"""Independent PPO (IPPO) trainer for TwoSO100Base-derived envs (step 2.1).

Run as: `python -m callosum.training.ippo --env-id FaceTurn-v0` (matching
docs/server-runbook.md). Server-only: needs GPU sim (Linux+CUDA), absent on
macOS, so this file cannot be run or even imported there -- see
docs/implementation-plan.md section 0.

Adapted from mani-skill's own PPO baseline
(examples/baselines/ppo/ppo.py @ v3.0.1, at the root of the ManiSkill repo --
not shipped in the pip package, so read from GitHub). Two independent
actor-critics (one per SO-100 arm, callosum.training._ppo_core.Agent)
instead of one, each seeing only its own observation slice
(callosum.training._agent_obs.build_agent_obs) and producing its own
action; GAE and the PPO update run separately per agent
(callosum.training._ppo_core.compute_gae / ppo_update) but share the same
env reward and termination/truncation signal, since the task reward is
cooperative/shared, not per-agent (docs/thesis/04-experiment-design.md).

Departures from the reference, each forced by our multi-agent setup or an
explicit plan constraint (docs/implementation-plan.md step 2.1):
- Actions are a dict keyed by agent uid ("so100-0"/"so100-1"), not one
  vector -- handled directly (no FlattenActionSpaceWrapper, which merges a
  Dict action space into ONE policy's output; the opposite of what two
  independent policies need).
- obs_mode="state_dict", not "state": "state" flattens both agents'
  observations into one shared tensor, destroying the per-agent structure
  needed to build each agent's own input honestly -- see
  callosum.training._agent_obs's module docstring for exactly how
  "state_dict" avoids that.
- No video capture / RecordEpisode / wandb: the training server has no
  Vulkan/EGL display setup (state-based training needs none, see
  docs/setup.md), and the plan authorizes adding only `tensorboard` to the
  lockfile for this step, not `wandb`.
- argparse instead of tyro (see callosum.configs.ippo) -- not an authorized
  new dependency.

`args.include_partner` (default False) is a diagnostic "oracle" condition:
if the decentralized baseline fails to learn on FaceTurn-v0, flipping it on
isolates whether that's a task/reward-shaping problem or genuinely requires
partner information, without spending a second paid session guessing. It is
threaded through every build_agent_obs call site below (obs-dim sizing,
rollout, eval, the final_observation bootstrap, and the post-rollout
next_value) -- all of them must agree, since the value function has to see
the same observation as the policy it is scoring, or advantage estimates
become meaningless. See callosum.training._agent_obs's module docstring for
why this is the env-level-impossible half of the step 1.4 partner_obs flag
finally becoming expressible.
"""

import random
import time
from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from callosum.configs.ippo import IPPOConfig, parse_args

# Registers FaceTurn-v0/TwoSO100-v0 (import side effect); callosum.envs
# itself stays import-clean on macOS/CI (no mani_skill there), so it
# deliberately does not re-export these submodules. Bound to distinct names
# (not two `import callosum.envs.x` statements) so ruff's unused-import
# check tracks each one individually instead of one shadowing the other --
# empirically, `ruff check --fix` silently deleted both when they collided
# on the single name `callosum`.
from callosum.envs import face_turn as _face_turn  # noqa: F401
from callosum.envs import two_so100_base as _two_so100_base  # noqa: F401
from callosum.training._agent_obs import build_agent_obs
from callosum.training._ppo_core import Agent, compute_gae, ppo_update


def _make_env(args: IPPOConfig, num_envs: int, reconfiguration_freq: int | None) -> gym.Env:
    env_kwargs = {"obs_mode": "state_dict", "sim_backend": "physx_cuda"}
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    return gym.make(
        args.env_id, num_envs=num_envs, reconfiguration_freq=reconfiguration_freq, **env_kwargs
    )


def main(args: IPPOConfig) -> None:
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = args.exp_name or f"{args.env_id}__ippo__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = _make_env(args, args.num_envs, args.reconfiguration_freq)
    eval_envs = _make_env(args, args.num_eval_envs, args.eval_reconfiguration_freq)

    assert hasattr(envs.unwrapped.agent, "agents_dict"), f"{args.env_id} is not a multi-agent env"
    agent_uids = tuple(envs.unwrapped.agent.agents_dict.keys())
    assert len(agent_uids) == 2, f"this IPPO trainer assumes exactly 2 agents, got {agent_uids}"

    envs = ManiSkillVectorEnv(
        envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True
    )
    eval_envs = ManiSkillVectorEnv(
        eval_envs,
        args.num_eval_envs,
        ignore_terminations=not args.eval_partial_reset,
        record_metrics=True,
    )

    print(f"agents: {agent_uids}")
    writer = SummaryWriter(f"runs/{run_name}")
    hyperparams_table = "\n".join(f"|{k}|{v}|" for k, v in vars(args).items())
    writer.add_text("hyperparameters", f"|param|value|\n|-|-|\n{hyperparams_table}")

    next_raw_obs, _ = envs.reset(seed=args.seed)
    eval_raw_obs, _ = eval_envs.reset(seed=args.seed)

    obs_dims = [
        build_agent_obs(next_raw_obs, i, agent_uids, args.include_partner).shape[-1]
        for i in range(2)
    ]
    action_dims = [int(np.prod(envs.single_action_space[uid].shape)) for uid in agent_uids]
    action_low = [
        torch.from_numpy(envs.single_action_space[uid].low).to(device) for uid in agent_uids
    ]
    action_high = [
        torch.from_numpy(envs.single_action_space[uid].high).to(device) for uid in agent_uids
    ]
    print(f"obs_dims={obs_dims} action_dims={action_dims}")

    agents = [Agent(obs_dims[i], action_dims[i]).to(device) for i in range(2)]
    optimizers = [
        optim.Adam(agents[i].parameters(), lr=args.learning_rate, eps=1e-5) for i in range(2)
    ]

    def clip_action(i: int, action: torch.Tensor) -> torch.Tensor:
        return torch.clamp(action.detach(), action_low[i], action_high[i])

    obs_buf = [
        torch.zeros((args.num_steps, args.num_envs, obs_dims[i]), device=device) for i in range(2)
    ]
    actions_buf = [
        torch.zeros((args.num_steps, args.num_envs, action_dims[i]), device=device)
        for i in range(2)
    ]
    logprobs_buf = [torch.zeros((args.num_steps, args.num_envs), device=device) for _ in range(2)]
    values_buf = [torch.zeros((args.num_steps, args.num_envs), device=device) for _ in range(2)]
    # Shared: the task reward is cooperative (one scalar per env), and
    # termination/truncation apply to the whole env, not per agent.
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
        for agent in agents:
            agent.eval()

        if iteration % args.eval_freq == 1:
            print(f"iteration={iteration}: evaluating")
            eval_raw_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_actions = {
                        uid: agents[i].get_action(
                            build_agent_obs(eval_raw_obs, i, agent_uids, args.include_partner),
                            deterministic=True,
                        )
                        for i, uid in enumerate(agent_uids)
                    }
                    eval_raw_obs, _, _, _, eval_infos = eval_envs.step(eval_actions)
                if "final_info" in eval_infos:
                    num_episodes += eval_infos["_final_info"].sum()
                    for k, v in eval_infos["final_info"]["episode"].items():
                        eval_metrics[k].append(v)
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                writer.add_scalar(f"eval/{k}", mean, global_step)
                print(f"  eval_{k}_mean={mean:.4f}")
            print(
                f"  evaluated {args.num_eval_steps * args.num_eval_envs} steps,"
                f" {num_episodes} episodes"
            )

        if args.save_model and iteration % args.eval_freq == 1:
            for i, uid in enumerate(agent_uids):
                torch.save(
                    agents[i].state_dict(), f"runs/{run_name}/agent_{uid}_ckpt_{iteration}.pt"
                )

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            for optimizer in optimizers:
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        final_values_buf = [
            torch.zeros((args.num_steps, args.num_envs), device=device) for _ in range(2)
        ]

        rollout_time = time.time()
        for step in range(args.num_steps):
            global_step += args.num_envs
            dones_buf[step] = next_done

            step_actions = {}
            with torch.no_grad():
                for i, uid in enumerate(agent_uids):
                    obs_i = build_agent_obs(next_raw_obs, i, agent_uids, args.include_partner)
                    obs_buf[i][step] = obs_i
                    action_i, logprob_i, _, value_i = agents[i].get_action_and_value(obs_i)
                    values_buf[i][step] = value_i.flatten()
                    actions_buf[i][step] = action_i
                    logprobs_buf[i][step] = logprob_i
                    step_actions[uid] = clip_action(i, action_i)

            next_raw_obs, reward, terminations, truncations, infos = envs.step(step_actions)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards_buf[step] = reward.view(-1) * args.reward_scale

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    writer.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                with torch.no_grad():
                    for i in range(2):
                        final_obs_i = build_agent_obs(
                            infos["final_observation"], i, agent_uids, args.include_partner
                        )
                        final_values_buf[i][
                            step, torch.arange(args.num_envs, device=device)[done_mask]
                        ] = agents[i].get_value(final_obs_i[done_mask]).view(-1)
        rollout_time = time.time() - rollout_time

        update_time = time.time()
        for i, uid in enumerate(agent_uids):
            with torch.no_grad():
                next_obs_i = build_agent_obs(next_raw_obs, i, agent_uids, args.include_partner)
                next_value_i = agents[i].get_value(next_obs_i).reshape(1, -1)
            advantages, returns = compute_gae(
                rewards_buf,
                values_buf[i],
                dones_buf,
                final_values_buf[i],
                next_value_i,
                next_done,
                args,
            )
            b_obs = obs_buf[i].reshape(-1, obs_dims[i])
            b_actions = actions_buf[i].reshape(-1, action_dims[i])
            b_logprobs = logprobs_buf[i].reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values_buf[i].reshape(-1)

            agents[i].train()
            metrics = ppo_update(
                agents[i],
                optimizers[i],
                b_obs,
                b_actions,
                b_logprobs,
                b_advantages,
                b_returns,
                b_values,
                args,
            )
            for k, v in metrics.items():
                writer.add_scalar(f"losses/{uid}/{k}", v, global_step)
        update_time = time.time() - update_time

        sps = int(global_step / (time.time() - start_time))
        print(f"iteration={iteration} global_step={global_step} SPS={sps}")
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("time/rollout_time", rollout_time, global_step)
        writer.add_scalar("time/update_time", update_time, global_step)

    if args.save_model:
        for i, uid in enumerate(agent_uids):
            torch.save(agents[i].state_dict(), f"runs/{run_name}/agent_{uid}_final_ckpt.pt")
    writer.close()
    envs.close()
    eval_envs.close()


if __name__ == "__main__":
    main(parse_args())
