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

Step 3.2 (Bi-JEPA) replaces that ad-hoc flag with `args.partner_input`
(oracle/predicted/none): a shared BiJEPA encoder+predictor produces a
partner-latent slot that is appended to each agent's decentralized base
(build_agent_obs, include_partner=False) via bijepa_step. The slot is
*detached* in the policy input (the policy consumes a fixed partner signal)
so the two policy optimizers never touch the shared encoder/predictor --
those are trained solely by the JEPA aux loss
(`callosum.agents.bijepa.jepa_loss`) through a dedicated `bijepa_optimizer`,
applied once per iteration. obs_dim is identical across all three modes (a
zero slot in "none") so the SAME network is reused and only the
information content varies -- the whole point of the ablation
(docs/thesis/03-method-bijepa.md §Фаза 1). The Bi-JEPA module lives in the
torch-only `callosum.training._bijepa_policy` / `callosum.agents.bijepa`
so the loss math is unit-tested on macOS (tests/test_bijepa*.py) before
this server-only file runs on the paid session -- this file cannot be
imported on macOS/CI, which has no mani_skill (docs/implementation-plan.md
section 0).
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
from tqdm import tqdm

from callosum.agents.bijepa import BiJEPA, jepa_loss
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
from callosum.training._bijepa_policy import bijepa_step
from callosum.training._metrics import MetricLogger
from callosum.training._ppo_core import Agent, compute_gae, ppo_update


def _make_env(args: IPPOConfig, num_envs: int, reconfiguration_freq: int | None) -> gym.Env:
    env_kwargs = {
        "obs_mode": "state_dict",
        "sim_backend": "physx_cuda",
        "render_backend": args.render_backend,
    }
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
    metric_log = MetricLogger(writer)
    hyperparams_table = "\n".join(f"|{k}|{v}|" for k, v in vars(args).items())
    writer.add_text("hyperparameters", f"|param|value|\n|-|-|\n{hyperparams_table}")

    next_raw_obs, _ = envs.reset(seed=args.seed)
    eval_raw_obs, _ = eval_envs.reset(seed=args.seed)

    latent_dim = args.bijepa.latent_dim
    # Decentralized base dim (own proprio + own extra, partner pose dropped)
    # is shared by both arms; the partner-input latent slot is appended on top
    # (step 3.2 -- callosum.training._bijepa_policy). build_agent_obs is
    # evaluated once here only to size the network; the actual per-step
    # assembly goes through bijepa_step so z_i/z_j/z_hat_j are produced for the
    # JEPA aux loss.
    # Decentralized base dim; both arms must share it (identical SO-100 proprio
    # geometry) -- else the shared BiJEPA encoder would feed on mismatched
    # inputs. Asserted so an asymmetric future env (Phase-2 asymmetric
    # holder/rotator, method doc §Будущее) fails loud instead of silently
    # mis-sizing one arm's policy. (Л5)
    base_dim = build_agent_obs(next_raw_obs, 0, agent_uids, include_partner=False).shape[-1]
    assert (
        build_agent_obs(next_raw_obs, 1, agent_uids, include_partner=False).shape[-1] == base_dim
    ), "asymmetric agent obs dims: partner_input toggle assumes both arms share the base"
    obs_dims = [base_dim + latent_dim for _ in range(2)]
    action_dims = [int(np.prod(envs.single_action_space[uid].shape)) for uid in agent_uids]
    action_low = [
        torch.from_numpy(envs.single_action_space[uid].low).to(device) for uid in agent_uids
    ]
    action_high = [
        torch.from_numpy(envs.single_action_space[uid].high).to(device) for uid in agent_uids
    ]
    print(f"obs_dims={obs_dims} (base={base_dim}+latent={latent_dim}) action_dims={action_dims}")

    agents = [Agent(obs_dims[i], action_dims[i]).to(device) for i in range(2)]
    optimizers = [
        optim.Adam(agents[i].parameters(), lr=args.learning_rate, eps=1e-5) for i in range(2)
    ]

    # Shared Bi-JEPA encoder + partner predictor (Bi-symmetry: both arms share
    # E and P, docs/thesis/03-method-bijepa.md §Формулировка). The policy
    # consumes a *detached* latent slot (see bijepa_step), so the two policy
    # optimizers above never touch these params -- they are trained solely by
    # the JEPA aux loss through this dedicated optimizer (method doc §Решения C).
    bijepa = BiJEPA(args.bijepa, obs_dim=base_dim).to(device)
    bijepa_optimizer = optim.Adam(bijepa.parameters(), lr=args.learning_rate, eps=1e-5)
    # Л1: this trainer wires the Phase-1 current-only path (context=None, k=0 --
    # predict the partner latent from the SAME-step own latent). context_len>1
    # would silently mis-size PartnerPredictor's input and is only valid with the
    # LatentHistory wiring (Phase 2); fail loud here rather than in the loss.
    assert args.bijepa.context_len == 1, (
        "Phase-1 trainer uses context=None (k=0); set context_len=1 or wire LatentHistory"
    )

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

    # disable=None turns the bar off automatically when stdout is not a TTY,
    # i.e. under `nohup ... > log`, where a redrawing bar would be noise. The
    # per-iteration line below covers that case instead.
    bar = tqdm(range(1, args.num_iterations + 1), disable=None, unit="iter", dynamic_ncols=True)
    for iteration in bar:
        for agent in agents:
            agent.eval()

        if args.eval_freq == 1 or iteration % args.eval_freq == 1:
            print(f"iteration={iteration}: evaluating")
            eval_raw_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_actions = {
                        uid: agents[i].get_action(
                            bijepa_step(
                                bijepa.encoder,
                                bijepa.predictor,
                                eval_raw_obs,
                                i,
                                agent_uids,
                                args.partner_input,
                                latent_dim,
                            )[0],
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
                metric_log.log(f"eval/{k}", mean, global_step)
                print(f"  eval_{k}_mean={mean:.4f}")
            print(
                f"  evaluated {args.num_eval_steps * args.num_eval_envs} steps,"
                f" {num_episodes} episodes"
            )

        if args.save_model and (args.eval_freq == 1 or iteration % args.eval_freq == 1):
            for i, uid in enumerate(agent_uids):
                torch.save(
                    agents[i].state_dict(), f"runs/{run_name}/agent_{uid}_ckpt_{iteration}.pt"
                )
            torch.save(bijepa.state_dict(), f"runs/{run_name}/bijepa_ckpt_{iteration}.pt")

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            for optimizer in (*optimizers, bijepa_optimizer):
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        final_values_buf = [
            torch.zeros((args.num_steps, args.num_envs), device=device) for _ in range(2)
        ]
        # JEPA aux-loss buffers per iteration. IMPORTANT: these are LISTS, not
        # pre-allocated zeros tensors, because assigning a grad-bearing tensor
        # into a slice (`buf[step] = z_hat`) is an in-place data copy that SEVERS
        # the computation graph -- so z_hat's grad could never reach E/P and the
        # aux loss would train nothing. Appending the tensor itself and
        # torch.stack-ing it at update time preserves the graph. z_j is the
        # detached target; z_hat keeps grad (trains shared E+P).
        z_j_list = [[], []]
        z_hat_list = [[], []]
        # z_i buffered DETACHED for diagnostics only (runbook §3.1 collapse
        # gate: a shrinking z_std alongside a falling jepa_loss flags encoder
        # collapse that the loss alone cannot show -- review Б3).
        z_i_list = [[], []]

        rollout_time = time.time()
        for step in range(args.num_steps):
            global_step += args.num_envs
            dones_buf[step] = next_done

            step_actions = {}
            # Bi-JEPA encoding lives OUTSIDE no_grad: the predicted partner
            # latent z_hat must keep its graph so the JEPA aux loss can train
            # the shared encoder/predictor later in this iteration. The
            # policy-input slot is detached inside bijepa_step, so obs_i
            # carries no graph and is safe to store / feed the actor-critic
            # under no_grad below. (method doc §Решения C: Bi-JEPA is an
            # auxiliary task alongside RL, trained via its own optimizer.)
            bijepa_pi = []
            for i, uid in enumerate(agent_uids):
                # target_encoder routes z_j through the EMA copy when
                # ema_target=True (Б6: the toggle now genuinely drives the JEPA
                # target). With ema_target=False (Phase-1 default) it is None
                # and z_j comes from the online encoder then is detached in the
                # loss -- behaviour unchanged.
                obs_i, z_i_i, z_j_i, z_hat_i = bijepa_step(
                    bijepa.encoder,
                    bijepa.predictor,
                    next_raw_obs,
                    i,
                    agent_uids,
                    args.partner_input,
                    latent_dim,
                    target_encoder=bijepa.target_encoder,
                )
                obs_buf[i][step] = obs_i
                z_i_list[i].append(z_i_i.detach())
                z_j_list[i].append(z_j_i.detach())
                z_hat_list[i].append(z_hat_i)
                bijepa_pi.append(obs_i)
            with torch.no_grad():
                for i, uid in enumerate(agent_uids):
                    action_i, logprob_i, _, value_i = agents[i].get_action_and_value(bijepa_pi[i])
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
                    metric_log.log(f"train/{k}", v[done_mask].float().mean(), global_step)
                with torch.no_grad():
                    for i in range(2):
                        final_obs_i = bijepa_step(
                            bijepa.encoder,
                            bijepa.predictor,
                            infos["final_observation"],
                            i,
                            agent_uids,
                            args.partner_input,
                            latent_dim,
                        )[0]
                        final_values_buf[i][
                            step, torch.arange(args.num_envs, device=device)[done_mask]
                        ] = agents[i].get_value(final_obs_i[done_mask]).view(-1)
        rollout_time = time.time() - rollout_time

        update_time = time.time()
        for i, uid in enumerate(agent_uids):
            with torch.no_grad():
                next_obs_i = bijepa_step(
                    bijepa.encoder,
                    bijepa.predictor,
                    next_raw_obs,
                    i,
                    agent_uids,
                    args.partner_input,
                    latent_dim,
                )[0]
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
            update_metrics = ppo_update(
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
            for k, v in update_metrics.items():
                metric_log.log(f"losses/{uid}/{k}", v, global_step)

        # Bi-JEPA auxiliary update: train the shared encoder+predictor on the
        # JEPA aux loss (predict z_j from the agent's own latent), separate
        # from the two policy optimizers -- which never see these params
        # because the policy-input slot is detached (see bijepa_step). z_j is
        # already detached in z_j_list; z_hat retains its graph to the shared
        # encoder/predictor (stacked, not slice-assigned, so the graph survives).
        # Applied in ALL partner_input modes so the partner model still learns
        # under the decentralized 'none' ablation arm too.
        # (method doc §Решения C / §Фаза 1.)
        bijepa_optimizer.zero_grad()
        jepa_aux_loss = jepa_loss(torch.stack(z_j_list[0]), torch.stack(z_hat_list[0])) + jepa_loss(
            torch.stack(z_j_list[1]), torch.stack(z_hat_list[1])
        )
        (jepa_aux_loss * args.bijepa.aux_weight).backward()
        bijepa_optimizer.step()
        # Б6: advance the EMA target encoder IF ema_target=True. ema_update is
        # a no-op when target_encoder is None (Phase-1 default), so this is safe
        # to call unconditionally; guarded for clarity/skip when disabled.
        if args.bijepa.ema_target:
            bijepa.ema_update()
        jepa_loss_val = (jepa_aux_loss / 2).item()
        metric_log.log("losses/jepa_loss", jepa_loss_val, global_step)
        # Б3 collapse diagnostics: alongside the JEPA loss, log the per-latent
        # std of z_i (current) and z_j (CTDE target) and the mean |z_hat| of
        # the predictor output. A jepa_loss that "decreases" while z_std -> 0
        # is encoder collapse, not partner modelling (runbook §3.1 gate).
        with torch.no_grad():
            for i, uid in enumerate(agent_uids):
                z_i_std = torch.stack(z_i_list[i]).std(0).mean().item()
                z_j_std = torch.stack(z_j_list[i]).std(0).mean().item()
                z_hat_abs = torch.stack(z_hat_list[i]).abs().mean().item()
                metric_log.log(f"bijepa/{uid}/z_i_std", z_i_std, global_step)
                metric_log.log(f"bijepa/{uid}/z_j_std", z_j_std, global_step)
                metric_log.log(f"bijepa/{uid}/z_hat_abs", z_hat_abs, global_step)
        update_time = time.time() - update_time

        sps = int(global_step / (time.time() - start_time))
        metric_log.log("charts/SPS", sps, global_step)
        metric_log.log("time/rollout_time", rollout_time, global_step)
        metric_log.log("time/update_time", update_time, global_step)
        if bar.disable:
            print(
                metric_log.iteration_line(iteration, args.num_iterations, global_step, sps),
                flush=True,
            )
        else:
            bar.set_postfix(metric_log.headline())

    if args.save_model:
        for i, uid in enumerate(agent_uids):
            torch.save(agents[i].state_dict(), f"runs/{run_name}/agent_{uid}_final_ckpt.pt")
        torch.save(bijepa.state_dict(), f"runs/{run_name}/bijepa_final_ckpt.pt")
    bar.close()
    print(metric_log.summary(), flush=True)
    writer.close()
    envs.close()
    eval_envs.close()


if __name__ == "__main__":
    main(parse_args())
