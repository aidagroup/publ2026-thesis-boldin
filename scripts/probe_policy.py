"""What does a trained policy actually DO? Per-step reward breakdown.

Server-only (GPU sim). Written after the 2M-step FaceTurn-v0 runs of
2026-08-30, where the policy's return (11.4) sat barely above the value of
doing nothing at all (10.8 at the rest pose) and success stayed at 0. Aggregate
metrics cannot say WHY; this rolls out the checkpoint deterministically and
prints each reward term separately, so the dead ones are visible.

    uv run python scripts/probe_policy.py                      # newest FaceTurn run
    uv run python scripts/probe_policy.py --run runs/FaceTurn-v0__ippo__1__123
    uv run python scripts/probe_policy.py --random             # untrained baseline
"""

import argparse
import glob
import os

import gymnasium as gym
import torch
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import callosum.envs.face_turn  # noqa: F401  (registers FaceTurn-v0)
from callosum.agents.bijepa import BiJEPA
from callosum.configs.bijepa import BiJEPAConfig
from callosum.envs.face_turn import TARGET_FACE_ANGLE
from callosum.training._agent_obs import build_agent_obs
from callosum.training._bijepa_policy import bijepa_step
from callosum.training._ppo_core import Agent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None, help="run directory holding *_final_ckpt.pt")
    ap.add_argument("--env-id", default="FaceTurn-v0")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--every", type=int, default=10, help="print every N steps")
    ap.add_argument("--random", action="store_true", help="skip the checkpoint, act randomly")
    args = ap.parse_args()

    # Wrap exactly as callosum.training.ippo does: `single_action_space` lives on
    # ManiSkillVectorEnv, not on the TimeLimitWrapper that gym.make returns.
    # ignore_terminations=True keeps a successful episode running, so the whole
    # trajectory stays visible instead of auto-resetting mid-probe.
    n_envs = 64
    env = ManiSkillVectorEnv(
        gym.make(
            args.env_id,
            num_envs=n_envs,
            obs_mode="state_dict",
            sim_backend="physx_cuda",
            render_backend="none",
        ),
        n_envs,
        ignore_terminations=True,
        record_metrics=False,
    )
    base = env.unwrapped
    uids = tuple(base.agent.agents_dict.keys())
    raw_obs, _ = env.reset(seed=0)
    device = base.device

    agents = None
    if not args.random:
        run = args.run
        if run is None:
            # Newest run that actually HAS checkpoints, not merely the newest
            # directory: a run that dies during startup still creates its
            # directory (the TensorBoard writer opens first), and picking that
            # one fails with a confusing FileNotFoundError.
            cands = [
                d
                for d in glob.glob(f"runs/*{args.env_id.split('-')[0]}*/")
                if os.path.exists(os.path.join(d, "bijepa_final_ckpt.pt"))
            ]
            if not cands:
                print("no run with checkpoints found. Directories under runs/:")
                for d in sorted(glob.glob("runs/*/")):
                    n = len(glob.glob(os.path.join(d, "*.pt")))
                    print(f"  {d}  ({n} checkpoint files)")
                print("\nPass --run <dir>, or --random for an untrained baseline.")
                return 1
            run = max(cands, key=os.path.getmtime)
        cfg = BiJEPAConfig()
        bij = BiJEPA(cfg, obs_dim=build_agent_obs(raw_obs, 0, uids).shape[-1]).to(device)
        bij.load_state_dict(
            torch.load(os.path.join(run, "bijepa_final_ckpt.pt"), map_location=device)
        )
        agents = []
        for i, uid in enumerate(uids):
            pi = bijepa_step(bij.encoder, bij.predictor, raw_obs, i, uids, "none", cfg.latent_dim)[
                0
            ]
            a = Agent(pi.shape[-1], int(env.single_action_space[uid].shape[-1])).to(device)
            a.load_state_dict(
                torch.load(os.path.join(run, f"agent_{uid}_final_ckpt.pt"), map_location=device)
            )
            a.eval()
            agents.append((a, bij, cfg))
        print(f"loaded {run}")
    else:
        print("random policy (untrained baseline)")

    hdr = (
        f"{'step':>5} {'rot→face':>9} {'hold→body':>10} {'grasped':>8} "
        f"{'held':>6} {'angle':>7} {'reward':>8}"
    )
    print(f"\n{hdr}\n{'-' * len(hdr)}")
    with torch.no_grad():
        for step in range(args.steps + 1):
            # Same targets compute_dense_reward uses: the grasp HANDLES, not
            # the link origins (which sit inside solid geometry).
            d_rot = torch.linalg.norm(base.agent_b.tcp_pos - base.face_grasp_pos, dim=1)
            d_hold = torch.linalg.norm(base.agent_a.tcp_pos - base.body_grasp_pos, dim=1)
            grasped = base.agent_b.is_grasping(base.face_link).float()
            held = base.agent_a.is_grasping(base.body_link).float()
            angle = base.face_link.joint.qpos.squeeze(-1)
            info = base.get_info()
            rew = base.compute_dense_reward(None, None, info)
            if step % args.every == 0:
                print(
                    f"{step:>5} {d_rot.mean():>9.3f} {d_hold.mean():>10.3f} "
                    f"{grasped.mean():>8.2f} {held.mean():>6.2f} "
                    f"{angle.mean():>7.3f} {rew.mean():>8.3f}"
                )
            # Keep the last pre-reset breakdown: at step == max_episode_steps the
            # env has already auto-reset, so reading terms after the loop reports
            # the NEXT episode's start, not this policy's achievement.
            angle_rem = (TARGET_FACE_ANGLE - angle).clamp(min=0)
            cfg_r = base.reward_config
            last_terms = {
                "rotator_reach": float(
                    cfg_r.weight_rotator_reach * (1 - torch.tanh(5 * d_rot)).mean()
                ),
                "holder_reach": float(
                    cfg_r.weight_holder_reach * (1 - torch.tanh(5 * d_hold)).mean()
                ),
                "grasp": float(cfg_r.weight_grasp * grasped.mean()),
                "holder_grasp": float(cfg_r.weight_holder_grasp * held.mean()),
                "angle_progress": float(
                    cfg_r.weight_angle_progress * (1 - torch.tanh(2 * angle_rem)).mean()
                ),
            }
            if step == 0:
                start_reward = float(rew.mean())
            if step == args.steps:
                break

            actions = {}
            for i, uid in enumerate(uids):
                if agents is None:
                    actions[uid] = torch.from_numpy(env.action_space[uid].sample()).to(device)
                else:
                    a, bij, cfg = agents[i]
                    pi = bijepa_step(
                        bij.encoder, bij.predictor, raw_obs, i, uids, "none", cfg.latent_dim
                    )[0]
                    actions[uid] = a.get_action(pi, deterministic=True)
            raw_obs, _, _, _, _ = env.step(actions)

    print("\nreward terms at the last step BEFORE the episode reset:")
    for k, v in last_terms.items():
        print(f"  {k:16} {v:+8.4f}")
    total = sum(last_terms.values())
    # ManiSkill's default reward_mode is "normalized_dense" (first entry of
    # BaseEnv.SUPPORTED_REWARD_MODES), so the trainer's train/return and
    # eval/return are compute_dense_reward divided by the sum of the positive
    # term weights. Printing both scales avoids comparing one against the other,
    # which is exactly the mistake that produced a wrong diagnosis on 2026-08-30.
    #
    # Read off the env, never recomputed here: this local copy had already
    # gone stale once, silently omitting weight_holder_grasp after that term
    # was added.
    norm = base.reward_normalization_divisor
    print(f"  {'TOTAL (raw)':16} {total:+8.4f}")
    print(f"  {'TOTAL (normalized)':16} {total / norm:+8.4f}   <- the scale train/return uses")
    print()
    # Measured this run rather than hardcoded: the "doing nothing is worth X"
    # reference numbers move whenever the scene or the weights change, and a
    # stale constant here is worse than none.
    print(
        f"  {'start pose (step 0)':22} raw {start_reward:+7.4f}"
        f"  normalized {start_reward / norm:+7.4f}"
    )
    print(f"  {'perfect score':22} raw {norm:+7.4f}  normalized {1.0:+7.4f}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
