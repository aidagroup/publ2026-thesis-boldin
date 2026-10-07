"""Collect demonstrations of the scripted FaceTurn expert in the policies' own action space.

The expert (`callosum.experts.face_turn_expert`, the strategy of `scripts/probe_face_turn.py`) runs
in `control="delta"` mode: the env is the one the IPPO trainer uses (`FaceTurn-v0`,
`pd_joint_delta_pos`, the registered 400-step episode, `obs_mode="state"`,
`render_backend="none"`, `reward_mode="normalized_dense"`) and the arms are driven with delta
actions that track the expert's joint waypoints. Every env step is recorded: the flat state
observation before the step, each agent's 6-D action exactly as passed to `env.step`, the reward
and the terminated / truncated / success flags. Episodes are cut at their first success (where the
training env terminates) and only successful ones are kept unless `--keep-all`.

One batch = one `env.reset(seed=...)` of `--num-envs` envs (GPU: e.g. 256 at once; the CPU sim has
one env, so use `--num-batches` to loop over seeds). Batch `b` uses reset seed `--seed + b`.
The expert runs one asynchronous phase state machine per env (`callosum.experts._phases`): no
env waits for another, an env whose IK fails just holds. Every batch prints, per phase, how many
envs completed it and the min / median / max env step at which they did, plus the IK failures;
the summary (success rate, first-success step distribution) doubles as the self-check that the
expert fits into the episode under the training controller.

    uv run python scripts/collect_demos.py --num-envs 256 --num-batches 2 --out runs/demos/a.pt
    # Mac CPU sim, 20 episodes (see CLAUDE.md for the throwaway env):
    python scripts/collect_demos.py --sim-backend cpu --num-batches 20

The file layout is documented in `callosum.training._demos`. Exit status: 1 if no successful
episode was collected. A batch without any success prints how far its episodes got (expert
grasps, face angle, body drift) next to the per-phase table so that a failed collection can be
diagnosed from the log.

`--max-episode-steps N` overrides the registered 400-step episode (for diagnosis: does the
expert succeed at all, and when?). The value is stored in the file's `meta`, and the trainers
refuse demos whose episode length differs from the env they train in, so demos collected with a
non-default limit are only usable together with the same `--max-episode-steps` in training.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from _sim_utils import make_env, parse_args
from mani_skill.utils import gym_utils

import callosum.envs.face_turn  # noqa: F401  (registers FaceTurn-v0)
from callosum.configs.face_turn_expert import ExpertControlConfig
from callosum.envs.face_turn import TARGET_FACE_ANGLE
from callosum.experts._tracking import describe_progress
from callosum.experts.face_turn_expert import ArmModel, Rig, run_expert
from callosum.training._agent_obs import AGENT_UIDS, obs_layout
from callosum.training._demos import (
    assemble_demos,
    git_commit,
    layout_to_meta,
    save_demos,
    split_rollout,
)

ENV_ID = "FaceTurn-v0"
CONTROL_MODE = "pd_joint_delta_pos"
REWARD_MODE = "normalized_dense"


def _add_args(parser) -> None:
    parser.add_argument("--seed", type=int, default=0, help="Reset seed of the first batch.")
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Number of resets (seeds seed, seed+1, ...); each yields --num-envs episodes.",
    )
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="Also keep unsuccessful episodes (default: successful ones only).",
    )
    parser.add_argument(
        "--action-noise",
        type=float,
        default=0.0,
        help="DART-style noise: std of Gaussian noise added to the executed arm actions (normalised "
        "units; 0.1 = 0.005 rad), clipped to [-1, 1]. The recorded action is the clean expert "
        "action (the label). The gripper entry is never perturbed. Default 0 (off).",
    )
    parser.add_argument(
        "--no-overlap",
        action="store_true",
        help="Run each env's phases strictly one after the other (about 90 steps slower, see "
        "ExpertControlConfig.overlap_approach); default: overlapped.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Episode limit passed to gym.make (diagnosis only; default: the registered 400). "
        "Stored in the demo metadata; BC/IPPO must train with the same episode length.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output file (default: runs/demos/faceturn_delta_<timestamp>.pt).",
    )


def _recorder(rig: Rig, rows: dict[str, list]):
    """A `Rig` step callback that appends the step's data (on the CPU) to `rows`.

    The recorded action is the rig's clean expert action, which equals the executed one unless
    `--action-noise` perturbs the execution.
    """

    def record(obs, action, reward, terminated, truncated, info) -> None:
        rows["obs"].append(obs.detach().float().cpu())
        label = rig.last_clean_action
        rows["actions"].append(torch.stack([label[uid] for uid in AGENT_UIDS], dim=1).cpu())
        rows["rewards"].append(reward.detach().float().reshape(-1).cpu())
        rows["terminated"].append(terminated.reshape(-1).cpu())
        rows["truncated"].append(truncated.reshape(-1).cpu())
        rows["success"].append(info["success"].reshape(-1).cpu())

    return record


class _Progress:
    """Per-env running statistics of a batch (face angle, grasps, body drift), for diagnosis."""

    def __init__(self, n: int) -> None:
        self.face_max = np.zeros(n)
        self.face_end = np.zeros(n)
        self.holder = np.zeros(n, dtype=bool)
        self.rotator = np.zeros(n, dtype=bool)
        self.both = np.zeros(n, dtype=bool)
        self.body_rot = np.zeros(n)
        self.body_pos = np.zeros(n)

    def update(self, info: dict) -> None:
        def get(key: str) -> np.ndarray:
            return info[key].reshape(-1).detach().cpu().numpy()

        self.face_end = get("face_angle")
        self.face_max = np.maximum(self.face_max, self.face_end)
        holder, rotator = get("holder_grasp").astype(bool), get("rotator_grasp").astype(bool)
        self.holder |= holder
        self.rotator |= rotator
        self.both |= holder & rotator
        self.body_rot = np.maximum(self.body_rot, get("body_rot_drift"))
        self.body_pos = np.maximum(self.body_pos, get("body_pos_drift"))

    def describe(self, cfg) -> list[str]:
        return describe_progress(
            self.face_max,
            self.face_end,
            self.holder,
            self.rotator,
            self.both,
            self.body_rot,
            self.body_pos,
            TARGET_FACE_ANGLE,
            cfg.angle_tol,
            cfg.body_rot_tol,
            cfg.body_pos_tol,
        )


def _print_summary(episodes: list[dict], attempted: int, steps: np.ndarray) -> None:
    """Success rate over all attempted episodes and the first-success step distribution."""
    kept = len(episodes)
    succeeded = len(steps)
    print(f"attempted {attempted} episodes, successful {succeeded} ({succeeded / attempted:.1%})")
    if succeeded:
        q = np.percentile(steps, [0, 25, 50, 75, 100])
        print(
            f"first-success env step (episode length): mean {steps.mean():.1f}, "
            f"min {q[0]:.0f} / p25 {q[1]:.0f} / median {q[2]:.0f} / p75 {q[3]:.0f} / max {q[4]:.0f}"
        )
    print(f"kept {kept} episodes, {sum(len(e['rewards']) for e in episodes)} transitions")


def main() -> None:
    # TODO(review): GPU backend unverified for the async per-env phase machines: the per-env IK
    # plan runs in a Python loop at reset (slow for 256 envs, minutes per batch), and contact
    # behaviour of the delta-controlled expert may differ per env. The CPU sim (1 env) is where
    # the lockstep predecessor was checked.
    args = parse_args(__doc__, _add_args)
    start = time.time()
    extra = {}
    if args.max_episode_steps is not None:
        if args.max_episode_steps < 1:
            raise SystemExit(f"--max-episode-steps must be >= 1, got {args.max_episode_steps}")
        extra["max_episode_steps"] = args.max_episode_steps
    env = make_env(
        ENV_ID,
        args,
        control_mode=CONTROL_MODE,
        reward_mode=REWARD_MODE,
        reconfiguration_freq=0,  # as the trainer: the scene is never rebuilt on reset
        **extra,
    )
    base = env.unwrapped
    max_steps = int(gym_utils.find_max_episode_steps_value(env))
    print(f"episode limit: {max_steps} steps" + (" (overridden)" if extra else ""))
    control = ExpertControlConfig(control="delta", overlap_approach=not args.no_overlap)
    model = ArmModel()

    episodes: list[dict] = []
    first_steps: list[int] = []
    attempted = 0
    layout = None
    batch_seeds = [args.seed + b for b in range(args.num_batches)]
    for seed in batch_seeds:
        obs, _ = env.reset(seed=seed)
        if layout is None:
            layout = obs_layout(base.get_obs(unflattened=True))
        rig = Rig(
            env,
            model,
            control,
            obs=obs,
            max_steps=max_steps,
            stop_on_success=True,
            action_noise=args.action_noise,
            noise_seed=seed,
            tolerate_ik_failures=True,
        )
        rows: dict[str, list] = {
            k: [] for k in ("obs", "actions", "rewards", "terminated", "truncated", "success")
        }
        rig.callbacks.append(_recorder(rig, rows))
        progress = _Progress(base.num_envs)
        rig.callbacks.append(lambda *cb_args, p=progress: p.update(cb_args[-1]))
        run_expert(rig)
        batch = split_rollout(
            torch.stack(rows["obs"]),
            torch.stack(rows["actions"]),
            torch.stack(rows["rewards"]).reshape(len(rows["rewards"]), base.num_envs),
            torch.stack(rows["terminated"]).reshape(len(rows["rewards"]), base.num_envs),
            torch.stack(rows["truncated"]).reshape(len(rows["rewards"]), base.num_envs),
            torch.stack(rows["success"]).reshape(len(rows["rewards"]), base.num_envs),
            keep_all=True,
            batch_seed=seed,
        )
        attempted += len(batch)
        wins = [e for e in batch if e["first_success_step"] >= 0]
        first_steps += [e["first_success_step"] + 1 for e in wins]
        episodes += batch if args.keep_all else wins
        print(
            f"batch seed {seed}: {len(wins)}/{len(batch)} envs succeeded "
            f"({int(rig.ik_failed.sum())} with an IK failure) "
            f"({rig.steps} env steps, {time.time() - start:.0f}s elapsed)",
            flush=True,
        )
        print(
            f"  expert planning (IK) took {rig.plan_seconds:.0f}s; "
            f"{int(rig.ik_failed.sum())} envs with an IK failure (held, they never succeed)"
        )
        if rig.machine is not None:
            print("  per-phase completion (env step at which each env completed the phase):")
            print("\n".join("  " + line for line in rig.machine.describe()), flush=True)
        if not wins:
            print(
                f"  no success in this batch: the expert stopped at env step {rig.steps} "
                f"(episode limit {max_steps})"
            )
            print("\n".join(progress.describe(base.reward_config)), flush=True)
    env.close()

    _print_summary(episodes, attempted, np.array(first_steps))
    if not any(e["first_success_step"] >= 0 for e in episodes):
        print("no successful episode: nothing saved")
        sys.exit(1)

    meta = {
        "env_id": ENV_ID,
        "control_mode": CONTROL_MODE,
        "reward_mode": REWARD_MODE,
        "partner_obs": "full",
        "max_episode_steps": max_steps,
        "obs_layout": layout_to_meta(layout),
        "agent_uids": list(AGENT_UIDS),
        "obs_dim": int(sum(width for _, width in layout)),
        "action_dims": [6, 6],
        "num_attempted": attempted,
        "num_successful": len(first_steps),
        "batch_seeds": batch_seeds,
        "num_envs": int(base.num_envs),
        "sim_backend": args.sim_backend,
        "git_commit": git_commit(),
        "expert": dict(vars(control)),
        "keep_all": bool(args.keep_all),
        "action_noise": float(args.action_noise),
    }
    demos = assemble_demos(episodes, meta)
    out = Path(args.out or f"runs/demos/faceturn_delta_{int(time.time())}.pt")
    save_demos(out, demos)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
