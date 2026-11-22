"""[cloud] Smoke-check FaceTurnSkills-v0: forms, random rollout, and the conductor.

Sections (openspec tasks 2.1/2.5):
  1. observation/action forms: the flat state width (shared skills vector + per-agent
     proprioception), the Dict-of-Discrete action space, the info keys;
  2. a short random rollout (garbage decisions must not crash and must report the
     useless-action flags);
  3. the scripted conductor (oracle info, `callosum.experiments.arm.skills_conductor`): on a
     random plan of length 1 and on 2-3 moves by default, or on the FIXED plan given with
     `--plan-notation` (e.g. the pif-paf `"R U R' U'"` -> 6 executable moves, primes
     expanded). Prints the goal/plan, success rate, decisions used, flips executed, final
     plan index / active-layer angle / cube position and useless totals -- the environment's
     sanity validation;
  4. timing: batch reset and one decision step (the task 2.4 measurement; batched IK
     beyond this decision-boundary planning is future cloud work).

Never run on the Mac: the skill env needs the simulator (Linux/GPU or a Linux CPU
probe with --sim-backend cpu). Style follows scripts/experiments/arm/smoke_face_turn.py.

    uv run python scripts/experiments/arm/smoke_face_turn_skills.py --sim-backend gpu --num-envs 16
    # fixed pif-paf plan; needs a bigger internal budget (the config default is 150):
    ... --plan-notation "R U R' U'" --max-episode-decisions 600 --sim-backend cpu --num-envs 1
"""

import sys
import time

import numpy as np
import torch

import callosum.experiments.arm.skills_env  # noqa: F401  (registers FaceTurnSkills-v0)
from callosum.experiments.arm.sim_utils import make_env, parse_args
from callosum.experiments.arm.skills_agent_obs import SkillsObsBuilder
from callosum.experiments.arm.skills_conductor import HOLDER_ID, SkillsConductor
from callosum.experiments.arm.skills_config import FaceTurnSkillsConfig
from callosum.experiments.arm.skills_obs import (
    INFO_ACTIVE_LAYER_ANGLE_DEG,
    INFO_PLAN_INDEX,
    INFO_SUCCESS,
    INFO_TIME_TO_SUCCESS,
    INFO_USELESS_HOLDER,
    INFO_USELESS_ROTATOR,
)

_FLIP_IDS = (HOLDER_ID["flip90"], HOLDER_ID["flip180"])


def _extra_flags(parser):
    parser.add_argument(
        "--plan-length", type=int, default=1, help="Moves of run 3a. (default: %(default)s)"
    )
    parser.add_argument(
        "--conductor-plan",
        type=int,
        default=3,
        help="Moves of run 3b (2-3 for E2). (default: %(default)s)",
    )
    parser.add_argument(
        "--decisions", type=int, default=90, help="Decision budget of the conductor runs."
    )
    parser.add_argument(
        "--plan-notation",
        default="",
        help=(
            "Fixed plan for EVERY env/episode in standard face-turn notation, e.g. \"R U R' U'\" "
            "(the Sexy Move / pif-paf, 6 executable moves). Empty = random plans of "
            "--plan-length. Needs a bigger budget, see --max-episode-decisions."
        ),
    )
    parser.add_argument(
        "--max-episode-decisions",
        type=int,
        default=None,
        help=(
            "Internal episode budget in decisions (the env config's max_episode_decisions, "
            "default %(default)s = the config value); the pif-paf needs ~400-600. Also sets "
            "the conductor loop budget when it is larger than --decisions."
        ),
    )


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _skills_config(args, plan_length: int | None = None) -> FaceTurnSkillsConfig:
    """The env config with the CLI overrides (fixed plan and/or decision budget).

    `plan_notation` is authoritative: its prime expansion derives `plan_length` in the
    config's `__post_init__`, so a separate `plan_length` kwarg is only passed for the
    random-plan runs (and never together with a notation -- that would be a conflict).
    """
    kwargs: dict = {}
    if args.max_episode_decisions is not None:
        kwargs["max_episode_decisions"] = int(args.max_episode_decisions)
    if args.plan_notation:
        kwargs["plan_notation"] = args.plan_notation
    elif plan_length is not None:
        kwargs["plan_length"] = int(plan_length)
    return FaceTurnSkillsConfig(**kwargs)


def _plan_text(cfg: FaceTurnSkillsConfig) -> str:
    if cfg.plan_notation:
        return (
            f"fixed plan {cfg.plan_notation!r} -> {cfg.plan_length} executable moves "
            f"(primes expand to 90+180 deg)"
        )
    return f"random plan of {cfg.plan_length} move(s) per env/episode"


def _forms(env, base_env) -> None:
    obs, infos = env.reset(seed=0)
    print(f"num_envs: {base_env.num_envs}")
    print(f"flattened obs (obs_mode='state'): shape={tuple(obs.shape)}, dtype={obs.dtype}")
    print(f"action_space: {env.action_space}")
    print(f"info keys: {sorted(infos)}")
    cfg = base_env.skills_config
    print(f"decision_period: {cfg.decision_period} substeps")
    print(f"goal: {_plan_text(cfg)}")
    print(f"max_episode_decisions: {cfg.max_episode_decisions}")


def _random_rollout(env, steps: int = 5) -> None:
    env.reset(seed=0)
    for _ in range(steps):
        action = env.action_space.sample()  # unbatched: one id per agent, broadcast
        _obs, reward, _terminated, _truncated, info = env.step(action)
    print(f"after {steps} random decisions:")
    print(
        f"  reward: shape={tuple(reward.shape)}, dtype={reward.dtype}, range "
        f"[{reward.min().item():.3f}, {reward.max().item():.3f}]"
    )
    print(f"  success: {info[INFO_SUCCESS].any().item()} (expected False)")
    print(
        f"  useless h/r: {int(_np(info[INFO_USELESS_HOLDER]).sum())} / "
        f"{int(_np(info[INFO_USELESS_ROTATOR]).sum())}"
    )


def _conductor_run(args, cfg: FaceTurnSkillsConfig, label: str) -> None:
    env = make_env("FaceTurnSkills-v0", args, partner_mode="oracle", skills_config=cfg)
    try:
        base_env = env.unwrapped
        uids = tuple(env.action_space.spaces)
        obs, infos = env.reset(seed=0)
        builder = SkillsObsBuilder.from_env_obs(
            base_env.get_obs(unflattened=True), obs, "oracle", uids
        )
        n = base_env.num_envs
        conductors = [SkillsConductor() for _ in range(n)]
        successes = np.zeros(n, dtype=bool)
        decisions_used = np.zeros(n, dtype=np.int64)
        flips = np.zeros(n, dtype=np.int64)
        previous = np.full(n, -1, dtype=np.int64)
        useless = np.zeros((2, n), dtype=np.int64)
        budget = max(args.decisions, args.max_episode_decisions or 0)
        t0 = time.time()
        for step in range(budget):
            inputs = builder(obs, infos)
            holder_ids, rotator_ids = [], []
            for e in range(n):
                holder_ids.append(
                    conductors[e].act("holder", inputs[0][e].cpu().numpy(), _info_at(infos, e))
                )
                rotator_ids.append(
                    conductors[e].act("rotator", inputs[1][e].cpu().numpy(), _info_at(infos, e))
                )
            # Count flip EXECUTIONS (a flip primitive re-selected across decisions is one
            # flip, not one per decision): a transition into flip90/flip180.
            for e in range(n):
                if holder_ids[e] in _FLIP_IDS and previous[e] != holder_ids[e]:
                    flips[e] += 1
                previous[e] = holder_ids[e]
            action = {
                uids[0]: torch.as_tensor(holder_ids, dtype=torch.long, device=base_env.device),
                uids[1]: torch.as_tensor(rotator_ids, dtype=torch.long, device=base_env.device),
            }
            obs, _reward, terminated, truncated, infos = env.step(action)
            fresh = ~successes
            successes |= _np(infos[INFO_SUCCESS]).astype(bool) & fresh
            decisions_used[fresh] = step + 1
            useless[0] += _np(infos.get(INFO_USELESS_HOLDER, 0)).astype(np.int64)
            useless[1] += _np(infos.get(INFO_USELESS_ROTATOR, 0)).astype(np.int64)
            done = _np(terminated).astype(bool) | _np(truncated).astype(bool)
            for e in np.nonzero(done)[0]:
                conductors[int(e)] = SkillsConductor()  # fresh episode: fresh conductor
        wall = time.time() - t0
        rate = float(successes.mean())
        times = (
            _np(infos[INFO_TIME_TO_SUCCESS]).astype(np.int64)
            if INFO_TIME_TO_SUCCESS in infos
            else None
        )
        plan_index = _np(infos[INFO_PLAN_INDEX]).astype(np.int64)
        angle = _np(infos[INFO_ACTIVE_LAYER_ANGLE_DEG])
        cube_pos = _np(base_env.cube.pose.p)
        print(
            f"conductor, {label}: success {rate:.3f} over {n} envs "
            f"({budget} decisions, {wall:.1f}s)"
        )
        print(f"  goal: {_plan_text(cfg)}; budget {cfg.max_episode_decisions} decisions")
        print(f"  decisions used: {decisions_used.tolist()}")
        print(f"  flips executed: {flips.tolist()}")
        print(f"  final plan index: {plan_index.tolist()} / {cfg.plan_length}")
        print(f"  final active layer angle (deg): {np.round(angle, 1).tolist()}")
        print(f"  final cube position (m): {np.round(cube_pos, 3).tolist()}")
        if successes.any():
            print(f"  sim steps to success (final envs): {times[successes].tolist()}")
        print(f"  useless actions h/r: {useless[0].sum()} / {useless[1].sum()}")
        if rate < 0.9:
            print("  WARNING: expected ~1.0 -- investigate before the E2 runs (task 2.5)")
    finally:
        env.close()


def _info_at(infos: dict, env: int) -> dict:
    """One env's row of the batched info dict (the conductor serves one env)."""
    return {
        key: (_np(value).reshape(-1)[env] if hasattr(value, "reshape") else value)
        for key, value in infos.items()
    }


def _timing(args, cfg: FaceTurnSkillsConfig) -> None:
    env = make_env("FaceTurnSkills-v0", args, partner_mode="none", skills_config=cfg)
    try:
        t0 = time.time()
        env.reset(seed=0)
        reset_s = time.time() - t0
        action = env.action_space.sample()
        t0 = time.time()
        env.step(action)
        decision_s = time.time() - t0
        print(
            f"timing: batch reset {reset_s:.3f}s, one decision {decision_s:.3f}s "
            f"({env.unwrapped.num_envs} envs; the task 2.4 note: primitive IK runs at "
            "decision boundaries, batched/GPU IK is deferred cloud work)"
        )
    finally:
        env.close()


def main() -> None:
    if sys.platform == "darwin":
        print("this smoke check needs the simulator; run it on the cloud GPU box (see --help)")
        return
    args = parse_args(__doc__, _extra_flags)
    cfg = _skills_config(args, plan_length=args.plan_length)
    env = make_env("FaceTurnSkills-v0", args, partner_mode="none", skills_config=cfg)
    try:
        _forms(env, env.unwrapped)
        _random_rollout(env)
    finally:
        env.close()
    if args.plan_notation:
        # The fixed-plan (pif-paf) run: one conductor over the whole explicit plan.
        _conductor_run(args, _skills_config(args), label=f"fixed plan {args.plan_notation!r}")
    else:
        _conductor_run(args, _skills_config(args, plan_length=args.plan_length), label="plan 3a")
        _conductor_run(args, _skills_config(args, plan_length=args.conductor_plan), label="plan 3b")
    _timing(args, _skills_config(args, plan_length=args.plan_length))


if __name__ == "__main__":
    main()
