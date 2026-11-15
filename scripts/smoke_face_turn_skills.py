"""[cloud] Smoke-check FaceTurnSkills-v0: forms, random rollout, and the conductor.

Sections (openspec tasks 2.1/2.5):
  1. observation/action forms: the flat state width (shared skills vector + per-agent
     proprioception), the Dict-of-Discrete action space, the info keys;
  2. a short random rollout (garbage decisions must not crash and must report the
     useless-action flags);
  3. the scripted conductor (oracle info, `callosum.experts.skills_conductor`) on a
     plan of length 1 and on 2-3 moves: success rate, decisions and sim steps to
     success, useless totals -- the environment's sanity validation;
  4. timing: batch reset and one decision step (the task 2.4 measurement; batched IK
     beyond this decision-boundary planning is future cloud work).

Never run on the Mac: the skill env needs the simulator (Linux/GPU or a Linux CPU
probe with --sim-backend cpu). Style follows scripts/smoke_face_turn.py.

    uv run python scripts/smoke_face_turn_skills.py --sim-backend gpu --num-envs 16
"""

import sys
import time

import numpy as np
import torch
from _sim_utils import make_env, parse_args

import callosum.envs.face_turn_skills  # noqa: F401  (registers FaceTurnSkills-v0)
from callosum.envs._skills_obs import (
    INFO_SUCCESS,
    INFO_TIME_TO_SUCCESS,
    INFO_USELESS_HOLDER,
    INFO_USELESS_ROTATOR,
)
from callosum.experts.skills_conductor import SkillsConductor
from callosum.training._skills_agent_obs import SkillsObsBuilder


def _extra_flags(parser):
    parser.add_argument("--plan-length", type=int, default=1, help="Moves of run 3a. (default: %(default)s)")
    parser.add_argument(
        "--conductor-plan", type=int, default=3, help="Moves of run 3b (2-3 for E2). (default: %(default)s)"
    )
    parser.add_argument("--decisions", type=int, default=90, help="Decision budget of the conductor runs.")


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _forms(env, base_env) -> None:
    obs, infos = env.reset(seed=0)
    print(f"num_envs: {base_env.num_envs}")
    print(f"flattened obs (obs_mode='state'): shape={tuple(obs.shape)}, dtype={obs.dtype}")
    print(f"action_space: {env.action_space}")
    print(f"info keys: {sorted(infos)}")
    decisions = base_env.skills_config.decision_period
    print(
        f"decision_period: {decisions} substeps | max_episode_decisions: "
        f"{base_env.skills_config.max_episode_decisions}"
    )


def _random_rollout(env, steps: int = 5) -> None:
    env.reset(seed=0)
    for _ in range(steps):
        action = env.action_space.sample()  # unbatched: one id per agent, broadcast
        _obs, reward, _terminated, _truncated, info = env.step(action)
    print(f"after {steps} random decisions:")
    print(f"  reward: shape={tuple(reward.shape)}, dtype={reward.dtype}, range "
          f"[{reward.min().item():.3f}, {reward.max().item():.3f}]")
    print(f"  success: {info[INFO_SUCCESS].any().item()} (expected False)")
    print(f"  useless h/r: {int(_np(info[INFO_USELESS_HOLDER]).sum())} / "
          f"{int(_np(info[INFO_USELESS_ROTATOR]).sum())}")


def _conductor_run(args, plan_length: int) -> None:
    env = make_env(
        "FaceTurnSkills-v0", args, partner_mode="oracle", plan_length=plan_length
    )
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
        useless = np.zeros((2, n), dtype=np.int64)
        t0 = time.time()
        for step in range(args.decisions):
            inputs = builder(obs, infos)
            holder_ids, rotator_ids = [], []
            for e in range(n):
                holder_ids.append(
                    conductors[e].act("holder", inputs[0][e].cpu().numpy(), _info_at(infos, e))
                )
                rotator_ids.append(
                    conductors[e].act("rotator", inputs[1][e].cpu().numpy(), _info_at(infos, e))
                )
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
        times = _np(infos[INFO_TIME_TO_SUCCESS]).astype(np.int64) if INFO_TIME_TO_SUCCESS in infos else None
        print(f"conductor, plan_length={plan_length}: success {rate:.3f} over {n} envs "
              f"({args.decisions} decisions, {wall:.1f}s)")
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


def _timing(args) -> None:
    env = make_env("FaceTurnSkills-v0", args, partner_mode="none", plan_length=1)
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
    env = make_env("FaceTurnSkills-v0", args, partner_mode="none", plan_length=args.plan_length)
    try:
        _forms(env, env.unwrapped)
        _random_rollout(env)
    finally:
        env.close()
    _conductor_run(args, args.plan_length)
    _conductor_run(args, args.conductor_plan)
    _timing(args)


if __name__ == "__main__":
    main()
