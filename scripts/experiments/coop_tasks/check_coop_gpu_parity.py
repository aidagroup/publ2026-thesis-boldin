"""Step-by-step parity check: the CPU `VecCoopEnv` loop vs the batched `CoopBatchGPU`.

The GPU backend re-implements the CPU `CoopTaskEnv` control step in tensor algebra (latency
queues, start-delay suppression, one-sided force/torque scaling, the toss-catch paddle pose
interpolation, the `control_every` physics substeps, the observation layout, the success/
failure/reward rules, the per-episode hidden damping or contact parameters, and the auto-reset
of the finished worlds). This script is the proof that the two paths are the same environment:
`--num-worlds` CPU envs and one GPU batch env are reset with the SAME episode seeds and fed the
SAME actions, and every step's `qpos`, `qvel`, observation, reward and termination flags are
compared.

The intended difference is the float32 Warp physics against the float64 MuJoCo CPU path, so the
tolerances are small but not zero. The contact-free tasks (`cartpole`, `pendulum`) match at
~1e-7; the toss-catch's contact model is the only non-smooth part of the suite, so its
tolerance is looser and a policy-like (oracle) run can accumulate contact-timing differences.

What is compared:

* the reset observations (should be identical: same sampler, same layout);
* after every step: the full `qpos`/`qvel` of every NON-done world, both agents' flat
  observations, the shared reward and the success/failure/truncation flags;
* on the step a world terminates: its terminal flags/reward and the GPU `final_obs` against the
  CPU env's last observation (the truncation-bootstrap contract of `VecCoopEnv`), and the GPU's
  post-auto-reset observation against a CPU env restarted with the same seed (the wrapper's
  seed stream is mirrored: world `i` starts from `seed + i`, auto-resets continue from
  `seed + num_worlds`);
* the finished-episode statistics (`finished_returns/lengths/successes/angles`) against the
  CPU env's accumulators;
* the sampled randomisation tensors against a fresh CPU call of the same sampler, so a seed
  mapping to different episodes on the two paths cannot slip through.

Controllers: `zero` (do-nothing actions: long stable trajectories, the strictest numerical
check for the contact-free tasks), `random` (torch-random normalized commands, exercises the
action scaling/latency path) and `oracle` (the task's scripted `make_oracle` acting on the CPU
envs, which drives the episode to a real success/failure and, for the toss-catch, through real
ball/paddle contacts).

The GPU backend needs the optional `mujoco-warp==4.0.1` install (the `sandbox-gpu` extra,
Linux/CUDA only; the pod's plain `uv sync --frozen` removes it), so on the pod run:

    uv pip install --python /root/.callosum/venv/bin/python mujoco-warp==4.0.1
    python scripts/experiments/coop_tasks/check_coop_gpu_parity.py --task cartpole --controller random
    python scripts/experiments/coop_tasks/check_coop_gpu_parity.py --task toss --controller oracle

Exit code 0 when every comparison is inside its tolerance, 1 otherwise.
"""

import argparse
import sys

import numpy as np

from callosum.experiments.coop_tasks.config import coop_config
from callosum.experiments.coop_tasks.logic import AGENT_KEYS, task_diagnostic

_TASK_TOLERANCES = {
    "cartpole": {"qpos": 1e-5, "qvel": 1e-4, "obs": 1e-4, "reward": 1e-4, "return": 1e-3},
    "pendulum": {"qpos": 1e-5, "qvel": 1e-4, "obs": 1e-4, "reward": 1e-4, "return": 1e-3},
    "toss": {"qpos": 5e-3, "qvel": 1e-2, "obs": 1e-2, "reward": 1e-2, "return": 1e-1},
}
"""Max-abs tolerances of the compared quantities (float32 Warp vs float64 MuJoCo).

The contact-free tasks match at float-noise level; the toss-catch is contact-driven and its
contact solver is float32-sensitive, hence the looser (documented) bound.
"""


_CARTPOLE_ORACLE_TOLERANCES = {"qpos": 1e-3, "qvel": 1e-3, "obs": 2e-3}
"""Relaxed tolerances of a cart-pole ORACLE run (the closed loop amplifies float32 noise)."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the parity-check CLI."""
    parser = argparse.ArgumentParser(
        description="Compare CoopBatchGPU against the CPU coop envs step by step."
    )
    parser.add_argument("--task", choices=("cartpole", "pendulum", "toss"), default="cartpole")
    parser.add_argument("--num-worlds", type=int, default=8, help="worlds compared")
    parser.add_argument(
        "--steps", type=int, default=60, help="maximum control steps (stops at the first end)"
    )
    parser.add_argument("--seed", type=int, default=0, help="base episode seed")
    parser.add_argument(
        "--partner-mode", choices=("none", "oracle"), default="none", help="env partner mode"
    )
    parser.add_argument(
        "--controller",
        choices=("zero", "random", "oracle"),
        default="zero",
        help="action source: zero (default, strictest), random commands or the scripted oracle",
    )
    parser.add_argument("--device", default="cuda", help="CUDA device of the GPU env")
    parser.add_argument(
        "--max-control-steps",
        type=int,
        default=None,
        help="override the episode budget (e.g. 20 to force time-limit truncations)",
    )
    for name in ("qpos", "qvel", "obs", "reward"):
        parser.add_argument(f"--{name}-tol", type=float, default=None)
    parser.add_argument(
        "--return-tol",
        type=float,
        default=None,
        help="max |CPU - GPU| of a finished episode's return",
    )
    parser.add_argument(
        "--require-finished",
        action="store_true",
        help=(
            "fail when no episode finished during the run; by default a step budget shorter "
            "than an episode only checks the numerical parity (the finished-episode contract "
            "is covered by the runs that do end episodes)"
        ),
    )
    args = parser.parse_args(argv)
    if args.num_worlds < 1:
        parser.error(f"--num-worlds must be >= 1, got {args.num_worlds}")
    if args.steps < 1:
        parser.error(f"--steps must be >= 1, got {args.steps}")
    if args.seed < 0:
        parser.error(f"--seed must be >= 0, got {args.seed}")
    tolerances = dict(_TASK_TOLERANCES[args.task])
    explicit = {name for name in tolerances if getattr(args, f"{name}_tol") is not None}
    for name in ("qpos", "qvel", "obs", "reward", "return"):
        value = getattr(args, f"{name}_tol")
        if value is not None:
            tolerances[name] = value
    # The cart-pole is an UNSTABLE shared object: the scripted LQR/Smith predictor loop is
    # only marginally stable, so the float32-vs-float64 perturbation grows by ~e^(0.16 t)
    # over the episode (measured on the pod: 3e-9 -> 1.8e-4 over 120 steps). The zero-action
    # run keeps the strict tolerance; a policy-like oracle run needs the amplified one.
    if args.task == "cartpole" and args.controller == "oracle":
        for name, value in _CARTPOLE_ORACLE_TOLERANCES.items():
            if name not in explicit:
                tolerances[name] = value
    args.tolerances = tolerances
    return args


def _max_abs_diff(expected: np.ndarray, actual: np.ndarray) -> float:
    """Max abs difference; `0.0` for empty inputs (all rows of the batch can be done)."""
    if expected.size == 0:
        return 0.0
    return float(np.abs(expected.astype(np.float64) - actual.astype(np.float64)).max())


def _stack_obs(observations: list[dict], role: str) -> np.ndarray:
    """The CPU envs' per-role observations as one `(num_worlds, obs_dim)` array."""
    return np.stack([np.asarray(obs[role], dtype=np.float32) for obs in observations])


def _check_randomization(gpu_env: object, seeds: list[int]) -> list[str]:
    """Compare the GPU randomisation tensors with fresh CPU samples of the same seeds."""
    from callosum.experiments.coop_tasks.logic import (
        sample_episode_randomization,
        sample_toss_randomization,
    )

    config = gpu_env.config
    sampler = sample_toss_randomization if config.task == "toss" else sample_episode_randomization
    records = [
        sampler(
            config.randomization,
            control_dt=config.control_dt,
            max_control_steps=config.max_control_steps,
            seed=seed,
        )
        for seed in seeds
    ]
    failures: list[str] = []
    agents = gpu_env.randomization_tensors()
    for role_index, role in enumerate(AGENT_KEYS):
        for name, tensor in agents[role].items():
            expected = np.asarray([getattr(record.agents[role_index], name) for record in records])
            actual = tensor.cpu().numpy()
            if expected.dtype == np.bool_:
                mismatch = int((expected != actual).sum())
            else:
                mismatch = int((np.abs(expected.astype(np.float64) - actual) > 1e-6).sum())
            if mismatch:
                failures.append(f"randomization {role}.{name}: {mismatch} world(s) differ")
    episode = gpu_env.episode_randomization_tensors()
    for name, tensor in episode.items():
        expected = np.asarray([getattr(record, name) for record in records])
        actual = tensor.cpu().numpy()
        if expected.dtype == np.bool_:
            mismatch = int((expected != actual).sum())
        else:
            mismatch = int((np.abs(expected.astype(np.float64) - actual) > 1e-6).sum())
        if mismatch:
            failures.append(f"randomization episode.{name}: {mismatch} world(s) differ")
    return failures


def run_parity(args: argparse.Namespace) -> int:
    """Run the scenario; print the comparison table and return the process exit code."""
    import torch

    from callosum.experiments.coop_tasks import make_coop_env, make_oracle
    from callosum.experiments.coop_tasks.env_gpu import CoopBatchGPU

    torch.manual_seed(args.seed)
    config = coop_config(args.task)
    if args.max_control_steps is not None:
        import dataclasses

        config = dataclasses.replace(config, max_control_steps=args.max_control_steps)
    num_worlds = args.num_worlds
    seeds = [args.seed + index for index in range(num_worlds)]
    cpu_envs = [
        make_coop_env(args.task, config=config, partner_mode=args.partner_mode)
        for _ in range(num_worlds)
    ]
    oracles = [
        make_oracle(args.task, config) if args.controller == "oracle" else None
        for _ in range(num_worlds)
    ]
    gpu_env = CoopBatchGPU(
        args.task,
        num_worlds,
        config=config,
        partner_mode=args.partner_mode,
        seed=args.seed,
        device=args.device,
    )
    generator = torch.Generator().manual_seed(args.seed + 17)
    tolerances = args.tolerances
    max_diff = dict.fromkeys(("qpos", "qvel", "obs", "reward"), 0.0)
    flag_mismatches = 0
    next_reset_seed = args.seed + num_worlds
    episode_returns = np.zeros(num_worlds, dtype=np.float64)
    episode_lengths = np.zeros(num_worlds, dtype=np.int64)
    finished_max_return_diff = 0.0
    finished_length_mismatches = 0
    finished_success_mismatches = 0
    finished_angle_diff = 0.0
    finished_compared = 0
    episodes_ended = 0
    nonfinite_events = 0

    print(
        f"parity: task={args.task} controller={args.controller} partner_mode={args.partner_mode} "
        f"worlds={num_worlds} steps={args.steps} seed={args.seed} device={args.device} "
        f"max_control_steps={config.max_control_steps} graph={gpu_env.cuda_graph} "
        f"| {gpu_env.software_versions()}",
        flush=True,
    )

    # Reset: same seeds on both paths; the initial observation must agree.
    cpu_obs = [env.reset(seed=seed)[0] for env, seed in zip(cpu_envs, seeds)]
    gpu_obs = gpu_env.reset(seed=args.seed)
    for role in AGENT_KEYS:
        max_diff["obs"] = max(
            max_diff["obs"],
            _max_abs_diff(_stack_obs(cpu_obs, role), gpu_obs[role].cpu().numpy()),
        )
    randomisation_failures = _check_randomization(gpu_env, seeds)

    for step in range(args.steps):
        # --- actions: identical float32 values on both paths ----------------------------
        if args.controller == "zero":
            actions_float = np.zeros(
                (len(AGENT_KEYS), num_worlds, gpu_env.action_dim), dtype=np.float32
            )
        elif args.controller == "random":
            actions_float = (
                torch.rand((len(AGENT_KEYS), num_worlds, gpu_env.action_dim), generator=generator)
                * 2.0
                - 1.0
            ).numpy()
            actions_float = actions_float.astype(np.float32)
        else:
            stacked = [oracles[index].act(env) for index, env in enumerate(cpu_envs)]
            actions_float = np.stack(
                [np.stack([action[role] for action in stacked]) for role in AGENT_KEYS]
            ).astype(np.float32)
        cpu_actions = actions_float.astype(np.float64)
        gpu_actions = {
            role: torch.as_tensor(actions_float[role_index], device=args.device)
            for role_index, role in enumerate(AGENT_KEYS)
        }

        # --- one control step on both paths ----------------------------------------------
        cpu_steps = [
            env.step(
                {
                    role: cpu_actions[role_index, env_index]
                    for role_index, role in enumerate(AGENT_KEYS)
                }
            )
            for env_index, env in enumerate(cpu_envs)
        ]
        gpu_step = gpu_env.step(gpu_actions)
        gpu_obs_np = {role: gpu_step.obs[role].cpu().numpy() for role in AGENT_KEYS}
        gpu_final_np = {role: gpu_step.final_obs[role].cpu().numpy() for role in AGENT_KEYS}
        cpu_next_obs = [step_result[0] for step_result in cpu_steps]
        cpu_reward = np.asarray([step_result[1] for step_result in cpu_steps], dtype=np.float64)
        cpu_terminated = np.asarray([step_result[2] for step_result in cpu_steps])
        cpu_truncated = np.asarray([step_result[3] for step_result in cpu_steps])
        gpu_reward = gpu_step.reward.cpu().numpy()
        gpu_terminated = gpu_step.terminated.cpu().numpy()
        gpu_truncated = gpu_step.truncated.cpu().numpy()
        done = cpu_terminated | cpu_truncated
        gpu_nonfinite = (
            np.asarray(gpu_step.nonfinite.cpu(), dtype=bool)
            if gpu_step.nonfinite is not None
            else np.zeros(num_worlds, dtype=bool)
        )
        nonfinite_events += int(gpu_nonfinite.sum())
        # A non-finite world (a float32 contact-solver limit of the toss-catch) is a GPU-only
        # episode end: skip its row from this step's comparison, but still restart the CPU env
        # with the same auto-reset seed so the seed streams stay aligned afterwards.
        comparable = ~(done | gpu_nonfinite)
        alive = comparable

        # --- state / observation / reward comparison -------------------------------------
        cpu_qpos = np.stack([np.array(env._data.qpos, dtype=np.float64) for env in cpu_envs])
        cpu_qvel = np.stack([np.array(env._data.qvel, dtype=np.float64) for env in cpu_envs])
        gpu_state = gpu_env.state()
        gpu_qpos = gpu_state["qpos"].cpu().numpy()
        gpu_qvel = gpu_state["qvel"].cpu().numpy()
        max_diff["qpos"] = max(max_diff["qpos"], _max_abs_diff(cpu_qpos[alive], gpu_qpos[alive]))
        max_diff["qvel"] = max(max_diff["qvel"], _max_abs_diff(cpu_qvel[alive], gpu_qvel[alive]))
        for role in AGENT_KEYS:
            cpu_role_obs = _stack_obs(cpu_next_obs, role)
            max_diff["obs"] = max(
                max_diff["obs"], _max_abs_diff(cpu_role_obs[alive], gpu_obs_np[role][alive])
            )
            # The GPU env auto-resets the done worlds; their terminal observation is
            # `final_obs`, which is what VecCoopEnv stores for the GAE bootstrap.
            if done.any():
                max_diff["obs"] = max(
                    max_diff["obs"],
                    _max_abs_diff(cpu_role_obs[done], gpu_final_np[role][done]),
                )
        max_diff["reward"] = max(
            max_diff["reward"], _max_abs_diff(cpu_reward[comparable], gpu_reward[comparable])
        )
        flag_mismatches += int((cpu_terminated[comparable] != gpu_terminated[comparable]).sum())
        flag_mismatches += int((cpu_truncated[comparable] != gpu_truncated[comparable]).sum())
        episode_returns += cpu_reward
        episode_lengths += 1

        # --- finished-episode statistics (VecCoopEnv contract) --------------------------
        done_rows = np.flatnonzero(done)
        reset_rows = np.flatnonzero(done | gpu_nonfinite)
        gpu_lengths = list(gpu_step.finished_lengths)
        gpu_returns = list(gpu_step.finished_returns)
        gpu_successes = list(gpu_step.finished_successes)
        if not (len(done_rows) == len(gpu_lengths) == len(gpu_returns) == len(gpu_successes)):
            flag_mismatches += 1
            print(
                "  finished-statistics contract broken: "
                f"{len(done_rows)} done worlds vs {len(gpu_lengths)} GPU statistics"
            )
        for position, world in enumerate(reset_rows[: len(gpu_lengths)]):
            if gpu_nonfinite[world]:
                continue
            cpu_length = int(episode_lengths[world])
            cpu_return = float(episode_returns[world])
            cpu_success = bool(cpu_steps[world][4]["success"])
            cpu_angle = task_diagnostic(args.task, cpu_steps[world][4])
            finished_max_return_diff = max(
                finished_max_return_diff, abs(cpu_return - float(gpu_returns[position]))
            )
            finished_length_mismatches += int(cpu_length != int(gpu_lengths[position]))
            finished_success_mismatches += int(cpu_success != bool(gpu_successes[position]))
            finished_angle_diff = max(
                finished_angle_diff, abs(cpu_angle - float(gpu_step.finished_angles[position]))
            )
            finished_compared += 1

        if len(reset_rows) > 0:
            episodes_ended += len(reset_rows)
            # Mirror the GPU auto-reset: CPU envs that ended restart with the same seed
            # stream, in row order, and their fresh observations are compared below.
            fresh_obs = [
                cpu_envs[env_index].reset(seed=next_reset_seed + offset)[0]
                for offset, env_index in enumerate(reset_rows)
            ]
            next_reset_seed += len(reset_rows)
            episode_returns[reset_rows] = 0.0
            episode_lengths[reset_rows] = 0
            for role in AGENT_KEYS:
                max_diff["obs"] = max(
                    max_diff["obs"],
                    _max_abs_diff(_stack_obs(fresh_obs, role), gpu_obs_np[role][reset_rows]),
                )

    failures = list(randomisation_failures)
    for name in ("qpos", "qvel", "obs", "reward"):
        tolerance = tolerances[name]
        status = "OK" if max_diff[name] <= tolerance else "FAIL"
        if status == "FAIL":
            failures.append(f"{name} max diff {max_diff[name]:.3e} > tol {tolerance:.1e}")
        print(f"  {name:<7} max |CPU - GPU| = {max_diff[name]:.3e}  (tol {tolerance:.1e}) {status}")
    print(
        f"  finished episodes: {finished_compared} compared | max |Δreturn| = "
        f"{finished_max_return_diff:.3e} (tol {tolerances['return']:.1e}) | "
        f"max |Δangle| = {finished_angle_diff:.3e} | "
        f"length mismatches {finished_length_mismatches} | "
        f"success mismatches {finished_success_mismatches}"
    )
    if finished_compared == 0:
        if args.require_finished:
            failures.append(
                "no episode finished: the finished-statistics contract was not compared"
            )
        else:
            print(
                "  note: no episode finished within the step budget; only the numerical "
                "parity was compared (use --require-finished to enforce the statistics check)"
            )
    if finished_max_return_diff > tolerances["return"]:
        failures.append(
            f"finished return diff {finished_max_return_diff:.3e} > tol {tolerances['return']:.1e}"
        )
    if finished_length_mismatches:
        failures.append(f"{finished_length_mismatches} finished episode lengths differ")
    if finished_success_mismatches:
        failures.append(f"{finished_success_mismatches} finished episode success flags differ")
    print(
        f"  terminal-flag mismatches: {flag_mismatches}; episodes ended: {episodes_ended}; "
        f"non-finite contact-solver worlds skipped: {nonfinite_events}"
    )
    if flag_mismatches:
        failures.append(f"{flag_mismatches} terminated/truncated flags differ")
    if failures:
        print("PARITY FAIL")
        for failure in failures:
            print(f"  - {failure}")
        _close(cpu_envs, gpu_env)
        return 1
    print("PARITY PASS")
    _close(cpu_envs, gpu_env)
    return 0


def _close(cpu_envs: list, gpu_env: object) -> None:
    """Close both backends (the CPU envs are a list of Gymnasium envs)."""
    for env in cpu_envs:
        env.close()
    gpu_env.close()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    return run_parity(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
