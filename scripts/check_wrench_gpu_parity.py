"""Step-by-step parity check: the CPU `WrenchCubeEnv` vs the batched `WrenchCubeBatchGPU`.

The GPU backend re-implements the CPU env's control step in tensor algebra (latency queues,
start-delay / slip injections, `xfrc_applied`, 8 physics substeps, the observation layout,
success/failure/reward, per-episode layer friction, auto-reset). This script is the proof
that the two paths are the same environment: `--num-worlds` CPU envs and one GPU batch env are
reset with the SAME episode seeds and fed the SAME actions, and every step's qpos, qvel,
observation, reward and termination flags are compared. The only intended difference is the
float32 Warp physics against the float64 MuJoCo CPU path, so the tolerances are small but not
zero (`qpos` 1e-5, `obs` 1e-4 by default; qvel/reward 1e-4).

What is compared:

* the reset observations (should be identical: same sampler, same layout);
* after every step: the full `qpos` (8) and `qvel` (7) of every NON-done world, both agents'
  flat observations, the shared reward and the success/failure/truncation flags;
* on the step a world terminates: its terminal flags/reward and the GPU `final_obs` against
  the CPU env's last observation (the truncation-bootstrap contract of `VecWrenchCube`), and
  the GPU's post-auto-reset observation against a CPU env restarted with the same seed (the
  wrapper's seed stream is mirrored: world `i` starts from `seed + i`, auto-resets continue
  from `seed + num_worlds`);
* the sampled randomisation tensors against a fresh CPU call of the same sampler, so a seed
  mapping to different episodes on the two paths cannot slip through.

Controllers: `random` (torch-random normalized wrenches, physics/obs parity) and `oracle`
(the scripted `OracleController` acting on the CPU envs, which drives the episode to a real
success/failure so the terminal reward and the final observation are compared too).

Randomisation: `--randomization default|harsh` selects the episode-randomness preset of
BOTH envs (`callosum.experiments.cube_wrench.config.wrench_cube_config`); the physics/timing/reward are
identical between the presets, only the private-randomness ranges differ.

Friction: `--friction sampled` (the production path: the per-episode Coulomb friction the CPU
env writes into `model.dof_frictionloss` is reproduced per world by
`put_model(batch_sizes={"dof_frictionloss": nworld})`) or `--friction disabled` (both paths
built from `WrenchCubeConfig(hinge_frictionloss=0, layer_frictionloss_range=(0, 0))`; a
numerical-isolation mode for the random-action stress test).

Measured on the pod (A40, mujoco_warp 4.0.1, 8 worlds, 2026-11-21):

* `--controller oracle --friction sampled` (120 steps, 15 mirrored auto-resets): qpos 1.5e-6,
  qvel 6.8e-7, obs 1.4e-6, reward 4.1e-6 -- PASS. This is the production path (per-world
  sampled Coulomb friction + latency + injections) over whole episodes, so the CPU and GPU
  environments are the same for policy-like trajectories;
* `--controller oracle --friction sampled --partner-mode none` (60 steps): qpos 9.5e-7,
  obs 9.5e-7 -- PASS (the `none` observation block has no partner leakage);
* `--controller random --friction disabled` (120 steps, 74 mirrored auto-resets): qpos 1.3e-7,
  qvel 1.9e-6, obs 1.9e-6, reward 3.4e-6 -- PASS (the auto-reset seed stream, observations,
  rewards and flags agree for several episodes per world);
* `--controller random --friction sampled` (60 steps): qpos 1.7e-4, qvel 8.5e-4, obs 1.4e-3
  -- FAIL, and NOT a semantics difference: the Coulomb-friction constraint is non-smooth (its
  direction flips with the hinge velocity), and the ~1e-6 float32-vs-float64 state difference
  decides WHICH substep sees the flip, i.e. a discontinuous impulse difference of
  `frictionloss * dt / armature` order. Random wrenches reverse the hinge direction every few
  steps, so they hit that switch immediately; the oracle/policy-like path does not (bullet 1).
  The `friction disabled` run isolates the effect: with the same controllers/pipeline but the
  only non-smooth constraint removed, agreement returns to 1e-7. MuJoCo-Warp is float32-only
  (the C MuJoCo reference here is float64), so this is an expected numerical limit of the GPU
  backend for adversarial actions, documented rather than worked around.

The GPU backend needs the optional `mujoco-warp==4.0.1` install (the `sandbox-gpu` extra,
Linux/CUDA only; the pod's plain `uv sync --frozen` removes it), so on the pod run:

    uv pip install --python /root/.callosum/venv/bin/python mujoco-warp==4.0.1
    python scripts/check_wrench_gpu_parity.py --controller random --partner-mode oracle
    python scripts/check_wrench_gpu_parity.py --controller oracle --partner-mode oracle

Exit code 0 when every comparison is inside its tolerance, 1 otherwise.
"""

import argparse
import dataclasses
import sys

import numpy as np

from callosum.experiments.cube_wrench.config import RANDOMIZATION_PRESETS, wrench_cube_config

TOLERANCES = {
    "qpos": 1e-5,
    "qvel": 1e-4,
    "obs": 1e-4,
    "reward": 1e-4,
}
"""Max-abs tolerances of the compared quantities (float32 Warp vs float64 MuJoCo)."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the parity-check CLI."""
    parser = argparse.ArgumentParser(
        description="Compare WrenchCubeBatchGPU against the CPU WrenchCubeEnv step by step."
    )
    parser.add_argument("--num-worlds", type=int, default=8, help="worlds compared")
    parser.add_argument(
        "--steps", type=int, default=60, help="maximum control steps (stops at the first end)"
    )
    parser.add_argument("--seed", type=int, default=0, help="base episode seed")
    parser.add_argument(
        "--partner-mode", choices=("none", "oracle"), default="oracle", help="env partner mode"
    )
    parser.add_argument(
        "--controller",
        choices=("random", "oracle"),
        default="oracle",
        help="action source: the scripted oracle (default, policy-like) or random wrenches",
    )
    parser.add_argument(
        "--randomization",
        choices=RANDOMIZATION_PRESETS,
        default="default",
        help=(
            "episode-randomness preset of both envs (default: the dataclass defaults; harsh: "
            "the harder private-randomness ranges of callosum.experiments.cube_wrench.config)"
        ),
    )
    parser.add_argument(
        "--friction",
        choices=("sampled", "disabled"),
        default="sampled",
        help=(
            "sampled: the production per-episode Coulomb friction; disabled: both paths use a "
            "zero hinge friction (numerical isolation of the random-action stress test)"
        ),
    )
    parser.add_argument("--device", default="cuda", help="CUDA device of the GPU env")
    parser.add_argument(
        "--max-control-steps",
        type=int,
        default=None,
        help="override the episode budget (e.g. 20 to force time-limit truncations)",
    )
    parser.add_argument(
        "--force-slip",
        action="store_true",
        help="force slip_prob_range=(1, 1) so every agent slips every episode",
    )
    parser.add_argument(
        "--slip-decisions",
        type=int,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="override slip_decisions_range (e.g. 2 2)",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="override start_delay_range (e.g. 0.2 0.2)",
    )
    parser.add_argument(
        "--force-impulse",
        action="store_true",
        help="force impulse_prob_range=(1, 1) so every episode has an impulse",
    )
    parser.add_argument("--qpos-tol", type=float, default=TOLERANCES["qpos"])
    parser.add_argument("--qvel-tol", type=float, default=TOLERANCES["qvel"])
    parser.add_argument("--obs-tol", type=float, default=TOLERANCES["obs"])
    parser.add_argument("--reward-tol", type=float, default=TOLERANCES["reward"])
    parser.add_argument(
        "--return-tol",
        type=float,
        default=1e-4,
        help="max |CPU - GPU| of a finished episode's undiscounted return",
    )
    args = parser.parse_args(argv)
    if args.num_worlds < 1:
        parser.error(f"--num-worlds must be >= 1, got {args.num_worlds}")
    if args.steps < 1:
        parser.error(f"--steps must be >= 1, got {args.steps}")
    if args.seed < 0:
        parser.error(f"--seed must be >= 0, got {args.seed}")
    return args


def _cpu_state(env: object) -> tuple[np.ndarray, np.ndarray]:
    """`(qpos (8,), qvel (7,))` of a CPU env reconstructed from its public `state()`."""
    state = env.state()
    qpos = np.concatenate(
        [
            np.asarray(state["core_pos"], dtype=np.float64),
            np.asarray(state["core_quat"], dtype=np.float64),
            [float(state["layer_angle"])],
        ]
    )
    qvel = np.concatenate(
        [
            np.asarray(state["core_lin_vel"], dtype=np.float64),
            np.asarray(state["core_ang_vel"], dtype=np.float64),
            [float(state["layer_ang_vel"])],
        ]
    )
    return qpos, qvel


def _stack_obs(observations: list[dict], role: str) -> np.ndarray:
    """The CPU envs' per-role observations as one `(num_worlds, obs_dim)` array."""
    return np.stack([np.asarray(obs[role], dtype=np.float32) for obs in observations])


def _max_abs_diff(expected: np.ndarray, actual: np.ndarray) -> float:
    """Max abs difference; `0.0` for empty inputs (all rows of the batch can be done)."""
    if expected.size == 0:
        return 0.0
    return float(np.abs(expected.astype(np.float64) - actual.astype(np.float64)).max())


def _check_randomization(env: object, seeds: list[int]) -> list[str]:
    """Compare the GPU randomisation tensors with fresh CPU samples of the same seeds."""
    from callosum.experiments.cube_wrench.logic import AGENT_KEYS, sample_episode_randomization

    failures: list[str] = []
    agents = env.randomization_tensors()
    episode = env.episode_randomization_tensors()
    records = [sample_episode_randomization(env.config, seed) for seed in seeds]
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
    for name, tensor in episode.items():
        if name == "impulse_force":
            expected = np.asarray([record.impulse_force for record in records])
        else:
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

    from callosum.experiments.cube_wrench import OracleController, WrenchCubeEnv
    from callosum.experiments.cube_wrench.env_gpu import WrenchCubeBatchGPU

    torch.manual_seed(args.seed)
    roles = ("holder", "rotator")
    num_worlds = args.num_worlds
    seeds = [args.seed + index for index in range(num_worlds)]
    config = wrench_cube_config(args.randomization)
    # Edge-case overrides: force truncation / injections so the parity run exercises the
    # time-limit path, guaranteed slips, fixed start delays and the impulse in every episode.
    overrides: dict = {}
    if args.max_control_steps is not None:
        overrides["max_control_steps"] = args.max_control_steps
    if args.force_slip:
        overrides["slip_prob_range"] = (1.0, 1.0)
    if args.slip_decisions is not None:
        overrides["slip_decisions_range"] = tuple(args.slip_decisions)
    if args.start_delay is not None:
        overrides["start_delay_range"] = tuple(args.start_delay)
    if args.force_impulse:
        overrides["impulse_prob_range"] = (1.0, 1.0)
    if overrides:
        config = dataclasses.replace(config, **overrides)
    if args.friction == "disabled":
        config = dataclasses.replace(
            config, hinge_frictionloss=0.0, layer_frictionloss_range=(0.0, 0.0)
        )

    cpu_envs = [
        WrenchCubeEnv(config=config, partner_mode=args.partner_mode) for _ in range(num_worlds)
    ]
    gpu_env = WrenchCubeBatchGPU(
        num_worlds,
        config=config,
        partner_mode=args.partner_mode,
        seed=args.seed,
        device=args.device,
    )
    oracle = OracleController(cpu_envs[0].config)
    generator = torch.Generator().manual_seed(args.seed + 17)
    tolerances = {
        "qpos": args.qpos_tol,
        "qvel": args.qvel_tol,
        "obs": args.obs_tol,
        "reward": args.reward_tol,
    }

    print(
        f"parity: controller={args.controller} friction={args.friction} "
        f"randomization={args.randomization} partner_mode={args.partner_mode} "
        f"worlds={num_worlds} seed={args.seed} device={args.device} "
        f"max_control_steps={config.max_control_steps} "
        f"slip_prob={config.slip_prob_range} start_delay={config.start_delay_range} "
        f"impulse_prob={config.impulse_prob_range}",
        flush=True,
    )
    max_diff = dict.fromkeys(("qpos", "qvel", "obs", "reward"), 0.0)
    flag_mismatches = 0
    # The GPU wrapper's auto-reset seed stream (same contract as VecWrenchCube): the
    # constructor pre-computes `seed + nworld` and every auto-reset consumes the next seed.
    next_reset_seed = args.seed + num_worlds
    finished_episodes = 0
    last_end_step: int | None = None
    # Episode accumulators for the finished-statistics comparison (the trainer's
    # finished_returns/lengths/successes contract is not covered by the state/obs comparison).
    episode_returns = np.zeros(num_worlds, dtype=np.float64)
    episode_lengths = np.zeros(num_worlds, dtype=np.int64)
    finished_max_return_diff = 0.0
    finished_length_mismatches = 0
    finished_success_mismatches = 0
    finished_compared = 0

    # Reset: same seeds on both paths; the initial observation must agree.
    cpu_obs = [env.reset(seed=seed)[0] for env, seed in zip(cpu_envs, seeds)]
    gpu_obs = gpu_env.reset(seed=seeds[0])
    for role in roles:
        max_diff["obs"] = max(
            max_diff["obs"],
            _max_abs_diff(_stack_obs(cpu_obs, role), gpu_obs[role].cpu().numpy()),
        )
    randomisation_failures = _check_randomization(gpu_env, seeds)

    for step in range(args.steps):
        # --- actions: identical float32 values on both paths ----------------------------
        if args.controller == "random":
            raw = (torch.rand((num_worlds, 6), generator=generator) * 2.0 - 1.0).numpy()
            actions_float = np.stack([raw.astype(np.float32)] * len(roles))
        else:
            actions_float = np.stack(
                [np.stack([oracle.act(env)[role] for env in cpu_envs]) for role in roles]
            ).astype(np.float32)
        cpu_actions = actions_float.astype(np.float64)
        gpu_actions = {
            role: torch.as_tensor(actions_float[index], device=args.device)
            for index, role in enumerate(roles)
        }

        # --- one control step on both paths ----------------------------------------------
        cpu_steps = [
            env.step(
                {role: cpu_actions[role_index, env_index] for role_index, role in enumerate(roles)}
            )
            for env_index, env in enumerate(cpu_envs)
        ]
        gpu_step = gpu_env.step(gpu_actions)
        gpu_obs_np = {role: gpu_step.obs[role].cpu().numpy() for role in roles}
        gpu_final_np = {role: gpu_step.final_obs[role].cpu().numpy() for role in roles}
        cpu_next_obs = [step_result[0] for step_result in cpu_steps]
        cpu_reward = np.asarray([step_result[1] for step_result in cpu_steps], dtype=np.float64)
        cpu_terminated = np.asarray([step_result[2] for step_result in cpu_steps])
        cpu_truncated = np.asarray([step_result[3] for step_result in cpu_steps])
        gpu_reward = gpu_step.reward.cpu().numpy()
        gpu_terminated = gpu_step.terminated.cpu().numpy()
        gpu_truncated = gpu_step.truncated.cpu().numpy()
        done = cpu_terminated | cpu_truncated
        alive = ~done

        # --- state / observation / reward comparison -------------------------------------
        cpu_qpos = np.stack([_cpu_state(env)[0] for env in cpu_envs])
        cpu_qvel = np.stack([_cpu_state(env)[1] for env in cpu_envs])
        gpu_state = gpu_env.state()
        gpu_qpos = gpu_state["qpos"].cpu().numpy()
        gpu_qvel = gpu_state["qvel"].cpu().numpy()
        max_diff["qpos"] = max(max_diff["qpos"], _max_abs_diff(cpu_qpos[alive], gpu_qpos[alive]))
        max_diff["qvel"] = max(max_diff["qvel"], _max_abs_diff(cpu_qvel[alive], gpu_qvel[alive]))
        for role in roles:
            cpu_role_obs = _stack_obs(cpu_next_obs, role)
            max_diff["obs"] = max(
                max_diff["obs"],
                _max_abs_diff(cpu_role_obs[alive], gpu_obs_np[role][alive]),
            )
            # The GPU env auto-resets the done worlds; their terminal observation is
            # `final_obs`, which is what VecWrenchCube stores for the GAE bootstrap.
            if done.any():
                max_diff["obs"] = max(
                    max_diff["obs"],
                    _max_abs_diff(cpu_role_obs[done], gpu_final_np[role][done]),
                )
        max_diff["reward"] = max(max_diff["reward"], _max_abs_diff(cpu_reward, gpu_reward))
        flag_mismatches += int((cpu_terminated != gpu_terminated).sum())
        flag_mismatches += int((cpu_truncated != gpu_truncated).sum())
        episode_returns += cpu_reward
        episode_lengths += 1

        # --- finished-episode statistics (VecWrenchCube contract) ---------------------
        done_rows = np.flatnonzero(done)
        gpu_lengths = list(gpu_step.finished_lengths)
        gpu_returns = list(gpu_step.finished_returns)
        gpu_successes = list(gpu_step.finished_successes)
        if not (len(done_rows) == len(gpu_lengths) == len(gpu_returns) == len(gpu_successes)):
            flag_mismatches += 1
            print(
                "  finished-statistics contract broken: "
                f"{len(done_rows)} done worlds vs {len(gpu_lengths)} GPU statistics"
            )
        for position, world in enumerate(done_rows[: len(gpu_lengths)]):
            cpu_length = int(episode_lengths[world])
            cpu_return = float(episode_returns[world])
            cpu_success = bool(cpu_steps[world][4]["success"])
            finished_max_return_diff = max(
                finished_max_return_diff, abs(cpu_return - float(gpu_returns[position]))
            )
            finished_length_mismatches += int(cpu_length != int(gpu_lengths[position]))
            finished_success_mismatches += int(cpu_success != bool(gpu_successes[position]))
            finished_compared += 1

        if done.any():
            finished_episodes += int(done.sum())
            last_end_step = step
            # Mirror the GPU auto-reset: CPU envs that ended restart with the same seed
            # stream, in row order, and their fresh observations are compared below.
            done_rows = np.flatnonzero(done)
            fresh_obs = [
                cpu_envs[env_index].reset(seed=next_reset_seed + offset)[0]
                for offset, env_index in enumerate(done_rows)
            ]
            next_reset_seed += len(done_rows)
            episode_returns[done_rows] = 0.0
            episode_lengths[done_rows] = 0
            for role in roles:
                cpu_fresh = _stack_obs(fresh_obs, role)
                max_diff["obs"] = max(
                    max_diff["obs"],
                    _max_abs_diff(cpu_fresh, gpu_obs_np[role][done]),
                )

    failures = list(randomisation_failures)
    for name, tolerance in tolerances.items():
        status = "OK" if max_diff[name] <= tolerance else "FAIL"
        if status == "FAIL":
            failures.append(f"{name} max diff {max_diff[name]:.3e} > tol {tolerance:.1e}")
        print(f"  {name:<7} max |CPU - GPU| = {max_diff[name]:.3e}  (tol {tolerance:.1e}) {status}")
    print(
        f"  finished episodes: {finished_compared} compared | max |Δreturn| = "
        f"{finished_max_return_diff:.3e} | length mismatches {finished_length_mismatches} | "
        f"success mismatches {finished_success_mismatches}"
    )
    if finished_compared == 0:
        failures.append("no episode finished: the finished-statistics contract was not compared")
    if finished_max_return_diff > args.return_tol:
        failures.append(
            f"finished return diff {finished_max_return_diff:.3e} > tol {args.return_tol:.1e}"
        )
    if finished_length_mismatches:
        failures.append(f"{finished_length_mismatches} finished episode lengths differ")
    if finished_success_mismatches:
        failures.append(f"{finished_success_mismatches} finished episode success flags differ")
    print(f"  terminal-flag mismatches: {flag_mismatches}")
    if flag_mismatches:
        failures.append(f"{flag_mismatches} terminated/truncated flags differ")
    if last_end_step is None:
        print(f"  no episode ended within {args.steps} steps (compared {args.steps} steps)")
    else:
        print(
            f"  {finished_episodes} episode(s) ended (last at step {last_end_step}); the "
            f"auto-reset seed stream was mirrored for {finished_episodes} world restart(s)"
        )
    if failures:
        print("PARITY FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PARITY PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    return run_parity(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
