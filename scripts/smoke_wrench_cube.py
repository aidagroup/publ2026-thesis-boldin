"""[local] Smoke-check CubeWrench-v0: 1-2 envs, short scripted / random / zero rollouts.

The MuJoCo reduced model is CPU-only, so this is the ONE local simulation check the
project allows (1-2 envs, short episodes, no training). It creates the envs one after
another (the env is single-instance; the trainer will vectorise later), runs `--episodes`
episodes per env and prints, per episode, the resulting success / angle / core drift /
steps, then the aggregate.

Controllers (`--controller`):

* `oracle` -- the scripted two-agent oracle (holder PD stabilisation + rotator PD turn).
  This is the environment sanity gate: mean success must be >= 0.8, otherwise exit 1.
* `rotator-only` -- the NON-ADDITIVITY check: the same oracle rotator moment, but the
  holder command is forced to zero. Expected: success ~0 and the core rotation drift above
  the config's failure bound (the rotator alone cannot keep the core still). This is
  printed on its own line and exits 1 if the expectation fails. In gate G1 it runs next to
  `oracle`.
* `random` -- `env.action_space.sample()`; `--seed` seeds `np.random` and each env's
  action-space generator, so random runs are reproducible.
* `zero` -- the all-zero action (do nothing).

Exit code: 0 when the selected controller meets its expectation (the `random`/`zero`
controllers are diagnostics and always exit 0), 1 otherwise.

    ./.venv/bin/python scripts/smoke_wrench_cube.py --envs 2 --episodes 10 --controller oracle
    ./.venv/bin/python scripts/smoke_wrench_cube.py --envs 2 --episodes 10 --controller rotator-only
"""

import argparse
import sys

import numpy as np

from callosum.envs.wrench_cube import OracleController, WrenchCubeEnv

_ORACLE_SUCCESS_FLOOR = 0.8
_ROTATOR_ONLY_SUCCESS_CEIL = 0.1
"""The rotator alone must (almost) never succeed; 0.1 allows a little local noise."""


def parse_args() -> argparse.Namespace:
    """Parse the smoke-check CLI (env count, episodes, controller kind, base seed)."""
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-check CubeWrench-v0 with the scripted oracle, the rotator-only "
            "non-additivity check, random or zero actions."
        )
    )
    parser.add_argument("--envs", type=int, default=2, help="envs to build sequentially")
    parser.add_argument("--episodes", type=int, default=5, help="episodes per env")
    parser.add_argument(
        "--controller",
        choices=("oracle", "rotator-only", "random", "zero"),
        default="oracle",
        help=(
            "action source: 'oracle' (sanity gate), 'rotator-only' (holder forced to zero, "
            "non-additivity check), 'random' or 'zero' (diagnostics)"
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="base episode seed")
    return parser.parse_args()


def _episode_seed(base_seed: int, env_index: int, episode: int, episodes: int) -> int:
    """The deterministic seed of one smoke episode."""
    return base_seed + env_index * episodes + episode


def _zero_action(env: WrenchCubeEnv) -> dict[str, np.ndarray]:
    """The all-zero (do-nothing) action for the given env."""
    return {key: np.zeros(6, dtype=np.float32) for key in env.action_space.spaces}


def _controller_action(
    env: WrenchCubeEnv, controller: OracleController, kind: str
) -> dict[str, np.ndarray]:
    """The action of one step for the requested controller kind."""
    if kind == "oracle":
        return controller.act(env)
    if kind == "rotator-only":
        action = controller.act(env)
        action["holder"] = np.zeros(6, dtype=np.float32)
        return action
    if kind == "random":
        return env.action_space.sample()
    return _zero_action(env)


def run_episode(
    env: WrenchCubeEnv,
    controller: OracleController,
    kind: str,
    seed: int,
) -> tuple[dict, int]:
    """Run one episode to termination/truncation and return its info and step count."""
    env.reset(seed=seed)
    steps = 0
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        action = _controller_action(env, controller, kind)
        _obs, _reward, terminated, truncated, info = env.step(action)
        steps += 1
    return info, steps


def main() -> int:
    """Run the smoke check; return the process exit code."""
    args = parse_args()
    if args.envs < 1 or args.episodes < 1:
        print(f"FAIL: --envs and --episodes must be >= 1, got {args.envs}/{args.episodes}")
        return 2
    if args.controller == "random":
        # Gymnasium's `action_space.sample()` draws from the space's OWN generator, not
        # the global `np.random`, so both are seeded for a reproducible random baseline.
        np.random.seed(args.seed)
    successes: list[float] = []
    angles: list[float] = []
    drift_pos: list[float] = []
    drift_rot: list[float] = []
    config = None
    for env_index in range(args.envs):
        env = WrenchCubeEnv()
        try:
            if args.controller == "random":
                env.action_space.seed(args.seed + env_index)
            config = env.config
            controller = OracleController(env.config)
            for episode in range(args.episodes):
                seed = _episode_seed(args.seed, env_index, episode, args.episodes)
                info, steps = run_episode(env, controller, args.controller, seed)
                successes.append(float(info["success"]))
                angles.append(float(info["angle_deg"]))
                drift_pos.append(float(info["drift_pos"]))
                drift_rot.append(float(info["drift_rot"]))
                print(
                    f"env {env_index} episode {episode}: "
                    f"success={bool(info['success'])} "
                    f"angle_deg={info['angle_deg']:.1f} "
                    f"drift_mm={info['drift_pos'] * 1e3:.1f} "
                    f"drift_deg={np.degrees(info['drift_rot']):.1f} "
                    f"steps={steps}"
                )
        finally:
            env.close()
    mean_success = float(np.mean(successes))
    mean_drift_rot = float(np.mean(drift_rot))
    print(
        f"summary: controller={args.controller} envs={args.envs} episodes={args.episodes} "
        f"success={mean_success:.3f} "
        f"angle_deg={np.mean(angles):.1f} "
        f"drift_mm={np.mean(drift_pos) * 1e3:.1f} "
        f"drift_deg={np.degrees(mean_drift_rot):.1f}"
    )
    if args.controller == "oracle":
        if mean_success < _ORACLE_SUCCESS_FLOOR:
            print(f"FAIL: oracle mean success {mean_success:.3f} < {_ORACLE_SUCCESS_FLOOR}")
            return 1
        return 0
    if args.controller == "rotator-only":
        failure_bound = config.failure_rot_drift if config is not None else np.inf
        success_ok = mean_success <= _ROTATOR_ONLY_SUCCESS_CEIL
        drift_ok = mean_drift_rot > failure_bound
        print(
            f"rotator-only check: success={mean_success:.3f} "
            f"(expected <= {_ROTATOR_ONLY_SUCCESS_CEIL}), "
            f"drift_deg={np.degrees(mean_drift_rot):.1f} "
            f"(expected > failure_deg={np.degrees(failure_bound):.1f}) -> "
            f"{'PASS' if success_ok and drift_ok else 'FAIL'}"
        )
        return 0 if success_ok and drift_ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
