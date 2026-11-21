"""[local] Smoke-check CubeWrench-v0: build 1-2 envs and run oracle / random / zero rollouts.

The MuJoCo reduced model is CPU-only, so this is the ONE local simulation check the
project allows (1-2 envs, short episodes, no training). It creates the envs one after
another (the env is single-instance; the trainer will vectorise later), runs `--episodes`
episodes per env and prints, per episode, the resulting success / angle / core drift /
steps, then the aggregate.

Exit code: 0 when the oracle's mean success is >= 0.8 (the environment's local sanity
gate), 1 when the oracle fails that gate; the random/zero controllers are diagnostics and
always exit 0.

    ./.venv/bin/python scripts/smoke_wrench_cube.py --envs 2 --episodes 5 --controller oracle
"""

import argparse
import sys

import numpy as np

from callosum.envs.wrench_cube import OracleController, WrenchCubeEnv

_ORACLE_SUCCESS_FLOOR = 0.8


def parse_args() -> argparse.Namespace:
    """Parse the smoke-check CLI (env count, episodes, controller kind, base seed)."""
    parser = argparse.ArgumentParser(
        description="Smoke-check CubeWrench-v0 with a scripted oracle, random or zero actions."
    )
    parser.add_argument("--envs", type=int, default=2, help="envs to build sequentially")
    parser.add_argument("--episodes", type=int, default=5, help="episodes per env")
    parser.add_argument(
        "--controller",
        choices=("oracle", "random", "zero"),
        default="oracle",
        help="action source of the episodes",
    )
    parser.add_argument("--seed", type=int, default=0, help="base episode seed")
    return parser.parse_args()


def _episode_seed(base_seed: int, env_index: int, episode: int, episodes: int) -> int:
    """The deterministic seed of one smoke episode."""
    return base_seed + env_index * episodes + episode


def _zero_action(env: WrenchCubeEnv) -> dict[str, np.ndarray]:
    """The all-zero (do-nothing) action for the given env."""
    return {key: np.zeros(6, dtype=np.float32) for key in env.action_space.spaces}


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
        if kind == "oracle":
            action = controller.act(env)
        elif kind == "random":
            action = env.action_space.sample()
        else:
            action = _zero_action(env)
        _obs, _reward, terminated, truncated, info = env.step(action)
        steps += 1
    return info, steps


def main() -> int:
    """Run the smoke check; return the process exit code."""
    args = parse_args()
    successes: list[float] = []
    angles: list[float] = []
    drift_pos: list[float] = []
    drift_rot: list[float] = []
    for env_index in range(args.envs):
        env = WrenchCubeEnv()
        try:
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
    mean_success = float(np.mean(successes)) if successes else 0.0
    print(
        f"summary: controller={args.controller} envs={args.envs} episodes={args.episodes} "
        f"success={mean_success:.3f} "
        f"angle_deg={np.mean(angles):.1f} "
        f"drift_mm={np.mean(drift_pos) * 1e3:.1f} "
        f"drift_deg={np.degrees(np.mean(drift_rot)):.1f}"
    )
    if args.controller == "oracle" and mean_success < _ORACLE_SUCCESS_FLOOR:
        print(f"FAIL: oracle mean success {mean_success:.3f} < {_ORACLE_SUCCESS_FLOOR}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
