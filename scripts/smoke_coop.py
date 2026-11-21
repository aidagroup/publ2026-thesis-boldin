"""[local] Smoke-check the two-agent coop suite: 1-2 envs, short scripted rollouts.

The coop suite is CPU-only MuJoCo, so this is one of the allowed local simulation checks
(1-2 envs, short episodes, no training). It creates the envs one after another (the env is
single-instance; the trainer vectorises later), runs `--episodes` episodes per env for the
requested controller and prints, per episode, success / steps / task diagnostic, then the
aggregate.

Controllers (`--controller`):

* `oracle` -- the task's scripted two-agent oracle (`callosum.experiments.coop_tasks.oracle`). This is
  the environment sanity gate: mean success must be >= 0.8. With this controller the script
  ALSO runs the "only-left" baseline on the same seeds and prints it: the same oracle with
  the right agent absent must stay clearly below the oracle (the suite's non-additivity).
* `only-left` -- the absent-agent baseline on its own: the oracle commands with the right
  agent forced to zero (the one-sided actuators then cover only every other direction /
  half-cycle). It is expected to (almost) never succeed.
* `random` -- `env.action_space.sample()`; `--seed` seeds `np.random` and each env's
  action-space generator, so random runs are reproducible.

Exit code: 0 when the selected controller meets its expectation (the `random` controller is
a diagnostic and always exits 0), 1 otherwise.

    ./.venv/bin/python scripts/smoke_coop.py --task cartpole --controller oracle --envs 2 --episodes 10
    ./.venv/bin/python scripts/smoke_coop.py --task pendulum --controller oracle --envs 2 --episodes 10
    ./.venv/bin/python scripts/smoke_coop.py --task cartpole --controller only-left --envs 2 --episodes 10
"""

import argparse
import sys

import numpy as np

from callosum.experiments.coop_tasks import make_coop_env, make_oracle

_ORACLE_SUCCESS_FLOOR = 0.8
"""The scripted oracle must reach (almost) every episode; 0.8 leaves room for local noise."""

_ONLY_LEFT_SUCCESS_CEIL = 0.1
"""The absent-agent baseline must (almost) never succeed; 0.1 allows a little local noise."""

_MIN_NON_ADDITIVITY_GAP = 0.3
"""The oracle-vs-only-left success gap the smoke check requires."""


def parse_args() -> argparse.Namespace:
    """Parse the smoke-check CLI (task, env count, episodes, controller kind, base seed)."""
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-check the two-agent coop suite with the scripted oracle, the absent-agent "
            "baseline or random actions."
        )
    )
    parser.add_argument("--task", choices=("cartpole", "pendulum"), default="cartpole")
    parser.add_argument("--envs", type=int, default=2, help="envs to build sequentially")
    parser.add_argument("--episodes", type=int, default=5, help="episodes per env")
    parser.add_argument(
        "--controller",
        choices=("oracle", "only-left", "random"),
        default="oracle",
        help=(
            "action source: 'oracle' (sanity gate + non-additivity baseline), 'only-left' "
            "(absent-agent baseline) or 'random' (diagnostic)"
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="base episode seed")
    return parser.parse_args()


def _episode_seed(base_seed: int, env_index: int, episode: int, episodes: int) -> int:
    """The deterministic seed of one smoke episode."""
    return base_seed + env_index * episodes + episode


def _diagnostic(info: dict, task: str) -> float:
    """The task's primary amplitude/angle diagnostic from a step info dict."""
    if task == "cartpole":
        return float(info["angle_deg"])
    return float(info["amplitude_deg"])


def run_episode(env, oracle, kind: str, seed: int, task: str) -> tuple[dict, int]:
    """Run one episode to termination/truncation and return its info and step count."""
    env.reset(seed=seed)
    steps = 0
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        if kind == "oracle":
            action = oracle.act(env)
        elif kind == "only-left":
            action = oracle.act(env)
            action["right"] = np.zeros_like(action["right"])
        else:
            action = env.action_space.sample()
        _obs, _reward, terminated, truncated, info = env.step(action)
        steps += 1
    return info, steps


def run_controller(
    args: argparse.Namespace, kind: str
) -> tuple[list[float], list[int], list[float]]:
    """Run `--envs x --episodes` episodes of one controller; returns success/steps/diagnostics."""
    successes: list[float] = []
    steps: list[int] = []
    diagnostics: list[float] = []
    for env_index in range(args.envs):
        env = make_coop_env(args.task)
        try:
            if kind == "random":
                env.action_space.seed(args.seed + env_index)
            oracle = make_oracle(args.task, env.config)
            for episode in range(args.episodes):
                seed = _episode_seed(args.seed, env_index, episode, args.episodes)
                info, episode_steps = run_episode(env, oracle, kind, seed, args.task)
                successes.append(float(info["success"]))
                steps.append(episode_steps)
                diagnostics.append(_diagnostic(info, args.task))
                print(
                    f"env {env_index} episode {episode}: "
                    f"success={bool(info['success'])} steps={episode_steps} "
                    f"{'angle_deg' if args.task == 'cartpole' else 'amplitude_deg'}="
                    f"{diagnostics[-1]:.1f}"
                )
        finally:
            env.close()
    return successes, steps, diagnostics


def summarize(
    label: str, successes: list[float], steps: list[int], diagnostics: list[float]
) -> float:
    """Print one controller's aggregate and return its mean success."""
    mean_success = float(np.mean(successes))
    print(
        f"summary {label}: success={mean_success:.3f} "
        f"steps={np.mean(steps):.1f} diagnostic={np.mean(diagnostics):.1f}"
    )
    return mean_success


def main() -> int:
    """Run the smoke check; return the process exit code."""
    args = parse_args()
    if args.envs < 1 or args.episodes < 1:
        print(f"FAIL: --envs and --episodes must be >= 1, got {args.envs}/{args.episodes}")
        return 2
    if args.controller == "random":
        # Gymnasium's `action_space.sample()` draws from the space's OWN generator, not the
        # global `np.random`, so both are seeded for a reproducible random baseline.
        np.random.seed(args.seed)
    successes, steps, diagnostics = run_controller(args, args.controller)
    mean_success = summarize(args.controller, successes, steps, diagnostics)
    if args.controller == "random":
        return 0
    if args.controller == "only-left":
        ok = mean_success <= _ONLY_LEFT_SUCCESS_CEIL
        print(
            f"only-left check: success={mean_success:.3f} "
            f"(expected <= {_ONLY_LEFT_SUCCESS_CEIL}) -> {'PASS' if ok else 'FAIL'}"
        )
        return 0 if ok else 1
    # The oracle gate also runs the absent-agent baseline on the same seeds.
    baseline_successes, baseline_steps, baseline_diagnostics = run_controller(args, "only-left")
    baseline_success = summarize(
        "only-left", baseline_successes, baseline_steps, baseline_diagnostics
    )
    gap = mean_success - baseline_success
    checks = {
        "oracle": mean_success >= _ORACLE_SUCCESS_FLOOR,
        "baseline": baseline_success <= _ONLY_LEFT_SUCCESS_CEIL,
        "gap": gap >= _MIN_NON_ADDITIVITY_GAP,
    }
    print(
        f"task={args.task} oracle={mean_success:.3f} | only-left={baseline_success:.3f} | "
        f"gap={gap:+.3f} (required >= {_MIN_NON_ADDITIVITY_GAP}) -> "
        f"{'PASS' if all(checks.values()) else 'FAIL'}"
    )
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
