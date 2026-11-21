"""Two-agent reduced coop suite (`CartPoleCoop-v0`, `PendulumPump-v0`) in MuJoCo.

The suite (`PLAN.md` "Two-robot task suite") is a set of simple tasks in which two
decentralised agents cooperate on one shared object: the one-sided pushers of
`CartPoleCoop-v0` balance and move a shared cart-pole, the one-sided torque actuators of
`PendulumPump-v0` pump a pendulum to a target amplitude and hold it. Both tasks share the
harness `callosum.envs.coop.base.CoopTaskEnv`, the pure logic
`callosum.envs._coop_logic` and this package's scripted oracles.

Importing this package is what registers the gymnasium ids `CartPoleCoop-v0` and
`PendulumPump-v0`; it needs the `sandbox` extra (mujoco), which CI does not install, so
never import it from pure-logic modules or from tests that must run without the simulator.
The pure logic lives in `callosum.envs._coop_logic`, the configs in
`callosum.configs.coop_tasks`.
"""

from __future__ import annotations

import gymnasium

from callosum.configs.coop_tasks import CartPoleConfig, PendulumConfig
from callosum.envs.coop.cartpole import CartPoleEnv
from callosum.envs.coop.oracle import (
    CartPoleOracle,
    CartPoleOracleParams,
    PendulumOracle,
    PendulumOracleParams,
    make_oracle,
)
from callosum.envs.coop.pendulum import PendulumEnv

TASK_ENV_IDS = {
    "cartpole": "CartPoleCoop-v0",
    "pendulum": "PendulumPump-v0",
}
"""Gymnasium id of each suite task."""

gymnasium.register(
    id="CartPoleCoop-v0",
    entry_point="callosum.envs.coop.cartpole:CartPoleEnv",
    disable_env_checker=True,
)
gymnasium.register(
    id="PendulumPump-v0",
    entry_point="callosum.envs.coop.pendulum:PendulumEnv",
    disable_env_checker=True,
)


def make_coop_env(
    task: str,
    config: CartPoleConfig | PendulumConfig | None = None,
    partner_mode: str = "none",
) -> CartPoleEnv | PendulumEnv:
    """Build the env of a suite task.

    The factory is what the vectorisation (`callosum.training._vec_coop`), the evaluator and
    the smoke scripts call, so the task -> env mapping lives in one place.

    Args:
        task: `"cartpole"` or `"pendulum"`.
        config: The task config; the task's default config when `None`.
        partner_mode: Observation visibility of the partner, `"none"` or `"oracle"`.

    Returns:
        The concrete `CartPoleEnv` or `PendulumEnv`.

    Raises:
        ValueError: for an unknown task or a config of the wrong type.
    """
    if task == "cartpole":
        if config is not None and not isinstance(config, CartPoleConfig):
            raise ValueError(f"cartpole needs a CartPoleConfig, got {type(config)}")
        return CartPoleEnv(config=config, partner_mode=partner_mode)
    if task == "pendulum":
        if config is not None and not isinstance(config, PendulumConfig):
            raise ValueError(f"pendulum needs a PendulumConfig, got {type(config)}")
        return PendulumEnv(config=config, partner_mode=partner_mode)
    raise ValueError(f"task must be one of {tuple(TASK_ENV_IDS)}, got {task!r}")


__all__ = [
    "TASK_ENV_IDS",
    "CartPoleConfig",
    "CartPoleEnv",
    "CartPoleOracle",
    "CartPoleOracleParams",
    "PendulumConfig",
    "PendulumEnv",
    "PendulumOracle",
    "PendulumOracleParams",
    "make_coop_env",
    "make_oracle",
]
