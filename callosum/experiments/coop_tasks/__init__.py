"""Two-agent reduced coop suite (`CartPoleCoop-v0`, `PendulumPump-v0`, `TossCatch-v0`).

The suite (`PLAN.md` "Two-robot task suite") is a set of simple tasks in which two
decentralised agents cooperate on one shared object: the one-sided pushers of
`CartPoleCoop-v0` balance and move a shared cart-pole, the one-sided torque actuators of
`PendulumPump-v0` pump a pendulum to a target amplitude and hold it, and the two tilted
position-controlled paddles of `TossCatch-v0` catch and return a free ball for N consecutive
passes. All tasks share the harness `callosum.experiments.coop_tasks.base.CoopTaskEnv`, the
pure logic `callosum.experiments.coop_tasks.logic` and this package's scripted oracles.

Importing this package is what registers the gymnasium ids `CartPoleCoop-v0`,
`PendulumPump-v0` and `TossCatch-v0`, plus the hidden-partner variant
`TossCatchOccluded-v0` (`TossConfig.occlusion="partner"`, `PLAN.md` "Hidden-partner
regime"); it needs the `sandbox` extra (mujoco), which CI does not install, so never import
it from pure-logic modules or from tests that must run without the simulator. The pure logic
lives in `callosum.experiments.coop_tasks.logic`, the configs in
`callosum.experiments.coop_tasks.config`.
"""

from __future__ import annotations

import gymnasium

from callosum.experiments.coop_tasks.cartpole import CartPoleEnv
from callosum.experiments.coop_tasks.config import CartPoleConfig, PendulumConfig, TossConfig
from callosum.experiments.coop_tasks.oracle import (
    CartPoleOracle,
    CartPoleOracleParams,
    PendulumOracle,
    PendulumOracleParams,
    TossOracle,
    TossOracleParams,
    make_oracle,
)
from callosum.experiments.coop_tasks.pendulum import PendulumEnv
from callosum.experiments.coop_tasks.toss_catch import TossCatchEnv

TASK_ENV_IDS = {
    "cartpole": "CartPoleCoop-v0",
    "pendulum": "PendulumPump-v0",
    "toss": "TossCatch-v0",
}
"""Gymnasium id of each suite task."""

OCCLUDED_ENV_IDS = {"toss": "TossCatchOccluded-v0"}
"""Gymnasium id of each task's hidden-partner variant (`PLAN.md` "Hidden-partner regime").

`TossCatchOccluded-v0` is `TossCatch-v0` with `TossConfig(occlusion="partner")`: the agent
keeps the ball and its OWN paddle but not the partner paddle's pose/velocity/tilt, while the
privileged `partner_mode="oracle"` block is unchanged. The other suite tasks have no
per-agent shared fields, so they have no occluded id."""

gymnasium.register(
    id="CartPoleCoop-v0",
    entry_point="callosum.experiments.coop_tasks.cartpole:CartPoleEnv",
    disable_env_checker=True,
)
gymnasium.register(
    id="PendulumPump-v0",
    entry_point="callosum.experiments.coop_tasks.pendulum:PendulumEnv",
    disable_env_checker=True,
)
gymnasium.register(
    id="TossCatch-v0",
    entry_point="callosum.experiments.coop_tasks.toss_catch:TossCatchEnv",
    disable_env_checker=True,
)


def make_occluded_toss_env(
    config: TossConfig | None = None,
    partner_mode: str = "none",
    render_mode: str | None = None,
) -> TossCatchEnv:
    """Factory of `TossCatchOccluded-v0` (the hidden-partner toss-catch task).

    The occluded variant is the same env with `TossConfig.occlusion="partner"`; this factory
    is the gymnasium entry point, so `gymnasium.make("TossCatchOccluded-v0")` builds the
    hidden-partner task without the caller having to know the config field.

    Args:
        config: The task config; its occlusion must be `"partner"`. When `None` the default
            toss config is used with `occlusion="partner"`.
        partner_mode: Observation visibility of the partner, `"none"` or `"oracle"`.
        render_mode: Unused (the coop suite renders nothing); passed through to the env for
            its usual error.

    Returns:
        `TossCatchEnv` with the partner paddle hidden from the shared observation block.

    Raises:
        ValueError: for a non-`TossConfig` config or one with `occlusion != "partner"`.
    """
    if config is None:
        config = TossConfig(occlusion="partner")
    elif not isinstance(config, TossConfig):
        raise ValueError(f"TossCatchOccluded-v0 needs a TossConfig, got {type(config)}")
    elif config.occlusion != "partner":
        raise ValueError(
            "TossCatchOccluded-v0 needs TossConfig(occlusion='partner'), got "
            f"occlusion={config.occlusion!r}; build TossCatch-v0 for the full layout"
        )
    return TossCatchEnv(config=config, partner_mode=partner_mode, render_mode=render_mode)


gymnasium.register(
    id="TossCatchOccluded-v0",
    entry_point=f"{__name__}:make_occluded_toss_env",
    disable_env_checker=True,
)


def make_coop_env(
    task: str,
    config: CartPoleConfig | PendulumConfig | TossConfig | None = None,
    partner_mode: str = "none",
) -> CartPoleEnv | PendulumEnv | TossCatchEnv:
    """Build the env of a suite task.

    The factory is what the vectorisation (`callosum.experiments.coop_tasks.vec_env`), the evaluator and
    the smoke scripts call, so the task -> env mapping lives in one place.

    Args:
        task: `"cartpole"`, `"pendulum"` or `"toss"`.
        config: The task config; the task's default config when `None`. For `"toss"` the
            config's `occlusion` field selects the observation variant (`"none"` = full
            shared block, `"partner"` = the hidden-partner `OccludedToss-v0` layout).
        partner_mode: Observation visibility of the partner, `"none"` or `"oracle"`.

    Returns:
        The concrete `CartPoleEnv`, `PendulumEnv` or `TossCatchEnv`.

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
    if task == "toss":
        if config is not None and not isinstance(config, TossConfig):
            raise ValueError(f"toss needs a TossConfig, got {type(config)}")
        return TossCatchEnv(config=config, partner_mode=partner_mode)
    raise ValueError(f"task must be one of {tuple(TASK_ENV_IDS)}, got {task!r}")


__all__ = [
    "OCCLUDED_ENV_IDS",
    "TASK_ENV_IDS",
    "CartPoleConfig",
    "CartPoleEnv",
    "CartPoleOracle",
    "CartPoleOracleParams",
    "PendulumConfig",
    "PendulumEnv",
    "PendulumOracle",
    "PendulumOracleParams",
    "TossCatchEnv",
    "TossConfig",
    "TossOracle",
    "TossOracleParams",
    "make_coop_env",
    "make_occluded_toss_env",
    "make_oracle",
]
