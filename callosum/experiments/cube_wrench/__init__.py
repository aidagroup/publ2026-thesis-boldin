"""CubeWrench-v0: the MuJoCo reduced-mode wrench-level cube package.

Importing this package is what registers the gymnasium id `CubeWrench-v0`; it needs the
`sandbox` extra (mujoco), which CI does not install, so never import it from pure-logic
modules or from tests that must run without the simulator. The pure logic lives in
`callosum.experiments.cube_wrench.logic`, the config in `callosum.experiments.cube_wrench.config`.
"""

import gymnasium

from callosum.experiments.cube_wrench.env import WrenchCubeEnv
from callosum.experiments.cube_wrench.oracle import OracleController, OracleParams

gymnasium.register(
    id="CubeWrench-v0",
    entry_point="callosum.experiments.cube_wrench.env:WrenchCubeEnv",
    disable_env_checker=True,
)

__all__ = ["OracleController", "OracleParams", "WrenchCubeEnv"]
