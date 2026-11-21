"""CubeWrench-v0: the MuJoCo reduced-mode wrench-level cube package.

Importing this package is what registers the gymnasium id `CubeWrench-v0`; it needs the
`sandbox` extra (mujoco), which CI does not install, so never import it from pure-logic
modules or from tests that must run without the simulator. The pure logic lives in
`callosum.envs._wrench_logic`, the config in `callosum.configs.wrench_cube`.
"""

import gymnasium

from callosum.envs.wrench_cube.env import WrenchCubeEnv
from callosum.envs.wrench_cube.oracle import OracleController, OracleParams

gymnasium.register(
    id="CubeWrench-v0",
    entry_point="callosum.envs.wrench_cube.env:WrenchCubeEnv",
    disable_env_checker=True,
)

__all__ = ["OracleController", "OracleParams", "WrenchCubeEnv"]
