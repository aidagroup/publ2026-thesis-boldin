"""Batched GPU backend of CubeWrench-v0 (`mujoco_warp`; optional `sandbox-gpu` extra).

The CPU env (`callosum.experiments.cube_wrench.env.WrenchCubeEnv`) is single-instance and driven by a
Python loop of `mj_step` calls, which caps the trainer at a few thousand environment steps per
second no matter how small the reduced model is. This module is the batched GPU twin: all
`nworld` worlds live in ONE MuJoCo-Warp `Model`/`Data` pair, the whole control step (latency
queues, failure injections, `xfrc_applied`, 8 physics substeps, observations, reward and the
auto-reset of finished worlds) is tensor algebra on the GPU, and a single Python call steps
thousands of environments. The physics, the randomisation and the observation/reward layout
are the SAME contract as the CPU env -- `scripts/check_wrench_gpu_parity.py` compares the two
step by step (qpos/qvel within 1e-5, observations within 1e-4; float32 Warp against float64
MuJoCo is the only intended difference).

Install (GPU pod only): the dependency is Linux/CUDA-only, so it lives in the OPTIONAL
`sandbox-gpu` extra and is never installed by the default/macOS/CI sync. The extra is in
`uv.lock` with a `sys_platform == 'linux'` marker, so the clean way is

    uv sync --frozen --extra sandbox-gpu

but the pod checkout is usually synced WITHOUT that extra and `uv sync --frozen` removes the
package again (it is not in the selected extra set), so before a run either sync with the
extra or install into the same `python` that runs the trainer (a ~10 s cached install):

    uv pip install --python /root/.callosum/venv/bin/python mujoco-warp==4.0.1

This module imports `mujoco`, `mujoco_warp`, `warp` and `torch` lazily from the constructor, so
importing the package on a machine without the GPU stack fails only when an instance is built
(scripts/tests gate on `importorskip`); the pure packing helper `pack_episode_randomization`
is torch-only and unit-tested on the Mac.

Implementation notes (all verified against the installed `mujoco-warp==4.0.1`):

* `mjwarp.put_model(mjm, batch_sizes={"dof_frictionloss": nworld})` creates a PER-WORLD
  `dof_frictionloss`, which is exactly what the CPU env's per-episode layer friction needs
  (the constraint kernel indexes it by `worldid % shape[0]`), so the sampled Coulomb friction
  is reproduced without approximating it away.
* `mjwarp.put_data(mjm, mjd, nworld=nworld)` replicates the host state; `Data.qpos` is
  `(nworld, nq)`, `Data.xfrc_applied` is `(nworld, nbody, 6)` in MuJoCo's
  `[force xyz, torque xyz]` order, and `warp.to_torch` gives zero-copy CUDA views.
* `mjwarp.step(mw, d)` advances one physics substep; the control step calls it
  `config.control_every` times. The substep loop is captured ONCE into a CUDA graph
  (`warp.ScopedCapture`, the pattern of mujoco_warp's own viewer/testspeed) and replayed per
  control step: without the graph the tiny model is launch-bound and a control step takes
  ~35 ms at ANY world count; the graph replay takes ~1.5 ms (64 worlds) / ~2 ms (1024) /
  ~3 ms (4096), i.e. it is what makes the GPU backend actually faster than the CPU loop. The
  graph references the fixed Warp arrays, so the per-step torch writes of `qpos`/`qvel`/
  `xfrc_applied`/per-world friction are seen by the replay; if a capture fails (older
  toolkit) the env logs and falls back to the eager substep loop.
* The per-world randomisation is sampled on the CPU with the SAME stdlib sampler as the CPU
  env (`_wrench_logic.sample_episode_randomization`), then uploaded as tensors (the task
  explicitly allows the CPU sampler for the MVP); an auto-reset batch only samples the worlds
  that finished. The sampling is therefore bit-identical to the CPU path for a given seed.

Known limits of this backend: one GPU/one device per instance; `torch.float32` physics
(float64 CPU parity, not bit-for-bit); observations/reward therefore match within the parity
tolerances above, so the two backends are interchangeable for training/evaluation but a curve
cannot be expected to be bit-identical across them. The layout of `assets/cube_wrench.xml`
(nq=8: free core joint + one hinge; `hinge_qpos == 7`) and of
`callosum.experiments.cube_wrench.logic` (field blocks) is asserted at construction, so a layout change
fails loudly instead of silently producing a wrong observation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:  # `torch` is part of the `train` extra; the module must still import without it.
    import torch
except ImportError:  # pragma: no cover - the GPU backend itself always needs torch
    torch = None  # type: ignore[assignment]

from callosum.experiments.cube_wrench.config import WrenchCubeConfig
from callosum.experiments.cube_wrench.logic import (
    AGENT_KEYS,
    CUBE_STATE_FIELDS,
    GOAL_FIELDS,
    OWN_PARAM_FIELDS,
    OWN_PRIVILEGED_FIELDS,
    OWN_WRENCH_FIELDS,
    PARTNER_COMMAND_FIELDS,
    PARTNER_PARAM_FIELDS,
    PARTNER_PRIVILEGED_FIELDS,
    PARTNER_WRENCH_FIELDS,
    EpisodeRandomization,
    obs_fields,
    obs_size,
    partner_of,
    sample_episode_randomization,
    validate_partner_mode,
)

_ASSET_PATH = Path(__file__).resolve().parent / "assets" / "cube_wrench.xml"
_HINGE_JOINT = "layer_hinge"
_CORE_BODY = "core"
_LAYER_BODY = "layer"
_WRENCH_SIZE = 6
"""Width of one agent's action (a normalized 6D wrench)."""

_AGENT_RANDOMIZATION_FIELDS = (
    "latency_s",
    "latency_steps",
    "gain",
    "limit_scale",
    "slip_active",
    "slip_step",
    "slip_steps",
    "start_delay_s",
    "start_delay_steps",
)
"""Per-agent fields of `_wrench_logic.AgentRandomization`, in packing order."""

_INT_AGENT_FIELDS = frozenset({"latency_steps", "slip_step", "slip_steps", "start_delay_steps"})
_BOOL_AGENT_FIELDS = frozenset({"slip_active"})


# --- episode randomisation packing --------------------------------------------------------


def pack_episode_randomization(
    records: list[EpisodeRandomization], device: torch.device | str = "cpu"
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Pack a batch of per-episode randomisation records into `(nworld, ...)` tensors.

    Pure torch (no Warp), so the packing is unit-testable on the Mac and reusable by the
    parity script. The tensor values are exactly the sampled dataclass values (the sampler is
    `_wrench_logic.sample_episode_randomization`, the same one the CPU env uses), so a batch
    world reproduces the CPU env of the same seed -- subject only to the dtype (the CPU env
    keeps float64; the GPU path uploads float32, which is what the Warp model uses).

    Args:
        records: One record per world, in world order.
        device: Torch device of the produced tensors.

    Returns:
        `(agents, episode)`: `agents[role][field]` tensors of shape `(nworld,)` (int64 for
        the step counts, bool for `slip_active`, float32 for the rest), and `episode` with
        `layer_initial_angle` (nworld,), `layer_frictionloss` (nworld,), `impulse_active`
        (nworld,) bool, `impulse_step` (nworld,) int64 and `impulse_force` (nworld, 3).
    """
    if torch is None:  # pragma: no cover - only reachable without the train extra
        raise ImportError("pack_episode_randomization needs torch; install the `train` extra")
    if not records:
        raise ValueError("records must contain at least one EpisodeRandomization")
    device = torch.device(device)
    agents: dict[str, dict[str, torch.Tensor]] = {}
    for role_index, role in enumerate(AGENT_KEYS):
        fields: dict[str, torch.Tensor] = {}
        for name in _AGENT_RANDOMIZATION_FIELDS:
            values = [getattr(record.agents[role_index], name) for record in records]
            if name in _BOOL_AGENT_FIELDS:
                tensor = torch.tensor(values, dtype=torch.bool, device=device)
            elif name in _INT_AGENT_FIELDS:
                tensor = torch.tensor(values, dtype=torch.int64, device=device)
            else:
                tensor = torch.tensor(values, dtype=torch.float32, device=device)
            fields[name] = tensor
        agents[role] = fields
    episode = {
        "layer_initial_angle": torch.tensor(
            [record.layer_initial_angle for record in records],
            dtype=torch.float32,
            device=device,
        ),
        "layer_frictionloss": torch.tensor(
            [record.layer_frictionloss for record in records],
            dtype=torch.float32,
            device=device,
        ),
        "impulse_active": torch.tensor(
            [record.impulse_active for record in records], dtype=torch.bool, device=device
        ),
        "impulse_step": torch.tensor(
            [record.impulse_step for record in records], dtype=torch.int64, device=device
        ),
        "impulse_force": torch.tensor(
            [record.impulse_force for record in records], dtype=torch.float32, device=device
        ),
    }
    return agents, episode


# --- observation layout -------------------------------------------------------------------

_SHARED_BLOCKS = (
    ("own_wrench", OWN_WRENCH_FIELDS),
    ("own_params", OWN_PARAM_FIELDS),
    ("own_privileged", OWN_PRIVILEGED_FIELDS),
    ("cube", CUBE_STATE_FIELDS),
    ("goal", GOAL_FIELDS),
)
"""The observation blocks that do not depend on the partner mode, in `obs_fields` order."""

_PARTNER_BLOCKS = (
    ("partner_wrench", PARTNER_WRENCH_FIELDS),
    ("partner_command", PARTNER_COMMAND_FIELDS),
    ("partner_params", PARTNER_PARAM_FIELDS),
    ("partner_privileged", PARTNER_PRIVILEGED_FIELDS),
)
"""The partner block under `partner_mode="oracle"`, in `obs_fields` order."""

_TENSOR_KEYS = {
    "own_wrench": "wrench",
    "own_params": "params",
    "own_privileged": "privileged",
    "partner_wrench": "wrench",
    "partner_command": "command",
    "partner_params": "params",
    "partner_privileged": "privileged",
}
"""Observation block name -> per-agent tensor key (the same key for own and partner blocks)."""


def observation_blocks(partner_mode: str) -> dict[str, tuple[tuple[str, str | None], ...]]:
    """The ordered `(tensor_key, owner)` blocks of each agent's flat observation.

    `owner` is the role whose tensors the block is read from (the agent itself for its own
    blocks, the partner under `"oracle"`), or `None` for the shared cube-state/goal blocks.
    The function also validates that concatenating the block field names reproduces
    `_wrench_logic.obs_fields(partner_mode)` exactly, so a change of the pure observation
    layout (a new field, a reordered block) fails at construction instead of silently
    producing an observation the policy was not trained on.

    Args:
        partner_mode: Observation visibility mode, `"none"` or `"oracle"`.

    Returns:
        `{role: ((tensor_key, owner), ...)}` in observation column order.

    Raises:
        RuntimeError: if the blocks do not concatenate to `_wrench_logic.obs_fields`.
    """
    validate_partner_mode(partner_mode)
    expected_fields: list[str] = []
    for _name, block_fields in _SHARED_BLOCKS:
        expected_fields.extend(block_fields)
    if partner_mode != "none":
        for _name, block_fields in _PARTNER_BLOCKS:
            expected_fields.extend(block_fields)
    expected = tuple(expected_fields)
    if expected != obs_fields(partner_mode):
        raise RuntimeError(
            "the GPU observation blocks and callosum.experiments.cube_wrench.logic.obs_fields disagree: "
            f"blocks concatenate to {len(expected)} fields, obs_fields has "
            f"{len(obs_fields(partner_mode))} ({obs_fields(partner_mode)}); update "
            "callosum/experiments/cube_wrench/env_gpu.py next to the logic module"
        )
    blocks: dict[str, tuple[tuple[str, str | None], ...]] = {}
    for role in AGENT_KEYS:
        spec: list[tuple[str, str | None]] = []
        for name, _fields in _SHARED_BLOCKS:
            owner = None if name in ("cube", "goal") else role
            spec.append((_TENSOR_KEYS.get(name, name), owner))
        if partner_mode != "none":
            partner = partner_of(role)
            for name, _fields in _PARTNER_BLOCKS:
                spec.append((_TENSOR_KEYS[name], partner))
        blocks[role] = tuple(spec)
    return blocks


# --- step result --------------------------------------------------------------------------


@dataclass
class GpuStep:
    """One control step of all worlds (the auto-reset contract mirrors `VecWrenchCube`).

    The tensors live on the env's CUDA device and are NOT copied to the host, except the
    small `finished_*` lists of the episodes that ended on this step (the trainer/evaluator
    consume them as Python values, as with the CPU vec env).

    Attributes:
        obs: Next-step observation per role, `(nworld, obs_dim)`; done worlds already
            restarted, so `obs` is always the input of the next policy call.
        reward: Shared team reward, `(nworld,)` float32.
        terminated: Success/failure flags, `(nworld,)` bool.
        truncated: Time-limit flags, `(nworld,)` bool.
        final_obs: Per-role observation at the END of the episode, `(nworld, obs_dim)`; only
            the rows of done worlds are meaningful (the truncation bootstrap reads the
            `truncated` ones).
        finished_returns: Undiscounted returns of the episodes that ended on this step.
        finished_lengths: Lengths (control steps) of those episodes.
        finished_successes: Success flags of those episodes.
        finished_angles: Final layer angle (degrees) of those episodes.
        finished_drift_pos: Final core position drift (m) of those episodes.
        finished_drift_rot: Final core rotation drift (rad) of those episodes.
    """

    obs: dict[str, torch.Tensor]
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    final_obs: dict[str, torch.Tensor]
    finished_returns: list[float] = field(default_factory=list)
    finished_lengths: list[int] = field(default_factory=list)
    finished_successes: list[bool] = field(default_factory=list)
    finished_angles: list[float] = field(default_factory=list)
    finished_drift_pos: list[float] = field(default_factory=list)
    finished_drift_rot: list[float] = field(default_factory=list)

    @property
    def done(self) -> torch.Tensor:
        """`terminated | truncated`, `(nworld,)` bool."""
        return torch.logical_or(self.terminated, self.truncated)


# --- the batch env ------------------------------------------------------------------------


def _normalize_device(device: str) -> str:
    """Normalize a CUDA device string for Warp (`"cuda"` -> `"cuda:0"`)."""
    normalized = str(device)
    if normalized == "cuda":
        normalized = "cuda:0"
    if not normalized.startswith("cuda"):
        raise ValueError(
            f"WrenchCubeBatchGPU runs on a CUDA device of mujoco_warp, got device={device!r}"
        )
    return normalized


def _steps(config: WrenchCubeConfig, seconds: float) -> int:
    """Convert seconds to whole control steps (same rounding as `_wrench_logic._steps`)."""
    return max(0, round(seconds / config.control_dt))


class WrenchCubeBatchGPU:
    """`nworld` CubeWrench-v0 worlds stepped together on the GPU through `mujoco_warp`.

    The public contract (reset/step, action dict of normalized 6D wrenches, flat per-agent
    observations, shared team reward, success/failure/truncation, auto-reset) is the CPU
    `WrenchCubeEnv`'s; the difference is that every value is a `(nworld, ...)` tensor and one
    `step` advances all worlds. `torch_io = True` marks the tensor contract for
    `callosum.experiments.cube_wrench.train` / `evaluate_greedy`, which then pass CUDA tensors instead
    of numpy arrays.

    The class is not a `gymnasium.Env` on purpose: there is no vector-of-envs observation
    space to describe and the trainer uses the vec-env contract anyway.
    """

    torch_io = True
    """Marker for the trainer/evaluator: `reset`/`step` consume and return torch tensors."""

    def __init__(
        self,
        nworld: int,
        config: WrenchCubeConfig | None = None,
        partner_mode: str = "none",
        seed: int = 0,
        device: str = "cuda",
        use_cuda_graph: bool = True,
    ) -> None:
        """Build the Warp model/data pair and the per-world episode tensors.

        Args:
            nworld: Number of worlds stepped together (>= 1).
            config: Environment config; `WrenchCubeConfig()` defaults when `None`.
            partner_mode: Observation visibility mode, `"none"` or `"oracle"`.
            seed: Base episode seed of the auto-reset stream (`reset(seed)` seeds world `i`
                with `seed + i`; auto-resets continue from `seed + nworld`, exactly like the
                CPU `VecWrenchCube`).
            device: CUDA device string (`"cuda"` or `"cuda:N"`).
            use_cuda_graph: Capture the `control_every`-substep physics loop into a CUDA
                graph (see the module docstring); `False` forces the eager loop (debugging
                and the capture-failure fallback path).

        Raises:
            ValueError: on a non-positive `nworld`, a bad partner mode or a non-CUDA device.
            RuntimeError: if the model layout does not match this backend's assumptions.
        """
        if torch is None:  # pragma: no cover - only reachable without the train extra
            raise ImportError("WrenchCubeBatchGPU needs torch; install the `train` extra")
        if nworld < 1:
            raise ValueError(f"nworld must be >= 1, got {nworld}")
        validate_partner_mode(partner_mode)
        device = _normalize_device(device)

        # Lazy simulator imports: the pure-logic/torch parts of the package stay importable
        # without the GPU stack (and the macOS test suite skips these tests).
        import mujoco
        import mujoco_warp
        import warp as wp

        wp.init()
        wp.set_device(device)
        self._warp = wp
        self._mujoco_warp = mujoco_warp
        self._software_versions = {
            "mujoco": mujoco.__version__,
            "mujoco_warp": getattr(mujoco_warp, "__version__", "unknown"),
            "warp": wp.__version__,
            "torch": torch.__version__,
        }

        self.device = device
        self._torch_device = torch.device(device)
        self.config = config if config is not None else WrenchCubeConfig()
        cfg = self.config
        self.partner_mode = partner_mode
        self.num_envs = int(nworld)
        self.obs_dim = obs_size(partner_mode)
        self._base_seed = int(seed)
        self._next_seed = self._base_seed + self.num_envs

        # --- host model with the config as truth (same writes as the CPU env) -------------
        model = mujoco.MjModel.from_xml_path(str(_ASSET_PATH))
        hinge_joint = model.joint(_HINGE_JOINT)
        core_joint = model.joint("core_joint")
        self._core_id = int(model.body(_CORE_BODY).id)
        self._layer_id = int(model.body(_LAYER_BODY).id)
        self._hinge_qpos = int(model.jnt_qposadr[hinge_joint.id])
        self._hinge_dof = int(model.jnt_dofadr[hinge_joint.id])
        if self._hinge_qpos != 7:
            raise RuntimeError(
                "this GPU backend assumes the cube_wrench.xml layout (nq=8: free core joint "
                f"then the hinge at qpos address 7), got hinge_qpos={self._hinge_qpos}"
            )
        model.opt.timestep = cfg.dt
        model.jnt_range[hinge_joint.id] = (0.0, cfg.hinge_limit)
        core_dof0 = int(model.jnt_dofadr[core_joint.id])
        model.dof_damping[core_dof0 : core_dof0 + 3] = cfg.core_lin_damping
        model.dof_damping[core_dof0 + 3 : core_dof0 + 6] = cfg.core_ang_damping
        model.dof_damping[self._hinge_dof] = cfg.hinge_damping
        model.dof_armature[self._hinge_dof] = cfg.hinge_armature
        # The host value only sizes the per-world friction-loss constraint row (a zero would
        # allocate no row at all); every world's value is overwritten on reset.
        model.dof_frictionloss[self._hinge_dof] = max(
            cfg.hinge_frictionloss, cfg.layer_frictionloss_range[1], 1e-9
        )

        self._model = mujoco_warp.put_model(model, batch_sizes={"dof_frictionloss": self.num_envs})
        host_data = mujoco.MjData(model)
        mujoco.mj_forward(model, host_data)
        self._data = mujoco_warp.put_data(model, host_data, nworld=self.num_envs)
        self._physics_step = mujoco_warp.step

        # Zero-copy torch views of the Warp state (the arrays are never reallocated).
        self._qpos = wp.to_torch(self._data.qpos)
        self._qvel = wp.to_torch(self._data.qvel)
        self._xfrc = wp.to_torch(self._data.xfrc_applied)
        self._qacc = wp.to_torch(self._data.qacc)
        self._qacc_warmstart = wp.to_torch(self._data.qacc_warmstart)
        self._qfrc_applied = wp.to_torch(self._data.qfrc_applied)
        self._time = wp.to_torch(self._data.time)
        try:  # nu == 0 for the reduced model; keep the field optional across warp versions
            self._ctrl = wp.to_torch(self._data.ctrl)
        except (AttributeError, TypeError):  # pragma: no cover - warp-version dependent
            self._ctrl = None
        self._friction = wp.to_torch(self._model.dof_frictionloss)
        if self._qpos.dtype != torch.float32:
            raise RuntimeError(f"unexpected qpos dtype {self._qpos.dtype}")

        # --- per-world episode state ------------------------------------------------------
        self._authority = {
            role: torch.tensor(
                cfg.holder_authority if index == 0 else cfg.rotator_authority,
                dtype=torch.float32,
                device=self._torch_device,
            )
            for index, role in enumerate(AGENT_KEYS)
        }
        # The latency queue: `latency_steps` commands are in flight and the command of
        # `latency_steps` steps ago takes effect (a zero command during the initial latency,
        # exactly like the CPU env's deque initialised with zeros). `_delay_size` is one more
        # than the longest sampled latency, so "(write position - latency) mod size" points at
        # the command exactly `latency` steps back.
        self._delay_size = max(1, _steps(cfg, cfg.latency_range[1]) + 1)
        self._write_pos = 0
        zeros = self._torch_device
        self._delay = {
            role: torch.zeros(
                (self.num_envs, self._delay_size, _WRENCH_SIZE),
                dtype=torch.float32,
                device=zeros,
            )
            for role in AGENT_KEYS
        }
        self._last_applied = {
            role: torch.zeros((self.num_envs, _WRENCH_SIZE), dtype=torch.float32, device=zeros)
            for role in AGENT_KEYS
        }
        self._last_command = {
            role: torch.zeros_like(self._last_applied[role]) for role in AGENT_KEYS
        }
        self._params = {
            role: torch.zeros(
                (self.num_envs, len(OWN_PARAM_FIELDS)), dtype=torch.float32, device=zeros
            )
            for role in AGENT_KEYS
        }
        self._rand_agents = {
            role: self._new_agent_randomization_buffers(self.num_envs) for role in AGENT_KEYS
        }
        self._rand_episode = self._new_episode_randomization_buffers(self.num_envs)
        self._step = torch.zeros(self.num_envs, dtype=torch.int64, device=zeros)
        self._goal_time = torch.zeros(
            (self.num_envs, len(GOAL_FIELDS)), dtype=torch.float32, device=zeros
        )
        self._goal_time[:, 0] = cfg.goal_angle
        self._initial_core_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=zeros)
        self._initial_core_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=zeros)
        self._initial_core_quat[:, 0] = 1.0
        self._angle_initial = torch.zeros(self.num_envs, dtype=torch.float32, device=zeros)
        self._prev_potential = torch.zeros(self.num_envs, dtype=torch.float32, device=zeros)
        self._episode_return = torch.zeros(self.num_envs, dtype=torch.float64, device=zeros)
        self._episode_length = torch.zeros(self.num_envs, dtype=torch.int64, device=zeros)
        self._obs_blocks = observation_blocks(partner_mode)
        self._zero = torch.zeros((), dtype=torch.float32, device=zeros)
        self._reward_success = torch.tensor(
            cfg.success_bonus - cfg.step_penalty, dtype=torch.float32, device=zeros
        )
        self._reward_failure = torch.tensor(cfg.failure_penalty, dtype=torch.float32, device=zeros)
        self._reward_step = torch.tensor(-cfg.step_penalty, dtype=torch.float32, device=zeros)
        self._ready = False
        self._cuda_graph = None
        self._cuda_graph_error: str | None = None
        if use_cuda_graph:
            self._cuda_graph = self._capture_physics_graph()

    # --- CUDA graph ---------------------------------------------------------------------
    def _capture_physics_graph(self):
        """Capture the `control_every`-substep physics loop into a CUDA graph (or fail soft).

        The captured loop touches only the fixed Warp model/data arrays, so it can be
        replayed after the per-step torch writes of `qpos`/`qvel`/`xfrc_applied`/friction.
        A capture failure (unsupported toolkit/driver) is not fatal: the env falls back to
        the eager substep loop and records the reason in `cuda_graph_error`.

        Returns:
            The captured `warp.Graph`, or `None` when capture is unsupported here.
        """
        try:
            # Warm up the lazy kernel compilation / solver scratch allocations outside the
            # capture (a CUDA graph cannot allocate); the state is overwritten by reset().
            self._physics_step(self._model, self._data)
            with self._warp.ScopedCapture() as capture:
                for _ in range(self.config.control_every):
                    self._physics_step(self._model, self._data)
            return capture.graph
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - toolkit-dependent
            # A capture failure must never break the env: warn once and fall back to eager.
            self._cuda_graph_error = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                "CUDA graph capture of the physics substep loop failed "
                f"({self._cuda_graph_error}); falling back to the eager loop, which is "
                "~20x slower per control step",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    @property
    def cuda_graph(self) -> bool:
        """Whether the physics substep loop is replayed from a captured CUDA graph."""
        return self._cuda_graph is not None

    @property
    def cuda_graph_error(self) -> str | None:
        """Why the CUDA graph capture failed (`None` when the graph is active)."""
        return self._cuda_graph_error

    def software_versions(self) -> str:
        """The simulator/GPU software versions of this env, for the trainer's startup log."""
        return " | ".join(f"{name} {version}" for name, version in self._software_versions.items())

    # --- buffers ------------------------------------------------------------------------
    def _new_agent_randomization_buffers(self, nworld: int) -> dict[str, torch.Tensor]:
        """Zero buffers of one agent's randomisation fields (same dtypes as the packer)."""
        buffers: dict[str, torch.Tensor] = {}
        for name in _AGENT_RANDOMIZATION_FIELDS:
            if name in _BOOL_AGENT_FIELDS:
                buffers[name] = torch.zeros(nworld, dtype=torch.bool, device=self._torch_device)
            elif name in _INT_AGENT_FIELDS:
                buffers[name] = torch.zeros(nworld, dtype=torch.int64, device=self._torch_device)
            else:
                buffers[name] = torch.zeros(nworld, dtype=torch.float32, device=self._torch_device)
        return buffers

    def _new_episode_randomization_buffers(self, nworld: int) -> dict[str, torch.Tensor]:
        """Zero buffers of the episode-level randomisation fields."""
        return {
            "layer_initial_angle": torch.zeros(
                nworld, dtype=torch.float32, device=self._torch_device
            ),
            "layer_frictionloss": torch.zeros(
                nworld, dtype=torch.float32, device=self._torch_device
            ),
            "impulse_active": torch.zeros(nworld, dtype=torch.bool, device=self._torch_device),
            "impulse_step": torch.zeros(nworld, dtype=torch.int64, device=self._torch_device),
            "impulse_force": torch.zeros(
                (nworld, 3), dtype=torch.float32, device=self._torch_device
            ),
        }

    # --- randomisation / reset ----------------------------------------------------------
    def _install_randomization(self, seeds: list[int], indices: torch.Tensor | None) -> None:
        """Sample `seeds` on the CPU and install them (and the fresh episode state) on device.

        `indices=None` resets every world (a full `reset`); otherwise only the given rows are
        overwritten (the auto-reset of the worlds that finished). The sampler is the CPU
        env's, so a seed maps to the same episode randomisation in both backends.
        """
        records = [sample_episode_randomization(self.config, seed) for seed in seeds]
        agents, episode = pack_episode_randomization(records, self._torch_device)
        if indices is None:
            for role in AGENT_KEYS:
                for name, tensor in agents[role].items():
                    self._rand_agents[role][name].copy_(tensor)
                self._params[role].copy_(self._agent_params(agents[role]))
            for name, tensor in episode.items():
                self._rand_episode[name].copy_(tensor)
            self._qpos.zero_()
            self._qpos[:, 3] = 1.0
            self._qpos[:, self._hinge_qpos] = episode["layer_initial_angle"]
            self._qvel.zero_()
            self._reset_solver_state(None)
            self._step.zero_()
            self._friction[:, self._hinge_dof].copy_(episode["layer_frictionloss"])
            self._write_pos = 0
        else:
            for role in AGENT_KEYS:
                for name, tensor in agents[role].items():
                    self._rand_agents[role][name][indices] = tensor
                self._params[role][indices] = self._agent_params(agents[role])
            for name, tensor in episode.items():
                self._rand_episode[name][indices] = tensor
            self._qpos[indices] = 0.0
            self._qpos[indices, 3] = 1.0
            self._qpos[indices, self._hinge_qpos] = episode["layer_initial_angle"]
            self._qvel[indices] = 0.0
            self._reset_solver_state(indices)
            self._step[indices] = 0
            self._friction[:, self._hinge_dof].index_copy_(
                0, indices, episode["layer_frictionloss"]
            )
        # Episode accumulators and the observation bookkeeping of the (re)started worlds.
        angle_initial = self._qpos[:, self._hinge_qpos]
        if indices is None:
            self._initial_core_pos.copy_(self._qpos[:, 0:3])
            self._initial_core_quat.copy_(self._qpos[:, 3:7])
            self._angle_initial.copy_(angle_initial)
            self._prev_potential.zero_()
            self._episode_return.zero_()
            self._episode_length.zero_()
            for role in AGENT_KEYS:
                self._last_applied[role].zero_()
                self._last_command[role].zero_()
                self._delay[role].zero_()
        else:
            self._initial_core_pos[indices] = self._qpos[indices, 0:3]
            self._initial_core_quat[indices] = self._qpos[indices, 3:7]
            self._angle_initial[indices] = angle_initial[indices]
            self._prev_potential[indices] = 0.0
            self._episode_return[indices] = 0.0
            self._episode_length[indices] = 0
            for role in AGENT_KEYS:
                self._last_applied[role][indices] = 0.0
                self._last_command[role][indices] = 0.0
                self._delay[role][indices] = 0.0

    def _reset_solver_state(self, indices: torch.Tensor | None) -> None:
        """Zero the solver/actuator state the CPU env's `mj_resetData` clears on reset.

        `mj_resetData` resets the whole `mjData`: besides `qpos`/`qvel` it drops the solver
        warmstart, `qacc`, the applied generalised forces, the controls, `time` and the
        external forces. A fresh episode must start from that state (otherwise a residual
        `qacc_warmstart`/`qacc` of the old episode changes the first substep), so every reset
        zeroes the corresponding Warp data fields too -- for all worlds or only the
        auto-reset rows (`indices`).
        """
        data_fields = (
            self._qacc,
            self._qacc_warmstart,
            self._qfrc_applied,
            self._xfrc,
            self._time,
        )
        if self._ctrl is not None:
            data_fields = data_fields + (self._ctrl,)
        for data_field in data_fields:
            if indices is None:
                data_field.zero_()
            else:
                data_field[indices] = 0.0

    @staticmethod
    def _agent_params(fields: dict[str, torch.Tensor]) -> torch.Tensor:
        """`(nworld, 3)` tensor `[latency_s, gain, limit_scale]` in `OWN_PARAM_FIELDS` order."""
        return torch.stack((fields["latency_s"], fields["gain"], fields["limit_scale"]), dim=1)

    def reset(self, seed: int | None = None) -> dict[str, torch.Tensor]:
        """Reset every world and return the first observation of the new episodes.

        Mirrors `VecWrenchCube.reset`: world `i` is reset with episode seed `seed + i` (the
        wrapper's base seed when `seed` is `None`) and the auto-reset seed stream is NOT
        rewound, so `reset` with the same seed replays the same episodes.

        Args:
            seed: Base episode seed; `None` uses the constructor's seed.

        Returns:
            `{role: (nworld, obs_dim)}` first observation of each world.
        """
        base = self._base_seed if seed is None else int(seed)
        self._install_randomization([base + index for index in range(self.num_envs)], None)
        self._ready = True
        return self._build_observations()

    # --- physics / observations ---------------------------------------------------------
    def _injection_mask(self, role: str) -> torch.Tensor:
        """Whether the role's wrench is suppressed on this step (start delay or slip window)."""
        rand = self._rand_agents[role]
        in_start_delay = self._step < rand["start_delay_steps"]
        slip_end = rand["slip_step"] + rand["slip_steps"]
        in_slip = rand["slip_active"] & (rand["slip_step"] <= self._step) & (self._step < slip_end)
        return in_start_delay | in_slip

    def _slip_remaining(self, role: str) -> torch.Tensor:
        """Remaining steps of the role's slip window (0 outside it), for the observation."""
        rand = self._rand_agents[role]
        slip_end = rand["slip_step"] + rand["slip_steps"]
        in_slip = rand["slip_active"] & (rand["slip_step"] <= self._step) & (self._step < slip_end)
        remaining = (slip_end - self._step).to(torch.float32)
        return torch.where(in_slip, remaining, torch.zeros_like(remaining))

    def _build_observations(self) -> dict[str, torch.Tensor]:
        """Both agents' flat observation vectors for the CURRENT state and step counters.

        The blocks are assembled directly from the state tensors (one `torch.cat` per block
        chain instead of a per-field Python loop), in the order `observation_blocks` validated
        against `_wrench_logic.obs_fields`.
        """
        cfg = self.config
        self._goal_time[:, 1] = self._step.to(torch.float32) / cfg.max_control_steps
        cube = torch.cat(
            (
                self._qpos[:, 0:7],
                self._qvel[:, 0:6],
                self._qpos[:, self._hinge_qpos : self._hinge_qpos + 1],
                self._qvel[:, self._hinge_dof : self._hinge_dof + 1],
            ),
            dim=1,
        )
        shared = {"cube": cube, "goal": self._goal_time}
        per_agent = {}
        for role in AGENT_KEYS:
            rand = self._rand_agents[role]
            privileged = torch.cat(
                (rand["start_delay_s"].unsqueeze(1), self._slip_remaining(role).unsqueeze(1)),
                dim=1,
            )
            per_agent[role] = {
                "wrench": self._last_applied[role],
                "command": self._last_command[role],
                "params": self._params[role],
                "privileged": privileged,
            }
        observations: dict[str, torch.Tensor] = {}
        for role in AGENT_KEYS:
            parts: list[torch.Tensor] = []
            for key, owner in self._obs_blocks[role]:
                if owner is None:
                    parts.append(shared[key])
                else:
                    parts.append(per_agent[owner][key])
            observations[role] = torch.cat(parts, dim=1)
        return observations

    # --- gym-like step ------------------------------------------------------------------
    def step(self, actions: dict[str, torch.Tensor | np.ndarray]) -> GpuStep:
        """Advance all worlds by one control step (auto-resetting the finished ones).

        Args:
            actions: `{role: (nworld, 6)}` normalized wrenches, CUDA tensors or numpy arrays;
                clipped to `[-1, 1]` like the CPU env's action box.

        Returns:
            The `GpuStep` with the next-step observations, reward, flags and episode stats.
        """
        if not self._ready:
            raise RuntimeError("call reset() before step()")
        cfg = self.config
        commands = self._parse_actions(actions)
        self._last_command = commands
        wrenches: dict[str, torch.Tensor] = {}
        for role in AGENT_KEYS:
            rand = self._rand_agents[role]
            buffer = self._delay[role]
            write_position = self._write_pos
            buffer[:, write_position, :] = commands[role]
            gather_index = ((write_position - rand["latency_steps"]) % self._delay_size).view(
                -1, 1, 1
            )
            applied = torch.gather(buffer, 1, gather_index.expand(-1, 1, _WRENCH_SIZE)).squeeze(1)
            applied = torch.where(
                self._injection_mask(role).unsqueeze(1), torch.zeros_like(applied), applied
            )
            # The CPU env reports the effective (post-latency, post-injection) normalized
            # wrench in the observation and scales it by authority * limit_scale * gain for
            # the physics; both happen here.
            self._last_applied[role] = applied
            wrenches[role] = (
                applied
                * self._authority[role].unsqueeze(0)
                * rand["limit_scale"].unsqueeze(1)
                * rand["gain"].unsqueeze(1)
            )
        self._write_pos = (self._write_pos + 1) % self._delay_size

        xfrc = self._xfrc
        xfrc.zero_()
        xfrc[:, self._core_id, :] = wrenches[AGENT_KEYS[0]]
        xfrc[:, self._layer_id, :] = wrenches[AGENT_KEYS[1]]
        rand_episode = self._rand_episode
        impulse_active = rand_episode["impulse_active"] & (
            self._step == rand_episode["impulse_step"]
        )
        xfrc[:, self._core_id, :3] += rand_episode["impulse_force"] * (
            impulse_active.unsqueeze(1).to(torch.float32)
        )
        if self._cuda_graph is not None:
            self._warp.capture_launch(self._cuda_graph)
        else:
            for _ in range(cfg.control_every):
                self._physics_step(self._model, self._data)
        self._step += 1
        observations = self._build_observations()

        # --- diagnostics / success / failure / reward (the CPU env's formulas) ------------
        qpos, qvel = self._qpos, self._qvel
        angle = qpos[:, self._hinge_qpos]
        drift_pos = torch.linalg.vector_norm(qpos[:, 0:3] - self._initial_core_pos, dim=1)
        dot = torch.clamp(torch.abs((self._initial_core_quat * qpos[:, 3:7]).sum(dim=1)), 0.0, 1.0)
        drift_rot = 2.0 * torch.acos(dot)
        lin_speed = torch.linalg.vector_norm(qvel[:, 0:3], dim=1)
        ang_speed = torch.sqrt((qvel[:, 3:6] ** 2).sum(dim=1) + qvel[:, self._hinge_dof] ** 2)
        succeeded = (
            ((angle - cfg.goal_angle).abs() <= cfg.success_angle_tol)
            & (drift_pos <= cfg.success_pos_tol)
            & (drift_rot <= cfg.success_rot_tol)
            & (lin_speed <= cfg.success_lin_speed_tol)
            & (ang_speed <= cfg.success_ang_speed_tol)
        )
        failed = (drift_pos > cfg.failure_pos_drift) | (drift_rot > cfg.failure_rot_drift)
        reward = torch.where(
            succeeded,
            self._reward_success,
            torch.where(failed, self._reward_failure, self._reward_step),
        )
        # Potential-based shaping on non-terminal steps only (Φ telescopes; see the CPU env).
        progress = (
            (self._angle_initial - cfg.goal_angle).abs() - (angle - cfg.goal_angle).abs()
        ) / cfg.goal_angle
        progress = progress.clamp(0.0, 1.0)
        excess_pos = (
            (drift_pos - cfg.success_pos_tol) / (cfg.failure_pos_drift - cfg.success_pos_tol)
        ).clamp(min=0.0)
        excess_rot = (
            (drift_rot - cfg.success_rot_tol) / (cfg.failure_rot_drift - cfg.success_rot_tol)
        ).clamp(min=0.0)
        drift_excess = (excess_pos + excess_rot).clamp(max=1.0)
        potential_now = cfg.shaping_angle_coef * progress - cfg.shaping_drift_coef * drift_excess
        non_terminal = ~(succeeded | failed)
        reward = reward + torch.where(
            non_terminal, potential_now - self._prev_potential, self._zero
        )
        self._prev_potential = potential_now
        self._episode_return += reward.to(torch.float64)
        self._episode_length += 1
        terminated = succeeded | failed
        truncated = non_terminal & (self._step >= cfg.max_control_steps)

        # --- finished-episode statistics, then the auto-reset of the finished worlds -------
        final_obs = observations
        finished_returns: list[float] = []
        finished_lengths: list[int] = []
        finished_successes: list[bool] = []
        finished_angles: list[float] = []
        finished_drift_pos: list[float] = []
        finished_drift_rot: list[float] = []
        done = terminated | truncated
        done_rows = done.nonzero(as_tuple=False).flatten().cpu().tolist()
        if done_rows:
            indices = torch.as_tensor(done_rows, dtype=torch.int64, device=self._torch_device)
            rows = (
                torch.stack(
                    (
                        self._episode_return[indices],
                        self._episode_length[indices].to(torch.float64),
                        succeeded[indices].to(torch.float64),
                        torch.rad2deg(angle[indices].to(torch.float64)),
                        drift_pos[indices].to(torch.float64),
                        drift_rot[indices].to(torch.float64),
                    ),
                    dim=1,
                )
                .cpu()
                .tolist()
            )
            for (
                episode_return,
                length,
                success,
                angle_deg,
                drift_pos_value,
                drift_rot_value,
            ) in rows:
                finished_returns.append(float(episode_return))
                finished_lengths.append(int(length))
                finished_successes.append(bool(success))
                finished_angles.append(float(angle_deg))
                finished_drift_pos.append(float(drift_pos_value))
                finished_drift_rot.append(float(drift_rot_value))
            self._reset_worlds(indices)
        observations = self._build_observations()
        return GpuStep(
            obs=observations,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            final_obs=final_obs,
            finished_returns=finished_returns,
            finished_lengths=finished_lengths,
            finished_successes=finished_successes,
            finished_angles=finished_angles,
            finished_drift_pos=finished_drift_pos,
            finished_drift_rot=finished_drift_rot,
        )

    def _reset_worlds(self, indices: torch.Tensor) -> None:
        """Auto-reset the finished worlds with fresh seeds from the wrapper's seed stream."""
        count = int(indices.numel())
        seeds = list(range(self._next_seed, self._next_seed + count))
        self._next_seed += count
        self._install_randomization(seeds, indices)

    # --- inspection ---------------------------------------------------------------------
    def state(self) -> dict[str, torch.Tensor]:
        """The raw `qpos (nworld, nq)` / `qvel (nworld, nv)` CUDA views of the current state.

        Zero-copy views (like the CPU env's `state()` accessor, but batched): the parity
        script uses them to compare the physics of the two backends world by world. Do not
        keep them across a `step` if you need a snapshot -- use `.clone()`.
        """
        return {"qpos": self._qpos, "qvel": self._qvel}

    def randomization_tensors(self) -> dict[str, dict[str, torch.Tensor]]:
        """The per-agent sampled randomisation tensors of the current episodes (read-only).

        Exposed for the parity check and tests: this is the ON-DEVICE copy of
        `_wrench_logic.sample_episode_randomization` of each world's seed, so the parity
        script can verify that the CPU sampler's values reached the GPU unchanged.
        """
        return {role: dict(self._rand_agents[role]) for role in AGENT_KEYS}

    def episode_randomization_tensors(self) -> dict[str, torch.Tensor]:
        """The episode-level randomisation tensors (initial angle, friction, impulse)."""
        return dict(self._rand_episode)

    # --- helpers ------------------------------------------------------------------------
    def _parse_actions(
        self, actions: dict[str, torch.Tensor | np.ndarray]
    ) -> dict[str, torch.Tensor]:
        """Validate the action dict and return clipped float32 CUDA tensors."""
        if not isinstance(actions, dict):
            raise TypeError(f"actions must be a dict with keys {AGENT_KEYS}, got {type(actions)}")
        if set(actions) != set(AGENT_KEYS):
            raise ValueError(f"action keys must be exactly {AGENT_KEYS}, got {sorted(actions)}")
        expected = (self.num_envs, _WRENCH_SIZE)
        parsed: dict[str, torch.Tensor] = {}
        for role in AGENT_KEYS:
            value = actions[role]
            tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"action[{role!r}] must have shape {expected}, got {tuple(tensor.shape)}"
                )
            parsed[role] = tensor.to(device=self._torch_device, dtype=torch.float32).clamp(
                -1.0, 1.0
            )
        return parsed

    def synchronize(self) -> None:
        """Block until the queued Warp kernels of the current device have finished."""
        self._warp.synchronize()

    def close(self) -> None:
        """Release nothing (kept for symmetry with `VecWrenchCube.close`)."""
