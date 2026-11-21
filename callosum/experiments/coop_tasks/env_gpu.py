"""Batched GPU backend of the two-agent coop suite (`mujoco_warp`; optional `sandbox-gpu` extra).

The CPU path (`callosum.experiments.coop_tasks.vec_env.VecCoopEnv`) drives `num_envs` copies of
one `CoopTaskEnv` in a Python loop of `mj_step` calls; the pod's CPU quota turns the suite's
400-1200-update runs into hours. This module is the batched GPU twin for all three tasks
(`cartpole`, `pendulum`, `toss`): all `nworld` worlds live in ONE MuJoCo-Warp `Model`/`Data`
pair, the control step of `callosum.experiments.coop_tasks.base.CoopTaskEnv` (latency queues,
start-delay suppression, one-sided scaling, the paddle pose interpolation, the
`control_every` physics substeps, observations, reward, success/failure and the auto-reset of
the finished worlds) is tensor algebra on the GPU, and one Python `step()` advances thousands
of environments.

The env contract is `VecCoopEnv`'s: `reset(seed)`, `step({"left": ..., "right": ...})`,
per-agent observations, the shared team reward, `terminated`/`truncated`, `final_obs` for the
truncation bootstrap and the `finished_*` statistics of the episodes that ended on the step.
`torch_io = True` marks the tensor contract for `callosum.experiments.cube_wrench.train` and
`train.evaluate_greedy`, so the trainer consumes the CUDA tensors directly. The pure rules of
the suite are NOT duplicated: the observation layout comes from
`callosum.experiments.coop_tasks.logic.obs_fields`, the episode randomisation from its
samplers (`sample_episode_randomization`/`sample_toss_randomization`), and the
success/failure/potential formulas are transcribed from `logic.py` once, in `_task_reward`.

Task specifics:

* `cartpole` and `pendulum` are contact-free reduced models with two/one DOFs: per-world
  `dof_damping` (`put_model(batch_sizes={"dof_damping": nworld})`) carries the sampled hidden
  joint damping, the agents' forces/torques go through `xfrc_applied`, and the observation
  blocks are assembled from `qpos`/`qvel` exactly in the CPU layout/order.
* `toss` is the suite's contact/timing task (free ball + two MOCAP paddles). Per-world
  `geom_solref`/`geom_friction` carry the sampled contact restitution/friction, the paddles'
  poses are interpolated over the `control_every` substeps and written to `mocap_pos`/
  `mocap_quat`, and the ball/paddle contacts are detected per substep with small Warp kernels
  so the alternating pass counter of `toss_catch.py` keeps its CPU semantics -- including the
  anti-chatter gate: a pass counts only on the contacting paddle's own court half and at
  least `min_pass_interval_steps` control steps after the previous counted pass. The paddles'
  per-substep poses live in a constant pose/quaternion bank computed with torch before the
  graph replay, so no quaternion trigonometry happens inside the captured kernels.

CUDA graph: the whole control step (the substep loop with its per-substep paddle-pose writes,
contact flags and pass tracker) is captured ONCE into a CUDA graph (`warp.ScopedCapture`, the
pattern of `mujoco_warp`'s own speed tests) and replayed per control step: the tiny models are
launch-bound without it. The graph references the fixed Warp arrays, so the per-step torch
writes (`qpos`/`qvel`/`xfrc_applied`/per-world damping or contact fields/the paddle pose
banks) are seen by the replay. A capture failure is not fatal: the env warns once and falls
back to the eager loop (`use_cuda_graph=False` forces it).

Randomisation: the per-episode records are sampled on the CPU with the SAME stdlib samplers
as the CPU env (`pack_episode_randomization` uploads them as tensors; an auto-reset batch only
samples the worlds that finished), so a batch world reproduces the CPU env of the same seed
subject to the float32 Warp physics against the float64 MuJoCo CPU path.

Install (GPU pod only): the dependency is Linux/CUDA-only, so it lives in the OPTIONAL
`sandbox-gpu` extra and is never installed by the default/macOS/CI sync:

    uv pip install --python /root/.callosum/venv/bin/python mujoco-warp==4.0.1

The module imports `mujoco`, `mujoco_warp`, `warp` and `torch` lazily from the constructor, so
importing the package on a machine without the GPU stack fails only when an instance is built
(scripts/tests gate on `importorskip`); `pack_episode_randomization` and
`observation_blocks` are pure and unit-tested on the Mac.

Known limits: `torch.float32` physics (the CPU reference is float64), so the two backends are
interchangeable for training/evaluation but a curve is not bit-identical across them; the
`toss` contacts are floating-point-sensitive, so the parity tolerance there is looser than for
the contact-free tasks (see `scripts/experiments/coop_tasks/check_coop_gpu_parity.py`). The
float32 Warp contact solver can also produce a non-finite `qpos`/`qvel` on a rare deep
high-speed ball/paddle impact where float64 MuJoCo stays finite: the env treats such a world as
a FAILED episode and auto-resets it (`CoopBatchGPU.physics_failures` counts those events, which
the parity script reports as a separate diagnostic), so a run's policies never see NaN.
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

try:  # optional Linux/CUDA-only backend (`sandbox-gpu` extra); absent on macOS/CI.
    import warp as wp
except ImportError:  # pragma: no cover - only reachable without the GPU stack
    wp = None  # type: ignore[assignment]

from callosum.experiments.coop_tasks.config import (
    CartPoleConfig,
    PendulumConfig,
    TossConfig,
    coop_config,
)
from callosum.experiments.coop_tasks.logic import (
    AGENT_KEYS,
    TIME_FIELD,
    CoopTaskSpec,
    EpisodeRandomization,
    obs_fields,
    obs_size,
    partner_of,
    sample_episode_randomization,
    sample_toss_randomization,
    task_spec,
    validate_partner_mode,
    validate_task,
)

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_ASSET_FILES = {
    "cartpole": "cartpole.xml",
    "pendulum": "pendulum.xml",
    "toss": "toss_catch.xml",
}
"""MJCF file of each suite task under `callosum/experiments/coop_tasks/assets/`."""

_BASE_AGENT_FIELDS = (
    "latency_s",
    "latency_steps",
    "gain",
    "limit_scale",
    "start_delay_s",
    "start_delay_steps",
)
"""Per-agent fields of `logic.AgentRandomization`, in packing order."""

_INT_AGENT_FIELDS = frozenset({"latency_steps", "start_delay_steps"})

_MIN_CONTACT_FRICTION = 1e-4
"""Lower bound of the per-world toss-catch sliding friction written into `mujoco_warp`.

The CPU config samples `contact_friction` in `(0.0, 0.15)`, and MuJoCo's own `mj_setConst`
clamps a contact friction below `mjMINMU` (1e-5); `mujoco_warp` instead warns that such a
contact "may cause NaN" and the float32 contact solver does produce NaN on nearly frictionless
ball/paddle impacts. The GPU backend therefore clamps the written value (it does not change the
sampled randomisation record, which the parity script compares): the CPU value is within 1e-4
of the clamped one, i.e. physically indistinguishable from the frictionless limit the sampler
already allows, while the batched training step stays finite.
"""

_PARAM_FIELDS = ("latency_s", "gain", "limit_scale")
"""Per-agent fields of the `own_params` observation block, in the CPU order."""

_OWN_BLOCK_FIELDS = {
    "action": lambda spec: tuple(f"own_action_{index}" for index in range(spec.action_dim)),
    "params": lambda spec: ("own_latency", "own_gain", "own_limit_scale"),
    "privileged": lambda spec: ("own_start_delay",),
}
_PARTNER_BLOCK_FIELDS = {
    "action": lambda spec: tuple(f"partner_action_{index}" for index in range(spec.action_dim)),
    "command": lambda spec: tuple(f"partner_command_{index}" for index in range(spec.action_dim)),
    "params": lambda spec: ("partner_latency", "partner_gain", "partner_limit_scale"),
    "privileged": lambda spec: ("partner_start_delay",),
}


# --- observation layout -------------------------------------------------------------------


def observation_blocks(
    spec: CoopTaskSpec, partner_mode: str
) -> dict[str, tuple[tuple[str, str | None], ...]]:
    """The ordered `(tensor_key, owner)` blocks of each agent's flat observation.

    `owner` is the role whose tensors the block is read from (the agent itself for its own
    blocks, the partner under `"oracle"`), or `None` for the shared-state/goal/time blocks.
    The blocks are validated against `callosum.experiments.coop_tasks.logic.obs_fields`, so a
    change of the pure observation layout (a new field, a reordered block) fails at
    construction instead of silently producing an observation the policy was not trained on.

    Args:
        spec: The task's observation/action layout.
        partner_mode: Observation visibility mode, `"none"` or `"oracle"`.

    Returns:
        `{role: ((tensor_key, owner), ...)}` in observation column order.

    Raises:
        RuntimeError: if the blocks do not concatenate to `logic.obs_fields`.
    """
    validate_partner_mode(partner_mode)
    expected: list[str] = []
    for key in ("action", "params", "privileged"):
        expected.extend(_OWN_BLOCK_FIELDS[key](spec))
    expected.extend(spec.shared_fields)
    expected.extend(spec.goal_fields)
    expected.append(TIME_FIELD)
    if partner_mode != "none":
        for key in ("action", "command", "params", "privileged"):
            expected.extend(_PARTNER_BLOCK_FIELDS[key](spec))
    if tuple(expected) != obs_fields(spec, partner_mode):
        raise RuntimeError(
            "the GPU observation blocks and callosum.experiments.coop_tasks.logic.obs_fields "
            f"disagree for task {spec.name!r}: blocks concatenate to {len(expected)} fields, "
            f"obs_fields has {len(obs_fields(spec, partner_mode))}; update "
            "callosum/experiments/coop_tasks/env_gpu.py next to the logic module"
        )
    blocks: dict[str, tuple[tuple[str, str | None], ...]] = {}
    for role in AGENT_KEYS:
        role_blocks: list[tuple[str, str | None]] = [
            (key, role) for key in ("action", "params", "privileged")
        ]
        role_blocks.extend((key, None) for key in ("shared", "goal", "time"))
        if partner_mode != "none":
            role_blocks.extend(
                (key, partner_of(role)) for key in ("action", "command", "params", "privileged")
            )
        blocks[role] = tuple(role_blocks)
    return blocks


# --- episode randomisation packing --------------------------------------------------------


def _agent_tensor(values: list, name: str, device: torch.device) -> torch.Tensor:
    """One agent randomisation field as a tensor (int64 or float32) on `device`."""
    if name in _INT_AGENT_FIELDS:
        return torch.tensor(values, dtype=torch.int64, device=device)
    return torch.tensor(values, dtype=torch.float32, device=device)


def pack_episode_randomization(
    records: list[EpisodeRandomization], device: torch.device | str = "cpu"
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Pack a batch of per-episode randomisation records into `(nworld, ...)` tensors.

    Pure torch (no Warp), so the packing is unit-testable on the Mac. The tensor values are
    exactly the sampled dataclass values (the CPU env's samplers), so a batch world reproduces
    the CPU env of the same seed subject only to the float32 physics upload. A
    `TossEpisodeRandomization` additionally carries the hidden throw/contact fields; they are
    packed when the records have them.

    Args:
        records: One record per world, in world order.
        device: Torch device of the produced tensors.

    Returns:
        `(agents, episode)`: `agents[role][field]` tensors of shape `(nworld,)` (int64 for the
        step counts, float32 for the rest) and `episode` with `initial_value`, `damping`,
        `wind_active` (bool), `wind_step` (int64), `wind_force`, plus for the toss-catch
        `launch_velocity` (nworld, 2), `launch_position` (nworld, 2), `release_delay_steps`
        (int64), `contact_dampratio` and `contact_friction`.
    """
    if torch is None:  # pragma: no cover - only reachable without the train extra
        raise ImportError("pack_episode_randomization needs torch; install the `train` extra")
    if not records:
        raise ValueError("records must contain at least one EpisodeRandomization")
    device = torch.device(device)
    agents: dict[str, dict[str, torch.Tensor]] = {}
    for role_index, role in enumerate(AGENT_KEYS):
        agents[role] = {
            name: _agent_tensor(
                [getattr(record.agents[role_index], name) for record in records], name, device
            )
            for name in _BASE_AGENT_FIELDS
        }
    episode: dict[str, torch.Tensor] = {
        "initial_value": torch.tensor(
            [record.initial_value for record in records], dtype=torch.float32, device=device
        ),
        "damping": torch.tensor(
            [record.damping for record in records], dtype=torch.float32, device=device
        ),
        "wind_active": torch.tensor(
            [record.wind_active for record in records], dtype=torch.bool, device=device
        ),
        "wind_step": torch.tensor(
            [record.wind_step for record in records], dtype=torch.int64, device=device
        ),
        "wind_force": torch.tensor(
            [record.wind_force for record in records], dtype=torch.float32, device=device
        ),
    }
    if hasattr(records[0], "contact_dampratio"):
        episode.update(
            {
                "launch_velocity": torch.tensor(
                    [record.launch_velocity for record in records],
                    dtype=torch.float32,
                    device=device,
                ),
                "launch_position": torch.tensor(
                    [record.launch_position for record in records],
                    dtype=torch.float32,
                    device=device,
                ),
                "release_delay_steps": torch.tensor(
                    [record.release_delay_steps for record in records],
                    dtype=torch.int64,
                    device=device,
                ),
                "contact_dampratio": torch.tensor(
                    [record.contact_dampratio for record in records],
                    dtype=torch.float32,
                    device=device,
                ),
                "contact_friction": torch.tensor(
                    [record.contact_friction for record in records],
                    dtype=torch.float32,
                    device=device,
                ),
            }
        )
    return agents, episode


# --- Warp kernels used by the toss-catch control step -------------------------------------
#
# They are defined at module level only where Warp is importable (the `sandbox-gpu` extra), so
# importing this module on macOS/CI does not need the GPU stack.

if wp is not None:

    @wp.kernel
    def _apply_paddle_pose(
        pose: wp.array3d(dtype=wp.vec3),
        quat: wp.array3d(dtype=wp.quat),
        substep: int,
        pad_mocap: wp.array(dtype=int),
        mocap_pos: wp.array2d(dtype=wp.vec3),
        mocap_quat: wp.array2d(dtype=wp.quat),
        flags: wp.array2d(dtype=int),
    ):
        """Write the substep's interpolated paddle pose and clear its contact flag."""
        world, side = wp.tid()
        mocap = pad_mocap[side]
        pose_world = pose[substep, world, side]
        mocap_pos[world, mocap] = wp.vec3(pose_world[0], 0.0, pose_world[1])
        mocap_quat[world, mocap] = quat[substep, world, side]
        flags[world, side] = 0

    @wp.kernel
    def _contact_flags(
        contact_geom: wp.array(dtype=wp.vec2i),
        contact_worldid: wp.array(dtype=int),
        nacon: wp.array(dtype=int),
        ball_geom: int,
        left_geom: int,
        right_geom: int,
        flags: wp.array2d(dtype=int),
    ):
        """Mark the `(world, paddle)` pairs whose ball contact exists in this substep."""
        index = wp.tid()
        if index < nacon[0]:
            geom = contact_geom[index]
            world = contact_worldid[index]
            first = geom[0]
            second = geom[1]
            if (first == ball_geom and second == left_geom) or (
                first == left_geom and second == ball_geom
            ):
                flags[world, 0] = 1
            if (first == ball_geom and second == right_geom) or (
                first == right_geom and second == ball_geom
            ):
                flags[world, 1] = 1

    @wp.kernel
    def _update_passes(
        flags: wp.array2d(dtype=int),
        contact_side: wp.array2d(dtype=int),
        passes: wp.array(dtype=int),
        last_side: wp.array(dtype=int),
        cooldown: wp.array(dtype=int),
        qpos: wp.array2d(dtype=wp.float32),
        ball_qpos: int,
        min_interval: int,
    ):
        """The CPU `toss_catch.TossCatchEnv._on_substep` pass gate, one substep at a time.

        A new ball/paddle contact is counted only when the sides alternate, the ball centre
        is on the contacting paddle's own half of the court (left: `x < 0`, right: `x > 0`,
        the CPU `logic.toss_ball_on_side` rule) and no counted pass happened in the last
        `min_interval` control steps. `cooldown` is aged once per control step by
        `CoopBatchGPU.step` and armed to `min_interval` here, so this per-substep kernel
        needs no step counter (and stays identical between the graph and the eager path).
        """
        world = wp.tid()
        ball_x = qpos[world, ball_qpos]
        now_left = flags[world, 0]
        if (
            now_left == 1
            and contact_side[world, 0] == 0
            and last_side[world] != 0
            and ball_x < 0.0
            and cooldown[world] == 0
        ):
            passes[world] = passes[world] + 1
            last_side[world] = 0
            cooldown[world] = min_interval
        contact_side[world, 0] = now_left
        now_right = flags[world, 1]
        if (
            now_right == 1
            and contact_side[world, 1] == 0
            and last_side[world] != 1
            and ball_x > 0.0
            and cooldown[world] == 0
        ):
            passes[world] = passes[world] + 1
            last_side[world] = 1
            cooldown[world] = min_interval
        contact_side[world, 1] = now_right


# --- step result --------------------------------------------------------------------------


@dataclass
class CoopGpuStep:
    """One control step of all worlds (the auto-reset contract mirrors `VecCoopEnv`).

    The tensors live on the env's CUDA device and are NOT copied to the host, except the small
    `finished_*` lists of the episodes that ended on this step (the trainer/evaluator consume
    them as Python values, as with the CPU vec env).

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
        finished_angles: Final primary diagnostic of those episodes (`angle_deg` for the
            cart-pole, `amplitude_deg` for the pendulum, `passes` for the toss-catch), the
            same value `logic.task_diagnostic` gives on the CPU.
        finished_infos: The CPU env's terminal info dicts (success + the task diagnostics) of
            those episodes, in the same order as the other `finished_*` lists.
        nonfinite: `(nworld,)` bool mask of the worlds that the float32 contact solver made
            non-finite on this step (treated as a FAILED episode and auto-reset; the CPU
            float64 path has no such event, so the parity script reports/skips those rows).
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
    finished_infos: list[dict] = field(default_factory=list)
    nonfinite: torch.Tensor | None = None

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
            f"CoopBatchGPU runs on a CUDA device of mujoco_warp, got device={device!r}"
        )
    return normalized


def _steps(control_dt: float, seconds: float) -> int:
    """Convert seconds to whole control steps (the same rounding as `logic._steps`)."""
    return max(0, round(seconds / control_dt))


class CoopBatchGPU:
    """`nworld` worlds of one coop task stepped together on the GPU through `mujoco_warp`.

    The public contract (reset/step, the per-role action dict of normalized commands, flat
    per-role observations, the shared team reward, success/failure/truncation, auto-reset) is
    the CPU `VecCoopEnv`'s; the difference is that every value is a `(nworld, ...)` CUDA
    tensor and one `step` advances all worlds. `torch_io = True` marks the tensor contract
    for `callosum.experiments.cube_wrench.train` / `evaluate_greedy`.
    """

    torch_io = True
    """Marker for the trainer/evaluator: `reset`/`step` consume and return torch tensors."""

    def __init__(
        self,
        task: str,
        num_envs: int,
        partner_mode: str = "none",
        config: CartPoleConfig | PendulumConfig | TossConfig | None = None,
        seed: int = 0,
        device: str = "cuda",
        use_cuda_graph: bool = True,
    ) -> None:
        """Build the Warp model/data pair and the per-world episode tensors.

        Args:
            task: The coop task (`"cartpole"`, `"pendulum"` or `"toss"`).
            num_envs: Number of worlds stepped together (>= 1).
            partner_mode: Observation visibility mode, `"none"` or `"oracle"`.
            config: The task config; the task's default config when `None`.
            seed: Base episode seed of the auto-reset stream (`reset(seed)` seeds world `i`
                with `seed + i`; auto-resets continue from `seed + num_envs`, exactly like
                `VecCoopEnv`).
            device: CUDA device string (`"cuda"` or `"cuda:N"`).
            use_cuda_graph: Capture the control step (the `control_every`-substep physics
                loop plus, for the toss-catch, the per-substep paddle-pose/contact kernels)
                into a CUDA graph; `False` forces the eager loop (debugging and the
                capture-failure fallback path).

        Raises:
            ValueError: on a bad task/nworld/partner mode/device or a config of another task.
            RuntimeError: if the model layout does not match this backend's assumptions.
        """
        if torch is None:  # pragma: no cover - only reachable without the train extra
            raise ImportError("CoopBatchGPU needs torch; install the `train` extra")
        validate_task(task)
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        validate_partner_mode(partner_mode)
        device = _normalize_device(device)

        # Lazy simulator imports: the pure-logic/torch parts of the package stay importable
        # without the GPU stack (and the macOS test suite skips these tests).
        import mujoco
        import mujoco_warp

        if wp is None:  # pragma: no cover - defined only when warp imports
            raise ImportError("CoopBatchGPU needs warp; install mujoco-warp==4.0.1")
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

        self.task = task
        self.device = device
        self._torch_device = torch.device(device)
        self.config = config if config is not None else coop_config(task)
        if getattr(self.config, "task", task) != task:
            raise ValueError(
                f"config is for task {getattr(self.config, 'task', None)!r}, got task={task!r}"
            )
        cfg = self.config
        self.partner_mode = partner_mode
        self.num_envs = int(num_envs)
        self._spec = task_spec(task)
        self.obs_dim = obs_size(self._spec, partner_mode)
        self.action_dim = self._spec.action_dim
        self._base_seed = int(seed)
        self._next_seed = self._base_seed + self.num_envs

        # --- host model with the config as truth (same writes as the CPU env) -------------
        model = mujoco.MjModel.from_xml_path(str(_ASSETS_DIR / _ASSET_FILES[task]))
        self._resolve_model_ids(model)
        self._write_host_model(model, cfg)
        self._model = mujoco_warp.put_model(model, batch_sizes=self._batch_sizes())
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
        try:  # nu == 0 for the reduced models; keep the field optional across warp versions
            self._ctrl = wp.to_torch(self._data.ctrl)
        except (AttributeError, TypeError):  # pragma: no cover - warp-version dependent
            self._ctrl = None
        if self._qpos.dtype != torch.float32:
            raise RuntimeError(f"unexpected qpos dtype {self._qpos.dtype}")
        self._mocap_pos = None
        self._mocap_quat = None
        self._contact_geom = None
        self._contact_worldid = None
        self._model_damping = None
        self._model_solref = None
        self._model_friction = None
        if task == "toss":
            self._mocap_pos = wp.to_torch(self._data.mocap_pos)
            self._mocap_quat = wp.to_torch(self._data.mocap_quat)
            self._contact_geom = wp.to_torch(self._data.contact.geom)
            self._contact_worldid = wp.to_torch(self._data.contact.worldid)
            self._model_solref = wp.to_torch(self._model.geom_solref)
            self._model_friction = wp.to_torch(self._model.geom_friction)
        if task in ("cartpole", "pendulum"):
            self._model_damping = wp.to_torch(self._model.dof_damping)

        # --- per-agent episode state ------------------------------------------------------
        zeros = self._torch_device
        self._authority = {
            role: torch.tensor(float(cfg.agent_authority(index)), dtype=torch.float32, device=zeros)
            for index, role in enumerate(AGENT_KEYS)
        }
        self._signs = {
            role: torch.tensor(float(sign), dtype=torch.float32, device=zeros)
            for role, sign in zip(AGENT_KEYS, self._agent_signs())
        }
        # The latency queue: `latency_steps` commands are in flight and the command of
        # `latency_steps` steps ago takes effect (a zero command during the initial latency,
        # exactly like the CPU env's deque initialised with zeros).
        self._delay_size = max(1, _steps(cfg.control_dt, cfg.randomization.latency_range[1]) + 1)
        self._write_pos = 0
        self._delay = {
            role: torch.zeros(
                (self.num_envs, self._delay_size, self.action_dim),
                dtype=torch.float32,
                device=zeros,
            )
            for role in AGENT_KEYS
        }
        self._last_applied = {
            role: torch.zeros((self.num_envs, self.action_dim), dtype=torch.float32, device=zeros)
            for role in AGENT_KEYS
        }
        self._last_command = {
            role: torch.zeros_like(self._last_applied[role]) for role in AGENT_KEYS
        }
        self._params = {
            role: torch.zeros(
                (self.num_envs, len(_PARAM_FIELDS)), dtype=torch.float32, device=zeros
            )
            for role in AGENT_KEYS
        }
        self._rand_agents = {role: self._new_agent_randomization_buffers() for role in AGENT_KEYS}
        self._rand_episode = self._new_episode_randomization_buffers()
        self._step = torch.zeros(self.num_envs, dtype=torch.int64, device=zeros)
        self._goal = torch.zeros(
            (self.num_envs, len(self._spec.goal_fields)), dtype=torch.float32, device=zeros
        )
        self._time_fraction = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=zeros)
        self._prev_potential = torch.zeros(self.num_envs, dtype=torch.float32, device=zeros)
        self._episode_return = torch.zeros(self.num_envs, dtype=torch.float64, device=zeros)
        self._episode_length = torch.zeros(self.num_envs, dtype=torch.int64, device=zeros)
        self._obs_blocks = observation_blocks(self._spec, partner_mode)
        self._zero = torch.zeros((), dtype=torch.float32, device=zeros)
        self._reward_success = torch.tensor(
            cfg.reward.success_bonus - cfg.reward.step_penalty,
            dtype=torch.float32,
            device=zeros,
        )
        self._reward_failure = torch.tensor(
            cfg.reward.failure_penalty, dtype=torch.float32, device=zeros
        )
        self._reward_step = torch.tensor(
            -cfg.reward.step_penalty, dtype=torch.float32, device=zeros
        )
        self._task_state = self._new_task_state_buffers()
        self._physics_failures = 0
        """Worlds whose qpos/qvel became non-finite in the float32 contact solver (guarded)."""
        self._ready = False
        self._cuda_graph = None
        self._cuda_graph_error: str | None = None
        if use_cuda_graph:
            self._cuda_graph = self._capture_physics_graph()

    # --- model setup --------------------------------------------------------------------
    def _agent_signs(self) -> tuple[float, float]:
        """Physical sign each agent commands (the CPU env's `AGENT_SIGNS`)."""
        if self.task == "toss":
            return (1.0, 1.0)
        return (1.0, -1.0)

    def _resolve_model_ids(self, model) -> None:
        """Resolve the task's joint/body/geom/mocap ids after loading the MJCF."""
        self._cart_joint = -1
        self._pole_joint = -1
        self._cart_qpos = self._cart_dof = self._pole_qpos = self._pole_dof = -1
        self._cart_body = self._pole_body = -1
        self._hinge_joint = -1
        self._hinge_qpos = self._hinge_dof = -1
        self._bob_body = -1
        self._ball_body = self._ball_geom = -1
        self._ball_qpos = self._ball_dof = -1
        self._pad_geom: dict[str, int] = {}
        self._pad_mocap: dict[str, int] = {}
        self._pad_mocap_ids: dict[str, int] = {}
        self._mocap_count = 0
        if self.task == "cartpole":
            self._cart_joint = int(model.joint("cart_slide").id)
            self._pole_joint = int(model.joint("pole_hinge").id)
            self._cart_qpos = int(model.jnt_qposadr[self._cart_joint])
            self._cart_dof = int(model.jnt_dofadr[self._cart_joint])
            self._pole_qpos = int(model.jnt_qposadr[self._pole_joint])
            self._pole_dof = int(model.jnt_dofadr[self._pole_joint])
            self._cart_body = int(model.body("cart").id)
            self._pole_body = int(model.body("pole").id)
            layout = (self._cart_qpos, self._cart_dof, self._pole_qpos, self._pole_dof)
            if layout != (0, 0, 1, 1):
                raise RuntimeError(
                    "this GPU backend assumes the cartpole.xml layout (cart_slide at qpos/dof "
                    f"0, pole_hinge at 1), got {layout}"
                )
        elif self.task == "pendulum":
            self._hinge_joint = int(model.joint("pivot_hinge").id)
            self._hinge_qpos = int(model.jnt_qposadr[self._hinge_joint])
            self._hinge_dof = int(model.jnt_dofadr[self._hinge_joint])
            self._bob_body = int(model.body("bob").id)
            layout = (self._hinge_qpos, self._hinge_dof)
            if layout != (0, 0):
                raise RuntimeError(
                    "this GPU backend assumes the pendulum.xml layout (pivot_hinge at qpos/dof "
                    f"0), got {layout}"
                )
        elif self.task == "toss":
            ball_joint = int(model.joint("ball_free").id)
            self._ball_qpos = int(model.jnt_qposadr[ball_joint])
            self._ball_dof = int(model.jnt_dofadr[ball_joint])
            self._ball_body = int(model.body("ball").id)
            self._ball_geom = int(model.geom("ball_geom").id)
            for side, body_name in (("left", "pad_left"), ("right", "pad_right")):
                body = model.body(body_name)
                self._pad_geom[side] = int(model.geom(f"{body_name}_geom").id)
                self._pad_mocap[side] = int(body.mocapid[0])
            layout = (self._ball_qpos, self._ball_dof, model.nq, model.nv)
            if layout != (0, 0, 7, 6):
                raise RuntimeError(
                    "this GPU backend assumes the toss_catch.xml layout (one free ball joint "
                    f"at qpos/dof 0, nq=7, nv=6), got qpos={self._ball_qpos}, "
                    f"dof={self._ball_dof}, nq={model.nq}, nv={model.nv}"
                )
            self._mocap_count = int(model.nmocap)

    def _write_host_model(self, model, cfg) -> None:
        """Write the config's timing/physics into the host model (config as truth)."""
        model.opt.timestep = cfg.dt
        model.opt.gravity[:] = (0.0, 0.0, -cfg.gravity)
        if self.task == "cartpole":
            model.jnt_range[self._cart_joint] = (-cfg.rail_limit, cfg.rail_limit)
            model.dof_damping[self._cart_dof] = cfg.cart_damping
            model.dof_damping[self._pole_dof] = cfg.pole_damping
            model.dof_armature[self._pole_dof] = cfg.pole_armature
        elif self.task == "pendulum":
            model.jnt_range[self._hinge_joint] = (-cfg.hinge_limit, cfg.hinge_limit)
            model.dof_damping[self._hinge_dof] = cfg.hinge_damping
            model.dof_armature[self._hinge_dof] = cfg.hinge_armature
        elif self.task == "toss":
            # The per-episode damping ratio/friction are batched per world and overwritten on
            # every reset; the host values below only seed the broadcast, and the nonzero
            # friction avoids mujoco_warp's zero-friction NaN warning at construction.
            for geom in self._toss_contact_geoms():
                model.geom_solref[geom, 1] = 0.14
                model.geom_friction[geom, 0] = 0.5

    def _batch_sizes(self) -> dict[str, int]:
        """The per-world model fields `mujoco_warp.put_model` must batch."""
        if self.task in ("cartpole", "pendulum"):
            return {"dof_damping": self.num_envs}
        return {"geom_solref": self.num_envs, "geom_friction": self.num_envs}

    def _toss_contact_geoms(self) -> tuple[int, ...]:
        """The ball/paddle geom ids whose sampled contact parameters are per-world."""
        return (self._ball_geom, self._pad_geom["left"], self._pad_geom["right"])

    # --- CUDA graph ---------------------------------------------------------------------
    def _control_step_physics(self) -> None:
        """Run the physics part of one control step (what the CUDA graph captures).

        For the contact-free tasks this is `control_every` plain `mujoco_warp.step` calls. The
        toss-catch additionally writes the substep's interpolated paddle pose (from the
        per-step pose/quaternion banks), clears the substep contact flags, detects the
        ball/paddle contacts and updates the alternating pass counter -- all inside the same
        captured sequence, so the graph replay reproduces the CPU `_on_substep` loop.
        """
        cfg = self.config
        if self.task != "toss":
            for _ in range(cfg.control_every):
                self._physics_step(self._model, self._data)
            return
        warp = self._warp
        for substep in range(cfg.control_every):
            warp.launch(
                _apply_paddle_pose,
                dim=(self.num_envs, 2),
                inputs=[
                    self._pose_bank,
                    self._quat_bank,
                    substep,
                    self._pad_mocap_array,
                    self._data.mocap_pos,
                    self._data.mocap_quat,
                    self._flags_warp,
                ],
            )
            self._physics_step(self._model, self._data)
            warp.launch(
                _contact_flags,
                dim=self._contact_geom.shape[0],
                inputs=[
                    self._data.contact.geom,
                    self._data.contact.worldid,
                    self._data.nacon,
                    self._ball_geom,
                    self._pad_geom["left"],
                    self._pad_geom["right"],
                    self._flags_warp,
                ],
            )
            warp.launch(
                _update_passes,
                dim=self.num_envs,
                inputs=[
                    self._flags_warp,
                    self._contact_side_warp,
                    self._passes_warp,
                    self._last_side_warp,
                    self._pass_cooldown_warp,
                    self._data.qpos,
                    self._ball_qpos,
                    cfg.min_pass_interval_steps,
                ],
            )

    def _capture_physics_graph(self):
        """Capture the control-step physics into a CUDA graph (or fail soft).

        The captured sequence touches only the fixed Warp model/data arrays, so it can be
        replayed after the per-step torch writes of `qpos`/`qvel`/`xfrc_applied`/the per-world
        damping or contact fields/the toss paddle pose banks. A capture failure (unsupported
        toolkit/driver) is not fatal: the env falls back to the eager loop.

        Returns:
            The captured `warp.Graph`, or `None` when capture is unsupported here.
        """
        try:
            # Warm up the lazy kernel compilation / solver scratch allocations outside the
            # capture (a CUDA graph cannot allocate); the state is overwritten by reset().
            self._control_step_physics()
            with self._warp.ScopedCapture() as capture:
                self._control_step_physics()
            return capture.graph
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - toolkit-dependent
            # A capture failure must never break the env: warn once and fall back to eager.
            self._cuda_graph_error = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                "CUDA graph capture of the coop control step failed "
                f"({self._cuda_graph_error}); falling back to the eager loop, which is "
                "much slower per control step",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def _run_control_step(self) -> None:
        """Replay the captured control step, or run the eager sequence when capture failed."""
        if self._cuda_graph is not None:
            self._warp.capture_launch(self._cuda_graph)
        else:
            self._control_step_physics()

    @property
    def cuda_graph(self) -> bool:
        """Whether the control step is replayed from a captured CUDA graph."""
        return self._cuda_graph is not None

    @property
    def cuda_graph_error(self) -> str | None:
        """Why the CUDA graph capture failed (`None` when the graph is active)."""
        return self._cuda_graph_error

    @property
    def physics_failures(self) -> int:
        """Finished episodes ended by the non-finite-state guard (a float32 contact-solver limit).

        The CPU float64 path does not need this guard; the counter lets a run report how often
        the GPU contact solver produced a non-finite state instead of failing silently.
        """
        return self._physics_failures

    def software_versions(self) -> str:
        """The simulator/GPU software versions of this env, for the trainer's startup log."""
        return " | ".join(f"{name} {version}" for name, version in self._software_versions.items())

    # --- buffers ------------------------------------------------------------------------
    def _new_agent_randomization_buffers(self) -> dict[str, torch.Tensor]:
        """Zero buffers of one agent's randomisation fields (same dtypes as the packer)."""
        buffers: dict[str, torch.Tensor] = {}
        for name in _BASE_AGENT_FIELDS:
            if name in _INT_AGENT_FIELDS:
                buffers[name] = torch.zeros(
                    self.num_envs, dtype=torch.int64, device=self._torch_device
                )
            else:
                buffers[name] = torch.zeros(
                    self.num_envs, dtype=torch.float32, device=self._torch_device
                )
        return buffers

    def _new_episode_randomization_buffers(self) -> dict[str, torch.Tensor]:
        """Zero buffers of the episode-level randomisation fields."""
        device = self._torch_device
        buffers = {
            "initial_value": torch.zeros(self.num_envs, dtype=torch.float32, device=device),
            "damping": torch.zeros(self.num_envs, dtype=torch.float32, device=device),
            "wind_active": torch.zeros(self.num_envs, dtype=torch.bool, device=device),
            "wind_step": torch.zeros(self.num_envs, dtype=torch.int64, device=device),
            "wind_force": torch.zeros(self.num_envs, dtype=torch.float32, device=device),
        }
        if self.task == "toss":
            buffers.update(
                {
                    "launch_velocity": torch.zeros(
                        (self.num_envs, 2), dtype=torch.float32, device=device
                    ),
                    "launch_position": torch.zeros(
                        (self.num_envs, 2), dtype=torch.float32, device=device
                    ),
                    "release_delay_steps": torch.zeros(
                        self.num_envs, dtype=torch.int64, device=device
                    ),
                    "contact_dampratio": torch.zeros(
                        self.num_envs, dtype=torch.float32, device=device
                    ),
                    "contact_friction": torch.zeros(
                        self.num_envs, dtype=torch.float32, device=device
                    ),
                }
            )
        return buffers

    def _new_task_state_buffers(self) -> dict[str, torch.Tensor]:
        """The task-specific episode trackers (pendulum amplitude, toss pass counter, paddles)."""
        device = self._torch_device
        cfg = self.config
        buffers: dict[str, torch.Tensor] = {}
        if self.task == "pendulum":
            buffers["peak_abs_angle"] = torch.zeros(
                self.num_envs, dtype=torch.float32, device=device
            )
            buffers["last_angle_sign"] = torch.ones(
                self.num_envs, dtype=torch.float32, device=device
            )
            buffers["completed_cycles"] = torch.zeros(
                self.num_envs, dtype=torch.int64, device=device
            )
        elif self.task == "toss":
            pads = 2
            self._pad_start = torch.zeros(
                (self.num_envs, pads, 3), dtype=torch.float32, device=device
            )
            self._pad_target = torch.zeros_like(self._pad_start)
            self._pad_velocity = torch.zeros_like(self._pad_start)
            self._launch_velocity = torch.zeros(
                (self.num_envs, 3), dtype=torch.float32, device=device
            )
            self._release_step = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
            self._pose_bank = wp.to_torch(
                wp.zeros((cfg.control_every, self.num_envs, pads), dtype=wp.vec3)
            )
            self._quat_bank = wp.to_torch(
                wp.zeros((cfg.control_every, self.num_envs, pads), dtype=wp.quat)
            )
            self._flags_warp = wp.zeros((self.num_envs, pads), dtype=int)
            self._contact_side_warp = wp.zeros((self.num_envs, pads), dtype=int)
            self._passes_warp = wp.zeros(self.num_envs, dtype=int)
            self._last_side_warp = wp.zeros(self.num_envs, dtype=int)
            self._pass_cooldown_warp = wp.zeros(self.num_envs, dtype=int)
            self._flags = wp.to_torch(self._flags_warp)
            self._contact_side = wp.to_torch(self._contact_side_warp)
            self._passes = wp.to_torch(self._passes_warp)
            self._last_side = wp.to_torch(self._last_side_warp)
            self._pass_cooldown = wp.to_torch(self._pass_cooldown_warp)
            self._last_pass_step = torch.full(
                (self.num_envs,), -1, dtype=torch.int64, device=device
            )
            self._pad_mocap_array = wp.array(
                np.asarray([self._pad_mocap["left"], self._pad_mocap["right"]], dtype=np.int32),
                dtype=int,
            )
            self._alpha = torch.tensor(
                [(substep + 1) / cfg.control_every for substep in range(cfg.control_every)],
                dtype=torch.float32,
                device=device,
            ).view(cfg.control_every, 1, 1)
            self._ready_pose = self._initial_pad_pose()
            self._pad_pose = torch.zeros_like(self._pad_start)
        return buffers

    def _initial_pad_pose(self) -> torch.Tensor:
        """The ready pose `(2, 3)` of the two paddles: `(x, z, tilt)` per side, CPU-identical."""
        cfg = self.config
        mid_x = cfg.pad_inner_x + 0.10
        pose = torch.empty((2, 3), dtype=torch.float32, device=self._torch_device)
        pose[0, 0] = -mid_x
        pose[1, 0] = mid_x
        pose[:, 1] = cfg.pad_start_z
        pose[0, 2] = cfg.pad_tilt
        pose[1, 2] = -cfg.pad_tilt
        return pose

    # --- randomisation / reset ----------------------------------------------------------
    def _sample_records(self, seeds: list[int]) -> list[EpisodeRandomization]:
        """Sample the episode randomisation of `seeds` with the CPU env's stdlib samplers."""
        if self.task == "toss":
            return [
                sample_toss_randomization(
                    self.config.randomization,
                    control_dt=self.config.control_dt,
                    max_control_steps=self.config.max_control_steps,
                    seed=seed,
                )
                for seed in seeds
            ]
        return [
            sample_episode_randomization(
                self.config.randomization,
                control_dt=self.config.control_dt,
                max_control_steps=self.config.max_control_steps,
                seed=seed,
            )
            for seed in seeds
        ]

    def _install_randomization(self, seeds: list[int], indices: torch.Tensor | None) -> None:
        """Sample `seeds` on the CPU and install them (and the fresh episode state) on device.

        `indices=None` resets every world (a full `reset`); otherwise only the given rows are
        overwritten (the auto-reset of the worlds that finished).
        """
        records = self._sample_records(seeds)
        agents, episode = pack_episode_randomization(records, self._torch_device)
        if indices is None:
            for role in AGENT_KEYS:
                for name, tensor in agents[role].items():
                    self._rand_agents[role][name].copy_(tensor)
                self._params[role].copy_(self._agent_params(agents[role]))
                self._last_applied[role].zero_()
                self._last_command[role].zero_()
                self._delay[role].zero_()
            for name, tensor in episode.items():
                self._rand_episode[name].copy_(tensor)
            self._qpos.zero_()
            self._qvel.zero_()
            self._install_initial_state(None)
            self._write_world_randomization(None)
            self._write_task_state(None)
            self._reset_solver_state(None)
            self._step.zero_()
            self._episode_return.zero_()
            self._episode_length.zero_()
            self._write_pos = 0
            self._reset_prev_potential(None)
            return
        for role in AGENT_KEYS:
            for name, tensor in agents[role].items():
                self._rand_agents[role][name][indices] = tensor
            self._params[role][indices] = self._agent_params(agents[role])
            self._last_applied[role][indices] = 0.0
            self._last_command[role][indices] = 0.0
            self._delay[role][indices] = 0.0
        for name, tensor in episode.items():
            self._rand_episode[name][indices] = tensor
        self._qpos[indices] = 0.0
        self._qvel[indices] = 0.0
        self._step[indices] = 0
        self._install_initial_state(indices)
        self._write_world_randomization(indices)
        self._write_task_state(indices)
        self._reset_solver_state(indices)
        self._episode_return[indices] = 0.0
        self._episode_length[indices] = 0
        self._reset_prev_potential(indices)

    def _reset_prev_potential(self, indices: torch.Tensor | None) -> None:
        """Set the shaping baseline to the potential of the freshly reset state.

        The CPU env stores `_prev_potential = _task_potential()` at the end of `reset`, so the
        first step's shaping delta is measured from the reset state; the GPU path must do the
        same, otherwise the first reward of every episode would carry the whole initial
        potential (`~0.5` for the cart-pole's saturated position term).
        """
        _, _, potential = self._task_reward()
        if indices is None:
            self._prev_potential.copy_(potential)
        else:
            self._prev_potential[indices] = potential[indices]

    def _install_initial_state(self, indices: torch.Tensor | None) -> None:
        """Write the episode's initial `qpos`/`qvel` (and ball pose for the toss-catch)."""
        initial = self._rand_episode["initial_value"]
        if self.task == "cartpole":
            if indices is None:
                self._qpos[:, self._pole_qpos] = initial
            else:
                self._qpos[indices, self._pole_qpos] = initial[indices]
        elif self.task == "pendulum":
            if indices is None:
                self._qpos[:, self._hinge_qpos] = initial
            else:
                self._qpos[indices, self._hinge_qpos] = initial[indices]
        else:
            launch = self._rand_episode["launch_position"]
            if indices is None:
                self._qpos[:, 0] = launch[:, 0]
                self._qpos[:, 1] = 0.0
                self._qpos[:, 2] = launch[:, 1]
                self._qpos[:, 3] = 1.0
                self._qpos[:, 4:7] = 0.0
            else:
                self._qpos[indices, 0] = launch[indices, 0]
                self._qpos[indices, 1] = 0.0
                self._qpos[indices, 2] = launch[indices, 1]
                self._qpos[indices, 3] = 1.0
                self._qpos[indices, 4:7] = 0.0

    def _write_world_randomization(self, indices: torch.Tensor | None) -> None:
        """Write the sampled hidden physical parameters into the per-world model fields."""
        if self.task == "toss":
            dampratio = self._rand_episode["contact_dampratio"]
            friction = self._rand_episode["contact_friction"].clamp(min=_MIN_CONTACT_FRICTION)
            for geom in self._toss_contact_geoms():
                if indices is None:
                    self._model_solref[:, geom, 1] = dampratio
                    self._model_friction[:, geom, 0] = friction
                else:
                    self._model_solref[indices, geom, 1] = dampratio[indices]
                    self._model_friction[indices, geom, 0] = friction[indices]
            return
        damping = self._rand_episode["damping"]
        dof = self._pole_dof if self.task == "cartpole" else self._hinge_dof
        if indices is None:
            self._model_damping[:, dof] = damping
        else:
            self._model_damping[indices, dof] = damping[indices]

    def _write_task_state(self, indices: torch.Tensor | None) -> None:
        """Reset the task-specific episode trackers (and the toss-catch paddles/mocap poses)."""
        if self.task == "pendulum":
            initial = self._rand_episode["initial_value"]
            values = initial if indices is None else initial[indices]
            peak = values.abs()
            sign = torch.where(values >= 0.0, 1.0, -1.0)
            if indices is None:
                self._task_state["peak_abs_angle"].copy_(peak)
                self._task_state["last_angle_sign"].copy_(sign)
                self._task_state["completed_cycles"].zero_()
            else:
                self._task_state["peak_abs_angle"][indices] = peak
                self._task_state["last_angle_sign"][indices] = sign
                self._task_state["completed_cycles"][indices] = 0
        elif self.task == "toss":
            launch = self._rand_episode["launch_velocity"]
            if indices is None:
                self._passes.zero_()
                self._contact_side.zero_()
                self._last_side.fill_(-1)
                self._pass_cooldown.zero_()
                self._last_pass_step.fill_(-1)
                self._launch_velocity[:, 0] = launch[:, 0]
                self._launch_velocity[:, 1] = 0.0
                self._launch_velocity[:, 2] = launch[:, 1]
                self._release_step.copy_(self._rand_episode["release_delay_steps"])
                self._pad_pose.copy_(self._ready_pose.unsqueeze(0).expand(self.num_envs, -1, -1))
                self._pad_velocity.zero_()
                for side_index, side in enumerate(AGENT_KEYS):
                    self._write_mocap_pose(None, side_index, side)
            else:
                self._passes[indices] = 0
                self._contact_side[indices] = 0
                self._last_side[indices] = -1
                self._pass_cooldown[indices] = 0
                self._last_pass_step[indices] = -1
                self._launch_velocity[indices, 0] = launch[indices, 0]
                self._launch_velocity[indices, 1] = 0.0
                self._launch_velocity[indices, 2] = launch[indices, 1]
                self._release_step[indices] = self._rand_episode["release_delay_steps"][indices]
                self._pad_pose[indices] = self._ready_pose
                self._pad_velocity[indices] = 0.0
                for side_index, side in enumerate(AGENT_KEYS):
                    self._write_mocap_pose(indices, side_index, side)
            self._pad_start.copy_(self._pad_pose)
            self._pad_target.copy_(self._pad_pose)

    def _write_mocap_pose(self, indices: torch.Tensor | None, side_index: int, side: str) -> None:
        """Write one paddle's ready pose into the Warp `mocap_pos`/`mocap_quat` arrays."""
        mocap = self._pad_mocap[side]
        pose = self._ready_pose[side_index]
        half = pose[2] * 0.5
        if indices is None:
            self._mocap_pos[:, mocap, 0] = pose[0]
            self._mocap_pos[:, mocap, 1] = 0.0
            self._mocap_pos[:, mocap, 2] = pose[1]
            self._mocap_quat[:, mocap, 0] = torch.cos(half)
            self._mocap_quat[:, mocap, 1] = 0.0
            self._mocap_quat[:, mocap, 2] = torch.sin(half)
            self._mocap_quat[:, mocap, 3] = 0.0
            return
        self._mocap_pos[indices, mocap, 0] = pose[0]
        self._mocap_pos[indices, mocap, 1] = 0.0
        self._mocap_pos[indices, mocap, 2] = pose[1]
        self._mocap_quat[indices, mocap, 0] = torch.cos(half)
        self._mocap_quat[indices, mocap, 1] = 0.0
        self._mocap_quat[indices, mocap, 2] = torch.sin(half)
        self._mocap_quat[indices, mocap, 3] = 0.0

    def _reset_solver_state(self, indices: torch.Tensor | None) -> None:
        """Zero the solver/actuator state the CPU env's `mj_resetData` clears on reset."""
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
        """`(nworld, 3)` tensor `[latency_s, gain, limit_scale]` in the observation order."""
        return torch.stack(tuple(fields[name] for name in _PARAM_FIELDS), dim=1)

    def reset(self, seed: int | None = None) -> dict[str, torch.Tensor]:
        """Reset every world and return the first observation of the new episodes.

        Mirrors `VecCoopEnv.reset`: world `i` is reset with episode seed `seed + i` (the
        wrapper's base seed when `seed` is `None`), and `reset` with the same seed replays the
        same episodes.

        Args:
            seed: Base episode seed; `None` uses the constructor's seed.

        Returns:
            `{role: (nworld, obs_dim)}` first observation of each world.
        """
        base = self._base_seed if seed is None else int(seed)
        self._install_randomization([base + index for index in range(self.num_envs)], None)
        self._ready = True
        return self._build_observations()

    # --- observations -------------------------------------------------------------------
    def _build_observations(self) -> dict[str, torch.Tensor]:
        """Both agents' flat observation vectors for the CURRENT state and step counters."""
        cfg = self.config
        self._time_fraction[:, 0] = self._step.to(torch.float32) / cfg.max_control_steps
        shared = self._shared_state_tensor()
        goal = self._goal_tensor()
        per_agent = {}
        for role in AGENT_KEYS:
            per_agent[role] = {
                "action": self._last_applied[role],
                "command": self._last_command[role],
                "params": self._params[role],
                "privileged": self._rand_agents[role]["start_delay_s"].unsqueeze(1),
            }
        observations: dict[str, torch.Tensor] = {}
        for role in AGENT_KEYS:
            parts: list[torch.Tensor] = []
            for key, owner in self._obs_blocks[role]:
                if owner is None:
                    parts.append({"shared": shared, "goal": goal, "time": self._time_fraction}[key])
                else:
                    parts.append(per_agent[owner][key])
            observations[role] = torch.cat(parts, dim=1)
        return observations

    def _shared_state_tensor(self) -> torch.Tensor:
        """The task's shared-state block, in `spec.shared_fields` order."""
        if self.task == "cartpole":
            return torch.stack(
                (
                    self._qpos[:, self._cart_qpos],
                    self._qvel[:, self._cart_dof],
                    self._qpos[:, self._pole_qpos],
                    self._qvel[:, self._pole_dof],
                ),
                dim=1,
            )
        if self.task == "pendulum":
            return torch.stack(
                (self._qpos[:, self._hinge_qpos], self._qvel[:, self._hinge_dof]), dim=1
            )
        ball = torch.stack(
            (self._qpos[:, 0], self._qpos[:, 2], self._qvel[:, 0], self._qvel[:, 2]), dim=1
        )
        pads = torch.cat(
            (
                self._pad_pose[:, 0, :],
                self._pad_velocity[:, 0, :],
                self._pad_pose[:, 1, :],
                self._pad_velocity[:, 1, :],
            ),
            dim=1,
        )
        return torch.cat((ball, pads), dim=1)

    def _goal_tensor(self) -> torch.Tensor:
        """The task's goal block in `spec.goal_fields` order."""
        if self.task == "cartpole":
            self._goal[:, 0] = self.config.target_x
        elif self.task == "pendulum":
            self._goal[:, 0] = self.config.target_amplitude
            self._goal[:, 1] = float(self.config.hold_cycles)
        else:
            self._goal[:, 0] = self._passes.to(torch.float32)
            self._goal[:, 1] = float(self.config.n_passes)
        return self._goal

    # --- gym-like step ------------------------------------------------------------------
    def step(self, actions: dict[str, torch.Tensor | np.ndarray]) -> CoopGpuStep:
        """Advance all worlds by one control step (auto-resetting the finished ones).

        Args:
            actions: `{role: (nworld, action_dim)}` normalized commands, CUDA tensors or numpy
                arrays; clipped to `[-1, 1]` like the CPU env's action box.

        Returns:
            The `CoopGpuStep` with the next-step observations, reward, flags and episode stats.
        """
        if not self._ready:
            raise RuntimeError("call reset() before step()")
        cfg = self.config
        commands = self._parse_actions(actions)
        self._last_command = commands
        applied: dict[str, torch.Tensor] = {}
        for role in AGENT_KEYS:
            rand = self._rand_agents[role]
            buffer = self._delay[role]
            write_position = self._write_pos
            buffer[:, write_position, :] = commands[role]
            gather_index = ((write_position - rand["latency_steps"]) % self._delay_size).view(
                -1, 1, 1
            )
            delayed = torch.gather(buffer, 1, gather_index.expand(-1, 1, self.action_dim)).squeeze(
                1
            )
            applied[role] = torch.where(
                (self._step < rand["start_delay_steps"]).unsqueeze(1),
                torch.zeros_like(delayed),
                delayed,
            )
        self._last_applied = applied
        self._write_pos = (self._write_pos + 1) % self._delay_size

        self._xfrc.zero_()
        passes_before: torch.Tensor | None = None
        if self.task == "toss":
            # The pass gate's cooldown is counted in control steps: it is armed to
            # `min_pass_interval_steps` by the captured kernel on every counted pass and aged
            # here once per control step (0 stays 0, so a fresh episode can pass immediately).
            self._pass_cooldown.sub_(1).clamp_(min=0)
            passes_before = self._passes.clone()
            self._prepare_toss_step(applied)
        else:
            physical = {
                role: self._signs[role]
                * torch.clamp(applied[role][:, 0], min=0.0)
                * self._authority[role]
                * self._rand_agents[role]["limit_scale"]
                * self._rand_agents[role]["gain"]
                for role in AGENT_KEYS
            }
            wind = torch.where(
                self._rand_episode["wind_active"] & (self._step == self._rand_episode["wind_step"]),
                self._rand_episode["wind_force"],
                self._zero,
            )
            total = physical[AGENT_KEYS[0]] + physical[AGENT_KEYS[1]] + wind
            if self.task == "cartpole":
                self._xfrc[:, self._cart_body, 0] = total
            else:
                self._xfrc[:, self._bob_body, 4] = total
        self._run_control_step()
        self._step += 1
        # Numerical safety net: the float32 Warp contact solver can (rarely, in the toss-catch
        # task) produce a non-finite state from a deep/high-speed ball/paddle impact where
        # float64 MuJoCo stays finite. Such a world is treated as a FAILED episode and
        # auto-reset at the end of this step instead of poisoning the policy input with NaN.
        invalid = ~(torch.isfinite(self._qpos).all(dim=1) & torch.isfinite(self._qvel).all(dim=1))
        if bool(invalid.any()):
            self._physics_failures += int(invalid.sum())
            self._qpos[invalid] = 0.0
            self._qvel[invalid] = 0.0
        if self.task == "toss":
            self._finish_toss_step()
            if passes_before is not None:
                # At most one pass can be counted per control step (the cooldown gate), so
                # the increment identifies the worlds whose pass just happened; `self._step`
                # was already incremented, hence the `- 1` for the executed 0-based index.
                counted = self._passes > passes_before
                self._last_pass_step[counted] = self._step[counted] - 1
        elif self.task == "pendulum":
            self._update_pendulum_tracker()
        observations = self._build_observations()
        succeeded, failed, potential_now = self._task_reward()
        if bool(invalid.any()):
            succeeded = succeeded & ~invalid
            failed = failed | invalid
            potential_now = torch.where(invalid, self._zero, potential_now)
        # Failure has priority over success (the CPU `CoopTaskEnv.step` rule): a failure
        # step pays `failure_penalty` and reports success=False, never the success bonus.
        succeeded = succeeded & ~failed
        reward = torch.where(
            succeeded,
            self._reward_success,
            torch.where(failed, self._reward_failure, self._reward_step),
        )
        non_terminal = ~(succeeded | failed)
        reward = reward + torch.where(
            non_terminal, potential_now - self._prev_potential, self._zero
        )
        self._prev_potential = potential_now
        self._episode_return += reward.to(torch.float64)
        self._episode_length += 1
        terminated = succeeded | failed
        truncated = non_terminal & (self._step >= cfg.max_control_steps)

        final_obs = observations
        done = terminated | truncated
        done_rows = done.nonzero(as_tuple=False).flatten()
        finished_returns: list[float] = []
        finished_lengths: list[int] = []
        finished_successes: list[bool] = []
        finished_angles: list[float] = []
        finished_infos: list[dict] = []
        if int(done_rows.numel()) > 0:
            (
                finished_returns,
                finished_lengths,
                finished_successes,
                finished_angles,
                finished_infos,
            ) = self._finished_statistics(done_rows, succeeded)
            self._reset_worlds(done_rows)
        observations = self._build_observations()
        return CoopGpuStep(
            obs=observations,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            final_obs=final_obs,
            finished_returns=finished_returns,
            finished_lengths=finished_lengths,
            finished_successes=finished_successes,
            finished_angles=finished_angles,
            finished_infos=finished_infos,
            nonfinite=invalid,
        )

    def _parse_actions(
        self, actions: dict[str, torch.Tensor | np.ndarray]
    ) -> dict[str, torch.Tensor]:
        """Validate the action dict and return clipped float32 CUDA tensors."""
        if not isinstance(actions, dict):
            raise TypeError(f"actions must be a dict with keys {AGENT_KEYS}, got {type(actions)}")
        if set(actions) != set(AGENT_KEYS):
            raise ValueError(f"action keys must be exactly {AGENT_KEYS}, got {sorted(actions)}")
        expected = (self.num_envs, self.action_dim)
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

    # --- task-specific step mechanics ---------------------------------------------------
    def _prepare_toss_step(self, applied: dict[str, torch.Tensor]) -> None:
        """Release the hidden throw impulse and write the paddles' per-substep pose banks."""
        cfg = self.config
        ball_dof = self._ball_dof
        release = self._step == self._release_step
        if bool(release.any()):
            current = self._qvel[:, ball_dof : ball_dof + 3].clone()
            self._qvel[:, ball_dof : ball_dof + 3] = torch.where(
                release.unsqueeze(1), self._launch_velocity, current
            )
        start = self._pad_pose
        max_step = cfg.pad_speed * cfg.control_dt
        max_dtilt = cfg.pad_tilt_rate * cfg.control_dt
        target = torch.empty_like(start)
        x_ranges = (
            (-cfg.pad_outer_x, -cfg.pad_inner_x),
            (cfg.pad_inner_x, cfg.pad_outer_x),
        )
        for side_index, role in enumerate(AGENT_KEYS):
            rand = self._rand_agents[role]
            authority = rand["gain"] * rand["limit_scale"]
            command = applied[role]
            low_x, high_x = x_ranges[side_index]
            target[:, side_index, 0] = torch.clamp(
                start[:, side_index, 0] + command[:, 0] * max_step * authority, low_x, high_x
            )
            target[:, side_index, 1] = torch.clamp(
                start[:, side_index, 1] + command[:, 1] * max_step * authority,
                cfg.pad_min_z,
                cfg.pad_max_z,
            )
            target[:, side_index, 2] = torch.clamp(
                start[:, side_index, 2] + command[:, 2] * max_dtilt * authority,
                -cfg.pad_tilt_limit,
                cfg.pad_tilt_limit,
            )
        self._pad_start.copy_(start)
        self._pad_target.copy_(target)
        self._write_pose_banks()

    def _write_pose_banks(self) -> None:
        """Interpolate the paddles' poses over the substeps into the captured-graph banks."""
        pads = self._pad_start
        poses = pads.unsqueeze(0) + self._alpha.unsqueeze(-1) * (
            self._pad_target.unsqueeze(0) - pads.unsqueeze(0)
        )
        half = poses[:, :, :, 2] * 0.5
        quats = torch.stack(
            (
                torch.cos(half),
                torch.zeros_like(half),
                torch.sin(half),
                torch.zeros_like(half),
            ),
            dim=-1,
        )
        self._pose_bank.copy_(poses)
        self._quat_bank.copy_(quats)

    def _finish_toss_step(self) -> None:
        """Update the paddles' reported pose/velocity after the step (CPU `_after_step`)."""
        self._pad_pose.copy_(self._pad_target)
        self._pad_velocity.copy_((self._pad_target - self._pad_start) / self.config.control_dt)

    def _update_pendulum_tracker(self) -> None:
        """The CPU `PendulumEnv._after_step` amplitude / zero-crossing tracker, batched."""
        cfg = self.config
        angle = self._qpos[:, self._hinge_qpos]
        peak = torch.maximum(self._task_state["peak_abs_angle"], angle.abs())
        sign = torch.where(angle >= 0.0, 1.0, -1.0)
        crossed = (angle != 0.0) & (sign != self._task_state["last_angle_sign"])
        completed = self._task_state["completed_cycles"] + (
            crossed & (peak >= cfg.target_amplitude)
        ).to(torch.int64)
        self._task_state["peak_abs_angle"] = torch.where(crossed, angle.abs(), peak)
        self._task_state["last_angle_sign"] = torch.where(
            crossed, sign, self._task_state["last_angle_sign"]
        )
        self._task_state["completed_cycles"] = completed

    def _task_reward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(succeeded, failed, potential_now)` of the current state (the CPU rules)."""
        cfg = self.config
        if self.task == "cartpole":
            cart_x = self._qpos[:, self._cart_qpos]
            cart_vx = self._qvel[:, self._cart_dof]
            angle = self._qpos[:, self._pole_qpos]
            angvel = self._qvel[:, self._pole_dof]
            succeeded = (
                ((cart_x - cfg.target_x).abs() <= cfg.success_pos_tol)
                & (angle.abs() <= cfg.success_angle_tol)
                & (cart_vx.abs() <= cfg.success_lin_speed_tol)
                & (angvel.abs() <= cfg.success_ang_speed_tol)
            )
            failed = angle.abs() > cfg.failure_angle
            pos_scale = max(abs(cfg.target_x), 1e-6)
            pos_excess = ((cart_x - cfg.target_x).abs() / pos_scale).clamp(max=1.0)
            angle_excess = (angle.abs() / cfg.failure_angle).clamp(max=1.0)
            potential = -cfg.shaping_pos_coef * pos_excess - cfg.shaping_angle_coef * angle_excess
            return succeeded, failed, potential
        if self.task == "pendulum":
            angle = self._qpos[:, self._hinge_qpos]
            peak = self._task_state["peak_abs_angle"]
            succeeded = self._task_state["completed_cycles"] >= cfg.hold_cycles
            failed = angle.abs() > cfg.failure_angle
            potential = cfg.shaping_amplitude_coef * (peak / cfg.target_amplitude).clamp(
                min=0.0, max=1.0
            )
            return succeeded, failed, potential
        ball_x = self._qpos[:, 0]
        ball_y = self._qpos[:, 1]
        ball_z = self._qpos[:, 2]
        failed = (
            (ball_z < cfg.floor_z) | (ball_x.abs() > cfg.court_x) | (ball_y.abs() > cfg.court_y)
        )
        # The CPU `logic.toss_success` conjunction: enough passes AND the ball still in play
        # above `floor_z + success_floor_margin` (and not a floor/court failure), so an
        # episode cannot be won on the step that drops the ball out of the court.
        succeeded = (
            (self._passes >= cfg.n_passes)
            & (ball_z > cfg.floor_z + cfg.success_floor_margin)
            & ~failed
        )
        progress = (self._passes.to(torch.float32) / cfg.n_passes).clamp(min=0.0, max=1.0)
        height = (ball_z / cfg.shaping_height_scale).clamp(min=0.0, max=1.0)
        potential = cfg.shaping_pass_coef * progress + cfg.shaping_height_coef * height
        return succeeded, failed, potential

    # --- finished-episode statistics / auto-reset ---------------------------------------
    def _finished_statistics(
        self, rows: torch.Tensor, succeeded: torch.Tensor
    ) -> tuple[list[float], list[int], list[bool], list[float], list[dict]]:
        """Gather the finished worlds' episode statistics and the CPU-format terminal infos."""
        columns = torch.stack(
            (
                self._episode_return[rows],
                self._episode_length[rows].to(torch.float64),
                succeeded[rows].to(torch.float64),
                self._task_diagnostic(rows),
            ),
            dim=1,
        )
        rows_list = columns.cpu().tolist()
        infos = self._finished_infos(rows, succeeded)
        finished_returns: list[float] = []
        finished_lengths: list[int] = []
        finished_successes: list[bool] = []
        finished_angles: list[float] = []
        for index, row in enumerate(rows_list):
            episode_return, length, success, angle = row
            finished_returns.append(float(episode_return))
            finished_lengths.append(int(length))
            finished_successes.append(bool(success))
            finished_angles.append(float(angle))
        return finished_returns, finished_lengths, finished_successes, finished_angles, infos

    def _task_diagnostic(self, rows: torch.Tensor) -> torch.Tensor:
        """The task's primary diagnostic of `rows` in the CPU `task_diagnostic` units."""
        if self.task == "cartpole":
            return torch.rad2deg(self._qpos[rows, self._pole_qpos].to(torch.float64))
        if self.task == "pendulum":
            return torch.rad2deg(self._task_state["peak_abs_angle"][rows].to(torch.float64))
        return self._passes[rows].to(torch.float64)

    def _finished_infos(self, rows: torch.Tensor, succeeded: torch.Tensor) -> list[dict]:
        """The CPU env's terminal `info` dicts (success + diagnostics) of `rows`."""
        cfg = self.config
        step_time = self._step[rows].to(torch.float64) * cfg.control_dt
        success = succeeded[rows].cpu().tolist()
        infos: list[dict] = []
        if self.task == "cartpole":
            values = torch.stack(
                (
                    torch.rad2deg(self._qpos[rows, self._pole_qpos].to(torch.float64)),
                    self._qpos[rows, self._cart_qpos].to(torch.float64),
                    self._qvel[rows, self._cart_dof].to(torch.float64),
                    self._qpos[rows, self._pole_qpos].to(torch.float64),
                    self._qvel[rows, self._pole_dof].to(torch.float64),
                    (self._qpos[rows, self._cart_qpos] - cfg.target_x).abs().to(torch.float64),
                    step_time,
                ),
                dim=1,
            )
            keys = (
                "angle_deg",
                "cart_x",
                "cart_vx",
                "pole_angle",
                "pole_angvel",
                "distance_to_goal",
                "time",
            )
        elif self.task == "pendulum":
            peak = self._task_state["peak_abs_angle"][rows]
            values = torch.stack(
                (
                    torch.rad2deg(self._qpos[rows, self._hinge_qpos].abs().to(torch.float64)),
                    torch.rad2deg(peak.to(torch.float64)),
                    self._task_state["completed_cycles"][rows].to(torch.float64),
                    torch.full_like(step_time, float(cfg.hold_cycles)),
                    step_time,
                ),
                dim=1,
            )
            keys = ("angle_deg", "amplitude_deg", "completed_cycles", "hold_cycles", "time")
        else:
            ball_vx = self._qvel[rows, 0].to(torch.float64)
            ball_vz = self._qvel[rows, 2].to(torch.float64)
            values = torch.stack(
                (
                    self._passes[rows].to(torch.float64),
                    torch.full_like(step_time, float(cfg.n_passes)),
                    self._last_pass_step[rows].to(torch.float64),
                    torch.full_like(step_time, float(cfg.min_pass_interval_steps)),
                    self._qpos[rows, 0].to(torch.float64),
                    self._qpos[rows, 2].to(torch.float64),
                    torch.sqrt(ball_vx**2 + ball_vz**2),
                    step_time,
                ),
                dim=1,
            )
            keys = (
                "passes",
                "target_passes",
                "last_pass_step",
                "min_pass_interval_steps",
                "ball_x",
                "ball_z",
                "ball_speed",
                "time",
            )
        for position, row_values in enumerate(values.cpu().tolist()):
            info = {"success": bool(success[position])}
            info.update(zip(keys, (float(value) for value in row_values)))
            infos.append(info)
        return infos

    def _reset_worlds(self, indices: torch.Tensor) -> None:
        """Auto-reset the finished worlds with fresh seeds from the wrapper's seed stream."""
        count = int(indices.numel())
        seeds = list(range(self._next_seed, self._next_seed + count))
        self._next_seed += count
        self._install_randomization(seeds, indices)

    # --- inspection ---------------------------------------------------------------------
    def state(self) -> dict[str, torch.Tensor]:
        """The raw `qpos (nworld, nq)` / `qvel (nworld, nv)` CUDA views of the current state.

        Zero-copy views (the parity script compares them against the CPU env's `_data`). Do
        not keep them across a `step` if you need a snapshot -- use `.clone()`.
        """
        return {"qpos": self._qpos, "qvel": self._qvel}

    def randomization_tensors(self) -> dict[str, dict[str, torch.Tensor]]:
        """The per-agent sampled randomisation tensors of the current episodes (read-only)."""
        return {role: dict(self._rand_agents[role]) for role in AGENT_KEYS}

    def episode_randomization_tensors(self) -> dict[str, torch.Tensor]:
        """The episode-level randomisation tensors (initial state, damping/contact parameters)."""
        return dict(self._rand_episode)

    def synchronize(self) -> None:
        """Block until the queued Warp kernels of the current device have finished."""
        self._warp.synchronize()

    def close(self) -> None:
        """Release nothing (kept for symmetry with `VecCoopEnv.close`)."""
