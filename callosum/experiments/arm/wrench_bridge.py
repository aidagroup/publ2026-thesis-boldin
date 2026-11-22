"""Intent->skills bridge: the learned `CubeWrench-v0` wrench policy -> `FaceTurnSkills-v0`.

The reduced wrench-level cube and the skill-level SO-ARM101 environment share neither the
observation nor the action space, so **no weight transfer is possible and none is attempted**.
This module makes the bridge between the two lines explicit and measurable instead:

1. **Reduced rollout.** A greedy (reduced) learned policy from a `wrench_ppo` checkpoint is
   rolled out in the GPU `CubeWrench-v0` backend (`mujoco_warp`, one world, fixed seed) and
   every control step is recorded: both agents' APPLIED normalized wrenches converted to
   physical forces/torques, the layer angle, the core drift and the episode flags. Only the
   public observation contract (`callosum.experiments.cube_wrench.logic.obs_fields`) is read,
   so the recorder works for every `input_mode` (`none`/`oracle`/`predicted`).
2. **Phase segmentation.** The trace is classified into `idle` / `hold` / `turn` /
   `release` phases: `hold` = the holder applies a stabilising wrench (nonzero force or
   torque, small core drift) while the layer does not turn; `turn` = the rotator's
   `torque_z` dominates and the layer angle rises.
3. **Primitive schedule.** Each phase maps to a per-role arm primitive
   (holder: `approach_body`/`grasp`/`wait`/`release`; rotator:
   `approach_face`/`grasp`/`turn90`/`release`) and the phase timeline is expressed on the
   arm's DISCRETE decision grid (one `FaceTurnSkills-v0` decision = `decision_period`
   substeps of `STEP_SECONDS` each). Two allocation modes are reported: `dt` (naively
   faithful to the reduced physical time) and `share` (phase durations as shares of a
   configured arm decision budget).
4. **Timing measurement.** `compare_phase_timing` scores how well the arm execution
   reproduces the reduced phase structure: phase-share error, turn-onset offset, the
   decision/control-step time-scale ratio and the success flags.

HONESTY NOTE -- this is an intent->skills bridge, NOT policy transfer. What carries over is
the phase ORDER and the per-role intent; the arm keeps its own primitive durations. The
reduced policy's absolute timing is not transferable: one arm decision spans
`ARM_DECISION_SECONDS / control_dt` ~ 12.5 reduced control steps, while a single arm
primitive (an approach leg) already needs tens of decisions.

Everything above the rollout is pure (stdlib + numpy, importable without torch and without
the simulator), so the classification/schedule/metrics are unit-tested on macOS; the GPU
imports are lazy and `torch` is optional at import time.
"""

from __future__ import annotations

import contextlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:  # `torch` is only needed for the reduced rollout (the `train` extra).
    import torch
except ImportError:  # pragma: no cover - pure classification/schedule paths need no torch
    torch = None  # type: ignore[assignment]

from callosum.experiments.cube_wrench.logic import AGENT_KEYS, obs_fields

# --- timing of the two sides ---------------------------------------------------------------

ARM_DECISION_PERIOD = 10
"""Substeps per `FaceTurnSkills-v0` decision (`FaceTurnSkillsConfig.decision_period`)."""

ARM_SUBSTEP_SECONDS = 0.05
"""One arm control substep (s), `skills_runtime.STEP_SECONDS`."""

ARM_DECISION_SECONDS = ARM_DECISION_PERIOD * ARM_SUBSTEP_SECONDS
"""Wall-clock length of one arm decision (0.5 s), the arm side's discrete time unit."""

# --- phase-classification thresholds (physical units) --------------------------------------
HOLDER_FORCE_ON_N = 0.05
"""Holder force (N) above which the holder counts as applying a wrench. `holder_max_force`
is 0.5 N, so this is 10% of the authority."""

HOLDER_TORQUE_ON_NM = 5.0e-4
"""Holder torque (Nm) above which the holder counts as applying a wrench (10% of the
5e-3 Nm authority)."""

ROTATOR_TAU_ON_NM = 5.0e-4
"""Rotator `torque_z` (Nm) above which the layer counts as being driven (25% of the
2e-3 Nm authority)."""

TURN_RISE_DEG = 0.3
"""Layer rise (deg per reduced control step) above which the layer counts as turning."""

MIN_SEGMENT_STEPS = 2
"""Raw phase runs shorter than this (control steps) are merged into the neighbouring
phase: a one-step wrench flicker must not split a hold into three segments."""

PHASES = ("idle", "hold", "turn", "release")
"""The bridge's phase labels, in the order they appear inside an episode."""

PHASE_INTENTS = {
    # Entry primitive = the first decision of the segment; hold primitive = the state the
    # segment settles into (re-selecting a primitive continues it, so re-selecting is the
    # skill level's "keep executing this" signal).
    "idle": {
        "holder": ("approach_body", "approach_body"),
        "rotator": ("approach_face", "approach_face"),
    },
    "hold": {"holder": ("grasp", "wait"), "rotator": ("grasp", "wait")},
    "turn": {"holder": ("wait", "wait"), "rotator": ("turn90", "turn90")},
    "release": {"holder": ("release", "retreat"), "rotator": ("release", "retreat")},
}
"""Per phase `(entry primitive, held primitive)` for each role."""

ARM_TURN_PRIMITIVES = ("turn90", "turn180")
"""Rotator primitives that drive the layer (the arm side's turn phase)."""

ARM_RELEASE_PRIMITIVES = ("release", "retreat")
"""Primitives of the terminating release phase."""

ARM_TURN_RISE_DEG = 0.5
"""Arm-side layer rise (deg per decision) above which the arm is in its turn phase."""


# --- the reduced rollout -------------------------------------------------------------------


@dataclass
class ReducedStep:
    """One reduced control step's bridge signals (SI units unless the name says otherwise).

    The wrenches are the APPLIED (post-latency, post-injection) normalized wrenches scaled
    exactly like the env scales them for the physics (`authority * limit_scale * gain`), so
    they are the physical forces/torques the reduced policy put on the cube.
    """

    t: int
    time_s: float
    layer_angle_rad: float
    layer_ang_vel: float
    core_drift_pos_m: float
    core_drift_rot_deg: float
    holder_force: tuple[float, float, float]
    holder_torque: tuple[float, float, float]
    rotator_force: tuple[float, float, float]
    rotator_torque: tuple[float, float, float]
    core_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    core_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    holder_command: tuple[float, ...] = ()
    rotator_command: tuple[float, ...] = ()
    angle_rise_deg: float = 0.0
    phase: str = ""
    intended_holder: str = ""
    intended_rotator: str = ""

    @property
    def layer_angle_deg(self) -> float:
        """Layer angle in degrees."""
        return math.degrees(self.layer_angle_rad)

    @property
    def holder_force_norm_n(self) -> float:
        """Holder force magnitude (N)."""
        return _norm(self.holder_force)

    @property
    def holder_torque_norm_nm(self) -> float:
        """Holder torque magnitude (Nm)."""
        return _norm(self.holder_torque)

    @property
    def rotator_tau_z_nm(self) -> float:
        """Rotator `torque_z` (Nm) -- the hinge-axis drive of the turn phase."""
        return float(self.rotator_torque[2])

    @property
    def rotator_force_norm_n(self) -> float:
        """Rotator force magnitude (N)."""
        return _norm(self.rotator_force)

    def signals(self) -> PhaseSignals:
        """The classifier's input view of this step."""
        return PhaseSignals(
            holder_force_norm_n=self.holder_force_norm_n,
            holder_torque_norm_nm=self.holder_torque_norm_nm,
            rotator_tau_z_nm=self.rotator_tau_z_nm,
            angle_rise_deg=self.angle_rise_deg,
        )

    def to_json(self) -> dict:
        """The JSON form of one trace step (the schema the task report asks for)."""
        return {
            "phase": self.phase,
            "t": self.t,
            "time_s": self.time_s,
            "holder_force": list(self.holder_force),
            "holder_torque": list(self.holder_torque),
            "holder_force_norm_n": self.holder_force_norm_n,
            "holder_torque_norm_nm": self.holder_torque_norm_nm,
            "rotator_tau_z": self.rotator_tau_z_nm,
            "rotator_force_norm_n": self.rotator_force_norm_n,
            "rotator_torque": list(self.rotator_torque),
            "layer_angle": self.layer_angle_rad,
            "layer_angle_deg": self.layer_angle_deg,
            "layer_ang_vel": self.layer_ang_vel,
            "angle_rise_deg": self.angle_rise_deg,
            "core_drift_pos_m": self.core_drift_pos_m,
            "core_drift_rot_deg": self.core_drift_rot_deg,
            "core_pos": list(self.core_pos),
            "core_quat_wxyz": list(self.core_quat),
            "intended_primitive": {
                "holder": self.intended_holder,
                "rotator": self.intended_rotator,
            },
        }


@dataclass
class PhaseSegment:
    """A contiguous run of one phase in the reduced trace."""

    phase: str
    t_start: int
    t_end: int
    duration_s: float

    @property
    def steps(self) -> int:
        """Number of reduced control steps in the segment."""
        return self.t_end - self.t_start + 1

    def to_json(self) -> dict:
        """The JSON form of the segment."""
        return {
            "phase": self.phase,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "steps": self.steps,
            "duration_s": self.duration_s,
        }


@dataclass
class ReducedRollout:
    """One greedy episode of the learned reduced policy plus its bridge segmentation."""

    checkpoint: str
    input_mode: str
    env_mode: str
    seed: int
    device: str
    backend: str
    control_dt_s: float
    steps: list[ReducedStep] = field(default_factory=list)
    phase_segments: list[PhaseSegment] = field(default_factory=list)
    success: bool = False
    terminal: str = ""
    final_angle_deg: float = 0.0
    final_drift_pos_m: float = 0.0
    final_drift_rot_deg: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        """Simulated duration of the episode (s)."""
        return len(self.steps) * self.control_dt_s

    @property
    def segments(self) -> list[PhaseSegment]:
        """The classified phase segments (filled by `classify_rollout`)."""
        return self.phase_segments

    def to_json(self, include_trace: bool = True) -> dict:
        """JSON-ready reduced block of the bridge report."""
        payload = {
            "checkpoint": self.checkpoint,
            "input_mode": self.input_mode,
            "env_partner_mode": self.env_mode,
            "seed": self.seed,
            "backend": self.backend,
            "device": self.device,
            "control_dt_s": self.control_dt_s,
            "steps": len(self.steps),
            "duration_s": self.duration_s,
            "success": self.success,
            "terminal": self.terminal,
            "final_angle_deg": self.final_angle_deg,
            "final_drift_pos_m": self.final_drift_pos_m,
            "final_drift_rot_deg": self.final_drift_rot_deg,
            "phase_segments": [segment.to_json() for segment in self.phase_segments],
            "meta": dict(self.meta),
        }
        if include_trace:
            payload["trace"] = [step.to_json() for step in self.steps]
        return payload


@dataclass(frozen=True)
class ReducedPolicy:
    """A loaded greedy reduced policy plus what the rollout needs to build its input."""

    policies: list
    builder: object | None
    env_mode: str
    input_mode: str
    env_config: object
    latent_dim: int
    checkpoint: str


def _norm(vector) -> float:
    """Euclidean norm of a 3-vector as a python float."""
    return float(math.sqrt(sum(float(value) ** 2 for value in vector)))


def _authority(config, role: str) -> tuple[float, ...]:
    """The role's wrench authority (force xyz, torque xyz) from the env config."""
    if role == "holder":
        return tuple(config.holder_authority)
    return tuple(config.rotator_authority)


def _physical_wrench(obs: np.ndarray, mode: str, role: str, config) -> tuple[tuple, tuple]:
    """`(force, torque)` the agent APPLIED at this step, in N / Nm.

    The observation reports the applied NORMALIZED wrench (`own_wrench_*`); the env scales
    it by `authority * limit_scale * gain` before writing `xfrc_applied`, and the agent's own
    `gain`/`limit_scale` are in its own block, so this recovers the physical wrench without
    touching the simulator's internals.
    """
    fields = obs_fields(mode)
    axes = ("fx", "fy", "fz", "tx", "ty", "tz")
    applied = np.asarray(
        [obs[fields.index(f"own_wrench_{axis}")] for axis in axes], dtype=np.float64
    )
    gain = obs[fields.index("own_gain")]
    limit_scale = obs[fields.index("own_limit_scale")]
    scaled = applied * np.asarray(_authority(config, role)) * float(gain) * float(limit_scale)
    return tuple(float(value) for value in scaled[:3]), tuple(float(value) for value in scaled[3:])


def _core_state(obs: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """The shared core state from an observation vector: `(core_pos, core_quat)`, wxyz."""
    fields = obs_fields(mode)
    pos = np.asarray([obs[fields.index(f"core_{axis}")] for axis in ("x", "y", "z")])
    quat = np.asarray([obs[fields.index(f"core_q{axis}")] for axis in ("w", "x", "y", "z")])
    return pos, quat


def _drift(obs: np.ndarray, mode: str, initial_pos: np.ndarray, initial_quat: np.ndarray) -> tuple:
    """`(position drift m, rotation drift deg)` of the core from the episode's start."""
    pos, quat = _core_state(obs, mode)
    drift_pos = float(np.linalg.norm(pos - initial_pos))
    dot = float(np.clip(abs(float(np.dot(quat, initial_quat))), 0.0, 1.0))
    return drift_pos, math.degrees(2.0 * math.acos(dot))


# --- policy loading and the GPU rollout ----------------------------------------------------


def load_reduced_policy(
    checkpoint: str | Path,
    input_mode: str | None = None,
    device: str = "cuda",
) -> ReducedPolicy:
    """Load a `wrench_ppo` checkpoint as a greedy reduced policy (weights + Bi-JEPA).

    Args:
        checkpoint: Path to a `latest.pt` of `callosum.experiments.cube_wrench.train`.
        input_mode: One of `none`/`oracle`/`predicted`; `None` reads it from the checkpoint.
        device: Torch device of the policies (the rollout's GPU device).

    Returns:
        The loaded `ReducedPolicy`.
    """
    if torch is None:  # pragma: no cover - only reachable without the `train` extra
        raise ImportError("load_reduced_policy needs torch; install the `train` extra")
    from callosum.experiments.cube_wrench.eval import resolve_input_mode
    from callosum.experiments.cube_wrench.train import (
        LatentInputBuilder,
        check_checkpoint_compat,
        config_from_payload,
        load_bijepa_weights,
        load_checkpoint,
        load_weights,
        make_bijepa,
        make_policies,
    )

    payload = load_checkpoint(checkpoint)
    args = SimpleNamespace(checkpoint=str(checkpoint), input_mode=input_mode, partner_mode=None)
    mode = resolve_input_mode(payload, args)
    env_mode, latent_dim = check_checkpoint_compat(payload, mode)
    cfg = config_from_payload(payload)
    policies = make_policies(env_mode, device, latent_dim=latent_dim)
    load_weights(payload, policies)
    for policy in policies:
        policy.eval()
    builder = None
    bijepa = make_bijepa(cfg, device)
    if bijepa is not None:
        load_bijepa_weights(payload, bijepa)
        bijepa.eval()
        builder = LatentInputBuilder(bijepa, device=device)
    return ReducedPolicy(
        policies=policies,
        builder=builder,
        env_mode=env_mode,
        input_mode=mode,
        env_config=cfg.env_config(),
        latent_dim=latent_dim,
        checkpoint=str(checkpoint),
    )


@dataclass
class RolloutConfig:
    """Settings of the greedy reduced rollout."""

    seed: int = 0
    """Episode seed (one world, so the episode is exactly this seed's randomisation)."""

    device: str = "cuda"
    """CUDA device of the `mujoco_warp` backend."""

    max_steps: int | None = None
    """Hard cap on control steps; `None` uses the env config's `max_control_steps`."""

    use_cuda_graph: bool = True
    """Capture the physics substep loop into a CUDA graph (the backend's fast path)."""


def _no_grad():
    """`torch.no_grad()` when torch is installed, else a no-op context (pure paths)."""
    if torch is None:  # pragma: no cover - only the pure paths run without torch
        return contextlib.nullcontext()
    return torch.no_grad()


def rollout_reduced(
    policy: ReducedPolicy,
    config: RolloutConfig | None = None,
) -> ReducedRollout:
    """Run one greedy episode of the reduced policy on the GPU backend and record it.

    One `mujoco_warp` world is reset with `config.seed`, the two greedy policies act on
    their observations (or on the Bi-JEPA input for `predicted`), and every control step is
    recorded from the END-of-step observation (`GpuStep.final_obs`), so the recorded applied
    wrench is the one that produced the recorded state.

    Args:
        policy: A loaded `ReducedPolicy` (see `load_reduced_policy`).
        config: Rollout settings; defaults to `RolloutConfig()`.

    Returns:
        The recorded `ReducedRollout` with the phase classification filled in.
    """
    if torch is None:  # pragma: no cover - only reachable without the `train` extra
        raise ImportError("rollout_reduced needs torch; install the `train` extra")
    from callosum.experiments.cube_wrench.env_gpu import WrenchCubeBatchGPU

    cfg = config if config is not None else RolloutConfig()
    env_config = policy.env_config
    env = WrenchCubeBatchGPU(
        1,
        config=env_config,
        partner_mode=policy.env_mode,
        seed=cfg.seed,
        device=cfg.device,
        use_cuda_graph=cfg.use_cuda_graph,
    )
    max_steps = int(env_config.max_control_steps if cfg.max_steps is None else cfg.max_steps)
    steps: list[ReducedStep] = []
    previous_angle = None
    initial_pos = initial_quat = None
    success = False
    terminal = "truncated"
    final_angle = final_drift_pos = final_drift_rot = 0.0
    try:
        with _no_grad():
            obs = env.reset(seed=cfg.seed)
            inputs = policy.builder.reset(obs) if policy.builder is not None else obs
            for step in range(max_steps):
                actions = {}
                for name, net in zip(AGENT_KEYS, policy.policies):
                    action = net.get_action(inputs[name], deterministic=True)
                    actions[name] = torch.clamp(action, -1.0, 1.0)
                result = env.step(actions)
                final_obs = result.final_obs
                holder_obs = final_obs["holder"][0].detach().cpu().numpy().astype(np.float64)
                rotator_obs = final_obs["rotator"][0].detach().cpu().numpy().astype(np.float64)
                if initial_pos is None:
                    initial_pos, initial_quat = _core_state(holder_obs, policy.env_mode)
                recorded = _record_step(
                    step=step,
                    holder_obs=holder_obs,
                    rotator_obs=rotator_obs,
                    actions=actions,
                    config=env_config,
                    mode=policy.env_mode,
                    previous_angle=previous_angle,
                    initial_pos=initial_pos,
                    initial_quat=initial_quat,
                )
                previous_angle = recorded.layer_angle_rad
                steps.append(recorded)
                done = bool(result.terminated[0] or result.truncated[0])
                if done:
                    success = bool(result.terminated[0]) and bool(
                        result.finished_successes[0] if result.finished_successes else False
                    )
                    if result.truncated[0]:
                        terminal = "truncated"
                    elif success:
                        terminal = "success"
                    else:
                        terminal = "failure"
                    if result.finished_angles:
                        final_angle = float(result.finished_angles[0])
                        final_drift_pos = float(result.finished_drift_pos[0])
                        final_drift_rot = float(result.finished_drift_rot[0])
                    else:  # pragma: no cover - the GPU env always reports the stats
                        final_angle = recorded.layer_angle_deg
                        final_drift_pos = recorded.core_drift_pos_m
                        final_drift_rot = recorded.core_drift_rot_deg
                    break
                if policy.builder is not None:
                    done_tensor = torch.zeros(1, device=cfg.device)
                    inputs = policy.builder.step(result.obs, done_tensor)
                else:
                    inputs = result.obs
        rollout = ReducedRollout(
            checkpoint=policy.checkpoint,
            input_mode=policy.input_mode,
            env_mode=policy.env_mode,
            seed=cfg.seed,
            device=cfg.device,
            backend="warp",
            control_dt_s=float(env_config.control_dt),
            steps=steps,
            success=success,
            terminal=terminal,
            final_angle_deg=final_angle,
            final_drift_pos_m=final_drift_pos,
            final_drift_rot_deg=final_drift_rot,
            meta={
                "goal_angle_deg": math.degrees(float(env_config.goal_angle)),
                "success_angle_tol_deg": math.degrees(float(env_config.success_angle_tol)),
                "use_cuda_graph": bool(env.cuda_graph),
                "cuda_graph_error": env.cuda_graph_error,
                "policy_input_mode": policy.input_mode,
                "policy_input_size": len(obs_fields(policy.env_mode)) + policy.latent_dim,
            },
        )
    finally:
        env.close()
    classify_rollout(rollout)
    return rollout


def _record_step(
    *,
    step: int,
    holder_obs: np.ndarray,
    rotator_obs: np.ndarray,
    actions: dict,
    config,
    mode: str,
    previous_angle: float | None,
    initial_pos: np.ndarray,
    initial_quat: np.ndarray,
) -> ReducedStep:
    """Turn one end-of-step observation pair into a `ReducedStep`."""
    fields = obs_fields(mode)
    angle = float(holder_obs[fields.index("layer_angle")])
    ang_vel = float(holder_obs[fields.index("layer_ang_vel")])
    rise = 0.0 if previous_angle is None else math.degrees(angle - previous_angle)
    drift_pos, drift_rot = _drift(holder_obs, mode, initial_pos, initial_quat)
    core_pos, core_quat = _core_state(holder_obs, mode)
    holder_force, holder_torque = _physical_wrench(holder_obs, mode, "holder", config)
    rotator_force, rotator_torque = _physical_wrench(rotator_obs, mode, "rotator", config)
    return ReducedStep(
        t=step,
        time_s=step * float(config.control_dt),
        layer_angle_rad=angle,
        layer_ang_vel=ang_vel,
        core_drift_pos_m=drift_pos,
        core_drift_rot_deg=drift_rot,
        holder_force=holder_force,
        holder_torque=holder_torque,
        rotator_force=rotator_force,
        rotator_torque=rotator_torque,
        core_pos=tuple(float(value) for value in core_pos),
        core_quat=tuple(float(value) for value in core_quat),
        holder_command=tuple(float(value) for value in actions["holder"].reshape(-1).cpu()),
        rotator_command=tuple(float(value) for value in actions["rotator"].reshape(-1).cpu()),
        angle_rise_deg=rise,
    )


# --- phase classification (pure) -----------------------------------------------------------


@dataclass(frozen=True)
class PhaseSignals:
    """The per-step numbers the classifier reads (a subset of `ReducedStep`)."""

    holder_force_norm_n: float
    holder_torque_norm_nm: float
    rotator_tau_z_nm: float
    angle_rise_deg: float


def classify_phase(
    signals: PhaseSignals,
    *,
    holder_force_on: float = HOLDER_FORCE_ON_N,
    holder_torque_on: float = HOLDER_TORQUE_ON_NM,
    rotator_tau_on: float = ROTATOR_TAU_ON_NM,
    turn_rise_deg: float = TURN_RISE_DEG,
) -> str:
    """Classify one reduced control step into `hold`, `turn` or `idle`.

    `turn` takes precedence: the rotator's hinge torque is above threshold AND the layer
    rose during the step. Otherwise a nonzero holder wrench (force or torque) is `hold`;
    nothing above threshold is `idle`. (`release` is assigned by the segmenter as the
    trailing run of a finished episode, see `segment_phases`.)

    Args:
        signals: The step's classified quantities.
        holder_force_on: Holder force threshold (N).
        holder_torque_on: Holder torque threshold (Nm).
        rotator_tau_on: Rotator `torque_z` threshold (Nm).
        turn_rise_deg: Layer-rise threshold (deg per control step).

    Returns:
        One of `"turn"`, `"hold"`, `"idle"`.
    """
    turning = abs(signals.rotator_tau_z_nm) >= rotator_tau_on and (
        signals.angle_rise_deg >= turn_rise_deg
    )
    if turning:
        return "turn"
    holding = (
        signals.holder_force_norm_n >= holder_force_on
        or signals.holder_torque_norm_nm >= holder_torque_on
    )
    return "hold" if holding else "idle"


def intended_primitives(phase: str, entry: bool = False) -> tuple[str, str]:
    """The `(holder, rotator)` arm primitives a phase maps to.

    Args:
        phase: One of `PHASES`.
        entry: Whether this is the first decision of the segment (entry primitive) or a
            later one (the held primitive).

    Returns:
        The per-role primitive names.
    """
    if phase not in PHASE_INTENTS:
        raise ValueError(f"unknown phase {phase!r}, expected one of {PHASES}")
    intent = PHASE_INTENTS[phase]
    index = 0 if entry else 1
    return intent["holder"][index], intent["rotator"][index]


def segment_phases(
    steps: list[ReducedStep], control_dt_s: float, success: bool = False
) -> list[PhaseSegment]:
    """Group classified steps into phase segments, merging too-short raw runs.

    Args:
        steps: Steps with a `phase` already assigned (or `""`, classified on the fly).
        control_dt_s: Seconds per reduced control step.
        success: Whether the episode ended in success; a successful episode's trailing
            non-turn run is relabelled `release` (the wrenches drop as the policy finishes).

    Returns:
        The ordered phase segments.
    """
    runs: list[dict] = []
    for step in steps:
        phase = step.phase or classify_phase(step.signals())
        if runs and runs[-1]["phase"] == phase:
            runs[-1]["end"] = step.t
        else:
            runs.append({"phase": phase, "start": step.t, "end": step.t})
    merged = _merge_short_runs(runs)
    if success and merged and merged[-1]["phase"] != "turn":
        merged[-1]["phase"] = "release"
    return [
        PhaseSegment(
            phase=run["phase"],
            t_start=run["start"],
            t_end=run["end"],
            duration_s=(run["end"] - run["start"] + 1) * float(control_dt_s),
        )
        for run in merged
    ]


def _merge_short_runs(runs: list[dict]) -> list[dict]:
    """Merge runs shorter than `MIN_SEGMENT_STEPS` into their neighbour, then re-collapse."""
    merged: list[dict] = []
    for run in runs:
        steps = run["end"] - run["start"] + 1
        if merged and steps < MIN_SEGMENT_STEPS and run["phase"] != merged[-1]["phase"]:
            merged[-1]["end"] = run["end"]
        else:
            merged.append(dict(run))
    collapse: list[dict] = []
    for run in merged:
        if collapse and collapse[-1]["phase"] == run["phase"]:
            collapse[-1]["end"] = run["end"]
        else:
            collapse.append(run)
    return collapse


def classify_rollout(rollout: ReducedRollout) -> None:
    """Fill `phase`, the intended primitives of every step and the phase segments."""
    for step in rollout.steps:
        step.phase = classify_phase(step.signals())
    rollout.phase_segments = segment_phases(
        rollout.steps, rollout.control_dt_s, success=rollout.success
    )
    for segment in rollout.phase_segments:
        for offset, index in enumerate(range(segment.t_start, segment.t_end + 1)):
            step = rollout.steps[index]
            step.phase = segment.phase
            step.intended_holder, step.intended_rotator = intended_primitives(
                segment.phase, entry=offset == 0
            )


# --- primitive schedule (pure) -------------------------------------------------------------


@dataclass
class ScheduleSegment:
    """One arm-decision interval of the primitive schedule."""

    phase: str
    decisions: int
    holder: str
    rotator: str
    source: str = "reduced"
    """`reduced` = derived from the reduced trace; `arm-native` = the arm's own
    bracketing phase (engagement / release) that has no reduced-trace counterpart."""

    def to_json(self) -> dict:
        """The JSON form of the schedule segment."""
        holder_entry, rotator_entry = intended_primitives(self.phase, True)
        return {
            "phase": self.phase,
            "decisions": self.decisions,
            "holder": self.holder,
            "rotator": self.rotator,
            "holder_entry": holder_entry,
            "rotator_entry": rotator_entry,
            "source": self.source,
        }


def _largest_remainder(weights: list[float], total: int) -> list[int]:
    """Allocate `total` integer slots proportionally to `weights` (largest remainder).

    Every weight gets at least one slot (a phase must stay representable), so the result
    may exceed `total` when there are more phases than slots -- callers pass a budget large
    enough for the phase count.
    """
    if total < 1:
        raise ValueError(f"total must be >= 1, got {total}")
    positive = [max(float(weight), 0.0) for weight in weights]
    if sum(positive) <= 0:
        positive = [1.0 for _ in weights]
    exact = [total * weight / sum(positive) for weight in positive]
    counts = [max(1, math.floor(value)) for value in exact]
    remainder = total - sum(counts)
    if remainder > 0:
        order = sorted(
            range(len(weights)),
            key=lambda index: exact[index] - math.floor(exact[index]),
            reverse=True,
        )
        for index in order[:remainder]:
            counts[index] += 1
    return counts


def build_primitive_schedule(
    segments: list[PhaseSegment],
    *,
    timing_mode: str = "share",
    total_decisions: int = 40,
    arm_decision_seconds: float = ARM_DECISION_SECONDS,
    engage_decisions: int = 8,
    release_decisions: int = 3,
) -> list[ScheduleSegment]:
    """Map the reduced phase segments onto the arm's discrete decision grid.

    Two timing modes:

    * `"dt"` -- naive physical-time faithfulness: `round(segment_seconds / arm_decision_seconds)`
      decisions, at least one. This is what an honest dt-faithful port gives, and it is how
      the time-scale mismatch becomes visible (a whole 1.2 s reduced episode is ~3 arm
      decisions).
    * `"share"` -- preserve the reduced phase DURATION SHARES over a configured arm decision
      budget (`total_decisions`, largest remainder, at least one decision per phase). The arm
      keeps its own primitive durations, so this is a *target* the executor can miss.

    The reduced phase sequence is bracketed by two arm-native segments: an `idle`
    engagement (approach+grasp has no reduced counterpart -- the reduced agent commands a
    wrench, not a reach) and a terminating `release` (the reduced episode ends on success,
    so its release is implicit).

    Args:
        segments: The reduced phase segments.
        timing_mode: `"dt"` or `"share"`.
        total_decisions: Arm decision budget of the reduced part in the `"share"` mode.
        arm_decision_seconds: Length of one arm decision (s).
        engage_decisions: Decisions of the synthetic engagement segment.
        release_decisions: Decisions of the synthetic release segment.

    Returns:
        The ordered `ScheduleSegment`s in trace order.
    """
    if timing_mode not in ("dt", "share"):
        raise ValueError(f"timing_mode must be 'dt' or 'share', got {timing_mode!r}")
    collapsed = _collapse_reduced(segments)
    if timing_mode == "dt":
        counts = [
            max(1, round(segment.duration_s / float(arm_decision_seconds))) for segment in collapsed
        ]
    else:
        counts = _largest_remainder(
            [max(segment.duration_s, 1e-6) for segment in collapsed], int(total_decisions)
        )
    schedule: list[ScheduleSegment] = [
        ScheduleSegment(
            phase="idle",
            decisions=max(int(engage_decisions), 1),
            holder=PHASE_INTENTS["idle"]["holder"][0],
            rotator=PHASE_INTENTS["idle"]["rotator"][0],
            source="arm-native",
        )
    ]
    for segment, count in zip(collapsed, counts):
        if segment.phase == "idle":
            # The reduced idle runs are the arm engagement, already emitted above.
            continue
        holder, rotator = intended_primitives(segment.phase, entry=True)
        schedule.append(
            ScheduleSegment(
                phase=segment.phase,
                decisions=int(count),
                holder=holder,
                rotator=rotator,
                source="reduced",
            )
        )
    schedule.append(
        ScheduleSegment(
            phase="release",
            decisions=max(int(release_decisions), 1),
            holder=PHASE_INTENTS["release"]["holder"][0],
            rotator=PHASE_INTENTS["release"]["rotator"][0],
            source="arm-native",
        )
    )
    return schedule


def _collapse_reduced(segments: list[PhaseSegment]) -> list[PhaseSegment]:
    """Merge adjacent segments of the same phase (the segmentation may still produce ties)."""
    collapsed: list[PhaseSegment] = []
    for segment in segments:
        if collapsed and collapsed[-1].phase == segment.phase:
            collapsed[-1].t_end = segment.t_end
            collapsed[-1].duration_s += segment.duration_s
        else:
            collapsed.append(
                PhaseSegment(
                    phase=segment.phase,
                    t_start=segment.t_start,
                    t_end=segment.t_end,
                    duration_s=segment.duration_s,
                )
            )
    return collapsed


def schedule_json(schedule: list[ScheduleSegment]) -> dict:
    """The JSON block of a schedule (including the arm-side timing constants)."""
    return {
        "arm_decision_period_substeps": ARM_DECISION_PERIOD,
        "arm_substep_s": ARM_SUBSTEP_SECONDS,
        "arm_decision_s": ARM_DECISION_SECONDS,
        "segments": [segment.to_json() for segment in schedule],
        "total_decisions": sum(segment.decisions for segment in schedule),
    }


# --- arm side: the realized phase trace and the timing comparison --------------------------


@dataclass
class ArmDecision:
    """One realized arm decision of the bridge execution."""

    decision: int
    holder: str
    rotator: str
    layer_angle_deg: float
    angle_rise_deg: float
    holder_holding: bool
    rotator_grasp: bool
    plan_index: int
    success: bool
    holder_opening: float = 1.0
    phase: str = ""

    def to_json(self) -> dict:
        """The JSON form of one arm decision."""
        return {
            "decision": self.decision,
            "phase": self.phase,
            "holder": self.holder,
            "rotator": self.rotator,
            "layer_angle_deg": self.layer_angle_deg,
            "angle_rise_deg": self.angle_rise_deg,
            "holder_holding": self.holder_holding,
            "holder_opening": self.holder_opening,
            "rotator_grasp": self.rotator_grasp,
            "plan_index": self.plan_index,
            "success": self.success,
        }


def classify_arm_phase(decision: ArmDecision) -> str:
    """Classify one realized arm decision into the bridge's phase labels.

    The arm side has no wrenches, so its evidence is the primitive stream plus the layer
    motion: a release/retreat primitive is `release`; a turn primitive with a rising layer
    is `turn`; an engaged hold (`grasp`/`wait` or the holder's grasp fact) is `hold`;
    everything else is `idle`.

    Args:
        decision: The recorded decision.

    Returns:
        One of `PHASES`.
    """
    if decision.rotator in ARM_RELEASE_PRIMITIVES or decision.holder in ARM_RELEASE_PRIMITIVES:
        return "release"
    if decision.rotator in ARM_TURN_PRIMITIVES and decision.angle_rise_deg >= ARM_TURN_RISE_DEG:
        return "turn"
    if decision.holder in ("grasp", "wait") or decision.holder_holding:
        return "hold"
    return "idle"


def classify_arm_trace(decisions: list[ArmDecision]) -> None:
    """Fill `phase` of every realized arm decision."""
    for decision in decisions:
        decision.phase = classify_arm_phase(decision)


def arm_phase_segments(decisions: list[ArmDecision]) -> list[PhaseSegment]:
    """The realized arm phase runs as `PhaseSegment`s (`duration_s` at decision length)."""
    runs: list[dict] = []
    for decision in decisions:
        phase = decision.phase or classify_arm_phase(decision)
        if runs and runs[-1]["phase"] == phase:
            runs[-1]["end"] = decision.decision
        else:
            runs.append({"phase": phase, "start": decision.decision, "end": decision.decision})
    return [
        PhaseSegment(
            phase=run["phase"],
            t_start=run["start"],
            t_end=run["end"],
            duration_s=(run["end"] - run["start"] + 1) * ARM_DECISION_SECONDS,
        )
        for run in runs
    ]


def _phase_shares(phases: list[str], only: tuple[str, ...] = PHASES) -> dict[str, float]:
    """Fraction of units (steps or decisions) spent in every phase (0 for absent ones)."""
    total = max(len(phases), 1)
    return {phase: phases.count(phase) / total for phase in only}


def _derived_phases(phases: list[str]) -> list[str]:
    """The reduced-derived phases only (`hold`/`turn`), i.e. without the arm-native brackets."""
    return [phase for phase in phases if phase in ("hold", "turn")]


def _first_index(phases: list[str], phase: str) -> float | None:
    """Normalized onset of the first `phase` occurrence in [0, 1] (`None` if absent)."""
    if phase not in phases:
        return None
    return phases.index(phase) / max(len(phases) - 1, 1)


def compare_phase_timing(
    reduced_segments: list[PhaseSegment],
    arm_decisions: list[ArmDecision],
    *,
    reduced_control_dt_s: float,
    reduced_success: bool,
    arm_success: bool,
    plan_length: int = 1,
) -> dict:
    """Measure how faithfully the arm execution reproduces the reduced phase structure.

    The comparison is on NORMALIZED time (phase shares / onset fractions), because the two
    sides' absolute times are not comparable; the physical time scale is reported separately
    through `decision_over_control_step` and the per-phase seconds.

    Args:
        reduced_segments: The reduced phase segments.
        arm_decisions: The realized arm decisions (with `phase` already classified).
        reduced_control_dt_s: Seconds per reduced control step.
        reduced_success: Whether the reduced episode succeeded.
        arm_success: Whether the arm episode reached its env success.
        plan_length: Number of executable moves of the arm plan.

    Returns:
        A JSON-ready metrics dict.
    """
    reduced_phases = [segment.phase for segment in reduced_segments for _ in range(segment.steps)]
    arm_phases = [decision.phase for decision in arm_decisions]
    reduced_shares = _phase_shares(reduced_phases)
    arm_shares = _phase_shares(arm_phases)
    # The arm-native engagement/release brackets have no reduced counterpart, so the fair
    # phase-structure comparison is on the reduced-derived hold/turn subsequences alone.
    derived_reduced = _derived_phases(reduced_phases)
    derived_arm = _derived_phases(arm_phases)
    derived_reduced_shares = _phase_shares(derived_reduced, ("hold", "turn"))
    derived_arm_shares = _phase_shares(derived_arm, ("hold", "turn"))
    derived_l1 = sum(
        abs(derived_reduced_shares[phase] - derived_arm_shares[phase]) for phase in ("hold", "turn")
    )
    reduced_turn_s = sum(
        segment.duration_s for segment in reduced_segments if segment.phase == "turn"
    )
    arm_turn_decisions = sum(1 for decision in arm_decisions if decision.phase == "turn")
    turn_onset_reduced = _first_index(reduced_phases, "turn")
    turn_onset_arm = _first_index(arm_phases, "turn")
    return {
        "reduced_steps": len(reduced_phases),
        "arm_decisions": len(arm_phases),
        "reduced_duration_s": len(reduced_phases) * float(reduced_control_dt_s),
        "arm_duration_s": len(arm_phases) * ARM_DECISION_SECONDS,
        "decision_over_control_step": ARM_DECISION_SECONDS / float(reduced_control_dt_s),
        "reduced_phase_shares": reduced_shares,
        "arm_phase_shares": arm_shares,
        "phase_share_l1_error": sum(
            abs(reduced_shares[phase] - arm_shares[phase]) for phase in PHASES
        ),
        "derived_phase_shares_reduced": derived_reduced_shares,
        "derived_phase_shares_arm": derived_arm_shares,
        "derived_phase_share_l1_error": derived_l1,
        "derived_turn_onset_fraction_reduced": _first_index(derived_reduced, "turn"),
        "derived_turn_onset_fraction_arm": _first_index(derived_arm, "turn"),
        "reduced_turn_duration_s": reduced_turn_s,
        "arm_turn_duration_s": arm_turn_decisions * ARM_DECISION_SECONDS,
        "turn_onset_fraction_reduced": turn_onset_reduced,
        "turn_onset_fraction_arm": turn_onset_arm,
        "turn_onset_offset": (
            None
            if turn_onset_reduced is None or turn_onset_arm is None
            else arm_onset_offset(arm_onset=turn_onset_arm, reduced_onset=turn_onset_reduced)
        ),
        "hold_onset_fraction_reduced": _first_index(reduced_phases, "hold"),
        "hold_onset_fraction_arm": _first_index(arm_phases, "hold"),
        "hold_grasp_to_turn_delay_decisions": _grasp_to_turn_delay(arm_decisions),
        "reduced_success": bool(reduced_success),
        "arm_success": bool(arm_success),
        "plan_length": int(plan_length),
        "reduced_success_fraction": _success_fraction(reduced_success, plan_length),
        "arm_success_fraction": _success_fraction(arm_success, plan_length),
    }


def arm_onset_offset(*, arm_onset: float, reduced_onset: float) -> float:
    """Signed onset difference of a phase: arm fraction minus reduced fraction.

    Positive means the arm starts the phase later than the reduced policy did.
    """
    return float(arm_onset) - float(reduced_onset)


def _grasp_to_turn_delay(arm_decisions: list[ArmDecision]) -> int | None:
    """Arm decisions between the first holder grasp (or hold) and the first turn choice."""
    first_grasp = next(
        (item.decision for item in arm_decisions if item.holder == "grasp" or item.holder_holding),
        None,
    )
    first_turn = next(
        (item.decision for item in arm_decisions if item.rotator in ARM_TURN_PRIMITIVES),
        None,
    )
    if first_grasp is None or first_turn is None:
        return None
    return int(first_turn - first_grasp)


def _success_fraction(success: bool, plan_length: int) -> float:
    """Fraction of the plan the episode actually executed (whole-plan success is 0 or 1)."""
    return 1.0 if success and plan_length > 0 else 0.0


# --- report --------------------------------------------------------------------------------


LIMITATIONS = (
    (
        "Intent->skills bridge, not policy transfer: the reduced wrench policy's weights "
        "cannot act on the arm observation/action space."
    ),
    (
        "Only the phase ORDER and the per-role intent transfer; the arm keeps its own "
        "primitive durations. One arm decision spans ~12.5 reduced control steps, so the "
        "reduced absolute timing is not transferable."
    ),
    (
        "The reduced release is implicit (the episode terminates on success); the schedule's "
        "release/retreat tail is arm-native."
    ),
    (
        "The reduced side is a single greedy episode on a fixed seed (the exemplar the arm "
        "executes); the arm side is a single episode of its fixed plan."
    ),
    (
        "The reduced model has no arms, contacts or gravity, so no physical quantity is "
        "shared; the bridge compares normalised phase structure only."
    ),
)


def bridge_report(
    rollout: ReducedRollout,
    schedule: list[ScheduleSegment],
    *,
    timing_mode: str,
    arm: dict | None = None,
    conductor: dict | None = None,
    timing: dict | None = None,
    notes: list[str] | None = None,
) -> dict:
    """Assemble the bridge JSON: reduced trace + schedule + arm result + timing metrics."""
    return {
        "kind": "intent_to_skills_bridge",
        "note": (
            "learned reduced CubeWrench-v0 policy -> FaceTurnSkills-v0 arm primitives; "
            "an explicit intent bridge, not a weight transfer"
        ),
        "timing_mode": timing_mode,
        "reduced": rollout.to_json(),
        "schedule": schedule_json(schedule),
        "arm": arm,
        "conductor_baseline": conductor,
        "timing": timing,
        "limitations": list(LIMITATIONS),
        "notes": list(notes or []),
    }


def write_json(payload: dict, path: str | Path) -> Path:
    """Write a bridge payload as indented JSON (creating parent directories)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out
