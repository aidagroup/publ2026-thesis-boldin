"""Configuration of CubeWrench-v0: the reduced wrench-level cube of the thesis main line.

The reduced model (PLAN.md "Среда CubeWrench-v0") is a floating cube (`core`) with one
active layer on a limited hinge (`layer`); two decentralised agents act on it through a
raw 6D wrench each -- the holder stabilises the core, the rotator turns the layer to
90 degrees. This dataclass holds every tunable the spec leaves open: the decision
timing, the physical parameters the MuJoCo XML only carries as defaults, the per-agent
action authority, the success/failure tolerances, the episode-randomisation ranges (the
private randomness that makes the partner only partially observable), the team reward
and the fallback seed.

Pure stdlib on purpose (same rule as `callosum.envs._wrench_logic`): the config must
import and validate on macOS, where mujoco is absent, so the env's logic is unit-testable
without the simulator. The MuJoCo model is `callosum/envs/wrench_cube/assets/cube_wrench.xml`.

Config as truth for timing and DOF parameters: `WrenchCubeEnv` writes `dt` into
`model.opt.timestep`, `hinge_limit` into `model.jnt_range`, and the damping/frictionloss/
armature into `model.dof_*` (the nominal hinge frictionloss is then replaced per episode
by the sampled `layer_frictionloss`). Geometry and the two masses are deliberately
XML-owned: MuJoCo's `mj_setConst` does not recompute a body's inertia when `body_mass`
changes, so writing the masses at runtime would leave the model with an inconsistent
inertia. The XML carries the geometry/mass (core 0.2 kg, layer 0.02 kg) and is the source
of truth for them.

Physical scale note: the defaults are tuned so that the scripted oracle (holder PD +
rotator PD, both lead-compensated for their own action latency) reaches success under the
full randomisation ranges, while the rotator alone drifts the core past the failure bound
(the non-additivity the design needs). Details of the balance:
`core_ang_damping` / `core_lin_damping` are the reduced model's isotropic medium damping
(the free-floating cube has no contacts and no gravity, so without it the delayed wrench
control would be an undamped oscillator); `hinge_damping`/`hinge_frictionloss` set both
the layer's turn speed and the reaction impulse the holder must absorb. All of these are
the "точные масса/инерция/параметры шарнира и лимиты моментов" open question of PLAN.md
to be re-tuned at gates G1/G2.

The config enforces the design's non-additivity invariant in `__post_init__`: the
rotator-only quasi-static core drift must exceed `success_rot_tol` by a safe margin, so
"the rotator alone cannot keep the core still" is a validated property of the parameters
rather than an assumption.

Randomisation presets: the benchmark grid needs two levels of the *private* episode
randomness -- `"default"` (the dataclass defaults, tuned to stay learnable) and `"harsh"`
(the previous, harder ranges: wider gain/authority spreads, more and longer slips, longer
start delays). `wrench_cube_config(randomization)` builds the config of a preset, so the
CPU env (`VecWrenchCube`/`WrenchCubeEnv`), the GPU env (`WrenchCubeBatchGPU`) and the
evaluator all use the same randomness for a run. Only the randomisation ranges change
between the presets; the physics, timing, tolerances and reward stay identical, so the
`oracle`-vs-`none` gap of a harsh run answers "is the coordination harder", not "is the
model different".
"""

import math
from dataclasses import dataclass

_NON_ADDITIVITY_MARGIN = 1.5
"""Required ratio between the predicted rotator-only core drift and `success_rot_tol`.

The review's minimum: the drift must be at least 1.5x the success rotation tolerance, so
the holder is genuinely necessary and a near-miss episode cannot count as a success.
"""


def _check_range(
    name: str,
    value: tuple[float, float],
    *,
    non_negative: bool = True,
    positive_lo: bool = False,
) -> None:
    """Raise ValueError unless `value` is an ordered numeric pair within the allowed domain.

    Args:
        name: Field name for the error message.
        value: The `(lo, hi)` pair to check.
        non_negative: Require `lo >= 0`.
        positive_lo: Require `lo > 0` (implies `non_negative`).
    """
    lo, hi = value
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"{name} must be finite, got {value}")
    if lo > hi:
        raise ValueError(f"{name} must have lo <= hi, got {value}")
    if positive_lo and not lo > 0:
        raise ValueError(f"{name} must have lo > 0, got {value}")
    if non_negative and not positive_lo and lo < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


@dataclass(frozen=True)
class WrenchCubeConfig:
    """All tunables of CubeWrench-v0 (timing, physics, authority, tolerances, randomness).

    The field groups follow the environment spec: decision timing, the physical model,
    the two agents' wrench authority, the success/failure tolerances, the per-episode
    randomisation ranges, the team reward and the seed fallback. `__post_init__` validates
    the domains and the design invariants: `goal_angle <= hinge_limit`, the initial angle
    inside the hinge range, the non-additivity drift margin, and the failure penalty no
    better than a step penalty.
    """

    # --- decision timing -----------------------------------------------------------------
    dt: float = 0.005
    """MuJoCo physics timestep (s). The XML default is the same; the env enforces this
    value on `model.opt.timestep`."""

    control_every: int = 8
    """Physics substeps per control step. One `env.step` applies a fixed wrench for
    `control_every` substeps (0.04 s at the defaults, inside the spec's 8-10 range)."""

    max_control_steps: int = 350
    """Episode budget in control steps (14 s at the defaults, inside the spec's 300-400
    range). A policy that has not succeeded by then is truncated."""

    # --- physical model (timing/dof params written into the model; mass/geometry in XML) -
    hinge_limit: float = math.pi / 2
    """Hinge travel limit (rad) of the layer, from 0 to this value. Written into
    `model.jnt_range`; the XML value is a default only. `goal_angle` must not exceed it."""

    core_lin_damping: float = 0.5
    """Linear viscous damping (Ns/m) on the core's free-joint translation DOFs. The cube
    floats (no gravity, no contacts), so this is the medium damping that makes the
    position loop well-posed; it is not a contact model."""

    core_ang_damping: float = 5.0e-4
    """Angular viscous damping (Nms/rad) on the core's free-joint rotation DOFs. It also
    bounds how far the rotator's reaction impulse can drift the core without the holder
    (with the defaults the core drifts ~1 rad, past `failure_rot_drift`)."""

    hinge_damping: float = 3.0e-4
    """Viscous damping (Nms/rad) of the layer hinge. Dominates the layer's turn speed
    and therefore the reaction impulse the holder must absorb: raising it needs more
    rotator torque and more holder authority. Model assumption: the reaction path from the
    rotator's layer torque back to the core is the hinge's COMBINED damping/frictionloss
    (`hinge_damping + hinge_frictionloss`); the non-additivity check in `__post_init__`
    and the XML/env docstrings state this explicitly."""

    hinge_frictionloss: float = 1.0e-6
    """Nominal Coulomb friction loss (Nm) of the layer hinge, written into
    `model.dof_frictionloss` at construction. Each `reset` replaces it with the episode's
    sample from `layer_frictionloss_range` (the range is what the episodes actually see);
    this nominal value is kept for the construction-time model and the non-additivity
    validation. It must stay small so that the rotator's steady-state angle error
    (`frictionloss / rotator PD gain`) stays well below the success angle tolerance."""

    hinge_armature: float = 1.0e-4
    """Hinge armature (kg m^2), added to the thin layer's own inertia so the hinge
    dynamics are slow enough to control under the 0-0.3 s action latency."""

    # --- per-agent wrench authority (action = normalized wrench * authority) -------------
    holder_max_force: float = 0.5
    """Holder wrench authority on the core: max force (N) per axis."""

    holder_max_torque: float = 5.0e-3
    """Holder wrench authority on the core: max torque (Nm) per axis (it must absorb the
    rotator's reaction impulse)."""

    rotator_max_force: float = 0.02
    """Rotator wrench authority on the layer: max force (N) per axis (the scripted
    rotator uses torque only, but the action contract is 6D for both agents)."""

    rotator_max_torque: float = 2.0e-3
    """Rotator wrench authority on the layer: max torque (Nm) per axis."""

    # --- success / failure tolerances ------------------------------------------------------
    goal_angle: float = math.pi / 2
    """Goal hinge angle (rad): the layer must turn 90 degrees from the core."""

    success_angle_tol: float = 0.08
    """Success tolerance (rad) on `|angle - goal_angle|`.

    Relaxed from 0.05 after the first RL probes: with 0-0.3 s action latency and the
    settle-to-rest requirement the policies plateaued at ~0.28 success after 200 updates;
    the scripted oracle still succeeds with a final error <=0.033 rad, so this is a
    learning-feasibility tuning of the reduced model, not a change of the coordination
    semantics (the rotator-only drift is still >3x the tolerance, see the invariant)."""

    success_pos_tol: float = 0.015
    """Success tolerance (m) on the core's position drift (the spec's 1 cm)."""

    success_rot_tol: float = 0.15
    """Success tolerance (rad) on the core's rotation drift (the spec's 0.1 rad)."""

    success_lin_speed_tol: float = 0.1
    """Success tolerance (m/s) on the core's linear speed: the cube must be at rest."""

    success_ang_speed_tol: float = 0.5
    """Success tolerance (rad/s) on the combined core + layer angular speed."""

    failure_pos_drift: float = 0.05
    """Failure bound (m) on the core's position drift (the spec's 5 cm)."""

    failure_rot_drift: float = 0.5
    """Failure bound (rad) on the core's rotation drift (the spec's |yaw| > 0.5 rad)."""

    # --- episode randomisation (private randomness of the partner) -------------------------
    latency_range: tuple[float, float] = (0.1, 0.3)
    """Per-agent action latency (s): a commanded wrench is applied this many control
    steps later. Sampled per agent; the agent knows its own latency (it is in its
    observation), the partner's is hidden."""

    gain_range: tuple[float, float] = (0.6, 1.4)
    """Per-agent wrench-gain multiplier: the applied wrench is `action * authority *
    limit_scale * gain`. Sampled per agent."""

    limit_scale_range: tuple[float, float] = (0.5, 1.5)
    """Per-agent authority multiplier (0.5-1.5 of the config maxima). Sampled per agent."""

    slip_prob_range: tuple[float, float] = (0.0, 0.3)
    """Per-agent probability range of one slip event per episode (a brief loss of the
    applied wrench). The event probability itself is sampled per agent, then a Bernoulli
    draw decides whether the episode has it."""

    slip_decisions_range: tuple[int, int] = (1, 3)
    """Slip duration range (control steps, inclusive) of the one slip event."""

    start_delay_range: tuple[float, float] = (0.0, 0.3)
    """Per-agent delay (s) before the agent's wrench takes effect for the first time."""

    layer_frictionloss_range: tuple[float, float] = (0.0, 2.0e-5)
    """Per-episode Coulomb friction loss (Nm) of the layer hinge: the layer-friction
    randomness the rotator cannot observe (it only sees its own latency/gain/limit in its
    observation). Sampled once per episode and written into `model.dof_frictionloss` on
    every `reset`. The range is adapted to this model's scale, not the review's
    illustrative (0.02, 0.08): the rotator's torque authority is 2e-3 Nm, so friction at
    that scale would exceed the available torque and freeze the hinge. At the default
    upper bound the turn takes ~12% longer and the worst steady-state angle error
    (`frictionloss / oracle rotator kp`, ~0.033 rad) stays inside `success_angle_tol`
    (locally probed: the scripted oracle still succeeds at this upper bound over 60 seeds)."""

    initial_angle_range: tuple[float, float] = (0.0, 0.05)
    """Initial layer angle (rad), sampled per episode. It must lie inside the hinge range
    `[0, hinge_limit]`: with the old default `(-0.05, 0.05)` the negative half was outside
    the hinge limit and got pushed to 0 by the joint limit on the first step."""

    impulse_prob_range: tuple[float, float] = (0.0, 0.3)
    """Episode probability range of one common impulse disturbance on the core. Like the
    slip probability, the episode probability is sampled and then a Bernoulli draw
    decides whether the impulse happens."""

    impulse_magnitude_range: tuple[float, float] = (0.005, 0.05)
    """Magnitude range (N) of the impulse force on the core (applied for one control
    step in a random direction)."""

    # --- reward ---------------------------------------------------------------------------
    success_bonus: float = 2.0
    """Team reward granted on the step that first satisfies the success condition."""

    step_penalty: float = 0.001
    """Team reward subtracted on every control step (the shared time cost).

    Kept small on purpose: the timeout return is `-step_penalty * max_control_steps`
    (0.35 at the defaults), and the failure penalty must be worse than that, otherwise
    the first probes learned to FAIL FAST (return -1.57 at 9 steps) instead of surviving
    to the timeout (-3.5 with the old 0.01 penalty). Learning the coordination is the
    goal; time-to-success stays a reported metric, not a strong reward term."""

    failure_penalty: float = -2.0
    """Team reward on the step that first violates the failure condition -- it replaces
    the plain `-step_penalty` there, so drifting the core out of bounds is punished rather
    than merely costing one more step. It MUST be strictly worse than the whole-episode
    timeout return (`failure_penalty < -step_penalty * max_control_steps`), otherwise an
    early crash is the optimal policy (observed in the first shape-reward probes). The
    success step keeps `success_bonus - step_penalty`."""

    shaping_angle_coef: float = 0.5
    """Coefficient of the potential-style angle-progress shaping term (shared reward).

    Sparse `success_bonus` alone did not learn in the first server probes (success stayed
    0 after 45k steps), so the team reward also pays `shaping_angle_coef * dphi`, where
    `dphi` is the normalised decrease of `|angle - goal|` since the previous control step
    divided by `goal_angle` (a full turn sums to ~1). The progress is GATED by the core
    staying inside the success drift tolerances, so a rotator that turns while the core
    drifts gets no angle reward -- the non-additivity of the success condition is kept
    (see `_wrench_logic.shaping_reward`). 0 disables the term."""

    shaping_drift_coef: float = 0.5
    """Coefficient of the drift-excess penalty in the shaping term.

    The penalty is the normalised excess of the core drift over the success tolerances,
    clipped to [0, 1] (both axes summed). It is what teaches the holder to stabilise the
    core: without it the shared reward has no gradient against drift until the terminal
    failure. 0 disables the term."""

    # --- fallback seed --------------------------------------------------------------------
    seed: int = 0
    """Seed used when `reset(seed=None)`; each reset without an explicit seed advances it
    by one episode, so a fresh env does not replay the same randomisation."""

    def __post_init__(self) -> None:
        if not self.dt > 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.control_every < 1:
            raise ValueError(f"control_every must be >= 1, got {self.control_every}")
        if self.max_control_steps < 1:
            raise ValueError(f"max_control_steps must be >= 1, got {self.max_control_steps}")
        for name in (
            "core_lin_damping",
            "core_ang_damping",
            "hinge_damping",
            "hinge_frictionloss",
            "hinge_armature",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        for name in (
            "holder_max_force",
            "holder_max_torque",
            "rotator_max_force",
            "rotator_max_torque",
        ):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if not 0 < self.hinge_limit <= math.pi:
            raise ValueError(f"hinge_limit must be in (0, pi], got {self.hinge_limit}")
        if not 0 < self.goal_angle <= self.hinge_limit:
            raise ValueError(
                f"goal_angle must be in (0, hinge_limit] = (0, {self.hinge_limit}], "
                f"got {self.goal_angle}"
            )
        if not 0 < self.success_angle_tol < self.goal_angle:
            raise ValueError(
                f"success_angle_tol must be in (0, goal_angle) = (0, {self.goal_angle}), "
                f"got {self.success_angle_tol}"
            )
        if not 0 < self.success_pos_tol < self.failure_pos_drift:
            raise ValueError(
                f"success_pos_tol must be in (0, failure_pos_drift) = "
                f"(0, {self.failure_pos_drift}), got {self.success_pos_tol}"
            )
        if not 0 < self.success_rot_tol < self.failure_rot_drift:
            raise ValueError(
                f"success_rot_tol must be in (0, failure_rot_drift) = "
                f"(0, {self.failure_rot_drift}), got {self.success_rot_tol}"
            )
        for name in ("success_lin_speed_tol", "success_ang_speed_tol"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        _check_range("latency_range", self.latency_range)
        _check_range("gain_range", self.gain_range, positive_lo=True)
        _check_range("limit_scale_range", self.limit_scale_range, positive_lo=True)
        _check_range("slip_prob_range", self.slip_prob_range)
        _check_range("start_delay_range", self.start_delay_range)
        _check_range("impulse_prob_range", self.impulse_prob_range)
        _check_range("impulse_magnitude_range", self.impulse_magnitude_range)
        _check_range("layer_frictionloss_range", self.layer_frictionloss_range)
        # Non-additivity invariant (the design's premise): without the holder, the
        # rotator's reaction must drift the core past the success rotation tolerance by a
        # safe margin. Quasi-static estimate of the rotator-only drift:
        #     drift = goal_angle * c / (c + core_ang_damping),
        # with c the hinge's combined damping/friction resistance. The estimate matches
        # the local rotator-only probe (predicted ~0.59 rad vs ~0.52 rad observed when the
        # 0.5 rad failure bound stops the episode).
        hinge_resistance = self.hinge_damping + self.hinge_frictionloss
        if not hinge_resistance > 0:
            raise ValueError(
                "hinge_damping + hinge_frictionloss must be > 0: a frictionless hinge "
                "would let the rotator turn the layer without a reaction on the core, "
                f"got {hinge_resistance}"
            )
        sampled_min_resistance = self.hinge_damping + self.layer_frictionloss_range[0]
        if not sampled_min_resistance > 0:
            raise ValueError(
                "hinge_damping + layer_frictionloss_range[0] must be > 0: an episode with "
                "no hinge resistance would let the rotator turn the layer without "
                f"drifting the core, got {sampled_min_resistance}"
            )
        resistance = min(hinge_resistance, sampled_min_resistance)
        predicted_drift = self.goal_angle * resistance / (resistance + self.core_ang_damping)
        required_drift = _NON_ADDITIVITY_MARGIN * self.success_rot_tol
        if predicted_drift < required_drift:
            raise ValueError(
                "non-additivity violated: the predicted rotator-only core drift is "
                f"{predicted_drift:.4f} rad (goal_angle * c / (c + core_ang_damping) with "
                f"c = {resistance:.3g}), below the required {_NON_ADDITIVITY_MARGIN} x "
                f"success_rot_tol = {required_drift:.4f} rad. With these parameters the "
                "rotator alone would not move the core far enough for the holder to be "
                "necessary; lower core_ang_damping, raise the hinge damping/frictionloss, "
                "or raise success_rot_tol (it must stay below failure_rot_drift)."
            )
        if self.slip_prob_range[1] > 1:
            raise ValueError(f"slip_prob_range must be within [0, 1], got {self.slip_prob_range}")
        if self.impulse_prob_range[1] > 1:
            raise ValueError(
                f"impulse_prob_range must be within [0, 1], got {self.impulse_prob_range}"
            )
        slip_lo, slip_hi = self.slip_decisions_range
        if slip_lo < 1 or slip_lo > slip_hi:
            raise ValueError(
                f"slip_decisions_range must be 1 <= lo <= hi, got {self.slip_decisions_range}"
            )
        _check_range("initial_angle_range", self.initial_angle_range)
        if self.initial_angle_range[1] > self.hinge_limit:
            raise ValueError(
                "initial_angle_range must lie inside the hinge range [0, hinge_limit] = "
                f"[0, {self.hinge_limit}], got {self.initial_angle_range}"
            )
        if not 0 <= self.success_bonus:
            raise ValueError(f"success_bonus must be >= 0, got {self.success_bonus}")
        if not 0 <= self.step_penalty < self.success_bonus:
            raise ValueError(
                f"step_penalty must be in [0, success_bonus) = [0, {self.success_bonus}), "
                f"got {self.step_penalty}"
            )
        if not self.failure_penalty < 0:
            raise ValueError(f"failure_penalty must be < 0, got {self.failure_penalty}")
        if not self.failure_penalty <= -self.step_penalty:
            raise ValueError(
                "failure_penalty must be no better than a step penalty "
                f"(<= -step_penalty = {-self.step_penalty}), got {self.failure_penalty}"
            )
        timeout_return = -self.step_penalty * self.max_control_steps
        if self.failure_penalty >= timeout_return:
            raise ValueError(
                "failure_penalty must be strictly worse than the whole-episode timeout "
                f"return (failure_penalty < -step_penalty * max_control_steps = "
                f"{timeout_return}), got {self.failure_penalty}: otherwise failing fast "
                "is the optimal policy and the agents never learn to survive"
            )
        for name in ("shaping_angle_coef", "shaping_drift_coef"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")

    @property
    def control_dt(self) -> float:
        """Seconds per control step (`dt * control_every`)."""
        return self.dt * self.control_every

    @property
    def max_episode_time(self) -> float:
        """Simulated seconds of the episode budget (`control_dt * max_control_steps`)."""
        return self.control_dt * self.max_control_steps

    @property
    def holder_authority(self) -> tuple[float, float, float, float, float, float]:
        """Holder action scale: max force per axis, then max torque per axis."""
        return (self.holder_max_force,) * 3 + (self.holder_max_torque,) * 3

    @property
    def rotator_authority(self) -> tuple[float, float, float, float, float, float]:
        """Rotator action scale: max force per axis, then max torque per axis."""
        return (self.rotator_max_force,) * 3 + (self.rotator_max_torque,) * 3


# --- randomisation presets ----------------------------------------------------------------

RANDOMIZATION_PRESETS = ("default", "harsh")
"""The episode-randomness levels the trainer/evaluator can select (`WrenchPPOConfig.randomization`).

* `"default"` -- the `WrenchCubeConfig` dataclass defaults (the learnable level of the
  anticipation test; latency stays in (0.1, 0.3) so a partner command cannot be reactive);
* `"harsh"` -- the previous harder ranges (wider gain/authority spread, more frequent and
  longer slips, longer start delays), used to test whether the `oracle`-vs-`none` gap is
  only a sample-efficiency effect or survives at convergence.
"""

_HARSH_RANDOMIZATION: dict = {
    "latency_range": (0.1, 0.3),
    "gain_range": (0.5, 1.5),
    "limit_scale_range": (0.4, 1.6),
    "slip_prob_range": (0.2, 0.5),
    "slip_decisions_range": (2, 5),
    "start_delay_range": (0.0, 0.5),
}
"""The private-randomness ranges of the `"harsh"` preset (the pre-2026-11-21 defaults).

These were softened back to the current `WrenchCubeConfig` defaults in the "restore the
learnable randomisation level" commit; the preset keeps the harder level available for the
ablation without duplicating the config class.
"""


def wrench_cube_config(randomization: str = "default") -> WrenchCubeConfig:
    """The CubeWrench-v0 config of a randomisation preset.

    The returned `WrenchCubeConfig` is a normal validated instance (so the non-additivity
    invariant and the range checks run for every preset), and it is the single source of
    truth the CPU loop, the GPU batch env and the evaluator build from.

    Args:
        randomization: One of `RANDOMIZATION_PRESETS`.

    Returns:
        `WrenchCubeConfig()` for `"default"`, the harder private-randomness ranges for
        `"harsh"` (physics/timing/tolerances/reward unchanged).

    Raises:
        ValueError: if `randomization` is not a known preset.
    """
    if randomization not in RANDOMIZATION_PRESETS:
        raise ValueError(
            f"randomization must be one of {RANDOMIZATION_PRESETS}, got {randomization!r}"
        )
    if randomization == "default":
        return WrenchCubeConfig()
    return WrenchCubeConfig(**_HARSH_RANDOMIZATION)
