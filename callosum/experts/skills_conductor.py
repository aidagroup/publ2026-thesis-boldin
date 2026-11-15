"""The scripted conductor of FaceTurnSkills-v0: an oracle-informed hard program.

Not a result, a validation: the conductor drives both hands' primitive choices from
information a policy input carries (its own observation block) plus the oracle `info`
keys (`_skills_obs`: partner primitive ids, plan index, layer angle, success) and is
expected to reach success ~100% on E1/E2 -- the environment's sanity check and upper
baseline (skill-action-space spec: "Дирижёр — только валидация"). Playing its decision
sequence back as a demo is forbidden; the demo must run trained policies.

Realism constraints honoured (it may NOT peek at hidden state): it tracks the
partner's grasp through the partner's CURRENT primitive id in `info` (a grasp
streak establishes a hold belief that survives the partner's `wait` between plan
moves; releases, retreats, approaches and flips clear it), verifies its own
outcomes through its own observation (the flip's orientation change, the turn's
layer angle, its own gripper opening after a grasp, the plan cursor advancing in
`info`) and retries on failure -- a grasp-failure injection or a mid-turn slip
sends it back to a re-approach, never into a wall.

Pure stdlib + numpy (no mani_skill/torch): unit-testable on macOS with mock
observations (`tests/test_face_turn_skills_logic.py`). One instance serves ONE env;
`act(agent_key, obs, info)` returns that hand's primitive id for this decision, where
`obs` is the hand's policy input (own block, optionally with an oracle partner block
appended -- ignored; the oracle facts arrive via `info`).
"""

import math

import numpy as np

from callosum.envs._plans import FACES, Flip
from callosum.envs._skills import HOLDER_PRIMITIVES, ROTATOR_PRIMITIVES, Role, num_primitives
from callosum.envs._skills_obs import (
    INFO_ACTIVE_LAYER_ANGLE_DEG,
    INFO_PRIMITIVE_HOLDER,
    own_block_slices,
)
from callosum.experts.skills_runtime import (
    orientation_from_quat,
    turn_target_rad,
    up_face_from_quat,
)

HOLDER_ID = {p.name: p.index for p in HOLDER_PRIMITIVES}
ROTATOR_ID = {p.name: p.index for p in ROTATOR_PRIMITIVES}

_SETTLE_DECISIONS = 2
"""Extra decisions to wait after a primitive reports done (progress 1): the arm
settles and the observation catches up before the conductor commits to the next step."""
_GRASP_PARTNER_STREAK = 2
"""The rotator waits this many consecutive decisions of the holder's `info` primitive
being `grasp` before trusting the grasp -- the actual contact is not observable."""
_STATE_TIMEOUT = 30
"""Decisions a conductor state may take before it resynchronizes (release/retreat/
re-approach) -- the injected failures are recovered through this escape hatch."""
_ANGLE_TOL_DEG = 6.0  # verification window around the move target (config tol + margin)
_FLIP_RETRIES = 8  # flip attempts before holding anyway (the episode budget caps work)
_GRASP_RETRIES = 8  # grasp attempts (a failed one reads as open jaws) before giving up
_SEQ_HOLD = 2  # decisions each step of a release/retreat sequence is held
_NOT_HOLDING_OPENING = 0.9
_HOLDING_OPENING_MIN = 0.1
"""A hand verifies its grasp through its own `own_gripper_opening`: fully open is
1.0 (an injected failure never closes the jaws), jaws fully closed (0.0) closed
on air, and a real clamp reads the grasped width: the holder pinching the full
cube (~57 mm) settles around 0.77, the rotator pinching the layer rim (~19 mm)
around 0.16-0.26. Only the (min, max] band counts as a hold."""


def _to_float(value) -> float:
    """Scalar of a 0-d/1-element tensor, numpy scalar or python number."""
    array = np.asarray(value, dtype=float).reshape(-1)
    return float(array[0]) if len(array) else float(value)


def _to_int(value) -> int:
    array = np.asarray(value).reshape(-1)
    return int(array[0]) if len(array) else int(value)


class _HandState:
    """Bookkeeping one hand's conductor side needs between decisions."""

    def __init__(self) -> None:
        self.phase = "wait"
        self.in_phase = 0  # decisions spent in the current phase
        self.done_settle = 0  # consecutive decisions the own primitive reported done
        self.retries = 0
        self.streak = 0  # consecutive decisions the watched partner condition held
        self.partner_hold = False  # believed state of the partner's grasp (rotator)
        self.grasp_failures = 0  # grasp attempts that ended with open jaws
        self.step = 0  # position inside a multi-decision sequence phase
        self.seq_steps: list = []  # the pending sequence's primitive ids
        self.seq_then = "wait"  # the phase entered when the sequence completes
        self.target_deg: float | None = None  # the turn target remembered at issuance


class SkillsConductor:
    """Oracle-informed scripted chooser of both hands' primitives; see the module doc.

    The observation is read ONLY through the `_skills_obs` own-block layout (indices
    via `own_block_slices`), so a policy input built by
    `callosum.training._skills_agent_obs.build_agent_obs` drops in directly.
    """

    def __init__(self, settle_decisions: int = _SETTLE_DECISIONS) -> None:
        self.settle_decisions = settle_decisions
        self.holder = _HandState()
        self.rotator = _HandState()
        self._last = {Role.HOLDER: HOLDER_ID["wait"], Role.ROTATOR: ROTATOR_ID["wait"]}

    # --- interface ---------------------------------------------------------------------------

    def act(self, agent_key: str, obs, info: dict) -> int:
        """One hand's primitive id for this decision (`agent_key` holder/rotator or uid)."""
        obs = np.asarray(obs, dtype=float).reshape(-1)
        if "holder" in agent_key:
            state, role = self.holder, Role.HOLDER
        elif "rotator" in agent_key:
            state, role = self.rotator, Role.ROTATOR
        else:
            raise ValueError(f"agent_key must name the holder or rotator, got {agent_key!r}")
        state.in_phase += 1
        fields = self._fields(obs, role)
        progress = float(fields["own_primitive_progress"][0])
        state.done_settle = state.done_settle + 1 if progress >= 0.999 else 0
        choice = self._holder_act(fields, info) if role is Role.HOLDER else self._rotator_act(
            fields, info
        )
        self._last[role] = int(choice)
        return int(choice)

    # --- observation reading (the _skills_obs own-block contract) ---------------------------

    def _fields(self, obs: np.ndarray, role: Role) -> dict:
        n_own = num_primitives(role)
        slices = own_block_slices(n_own)
        needed = slices["own_primitive_progress"].stop
        if len(obs) < needed:
            raise ValueError(
                f"observation has {len(obs)} columns, too narrow for the "
                f"{role.value} own block ({needed})"
            )
        return {name: obs[slice_] for name, slice_ in slices.items()}

    def _move_face(self, fields: dict) -> str | None:
        onehot = fields["plan_face_onehot"]
        if float(onehot.max()) <= 0.0:
            return None  # plan done (or empty): no pending move
        return FACES[int(np.argmax(onehot))]

    def _move_quarter(self, fields: dict) -> int:
        return max(round(float(fields["plan_quarter_turns"][0]) * 2), 1)

    def _up_face(self, fields: dict) -> str:
        return up_face_from_quat(fields["cube_orientation_quat"])

    def _holding(self, fields: dict) -> bool:
        """Whether this hand's own jaws read as clamping something (proprioception).

        Clamped on the slab the normalized opening settles around 0.8; an injected
        failure leaves the jaws open (1.0) and a grasp that missed its target closes
        fully on air (0.0) -- only the in-between band counts as a hold."""
        opening = float(fields["own_gripper_opening"][0])
        return _HOLDING_OPENING_MIN < opening <= _NOT_HOLDING_OPENING

    def _needed_flip(self, fields: dict) -> Flip | None:
        face = self._move_face(fields)
        if face is None:
            return None
        orientation = orientation_from_quat(fields["cube_orientation_quat"])
        flips, _ = orientation.flips_to_bring_up(face)
        return flips[0] if flips else None

    # --- the holder ----------------------------------------------------------------------

    def _holder_act(self, fields: dict, info: dict) -> int:
        state = self.holder
        ids = HOLDER_ID
        del info  # the holder's oracle needs are served by its own observation
        if state.phase == "done":
            return ids["wait"]
        if self._move_face(fields) is None:
            # Plan finished: let go and stand by (success terminates the episode).
            return self._sequence(state, [ids["release"], ids["retreat"]], then="done")
        needed = self._needed_flip(fields)
        phase = state.phase
        if phase in ("wait", "done"):
            return self._enter(state, "approach", ids["approach_body"])
        if phase == "approach":
            return self._after_settle(state, "grasp", ids["grasp"])
        if phase == "grasp":
            if self._timed_out(state):
                return self._resync(state)
            return self._after_settle(state, "serving", ids["wait"])
        if phase == "serving":
            if not self._holding(fields):
                # The grasp never engaged (an injected failure leaves the jaws open)
                # or was lost: re-approach and re-grasp instead of holding air.
                if state.grasp_failures >= _GRASP_RETRIES:
                    return ids["wait"]  # give up; the episode budget decides
                state.grasp_failures += 1
                return self._enter(state, "approach", ids["approach_body"])
            state.grasp_failures = 0
            # Holding the core: flip the needed face up, or hold while the rotator turns.
            if needed is None:
                return ids["wait"]
            if state.retries >= _FLIP_RETRIES:
                return ids["wait"]  # give up flipping; the episode budget decides
            state.retries += 1
            flip_id = ids["flip90"] if needed.quarter_turns == 1 else ids["flip180"]
            return self._enter(state, "flipping", flip_id)
        if phase == "flipping":
            if state.done_settle >= self.settle_decisions:
                # Verify through the OWN observation: did the move's face come up?
                if self._up_face(fields) != self._move_face(fields):
                    # The grasp failed/slid, or the grip's finger could not aim at
                    # this flip's axis (the roll gate parked the macro, grip intact):
                    # the resync releases and re-approaches aimed at THIS move's axis.
                    return self._resync(state)
                # Keep the grip: the lower leg cannot press the cube onto the table
                # anyway (it ends holding it at the stall height), and the rotator
                # turns the layer of the hanging cube.
                return self._enter(state, "serving", ids["wait"])
            return self._reselect(state)
        if phase == "seq":
            return self._sequence(state, state.seq_steps, state.seq_then)
        return self._enter(state, "wait", ids["wait"])

    # --- the rotator ---------------------------------------------------------------------

    def _rotator_act(self, fields: dict, info: dict) -> int:
        state = self.rotator
        ids = ROTATOR_ID
        partner = _to_int(info.get(INFO_PRIMITIVE_HOLDER, 0))
        if partner == HOLDER_ID["grasp"]:
            state.streak += 1
        else:
            state.streak = 0
        # Partner-hold belief, from the holder's primitive stream alone: a grasp
        # held for the streak margin means the jaws closed on the core (and stay
        # closed through the holder's `wait` AND its flips -- the flip macro
        # keeps the grip by design), so the belief survives the holder idling
        # between plan moves and across flips -- the streak alone would never
        # re-arm after one. Only selections that open the jaws or park them away
        # (release/retreat/approach) clear it.
        if partner in (
            HOLDER_ID["release"],
            HOLDER_ID["retreat"],
            HOLDER_ID["approach_body"],
        ):
            state.partner_hold = False
        elif state.streak >= _GRASP_PARTNER_STREAK:
            state.partner_hold = True
        face = self._move_face(fields)
        if state.phase == "done":
            return ids["wait"]
        if face is None:
            return self._sequence(state, [ids["release"], ids["retreat"]], then="done")
        if state.phase not in ("wait", "done", "resync") and self._timed_out(state):
            return self._resync(state)

        phase = state.phase
        if phase in ("wait", "done"):
            if self._up_face(fields) == face and (
                state.streak >= _GRASP_PARTNER_STREAK or state.partner_hold
            ):
                return self._enter(state, "approach", ids["approach_face"])
            return ids["wait"]
        if phase == "approach":
            return self._after_settle(state, "grasp", ids["grasp"])
        if phase == "grasp":
            if state.done_settle >= self.settle_decisions:
                if not self._holding(fields):
                    # The jaws missed the layer (fully closed on air reads 0.0) or
                    # never closed (an injected failure reads fully open): retry.
                    return self._resync(state)
                # Remember the move's target NOW: after the turn the angle has moved,
                # so recomputing `turn_target_rad` later would target the NEXT detent.
                state.target_deg = math.degrees(
                    turn_target_rad(self._angle_rad(fields), self._move_quarter(fields))
                )
                return self._enter(state, "turn", self._turn_choice(fields))
            return self._reselect(state)
        if phase == "turn":
            if state.step == 1:
                # The one-decision gap interrupted the finished turn; re-issue a
                # FRESH one (the changed selection replans from the current angle).
                state.step = 0
                return self._turn_choice(fields)
            if state.done_settle >= self.settle_decisions:
                angle = _to_float(info.get(INFO_ACTIVE_LAYER_ANGLE_DEG, 0.0))
                target = state.target_deg if state.target_deg is not None else angle
                if abs(angle - target) <= _ANGLE_TOL_DEG:
                    return self._sequence(state, [ids["release"], ids["retreat"]], then="wait")
                if state.retries >= 2:
                    return self._resync(state)  # slipped: re-approach and re-grasp
                state.retries += 1
                state.step = 1
                # Interrupt the finished turn with `wait` this decision: re-selecting
                # the same turn id would only CONTINUE (hold) the completed primitive.
                return ids["wait"]
            return self._reselect(state)
        if phase == "seq":
            return self._sequence(state, state.seq_steps, state.seq_then)
        return self._enter(state, "wait", ids["wait"])

    def _angle_rad(self, fields: dict) -> float:
        return float(fields["active_layer_angle"][0]) * math.pi

    def _turn_choice(self, fields: dict) -> int:
        """`turn90` vs `turn180`: whichever covers the remaining angle to the target."""
        angle = self._angle_rad(fields)
        if self.rotator.target_deg is not None:
            target = math.radians(self.rotator.target_deg)
        else:
            target = turn_target_rad(angle, self._move_quarter(fields))
        remaining = (target - angle) % (2 * math.pi)
        if remaining > math.pi:
            remaining -= 2 * math.pi
        return (
            ROTATOR_ID["turn90"]
            if abs(remaining) <= math.pi / 2 + _ANGLE_TOL_DEG / 2
            else ROTATOR_ID["turn180"]
        )

    # --- small FSM helpers ------------------------------------------------------------------

    def _enter(self, state: _HandState, phase: str, choice: int) -> int:
        if state.phase != phase:
            state.phase = phase
            state.in_phase = 0
            state.done_settle = 0
            state.retries = 0
            state.step = 0
        return int(choice)

    def _after_settle(self, state: _HandState, next_phase: str, next_choice: int) -> int:
        """Hold the running choice until it settled, then enter the next phase."""
        if state.done_settle >= self.settle_decisions:
            return self._enter(state, next_phase, next_choice)
        return self._reselect(state)

    def _reselect(self, state: _HandState) -> int:
        """Re-issue the last choice: re-selecting the running primitive continues it."""
        role = Role.HOLDER if state is self.holder else Role.ROTATOR
        return int(self._last[role])

    def _timed_out(self, state: _HandState) -> bool:
        return state.in_phase > _STATE_TIMEOUT

    def _resync(self, state: _HandState) -> int:
        """Escape hatch: release, retreat, then re-approach (recovers every injection)."""
        approach = (
            HOLDER_ID["approach_body"] if state is self.holder else ROTATOR_ID["approach_face"]
        )
        release = HOLDER_ID["release"] if state is self.holder else ROTATOR_ID["release"]
        retreat = HOLDER_ID["retreat"] if state is self.holder else ROTATOR_ID["retreat"]
        return self._sequence(state, [release, retreat, approach], then="approach")

    def _sequence(self, state: _HandState, steps: list, then: str) -> int:
        """Walk `steps` (`_SEQ_HOLD` decisions each), then enter phase `then`.

        Interrupting a retreat mid-flight is safe by construction: the approach phase
        that follows plans from the CURRENT state, so nothing yanks anywhere."""
        if state.phase != "seq":
            state.phase = "seq"
            state.in_phase = 0
            state.done_settle = 0
            state.retries = 0
            state.step = 0
            state.seq_steps = list(steps)
            state.seq_then = then
            return int(steps[0])
        if state.in_phase >= _SEQ_HOLD * (state.step + 1):
            state.step += 1
        if state.step >= len(state.seq_steps):
            return self._enter(state, state.seq_then, state.seq_steps[-1])
        return int(state.seq_steps[state.step])
