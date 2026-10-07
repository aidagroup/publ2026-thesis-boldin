"""The scripted FaceTurn program as per-arm lists of phases (pure numpy, unit-testable).

The strategy is the one described in `callosum.experts.face_turn_expert`; here it is written as the
two arms' programs of `callosum.experts._phases` (one primitive per phase, ordering between the
arms as gates) over an `ExpertPlan`, the per-env joint waypoints the expert planned with IK.

Holder (agent_a):  `holder pregrasp` -> `holder grasp` -> `holder close`
                   (-> `holder release` after the rotator's half turn, `release_holder` only)
Rotator (agent_b): `rotator pregrasp` -> `rotator descent` -> `rotator close`
                   (-> `half turn`) -> `turn` -> `release`

Ordering constraints, all per env:

* overlap (`ExpertControlConfig.overlap_approach`): the rotator flies to its pre-grasp pose (4 cm
  above the face layer, clear of the holder) together with the holder, and starts its descent
  only once THIS env's holder has started closing its jaws (`started("holder close")`).
  Without overlap the rotator starts only after the holder's jaws have closed.
* the turn needs this env's rotator grasp and holder grasp both completed.
* with `release_holder` the holder opens after the half turn and the full turn waits for that.
"""

from dataclasses import dataclass

import numpy as np

from callosum.configs.face_turn_expert import ExpertControlConfig
from callosum.experts._phases import (
    Context,
    Grip,
    PhaseSpec,
    Planner,
    TrackPath,
    after,
    all_of,
    started,
)

SETTLE_STEPS = 10  # settle after a movement phase (the delta mode waits less, see the config)
GRIPPER_OPEN, GRIPPER_CLOSED = 1.0, -1.0  # normalised absolute gripper action
CLOSE_STEPS = 20  # a jaw close / open phase ends at the latest after this many steps
HOLDER_RELEASE_STEPS = 10  # `release_holder`: the holder's jaws open within this many steps
HALF_TURN_SETTLE = 30  # `release_holder`: settle after the half turn (capped by `settle_cap`)

HOLDER_PREGRASP = "holder pregrasp"
HOLDER_GRASP = "holder grasp"
HOLDER_CLOSE = "holder close"
HOLDER_RELEASE = "holder release"
ROTATOR_PREGRASP = "rotator pregrasp"
ROTATOR_DESCENT = "rotator descent"
ROTATOR_CLOSE = "rotator close"
HALF_TURN = "half turn"
TURN = "turn"
RELEASE = "release"


@dataclass
class ExpertPlan:
    """Joint waypoints planned per env, `(n, W, 5)` each (`(n, 1, 5)` for the single targets).

    Everything depends only on the cube pose at reset (the paths chain: each starts from the end
    of the previous one), so the whole plan is computed once per reset. `ok` is False for envs
    where the IK found no valid solution (those envs hold and never succeed).
    """

    pregrasp_a: np.ndarray
    grasp_a: np.ndarray
    pregrasp_b: np.ndarray
    grasp_b: np.ndarray
    half_turn: np.ndarray
    turned: np.ndarray
    ok: np.ndarray


def lookup(waypoints: np.ndarray, ok: np.ndarray) -> Planner:
    """A `Planner` that returns the precomputed `(n, W, 5)` waypoints of the envs entering."""
    return lambda envs: (waypoints[envs], ok[envs])


def _face_turned(ctx: Context) -> np.ndarray:
    return ctx.obs.face_turned


def build_programs(
    plan: ExpertPlan,
    cfg: ExpertControlConfig,
    *,
    no_holder: bool = False,
    release_holder: bool = False,
) -> list[list[PhaseSpec]]:
    """The holder's and the rotator's programs `[holder, rotator]` over the plan (see the module
    doc). `no_holder` leaves the holder with an empty program (the rotator turns alone)."""
    n = len(plan.ok)
    overlap = cfg.overlap_approach and not no_holder
    holder: list[PhaseSpec] = []
    if not no_holder:
        holder = [
            PhaseSpec(
                TrackPath(
                    n,
                    HOLDER_PREGRASP,
                    lookup(plan.pregrasp_a, plan.ok),
                    fly_by=overlap,
                    settle=SETTLE_STEPS,
                )
            ),
            PhaseSpec(
                TrackPath(
                    n,
                    HOLDER_GRASP,
                    lookup(plan.grasp_a, plan.ok),
                    settle=0 if overlap else SETTLE_STEPS,
                )
            ),
            PhaseSpec(Grip(n, HOLDER_CLOSE, GRIPPER_CLOSED, CLOSE_STEPS)),
        ]
        if release_holder:
            holder.append(
                PhaseSpec(
                    Grip(n, HOLDER_RELEASE, GRIPPER_OPEN, HOLDER_RELEASE_STEPS),
                    gate=after(HALF_TURN),
                )
            )

    # The holder goes first: while its jaws are open next to the cube they would hit the
    # rotator's open jaws, so the rotator only comes down to the face layer once this env's
    # holder is at its grasp pose and closing.
    pregrasp_gate = None if (overlap or no_holder) else after(HOLDER_CLOSE)
    descent_gate = None if no_holder else started(HOLDER_CLOSE)
    turn_gate = (
        after(ROTATOR_CLOSE) if no_holder else all_of(after(ROTATOR_CLOSE), after(HOLDER_CLOSE))
    )
    rotator = [
        PhaseSpec(
            TrackPath(
                n,
                ROTATOR_PREGRASP,
                lookup(plan.pregrasp_b, plan.ok),
                fly_by=overlap,
                settle=SETTLE_STEPS,
            ),
            gate=pregrasp_gate,
        ),
        PhaseSpec(
            TrackPath(n, ROTATOR_DESCENT, lookup(plan.grasp_b, plan.ok), settle=SETTLE_STEPS),
            gate=descent_gate,
        ),
        PhaseSpec(Grip(n, ROTATOR_CLOSE, GRIPPER_CLOSED, CLOSE_STEPS)),
    ]
    turn_after = turn_gate
    if release_holder and not no_holder:
        # Half the roll, then the holder lets go and the rotator keeps rolling: with the lock the
        # face must stop where it was when the holder released.
        rotator.append(
            PhaseSpec(
                TrackPath(n, HALF_TURN, lookup(plan.half_turn, plan.ok), settle=HALF_TURN_SETTLE),
                gate=turn_gate,
            )
        )
        turn_after = after(HOLDER_RELEASE)
    # The turn ends as soon as THIS env's face is at its target: the wrist target overshoots the
    # face's limit on purpose, so the wrist's own error would never get within tolerance.
    rotator += [
        PhaseSpec(
            TrackPath(n, TURN, lookup(plan.turned, plan.ok), until=_face_turned, hold_on_done=True),
            gate=turn_after,
        ),
        PhaseSpec(Grip(n, RELEASE, GRIPPER_OPEN, CLOSE_STEPS)),
    ]
    return [holder, rotator]
