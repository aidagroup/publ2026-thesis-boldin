"""Assembly plans, the cube-orientation model and the lazy Kociemba wrapper.

Pure stdlib on purpose (the same macOS-testability rule as `_skills` and
`_partner_obs`): the plan/orientation bookkeeping must not depend on mani_skill or
torch. The env (task 2.x) consumes these to know, per plan move, which flips the
holder has to perform before the rotator's top-face turn is executable.

Vocabulary:

* A **plan move** (`PlanMove`) is a face turn of the puzzle: face in {U, D, L, R, F, B}
  (the cube's labeled faces, fixed to the BODY, i.e. the stickers) and quarter turns
  in {1, 2} -- a 90 or 180 degree turn of that face's outer layer. Counterclockwise
  turns do not exist as single moves because the layer joints span [0, 180] deg
  clockwise; a 270-deg (= prime) turn is expressed as 90 + 180 (two moves).
* An **orientation** (`CubeOrientation`) is the permutation that says which labeled
  face currently points along each world direction (up/down/front/back/right/left
  in the table frame). Only face directions are tracked: layer angles and sticker
  permutation live in the env's physics state, not here.
* A **flip** (`Flip`) is a whole-cube rotation about a horizontal world axis, in
  quarter turns -- what the holder's `flip90`/`flip180` primitives execute. A flip
  changes the orientation; a top-face plan move does not (a layer turns about the
  up axis, no face changes direction).
"""

import random
from dataclasses import dataclass

FACES = ("U", "D", "L", "R", "F", "B")
"""The six labeled faces (the task's fixed set order; registry-style stable order)."""

_WORLD_SLOTS = ("U", "D", "F", "B", "R", "L")
"""World directions a face can point along, in frame +z up, +y front, +x right:
U = up, D = down, F = front, B = back, R = right, L = left. The labels do double
duty (a face name and a world direction); the frame is the table's, not the body's."""

_FACE_VECTORS = {
    "U": (0, 0, 1),
    "D": (0, 0, -1),
    "F": (0, 1, 0),
    "B": (0, -1, 0),
    "R": (1, 0, 0),
    "L": (-1, 0, 0),
}
"""Unit vector of each label: a labeled face's body-frame normal at the upright
reference pose, and (same table) the world direction vector of each world slot."""

_VECTOR_TO_FACE = {vector: face for face, vector in _FACE_VECTORS.items()}


def _cross(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _quarter_turn(vector: tuple[int, int, int], axis: tuple[int, int, int]) -> tuple[int, int, int]:
    """Rotate `vector` by +90 deg about the unit `axis` (right-hand rule).

    Rodrigues at 90 deg reduces to `axis x vector + axis * (axis . vector)`, which is
    exact in integers for the +-x/+-y/+-z axes used here -- no floats, no drift.
    """
    dot = axis[0] * vector[0] + axis[1] * vector[1] + axis[2] * vector[2]
    crossed = _cross(axis, vector)
    return (
        crossed[0] + axis[0] * dot,
        crossed[1] + axis[1] * dot,
        crossed[2] + axis[2] * dot,
    )


# --- plan -------------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanMove:
    """One face turn of the plan: turn `face`'s outer layer by 90 (1) or 180 (2) deg."""

    face: str
    quarter_turns: int

    def __post_init__(self) -> None:
        if self.face not in FACES:
            raise ValueError(f"face must be one of {FACES}, got {self.face!r}")
        if self.quarter_turns not in (1, 2):
            raise ValueError(f"quarter_turns must be 1 or 2, got {self.quarter_turns}")

    def __str__(self) -> str:
        """Standard face-turn notation ("U" / "U2"; primes are never needed here)."""
        return self.face if self.quarter_turns == 1 else f"{self.face}2"


@dataclass(frozen=True)
class AssemblyPlan:
    """A sequence of plan moves given TO the environment as input (random for E2/E3,
    a Kociemba solution for E4); planning is not learned (thesis assumption)."""

    moves: tuple[PlanMove, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "moves", tuple(self.moves))  # accept lists, store a tuple
        for move in self.moves:
            if not isinstance(move, PlanMove):
                raise TypeError(f"plan entries must be PlanMove, got {move!r}")

    def __len__(self) -> int:
        return len(self.moves)

    def __iter__(self):
        return iter(self.moves)

    def __getitem__(self, index: int) -> PlanMove:
        return self.moves[index]

    def __str__(self) -> str:
        return " ".join(str(move) for move in self.moves)


def random_plan(length: int, seed: int) -> AssemblyPlan:
    """A uniformly random plan of `length` moves (E2: 2-3, E3: 5+).

    Each move picks a face and a quarter-turn count uniformly and independently (12
    options per move); consecutive moves on the same face are allowed and NOT merged
    (executing U then U2 sequentially is a legal 270-deg-equivalent turn). Negative
    lengths are rejected; length 0 is the empty (already solved) plan.
    """
    if length < 0:
        raise ValueError(f"length must be >= 0, got {length}")
    rng = random.Random(seed)
    moves = tuple(PlanMove(rng.choice(FACES), rng.choice((1, 2))) for _ in range(length))
    return AssemblyPlan(moves)


def parse_face_turn_sequence(text: str) -> AssemblyPlan:
    """Parse standard face-turn notation ("R U2 F'") into a plan.

    Token grammar: face + optional modifier, "" = clockwise 90, "2" = 180,
    "'" (prime) = counterclockwise. A prime expands into TWO moves (90 then 180),
    because the layer joints turn clockwise [0, 180] deg only. This is also what
    maps a Kociemba solution (which freely uses primes) onto executable moves.
    """
    moves: list[PlanMove] = []
    for token in text.split():
        face, modifier = token[0], token[1:]
        if face not in FACES or modifier not in ("", "2", "'"):
            raise ValueError(f"invalid face-turn token {token!r}")
        if modifier == "":
            moves.append(PlanMove(face, 1))
        elif modifier == "2":
            moves.append(PlanMove(face, 2))
        else:  # prime: -90 deg == +90 deg + 180 deg
            moves.extend((PlanMove(face, 1), PlanMove(face, 2)))
    return AssemblyPlan(tuple(moves))


# --- orientation ------------------------------------------------------------------------


@dataclass(frozen=True)
class Flip:
    """A whole-cube rotation about a horizontal world axis, in quarter turns.

    `axis` names the world direction the rotation axis points along: "R"/"L" are
    +-x, "F"/"B" are +-y; the rotation is +90 deg per quarter turn about that axis
    (right-hand rule), so "R" and "L" give opposite 90-deg flips and identical
    180-deg flips. `quarter_turns` 1 / 2 correspond to the holder's `flip90` /
    `flip180` primitives (a 180 flip is ONE flip180 execution, not two 90s).
    """

    axis: str
    quarter_turns: int

    def __post_init__(self) -> None:
        if self.axis not in ("R", "L", "F", "B"):
            raise ValueError(f"flip axis must be R, L, F or B, got {self.axis!r}")
        if self.quarter_turns not in (1, 2):
            raise ValueError(f"quarter_turns must be 1 or 2, got {self.quarter_turns}")


@dataclass(frozen=True)
class CubeOrientation:
    """Which labeled face points along each world direction.

    `slots[i]` is the face whose normal currently points along `_WORLD_SLOTS[i]`
    (up, down, front, back, right, left in the table frame). The orientation is a
    permutation of the six faces; layer angles and sticker state are elsewhere.
    """

    slots: tuple[str, str, str, str, str, str]

    @classmethod
    def upright(cls) -> "CubeOrientation":
        """The reference orientation: every labeled face on its own side."""
        return cls(_WORLD_SLOTS)

    def __post_init__(self) -> None:
        if len(self.slots) != 6 or set(self.slots) != set(FACES):
            raise ValueError(f"slots must be a permutation of {FACES}, got {self.slots}")

    def face_along(self, direction: str) -> str:
        """The labeled face pointing along the world `direction` (a slot label)."""
        return self.slots[_WORLD_SLOTS.index(direction)]

    @property
    def up(self) -> str:
        return self.face_along("U")

    @property
    def down(self) -> str:
        return self.face_along("D")

    def flipped(self, flip: Flip) -> "CubeOrientation":
        """The orientation after `flip`: a face that pointed along `d` now points
        along the rotated direction `R(d)`."""
        axis = _FACE_VECTORS[flip.axis]
        slots: list[str] = [""] * 6
        for slot, face in zip(_WORLD_SLOTS, self.slots, strict=True):
            vector = _FACE_VECTORS[slot]
            for _ in range(flip.quarter_turns):
                vector = _quarter_turn(vector, axis)
            slots[_WORLD_SLOTS.index(_VECTOR_TO_FACE[vector])] = face
        return CubeOrientation(tuple(slots))

    def flips_to_bring_up(self, face: str) -> tuple[tuple[Flip, ...], "CubeOrientation"]:
        """The minimal flip sequence that puts `face` on top, and the orientation
        after it.

        Every face reaches the top with AT MOST ONE flip about a horizontal axis:
        the top face needs none, a side face one 90-deg flip about the horizontal
        axis perpendicular to it, the bottom face one 180-deg flip (a single
        flip180, not two 90s). The search tries the no-flip case first, then the
        single flips in the fixed canonical order axes (R, L, F, B) x quarter turns
        (1, 2), so the answer is deterministic.
        """
        if face not in FACES:
            raise ValueError(f"face must be one of {FACES}, got {face!r}")
        if self.up == face:
            return ((), self)
        for axis in ("R", "L", "F", "B"):
            for quarter_turns in (1, 2):
                flip = Flip(axis, quarter_turns)
                turned = self.flipped(flip)
                if turned.up == face:
                    return ((flip,), turned)
        raise AssertionError(  # pragma: no cover - the geometry above is exhaustive
            f"no single flip brings {face} up from {self.slots}"
        )

    def after_move(self, move: PlanMove) -> "CubeOrientation":
        """The orientation after a plan move has been executed: unchanged, made
        explicit as a method so call sites state the invariant. A layer turn
        rotates material about the up axis; no face changes direction. (Which
        stickers end up where is sticker state, tracked by the env, not here.)"""
        return self


@dataclass(frozen=True)
class NextStep:
    """What has to happen before the next plan move is executable.

    `flips` is the (possibly empty) minimal sequence the holder must perform; after
    it, `orientation` has `move.face` on top and the rotator's turn can proceed.
    """

    move: PlanMove
    flips: tuple[Flip, ...]
    orientation: CubeOrientation


def next_step(plan: AssemblyPlan, index: int, orientation: CubeOrientation) -> NextStep:
    """The "next move" logic for plan position `index` under `orientation`.

    If the move's target face is not on top, the returned flips bring it up (and
    the returned orientation is the post-flip one); otherwise `flips` is empty and
    the orientation passes through. A bad `index` raises IndexError, like plan
    indexing itself.
    """
    move = plan[index]
    flips, oriented = orientation.flips_to_bring_up(move.face)
    return NextStep(move=move, flips=flips, orientation=oriented)


# --- kociemba wrapper -------------------------------------------------------------------


def solve_with_kociemba(facelets: str) -> AssemblyPlan:
    """Solve a cube state with the two-phase Kociemba solver into an executable plan.

    `facelets` is the solver's 54-character URFDLB facelet string (the convention of
    the `kociemba` package); the returned solution string is parsed with
    `parse_face_turn_sequence`, so primes expand into 90+180 move pairs. Building
    the facelet string FROM a scramble is E4 env work and stays out of this module.

    The import is deliberately lazy and the package is deliberately NOT a project
    dependency: E3/E4 are not the nearest goal and the module must import without
    it (it is only needed on the machine that generates solver-based plans).
    TODO(review): add kociemba to an extra and exercise this on E3/E4 hardware.
    """
    try:
        import kociemba
    except ImportError as exc:
        raise RuntimeError(
            "kociemba is not installed: solver-based (E3/E4) plans need it. Install it "
            "manually, e.g. `uv pip install kociemba` -- it is intentionally not a "
            "project dependency yet (see TODO(review) in _plans.solve_with_kociemba)"
        ) from exc
    return parse_face_turn_sequence(kociemba.solve(facelets))
