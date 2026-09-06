"""3x3x3 Rubik's cube state, moves and scrambling. Pure numpy, no simulator."""

import numpy as np

# Facelet indices: 9 per face, faces in this order.
FACES = ("U", "D", "F", "B", "L", "R")
FACE_INDEX = {f: i for i, f in enumerate(FACES)}
N_FACELETS = 54

# World-frame normal of each face for the cube resting on the table, +y toward
# the rotator. Used to decide which face the rotator is looking at.
FACE_NORMAL = {
    "U": np.array([0.0, 0.0, 1.0]),
    "D": np.array([0.0, 0.0, -1.0]),
    "F": np.array([0.0, 1.0, 0.0]),
    "B": np.array([0.0, -1.0, 0.0]),
    "L": np.array([-1.0, 0.0, 0.0]),
    "R": np.array([1.0, 0.0, 0.0]),
}

MOVES = tuple(f"{f}{s}" for f in FACES for s in ("", "'"))


def _cycle(perm: np.ndarray, items: list[int]) -> None:
    """In-place 4-cycle: items[0] <- items[3] <- items[2] <- items[1] <- items[0]."""
    a, b, c, d = items
    perm[a], perm[b], perm[c], perm[d] = perm[d], perm[a], perm[b], perm[c]


def _face(f: str) -> int:
    return FACE_INDEX[f] * 9


# Facelet cycles for a clockwise quarter turn of each face, viewed from
# outside that face. Each entry is a list of 4-cycles over global indices.
def _build_move_tables() -> dict[str, np.ndarray]:
    u, d, f, b, ll, r = (_face(x) for x in FACES)

    def ring(face_start: int) -> list[list[int]]:
        s = face_start
        return [[s + 0, s + 2, s + 8, s + 6], [s + 1, s + 5, s + 7, s + 3]]

    side = {
        "U": [
            [f + 0, ll + 0, b + 0, r + 0],
            [f + 1, ll + 1, b + 1, r + 1],
            [f + 2, ll + 2, b + 2, r + 2],
        ],
        "D": [
            [f + 6, r + 6, b + 6, ll + 6],
            [f + 7, r + 7, b + 7, ll + 7],
            [f + 8, r + 8, b + 8, ll + 8],
        ],
        "F": [
            [u + 6, r + 0, d + 2, ll + 8],
            [u + 7, r + 3, d + 1, ll + 5],
            [u + 8, r + 6, d + 0, ll + 2],
        ],
        "B": [
            [u + 2, ll + 0, d + 6, r + 8],
            [u + 1, ll + 3, d + 7, r + 5],
            [u + 0, ll + 6, d + 8, r + 2],
        ],
        "L": [
            [u + 0, f + 0, d + 0, b + 8],
            [u + 3, f + 3, d + 3, b + 5],
            [u + 6, f + 6, d + 6, b + 2],
        ],
        "R": [
            [u + 8, b + 0, d + 8, f + 8],
            [u + 5, b + 3, d + 5, f + 5],
            [u + 2, b + 6, d + 2, f + 2],
        ],
    }
    tables = {}
    for name in FACES:
        perm = np.arange(N_FACELETS)
        for cyc in ring(_face(name)) + side[name]:
            _cycle(perm, cyc)
        tables[name] = perm
        tables[name + "'"] = np.argsort(perm)
    return tables


MOVE_TABLES = _build_move_tables()

SOLVED = np.repeat(np.arange(6), 9)


class RubikState:
    """Facelet colours of one cube. `colours[i]` is the face colour at facelet i."""

    __slots__ = ("colours",)

    def __init__(self, colours: np.ndarray | None = None):
        self.colours = SOLVED.copy() if colours is None else np.asarray(colours).copy()

    def copy(self) -> "RubikState":
        return RubikState(self.colours)

    def apply(self, move: str) -> "RubikState":
        self.colours = self.colours[MOVE_TABLES[move]]
        return self

    def apply_all(self, moves) -> "RubikState":
        for m in moves:
            self.apply(m)
        return self

    @property
    def solved(self) -> bool:
        return bool(np.array_equal(self.colours, SOLVED))

    @property
    def solved_facelets(self) -> int:
        """How many facelets match their face's centre. 54 when solved."""
        centres = self.colours[4::9]
        return int((self.colours == np.repeat(centres, 9)).sum())

    def one_hot(self) -> np.ndarray:
        """(54, 6) float observation of the colour state."""
        out = np.zeros((N_FACELETS, 6), dtype=np.float32)
        out[np.arange(N_FACELETS), self.colours] = 1.0
        return out


def scramble(depth: int, rng: np.random.Generator) -> tuple[RubikState, list[str]]:
    """A state `depth` quarter turns from solved, avoiding immediate undo."""
    state, moves, last = RubikState(), [], ""
    for _ in range(depth):
        choices = [m for m in MOVES if m[0] != last]
        move = str(rng.choice(choices))
        state.apply(move)
        moves.append(move)
        last = move[0]
    return state, moves


def inverse(moves) -> list[str]:
    return [m[0] if m.endswith("'") else m + "'" for m in reversed(moves)]


def facing_face(cube_quat: np.ndarray, toward: np.ndarray) -> str:
    """Which face's outward normal points most along `toward`, given the cube's pose.

    `cube_quat` is (w, x, y, z). Used to decide which face the rotator's grip
    would actually turn, so a physical quarter turn maps to the right move.
    """
    w, x, y, z = cube_quat
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    toward = np.asarray(toward, dtype=float)
    toward = toward / (np.linalg.norm(toward) + 1e-12)
    best, score = "F", -np.inf
    for name, normal in FACE_NORMAL.items():
        s = float((rot @ normal) @ toward)
        if s > score:
            best, score = name, s
    return best
