"""What the SO-100 gripper can hold, measured from its own collision meshes.

Written on 2026-09-02, after the third round of re-deriving grasp geometry
from prose. `callosum/envs/_cube_geometry.py` carries a long comment claiming
a bare cube "is not graspable by this arm in the pose the task needs"; that
claim came from an analysis nobody could re-run. This is that analysis, as
code, so the next person can disagree with a number instead of a paragraph.

Everything is expressed in the **Fixed_Jaw frame**, read straight out of
`so100.urdf`: the tool points along -y, the jaws close along x, and the
gripper hinge is z. The collision geometry is four small pads (`*_part*.ply`),
prisms in z, so the whole question is two-dimensional in the x-y plane.

Two things come out of it, and the second is the one that decides the task:

  1. What the jaws can CLAMP -- how far the moving jaw must close on an object
     of a given width, and whether it fouls whatever sits behind it.
  2. What the jaws can TURN. The wrist_roll axis is the Fixed_Jaw frame's own
     y axis, i.e. the line x = z = 0, while the fixed blade's gripping face is
     the plane x = +0.0079. An object seats against that face, so its centre
     lands `width/2 - 0.0079` off the roll axis. Rolling the wrist therefore
     spins a narrow object about itself and swings a wide one around an arc.

    uv run python scripts/measure_gripper.py
    uv run python scripts/measure_gripper.py --mesh-dir path/to/so100/meshes
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Fixed_Jaw collision pad `Fixed_Jaw_part2.ply`, minimum x: the flat face the
# object is pressed against. Recomputed below; this is only the default for
# the report's arithmetic.
FACE_X = 0.0079
# The gripper joint, from so100.urdf: parent Fixed_Jaw, child Moving_Jaw.
JOINT_XYZ = np.array([-0.0202, -0.0244, 0.0])
JOINT_RPY = (0.0, 3.14159, -0.9)
JOINT_LIMIT = 1.1
# Interpenetration that counts as contact rather than as floating-point dust.
TOUCH = 2e-4


def _mesh_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    try:
        import mani_skill
    except ImportError:
        sys.exit(
            "mani_skill is not installed here (it is Linux+CUDA only) -- pass"
            " --mesh-dir, e.g. from an unpacked wheel"
        )
    return Path(mani_skill.__file__).parent / "assets/robots/so100/meshes"


def load_ply(path: Path) -> np.ndarray:
    """Vertices of a little binary PLY. These pads are 9-16 vertices each."""
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:end].decode("ascii", "replace")
    n = int(
        next(line for line in head.splitlines() if line.startswith("element vertex")).split()[-1]
    )
    k = sum(line.startswith("property float") for line in head.splitlines())
    return np.frombuffer(raw[end : end + n * 4 * k], dtype="<f4").reshape(n, k)[:, :3].astype(float)


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def hull2d(points: np.ndarray) -> np.ndarray:
    """Convex hull of the xy projection (monotone chain)."""
    p = np.unique(np.round(points[:, :2], 9), axis=0)
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(seq):
        out: list[np.ndarray] = []
        for point in seq:
            while len(out) > 1 and _cross(out[-1] - out[-2], point - out[-2]) <= 1e-12:
                out.pop()
            out.append(point)
        return out

    return np.array(half(p)[:-1] + half(p[::-1])[:-1])


def _normals(poly: np.ndarray) -> np.ndarray:
    edges = np.roll(poly, -1, 0) - poly
    n = np.stack([-edges[:, 1], edges[:, 0]], 1)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def penetration(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap depth of two convex polygons; 0.0 when they are apart (SAT)."""
    best = np.inf
    for axis in np.vstack([_normals(a), _normals(b)]):
        pa, pb = a @ axis, b @ axis
        depth = min(pa.max() - pb.min(), pb.max() - pa.min())
        if depth <= 0:
            return 0.0
        best = min(best, depth)
    return float(best)


def _rotz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    return (
        _rotz(yaw)
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )


def rect(x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


class Gripper:
    """The four collision pads, as 2-D polygons in the Fixed_Jaw frame."""

    def __init__(self, meshes: Path):
        self.palm = hull2d(load_ply(meshes / "Fixed_Jaw_part1.ply"))
        self.blade = hull2d(load_ply(meshes / "Fixed_Jaw_part2.ply"))
        self._moving = [load_ply(meshes / f"Moving_Jaw_part{i}.ply") for i in (1, 2, 3)]
        self._joint_rot = _rpy(*JOINT_RPY)
        self.face_x = float(self.blade[:, 0].min())
        self.blade_span = (float(self.blade[:, 1].min()), float(self.blade[:, 1].max()))

    def moving(self, qpos: float) -> list[np.ndarray]:
        rot = self._joint_rot @ _rotz(qpos)
        return [hull2d(p @ rot.T + JOINT_XYZ) for p in self._moving]

    def closes_on(self, box: np.ndarray) -> float | None:
        """The gripper qpos at which the moving jaw first touches `box`."""
        for qpos in np.arange(JOINT_LIMIT, -JOINT_LIMIT - 1e-9, -0.002):
            if any(penetration(p, box) > TOUCH for p in self.moving(qpos)):
                return float(qpos)
        return None


def report_clamp(grip: Gripper, width: float, depth: float, behind: float, label: str) -> None:
    """Sweep grasp depth: where can this object be clamped, and on what else."""
    print(f"\n{label}")
    print(f"  {width * 100:.1f} cm across the jaws, {depth * 100:.1f} cm deep")
    print(
        f"  {'front face y':>12} {'clamps at':>10} {'jaw on what is behind':>22} {'palm fouls':>11}"
    )
    clean = []
    for y0 in np.arange(-0.030, -0.1051, -0.005):
        obj = rect(grip.face_x - width, grip.face_x, y0 - depth, y0)
        body = rect(grip.face_x - width, grip.face_x, y0 - depth - behind, y0 - depth)
        qpos = grip.closes_on(obj)
        if qpos is None:
            print(f"  {y0:>12.4f} {'never':>10} {'-':>22} {'-':>11}")
            continue
        jaws = grip.moving(qpos)
        on_body = behind > 0 and any(penetration(p, body) > TOUCH for p in jaws)
        fouls = penetration(grip.palm, obj) > TOUCH
        if not on_body and not fouls:
            clean.append(y0)
        print(
            f"  {y0:>12.4f} {qpos:>10.3f} {'YES' if on_body else 'no':>22}"
            f" {'YES' if fouls else 'no':>11}"
        )
    print(f"  clean clamp at depth: {', '.join(f'{v:.3f}' for v in clean) if clean else 'NOWHERE'}")


def report_roll(grip: Gripper, widths: dict[str, float]) -> None:
    """How far off the wrist_roll axis each object's centre ends up.

    The roll axis IS the Fixed_Jaw frame's y axis, so this offset is the
    radius of the arc a wrist roll drags the object along. Turning something
    about its OWN axis needs that radius to be ~0.
    """
    print("\nturning: distance from the wrist_roll axis to the object's centre")
    print(f"  the fixed blade's gripping face is the plane x = {grip.face_x:+.4f}")
    print(f"  {'object':>34} {'width':>7} {'offset':>8} {'swing over a quarter turn':>27}")
    for label, width in widths.items():
        offset = abs(width / 2 - grip.face_x)
        swing = offset * np.sqrt(2)  # chord of a 90 deg arc
        print(
            f"  {label:>34} {width * 100:>5.1f} cm {offset * 100:>6.2f} cm {swing * 100:>24.2f} cm"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh-dir", default=None, help="SO-100 meshes/ (default: the installed one)")
    ap.add_argument("--cube", type=float, default=0.057, help="cube edge length, metres")
    args = ap.parse_args()

    grip = Gripper(_mesh_dir(args.mesh_dir))
    lo, hi = grip.blade_span
    print("SO-100 gripper, from the URDF collision pads (Fixed_Jaw frame, metres)")
    print(f"  fixed blade: gripping face x = {grip.face_x:+.4f}, spans y {lo:+.4f} .. {hi:+.4f}")
    print(f"  blade length {(hi - lo) * 100:.2f} cm, gripper travel +-{JOINT_LIMIT} rad")

    layer = args.cube / 3
    report_clamp(grip, 0.020, 0.035, 0.0, "the 2 cm grasp handle we build today")
    report_clamp(grip, args.cube, layer, args.cube - layer, "a bare Rubik's layer")
    report_clamp(grip, args.cube, args.cube, 0.0, "the whole bare cube")

    report_roll(
        grip,
        {
            "the 2 cm grasp handle": 0.020,
            "a bare Rubik's layer": args.cube,
            "widest object still on the roll axis": 2 * grip.face_x,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
