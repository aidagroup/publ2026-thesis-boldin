"""Re-derive the scripted joint waypoints from the cube's geometry.

This is where the numbers in `callosum.envs._so100_kinematics` come from.
ManiSkill has no IK on the GPU backend (`Articulation.create_pinocchio_model`
raises when GPU sim is enabled), so the arms are driven through precomputed
JOINT waypoints -- and three server round trips were lost to waypoints that
had quietly stopped matching the scene. tests/test_grasp_waypoints.py stops
them drifting; this stops them being unreproducible.

Pure numpy, no simulator, so it runs on the dev machine.

    uv run python scripts/solve_waypoints.py
    uv run python scripts/solve_waypoints.py --scan     # the reach envelope

The `--scan` report is the one that set the layout. Sweeping the whole joint
range, it asks where each arm can put the jaws' pocket while holding its tool
HORIZONTAL -- which the task requires, because a face turn is a wrist roll
about the cube's axis and that axis is horizontal. The answer is why
`ARM_BASE_OFFSET` had to grow from 0.28 to 0.34: pointing flat costs reach at
the near end, and at 0.28 the holder could not reach the cube on the table at
all.
"""

import argparse

import numpy as np

from callosum.envs import _cube_geometry as cube
from callosum.envs import _so100_kinematics as kin

TOWARD_HOLDER = np.array([0.0, -1.0, 0.0])
TOWARD_ROTATOR = np.array([0.0, 1.0, 0.0])


def batch_fixed_jaw(qpos: np.ndarray, base: np.ndarray) -> np.ndarray:
    """`fixed_jaw_pose` over a batch of arm configurations. (N,5) -> (N,4,4)."""
    out = np.broadcast_to(np.asarray(base, float), (len(qpos), 4, 4)).copy()
    for i, (xyz, rpy, axis) in enumerate(kin._ARM_CHAIN):
        static = np.eye(4)
        static[:3, :3] = kin._rpy(*rpy)
        static[:3, 3] = xyz
        a = np.asarray(axis, float)
        skew = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        theta = qpos[:, i]
        rot = (
            np.eye(3)
            + np.sin(theta)[:, None, None] * skew
            + (1 - np.cos(theta))[:, None, None] * (skew @ skew)
        )
        joint = np.zeros((len(qpos), 4, 4))
        joint[:, :3, :3] = rot
        joint[:, 3, 3] = 1.0
        out = out @ static @ joint
    return out


def _cost(qpos, base, target, tool, half_width):
    poses = batch_fixed_jaw(qpos, base)
    pocket = poses[:, :3, 3] + np.einsum(
        "nij,j->ni", poses[:, :3, :3], kin.pocket_offset(half_width)
    )
    # Position in metres, orientation weighted so a degree of tilt trades
    # against about a millimetre of miss -- the grip absorbs a millimetre far
    # more happily than it absorbs a tilted roll axis.
    return np.linalg.norm(pocket - target, axis=1) + 0.10 * np.linalg.norm(
        -poses[:, :3, 1] - tool, axis=1
    )


def solve(base, target, tool, half_width, grid=81, iters=400):
    """Coarse grid over the three pitch joints, then Gauss-Newton on all five.

    Multi-start rather than a single seed: the three parallel pitch joints
    give this arm elbow-up and elbow-down branches, and only one of them keeps
    the tool horizontal at the near end of the workspace.
    """
    limits = kin.JOINT_LIMITS[:5]
    axes = [np.linspace(lo, hi, grid) for lo, hi in limits[1:4]]
    lift, elbow, flex = np.meshgrid(*axes, indexing="ij")
    qpos = np.stack(
        [
            np.zeros(lift.size),
            lift.ravel(),
            elbow.ravel(),
            flex.ravel(),
            np.full(lift.size, np.pi / 2),
        ],
        axis=1,
    )
    qpos = qpos[np.argsort(_cost(qpos, base, target, tool, half_width))[:30]]
    for _ in range(iters):
        cost = _cost(qpos, base, target, tool, half_width)
        jac = np.zeros((len(qpos), 5))
        for i in range(5):
            step = np.zeros((1, 5))
            step[0, i] = 1e-6
            jac[:, i] = (_cost(qpos + step, base, target, tool, half_width) - cost) / 1e-6
        delta = -(jac * cost[:, None]) / (np.sum(jac * jac, axis=1) + 1e-9)[:, None]
        qpos = np.clip(qpos + np.clip(delta, -0.05, 0.05), limits[:, 0], limits[:, 1])
    best = qpos[np.argmin(_cost(qpos, base, target, tool, half_width))]
    pose = batch_fixed_jaw(best[None], base)[0]
    pocket = pose[:3, 3] + pose[:3, :3] @ kin.pocket_offset(half_width)
    tilt = np.degrees(np.arccos(np.clip((-pose[:3, 1]) @ tool, -1.0, 1.0)))
    return best, float(np.linalg.norm(pocket - target)), float(tilt)


def _plan():
    """(name, base pose, pocket target, tool direction, object half-width)."""
    nub_y = cube.CUBE_HALF_SIZE + cube.NUB_GRASP_DEPTH
    body_y = cube.BODY_GRASP_OFFSET[1]
    # The holder's pocket sits on the cube's axis; its TOOL axis is 2.16 cm to
    # the side, which is what the base's x offset is there to line up with.
    grasp_z = cube.CUBE_HALF_SIZE + cube.BODY_GRASP_OFFSET[2]
    lift_z = cube.LIFT_HEIGHT + cube.BODY_GRASP_OFFSET[2]
    rot, hold = kin.ROTATOR_BASE_POSE, kin.HOLDER_BASE_POSE
    nub, body = cube.NUB_HALF_WIDTH, cube.CUBE_HALF_SIZE
    return [
        ("HOLDER_PREGRASP", hold, [0.0, body_y, grasp_z + kin.PREGRASP_LIFT], TOWARD_ROTATOR, body),
        ("HOLDER_GRASP", hold, [0.0, body_y, grasp_z], TOWARD_ROTATOR, body),
        ("HOLDER_LIFT", hold, [0.0, body_y, lift_z], TOWARD_ROTATOR, body),
        (
            "ROTATOR_PREGRASP",
            rot,
            [0.0, nub_y, cube.LIFT_HEIGHT + kin.PREGRASP_LIFT],
            TOWARD_HOLDER,
            nub,
        ),
        ("ROTATOR_GRASP", rot, [0.0, nub_y, cube.LIFT_HEIGHT], TOWARD_HOLDER, nub),
    ]


def report_waypoints() -> None:
    stored = {
        "HOLDER_PREGRASP": kin.HOLDER_PREGRASP,
        "HOLDER_GRASP": kin.HOLDER_GRASP,
        "HOLDER_LIFT": kin.HOLDER_LIFT,
        "ROTATOR_PREGRASP": kin.ROTATOR_PREGRASP,
        "ROTATOR_GRASP": kin.ROTATOR_GRASP,
    }
    print(f"solved against ARM_BASE_OFFSET = {kin.ARM_BASE_OFFSET}")
    print(f"  {'waypoint':>17} {'err':>9} {'tilt':>9} {'vs stored':>11}  qpos")
    for name, base, target, tool, half_width in _plan():
        qpos, err, tilt = solve(base, np.asarray(target, float), tool, half_width)
        drift = float(np.max(np.abs(qpos - stored[name])))
        flag = "ok" if err < 1.5e-3 and tilt < 0.5 else "!!"
        joints = ", ".join(f"{v:.4f}" for v in qpos)
        print(
            f"{flag} {name:>17} {err * 1000:>6.2f} mm {tilt:>6.2f} deg"
            f" {drift:>10.4f}  np.array([{joints}])"
        )
    print("\n  'vs stored' is the largest per-joint disagreement with")
    print("  callosum.envs._so100_kinematics. Anything above ~0.01 rad means the")
    print("  geometry moved and the stored waypoints have not caught up.")


def report_scan(grid: int = 181) -> None:
    """Where can each arm put the pocket while keeping its tool horizontal?"""
    limits = kin.JOINT_LIMITS[:5]
    axes = [np.linspace(lo, hi, grid) for lo, hi in limits[1:4]]
    lift, elbow, flex = np.meshgrid(*axes, indexing="ij")
    qpos = np.stack(
        [
            np.zeros(lift.size),
            lift.ravel(),
            elbow.ravel(),
            flex.ravel(),
            np.full(lift.size, np.pi / 2),
        ],
        axis=1,
    )
    for label, base, tool, half_width in (
        ("ROTATOR", kin.ROTATOR_BASE_POSE, TOWARD_HOLDER, cube.NUB_HALF_WIDTH),
        ("HOLDER", kin.HOLDER_BASE_POSE, TOWARD_ROTATOR, cube.CUBE_HALF_SIZE),
    ):
        poses = batch_fixed_jaw(qpos, base)
        tilt = np.degrees(np.arccos(np.clip(-poses[:, :3, 1] @ tool, -1.0, 1.0)))
        keep = tilt < 2.0
        pocket = (
            poses[:, :3, 3]
            + np.einsum("nij,j->ni", poses[:, :3, :3], kin.pocket_offset(half_width))[:, :]
        )
        reach = np.abs(pocket[keep, 1] - base[1, 3])
        print(f"\n{label}: {keep.sum()} of {keep.size} poses hold the tool within 2 deg of flat")
        print(f"  distance from its own base along y: {reach.min():.3f} .. {reach.max():.3f} m")
        for lo, hi in ((0.02, 0.04), (0.04, 0.06), (0.06, 0.08), (0.08, 0.10), (0.10, 0.14)):
            band = keep.copy()
            band[keep] = (pocket[keep, 2] >= lo) & (pocket[keep, 2] < hi)
            if band.any():
                d = np.abs(pocket[band, 1] - base[1, 3])
                print(f"    pocket z {lo:.2f}-{hi:.2f} m: reaches {d.min():.3f} .. {d.max():.3f} m")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan", action="store_true", help="print the reach envelope instead")
    args = ap.parse_args()
    if args.scan:
        report_scan()
    else:
        report_waypoints()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
