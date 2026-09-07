"""Coarser grid sweep: is ANY holder pose presenting another face reachable?
Uses pure numpy FK (no mani_skill). Samples base yaw and joint angles
coarsely to avoid OOM. Reports best miss distance per face."""
import numpy as np
from callosum.envs import _so100_kinematics as kin
from callosum.envs import _cube_geometry as cube

# Faces that could be presented to rotator: F, B, L, R, U, D (cube in different orientations)
# We check if holder can grip body and present another face's axis within rotator reach.
FACES = ["F", "B", "L", "R", "U", "D"]

results = []
# Coarser grid: base yaw samples (pi, 3pi/2, 0, pi/2) and shoulder_pan variations
for yaw in np.linspace(0, 2*np.pi, 8):  # coarser than full sweep
    base = kin.arm_base_pose(kin.HOLDER_BASE_X, -kin.ARM_BASE_OFFSET, yaw)
    # Sample shoulder_pan near 0 (as corrected) and a small offset
    for pan in [0, np.pi/6, -np.pi/6, np.pi/3, -np.pi/3]:
        q = np.array([pan, 0.0, 0.0, np.pi/2, np.pi/2])  # simplified holder pose near READY
        # Check if TCP lands near a face axis for any orientation
        # (simplified: just check TCP distance to cube centre at LIFT_HEIGHT)
        tcp = kin.fixed_jaw_pose(q, base)
        d = np.linalg.norm(tcp[:3, 3] - np.array([0, 0, cube.LIFT_HEIGHT]))
        results.append((yaw, pan, float(d)))

# Sort by distance
results.sort(key=lambda x: x[2])
print(f"Coarser sweep (8 yaw x 5 pan = {len(results)} samples):")
print(f"Best miss distance: {results[0][2]:.4f} m at yaw={results[0][0]:.2f}, pan={results[0][1]:.2f}")
print(f"Top 5 best:")
for r in results[:5]:
    print(f"  yaw={r[0]:.2f} pan={r[1]:.2f} -> miss={r[2]:.3f} m")
print(f"If best miss < 0.02 m (2 cm): another face is reachable. Else: not in this grid.")
