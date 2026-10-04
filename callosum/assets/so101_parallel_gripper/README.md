# SO-ARM101 + Robonine parallel gripper (simulation assets)

Robot description used by `callosum.robots.so101_parallel_gripper` (ManiSkill agent
`so101_pg`). Simulation only: there is no real hardware in this project.

## Source

- Repository: <https://github.com/roboninecom/SO-ARM100-101-Parallel-Gripper>
- Commit: `305ad0f6e8f19e4e739616160cbdc7cae1ab153f`
- Files taken: `simulation/so_arm_101_description/urdf/so_101.urdf.xacro` and
  `simulation/so_arm_101_description/meshes/{visual,collision}/*.stl`.

## Modifications

The files here are modified versions of the upstream files:

- `so101_parallel_gripper.urdf`: the upstream xacro flattened into plain URDF.
  - ROS-only parts removed (xacro args/includes, mujoco tag, world link, safety_controller);
    `base_link` is the root.
  - Mesh paths made relative.
  - `clamp_1`/`clamp_2` collision: the upstream convex hulls fill the gap between the jaws,
    so each clamp uses two box collision shapes (fingertip pad + finger body), computed from
    the visual mesh, instead. The gear rack (inside the gripper housing) has no collision.
  - Added fixed `clamp_1_pad` / `clamp_2_pad` frames at the centre of each jaw's inner face
    (used for the TCP and grasp checks).
- `meshes/visual/*.stl`: decimated about 10x (fewer triangles, same shape).
- `meshes/collision/*.stl`: the upstream convex hulls of the arm links, unchanged. Hulls for
  `clamp_1`/`clamp_2` are intentionally not shipped (see above).

## Licences

Every file is covered by exactly one licence, assigned per file as upstream does
(see upstream `LICENSING.md`, `NOTICE` and `simulation/so_arm_101_description/meshes/LICENSE.md`).
The full texts are in `LICENSE-Apache-2.0.txt` and `LICENSE-CERN-OHL-P-2.0.txt` (copied from
the upstream repository at the commit above).

| Files (visual and collision meshes) | Licence | Copyright |
|---|---|---|
| `clamp_1.stl`, `clamp_2.stl` | CERN-OHL-P-2.0 | (c) 2025 Robonine |
| `base_link.stl` | Apache-2.0 | TheRobotStudio (SO-ARM100 project, <https://github.com/TheRobotStudio/SO-ARM100>) |
| `link1_1.stl`, `link2_1.stl`, `link3_1.stl`, `link4_1.stl` | Apache-2.0 | TheRobotStudio (SO-ARM100 project) |
| `link5_1.stl` | Apache-2.0 | fused mesh: upstream SO-ARM101 wrist + Robonine main frame, Apache-2.0 in its entirety |
| `so101_parallel_gripper.urdf` | Apache-2.0 | derived from Robonine's Apache-2.0 robot description (software) |

Recommended CERN-OHL-P-2.0 notice for the clamp meshes:

```
Copyright (c) 2025 Robonine

This source describes Open Hardware and is licensed under the CERN-OHL-P v2.
You may redistribute and modify this source and make products using it under
the terms of the CERN-OHL-P v2 (https://ohwr.org/cern_ohl_p_v2.txt).

This source is distributed WITHOUT ANY EXPRESS OR IMPLIED WARRANTY, INCLUDING
OF MERCHANTABILITY, SATISFACTORY QUALITY AND FITNESS FOR A PARTICULAR PURPOSE.
```

The decimated meshes are modified copies; they keep the licence of the file they derive from.
