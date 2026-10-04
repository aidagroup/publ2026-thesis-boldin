"""Scripted two-arm expert for FaceTurn-v0: is the task physically feasible with two real arms?

The readiness check for step 1.5, and a worked example of a feasible strategy. There is no
teleporting or pinning of anything: both arms are driven only through their `pd_joint_pos`
controllers and the cube is only ever touched by the grippers.

The strategy (holder clamps the body, rotator clamps the face and rolls its wrist by 90 degrees,
four phases) is documented in `_face_turn_expert.py`, which `render_episode.py` shares.

Joint targets come from a small numpy IK of the URDF (`ArmModel`); the cube pose is read from
the sim (privileged). After every phase the script prints: whether the holder grasps the body,
whether the rotator grasps the face, the face angle, the body drift (position, rotation) and the
env's success flag. With several envs (GPU) floats are shown as min/mean/max and flags as
`count/num_envs`.

Run it on either backend (`--sim-backend cpu` is one env, also on macOS). Exit status 1 if
success is not reached in every env after the release.
"""

import math
import sys

from _face_turn_expert import ArmModel, Rig, run_face_turn
from _sim_utils import make_env, parse_args

import callosum.envs.face_turn  # noqa: F401  (registers FaceTurn-v0)
from callosum.envs.face_turn import TARGET_FACE_ANGLE


def main() -> None:
    # TODO(review): only verified on the CPU backend (one env). On the GPU backend the per-env
    # IK loops in Python (slow-ish for many envs) and contact behaviour may differ.
    args = parse_args(
        __doc__, lambda p: p.add_argument("--seed", type=int, default=0, help="Reset seed.")
    )
    env = make_env("FaceTurn-v0", args, control_mode="pd_joint_pos")
    env.reset(seed=args.seed)
    rig = Rig(env, ArmModel())
    rig.report("reset")

    final: dict = {}

    def on_phase(name: str) -> None:
        final.update(rig.report(name))

    run_face_turn(rig, on_phase)
    print(f"target face angle: {math.degrees(TARGET_FACE_ANGLE):.1f} deg")

    env.close()
    sys.exit(0 if final["success"].all() else 1)


if __name__ == "__main__":
    main()
