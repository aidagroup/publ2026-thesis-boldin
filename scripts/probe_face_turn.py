"""Scripted two-arm expert for FaceTurn-v0: is the task physically feasible with two real arms?

The readiness check for step 1.5, and a worked example of a feasible strategy. There is no
teleporting or pinning of anything: both arms are driven only through their joint controllers
and the cube is only ever touched by the grippers. The strategy itself lives in
`callosum.experts.face_turn_expert` (shared with `scripts/collect_demos.py`); `--control delta`
runs it under the training controller (`pd_joint_delta_pos`) instead of `pd_joint_pos`.

Strategy (arms stand 90 degrees apart around the cube, see `callosum.configs.layout`):

  1. `holder pregrasp`, `holder grasp`: the holder (agent_a) comes in low and almost horizontal,
     fingers pointing at the cube 10 degrees below horizontal and jaws open horizontally across
     the approach, and clamps the body (the lower layer, below the face) from its sides. It grips
     2 cm in front of the body centre so that its wrist housing stays clear of the rotator.
  2. `rotator pregrasp`, `rotator grasp`: only then does the rotator (agent_b) come in top-down
     (fingers 6 degrees off vertical), lowers its open jaws around the face layer and clamps it
     from its sides. The two arms must not have their open jaws near the cube at the same time.
  3. `turn`: the rotator rolls its wrist by -90 degrees (a bit more, so the face joint's limit
     stops the face at exactly 90 degrees); the face turns by +90 degrees about the vertical axis
     while the holder keeps the body still.
  4. `release`: the rotator opens its jaws; success is evaluated again.

Joint targets come from a small numpy IK of the URDF (`ArmModel`); the cube pose is read from
the sim (privileged). After every phase the script prints: the env step count, the dense reward
(raw and normalised; the sanity check that the shaping rewards the phases in order), whether the
holder grasps the body, whether the rotator grasps the face, the face angle, the body drift
(position, rotation) and the env's success flag. With several envs (GPU) floats are shown as min/mean/max and flags as
`count/num_envs`.

Face-lock checks (the rule "the face can only be turned while the holder grasps the body",
`FaceTurnPhysicsConfig.lock_face_unless_held`): `--no-holder` skips the holder phases (rotator
alone), `--release-holder` makes the holder let go half way through the turn, `--no-lock`
disables the rule (for the record: what the rotator alone can do without it), and
`--face-friction` / `--face-damping` override the face joint's resistance. Every report line also
prints `face_lock_steps`, the control steps each env spent locked, which on the GPU shows that the
lock acts per env.

With `--control delta` every report also prints `phase_steps` (the env steps of that phase),
`reaches` (how the phase's closed-loop reaches ended: tolerance, stall or timeout, and the worst
remaining joint error, see `callosum.experts._tracking`) and, at the end, the step count of each
phase.

Run it on either backend (`--sim-backend cpu` is one env, also on macOS). Exit status: in the
default mode 1 if success is not reached in every env after the release; with `--no-holder` or
`--release-holder` and the lock on, 1 if any env reaches success or its face turns past 5 degrees
(`--release-holder` allows the half turn done before the release, 45 degrees plus 5); with
`--no-lock` it is always 0.
"""

import math
import sys

import numpy as np
import torch
from _sim_utils import make_env, parse_args
from mani_skill.utils import common

import callosum.envs.face_turn  # noqa: F401  (registers FaceTurn-v0)
from callosum.configs.face_turn import FaceTurnPhysicsConfig
from callosum.configs.face_turn_expert import CONTROL_MODES, ExpertControlConfig
from callosum.envs.face_turn import TARGET_FACE_ANGLE
from callosum.experts._tracking import summarize_reaches
from callosum.experts.face_turn_expert import ArmModel, Rig, run_expert


def report(rig: Rig, phase: str, steps_before: int = 0) -> dict:
    """Print the per-phase status line and return it as a dict of numpy values.

    `steps_before` is the env step count at the end of the previous phase, so that the report
    shows the length of this phase (`phase_steps`) next to the running count (`env_steps`).
    """
    b = rig.base
    info = b.evaluate()
    body = b.cube.links_map["body"]
    face = b.face_link
    pos_drift = torch.linalg.norm(b.cube.pose.p - b.body_init_pos, dim=1)
    rot_drift = common.quat_diff_rad(b.cube.pose.q, b.body_init_q)
    reward = b.compute_dense_reward(obs=None, action=None, info=info)
    # The face-angle term alone (normalised): weight * hard-gate(holder grasp) * progress,
    # to check that it drops to 0 once the holder lets go.
    cfg = b.reward_config
    holder_grasp = rig.agents[0].is_grasping(body).float()
    remaining = (TARGET_FACE_ANGLE - info["face_angle"]).clamp(min=0)
    progress = (1 - remaining / TARGET_FACE_ANGLE).clamp(0, 1)
    angle_term = (
        cfg.weight_angle_progress
        * cfg.order_gate(cfg.angle_gate_floor, holder_grasp)
        * progress
        / cfg.max_positive_reward
    )
    row = {
        "env_steps": np.array([int(b.elapsed_steps[0])]),
        "dense_reward": reward.cpu().numpy(),
        "normalized_reward": (reward / b.reward_config.max_positive_reward).cpu().numpy(),
        "angle_term_norm": angle_term.cpu().numpy(),
        "holder_grasps_body": rig.agents[0].is_grasping(body).cpu().numpy(),
        "rotator_grasps_face": rig.agents[1].is_grasping(face).cpu().numpy(),
        "face_angle_deg": np.rad2deg(info["face_angle"].cpu().numpy()),
        "body_drift_cm": pos_drift.cpu().numpy() * 100,
        "body_rot_deg": np.rad2deg(rot_drift.cpu().numpy()),
        "success": info["success"].cpu().numpy(),
        "face_lock_steps": b.face_lock_engaged_steps.cpu().numpy(),
    }
    if rig.control.control == "delta":  # the pos-mode output stays as it always was
        steps = {"phase_steps": np.array([int(b.elapsed_steps[0]) - steps_before])}
        row = {"env_steps": row.pop("env_steps"), **steps, **row}
    print(f"[{phase}]")
    for key, val in row.items():
        if key in ("env_steps", "phase_steps"):
            shown = str(int(val[0]))
        elif key == "face_lock_steps":
            shown = (
                str(int(val[0]))
                if len(val) == 1
                else f"{val.min()} / {val.mean():.1f} / {val.max()}"
            )
        elif val.dtype.kind == "b":
            shown = str(bool(val[0])) if len(val) == 1 else f"{int(val.sum())}/{len(val)}"
        elif len(val) == 1:
            shown = f"{val[0]:.2f}"
        else:
            shown = f"{val.min():.2f} / {val.mean():.2f} / {val.max():.2f}"
        print(f"  {key:>20}: {shown}")
    if rig.control.control == "delta":
        # How the closed-loop reaches of this phase ended (tolerance / stall / timeout): the
        # timeouts are where the steps go if a phase is slower than its joint distances allow.
        print(f"  {'reaches':>20}: {summarize_reaches(rig.reach_log)}")
        rig.reach_log.clear()
    return row


def _add_args(parser) -> None:
    parser.add_argument("--seed", type=int, default=0, help="Reset seed.")
    parser.add_argument(
        "--control",
        choices=CONTROL_MODES,
        default="pos",
        help='"pos": pd_joint_pos, open loop (default); "delta": the training controller '
        "pd_joint_delta_pos, tracked closed loop.",
    )
    parser.add_argument(
        "--overlap",
        action="store_true",
        help="Overlap the approach phases (ExpertControlConfig.overlap_approach): the rotator "
        "flies to its pre-grasp pose with the holder and descends while the holder's jaws close.",
    )
    parser.add_argument(
        "--no-holder", action="store_true", help="Skip the holder phases: the rotator turns alone."
    )
    parser.add_argument(
        "--release-holder",
        action="store_true",
        help="The holder opens its jaws half way through the turn.",
    )
    parser.add_argument(
        "--no-lock", action="store_true", help="Disable the face lock (lock_face_unless_held)."
    )
    defaults = FaceTurnPhysicsConfig()
    parser.add_argument("--face-friction", type=float, default=defaults.face_friction)
    parser.add_argument("--face-damping", type=float, default=defaults.face_damping)


def main() -> None:
    # TODO(review): only verified on the CPU backend (one env). On the GPU backend the per-env
    # IK in the expert loops in Python (slow-ish for many envs) and contact behaviour may differ;
    # the face lock's GPU path (batched fetch/clamp/apply) is unverified: check `face_lock_steps`.
    args = parse_args(__doc__, _add_args)
    physics = FaceTurnPhysicsConfig(
        lock_face_unless_held=not args.no_lock,
        face_friction=args.face_friction,
        face_damping=args.face_damping,
    )
    control = ExpertControlConfig(control=args.control, overlap_approach=args.overlap)
    mode = "pd_joint_pos" if control.control == "pos" else "pd_joint_delta_pos"
    env = make_env("FaceTurn-v0", args, control_mode=mode, physics_config=physics)
    obs, _ = env.reset(seed=args.seed)
    rig = Rig(env, ArmModel(), control, obs=obs)
    report(rig, "reset")

    final: dict = {}
    phase_ends: list[tuple[str, int]] = []

    def on_phase(name: str) -> None:
        before = phase_ends[-1][1] if phase_ends else 0
        final.update(report(rig, name, before))
        phase_ends.append((name, int(rig.base.elapsed_steps[0])))

    run_expert(rig, no_holder=args.no_holder, release_holder=args.release_holder, on_phase=on_phase)
    if rig.first_success is not None:
        step, reward, before = rig.first_success
        print(
            f"first success at env step {step}: env.step reward {reward:.2f} "
            f"(the step before: {before:.2f})"
        )
    print(f"target face angle: {math.degrees(TARGET_FACE_ANGLE):.1f} deg")
    if control.control == "delta":
        steps = [(n, e - (phase_ends[i - 1][1] if i else 0)) for i, (n, e) in enumerate(phase_ends)]
        print("phase step counts: " + ", ".join(f"{n} {s}" for n, s in steps))

    env.close()
    if args.no_lock:
        sys.exit(0)
    if args.no_holder or args.release_holder:
        limit = 50.0 if args.release_holder else 5.0
        ok = not final["success"].any() and (final["face_angle_deg"] < limit).all()
    else:
        ok = final["success"].all()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
