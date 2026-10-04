"""Smoke-check FaceTurn-v0: plumbing (as smoke_env.py) plus the two checks
step 1.3's readiness criterion calls out explicitly:
  1. scripting the face joint straight to the target angle should flip
     evaluate()["success"] to True;
  2. displacing the body afterward should flip it back to False -- i.e. the
     body-drift penalty/instability check actually fires;
  3. the face lock (the face cannot turn unless the holder grasps the body): the face forced to
     the target angle snaps back to the locked angle on the next env step, since the holder is
     at rest. Checks 1 and 2 call `evaluate()` without stepping the env, so the lock does not
     touch them (it acts in `_before_control_step` / `_after_simulation_step`, i.e. in `step`).

Meant for the GPU server; `--sim-backend cpu` runs a single env locally (also on macOS).
See docs/implementation-plan.md, step 1.3, "Критерий готовности (СЕРВЕР)".
"""

import torch
from _sim_utils import make_env, parse_args
from mani_skill.utils.structs.pose import Pose

# Importing the module (not just the callosum.envs package) runs its
# @register_env("FaceTurn-v0", ...) decorator. callosum.envs itself stays
# import-clean on macOS/CI (no mani_skill there), so it deliberately does not
# re-export this submodule -- see docs/implementation-plan.md section 0.
import callosum.envs.face_turn  # noqa: F401
from callosum.envs.face_turn import TARGET_FACE_ANGLE


def main() -> None:
    args = parse_args(__doc__)
    env = make_env("FaceTurn-v0", args)
    base_env = env.unwrapped

    obs, _ = env.reset(seed=0)
    print(f"num_envs: {base_env.num_envs}")
    print(f"flattened obs (obs_mode='state'): shape={tuple(obs.shape)}, dtype={obs.dtype}")
    print(f"action_space: {env.action_space}")

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, _terminated, _truncated, info = env.step(action)
    print("after 20 random steps:")
    print(f"  obs: shape={tuple(obs.shape)}, dtype={obs.dtype}")
    print(f"  reward: shape={tuple(reward.shape)}, dtype={reward.dtype}")
    print(f"  success (random rollout -- expected all False): {info['success'].any().item()}")

    # Scripted check 1: force the face straight to the target angle (bypassing
    # robot control) and confirm evaluate() reports success. This isolates
    # success-detection correctness from whether a policy can actually
    # achieve the turn.
    env.reset(seed=0)
    n = base_env.num_envs
    base_env.cube.set_qpos(torch.full((n, 1), TARGET_FACE_ANGLE, device=base_env.device))
    info = base_env.evaluate()
    reward = base_env.compute_dense_reward(obs=None, action=None, info=info)
    print("scripted check 1 (face at target angle, body untouched):")
    print(f"  face_angle[0]: {info['face_angle'][0].item():.4f} (target {TARGET_FACE_ANGLE:.4f})")
    print(f"  is_body_stable: {info['is_body_stable'].all().item()} (expected True)")
    print(f"  success: {info['success'].all().item()} (expected True)")
    print(f"  dense_reward[0]: {reward[0].item():.3f}")

    # Scripted check 2: from that same success state, displace the body by
    # more than its position tolerance and confirm success flips back to
    # False -- i.e. the body-drift penalty/instability check actually fires.
    displaced_pose = Pose.create_from_pq(
        p=base_env.cube.pose.p + torch.tensor([0.05, 0, 0], device=base_env.device),
        q=base_env.cube.pose.q,
    )
    base_env.cube.set_pose(displaced_pose)
    info = base_env.evaluate()
    reward = base_env.compute_dense_reward(obs=None, action=None, info=info)
    print("scripted check 2 (face still at target angle, body displaced +5cm in x):")
    print(f"  is_body_stable: {info['is_body_stable'].all().item()} (expected False)")
    print(f"  success: {info['success'].any().item()} (expected False)")
    print(f"  dense_reward[0]: {reward[0].item():.3f} (expected lower than check 1)")

    # Scripted check 3: the face lock. Force the face to the target angle, then take one step with
    # zero actions: the holder is not grasping, so the face is clamped back to the angle latched
    # at the start of the step (0, the reset angle).
    # TODO(review): GPU path of the lock (batched fetch/clamp/apply) unverified until the server.
    env.reset(seed=0)
    base_env.cube.set_qpos(torch.full((n, 1), TARGET_FACE_ANGLE, device=base_env.device))
    if base_env.gpu_sim_enabled:
        base_env.scene._gpu_apply_all()
        base_env.scene._gpu_fetch_all()
    zero_action = {
        uid: torch.zeros(space.shape, device=base_env.device)
        for uid, space in env.action_space.spaces.items()
    }
    env.step(zero_action)
    angle = base_env.evaluate()["face_angle"]
    print("scripted check 3 (face forced to the target, holder idle, one env step):")
    print(f"  face_angle[0]: {angle[0].item():.4f} (expected ~0: locked)")
    print(f"  all envs locked: {(angle.abs() < 0.05).all().item()} (expected True)")
    print(
        f"  lock engaged control steps [min/max]: "
        f"{base_env.face_lock_engaged_steps.min().item()} / "
        f"{base_env.face_lock_engaged_steps.max().item()}"
    )

    env.close()


if __name__ == "__main__":
    main()
