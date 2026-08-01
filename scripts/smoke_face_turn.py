"""Smoke-check FaceTurn-v0: plumbing (as smoke_env.py) plus the two checks
step 1.3's readiness criterion calls out explicitly:
  1. scripting the face joint straight to the target angle should flip
     evaluate()["success"] to True;
  2. displacing the body afterward should flip it back to False -- i.e. the
     body-drift penalty/instability check actually fires.

Server-only (GPU sim) -- cannot run on macOS. See docs/implementation-plan.md,
step 1.3, "Критерий готовности (СЕРВЕР)".
"""

import gymnasium as gym
import torch
from mani_skill.utils.structs.pose import Pose

# Importing the module (not just the callosum.envs package) runs its
# @register_env("FaceTurn-v0", ...) decorator. callosum.envs itself stays
# import-clean on macOS/CI (no mani_skill there), so it deliberately does not
# re-export this submodule -- see docs/implementation-plan.md section 0.
import callosum.envs.face_turn  # noqa: F401
from callosum.envs.face_turn import TARGET_FACE_ANGLE


def main() -> None:
    env = gym.make("FaceTurn-v0", num_envs=16, obs_mode="state", sim_backend="gpu")
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
    base_env.face_link.joint.qpos = torch.full((n,), TARGET_FACE_ANGLE, device=base_env.device)
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

    env.close()


if __name__ == "__main__":
    main()
