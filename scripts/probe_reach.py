"""Can the arms actually reach the cube? (answers the y=+-0.3 spacing TODO)

Server-only (GPU sim). Motivated by the 2M-step FaceTurn-v0 run of 2026-08-30:
the policy climbed the shaped reward but never once succeeded, and the reward
level implied the tool centre points were stalling ~0.36 m from their targets.
This measures that directly instead of inferring it.

Two questions, answered separately:

  1. Where do the TCPs START? (rest keyframe -> distance to their reward targets)
  2. How CLOSE can they get at all? -> sample joint configurations inside the
     limits and report the minimum distance achieved. That is an empirical lower
     bound on the workspace: if even the best of thousands of samples cannot get
     near the cube, no policy will either, and the fix is the scene layout, not
     more training.

    uv run python scripts/probe_reach.py
    uv run python scripts/probe_reach.py --samples 20000
"""

import argparse

import gymnasium as gym
import torch

import callosum.envs.face_turn  # noqa: F401  (registers FaceTurn-v0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=8192, help="random joint configs to try")
    ap.add_argument("--env-id", default="FaceTurn-v0")
    args = ap.parse_args()

    env = gym.make(
        args.env_id,
        num_envs=args.samples,
        obs_mode="state",
        sim_backend="gpu",
        render_backend="none",
    )
    base = env.unwrapped
    base.reset(seed=0)

    # The reward's two reach targets (see FaceTurn.compute_dense_reward):
    # the rotator aims at the face link, the holder at the cube body.
    targets = {
        "rotator (agent_b) -> face": (base.agent_b, base.face_link.pose.p),
        "holder  (agent_a) -> body": (base.agent_a, base.cube.pose.p),
    }

    print(f"env={args.env_id}  num_envs={args.samples}")
    print(f"cube centre (env 0): {base.cube.pose.p[0].tolist()}")
    for name, (agent, target) in targets.items():
        d = torch.linalg.norm(agent.tcp_pos - target, dim=1)
        print(f"\n{name}")
        print(f"  at rest keyframe : {d.mean():.3f} m (min over envs {d.min():.3f})")
        print(f"  reward term there: {(1 - torch.tanh(5 * d)).mean():.4f}")

    # Sample the joint space. One configuration per parallel env, so the whole
    # sweep is a single batched write + read -- no stepping, no physics.
    print(f"\nsampling {args.samples} joint configurations per arm...")
    for name, (agent, target) in targets.items():
        lo, hi = agent.robot.get_qlimits()[0, :, 0], agent.robot.get_qlimits()[0, :, 1]
        lo, hi = lo.to(base.device), hi.to(base.device)
        q = lo + (hi - lo) * torch.rand((args.samples, lo.shape[0]), device=base.device)
        agent.robot.set_qpos(q)
        base.scene._gpu_apply_all()
        base.scene.px.gpu_update_articulation_kinematics()
        base.scene._gpu_fetch_all()

        d = torch.linalg.norm(agent.tcp_pos - target, dim=1)
        best = d.min()
        print(f"\n{name}")
        print(f"  closest reachable : {best:.3f} m")
        print(f"  reward term there : {(1 - torch.tanh(5 * best)):.4f}")
        print(f"  10th percentile   : {torch.quantile(d, 0.10):.3f} m")
        if best > 0.05:
            print("  >>> OUT OF REACH: the gripper cannot get near this target.")
            print("  >>> Fix the scene layout (arm spacing / cube position), not the policy.")
        else:
            print("  >>> reachable — the target is inside the workspace.")

    env.close()


if __name__ == "__main__":
    main()
