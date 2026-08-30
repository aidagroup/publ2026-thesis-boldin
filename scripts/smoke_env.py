"""Smoke-check the TwoSO100-v0 plumbing: agents, observations, actions, reward.

Server-only (GPU sim) -- cannot run on macOS. See docs/implementation-plan.md,
step 1.2, "Критерий готовности (СЕРВЕР)".
"""

import gymnasium as gym

# Importing the module (not just the callosum.envs package) runs its
# @register_env("TwoSO100-v0", ...) decorator. callosum.envs itself stays
# import-clean on macOS/CI (no mani_skill there), so it deliberately does not
# re-export this submodule -- see docs/implementation-plan.md section 0.
import callosum.envs.two_so100_base  # noqa: F401


def main() -> None:
    # render_backend="none": this training is state-based and never renders, so
    # the renderer is pure overhead. It also must be off wherever Vulkan is
    # unavailable -- ManiSkill's render_utils.can_render() only checks that a
    # render device was selected, not that Vulkan actually works, so it says yes
    # on a headless container and then RenderSystem() raises
    # "vk::createInstanceUnique: ErrorIncompatibleDriver". Disabling it is the
    # documented remedy (BaseEnv docstring). Phase 4 (vision) will need it back.
    env = gym.make(
        "TwoSO100-v0",
        num_envs=16,
        obs_mode="state",
        sim_backend="gpu",
        render_backend="none",
    )
    base_env = env.unwrapped

    obs, _ = env.reset(seed=0)
    print(f"num_envs: {base_env.num_envs}")
    print(f"flattened obs (obs_mode='state'): shape={tuple(obs.shape)}, dtype={obs.dtype}")

    # obs_mode="state" flattens agent+extra into one combined tensor (see
    # mani_skill.utils.common.flatten_state_dict), so per-agent structure is
    # not visible in `obs` above. Check it directly via the unflattened dict
    # and the (genuinely per-agent) action space instead.
    unflat_obs = base_env.get_obs(unflattened=True)
    print(f"agent keys (per-agent proprioception): {list(unflat_obs['agent'].keys())}")
    for uid, agent_obs in unflat_obs["agent"].items():
        qpos_shape = tuple(agent_obs["qpos"].shape)
        qvel_shape = tuple(agent_obs["qvel"].shape)
        print(f"  {uid}: qpos={qpos_shape}, qvel={qvel_shape}")
    print(f"extra keys (own+partner TCP poses, cube pose): {list(unflat_obs['extra'].keys())}")

    print(f"agents_dict keys: {list(base_env.agent.agents_dict.keys())}")
    print(f"action_space: {env.action_space}")

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    print("after 20 random steps:")
    print(f"  obs: shape={tuple(obs.shape)}, dtype={obs.dtype}")
    print(f"  reward: shape={tuple(reward.shape)}, dtype={reward.dtype}")
    print(f"  terminated: shape={tuple(terminated.shape)}")
    print(f"  truncated: shape={tuple(truncated.shape)}")
    print(f"  success (evaluate() stub -- expected all False): {info['success'].any().item()}")

    env.close()


if __name__ == "__main__":
    main()
