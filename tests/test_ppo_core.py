"""Unit tests for callosum.training._ppo_core (step 2.1) -- pure PyTorch,
no mani_skill dependency, so runnable on macOS/CI. These verify the RL
algorithm itself (network shapes, GAE, the PPO update) ahead of the paid
GPU session, where only the mani_skill-specific env glue in
callosum/training/ippo.py remains genuinely unverified.
"""

import numpy as np
import torch

from callosum.configs.ippo import IPPOConfig
from callosum.training._ppo_core import Agent, compute_gae, ppo_update


def test_agent_output_shapes() -> None:
    agent = Agent(obs_dim=10, action_dim=4)
    x = torch.randn(8, 10)
    assert agent.get_value(x).shape == (8, 1)
    action, logprob, entropy, value = agent.get_action_and_value(x)
    assert action.shape == (8, 4)
    assert logprob.shape == (8,)
    assert entropy.shape == (8,)
    assert value.shape == (8, 1)


def test_agent_deterministic_action_matches_actor_mean() -> None:
    agent = Agent(obs_dim=5, action_dim=2)
    x = torch.randn(3, 5)
    with torch.no_grad():
        assert torch.equal(agent.get_action(x, deterministic=True), agent.actor_mean(x))


def test_agent_action_replay_reproduces_same_logprob() -> None:
    # get_action_and_value(x, action=<given>) must score that action, not
    # resample -- required for the PPO ratio (newlogprob vs b_logprobs) to
    # be meaningful across minibatch epochs.
    agent = Agent(obs_dim=4, action_dim=3)
    x = torch.randn(6, 4)
    with torch.no_grad():
        action, logprob1, _, _ = agent.get_action_and_value(x)
        _, logprob2, _, _ = agent.get_action_and_value(x, action=action)
    assert torch.allclose(logprob1, logprob2)


def test_compute_gae_shapes_and_returns_identity() -> None:
    num_steps, num_envs = 6, 4
    args = IPPOConfig(gamma=0.99, gae_lambda=0.95)
    rewards = torch.randn(num_steps, num_envs)
    values = torch.randn(num_steps, num_envs)
    dones = torch.zeros(num_steps, num_envs)
    final_values = torch.zeros(num_steps, num_envs)
    next_value = torch.randn(1, num_envs)
    next_done = torch.zeros(num_envs)
    advantages, returns = compute_gae(
        rewards, values, dones, final_values, next_value, next_done, args
    )
    assert advantages.shape == (num_steps, num_envs)
    assert returns.shape == (num_steps, num_envs)
    assert torch.allclose(returns, advantages + values)


def test_compute_gae_all_zero_gives_zero_advantage() -> None:
    num_steps, num_envs = 5, 2
    args = IPPOConfig(gamma=0.99, gae_lambda=0.95)
    zeros = torch.zeros(num_steps, num_envs)
    advantages, returns = compute_gae(
        zeros, zeros, zeros, zeros, torch.zeros(1, num_envs), torch.zeros(num_envs), args
    )
    assert torch.allclose(advantages, torch.zeros_like(advantages))
    assert torch.allclose(returns, torch.zeros_like(returns))


def test_compute_gae_done_at_last_step_blocks_bootstrap() -> None:
    # One env, two steps, done at the last step: the huge next_value must be
    # ignored (next_not_done=0), so the last-step advantage is just that
    # step's immediate reward.
    args = IPPOConfig(gamma=0.99, gae_lambda=0.95)
    rewards = torch.tensor([[1.0], [1.0]])
    values = torch.zeros(2, 1)
    dones = torch.zeros(2, 1)
    final_values = torch.zeros(2, 1)
    next_value = torch.tensor([[100.0]])
    next_done = torch.tensor([1.0])
    advantages, _ = compute_gae(rewards, values, dones, final_values, next_value, next_done, args)
    assert torch.allclose(advantages[-1], torch.tensor([1.0]))


def test_compute_gae_final_values_feed_bootstrap_at_done_step() -> None:
    # One env, two steps: env is done mid-rollout (dones[1]=1, matching how
    # ManiSkillVectorEnv marks the step an episode ended on) and
    # final_values[0] carries the value of the true terminal observation
    # (as callosum.training.ippo populates it) -- it must be used as the
    # bootstrap for step 0's return instead of values[1] (the *reset*
    # observation's value, which would be wrong to bootstrap from).
    args = IPPOConfig(gamma=1.0, gae_lambda=1.0)
    rewards = torch.tensor([[0.0], [0.0]])
    values = torch.tensor([[0.0], [999.0]])  # values[1] is the post-reset obs; must be ignored
    dones = torch.tensor([[0.0], [1.0]])
    final_values = torch.tensor([[5.0], [0.0]])  # value of the true terminal obs at step 0
    next_value = torch.tensor([[0.0]])
    next_done = torch.tensor([0.0])
    _advantages, returns = compute_gae(
        rewards, values, dones, final_values, next_value, next_done, args
    )
    # step 0: next_not_done = 1 - dones[1] = 0, so real_next_value = final_values[0] = 5
    assert torch.allclose(returns[0], torch.tensor([5.0]))


def test_ppo_update_runs_and_produces_finite_losses() -> None:
    torch.manual_seed(0)
    agent = Agent(obs_dim=4, action_dim=2)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-2)
    batch = 64
    b_obs = torch.randn(batch, 4)
    with torch.no_grad():
        b_actions, b_logprobs, _, b_values = agent.get_action_and_value(b_obs)
        b_values = b_values.view(-1)
    b_advantages = torch.randn(batch)
    b_returns = b_values + b_advantages

    args = IPPOConfig(
        update_epochs=2,
        num_minibatches=4,
        minibatch_size=batch // 4,
        norm_adv=True,
        clip_coef=0.2,
        clip_vloss=False,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
    )
    metrics = ppo_update(
        agent, optimizer, b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values, args
    )
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"):
        assert np.isfinite(metrics[key]), f"{key} is not finite: {metrics[key]}"


def test_ppo_update_changes_parameters() -> None:
    torch.manual_seed(0)
    agent = Agent(obs_dim=3, action_dim=2)
    before = [p.clone() for p in agent.parameters()]
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-1)
    batch = 32
    b_obs = torch.randn(batch, 3)
    with torch.no_grad():
        b_actions, b_logprobs, _, b_values = agent.get_action_and_value(b_obs)
        b_values = b_values.view(-1)
    b_advantages = torch.randn(batch)
    b_returns = b_values + b_advantages
    args = IPPOConfig(
        update_epochs=1,
        minibatch_size=batch,
        norm_adv=False,
        clip_coef=0.2,
        clip_vloss=False,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
    )
    ppo_update(
        agent, optimizer, b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values, args
    )
    after = list(agent.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))
