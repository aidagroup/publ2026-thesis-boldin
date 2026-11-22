"""CategoricalPolicy: shapes, probability identities, act/evaluate consistency (CPU torch)."""

import math

import pytest

torch = pytest.importorskip("torch")  # CI installs only the dev extra; run with `make dev`

from callosum.common.categorical import CategoricalPolicy
from callosum.common.ppo_core import ppo_update
from callosum.experiments.arm.ppo_config import IPPOConfig


def test_forward_returns_logits_and_value() -> None:
    policy = CategoricalPolicy(obs_dim=6, n_actions=9)
    obs = torch.randn(16, 6)
    logits, value = policy(obs)
    assert logits.shape == (16, 9)
    assert value.shape == (16, 1)


def test_act_samples_valid_indices_with_event_logprobs() -> None:
    torch.manual_seed(0)
    policy = CategoricalPolicy(obs_dim=5, n_actions=7)
    obs = torch.randn(64, 5)
    action, log_prob, entropy = policy.act(obs)
    assert action.shape == (64,) and action.dtype == torch.int64
    assert bool(((action >= 0) & (action < 7)).all())
    probs = log_prob.exp()  # event probabilities of the sampled ids
    assert bool((probs > 0).all()) and bool((probs <= 1.0).all())
    assert bool((entropy > 0).all())
    assert bool((entropy <= math.log(7) + 1e-6).all())  # uniform is the entropy maximum


def test_initial_logits_are_near_uniform() -> None:
    torch.manual_seed(1)
    policy = CategoricalPolicy(obs_dim=8, n_actions=9)
    logits, _ = policy(torch.randn(32, 8))
    probs = torch.softmax(logits, dim=-1)
    assert bool(torch.allclose(probs, torch.full_like(probs, 1.0 / 9), atol=0.05))


def test_act_matches_evaluate() -> None:
    torch.manual_seed(2)
    policy = CategoricalPolicy(obs_dim=4, n_actions=5)
    obs = torch.randn(24, 4)
    with torch.no_grad():
        action, log_prob, entropy = policy.act(obs)
        eval_log_prob, eval_entropy, value = policy.evaluate(obs, action)
    assert torch.allclose(log_prob, eval_log_prob)
    assert torch.allclose(entropy, eval_entropy)
    assert torch.allclose(value, policy.get_value(obs))
    # The log-probs are the log-softmax of the forward logits at the sampled ids.
    logits, _ = policy(obs)
    assert torch.allclose(
        log_prob, torch.log_softmax(logits, dim=-1).gather(1, action[:, None])[:, 0]
    )


def test_evaluate_accepts_rollout_buffer_action_shape() -> None:
    torch.manual_seed(3)
    policy = CategoricalPolicy(obs_dim=4, n_actions=5)
    obs = torch.randn(24, 4)
    with torch.no_grad():
        action, log_prob, _, _ = policy.get_action_and_value(obs)
        log_prob_buf, _, _ = policy.evaluate(obs, action.unsqueeze(-1))  # (24, 1) as stored
    assert torch.allclose(log_prob_buf.squeeze(-1), log_prob)


def test_deterministic_uses_argmax() -> None:
    torch.manual_seed(4)
    policy = CategoricalPolicy(obs_dim=3, n_actions=6)
    obs = torch.randn(20, 3)
    logits, _ = policy(obs)
    assert torch.equal(policy.get_action(obs, deterministic=True), logits.argmax(dim=-1))
    action, _, _ = policy.act(obs, deterministic=True)
    assert torch.equal(action, logits.argmax(dim=-1))


def test_gradients_flow_through_evaluate() -> None:
    policy = CategoricalPolicy(obs_dim=3, n_actions=4)
    obs = torch.randn(8, 3)
    action = torch.randint(0, 4, (8,))
    log_prob, entropy, value = policy.evaluate(obs, action)
    (-(log_prob + entropy).mean() + value.mean()).backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in policy.actor_logits.parameters()
    )
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.critic.parameters())


def test_ppo_update_shifts_the_categorical_policy_toward_the_good_action() -> None:
    torch.manual_seed(5)
    cfg = IPPOConfig(num_envs=64, num_steps=4, num_minibatches=4, update_epochs=4, target_kl=None)
    policy = CategoricalPolicy(obs_dim=3, n_actions=4)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3, eps=1e-5)
    obs = torch.randn(cfg.batch_size, 3)
    with torch.no_grad():
        action, log_prob, _, _ = policy.get_action_and_value(obs)
    batch = {
        "obs": obs,
        "actions": action.unsqueeze(-1),  # the rollout buffer stores (batch, 1)
        "logprobs": log_prob,
        "advantages": (action == 2).float() - 0.5,  # primitive 2 is the good choice
        "returns": (action == 2).float(),
        "values": torch.zeros(cfg.batch_size),
    }
    with torch.no_grad():
        before = torch.softmax(policy(obs)[0], dim=-1)[:, 2].mean()
    for _ in range(8):
        ppo_update(policy, optimizer, batch, cfg)
    with torch.no_grad():
        after = torch.softmax(policy(obs)[0], dim=-1)[:, 2].mean()
    assert after > before
