"""PPO building blocks, metrics and checkpoints of the IPPO trainer (pure torch, CPU)."""

import math

import pytest

torch = pytest.importorskip("torch")  # CI installs only the dev extra; run with `make dev`

from callosum.configs.ippo import IPPOConfig
from callosum.training._checkpoint import (
    load_agent_weights,
    load_checkpoint,
    save_checkpoint,
)
from callosum.training._ppo_core import ActorCritic, compute_gae, ppo_update


def test_gae_matches_discounted_return_without_values() -> None:
    rewards = torch.ones(4, 1)
    zeros = torch.zeros(4, 1)
    adv, ret = compute_gae(
        rewards, zeros, zeros, zeros, torch.zeros(1, 1), torch.zeros(1), gamma=0.5, gae_lambda=1.0
    )
    expected = torch.tensor([[1 + 0.5 + 0.25 + 0.125], [1 + 0.5 + 0.25], [1 + 0.5], [1.0]])
    assert torch.allclose(adv, expected) and torch.allclose(ret, expected)


def test_gae_bootstraps_final_value_across_reset_and_cuts_after_it() -> None:
    # Two steps; the env resets at the end of step 0 (dones[1] = 1) with true last value 10.
    rewards = torch.tensor([[1.0], [1.0]])
    values = torch.zeros(2, 1)
    dones = torch.tensor([[0.0], [1.0]])
    final_values = torch.tensor([[10.0], [0.0]])
    adv, _ = compute_gae(
        rewards, values, dones, final_values, torch.zeros(1, 1), torch.zeros(1), 0.9, 1.0
    )
    assert math.isclose(adv[0].item(), 1 + 0.9 * 10.0)  # no leakage from step 1 after the reset
    assert math.isclose(adv[1].item(), 1.0)


def test_ppo_update_improves_a_trivial_problem() -> None:
    torch.manual_seed(0)
    cfg = IPPOConfig(num_envs=64, num_steps=4, num_minibatches=4, update_epochs=4, target_kl=None)
    agent = ActorCritic(obs_dim=5, action_dim=6)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-3, eps=1e-5)
    obs = torch.randn(cfg.batch_size, 5)
    with torch.no_grad():
        actions, logprobs, _, values = agent.get_action_and_value(obs)
    returns = obs[:, 0] * 2.0  # value target depends on the observation
    batch = {
        "obs": obs,
        "actions": actions,
        "logprobs": logprobs,
        "advantages": torch.randn(cfg.batch_size),
        "returns": returns,
        "values": values.flatten(),
    }
    first = ppo_update(agent, optimizer, batch, cfg)["value_loss"]
    for _ in range(30):
        last = ppo_update(agent, optimizer, batch, cfg)
    assert last["value_loss"] < first
    assert set(last) == {
        "policy_loss", "value_loss", "entropy", "old_approx_kl", "approx_kl", "clipfrac",
        "explained_variance",
    }  # fmt: skip


def test_ppo_update_survives_single_sample_minibatches() -> None:
    torch.manual_seed(0)
    cfg = IPPOConfig(num_envs=1, num_steps=5, num_minibatches=1, target_kl=None)
    cfg.num_minibatches = 5  # bypass the config validation: every minibatch has one sample
    assert cfg.minibatch_size == 1
    agent = ActorCritic(obs_dim=5, action_dim=6)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-3, eps=1e-5)
    obs = torch.randn(cfg.batch_size, 5)
    with torch.no_grad():
        actions, logprobs, _, values = agent.get_action_and_value(obs)
    batch = {
        "obs": obs,
        "actions": actions,
        "logprobs": logprobs,
        "advantages": torch.randn(cfg.batch_size),
        "returns": torch.randn(cfg.batch_size),
        "values": values.flatten(),
    }
    metrics = ppo_update(agent, optimizer, batch, cfg)
    assert all(math.isfinite(v) for k, v in metrics.items() if k != "explained_variance")
    assert all(torch.isfinite(p).all() for p in agent.parameters())


def test_checkpoint_roundtrip(tmp_path) -> None:
    cfg = IPPOConfig()
    agents = [ActorCritic(10, 6), ActorCritic(12, 6)]
    path = tmp_path / "latest.pt"
    save_checkpoint(path, agents, cfg, [10, 12], ["u0", "u1"], 5, 1000, {"success_once": 0.5})
    assert not (tmp_path / "latest.pt.tmp").exists()
    payload = load_checkpoint(path)
    assert payload["iteration"] == 5 and payload["obs_dims"] == [10, 12]
    assert payload["config"]["env_id"] == cfg.env_id
    fresh = [ActorCritic(10, 6), ActorCritic(12, 6)]
    load_agent_weights(payload, fresh)
    for old, new in zip(agents, fresh, strict=True):
        for a, b in zip(old.parameters(), new.parameters(), strict=True):
            assert torch.equal(a, b)
    assert path.stat().st_size < 5_000_000  # a few MB, the server quota is small
