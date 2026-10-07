"""PPO building blocks, metrics and checkpoints of the IPPO trainer (pure torch, CPU)."""

import math

import pytest

torch = pytest.importorskip("torch")  # CI installs only the dev extra; run with `make dev`

from callosum.configs.ippo import IPPOConfig
from callosum.training._checkpoint import (
    load_agent_weights,
    load_checkpoint,
    load_training_state,
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


def _adam_agents(steps: int = 3) -> tuple[list[ActorCritic], list]:
    """Two small agents and Adam optimizers after a few real update steps."""
    agents = [ActorCritic(10, 6), ActorCritic(12, 6)]
    optimizers = [torch.optim.Adam(a.parameters(), lr=1e-3, eps=1e-5) for a in agents]
    for agent, opt, dim in zip(agents, optimizers, (10, 12), strict=True):
        for _ in range(steps):
            opt.zero_grad()
            x = torch.randn(8, dim)
            loss = agent.get_value(x).pow(2).mean() + agent.actor_mean(x).pow(2).mean()
            loss = loss + agent.actor_logstd.sum()
            loss.backward()
            opt.step()
    return agents, optimizers


def test_checkpoint_roundtrip(tmp_path) -> None:
    cfg = IPPOConfig()
    agents, optimizers = _adam_agents()
    path = tmp_path / "latest.pt"
    save_checkpoint(
        path, agents, cfg, [10, 12], ["u0", "u1"], 5, 1000, {"success_once": 0.5},
        optimizers=optimizers, best_score=(0.5, -3.0), train_seconds=12.5,
    )  # fmt: skip
    assert not (tmp_path / "latest.pt.tmp").exists()
    payload = load_checkpoint(path)  # weights_only=True inside
    assert payload["format"] == 2
    assert payload["iteration"] == 5 and payload["obs_dims"] == [10, 12]
    assert payload["config"]["env_id"] == cfg.env_id
    assert payload["best_score"] == [0.5, -3.0]
    assert path.stat().st_size < 50_000_000
    fresh = [ActorCritic(10, 6), ActorCritic(12, 6)]
    load_agent_weights(payload, fresh)
    for old, new in zip(agents, fresh, strict=True):
        for a, b in zip(old.parameters(), new.parameters(), strict=True):
            assert torch.equal(a, b)


def test_training_state_roundtrip(tmp_path) -> None:
    cfg = IPPOConfig()
    agents, optimizers = _adam_agents()
    path = tmp_path / "latest.pt"
    save_checkpoint(
        path, agents, cfg, [10, 12], ["u0", "u1"], 7, 1792, {"success_once": 0.25, "return": 4.0},
        optimizers=optimizers, best_score=(0.25, 4.0), train_seconds=99.0,
    )  # fmt: skip
    expected_next = torch.rand(3)  # the RNG state right after saving

    fresh = [ActorCritic(10, 6), ActorCritic(12, 6)]
    fresh_opts = [torch.optim.Adam(a.parameters(), lr=1e-3, eps=1e-5) for a in fresh]
    torch.manual_seed(12345)  # scramble the RNG; loading must put it back
    state = load_training_state(load_checkpoint(path), fresh, fresh_opts)

    assert torch.equal(torch.rand(3), expected_next)
    assert (state.iteration, state.global_step) == (7, 1792)
    assert state.best_score == (0.25, 4.0) and isinstance(state.best_score, tuple)
    assert state.last_eval == {"success_once": 0.25, "return": 4.0}
    assert state.train_seconds == 99.0
    for old, new in zip(agents, fresh, strict=True):
        for a, b in zip(old.parameters(), new.parameters(), strict=True):
            assert torch.equal(a, b)
    for old_opt, new_opt in zip(optimizers, fresh_opts, strict=True):
        old_state, new_state = old_opt.state_dict(), new_opt.state_dict()
        assert old_state["param_groups"] == new_state["param_groups"]
        assert old_state["state"].keys() == new_state["state"].keys()
        for key, entry in old_state["state"].items():
            assert entry["step"] == new_state["state"][key]["step"] > 0
            assert torch.equal(entry["exp_avg"], new_state["state"][key]["exp_avg"])
            assert torch.equal(entry["exp_avg_sq"], new_state["state"][key]["exp_avg_sq"])


def test_best_score_none_roundtrips(tmp_path) -> None:
    agents, optimizers = _adam_agents(steps=1)
    path = tmp_path / "latest.pt"
    save_checkpoint(
        path, agents, IPPOConfig(), [10, 12], ["u0", "u1"], 1, 256, optimizers=optimizers
    )
    state = load_training_state(load_checkpoint(path), agents, optimizers)
    assert state.best_score is None and state.last_eval == {} and state.train_seconds == 0.0


def test_format_1_file_is_a_warm_start_only(tmp_path) -> None:
    agents = [ActorCritic(10, 6), ActorCritic(12, 6)]
    path = tmp_path / "old.pt"
    torch.save(
        {
            "format": 1,
            "agents": {"agent_a": agents[0].state_dict(), "agent_b": agents[1].state_dict()},
            "config": {},
            "obs_dims": [10, 12],
            "agent_uids": ["u0", "u1"],
            "iteration": 3,
            "global_step": 768,
            "eval": {},
        },
        path,
    )
    payload = load_checkpoint(path)  # still readable
    fresh = [ActorCritic(10, 6), ActorCritic(12, 6)]
    load_agent_weights(payload, fresh)  # warm start works
    for old, new in zip(agents, fresh, strict=True):
        for a, b in zip(old.parameters(), new.parameters(), strict=True):
            assert torch.equal(a, b)
    opts = [torch.optim.Adam(a.parameters()) for a in fresh]
    with pytest.raises(ValueError, match="weights only.*--checkpoint"):
        load_training_state(payload, fresh, opts)


def test_unknown_format_rejected(tmp_path) -> None:
    path = tmp_path / "x.pt"
    torch.save({"format": 99}, path)
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_checkpoint(path)
