"""BC pretraining and the IPPO fine-tune additions (critic warm-up, auxiliary BC loss), pure torch."""

import copy
import dataclasses
import json

import pytest

torch = pytest.importorskip("torch")  # CI installs only the dev extra; run with `make dev`

from callosum.configs.bc import BCConfig
from callosum.configs.ippo import IPPOConfig
from callosum.training import bc
from callosum.training._agent_obs import AGENT_UIDS, AgentObsBuilder
from callosum.training._checkpoint import (
    check_warm_start_compat,
    load_agent_weights,
    load_checkpoint,
)
from callosum.training._demos import assemble_demos, layout_to_meta, save_demos, split_rollout
from callosum.training._ppo_core import ActorCritic, bc_mse, ppo_update

LAYOUT = [
    (("agent", "so101_pg-0", "qpos"), 7),
    (("agent", "so101_pg-0", "qvel"), 7),
    (("agent", "so101_pg-1", "qpos"), 7),
    (("agent", "so101_pg-1", "qvel"), 7),
    (("extra", "agent_a_tcp_pose"), 7),
    (("extra", "agent_b_tcp_pose"), 7),
    (("extra", "cube_pose"), 7),
    (("extra", "face_angle"), 1),
    (("extra", "face_pose"), 7),
]
OBS_DIM = sum(w for _, w in LAYOUT)


def synthetic_demos(num_episodes: int = 12, length: int = 20, seed: int = 0) -> dict:
    """Episodes whose actions and rewards are smooth functions of the observation."""
    gen = torch.Generator().manual_seed(seed)
    builder = AgentObsBuilder(LAYOUT, "full", AGENT_UIDS)
    mix = [torch.randn(d, 6, generator=gen) * 0.3 for d in builder.obs_dims]
    obs = torch.randn(length, num_episodes, OBS_DIM, generator=gen)
    inputs = [builder(obs[t]) for t in range(length)]
    actions = torch.stack(
        [torch.stack([torch.tanh(x[i] @ mix[i]) for i in range(2)], dim=1) for x in inputs]
    )  # (T, n, 2, 6)
    rewards = torch.ones(length, num_episodes)
    rewards[-1] += 10.0  # a "success bonus" on the last step
    success = torch.zeros(length, num_episodes, dtype=torch.bool)
    success[-1] = True
    episodes = split_rollout(
        obs, actions, rewards, success.clone(), torch.zeros_like(success), success,
        keep_all=False, batch_seed=0,
    )  # fmt: skip
    meta = {
        "env_id": "FaceTurn-v0",
        "obs_layout": layout_to_meta(LAYOUT),
        "agent_uids": list(AGENT_UIDS),
        "git_commit": None,
    }
    return assemble_demos(episodes, meta)


def test_prepare_data_builds_per_agent_inputs_and_returns() -> None:
    demos = synthetic_demos()
    cfg = BCConfig(demos="x", gamma=0.5)
    data = bc.prepare_data(demos, cfg)
    assert [o.shape for o in data.obs] == [(240, 43), (240, 43)]  # same rule as IPPO
    assert [a.shape for a in data.actions] == [(240, 6), (240, 6)]
    builder = AgentObsBuilder(LAYOUT, "full", AGENT_UIDS)
    assert torch.equal(data.obs[1], builder(demos["obs"])[1])
    # The last step of an episode: return = its own reward; earlier steps discount the bonus.
    last = data.returns[19]
    assert last.item() == pytest.approx(11.0)
    assert data.returns[18].item() == pytest.approx(1.0 + 0.5 * 11.0)
    # `partner_obs="none"` removes the partner's TCP pose (7 columns) from both inputs.
    none = bc.prepare_data(demos, BCConfig(demos="x", partner_obs="none"))
    assert none.builder.obs_dims == [36, 36]


def test_only_success_filters_failed_episodes() -> None:
    demos = synthetic_demos(num_episodes=4)
    demos["episodes"]["success"][1] = False  # pretend episode 1 failed
    kept = bc.prepare_data(demos, BCConfig(demos="x"))
    assert kept.size == 3 * 20 and 1 not in kept.episode_id.tolist()
    everything = bc.prepare_data(demos, BCConfig(demos="x", only_success=False))
    assert everything.size == 4 * 20


def test_split_holds_out_whole_episodes() -> None:
    data = bc.prepare_data(synthetic_demos(), BCConfig(demos="x"))
    train, val = bc.split_episodes(data, 0.25, seed=0)
    assert val is not None
    assert train.size + val.size == data.size
    assert set(train.episode_id.tolist()).isdisjoint(val.episode_id.tolist())
    assert len(set(val.episode_id.tolist())) == 3  # 25% of 12 episodes
    again = bc.split_episodes(data, 0.25, seed=0)[1]
    assert torch.equal(again.episode_id, val.episode_id)  # seeded
    assert bc.split_episodes(data, 0.0, seed=0)[1] is None


def test_bc_reduces_loss_and_critic_is_on_the_return_scale() -> None:
    torch.manual_seed(0)
    cfg = BCConfig(demos="x", epochs=40, batch_size=64, log_every=40, val_fraction=0.25)
    data = bc.prepare_data(synthetic_demos(), cfg)
    train, val = bc.split_episodes(data, cfg.val_fraction, cfg.seed)
    agents = [ActorCritic(d, 6) for d in data.builder.obs_dims]
    before = bc.evaluate_losses(agents, val, 1.0)
    stats = bc.train_bc(agents, train, val, cfg, log=lambda line: None)
    after = bc.evaluate_losses(agents, val, 1.0)  # critics were scaled back: value_scale 1
    for name in ("agent_a", "agent_b"):
        assert after[f"{name}/actor_mse"] < 0.5 * before[f"{name}/actor_mse"]
        assert after[f"{name}/critic_rmse"] < before[f"{name}/critic_rmse"]
    assert stats["value_scale"] >= 1.0 and len(stats["history"]) == 2  # epoch 1 and epoch 40


def test_fold_value_scale_scales_the_critic_output() -> None:
    agent = ActorCritic(5, 6)
    obs = torch.randn(4, 5)
    before = agent.get_value(obs)
    bc.fold_value_scale(agent, 7.0)
    assert torch.allclose(agent.get_value(obs), 7.0 * before, atol=1e-5)


def test_run_writes_a_checkpoint_that_ippo_can_warm_start_from(tmp_path) -> None:
    demos_path = tmp_path / "demos.pt"
    save_demos(demos_path, synthetic_demos())
    cfg = BCConfig(
        demos=str(demos_path), runs_dir=str(tmp_path), exp_name="bc_test", epochs=3,
        batch_size=64, log_every=3, actor_logstd=-1.6, device="cpu",
    )  # fmt: skip
    run_dir = bc.run(cfg)
    assert json.loads((run_dir / "config.json").read_text())["actor_logstd"] == -1.6
    payload = load_checkpoint(run_dir / "bc.pt")  # weights_only=True loadable
    assert payload["format"] == 1 and payload["obs_dims"] == [43, 43]
    assert payload["agent_uids"] == list(AGENT_UIDS) and payload["config"]["partner_obs"] == "full"
    check_warm_start_compat(payload, [43, 43], payload["obs_fields"], "full")
    agents = [ActorCritic(43, 6), ActorCritic(43, 6)]
    load_agent_weights(payload, agents)
    for agent in agents:
        assert agent.actor_logstd.exp().mean().item() == pytest.approx(0.2, abs=0.01)


def test_warm_start_compat_rejects_mismatched_inputs() -> None:
    payload = {
        "obs_dims": [43, 43],
        "config": {"partner_obs": "full"},
        "obs_fields": [["a"], ["b"]],
    }
    check_warm_start_compat(payload, [43, 43], [["a"], ["b"]], "full")
    with pytest.raises(ValueError, match="width"):
        check_warm_start_compat(payload, [36, 36], None, "none")
    with pytest.raises(ValueError, match="partner_obs"):
        check_warm_start_compat(
            {"obs_dims": [43, 43], "config": {"partner_obs": "none"}}, [43, 43], None, "full"
        )
    with pytest.raises(ValueError, match="fields"):
        check_warm_start_compat(payload, [43, 43], [["a"], ["c"]], "full")
    check_warm_start_compat({"agents": {}}, [43, 43])  # old files without metadata still load


def ppo_batch(cfg: IPPOConfig, agent: ActorCritic, obs_dim: int = 5) -> dict:
    obs = torch.randn(cfg.batch_size, obs_dim)
    with torch.no_grad():
        actions, logprobs, _, values = agent.get_action_and_value(obs)
    return {
        "obs": obs,
        "actions": actions,
        "logprobs": logprobs,
        "advantages": torch.randn(cfg.batch_size),
        "returns": obs[:, 0] * 2.0,
        "values": values.flatten(),
    }


def snapshot(module: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]


def unchanged(module: torch.nn.Module, before: list[torch.Tensor]) -> bool:
    return all(torch.equal(p.detach(), b) for p, b in zip(module.parameters(), before))


def test_critic_warmup_freezes_the_actor_but_trains_the_critic() -> None:
    torch.manual_seed(0)
    cfg = IPPOConfig(num_envs=64, num_steps=4, num_minibatches=4, target_kl=None)
    agent = ActorCritic(5, 6)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-3, eps=1e-5)
    actor, critic = snapshot(agent.actor_mean), snapshot(agent.critic)
    logstd = agent.actor_logstd.detach().clone()
    batch = ppo_batch(cfg, agent)
    for _ in range(3):
        ppo_update(agent, optimizer, batch, cfg, update_actor=False)
    assert unchanged(agent.actor_mean, actor)
    assert torch.equal(agent.actor_logstd.detach(), logstd)
    assert not unchanged(agent.critic, critic)
    # With the actor released, the very same optimizer moves it again.
    ppo_update(agent, optimizer, batch, cfg, update_actor=True)
    assert not unchanged(agent.actor_mean, actor)


def test_critic_warmup_ignores_the_bc_loss() -> None:
    cfg = IPPOConfig(num_envs=64, num_steps=4, num_minibatches=4)
    agent = ActorCritic(5, 6)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-3, eps=1e-5)
    actor = snapshot(agent.actor_mean)
    demo = {"obs": torch.randn(50, 5), "actions": torch.rand(50, 6)}
    metrics = ppo_update(
        agent,
        optimizer,
        ppo_batch(cfg, agent),
        cfg,
        update_actor=False,
        demo_batch=demo,
        bc_coef=5.0,
    )
    assert unchanged(agent.actor_mean, actor) and metrics["bc_loss"] == 0.0


def test_auxiliary_bc_loss_pulls_the_policy_towards_the_demos() -> None:
    torch.manual_seed(0)
    cfg = IPPOConfig(
        num_envs=64, num_steps=4, num_minibatches=4, target_kl=None, bc_batch_size=128,
        demos="d.pt", bc_coef=1.0,
    )  # fmt: skip
    agent = ActorCritic(5, 6)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-3, eps=1e-5)
    demo_obs = torch.randn(256, 5)
    demo = {"obs": demo_obs, "actions": torch.tanh(demo_obs @ torch.randn(5, 6))}
    start = bc_mse(agent, demo["obs"], demo["actions"]).item()
    batch = ppo_batch(cfg, agent)
    batch["advantages"] = torch.zeros(cfg.batch_size)  # no policy gradient: only the BC loss acts
    losses = []
    for _ in range(40):
        losses.append(
            ppo_update(agent, optimizer, batch, cfg, demo_batch=demo, bc_coef=1.0)["bc_loss"]
        )
    assert losses[-1] < 0.5 * start and losses[-1] < losses[0]
    # Switched off (coefficient 0 or no demos), the metric is 0 and nothing is sampled.
    assert ppo_update(agent, optimizer, batch, cfg, demo_batch=demo, bc_coef=0.0)["bc_loss"] == 0.0
    assert ppo_update(agent, optimizer, batch, cfg, bc_coef=1.0)["bc_loss"] == 0.0


def test_default_ppo_update_is_unchanged_by_the_new_options() -> None:
    """Same seed, same batch: the plain call and the call with the options off agree exactly."""
    cfg = IPPOConfig(num_envs=64, num_steps=4, num_minibatches=4, target_kl=None)
    torch.manual_seed(1)
    agent = ActorCritic(5, 6)
    other = copy.deepcopy(agent)
    batch = ppo_batch(cfg, agent)
    opt_a = torch.optim.Adam(agent.parameters(), lr=3e-3)
    opt_b = torch.optim.Adam(other.parameters(), lr=3e-3)
    torch.manual_seed(2)
    ppo_update(agent, opt_a, batch, cfg)
    torch.manual_seed(2)
    ppo_update(other, opt_b, batch, cfg, update_actor=True, demo_batch=None, bc_coef=0.0)
    assert all(torch.equal(a, b) for a, b in zip(agent.parameters(), other.parameters()))


def test_bc_config_asdict_is_checkpointable() -> None:
    assert json.dumps(dataclasses.asdict(BCConfig(demos="x")))
