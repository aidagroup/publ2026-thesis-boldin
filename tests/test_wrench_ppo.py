"""Continuous-wrench IPPO: config, GAE, policy shapes, one PPO step, checkpoint roundtrip.

Pure-torch / stdlib tests: no mujoco and no real env runs (the trainer imports the vec env
lazily inside `run()`, so this module imports fine without the sandbox extra). CI installs
only the dev extra, so the torch import is skipped there.
"""

import dataclasses
import math
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # CI installs only the dev extra; run with `make dev`

from callosum.configs.wrench_ppo import WrenchPPOConfig
from callosum.envs._wrench_logic import AGENT_KEYS, obs_size
from callosum.training import wrench_ppo
from callosum.training.wrench_ppo import (
    CHECKPOINT_FORMAT,
    EvalSummary,
    compute_advantages,
    evaluate_greedy,
    load_checkpoint,
    load_optimizers,
    load_weights,
    make_policies,
    ppo_step,
    save_checkpoint,
)

_ACTION_DIM = 6


# --- config -------------------------------------------------------------------------------


def test_default_config_is_valid_and_sized_as_the_plan() -> None:
    cfg = WrenchPPOConfig()
    assert cfg.partner_mode == "none"
    assert cfg.num_envs == 128
    assert cfg.rollout_steps == 64
    assert cfg.batch_size == 128 * 64
    assert cfg.minibatch_size == cfg.batch_size // cfg.num_minibatches
    assert cfg.total_steps == cfg.batch_size * 300
    assert cfg.obs_size == obs_size("none") == 26
    assert WrenchPPOConfig(partner_mode="oracle").obs_size == obs_size("oracle") == 35


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("partner_mode", "predicted", "partner_mode"),
        ("partner_mode", "none ", "partner_mode"),
        ("num_envs", 1, "num_envs"),
        ("rollout_steps", 0, "rollout_steps"),
        ("total_updates", 0, "total_updates"),
        ("seed", -1, "seed"),
        ("gamma", 0.0, "gamma"),
        ("gamma", 1.5, "gamma"),
        ("gae_lambda", -0.1, "gae_lambda"),
        ("clip_coef", 1.0, "clip_coef"),
        ("update_epochs", 0, "update_epochs"),
        ("learning_rate", 0.0, "learning_rate"),
        ("ent_coef", -1.0, "ent_coef"),
        ("max_grad_norm", 0.0, "max_grad_norm"),
        ("target_kl", 0.0, "target_kl"),
        ("checkpoint_every", 0, "checkpoint_every"),
        ("eval_episodes", 0, "eval_episodes"),
        ("eval_every", 0, "eval_every"),
        ("device", "", "device"),
    ],
)
def test_config_rejects_invalid_values(field: str, value: object, match: str) -> None:
    values = dataclasses.asdict(WrenchPPOConfig())
    values[field] = value
    with pytest.raises(ValueError, match=match):
        WrenchPPOConfig(**values)


def test_config_requires_equal_minibatches_of_at_least_two() -> None:
    with pytest.raises(ValueError, match="divisible by num_minibatches"):
        # 64 * 32 = 2048 is not divisible by 3.
        WrenchPPOConfig(num_envs=64, rollout_steps=32, num_minibatches=3)
    with pytest.raises(ValueError, match="minibatch_size"):
        # batch 8 / 8 minibatches = a one-sample minibatch (undefined advantage std).
        WrenchPPOConfig(num_envs=2, rollout_steps=4, num_minibatches=8)
    assert WrenchPPOConfig(num_envs=8, rollout_steps=8, num_minibatches=4).minibatch_size == 16


# --- GAE ----------------------------------------------------------------------------------


def _reference_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    truncated: torch.Tensor,
    final_values: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The textbook GAE recursion with the truncation-only bootstrap written out by hand."""
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last = torch.zeros_like(next_done)
    for t in reversed(range(steps)):
        if t == steps - 1:
            next_not_done = 1.0 - next_done
            next_values = next_value.reshape(-1)
        else:
            next_not_done = 1.0 - dones[t + 1]
            next_values = values[t + 1]
        real_next = next_not_done * next_values + truncated[t] * final_values[t]
        delta = rewards[t] + gamma * real_next - values[t]
        last = delta + gamma * gae_lambda * next_not_done * last
        advantages[t] = last
    return advantages, advantages + values


def test_gae_matches_the_reference_recursion_with_truncation_only_bootstrap() -> None:
    torch.manual_seed(0)
    steps, num_envs = 6, 3
    rewards = torch.randn(steps, num_envs)
    values = torch.randn(steps, num_envs)
    # Episode boundaries: env 0 ends a truncated episode at step 2 and a terminated one at
    # step 4; env 1 only terminates at the last step. `dones[t]` is the done flag BEFORE t.
    dones = torch.zeros(steps, num_envs)
    truncated = torch.zeros(steps, num_envs)
    final_values = torch.zeros(steps, num_envs)
    truncated[2, 0] = 1.0
    final_values[2, 0] = 3.0  # the truncation bootstrap value
    dones[3, 0] = 1.0
    final_values[4, 0] = 7.0  # a TERMINATED episode's last value: must not bootstrap
    truncated[4, 0] = 0.0
    dones[5, 0] = 1.0
    truncated[5, 1] = 1.0
    final_values[5, 1] = -2.0
    dones[5, 1] = 1.0
    next_value = torch.randn(1, num_envs)
    next_done = torch.tensor([0.0, 1.0, 0.0])
    gamma, gae_lambda = 0.99, 0.95

    advantages, returns = compute_advantages(
        rewards=rewards,
        values=values,
        dones=dones,
        truncated=truncated,
        final_values=final_values,
        next_value=next_value,
        next_done=next_done,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    expected_adv, expected_ret = _reference_gae(
        rewards, values, dones, truncated, final_values, next_value, next_done, gamma, gae_lambda
    )
    assert torch.allclose(advantages, expected_adv, atol=1e-6)
    assert torch.allclose(returns, expected_ret, atol=1e-6)


def test_termination_is_terminal_and_truncation_bootstraps() -> None:
    # One step, one env, no future value: a terminated episode's advantage is just r - V, even
    # though the env returns a (meaningless) final value; a truncated one adds gamma * V.
    rewards = torch.tensor([[1.0]])
    values = torch.zeros(1, 1)
    dones = torch.zeros(1, 1)
    final_values = torch.tensor([[10.0]])
    zero = torch.zeros(1, 1)
    next_value = torch.zeros(1, 1)
    next_done = torch.zeros(1)

    terminated_adv, _ = compute_advantages(
        rewards=rewards,
        values=values,
        dones=dones,
        truncated=zero,
        final_values=final_values,
        next_value=next_value,
        next_done=next_done,
        gamma=0.9,
        gae_lambda=1.0,
    )
    truncated_adv, _ = compute_advantages(
        rewards=rewards,
        values=values,
        dones=dones,
        truncated=torch.ones(1, 1),
        final_values=final_values,
        next_value=next_value,
        next_done=next_done,
        gamma=0.9,
        gae_lambda=1.0,
    )
    assert math.isclose(terminated_adv[0, 0].item(), 1.0)
    assert math.isclose(truncated_adv[0, 0].item(), 1.0 + 0.9 * 10.0)


# --- policies -----------------------------------------------------------------------------


@pytest.mark.parametrize(("mode", "expected"), [("none", 26), ("oracle", 35)])
def test_policy_input_width_follows_the_observation_layout(mode: str, expected: int) -> None:
    policies = make_policies(mode)
    assert len(policies) == len(AGENT_KEYS) == 2
    for policy in policies:
        assert policy.actor_mean[0].in_features == expected
        assert policy.critic[0].in_features == expected
        assert policy.actor_mean[-1].out_features == _ACTION_DIM
        assert policy.actor_logstd.shape == (1, _ACTION_DIM)
    # The two policies are independent networks, not one shared module.
    holder, rotator = policies
    assert holder is not rotator
    assert all(a is not b for a, b in zip(holder.parameters(), rotator.parameters(), strict=True))


# --- one PPO step -------------------------------------------------------------------------


def _random_batch(cfg: WrenchPPOConfig, policy: object, obs_dim: int) -> dict:
    """A synthetic rollout batch of one policy, sized like one update."""
    obs = torch.randn(cfg.batch_size, obs_dim)
    with torch.no_grad():
        actions, logprobs, _, values = policy.get_action_and_value(obs)
    return {
        "obs": obs,
        "actions": actions,
        "logprobs": logprobs,
        "advantages": torch.randn(cfg.batch_size),
        "returns": obs[:, 0] * 2.0,
        "values": values.flatten(),
    }


def test_one_ppo_step_keeps_the_loss_finite_and_moves_the_parameters() -> None:
    torch.manual_seed(0)
    cfg = WrenchPPOConfig(
        num_envs=16, rollout_steps=8, num_minibatches=4, update_epochs=2, target_kl=None
    )
    policy = make_policies("none")[0]
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3, eps=1e-5)
    before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
    metrics = ppo_step(policy, optimizer, _random_batch(cfg, policy, cfg.obs_size), cfg)
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"):
        assert key in metrics
        assert math.isfinite(metrics[key])
    assert any(
        not torch.allclose(before[name], parameter) for name, parameter in policy.named_parameters()
    )


# --- checkpoints --------------------------------------------------------------------------


def test_checkpoint_roundtrip_restores_weights_optimizers_and_counters(tmp_path) -> None:
    torch.manual_seed(0)
    cfg = WrenchPPOConfig(
        partner_mode="oracle",
        num_envs=8,
        rollout_steps=8,
        num_minibatches=2,
        update_epochs=1,
        target_kl=None,
        exp_name="roundtrip",
    )
    policies = make_policies(cfg.partner_mode)
    optimizers = [
        torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate, eps=1e-5) for policy in policies
    ]
    for policy, optimizer in zip(policies, optimizers):
        ppo_step(policy, optimizer, _random_batch(cfg, policy, cfg.obs_size), cfg)
    path = tmp_path / "latest.pt"
    save_checkpoint(path, policies, optimizers, cfg, 7, 1234, {"success_rate": 0.5})

    payload = load_checkpoint(path)
    assert payload["format"] == CHECKPOINT_FORMAT
    assert payload["update"] == 7 and payload["global_step"] == 1234
    assert payload["partner_mode"] == "oracle" and payload["obs_size"] == obs_size("oracle")
    assert payload["eval"]["success_rate"] == 0.5
    assert payload["config"]["total_updates"] == cfg.total_updates
    assert set(payload["agents"]) == set(AGENT_KEYS)

    fresh = make_policies(cfg.partner_mode)
    load_weights(payload, fresh)
    for name, policy in zip(AGENT_KEYS, fresh):
        for parameter_name, parameter in policy.state_dict().items():
            assert torch.allclose(parameter, payload["agents"][name][parameter_name])

    fresh_optimizers = [
        torch.optim.Adam(policy.parameters(), lr=1e-4, eps=1e-5) for policy in fresh
    ]
    load_optimizers(payload, fresh_optimizers)
    for name, optimizer in zip(AGENT_KEYS, fresh_optimizers):
        restored = optimizer.state_dict()["state"]
        assert restored, f"{name}'s optimizer state was not restored"
        for key, state in restored.items():
            assert torch.allclose(
                state["exp_avg"], payload["optimizers"][name]["state"][key]["exp_avg"]
            )


def test_missing_checkpoint_file_is_a_clear_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")


# --- greedy evaluation (fake vec env, no simulator) ---------------------------------------


class _FakeVec:
    """Minimal `VecWrenchCube` stand-in that ends every episode after `episode_steps` steps."""

    def __init__(self, num_envs: int, obs_dim: int, episode_steps: int) -> None:
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.config = types.SimpleNamespace(max_control_steps=episode_steps)
        self._episode_steps = episode_steps
        self._elapsed = 0

    def reset(self, seed: int | None = None) -> dict:
        del seed
        self._elapsed = 0
        return self._obs()

    def _obs(self) -> dict:
        return {name: torch.zeros(self.num_envs, self.obs_dim) for name in AGENT_KEYS}

    def step(self, actions: dict) -> types.SimpleNamespace:
        del actions
        self._elapsed += 1
        done = self._elapsed >= self._episode_steps
        finished = self.num_envs if done else 0
        return types.SimpleNamespace(
            obs=self._obs(),
            reward=torch.zeros(self.num_envs),
            terminated=torch.zeros(self.num_envs, dtype=torch.bool),
            truncated=torch.zeros(self.num_envs, dtype=torch.bool),
            final_obs=self._obs(),
            infos=[{} for _ in range(self.num_envs)],
            finished_returns=[1.0] * finished,
            finished_lengths=[self._episode_steps] * finished,
            finished_successes=[True] * finished,
            finished_angles=[90.0] * finished,
        )


def test_evaluate_greedy_aggregates_finished_episodes_and_trims_overrun() -> None:
    cfg = WrenchPPOConfig()
    policies = make_policies("none")
    vec = _FakeVec(num_envs=3, obs_dim=cfg.obs_size, episode_steps=4)
    summary = evaluate_greedy(policies, vec, episodes=5)
    assert isinstance(summary, EvalSummary)
    # 5 finished episodes requested out of rounds of 3: the second round overruns and the
    # summary is trimmed to exactly 5 episodes (not 6).
    assert summary.episodes == 5
    assert summary.success_rate == 1.0
    assert summary.mean_return == 1.0
    assert summary.mean_angle_deg == 90.0
    assert summary.mean_length == 4.0


# --- training loop (toy vector env, no simulator) -----------------------------------------


class _ToyVecWrenchCube:
    """A mujoco-free `VecWrenchCube` stand-in: toy episodes that always succeed in 4 steps.

    It exposes the attributes `wrench_ppo.train` and `evaluate_greedy` rely on (`num_envs`,
    `config.max_control_steps`, `reset`, `step`), with the real auto-reset contract, so the
    update loop, GAE, PPO, greedy evaluation and checkpointing are exercised end to end
    without the sandbox extra.
    """

    def __init__(self, num_envs: int, obs_dim: int, episode_steps: int = 4) -> None:
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.config = types.SimpleNamespace(max_control_steps=episode_steps)
        self._episode_steps = episode_steps
        self._elapsed = 0
        self._episode_return = np.zeros(num_envs)
        self._episode_length = np.zeros(num_envs, dtype=np.int64)

    def reset(self, seed: int | None = None) -> dict:
        del seed
        self._elapsed = 0
        self._episode_return[:] = 0.0
        self._episode_length[:] = 0
        return self._obs()

    def _obs(self) -> dict:
        return {
            name: np.zeros((self.num_envs, self.obs_dim), dtype=np.float32) for name in AGENT_KEYS
        }

    def step(self, actions: dict) -> types.SimpleNamespace:
        del actions
        self._elapsed += 1
        truncated = np.full(self.num_envs, self._elapsed >= self._episode_steps)
        terminated = np.zeros(self.num_envs, dtype=bool)
        reward = np.full(self.num_envs, -0.01)
        done = np.logical_or(terminated, truncated)
        self._episode_return += reward
        self._episode_length += 1
        finished = int(done.sum())
        result = types.SimpleNamespace(
            obs=self._obs(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            final_obs=self._obs(),
            infos=[{} for _ in range(self.num_envs)],
            finished_returns=[float(self._episode_return[0])] * finished,
            finished_lengths=[self._episode_steps] * finished,
            finished_successes=[True] * finished,
            finished_angles=[90.0] * finished,
        )
        self._episode_return[done] = 0.0
        self._episode_length[done] = 0
        return result


def test_training_loop_runs_updates_and_writes_a_resumable_checkpoint(tmp_path) -> None:
    torch.manual_seed(0)
    cfg = WrenchPPOConfig(
        exp_name="toy",
        num_envs=4,
        rollout_steps=8,
        num_minibatches=2,
        update_epochs=1,
        total_updates=2,
        eval_every=1,
        eval_episodes=2,
        eval_envs=2,
        checkpoint_every=1,
        target_kl=None,
    )
    run_dir = tmp_path / "toy"
    run_dir.mkdir()
    torch.manual_seed(cfg.seed)
    train_vec = _ToyVecWrenchCube(cfg.num_envs, cfg.obs_size)
    eval_vec = _ToyVecWrenchCube(cfg.eval_envs, cfg.obs_size)
    wrench_ppo.train(cfg, run_dir, train_vec, eval_vec)

    assert (run_dir / "config.json").is_file()
    payload = load_checkpoint(run_dir / "latest.pt")
    assert payload["update"] == cfg.total_updates
    assert payload["global_step"] == cfg.total_steps
    assert payload["partner_mode"] == cfg.partner_mode
    assert payload["eval"]["success_rate"] == 1.0
    assert any(run_dir.iterdir())


def test_resume_continues_the_update_counter_and_restores_the_weights(tmp_path) -> None:
    torch.manual_seed(0)
    base = {
        "exp_name": "toy",
        "num_envs": 4,
        "rollout_steps": 8,
        "num_minibatches": 2,
        "update_epochs": 1,
        "eval_every": 10,
        "eval_episodes": 2,
        "eval_envs": 2,
        "checkpoint_every": 10,
        "target_kl": None,
    }
    cfg = WrenchPPOConfig(total_updates=1, **base)
    run_dir = tmp_path / "toy"
    run_dir.mkdir()
    wrench_ppo.train(
        cfg,
        run_dir,
        _ToyVecWrenchCube(cfg.num_envs, cfg.obs_size),
        _ToyVecWrenchCube(cfg.eval_envs, cfg.obs_size),
    )
    payload = load_checkpoint(run_dir / "latest.pt")
    assert payload["update"] == 1

    resumed_cfg = wrench_ppo._config_from_payload(payload, run_dir)
    assert resumed_cfg.total_updates == 1 and resumed_cfg.exp_name == "toy"
    assert resumed_cfg.resume == str(run_dir)
    # Extend the run: continue from update 1 up to update 3.
    extended = WrenchPPOConfig(**{**dataclasses.asdict(resumed_cfg), "total_updates": 3})
    wrench_ppo.train(
        extended,
        run_dir,
        _ToyVecWrenchCube(extended.num_envs, extended.obs_size),
        _ToyVecWrenchCube(extended.eval_envs, extended.obs_size),
        payload,
    )
    continued = load_checkpoint(run_dir / "latest.pt")
    assert continued["update"] == 3
    assert continued["global_step"] == 3 * extended.batch_size
