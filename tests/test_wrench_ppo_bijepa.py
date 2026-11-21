"""Bi-JEPA integration tests for `wrench_ppo` (PLAN.md step 3): pure torch, no mujoco.

Covers the `predicted` ablation arm end to end without the simulator:

* the policy input widths of the three input modes (28 / 39 / 28+32 = 60);
* the Bi-JEPA auxiliary loss is finite and its gradients reach the online encoder/predictor
  only (the EMA target encoders are stop-gradient and are never optimised);
* the `ẑ_partner` policy input is built from the agent's OWN observation history only, so
  ground-truth partner data cannot leak into the `predicted` execution path;
* checkpointing a `predicted` run (Bi-JEPA weights + optimizer + input-mode metadata) and the
  checkpoint/input-mode compatibility checks, including a legacy step-2 checkpoint;
* a toy-vector-env `train()` smoke run with `use_bijepa=True` (two updates: finite losses, a
  written checkpoint with the Bi-JEPA block);
* the closed-form ridge probe used for the report's "prediction quality" numbers.

Skipped when torch is absent (CI installs only the `dev` extra).
"""

import dataclasses
import math
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from callosum.configs.wrench_ppo import INPUT_MODES, WrenchPPOConfig
from callosum.envs._wrench_logic import AGENT_KEYS, obs_size
from callosum.training import wrench_ppo
from callosum.training._bijepa_probe import ridge_probe
from callosum.training.wrench_ppo import (
    LatentInputBuilder,
    aux_update,
    check_checkpoint_compat,
    compute_aux_loss,
    config_from_payload,
    env_partner_mode,
    load_bijepa_optimizer,
    load_bijepa_weights,
    load_checkpoint,
    load_weights,
    make_bijepa,
    make_bijepa_optimizer,
    make_policies,
    ppo_step,
    save_checkpoint,
)

_ACTION_DIM = 6


# --- helpers ------------------------------------------------------------------------------


def _cfg(**overrides: object) -> WrenchPPOConfig:
    """A small `predicted` config that runs a full update on a 64-transition batch."""
    base: dict[str, object] = {
        "use_bijepa": True,
        "num_envs": 8,
        "rollout_steps": 8,
        "num_minibatches": 2,
        "update_epochs": 1,
        "target_kl": None,
        "exp_name": "bijepa_test",
    }
    return WrenchPPOConfig(**{**base, **overrides})


def _histories(cfg: WrenchPPOConfig, batch: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic own-observation histories of the holder and the rotator."""
    return (
        torch.randn(batch, cfg.history_len, cfg.env_obs_size),
        torch.randn(batch, cfg.history_len, cfg.env_obs_size),
    )


def _random_policy_batch(cfg: WrenchPPOConfig, policy: object, obs_dim: int) -> dict:
    """A synthetic PPO rollout batch of one policy, sized like one update."""
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


# --- (a) input widths of the three modes --------------------------------------------------


def test_input_widths_of_none_oracle_and_predicted() -> None:
    baseline = WrenchPPOConfig()
    assert baseline.input_mode == "none"
    assert baseline.env_obs_size == baseline.obs_size == obs_size("none") == 28
    assert baseline.policy_input_size == 28

    oracle = WrenchPPOConfig(partner_mode="oracle")
    assert oracle.input_mode == "oracle"
    assert oracle.env_obs_size == obs_size("oracle") == 45
    assert oracle.policy_input_size == 45

    predicted = WrenchPPOConfig(use_bijepa=True)
    assert predicted.input_mode == "predicted"
    assert predicted.partner_mode == "none"
    assert predicted.env_obs_size == 28
    assert predicted.latent_dim == 32
    assert predicted.policy_input_size == 28 + 32 == 60
    assert WrenchPPOConfig(use_bijepa=True, latent_dim=16).policy_input_size == 44

    for policies, expected in (
        (make_policies("none"), 28),
        (make_policies("oracle"), 45),
        (make_policies("none", latent_dim=32), 60),
    ):
        for policy in policies:
            assert policy.actor_mean[0].in_features == expected
            assert policy.critic[0].in_features == expected
            assert policy.actor_mean[-1].out_features == _ACTION_DIM

    assert env_partner_mode("predicted") == "none"
    assert env_partner_mode("none") == "none"
    assert env_partner_mode("oracle") == "oracle"
    with pytest.raises(ValueError, match="input_mode"):
        env_partner_mode("predicted-direct")
    assert set(INPUT_MODES) == {"none", "oracle", "predicted"}


def test_predicted_requires_the_unprivileged_env_and_valid_bijepa_fields() -> None:
    with pytest.raises(ValueError, match="partner_mode"):
        WrenchPPOConfig(use_bijepa=True, partner_mode="oracle")
    for bad in ({"latent_dim": 0}, {"history_len": 0}, {"ema_momentum": 1.0}):
        with pytest.raises(ValueError):
            WrenchPPOConfig(use_bijepa=True, **bad)
    with pytest.raises(ValueError, match="aux_coef"):
        WrenchPPOConfig(use_bijepa=True, aux_coef=-0.1)
    # The non-Bi-JEPA config still validates through the same block (defaults are valid).
    assert WrenchPPOConfig().bijepa_config().ema_target is True


# --- (b) auxiliary loss: finite, gradients reach online modules only ----------------------


def test_aux_loss_is_finite_and_weighted_by_aux_coef() -> None:
    torch.manual_seed(0)
    cfg = _cfg(aux_coef=0.25)
    bijepa = make_bijepa(cfg)
    assert bijepa is not None
    hist_a, hist_b = _histories(cfg)
    terms = compute_aux_loss(bijepa, cfg, hist_a, hist_b)
    assert terms["loss"].ndim == 0 and terms["loss"].requires_grad
    for key in ("loss", "mse", "var", "cov", "mse_a", "mse_b"):
        assert torch.isfinite(terms[key]), key
    assert terms["z_pred_a"].shape == terms["z_target_a"].shape == (64, cfg.latent_dim)
    assert not terms["z_target_a"].requires_grad
    with torch.no_grad():
        expected = cfg.aux_coef * bijepa.forward_both(hist_a, hist_b)["loss"]
    assert torch.allclose(terms["loss"].detach(), expected)


def test_aux_gradients_flow_to_online_encoders_and_predictors_only() -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    bijepa = make_bijepa(cfg)
    assert bijepa is not None
    hist_a, hist_b = _histories(cfg)
    terms = compute_aux_loss(bijepa, cfg, hist_a, hist_b)
    terms["loss"].backward()

    for parameter in bijepa.encoders.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    for parameter in bijepa.predictors.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert any(p.grad.abs().sum() > 0 for p in bijepa.encoders.parameters())
    assert any(p.grad.abs().sum() > 0 for p in bijepa.predictors.parameters())
    # The EMA target encoders are stop-gradient: no path reaches them.
    assert bijepa.target_encoders is not None
    for parameter in bijepa.target_encoders.parameters():
        assert not parameter.requires_grad
        assert parameter.grad is None
    # The observation histories are inputs, not parameters: they receive no gradient.
    assert hist_a.grad is None and hist_b.grad is None


def test_aux_update_logs_the_required_tags_and_advances_online_and_ema_weights() -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    bijepa = make_bijepa(cfg)
    optimizer = make_bijepa_optimizer(cfg, bijepa)
    assert bijepa is not None and optimizer is not None
    hist_a, hist_b = _histories(cfg)
    online_before = [p.detach().clone() for p in bijepa.encoders.parameters()]
    target_before = [p.detach().clone() for p in bijepa.target_encoders.parameters()]

    metrics = aux_update(bijepa, optimizer, cfg, hist_a, hist_b)
    required = {
        "aux/loss",
        "aux/mse",
        "aux/var",
        "aux/cov",
        "aux/mse_a",
        "aux/mse_b",
        "latent/std",
        "latent/std_min",
        "latent/effective_rank",
        "aux/probe/wrench_mse",
        "aux/probe/wrench_r2",
        "aux/probe/wrench_baseline_mse",
        "aux/probe/latent_mse",
        "aux/probe/latent_r2",
    }
    assert required <= set(metrics)
    assert all(math.isfinite(value) for value in metrics.values()), metrics
    assert 1.0 <= metrics["latent/effective_rank"] <= float(cfg.latent_dim)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(online_before, bijepa.encoders.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(target_before, bijepa.target_encoders.parameters(), strict=True)
    )


# --- (c) predicted input carries no ground-truth partner data -----------------------------

_CHANGED_PARTNER = 100.0


def test_predicted_input_does_not_use_the_partner_observation() -> None:
    torch.manual_seed(0)
    cfg = _cfg(latent_dim=8)
    bijepa = make_bijepa(cfg)
    assert bijepa is not None
    builder = LatentInputBuilder(bijepa)
    num_envs = 3
    obs = {name: torch.randn(num_envs, cfg.env_obs_size) for name in AGENT_KEYS}
    inputs = builder.reset(obs)
    assert builder.input_size == cfg.env_obs_size + cfg.latent_dim == 36
    for name in AGENT_KEYS:
        assert inputs[name].shape == (num_envs, builder.input_size)
        # The own part of the input is the own current observation, verbatim.
        assert torch.allclose(inputs[name][:, : cfg.env_obs_size], obs[name])

    # Each agent's predicted input is a function of its OWN observation only: corrupting the
    # partner's observation leaves it unchanged, changing the own one moves the latent part.
    for index, name in enumerate(AGENT_KEYS):
        partner = AGENT_KEYS[1 - index]
        corrupted = {key: value.clone() for key, value in obs.items()}
        corrupted[partner] += _CHANGED_PARTNER
        assert torch.allclose(inputs[name], builder.reset(corrupted)[name])
        own_changed = {key: value.clone() for key, value in obs.items()}
        own_changed[name] += _CHANGED_PARTNER
        assert not torch.allclose(inputs[name], builder.reset(own_changed)[name])

    # The same holds through step(), the path the rollout loop uses every control step.
    builder.reset(obs)
    stepped = builder.step(obs, torch.zeros(num_envs))
    builder.reset(obs)
    partner_changed = dict(obs)
    partner_changed["rotator"] = obs["rotator"] + _CHANGED_PARTNER
    stepped_partner_changed = builder.step(partner_changed, torch.zeros(num_envs))
    assert torch.allclose(stepped["holder"], stepped_partner_changed["holder"])


def test_builder_resets_the_history_window_at_episode_boundaries() -> None:
    torch.manual_seed(0)
    cfg = _cfg(latent_dim=8, history_len=2)
    bijepa = make_bijepa(cfg)
    assert bijepa is not None
    builder = LatentInputBuilder(bijepa)
    num_envs = 2
    obs = {name: torch.randn(num_envs, cfg.env_obs_size) for name in AGENT_KEYS}
    builder.reset(obs)
    next_obs = {name: torch.randn(num_envs, cfg.env_obs_size) for name in AGENT_KEYS}
    done = torch.tensor([1.0, 0.0])
    builder.step(next_obs, done)
    for name in AGENT_KEYS:
        # Env 0 finished: its window is the fresh observation repeated.
        fresh = next_obs[name][0].unsqueeze(0).repeat(cfg.history_len, 1)
        assert torch.allclose(builder.history[name][0], fresh)
        # Env 1 continued: the window rolled (oldest observation replaced).
        assert not torch.allclose(
            builder.history[name][1], next_obs[name][1].unsqueeze(0).repeat(2, 1)
        )
    assert builder.step(next_obs, torch.zeros(num_envs)).get("holder") is not None
    # `history_inputs` must not commit the hypothetical observation to the history.
    before = {name: builder.history[name].clone() for name in AGENT_KEYS}
    builder.history_inputs(next_obs)
    for name in AGENT_KEYS:
        assert torch.allclose(before[name], builder.history[name])


# --- (d) checkpoint roundtrip and compatibility checks ------------------------------------


def test_predicted_checkpoint_roundtrip_restores_bijepa_and_metadata(tmp_path) -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    policies = make_policies(cfg.partner_mode, latent_dim=cfg.latent_dim)
    optimizers = [
        torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate, eps=1e-5) for policy in policies
    ]
    for policy, optimizer in zip(policies, optimizers):
        ppo_step(policy, optimizer, _random_policy_batch(cfg, policy, cfg.policy_input_size), cfg)
    bijepa = make_bijepa(cfg)
    bijepa_optimizer = make_bijepa_optimizer(cfg, bijepa)
    assert bijepa is not None and bijepa_optimizer is not None
    hist_a, hist_b = _histories(cfg)
    aux_update(bijepa, bijepa_optimizer, cfg, hist_a, hist_b)

    path = tmp_path / "latest.pt"
    save_checkpoint(
        path,
        policies,
        optimizers,
        cfg,
        3,
        99,
        {"success_rate": 0.25},
        bijepa=bijepa,
        bijepa_optimizer=bijepa_optimizer,
    )
    payload = load_checkpoint(path)
    assert payload["format"] == wrench_ppo.CHECKPOINT_FORMAT
    assert payload["input_mode"] == "predicted"
    assert payload["partner_mode"] == "none"
    assert payload["use_bijepa"] is True
    assert payload["obs_size"] == obs_size("none") == 28
    assert payload["policy_input_size"] == 60
    assert payload["latent_dim"] == cfg.latent_dim
    assert payload["config"]["use_bijepa"] is True
    assert payload["bijepa"] and payload["bijepa_optimizer"]
    assert check_checkpoint_compat(payload, "predicted") == ("none", cfg.latent_dim)
    # The checkpoint and both non-predicted modes disagree on the policy input mode/width.
    for wrong_mode in ("none", "oracle"):
        with pytest.raises(ValueError, match="input_mode"):
            check_checkpoint_compat(payload, wrong_mode)

    fresh_policies = make_policies("none", latent_dim=cfg.latent_dim)
    load_weights(payload, fresh_policies)
    for name, policy in zip(AGENT_KEYS, fresh_policies):
        for key, value in policy.state_dict().items():
            assert torch.allclose(value, payload["agents"][name][key])

    fresh_bijepa = make_bijepa(cfg)
    assert fresh_bijepa is not None
    load_bijepa_weights(payload, fresh_bijepa)
    for key, value in bijepa.state_dict().items():
        assert torch.allclose(value, fresh_bijepa.state_dict()[key])
    fresh_optimizer = make_bijepa_optimizer(cfg, fresh_bijepa)
    assert fresh_optimizer is not None
    load_bijepa_optimizer(payload, fresh_optimizer)
    assert fresh_optimizer.state_dict()["state"]

    restored = config_from_payload(payload)
    assert restored.use_bijepa is True
    assert restored.latent_dim == cfg.latent_dim
    assert restored.input_mode == "predicted"
    assert restored.policy_input_size == 60


def test_legacy_step_two_checkpoint_stays_loadable(tmp_path) -> None:
    torch.manual_seed(0)
    cfg = WrenchPPOConfig(
        num_envs=8, rollout_steps=8, num_minibatches=2, update_epochs=1, target_kl=None
    )
    policies = make_policies("none")
    optimizers = [
        torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate, eps=1e-5) for policy in policies
    ]
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, policies, optimizers, cfg, 1, 64)
    payload = load_checkpoint(path)
    # Simulate a genuine step-2 checkpoint: no Bi-JEPA block and none of the new metadata keys.
    legacy_keys = ("input_mode", "use_bijepa", "policy_input_size", "latent_dim")
    payload = {key: value for key, value in payload.items() if key not in legacy_keys}
    payload["config"] = {
        key: value for key, value in payload["config"].items() if key not in legacy_keys
    }
    # The legacy keys (`partner_mode`, `obs_size`) stay meaningful for the evaluator.
    assert check_checkpoint_compat(payload, "none") == ("none", 0)
    with pytest.raises(ValueError, match="no Bi-JEPA"):
        check_checkpoint_compat(payload, "predicted")
    restored = config_from_payload(payload)
    assert restored.use_bijepa is False
    assert restored.policy_input_size == 28
    assert restored.input_mode == "none"


# --- (e) training smoke on a toy vector env (no mujoco) -----------------------------------


class _ToyVecWrenchCube:
    """A mujoco-free `VecWrenchCube` stand-in exposing only the env's 28-d observation.

    It follows the real auto-reset contract: an episode truncates after `episode_steps` and
    the next observation belongs to a fresh episode, so the trainer's rollout loop, the
    Bi-JEPA history bookkeeping and the greedy evaluation all run end to end.
    """

    def __init__(self, num_envs: int, obs_dim: int, episode_steps: int = 4) -> None:
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.config = types.SimpleNamespace(max_control_steps=episode_steps)
        self._episode_steps = episode_steps
        self._elapsed = 0
        self._counter = 0
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
        self._counter += 1
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


@pytest.mark.parametrize("history_len", [1, 2])
def test_train_smoke_with_bijepa_writes_finite_losses_and_a_predicted_checkpoint(
    tmp_path, history_len: int
) -> None:
    torch.manual_seed(0)
    cfg = _cfg(
        num_envs=4,
        rollout_steps=8,
        history_len=history_len,
        eval_every=1,
        eval_episodes=2,
        eval_envs=2,
        checkpoint_every=1,
        total_updates=2,
    )
    run_dir = tmp_path / f"toy_bijepa_k{history_len}"
    run_dir.mkdir()
    torch.manual_seed(cfg.seed)
    train_vec = _ToyVecWrenchCube(cfg.num_envs, cfg.env_obs_size)
    eval_vec = _ToyVecWrenchCube(cfg.eval_envs, cfg.env_obs_size)
    wrench_ppo.train(cfg, run_dir, train_vec, eval_vec)

    payload = load_checkpoint(run_dir / "latest.pt")
    assert payload["update"] == cfg.total_updates == 2
    assert payload["global_step"] == cfg.total_steps
    assert payload["input_mode"] == "predicted"
    assert payload["policy_input_size"] == 60
    assert payload["bijepa"] and payload["bijepa_optimizer"]
    assert "latest.pt" in {path.name for path in run_dir.iterdir()}
    config = (run_dir / "config.json").read_text()
    assert '"input_mode": "predicted"' in config
    assert '"policy_input_size": 60' in config


# --- (f) the ridge probe ----------------------------------------------------------------


def test_ridge_probe_recovers_a_linear_signal_and_reports_baselines() -> None:
    torch.manual_seed(0)
    features = torch.randn(400, 8)
    weights = torch.randn(8, 6)
    targets = features @ weights + 0.01 * torch.randn(400, 6)
    metrics = ridge_probe(features, targets)
    assert {"mse", "r2", "baseline_mse", "n_train", "n_test"} == set(metrics)
    assert metrics["r2"] > 0.95
    assert metrics["mse"] < metrics["baseline_mse"]
    assert metrics["n_train"] + metrics["n_test"] == 400
    assert ridge_probe(features, targets)["r2"] == metrics["r2"]  # deterministic split
    # A linear probe of pure noise cannot beat the target mean.
    noise = ridge_probe(features, torch.randn(400, 6))
    assert noise["r2"] < 0.1
    # Constant held-out targets: nothing to explain, reported as a perfect probe.
    constant = ridge_probe(features, torch.full((400, 6), 3.0))
    assert constant["r2"] == 1.0 and constant["mse"] == 0.0


def test_ridge_probe_rejects_bad_shapes_and_arguments() -> None:
    features = torch.randn(16, 4)
    with pytest.raises(ValueError, match="row counts"):
        ridge_probe(features, torch.randn(8, 4))
    with pytest.raises(ValueError, match="2-D"):
        ridge_probe(torch.randn(16), torch.randn(16))
    with pytest.raises(ValueError, match="at least 2"):
        ridge_probe(torch.randn(1, 4), torch.randn(1, 2))
    with pytest.raises(ValueError, match="test_fraction"):
        ridge_probe(features, torch.randn(16, 2), test_fraction=1.0)
    with pytest.raises(ValueError, match="ridge"):
        ridge_probe(features, torch.randn(16, 2), ridge=-1.0)


def test_checkpoint_config_matches_the_dataclass_fields() -> None:
    """The saved config must be reconstructible from the dataclass fields (resume/eval path)."""
    cfg = _cfg()
    saved = {key: value for key, value in dataclasses.asdict(cfg).items()}
    assert saved["use_bijepa"] is True
    assert saved["latent_dim"] == 32
    assert saved["history_len"] == 1
    assert saved["aux_coef"] == 0.1
    assert saved["ema_momentum"] == 0.99
    assert saved["var_weight"] == 1.0 and saved["cov_weight"] == 0.04
