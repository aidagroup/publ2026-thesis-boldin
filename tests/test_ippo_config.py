"""Config and CLI parsing of the IPPO trainer (pure stdlib, runs without torch/mani_skill)."""

import pytest

from callosum.configs.ippo import (
    IPPOConfig,
    build_parser,
    clamp_envs_for_cpu_backend,
    parse_args,
    resolve_episode_length,
)


def test_defaults_are_consistent() -> None:
    cfg = IPPOConfig()
    assert cfg.num_envs == 256
    assert cfg.batch_size == cfg.num_envs * cfg.num_steps
    assert cfg.minibatch_size == cfg.batch_size // cfg.num_minibatches
    assert cfg.num_iterations == cfg.total_timesteps // cfg.batch_size
    assert cfg.reward_mode == "normalized_dense"
    assert cfg.gamma == 0.99  # not the baseline's 0.8, see the field comment


def test_every_field_has_a_flag() -> None:
    parser = build_parser()
    flags = {a.dest for a in parser._actions}
    assert {"env_id", "sim_backend", "num_envs", "partner_obs", "max_episode_steps"} <= flags
    assert {"anneal_lr", "target_kl", "exp_name"} <= flags


def test_parse_overrides() -> None:
    cfg = parse_args(
        [
            "--env-id", "TwoSO101-v0", "--num-envs", "64", "--num-steps", "50",
            "--total-timesteps", "100000", "--gamma", "0.95", "--anneal-lr",
            "--partner-obs", "none", "--exp-name", "run1", "--target-kl", "none",
            "--max-episode-steps", "200", "--no-partial-reset",
        ]
    )  # fmt: skip
    assert (cfg.env_id, cfg.num_envs, cfg.num_steps) == ("TwoSO101-v0", 64, 50)
    assert cfg.gamma == 0.95 and cfg.anneal_lr is True and cfg.partial_reset is False
    assert cfg.partner_obs == "none" and cfg.exp_name == "run1"
    assert cfg.target_kl is None and cfg.max_episode_steps == 200


def test_parse_defaults_equal_dataclass_defaults() -> None:
    assert parse_args([]) == IPPOConfig()
    # Default: use the env's registered episode length (400 FaceTurn-v0, 100 TwoSO101-v0).
    assert IPPOConfig().max_episode_steps is None


def test_max_episode_steps_none_keeps_env_registration() -> None:
    assert parse_args(["--max-episode-steps", "none"]).max_episode_steps is None


def test_rejects_bad_values() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--partner-obs", "predicted"])
    with pytest.raises(SystemExit):
        parse_args(["--sim-backend", "tpu"])
    with pytest.raises(ValueError, match="minibatch_size"):
        IPPOConfig(num_envs=1, num_steps=8, num_minibatches=32)
    with pytest.raises(ValueError, match="total_timesteps"):
        IPPOConfig(total_timesteps=10)


def test_batch_must_divide_into_equal_minibatches() -> None:
    # 1 * 101 // 4 = 25 would leave a 1-sample last minibatch (advantage std is NaN).
    with pytest.raises(ValueError, match="divisible by num_minibatches") as err:
        IPPOConfig(num_envs=1, num_steps=101, num_minibatches=4, total_timesteps=1000)
    assert "Valid num_minibatches: [1]" in str(err.value)
    with pytest.raises(ValueError, match=r"Valid num_minibatches: \[1, 2, 3, 4, 6, 8, 12\]"):
        IPPOConfig(num_envs=1, num_steps=24, num_minibatches=5, total_timesteps=1000)
    assert IPPOConfig(num_envs=1, num_steps=100, num_minibatches=4).minibatch_size == 25


def test_cpu_defaults_need_a_dividing_num_minibatches() -> None:
    # The CPU clamp makes the effective batch 1 * num_steps, which is what gets validated.
    with pytest.raises(ValueError, match="divisible by num_minibatches"):
        parse_args(["--sim-backend", "cpu", "--num-steps", "100", "--total-timesteps", "300"])


def test_eval_must_cover_a_full_episode() -> None:
    with pytest.raises(ValueError, match="full episode"):
        IPPOConfig(max_episode_steps=300, num_eval_steps=50)
    assert resolve_episode_length(IPPOConfig(), 300) == 300
    assert resolve_episode_length(IPPOConfig(num_eval_steps=400), 300) == 400
    # With the env's registered length (max_episode_steps=None) the check happens at run time.
    cfg = IPPOConfig(max_episode_steps=None, num_eval_steps=50)
    with pytest.raises(ValueError, match="full episode"):
        resolve_episode_length(cfg, 100)


def test_cpu_backend_clamps_env_counts(capsys: pytest.CaptureFixture) -> None:
    cfg = parse_args(
        ["--sim-backend", "cpu", "--total-timesteps", "300", "--num-steps", "100"]
        + ["--num-minibatches", "4"]
    )
    assert (cfg.num_envs, cfg.num_eval_envs) == (1, 1)
    assert "single env" in capsys.readouterr().out
    values = {"sim_backend": "gpu", "num_envs": 256}
    assert clamp_envs_for_cpu_backend(values) is None and values["num_envs"] == 256
    values = {"sim_backend": "cpu", "num_envs": 1, "num_eval_envs": 1}
    assert clamp_envs_for_cpu_backend(values) is None
