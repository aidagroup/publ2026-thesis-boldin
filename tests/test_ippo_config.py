"""Config and CLI parsing of the IPPO trainer (pure stdlib, runs without torch/mani_skill)."""

import dataclasses
import json
from pathlib import Path

import pytest

from callosum.configs.ippo import (
    IPPOConfig,
    build_parser,
    clamp_envs_for_cpu_backend,
    env_reset_seeds,
    learning_rate_at,
    load_resume_values,
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


def test_lr_anneals_linearly_to_zero_by_default() -> None:
    cfg = IPPOConfig(
        learning_rate=3e-4, num_envs=4, num_steps=100, num_minibatches=4, total_timesteps=4000
    )
    assert cfg.anneal_lr and cfg.num_iterations == 10
    assert learning_rate_at(cfg, 1) == pytest.approx(3e-4)
    assert learning_rate_at(cfg, 6) == pytest.approx(1.5e-4)
    assert learning_rate_at(cfg, 10) == pytest.approx(3e-5)  # last iteration still > 0
    lrs = [learning_rate_at(cfg, k) for k in range(1, cfg.num_iterations + 1)]
    assert lrs == sorted(lrs, reverse=True)


def test_lr_constant_without_annealing() -> None:
    cfg = parse_args(
        ["--no-anneal-lr", "--num-envs", "4", "--num-minibatches", "4", "--total-timesteps", "4000"]
    )
    assert cfg.anneal_lr is False
    assert {learning_rate_at(cfg, k) for k in (1, 5, 10)} == {cfg.learning_rate}


# --- --resume ----------------------------------------------------------------------------


def _write_run(tmp_path: Path, name: str = "run_a", **overrides) -> Path:
    """A run directory with a `config.json` as the trainer writes it (derived keys included)."""
    fields = {
        "env_id": "TwoSO101-v0", "seed": 7, "total_timesteps": 100_000, "num_envs": 64,
        "num_steps": 50, "num_minibatches": 4, "exp_name": name, "runs_dir": "runs",
    }  # fmt: skip
    cfg = IPPOConfig(**{**fields, **overrides})
    saved = dataclasses.asdict(cfg)
    saved.update(
        env_max_episode_steps=100,
        num_eval_steps=100,
        batch_size=cfg.batch_size,
        minibatch_size=cfg.minibatch_size,
        num_iterations=cfg.num_iterations,
        obs_dims=[40, 44],
    )
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(saved, indent=2))
    return run_dir


def test_resume_uses_the_saved_config(tmp_path) -> None:
    run_dir = _write_run(tmp_path, learning_rate=1e-4, target_kl=None, partner_obs="none")
    cfg = parse_args(["--resume", str(run_dir)])
    assert cfg.resume == str(run_dir)
    assert (cfg.env_id, cfg.seed, cfg.total_timesteps) == ("TwoSO101-v0", 7, 100_000)
    assert (cfg.num_envs, cfg.num_steps, cfg.num_minibatches) == (64, 50, 4)
    assert cfg.learning_rate == 1e-4 and cfg.target_kl is None and cfg.partner_obs == "none"
    assert cfg.num_iterations == 100_000 // (64 * 50)


def test_resume_run_dir_is_the_given_dir(tmp_path) -> None:
    run_dir = _write_run(tmp_path, name="orig")
    renamed = run_dir.rename(tmp_path / "renamed")  # config.json still says exp_name=orig
    cfg = parse_args(["--resume", str(renamed)])
    assert Path(cfg.runs_dir) / cfg.exp_name == renamed
    assert cfg.resume == str(renamed)
    # A trailing slash does not change it either.
    cfg = parse_args(["--resume", str(renamed) + "/"])
    assert Path(cfg.runs_dir) / cfg.exp_name == renamed


def test_resume_without_saved_exp_name(tmp_path) -> None:
    run_dir = _write_run(tmp_path)
    saved = json.loads((run_dir / "config.json").read_text())
    saved["exp_name"] = None  # a timestamped default name: the directory still decides
    (run_dir / "config.json").write_text(json.dumps(saved))
    cfg = parse_args(["--resume", str(run_dir)])
    assert cfg.exp_name == run_dir.name and cfg.runs_dir == str(tmp_path)


def test_resume_accepts_flags_equal_to_the_saved_values(tmp_path) -> None:
    run_dir = _write_run(tmp_path)
    cfg = parse_args(
        ["--resume", str(run_dir), "--seed", "7", "--env-id", "TwoSO101-v0", "--num-envs", "64",
         "--exp-name", run_dir.name, "--runs-dir", str(tmp_path), "--anneal-lr"]
    )  # fmt: skip
    assert cfg.seed == 7 and cfg.num_envs == 64


def test_resume_rejects_conflicting_flags(tmp_path) -> None:
    run_dir = _write_run(tmp_path)
    with pytest.raises(SystemExit):
        parse_args(["--resume", str(run_dir), "--seed", "8"])
    with pytest.raises(ValueError, match="--seed 8 .*--learning-rate 0.1") as info:
        load_resume_values(run_dir, {"seed": 8, "learning_rate": 0.1, "num_envs": 64})
    assert "--num-envs" not in str(info.value)  # equal to the saved value: no conflict
    with pytest.raises(ValueError, match="--exp-name other"):
        load_resume_values(run_dir, {"exp_name": "other"})
    with pytest.raises(ValueError, match="--runs-dir"):
        load_resume_values(run_dir, {"runs_dir": str(tmp_path / "elsewhere")})


def test_resume_rejects_checkpoint_and_missing_config(tmp_path) -> None:
    run_dir = _write_run(tmp_path)
    with pytest.raises(SystemExit):
        parse_args(["--resume", str(run_dir), "--checkpoint", "warm.pt"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_resume_values(run_dir, {"checkpoint": "warm.pt"})
    with pytest.raises(SystemExit):
        parse_args(["--resume", str(tmp_path / "does_not_exist")])


def test_resume_ignores_unknown_saved_keys_and_clamps_cpu_flags(tmp_path) -> None:
    run_dir = _write_run(tmp_path, sim_backend="cpu", num_envs=1, num_eval_envs=1, num_steps=100)
    saved = json.loads((run_dir / "config.json").read_text())
    saved["some_future_key"] = 1
    (run_dir / "config.json").write_text(json.dumps(saved))
    assert "obs_dims" not in load_resume_values(run_dir, {})
    # Re-running the original CPU command (which asked for 256 envs, clamped to 1) resumes.
    cfg = parse_args(["--resume", str(run_dir), "--sim-backend", "cpu", "--num-envs", "256"])
    assert cfg.num_envs == 1 and cfg.sim_backend == "cpu"


def test_env_reset_seeds() -> None:
    cfg = IPPOConfig(seed=5)
    assert env_reset_seeds(cfg, 0) == (5, 6)  # a fresh run keeps the historical seeds
    train, evaluation = env_reset_seeds(cfg, 40)
    assert (train, evaluation) != (5, 6) and train != evaluation
    assert env_reset_seeds(cfg, 40) != env_reset_seeds(cfg, 41)
