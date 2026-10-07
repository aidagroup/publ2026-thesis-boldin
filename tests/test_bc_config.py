"""Config / CLI of the BC pretraining, the expert's control config and the IPPO fine-tune flags
(pure stdlib, runs without torch / mani_skill)."""

import math

import pytest

from callosum.configs.bc import BCConfig, parse_args
from callosum.configs.face_turn_expert import ExpertControlConfig
from callosum.configs.ippo import (
    IPPOConfig,
    bc_coef_at,
    build_parser,
    critic_warmup_active,
    load_resume_values,
)
from callosum.configs.ippo import (
    parse_args as parse_ippo_args,
)


def test_bc_defaults_and_required_demos() -> None:
    cfg = BCConfig(demos="d.pt")
    assert cfg.partner_obs == "full" and cfg.actor_logstd == pytest.approx(-1.6)
    assert cfg.gamma == IPPOConfig().gamma  # the critic targets must use the trainer's gamma
    with pytest.raises(ValueError, match="demos is required"):
        BCConfig()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"val_fraction": 1.0},
        {"val_fraction": -0.1},
        {"gamma": 0.0},
        {"partner_obs": "everything"},
        {"device": "tpu"},
    ],
)
def test_bc_config_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        BCConfig(demos="d.pt", **kwargs)


def test_bc_cli_parses_flags() -> None:
    cfg = parse_args(
        ["--demos", "a.pt", "--partner-obs", "none", "--epochs", "5", "--no-anneal-lr",
         "--actor-logstd", "-1.2", "--exp-name", "x"]
    )  # fmt: skip
    assert (cfg.demos, cfg.partner_obs, cfg.epochs) == ("a.pt", "none", 5)
    assert cfg.anneal_lr is False and cfg.actor_logstd == -1.2 and cfg.exp_name == "x"
    with pytest.raises(SystemExit):  # --demos is missing
        parse_args(["--epochs", "5"])


def test_expert_control_config_validation() -> None:
    assert ExpertControlConfig().control == "pos"  # the probe's original behaviour
    ExpertControlConfig(control="delta", overlap_approach=True)
    for kwargs in ({"control": "vel"}, {"waypoint_tol": 0.0}, {"final_timeout": 0}, {"gain": -1}):
        with pytest.raises(ValueError):
            ExpertControlConfig(**kwargs)


def test_bc_coef_schedule_decays_linearly_by_iteration() -> None:
    cfg = IPPOConfig(demos="d.pt", bc_coef=2.0, bc_decay_iters=10)
    assert bc_coef_at(cfg, 1) == pytest.approx(2.0)
    assert bc_coef_at(cfg, 6) == pytest.approx(1.0)
    assert bc_coef_at(cfg, 11) == 0.0 and bc_coef_at(cfg, 500) == 0.0
    assert math.isclose(bc_coef_at(cfg, 2), 2.0 * 0.9)


def test_bc_coef_is_off_by_default_and_without_demos() -> None:
    assert bc_coef_at(IPPOConfig(), 1) == 0.0
    assert bc_coef_at(IPPOConfig(demos="d.pt"), 1) == 0.0  # coef 0
    assert critic_warmup_active(IPPOConfig(), 1) is False


def test_critic_warmup_window_is_by_iteration() -> None:
    cfg = IPPOConfig(critic_warmup_iters=3)
    assert [critic_warmup_active(cfg, k) for k in (1, 2, 3, 4)] == [True, True, True, False]


def test_ippo_finetune_flag_validation() -> None:
    with pytest.raises(ValueError, match="needs demos"):
        IPPOConfig(bc_coef=1.0)
    with pytest.raises(ValueError, match="checkpoint"):
        IPPOConfig(eval_only=True)
    with pytest.raises(ValueError, match=">= 0"):
        IPPOConfig(critic_warmup_iters=-1)
    with pytest.raises(ValueError, match=">= 1"):
        IPPOConfig(demos="d.pt", bc_coef=1.0, bc_decay_iters=0)
    # Eval-only runs do not need a valid PPO batch layout (the CPU sim has one env).
    IPPOConfig(eval_only=True, checkpoint="c.pt", num_envs=1, num_eval_envs=1)


def test_ippo_defaults_leave_behaviour_unchanged() -> None:
    cfg = IPPOConfig()
    assert (cfg.critic_warmup_iters, cfg.bc_coef, cfg.demos, cfg.eval_only) == (0, 0.0, None, False)
    flags = {a.dest for a in build_parser()._actions}
    assert {"critic_warmup_iters", "demos", "bc_coef", "bc_decay_iters", "eval_only"} <= flags
    parsed = parse_ippo_args(
        [
            "--checkpoint",
            "bc.pt",
            "--demos",
            "d.pt",
            "--bc-coef",
            "0.5",
            "--critic-warmup-iters",
            "5",
        ]
    )
    assert parsed.bc_coef == 0.5 and parsed.critic_warmup_iters == 5 and parsed.demos == "d.pt"


def test_resume_keeps_the_saved_finetune_flags(tmp_path) -> None:
    import dataclasses
    import json

    saved = IPPOConfig(demos="d.pt", bc_coef=1.5, bc_decay_iters=20, critic_warmup_iters=4)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(saved)))
    values = load_resume_values(run_dir, {"resume": str(run_dir)})
    cfg = IPPOConfig(**values)
    assert (cfg.bc_coef, cfg.bc_decay_iters, cfg.critic_warmup_iters) == (1.5, 20, 4)
    # The schedule is a pure function of the iteration, so a resumed run continues it.
    assert bc_coef_at(cfg, 11) == pytest.approx(1.5 * 0.5)
    with pytest.raises(ValueError, match="conflict"):
        load_resume_values(run_dir, {"resume": str(run_dir), "bc_coef": 3.0})
