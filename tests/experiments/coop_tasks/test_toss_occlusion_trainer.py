"""Trainer/CLI wiring tests of the OccludedToss-v0 hidden-partner variant.

The hidden-partner task is the same toss-catch env with `TossConfig(occlusion="partner")`; the
trainer side (PLAN.md "Hidden-partner regime") has to (a) expose `--occlusion` only for
`--task toss`, (b) size the policies from the masked observation width (20/30 instead of
26/36, so the Bi-JEPA `predicted` input is 20 + latent_dim), (c) build the env config with the
flag and (d) judge a checkpoint against the SAME occlusion layout it was trained under.

The config/CLI tests import `callosum.experiments.cube_wrench.ppo_config` only (stdlib + the
pure Bi-JEPA config), so they run in CI; the checkpoint-compatibility tests import the trainer
module (torch) and are skipped when the `train` extra is absent.
"""

import pytest

from callosum.experiments.coop_tasks.config import coop_obs_size
from callosum.experiments.cube_wrench.ppo_config import (
    WrenchPPOConfig,
    parse_args,
    parse_args_with_explicit,
    task_obs_size,
)

_TOSS_OBS_NONE = 26
"""`TossCatch-v0` observation width under `partner_mode="none"`."""
_TOSS_OBS_ORACLE = 36
"""`TossCatch-v0` observation width under `partner_mode="oracle"`."""
_TOSS_OBS_OCCLUDED_NONE = 20
"""`OccludedToss-v0` width under `partner_mode="none"` (ball + own paddle only)."""
_TOSS_OBS_OCCLUDED_ORACLE = 30
"""`OccludedToss-v0` width under `partner_mode="oracle"` (occluded shared + partner block)."""


def test_occlusion_is_a_toss_only_observation_width_change() -> None:
    """The mask shrinks the toss shared block; the trainer's size helper mirrors it."""
    assert coop_obs_size("toss", "none") == _TOSS_OBS_NONE
    assert coop_obs_size("toss", "oracle") == _TOSS_OBS_ORACLE
    assert coop_obs_size("toss", "none", occlusion="partner") == _TOSS_OBS_OCCLUDED_NONE
    assert coop_obs_size("toss", "oracle", occlusion="partner") == _TOSS_OBS_OCCLUDED_ORACLE
    assert task_obs_size("toss", "none", "partner") == _TOSS_OBS_OCCLUDED_NONE
    assert task_obs_size("toss", "oracle", "partner") == _TOSS_OBS_OCCLUDED_ORACLE
    for task in ("cartpole", "pendulum"):
        assert coop_obs_size(task, "none", occlusion="partner") == coop_obs_size(task, "none")
    with pytest.raises(ValueError, match="occlusion"):
        task_obs_size("cube", "none", "partner")


def test_wrench_config_carries_the_occlusion_into_the_env_and_policy_width() -> None:
    """`env_obs_size`/`coop_env_config` follow the flag, including the Bi-JEPA input width."""
    for partner_mode, width in (
        ("none", _TOSS_OBS_OCCLUDED_NONE),
        ("oracle", _TOSS_OBS_OCCLUDED_ORACLE),
    ):
        cfg = WrenchPPOConfig(task="toss", partner_mode=partner_mode, occlusion="partner")
        assert cfg.env_obs_size == cfg.obs_size == width
        assert cfg.coop_env_config().occlusion == "partner"
        assert cfg.policy_input_size == width
    predicted = WrenchPPOConfig(task="toss", occlusion="partner", use_bijepa=True)
    assert predicted.input_mode == "predicted"
    assert predicted.env_obs_size == _TOSS_OBS_OCCLUDED_NONE
    assert predicted.policy_input_size == _TOSS_OBS_OCCLUDED_NONE + predicted.latent_dim
    # The default is the unchanged full layout.
    assert WrenchPPOConfig(task="toss").occlusion == "none"
    assert WrenchPPOConfig(task="toss").env_obs_size == _TOSS_OBS_NONE
    assert WrenchPPOConfig(task="toss", partner_mode="oracle").env_obs_size == _TOSS_OBS_ORACLE
    assert WrenchPPOConfig(task="toss").coop_env_config().occlusion == "none"


@pytest.mark.parametrize("task", ("cube", "cartpole", "pendulum"))
def test_occlusion_is_rejected_outside_the_toss_task(task: str) -> None:
    """Only the toss-catch task has per-agent shared fields to mask."""
    with pytest.raises(ValueError, match="only defined for --task toss"):
        WrenchPPOConfig(task=task, occlusion="partner")
    assert WrenchPPOConfig(task=task).occlusion == "none"


def test_occlusion_mode_is_validated() -> None:
    with pytest.raises(ValueError, match="occlusion"):
        WrenchPPOConfig(task="toss", occlusion="predicted")
    # `predicted` is a policy-input mode, not an environment occlusion: it still needs the
    # special env partner mode rule, and cannot mix an oracle env block with Bi-JEPA.
    with pytest.raises(ValueError, match="partner_mode"):
        WrenchPPOConfig(task="toss", occlusion="partner", partner_mode="oracle", use_bijepa=True)


def test_cli_exposes_the_occlusion_flag_for_toss() -> None:
    """`--task toss --occlusion partner` is the documented CLI spelling of the occluded run."""
    cfg = parse_args(["--task", "toss", "--occlusion", "partner", "--backend", "cpu"])
    assert (cfg.task, cfg.occlusion) == ("toss", "partner")
    assert cfg.env_obs_size == _TOSS_OBS_OCCLUDED_NONE
    explicit_cfg, explicit = parse_args_with_explicit(
        ["--task", "toss", "--occlusion", "partner", "--backend", "cpu"]
    )
    assert explicit_cfg.occlusion == "partner" and "occlusion" in explicit
    assert parse_args(["--task", "toss", "--backend", "cpu"]).occlusion == "none"
    with pytest.raises(SystemExit):
        parse_args(["--task", "toss", "--occlusion", "partner-paddle"])


# --- checkpoint compatibility (imports the trainer module: needs torch) --------------------


def test_checkpoint_compat_uses_the_checkpoint_occlusion_layout() -> None:
    """A checkpoint is judged against its own occlusion layout, and a wrong width fails."""
    pytest.importorskip("torch")
    from callosum.experiments.cube_wrench.train import (
        check_checkpoint_compat,
        config_from_payload,
    )

    occluded = {
        "config": {"task": "toss", "occlusion": "partner"},
        "obs_size": _TOSS_OBS_OCCLUDED_NONE,
        "partner_mode": "none",
        "input_mode": "none",
        "policy_input_size": _TOSS_OBS_OCCLUDED_NONE,
        "action_size": 3,
    }
    assert check_checkpoint_compat(occluded, "none") == ("none", 0)
    # The full-layout checkpoint (26) is fine under its own config...
    legacy = {
        "config": {"task": "toss"},
        "obs_size": _TOSS_OBS_NONE,
        "partner_mode": "none",
        "input_mode": "none",
        "policy_input_size": _TOSS_OBS_NONE,
        "action_size": 3,
    }
    assert check_checkpoint_compat(legacy, "none") == ("none", 0)
    # ...but a full-layout width cannot be evaluated under an occluded config (mismatch), and
    # an occluded checkpoint's width cannot be evaluated under the full layout.
    with pytest.raises(ValueError, match="width"):
        check_checkpoint_compat(
            {**occluded, "obs_size": _TOSS_OBS_NONE, "policy_input_size": _TOSS_OBS_NONE},
            "none",
        )
    with pytest.raises(ValueError, match="width"):
        check_checkpoint_compat(
            {
                **legacy,
                "obs_size": _TOSS_OBS_OCCLUDED_NONE,
                "policy_input_size": _TOSS_OBS_OCCLUDED_NONE,
            },
            "none",
        )
    # The saved config round-trips the flag into the trainer config and its env width.
    cfg = config_from_payload({"config": {"task": "toss", "occlusion": "partner"}})
    assert cfg.occlusion == "partner"
    assert cfg.env_obs_size == _TOSS_OBS_OCCLUDED_NONE


def test_checkpoint_compat_predicted_occluded_width() -> None:
    """The Bi-JEPA `predicted` arm of the occluded task takes 20 env values + latent."""
    pytest.importorskip("torch")
    from callosum.experiments.cube_wrench.train import check_checkpoint_compat

    latent_dim = 32
    payload = {
        "config": {"task": "toss", "occlusion": "partner", "use_bijepa": True},
        "obs_size": _TOSS_OBS_OCCLUDED_NONE,
        "partner_mode": "none",
        "input_mode": "predicted",
        "policy_input_size": _TOSS_OBS_OCCLUDED_NONE + latent_dim,
        "action_size": 3,
        "latent_dim": latent_dim,
        "bijepa": {"config": {"latent_dim": latent_dim}},
    }
    assert check_checkpoint_compat(payload, "predicted") == ("none", latent_dim)
    # The same checkpoint cannot be judged as a plain env-observation policy.
    with pytest.raises(ValueError, match="input_mode"):
        check_checkpoint_compat(payload, "none")
    with pytest.raises(ValueError, match="width"):
        check_checkpoint_compat(
            {**payload, "policy_input_size": _TOSS_OBS_NONE + latent_dim}, "predicted"
        )
