"""Unit tests for the IPPO CLI config (step 2.1) -- pure Python/argparse,
no mani_skill/torch dependency, so runnable on macOS/CI.
"""

from callosum.configs.ippo import IPPOConfig, parse_args


def test_no_args_matches_dataclass_defaults() -> None:
    args = parse_args([])
    defaults = IPPOConfig()
    assert args.env_id == defaults.env_id
    assert args.num_envs == defaults.num_envs
    assert args.total_timesteps == defaults.total_timesteps


def test_overrides_env_id_and_total_timesteps() -> None:
    args = parse_args(["--env-id", "TwoSO100-v0", "--total-timesteps", "1000"])
    assert args.env_id == "TwoSO100-v0"
    assert args.total_timesteps == 1000
    assert isinstance(args.total_timesteps, int)


def test_bool_flag_accepts_true_and_false() -> None:
    assert parse_args(["--anneal-lr", "true"]).anneal_lr is True
    assert parse_args(["--anneal-lr", "false"]).anneal_lr is False


def test_optional_field_defaults_to_none() -> None:
    args = parse_args([])
    assert args.control_mode is None
    assert args.exp_name is None


def test_optional_field_can_be_overridden() -> None:
    args = parse_args(["--control-mode", "pd_joint_pos"])
    assert args.control_mode == "pd_joint_pos"


def test_optional_float_field_can_be_overridden() -> None:
    args = parse_args(["--target-kl", "0.05"])
    assert args.target_kl == 0.05


def test_computed_fields_start_at_zero() -> None:
    args = parse_args([])
    assert args.batch_size == 0
    assert args.minibatch_size == 0
    assert args.num_iterations == 0


def test_include_partner_defaults_to_false_and_is_settable() -> None:
    assert parse_args([]).include_partner is False
    assert parse_args(["--include-partner", "true"]).include_partner is True
    assert parse_args(["--include-partner", "false"]).include_partner is False
