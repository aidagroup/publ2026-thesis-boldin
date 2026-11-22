"""The non-finite-env guard of the IPPO trainer (pure torch, fake vector env)."""

import pytest

torch = pytest.importorskip("torch")  # CI installs only the dev extra; run with `make dev`

from callosum.training._nonfinite import (
    NonFiniteTracker,
    guard_step,
    nonfinite_fields,
    nonfinite_mask,
)

NAN = float("nan")
LAYOUT = [(("agent", "so101_pg-0", "qpos"), 2), (("extra", "face_angle"), 1)]


class FakeEnvs:
    """Records partial resets and returns a fixed fresh observation for every env."""

    def __init__(self, fresh: torch.Tensor) -> None:
        self.fresh = fresh
        self.reset_calls: list[list[int]] = []

    def reset(self, *, options):
        self.reset_calls.append(options["env_idx"].tolist())
        return self.fresh.clone(), {}


def test_nonfinite_mask_rows_and_vectors() -> None:
    obs = torch.tensor([[1.0, 2.0], [NAN, 0.0], [0.0, float("inf")]])
    assert nonfinite_mask(obs).tolist() == [False, True, True]
    assert nonfinite_mask(torch.tensor([1.0, NAN, -float("inf")])).tolist() == [False, True, True]


def test_nonfinite_fields_names_the_offending_fields() -> None:
    row = torch.tensor([0.0, NAN, 1.0])
    assert nonfinite_fields(LAYOUT, row) == ["agent/so101_pg-0/qpos"]


def test_all_finite_does_nothing() -> None:
    envs = FakeEnvs(torch.zeros(3, 3))
    assert guard_step(envs, torch.ones(3, 3), torch.ones(3), {}) is None
    assert envs.reset_calls == []


def test_nonfinite_obs_row_is_reset_and_its_reward_zeroed() -> None:
    fresh = torch.full((3, 3), 7.0)
    envs = FakeEnvs(fresh)
    obs = torch.tensor([[1.0, 1.0, 1.0], [NAN, 0.0, 0.0], [3.0, 3.0, 3.0]])
    reward = torch.tensor([1.0, NAN, 1.0])
    result = guard_step(envs, obs, reward, {}, layout=LAYOUT, describe=True)
    assert result is not None and result.count == 1
    assert envs.reset_calls == [[1]]
    assert result.poisoned.tolist() == [False, True, False]
    # Only the poisoned row is replaced; the others keep their own observation.
    assert result.obs.tolist() == [[1.0] * 3, [7.0] * 3, [3.0] * 3]
    assert result.reward.tolist() == [1.0, 0.0, 1.0]
    assert "env 1" in result.report and "agent/so101_pg-0/qpos" in result.report


def test_nonfinite_reward_alone_triggers_a_reset() -> None:
    envs = FakeEnvs(torch.zeros(2, 3))
    result = guard_step(envs, torch.ones(2, 3), torch.tensor([NAN, 1.0]), {})
    assert result is not None and envs.reset_calls == [[0]]
    assert result.reward.tolist() == [0.0, 1.0] and result.report == ""


def test_autoreset_env_with_nonfinite_final_obs_is_not_reset_again() -> None:
    envs = FakeEnvs(torch.zeros(2, 3))
    final_obs = torch.tensor([[NAN, 0.0, 0.0], [1.0, 1.0, 1.0]])
    infos = {"_final_info": torch.tensor([True, False]), "final_observation": final_obs}
    result = guard_step(envs, torch.ones(2, 3), torch.tensor([NAN, 1.0]), infos)
    assert result is not None
    assert envs.reset_calls == []  # the wrapper already reset env 0 in this step
    assert result.poisoned.tolist() == [True, False]
    assert result.obs.tolist() == [[1.0] * 3, [1.0] * 3]


def test_row_still_nonfinite_after_reset_is_zeroed_but_all_bad_raises() -> None:
    fresh = torch.tensor([[NAN, 0.0, 0.0], [5.0, 5.0, 5.0]])
    obs = torch.tensor([[NAN, 0.0, 0.0], [NAN, 0.0, 0.0]])
    result = guard_step(FakeEnvs(fresh), obs, torch.ones(2), {})
    assert result is not None and result.obs.tolist() == [[0.0] * 3, [5.0] * 3]
    with pytest.raises(FloatingPointError):
        guard_step(FakeEnvs(torch.full((2, 3), NAN)), obs, torch.ones(2), {})


def test_tracker_counts_and_prints_the_first_report_once(capsys) -> None:
    tracker = NonFiniteTracker()
    envs = FakeEnvs(torch.zeros(2, 3))
    obs = torch.tensor([[NAN, 0.0, 0.0], [1.0, 1.0, 1.0]])
    for _ in range(2):
        result = guard_step(envs, obs, torch.ones(2), {}, describe=not tracker.reported)
        tracker.record("train", result, "iteration 1, rollout step 2")
    assert tracker.totals == {"train": 2, "eval": 0}
    out = capsys.readouterr().out
    assert out.count("[nonfinite] first occurrence") == 1 and "iteration 1" in out
