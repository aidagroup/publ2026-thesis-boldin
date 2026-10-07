"""Demo files of the scripted expert: Monte-Carlo returns, episode cutting, file round trip.

`discounted_returns` and the layout helpers are numpy/stdlib only and run in CI; everything that
builds or reads tensors is skipped without torch.
"""

import math

import numpy as np
import pytest

from callosum.training._demos import (
    check_layout_matches,
    discounted_returns,
    layout_from_meta,
    layout_to_meta,
)

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


def test_returns_discount_within_one_episode() -> None:
    out = discounted_returns(np.array([1.0, 1.0, 1.0]), np.zeros(3, dtype=int), gamma=0.5)
    assert np.allclose(out, [1.75, 1.5, 1.0])


def test_returns_do_not_leak_across_episodes() -> None:
    rewards = np.array([1.0, 2.0, 10.0, 20.0])
    out = discounted_returns(rewards, np.array([0, 0, 1, 1]), gamma=0.9)
    assert np.allclose(out, [1 + 0.9 * 2, 2, 10 + 0.9 * 20, 20])


def test_success_bonus_is_discounted_back_and_value_after_it_is_zero() -> None:
    # 5 steps of 0.5, the last one carries the +100 bonus (success terminal).
    rewards = np.array([0.5, 0.5, 0.5, 0.5, 100.5])
    out = discounted_returns(rewards, np.zeros(5, dtype=int), gamma=0.99)
    assert math.isclose(out[-1], 100.5)  # nothing after the terminal step
    assert math.isclose(out[0], sum(0.5 * 0.99**k for k in range(4)) + 100.5 * 0.99**4)


def test_returns_reward_scale_and_validation() -> None:
    out = discounted_returns(np.array([1.0, 1.0]), np.zeros(2, dtype=int), 1.0, reward_scale=3.0)
    assert np.allclose(out, [6.0, 3.0])
    with pytest.raises(ValueError, match="gamma"):
        discounted_returns(np.ones(2), np.zeros(2, dtype=int), gamma=1.5)
    with pytest.raises(ValueError, match="1-D"):
        discounted_returns(np.ones(3), np.zeros(2, dtype=int), gamma=0.9)


def test_layout_meta_round_trip_and_match_check() -> None:
    meta = {"obs_layout": layout_to_meta(LAYOUT)}
    assert meta["obs_layout"][0] == ["agent/so101_pg-0/qpos", 7]
    assert layout_from_meta(meta) == LAYOUT
    check_layout_matches(meta, LAYOUT)
    swapped = [LAYOUT[1], LAYOUT[0], *LAYOUT[2:]]
    with pytest.raises(ValueError, match="differs"):
        check_layout_matches(meta, swapped)
    with pytest.raises(ValueError, match="differs"):
        check_layout_matches(meta, [*LAYOUT[:-1], (("extra", "face_pose"), 6)])


torch = pytest.importorskip("torch")  # the rest needs tensors

from callosum.training._demos import (
    assemble_demos,
    load_demos,
    save_demos,
    split_rollout,
    validate_demos,
)


def make_rollout(steps: int = 6, envs: int = 3):
    """Time-major synthetic rollout: env 0 succeeds at t=2, env 1 never, env 2 at t=4."""
    gen = torch.Generator().manual_seed(0)
    obs = torch.randn(steps, envs, OBS_DIM, generator=gen)
    actions = torch.rand(steps, envs, 2, 6, generator=gen) * 2 - 1
    rewards = torch.ones(steps, envs)
    success = torch.zeros(steps, envs, dtype=torch.bool)
    success[2, 0] = True
    success[3:, 0] = True  # the flag may stay on afterwards; the episode is cut at the first one
    success[4, 2] = True
    terminated = success.clone()
    truncated = torch.zeros(steps, envs, dtype=torch.bool)
    truncated[-1] = True
    return obs, actions, rewards, terminated, truncated, success


def test_split_rollout_cuts_at_first_success_and_drops_failures() -> None:
    episodes = split_rollout(*make_rollout(), keep_all=False, batch_seed=7)
    assert [e["env_index"] for e in episodes] == [0, 2]
    assert [len(e["rewards"]) for e in episodes] == [3, 5]
    assert [e["first_success_step"] for e in episodes] == [2, 4]
    assert all(e["batch_seed"] == 7 for e in episodes)
    assert episodes[0]["terminated"][-1] and not episodes[0]["terminated"][:-1].any()


def test_split_rollout_keep_all_keeps_the_full_failed_episode() -> None:
    episodes = split_rollout(*make_rollout(), keep_all=True, batch_seed=0)
    assert [len(e["rewards"]) for e in episodes] == [3, 6, 5]
    assert episodes[1]["first_success_step"] == -1 and not episodes[1]["success"].any()


def test_assemble_layout_and_file_round_trip(tmp_path) -> None:
    rollout = make_rollout()
    episodes = split_rollout(*rollout, keep_all=True, batch_seed=3)
    meta = {"env_id": "FaceTurn-v0", "obs_layout": layout_to_meta(LAYOUT), "git_commit": None}
    demos = assemble_demos(episodes, meta)
    assert demos["obs"].shape == (14, OBS_DIM) and demos["actions"].shape == (14, 2, 6)
    assert demos["episode_id"].tolist() == [0] * 3 + [1] * 6 + [2] * 5
    assert demos["step"].tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4]
    assert demos["episodes"]["success"].tolist() == [True, False, True]
    assert demos["episodes"]["length"].tolist() == [3, 6, 5]
    # The first transition of episode 2 is env 2's observation at t=0, action as applied.
    assert torch.equal(demos["obs"][9], rollout[0][0, 2])
    assert torch.equal(demos["actions"][9], rollout[1][0, 2])
    path = tmp_path / "demos" / "d.pt"
    save_demos(path, demos)
    loaded = load_demos(path)  # weights_only=True: tensors and plain containers only
    assert torch.equal(loaded["obs"], demos["obs"])
    assert loaded["meta"]["num_episodes"] == 3 and loaded["meta"]["format"] == 1


def test_assemble_rejects_empty_and_validate_catches_bad_shapes() -> None:
    with pytest.raises(ValueError, match="no episodes"):
        assemble_demos([], {"obs_layout": layout_to_meta(LAYOUT)})
    episodes = split_rollout(*make_rollout(), keep_all=False, batch_seed=0)
    demos = assemble_demos(episodes, {"obs_layout": layout_to_meta(LAYOUT)})
    bad = dict(demos, rewards=demos["rewards"][:-1])
    with pytest.raises(ValueError, match="rewards"):
        validate_demos(bad)
    bad_meta = dict(demos, meta=dict(demos["meta"], obs_layout=[["extra/x", 3]]))
    with pytest.raises(ValueError, match="obs_layout"):
        validate_demos(bad_meta)
