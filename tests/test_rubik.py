"""The cube's move tables have to be a real group, not a plausible-looking one."""

import numpy as np
import pytest

from callosum.envs import _rubik


@pytest.mark.parametrize("move", _rubik.MOVES)
def test_four_quarter_turns_is_identity(move) -> None:
    state = _rubik.RubikState()
    for _ in range(4):
        state.apply(move)
    assert state.solved


@pytest.mark.parametrize("move", _rubik.MOVES)
def test_a_move_and_its_inverse_cancel(move) -> None:
    inv = move[0] if move.endswith("'") else move + "'"
    assert _rubik.RubikState().apply(move).apply(inv).solved


@pytest.mark.parametrize("move", _rubik.MOVES)
def test_one_move_disturbs_exactly_twelve_facelets(move) -> None:
    """A quarter turn moves 8 facelets on its own face and 12 on the sides."""
    state = _rubik.RubikState().apply(move)
    assert int((state.colours != _rubik.SOLVED).sum()) == 12


def test_the_sexy_move_has_order_six() -> None:
    """(R U R' U') x6 is a solved cube. Fails on almost any wrong cycle table."""
    state = _rubik.RubikState()
    for _ in range(6):
        state.apply_all(["R", "U", "R'", "U'"])
    assert state.solved


def test_a_scramble_undone_is_solved() -> None:
    rng = np.random.default_rng(0)
    state, moves = _rubik.scramble(25, rng)
    assert not state.solved
    assert state.apply_all(_rubik.inverse(moves)).solved


def test_solved_facelets_counts_up_to_all_of_them() -> None:
    assert _rubik.RubikState().solved_facelets == _rubik.N_FACELETS
    state, _ = _rubik.scramble(20, np.random.default_rng(1))
    assert state.solved_facelets < _rubik.N_FACELETS


def test_centres_never_move() -> None:
    """Face centres are fixed by construction; a table that moves one is wrong."""
    state, _ = _rubik.scramble(50, np.random.default_rng(2))
    np.testing.assert_array_equal(state.colours[4::9], np.arange(6))


def test_facing_face_reads_the_cube_orientation() -> None:
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    assert _rubik.facing_face(identity, [0, 1, 0]) == "F"
    assert _rubik.facing_face(identity, [0, 0, 1]) == "U"
    assert _rubik.facing_face(identity, [-1, 0, 0]) == "L"
    # 90 deg about z maps the R face onto +y.
    quat = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    assert _rubik.facing_face(quat, [0, 1, 0]) == "R"


def test_one_hot_is_a_valid_observation() -> None:
    state, _ = _rubik.scramble(10, np.random.default_rng(3))
    obs = state.one_hot()
    assert obs.shape == (_rubik.N_FACELETS, 6)
    np.testing.assert_allclose(obs.sum(axis=1), 1.0)
