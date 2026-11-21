"""Unit tests for assembly plans and the cube-orientation model (no kociemba needed)."""

import pytest

from callosum.experiments.arm.skills_plans import (
    FACES,
    AssemblyPlan,
    CubeOrientation,
    Flip,
    NextStep,
    PlanMove,
    next_step,
    parse_face_turn_sequence,
    random_plan,
    solve_with_kociemba,
)


def flip_about(axis: str, quarter_turns: int, times: int, start: CubeOrientation | None = None):
    """Apply `Flip(axis, quarter_turns)` `times` times (helper for composition tests)."""
    orientation = CubeOrientation.upright() if start is None else start
    flip = Flip(axis, quarter_turns)
    for _ in range(times):
        orientation = orientation.flipped(flip)
    return orientation


# --- random plans -----------------------------------------------------------------------


def test_random_plan_is_deterministic_per_seed_and_differs_across_seeds() -> None:
    assert random_plan(4, seed=42) == random_plan(4, seed=42)
    assert random_plan(8, seed=0) != random_plan(8, seed=1)  # 12^8 outcomes, no collision worry


def test_random_plan_content_is_valid() -> None:
    plan = random_plan(5, seed=7)
    assert len(plan) == 5
    assert all(move.face in FACES for move in plan)
    assert all(move.quarter_turns in (1, 2) for move in plan)
    assert plan[0] == plan.moves[0]  # indexing delegates to the tuple


def test_random_plan_rejects_negative_length() -> None:
    with pytest.raises(ValueError, match="length"):
        random_plan(-1, seed=0)


def test_plan_move_and_plan_validation() -> None:
    with pytest.raises(ValueError, match="face"):
        PlanMove("X", 1)
    with pytest.raises(ValueError, match="quarter_turns"):
        PlanMove("U", 3)
    with pytest.raises(TypeError, match="PlanMove"):
        AssemblyPlan([("U", 1)])  # type: ignore[list-item]
    assert len(AssemblyPlan(())) == 0


# --- orientation ------------------------------------------------------------------------


def test_upright_orientation_maps_each_face_to_its_own_side() -> None:
    orientation = CubeOrientation.upright()
    assert orientation.up == "U"
    assert orientation.down == "D"
    assert orientation.face_along("F") == "F"
    assert orientation.face_along("R") == "R"


def test_flip_changes_orientation_top_move_does_not() -> None:
    flipped = CubeOrientation.upright().flipped(Flip("R", 1))
    assert flipped != CubeOrientation.upright()
    assert flipped.up == "F"  # +90 about +x: front comes up, up goes to back
    unchanged = CubeOrientation.upright().after_move(PlanMove("U", 2))
    assert unchanged == CubeOrientation.upright()


def test_flips_stay_permutations_of_all_faces() -> None:
    for axis in ("R", "L", "F", "B"):
        for quarter_turns in (1, 2):
            slots = CubeOrientation.upright().flipped(Flip(axis, quarter_turns)).slots
            assert sorted(slots) == sorted(FACES)


def test_four_quarter_flips_restore_identity_and_two_90s_equal_one_180() -> None:
    assert flip_about("R", 1, times=4) == CubeOrientation.upright()
    assert flip_about("F", 1, times=4) == CubeOrientation.upright()
    assert flip_about("R", 1, times=2) == flip_about("R", 2, times=1)
    assert flip_about("R", 1, times=2) == flip_about("L", 2, times=1)  # sign-free at 180


def test_flip_validation() -> None:
    with pytest.raises(ValueError, match="axis"):
        Flip("U", 1)  # only horizontal axes are flips
    with pytest.raises(ValueError, match="quarter_turns"):
        Flip("R", 3)


@pytest.mark.parametrize("face", ["D", "F", "B", "L", "R"])
def test_non_top_faces_need_exactly_one_flip(face: str) -> None:
    flips, turned = CubeOrientation.upright().flips_to_bring_up(face)
    assert len(flips) == 1
    assert turned.up == face
    assert turned == CubeOrientation.upright().flipped(flips[0])


def test_bottom_face_needs_a_single_180_flip() -> None:
    flips, turned = CubeOrientation.upright().flips_to_bring_up("D")
    assert flips == (Flip("R", 2),)
    assert turned.up == "D"


def test_top_face_needs_no_flip_and_passes_orientation_through() -> None:
    orientation = CubeOrientation.upright()
    flips, turned = orientation.flips_to_bring_up("U")
    assert flips == ()
    assert turned is orientation


def test_flips_to_bring_up_from_a_turned_orientation() -> None:
    # Front up after one flip; the (former) top face is at the back and needs one flip.
    front_up = CubeOrientation.upright().flipped(Flip("R", 1))
    assert front_up.up == "F"
    flips, turned = front_up.flips_to_bring_up("U")
    assert len(flips) == 1 and turned.up == "U"
    with pytest.raises(ValueError, match="face"):
        front_up.flips_to_bring_up("X")


# --- next-step logic --------------------------------------------------------------------


def test_next_step_returns_the_move_and_the_executable_orientation() -> None:
    plan = random_plan(6, seed=3)
    orientation = CubeOrientation.upright()
    # Walk the plan: after each next_step's flips the move's face is on top, and a
    # top-face move does not change the orientation -- sequences converge.
    for index in range(len(plan)):
        step = next_step(plan, index, orientation)
        assert isinstance(step, NextStep)
        assert step.move == plan[index]
        assert step.orientation.up == step.move.face
        if step.flips:
            assert step.orientation == orientation.flipped(step.flips[0])
        else:
            assert step.orientation == orientation
        orientation = step.orientation.after_move(step.move)
        assert sorted(orientation.slots) == sorted(FACES)


def test_next_step_on_a_u_move_needs_no_flips() -> None:
    plan = AssemblyPlan((PlanMove("U", 1),))
    step = next_step(plan, 0, CubeOrientation.upright())
    assert step.flips == ()
    assert step.orientation == CubeOrientation.upright()


def test_next_step_out_of_range_raises() -> None:
    plan = random_plan(2, seed=0)
    with pytest.raises(IndexError):
        next_step(plan, 5, CubeOrientation.upright())


# --- notation parsing (also the kociemba output contract) -------------------------------


def test_parse_round_trips_plan_strings() -> None:
    for seed in (0, 1, 2):
        plan = random_plan(7, seed=seed)
        assert parse_face_turn_sequence(str(plan)) == plan  # plans never contain primes


def test_parse_expands_modifiers() -> None:
    plan = parse_face_turn_sequence("R U2 F'")
    assert plan.moves == (
        PlanMove("R", 1),
        PlanMove("U", 2),
        PlanMove("F", 1),
        PlanMove("F", 2),  # prime: -90 deg == 90 deg + 180 deg
    )
    assert str(plan) == "R U2 F F2"


def test_parse_rejects_bad_tokens() -> None:
    for bad in ("X2", "U3", "u", 'F"', "2U"):
        with pytest.raises(ValueError, match="token"):
            parse_face_turn_sequence(bad)
    assert len(parse_face_turn_sequence("")) == 0


def test_kociemba_missing_package_raises_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force the lazy import to fail even where kociemba happens to be installed:
    # a None entry in sys.modules makes `import kociemba` raise ImportError.
    monkeypatch.setitem(__import__("sys").modules, "kociemba", None)
    with pytest.raises(RuntimeError, match="uv pip install kociemba"):
        solve_with_kociemba("U" * 54)


def test_orientation_math_is_exact_integers() -> None:
    # Sanity for the Rodrigues-at-90-deg integer rotation: rotating every world
    # slot direction four times about any axis returns each direction to itself.
    for axis in ("R", "L", "F", "B"):
        orientation = CubeOrientation.upright()
        for _ in range(4):
            orientation = orientation.flipped(Flip(axis, 1))
        assert orientation == CubeOrientation.upright()
