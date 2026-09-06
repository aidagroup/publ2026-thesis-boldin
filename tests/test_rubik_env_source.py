"""Source-level checks on callosum.envs.rubik, which CANNOT be imported here.

It needs mani_skill (Linux+CUDA only, see tests/test_ippo_source.py for the
same constraint on the trainer). These parse the source instead of importing
it, so a class of mistake that would otherwise only surface on the server
(env not registered, a per-env python loop where a batched gather was wanted,
evaluate() missing a key the trainer/tests expect, the face joint never reset
after a completed turn) is caught here for free.
"""

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "callosum" / "envs" / "rubik.py"
REQUIRED_EVALUATE_KEYS = {
    "success",
    "face_angle",
    "is_body_stable",
    "is_lifted",
    "solved_facelets",
    "moves_applied",
}


def _tree() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name)


def _find_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_the_env_registers_as_rubikcube_v0() -> None:
    tree = _tree()
    cls = _find_class(tree, "RubikCube")
    decorator = next(d for d in cls.decorator_list if isinstance(d, ast.Call))
    assert isinstance(decorator.func, ast.Name) and decorator.func.id == "register_env"
    assert isinstance(decorator.args[0], ast.Constant) and decorator.args[0].value == "RubikCube-v0"
    kwargs = {
        kw.arg: kw.value.value for kw in decorator.keywords if isinstance(kw.value, ast.Constant)
    }
    assert kwargs.get("max_episode_steps") == 300


def test_the_move_table_tensor_is_built_once() -> None:
    """A (12, 54) gather table, assigned exactly once -- not rebuilt per env."""
    tree = _tree()
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Attribute) and t.attr == "_move_tables"
    ]
    assert len(assignments) == 1, f"expected one _move_tables assignment, got {len(assignments)}"
    (node,) = assignments
    assert isinstance(node.value, ast.Call)
    assert isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "tensor"


def test_move_application_is_a_batched_gather_not_a_python_loop_over_envs() -> None:
    """The colour permutation itself must be one torch.gather call, not a
    per-env loop -- the per-env work (facing_face) is only over the envs
    that finished a turn this step, never over the full env batch."""
    src = SOURCE.read_text(encoding="utf-8")
    assert "torch.gather(" in src
    tree = _tree()
    cls = _find_class(tree, "RubikCube")
    apply_fn = _find_method(cls, "_apply_completed_turns")
    gather_calls = [
        node
        for node in ast.walk(apply_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "gather"
    ]
    assert gather_calls, "_apply_completed_turns must permute colours via torch.gather"
    # The only `for` loop in here is over the turned-envs batch (env_idx),
    # never `range(self.num_envs)`.
    for_loops = [node for node in ast.walk(apply_fn) if isinstance(node, ast.For)]
    for loop in for_loops:
        assert "num_envs" not in ast.dump(loop.iter), (
            "a per-turn loop must not range over the full env batch"
        )


def test_evaluate_returns_the_keys_the_trainer_and_tests_expect() -> None:
    tree = _tree()
    cls = _find_class(tree, "RubikCube")
    evaluate_fn = _find_method(cls, "evaluate")
    returns = [node for node in ast.walk(evaluate_fn) if isinstance(node, ast.Return)]
    assert returns, "evaluate() must return something"
    dict_returns = [r for r in returns if isinstance(r.value, ast.Dict)]
    assert dict_returns, "evaluate() must return a dict literal"
    (ret,) = dict_returns
    keys = {k.value for k in ret.value.keys if isinstance(k, ast.Constant)}
    missing = REQUIRED_EVALUATE_KEYS - keys
    assert not missing, f"evaluate() is missing required keys: {missing}"


def test_the_face_joint_is_reset_after_a_completed_turn() -> None:
    """Both qpos and qvel must be snapped back to 0 for envs whose turn just
    completed, or the next turn starts from wherever the last one ended."""
    tree = _tree()
    cls = _find_class(tree, "RubikCube")
    apply_fn = _find_method(cls, "_apply_completed_turns")
    reset_targets = set()
    for node in ast.walk(apply_fn):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Attribute)
            and node.targets[0].value.attr in ("qpos", "qvel")
        ):
            reset_targets.add(node.targets[0].value.attr)
    assert reset_targets == {"qpos", "qvel"}, f"expected qpos and qvel reset, got {reset_targets}"


def test_evaluate_detects_a_completed_turn_before_applying_it() -> None:
    """Guards against silently dropping the "both arms gripping" gate: a
    quarter turn must only count once holder AND rotator are grasping."""
    src = SOURCE.read_text(encoding="utf-8")
    assert "is_grasping(self.body_link)" in src
    assert "is_grasping(self.face_link)" in src
    assert "_apply_completed_turns" in src


def test_cube_colours_obs_is_a_shared_field_not_per_agent() -> None:
    src = SOURCE.read_text(encoding="utf-8")
    assert '"cube_colours"' in src
    assert "agent_a_cube_colours" not in src and "agent_b_cube_colours" not in src
