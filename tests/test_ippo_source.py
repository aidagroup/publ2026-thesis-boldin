"""Static checks on callosum.training.ippo, which CANNOT be imported here.

It needs mani_skill (Linux+CUDA only), so nothing in the normal test suite ever
executes it and a whole class of mistake reaches the server untouched. These
tests parse the source instead, and cost nothing.

Motivated by a real failure on 2026-08-30: the metric logger was introduced
under the name `metrics`, which the function already used for the dict returned
by ppo_update. A mechanical rewrite of `writer.add_scalar(...)` into
`metrics.log(...)` therefore turned one call into `dict.log(...)`, and the run
died 40 seconds in with `AttributeError: 'dict' object has no attribute 'log'`
-- after ruff and 68 unit tests had all passed.
"""

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "callosum" / "training" / "ippo.py"
LOGGER_METHODS = {"log", "iteration_line", "headline", "summary"}


def _tree() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def _assigned_names(tree: ast.Module) -> list[str]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)
    return names


def test_the_metric_logger_name_is_bound_exactly_once() -> None:
    """A second binding means something else now answers to that name."""
    tree = _tree()
    logger_names = {
        t.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "MetricLogger"
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    assert len(logger_names) == 1, f"expected one MetricLogger variable, got {logger_names}"
    name = logger_names.pop()
    assert _assigned_names(tree).count(name) == 1, (
        f"{name!r} is assigned more than once -- a later binding would shadow the logger"
    )


def test_logger_methods_are_only_called_on_the_logger() -> None:
    """`X.log(...)` must be the logger, not a dict that happens to be in scope."""
    tree = _tree()
    logger_name = next(
        t.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "MetricLogger"
        for t in node.targets
        if isinstance(t, ast.Name)
    )
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in LOGGER_METHODS
            and isinstance(node.func.value, ast.Name)
        ):
            assert node.func.value.id == logger_name, (
                f"line {node.lineno}: {node.func.value.id}.{node.func.attr}() "
                f"— expected the logger {logger_name!r}"
            )


def test_the_env_registration_imports_survive() -> None:
    """`ruff check --fix` once deleted these; without them no env registers."""
    src = SOURCE.read_text(encoding="utf-8")
    assert "from callosum.envs import face_turn as" in src
    assert "from callosum.envs import two_so100_base as" in src
