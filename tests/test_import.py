"""Smoke test: the callosum package and its subpackages must import cleanly."""

import callosum
import callosum.common
import callosum.configs
import callosum.envs
import callosum.robots
import callosum.training


def test_import_callosum() -> None:
    assert callosum.__version__


def test_import_subpackages() -> None:
    assert callosum.envs
    assert callosum.robots
    assert callosum.common
    assert callosum.training
    assert callosum.configs
