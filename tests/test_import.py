"""Smoke test: the callosum package and its subpackages must import cleanly."""

import callosum
import callosum.agents
import callosum.configs
import callosum.envs
import callosum.training


def test_import_callosum() -> None:
    assert callosum.__version__


def test_import_subpackages() -> None:
    assert callosum.envs
    assert callosum.agents
    assert callosum.training
    assert callosum.configs
