"""Smoke test: the callosum package and its subpackages must import cleanly."""

import callosum
import callosum.common
import callosum.experiments
import callosum.robots


def test_import_callosum() -> None:
    assert callosum.__version__


def test_import_subpackages() -> None:
    assert callosum.experiments
    assert callosum.robots
    assert callosum.common
