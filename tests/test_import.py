"""Smoke test: the callosum package and its subpackages must import cleanly.

CI runs with the `dev` extra only (ruff + pytest) -- NO torch, NO
benchmarl, NO mani_skill. So the four imports below must succeed on CI;
anything that needs the `train` extra (Bi-JEPA, BenchMARL, _ppo_core) is
pushed behind `pytest.importorskip` so it is skipped -- not failed --
when the dependency is absent. See docs/implementation-plan.md §0.
"""

import pytest

import callosum
import callosum.agents
import callosum.configs
import callosum.envs
import callosum.training


def test_import_callosum() -> None:
    assert callosum.__version__


def test_import_subpackages() -> None:
    # The CI contract (dev extra only) set:
    for sub in (callosum.envs, callosum.agents, callosum.training, callosum.configs):
        assert sub


def test_benchmarl_task_importable() -> None:
    # Needs the `train` extra (benchmarl/torchrl/torch); skip, don't fail, on CI.
    pytest.importorskip("torch")
    pytest.importorskip("benchmarl")
    import callosum.envs.benchmarl_task

    assert callosum.envs.benchmarl_task


def test_benchmarl_task_registered() -> None:
    # Importing the module side-effect-registers CallosumTask into BenchMARL's
    # task registry (train extra only) without requiring mani_skill/SAPIEN.
    pytest.importorskip("torch")
    pytest.importorskip("benchmarl")
    from benchmarl.environments import task_config_registry

    assert "callosum/face_turn" in task_config_registry
    assert "callosum/two_so100" in task_config_registry
