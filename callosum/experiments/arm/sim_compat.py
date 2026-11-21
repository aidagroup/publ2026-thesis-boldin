"""Simulator workarounds shared by the smoke/probe scripts and the trainer.

Imports of `mani_skill` / `sapien` happen inside the functions, never at module level, so this
module can be imported anywhere (macOS dev venv, CI) without the simulator installed.
"""

import sys

CPU_SIM_BACKENDS = ("cpu", "physx_cpu")
"""`sim_backend` values that select ManiSkill's CPU simulation (a single env per scene)."""


def is_cpu_backend(sim_backend: str) -> bool:
    """True if `sim_backend` selects ManiSkill's CPU simulation (limited to one env)."""
    return sim_backend in CPU_SIM_BACKENDS


def stub_rendering_for_macos() -> None:
    """Let the CPU sim run on a Mac without Vulkan.

    ManiSkill 3.0.1 assumes rendering is possible on macOS (`can_render` always returns True
    there), but SAPIEN has no Vulkan device on a Mac, so building any render material raises.
    `render_backend="none"` does not help here: ManiSkill 3.0.1 forces the CPU render backend on
    Darwin (`parse_sim_and_render_backend`) and `can_render` returns True there. The smoke scripts
    and the trainer never render, so turn rendering off and replace the render-material
    classes with inert stubs. Only call this on macOS, before creating an env.
    """
    import mani_skill.render.utils as render_utils
    import sapien
    from sapien.pysapien import render as pysapien_render
    from sapien.wrapper import urdf_loader

    class _InertRenderObject:
        def __init__(self, *args, **kwargs):
            self.base_color = (1.0, 1.0, 1.0, 1.0)

        def __getitem__(self, idx):
            return (1.0, 1.0, 1.0, 1.0)[idx]

    render_utils.can_render = lambda device: False
    for name in ("RenderMaterial", "RenderTexture2D", "RenderTexture"):
        setattr(sapien.render, name, _InertRenderObject)
        setattr(pysapien_render, name, _InertRenderObject)
    urdf_loader.RenderMaterial = _InertRenderObject
    urdf_loader.RenderTexture2D = _InertRenderObject


def prepare_sim_backend(sim_backend: str) -> None:
    """Apply the platform workarounds needed before `gym.make` with `sim_backend`.

    Currently only the macOS render stub for the CPU sim; a no-op everywhere else.
    """
    if is_cpu_backend(sim_backend) and sys.platform == "darwin":
        stub_rendering_for_macos()
