"""Shared helpers for the smoke and probe scripts: CLI flags and env construction.

Not part of the `callosum` package because it touches `mani_skill` / `sapien`
(Linux-only dependencies). The smoke scripts import it as a sibling module, which works
when they are run as `python scripts/<name>.py`.
"""

import argparse
import sys

import gymnasium as gym


def parse_args(description: str, extra_args=None) -> argparse.Namespace:
    """Parse the `--sim-backend` / `--render-backend` / `--num-envs` flags of the smoke scripts.

    `extra_args`, if given, is called with the parser to add script-specific flags.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--sim-backend",
        default="gpu",
        help='ManiSkill sim backend: "gpu" (server, default) or "cpu" (local, 1 env only).',
    )
    parser.add_argument(
        "--render-backend",
        default="none",
        help=(
            'ManiSkill render backend (default "none": no renderer, enough for obs_mode="state"). '
            'Use e.g. "gpu" only when rendering/vision is needed and a hardware Vulkan device '
            "exists; with the lavapipe software ICD the default CUDA render device cannot be created."
        ),
    )
    parser.add_argument("--num-envs", type=int, default=16, help="Parallel envs (default 16).")
    if extra_args is not None:
        extra_args(parser)
    return parser.parse_args()


def _stub_rendering_for_macos() -> None:
    """Let the CPU sim run on a Mac without Vulkan.

    ManiSkill 3.0.1 assumes rendering is possible on macOS (`can_render` always returns True
    there), but SAPIEN has no Vulkan device on a Mac, so building any render material raises.
    `render_backend="none"` does not help here: ManiSkill 3.0.1 forces the CPU render backend on
    Darwin (`parse_sim_and_render_backend`) and `can_render` returns True there. These smoke
    scripts never render, so turn rendering off and replace the render-material
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


def make_env(env_id: str, args: argparse.Namespace, **env_kwargs):
    """`gym.make` the env with the CLI flags applied (CPU sim is limited to one env).

    State observations need no renderer, so `--render-backend` defaults to "none": ManiSkill
    then skips `sapien.render.RenderSystem` (and lighting/sensor setup), which fails on a
    machine without a hardware Vulkan device. Any trainer must pass `render_backend="none"` too.

    Extra keyword arguments (e.g. `control_mode="pd_joint_pos"`) are passed on to the env.
    """
    num_envs = args.num_envs
    if args.sim_backend in ("cpu", "physx_cpu"):
        if num_envs != 1:
            print(f"note: the CPU sim backend supports a single env; using num_envs=1 ({num_envs})")
            num_envs = 1
        if sys.platform == "darwin":
            _stub_rendering_for_macos()
    return gym.make(
        env_id,
        num_envs=num_envs,
        obs_mode="state",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        render_mode=None,
        **env_kwargs,
    )
