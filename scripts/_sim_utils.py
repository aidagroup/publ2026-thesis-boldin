"""Shared helpers for the smoke scripts: CLI flags and env construction.

Not part of the `callosum` package because it touches `mani_skill` / `sapien`
(Linux-only dependencies). The smoke scripts import it as a sibling module, which works
when they are run as `python scripts/<name>.py`.
"""

import argparse
import sys

import gymnasium as gym


def parse_args(description: str) -> argparse.Namespace:
    """Parse the `--sim-backend` / `--num-envs` flags shared by the smoke scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--sim-backend",
        default="gpu",
        help='ManiSkill sim backend: "gpu" (server, default) or "cpu" (local, 1 env only).',
    )
    parser.add_argument("--num-envs", type=int, default=16, help="Parallel envs (default 16).")
    return parser.parse_args()


def _stub_rendering_for_macos() -> None:
    """Let the CPU sim run on a Mac without Vulkan.

    ManiSkill 3.0.1 assumes rendering is possible on macOS (`can_render` always returns True
    there), but SAPIEN has no Vulkan device on a Mac, so building any render material raises.
    These smoke scripts never render, so turn rendering off and replace the render-material
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


def make_env(env_id: str, args: argparse.Namespace):
    """`gym.make` the env with the CLI flags applied (CPU sim is limited to one env)."""
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
        render_mode=None,
    )
