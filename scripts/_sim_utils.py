"""Shared helpers for the smoke and probe scripts: CLI flags and env construction.

Not part of the `callosum` package because it touches `mani_skill` / `sapien`
(Linux-only dependencies). The smoke scripts import it as a sibling module, which works
when they are run as `python scripts/<name>.py`.
"""

import argparse

import gymnasium as gym

from callosum.envs._sim_compat import is_cpu_backend, prepare_sim_backend


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


def make_env(env_id: str, args: argparse.Namespace, **env_kwargs):
    """`gym.make` the env with the CLI flags applied (CPU sim is limited to one env).

    State observations need no renderer, so `--render-backend` defaults to "none": ManiSkill
    then skips `sapien.render.RenderSystem` (and lighting/sensor setup), which fails on a
    machine without a hardware Vulkan device. Any trainer must pass `render_backend="none"` too.

    Extra keyword arguments (e.g. `control_mode="pd_joint_pos"`) are passed on to the env; they
    may also override the defaults `obs_mode="state"` and `render_mode=None` (the video script
    asks for `obs_mode="rgb"` and `render_mode="rgb_array"`).
    """
    num_envs = args.num_envs
    if is_cpu_backend(args.sim_backend) and num_envs != 1:
        print(f"note: the CPU sim backend supports a single env; using num_envs=1 ({num_envs})")
        num_envs = 1
    prepare_sim_backend(args.sim_backend)
    env_kwargs = {"obs_mode": "state", "render_mode": None} | env_kwargs
    return gym.make(
        env_id,
        num_envs=num_envs,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        **env_kwargs,
    )
