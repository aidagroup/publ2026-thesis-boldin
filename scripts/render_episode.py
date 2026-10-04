"""Render one FaceTurn-v0 episode to an mp4: a scene view and both wrist cameras, in sync.

Each video frame is one sim (control) step. The left half is the fixed third-person scene camera
(`render_camera`, see `callosum.configs.cameras.SceneCameraConfig`, it sees both arms and the
cube); the right half stacks the holder's (`agent_a`) wrist camera over the rotator's
(`agent_b`). All panels come from the same sim step: the wrist images are part of the
observation `env.step` returns, and the scene image is rendered right after it, before the next
step. Every panel is labelled, and a footer shows the step index and the sim time
(`step / control_freq`); the video plays in real time by default (`--fps` = control freq).

Motion source: `--policy scripted` (default) is the scripted two-arm expert from
`_face_turn_expert.py` (the same one `probe_face_turn.py` uses; one full face turn is ~350 steps);
`--policy random` samples uniform random actions for a quick look. `--checkpoint <path>` plays
back a trained IPPO policy (`latest.pt` / `best.pt` of a `callosum.training.ippo` run, both
agents): the actors are rebuilt from the checkpoint's config and act on the env's state
observation (cut per agent exactly as in training, see `callosum.training._playback`), with the
actor mean by default (`--stochastic` samples). The episode ends on success unless
`--full-episode`, or at the episode length of the training config (400 for FaceTurn-v0). The
env uses the checkpoint's control mode and `partner_obs` but the `so101_pg_wristcam` robots and
`obs_mode="state+rgb"`; the footer shows the face angle and the success flag of every step.

Rendering needs a Vulkan device. On the lab server only the lavapipe *software* ICD exists, so the
env is created with `render_backend="cpu"` and rendering is slow (expect seconds per frame at the
default sizes); the physics runs on the CPU too by default (`--sim-backend cpu`, one env).
On macOS there is no Vulkan at all: use `--dry-run`, which steps the policy with rendering
stubbed out (`obs_mode="state"`, no sensors), feeds blank panels through the same compositing code
and reports the frame layout (no file is written). With `--checkpoint` it also prints the episode
summary (success, face angle, return), which makes it the local test of a checkpoint.

Usage (on the server, from the repo root):

    uv run python scripts/render_episode.py --name face_turn_scripted
    uv run python scripts/render_episode.py --checkpoint runs/faceturn_s1/best.pt \\
        --name faceturn_s1_policy

The file appears at `runs/videos/<name>.mp4`.
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import torch
from _face_turn_expert import ArmModel, Rig, run_face_turn
from _sim_utils import make_env, parse_args
from mani_skill.utils import gym_utils
from PIL import Image, ImageDraw, ImageFont

import callosum.envs.face_turn  # noqa: F401  (registers FaceTurn-v0)
from callosum.configs.cameras import SceneCameraConfig, WristCameraConfig
from callosum.training._playback import CheckpointPolicy, EpisodeSummary, state_of

ROBOT_UIDS = ("so101_pg_wristcam", "so101_pg_wristcam")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "runs" / "videos"
PANEL_TITLES = {"scene": "scene", "holder": "holder wrist", "rotator": "rotator wrist"}
FOOTER_HEIGHT = 28
DEFAULT_RANDOM_STEPS = 100  # the env's max_episode_steps


class _StopEpisode(Exception):
    """Raised from the per-step hook to end the episode early (`--max-steps`)."""


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)  # scalable font, Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font) -> None:
    """Draw `text` in white on a dark box at `xy` (top-left)."""
    left, top, right, bottom = draw.textbbox(xy, text, font=font)
    draw.rectangle((left - 4, top - 2, right + 4, bottom + 2), fill=(20, 20, 20))
    draw.text(xy, text, fill=(255, 255, 255), font=font)


def to_image(panel, height: int, width: int) -> np.ndarray:
    """A sensor/render output (a `(1, H, W, 3)` tensor or `(H, W, 3)` array) as uint8 `(H, W, 3)`.

    Resized to `height x width` if it does not already have that size.
    """
    if isinstance(panel, torch.Tensor):
        panel = panel.detach().cpu().numpy()
    panel = np.asarray(panel)
    if panel.ndim == 4:
        panel = panel[0]
    panel = panel[..., :3]
    if panel.dtype != np.uint8:
        panel = np.clip(panel * 255, 0, 255).astype(np.uint8)
    if panel.shape[:2] != (height, width):
        panel = np.asarray(Image.fromarray(panel).resize((width, height), Image.BILINEAR))
    return panel


def compose_frame(
    scene: np.ndarray,
    holder: np.ndarray,
    rotator: np.ndarray,
    step: int,
    sim_time: float,
    footer: str = "",
) -> np.ndarray:
    """One video frame: scene panel on the left, holder over rotator wrist panel on the right.

    `scene` is `(2H, Ws, 3)`, the wrist panels `(H, W, 3)` (all uint8). The result has a footer
    bar with the step index and the sim time; its sides are even (required by yuv420p).
    """
    panel_h = holder.shape[0]
    assert holder.shape == rotator.shape and scene.shape[0] == 2 * panel_h
    body = np.concatenate([scene, np.concatenate([holder, rotator], axis=0)], axis=1)
    height, width = body.shape[0] + FOOTER_HEIGHT, body.shape[1]
    canvas = np.zeros((height + height % 2, width + width % 2, 3), np.uint8)
    canvas[: body.shape[0], :width] = body
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font = _font(max(12, panel_h // 15))
    scene_w = scene.shape[1]
    _label(draw, (6, 6), PANEL_TITLES["scene"], font)
    _label(draw, (scene_w + 6, 6), PANEL_TITLES["holder"], font)
    _label(draw, (scene_w + 6, panel_h + 6), PANEL_TITLES["rotator"], font)
    text = f"step {step:4d}   t = {sim_time:6.2f} s"
    if footer:
        text += f"   |   {footer}"
    draw.text((8, body.shape[0] + 5), text, fill=(255, 255, 255), font=_font(16))
    # Thin separators between the panels.
    draw.line((scene_w, 0, scene_w, body.shape[0]), fill=(255, 255, 255), width=1)
    draw.line((scene_w, panel_h, width, panel_h), fill=(255, 255, 255), width=1)
    return np.asarray(image)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(sim_backend="cpu", render_backend="cpu", num_envs=1)
    parser.add_argument("--name", default=None, help="Output name (default: face_turn_<policy>).")
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Where to write <name>.mp4."
    )
    parser.add_argument("--policy", choices=["scripted", "random"], default="scripted")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "IPPO checkpoint (latest.pt / best.pt of a trainer run) to roll out; replaces "
            "--policy. Needs the train extra (torch)."
        ),
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="With --checkpoint: sample the actions instead of using the actor mean.",
    )
    parser.add_argument(
        "--full-episode",
        action="store_true",
        help="With --checkpoint: keep going after success until the episode length is reached.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Reset (and random policy) seed.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Stop after this many sim steps (default: the whole scripted episode, "
            f"{DEFAULT_RANDOM_STEPS} steps for --policy random)."
        ),
    )
    parser.add_argument(
        "--fps", type=float, default=None, help="Video fps (default: control freq)."
    )
    parser.add_argument("--width", type=int, default=320, help="Wrist panel width in pixels.")
    parser.add_argument("--height", type=int, default=240, help="Wrist panel height in pixels.")
    parser.add_argument(
        "--scene-shader",
        choices=["minimal", "default"],
        default=None,
        help='Shader pack of the scene camera ("default" has lighting, slower; unverified).',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Step the policy without rendering (works on macOS), report the frame layout only.",
    )


def make_render_env(args: argparse.Namespace, control_mode: str | None, **env_kwargs):
    """The FaceTurn env with wrist cameras, wrist images in the obs and a scene render camera.

    `control_mode=None` keeps the env's default. With `--checkpoint` the obs mode is
    `"state+rgb"` (the privileged state vector the policy needs, next to the images; see
    `callosum.training._playback`); `env_kwargs` are further env arguments (the checkpoint's).
    """
    scene_cam = dataclasses.replace(
        SceneCameraConfig(), width=args.width * 2, height=args.height * 2
    )
    if args.dry_run:
        # No renderer: no sensors are created, so a visual obs_mode would have nothing to read.
        args.render_backend = "none"
        mode = {"obs_mode": "state", "render_mode": None}
    else:
        # TODO(review): "state+rgb" (obs["state"] next to the sensor data) is verified on the Mac
        # only for the equivalent "state" mode (identical flattening); not rendered anywhere yet.
        obs_mode = "state+rgb" if args.checkpoint is not None else "rgb"
        mode = {"obs_mode": obs_mode, "render_mode": "rgb_array"}
    extra = dict(env_kwargs)
    if control_mode is not None:
        extra["control_mode"] = control_mode
    if args.scene_shader is not None:
        extra["human_render_camera_configs"] = {"shader_pack": args.scene_shader}
    return make_env(
        "FaceTurn-v0",
        args,
        robot_uids=ROBOT_UIDS,
        scene_camera=scene_cam,
        # Applies to every sensor camera, i.e. both wrist cameras.
        sensor_configs={"width": args.width, "height": args.height},
        **mode,
        **extra,
    )


class FrameRecorder:
    """Collects one composite frame per sim step (and feeds an optional mp4 writer)."""

    def __init__(self, env, args: argparse.Namespace, writer, max_steps: int | None) -> None:
        self.env = env
        self.base = env.unwrapped
        self.args = args
        self.writer = writer
        self.max_steps = max_steps
        self.dt = 1.0 / self.base.control_freq
        agent_keys = list(self.base.agent.agents_dict)
        cam_uid = WristCameraConfig().uid
        self.sensor_keys = {
            "holder": f"{agent_keys[0]}-{cam_uid}",
            "rotator": f"{agent_keys[1]}-{cam_uid}",
        }
        self.footer = f"seed {args.seed}   {args.policy}"
        self.status = ""  # per-step text appended to the footer (set by the caller)
        self.num_frames = 0
        self.frame_shape: tuple[int, ...] | None = None
        self.start = time.time()

    def _panels(self, obs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = self.args.height, self.args.width
        if self.args.dry_run:
            blank = np.zeros((h, w, 3), np.uint8)
            return np.zeros((2 * h, 2 * w, 3), np.uint8), blank, blank
        data = obs["sensor_data"]
        missing = [k for k in self.sensor_keys.values() if k not in data]
        if missing:
            raise KeyError(f"no sensor data for {missing}; got {list(data)}")
        holder = to_image(data[self.sensor_keys["holder"]]["rgb"], h, w)
        rotator = to_image(data[self.sensor_keys["rotator"]]["rgb"], h, w)
        scene = to_image(self.base.render(), 2 * h, 2 * w)
        return scene, holder, rotator

    def record(self, obs) -> None:
        """Compose the frame of the sim's current state (`obs` is that step's observation)."""
        step = self.num_frames
        scene, holder, rotator = self._panels(obs)
        footer = f"{self.footer}   {self.status}" if self.status else self.footer
        frame = compose_frame(scene, holder, rotator, step, step * self.dt, footer)
        self.frame_shape = frame.shape
        if self.writer is not None:
            self.writer.append_data(frame)
        self.num_frames += 1
        if self.num_frames % 10 == 0 or self.num_frames == 1:
            elapsed = time.time() - self.start
            print(
                f"  frame {self.num_frames} (t = {step * self.dt:.2f} s sim), "
                f"{self.num_frames / elapsed:.2f} frames/s",
                flush=True,
            )
        if self.max_steps is not None and step >= self.max_steps:
            raise _StopEpisode


def random_actions(base, rng: np.random.Generator) -> dict:
    """Uniform random normalised actions, one 6-D vector per arm (`pd_joint_delta_pos`)."""
    return {
        uid: torch.as_tensor(rng.uniform(-1, 1, (1, 6)), dtype=torch.float32, device=base.device)
        for uid in base.agent.agents_dict
    }


def bind_policy(policy: CheckpointPolicy, env, obs) -> None:
    """Attach `policy` to the (camera) env and verify that its inputs match the training ones.

    The structured state observation of the freshly reset env gives the field layout, which
    `CheckpointPolicy.bind` checks against the real flat state and the checkpoint's input widths.
    """
    base = env.unwrapped
    spaces = base.single_action_space.spaces
    env_uids = list(base.agent.agents_dict)
    policy.bind(
        base.get_obs(unflattened=True),
        state_of(obs),
        env_uids,
        [(spaces[uid].low, spaces[uid].high) for uid in env_uids],
    )
    for name, uid, dim, fields in zip(
        ("agent_a", "agent_b"), env_uids, policy.obs_dims, policy.builder.fields, strict=True
    ):
        print(f"  {name} ({uid}): policy input {dim} = {', '.join(fields)}")


def play_policy(
    env,
    policy: CheckpointPolicy,
    obs,
    recorder: FrameRecorder,
    summary: EpisodeSummary,
    full_episode: bool,
) -> dict:
    """Roll the policy out, recording one frame per step; returns the last step's info.

    Ends at the env's time limit, or on the first success unless `full_episode`.
    """
    info: dict = {}
    while True:
        obs, reward, _, truncated, info = env.step(policy.act(state_of(obs)))
        summary.update(float(reward.reshape(-1)[0]), info)
        recorder.status = summary.status()
        recorder.record(obs)
        if bool(torch.as_tensor(truncated).any()):
            return info
        if summary.success and not full_episode:
            return info


def main() -> None:
    args = parse_args(__doc__, add_args)
    if args.sim_backend in ("gpu", "physx_cuda") and args.checkpoint is not None:
        sys.exit("error: --checkpoint playback uses the single-env CPU sim (--sim-backend cpu).")
    if sys.platform == "darwin" and not args.dry_run:
        sys.exit(
            "error: rendering is not possible on macOS (SAPIEN has no Vulkan device there). "
            "Run this on the lab server, or pass --dry-run to only step the policy and report "
            "the frame layout."
        )
    if args.sim_backend not in ("cpu", "physx_cpu") and args.render_backend in (
        "cpu",
        "sapien_cpu",
    ):
        # ManiSkill 3.0.1 reads GPU-sim camera images through CUDA buffers of the render system
        # (`camera_group.get_picture_cuda`), which a CPU (lavapipe) renderer cannot provide.
        sys.exit(
            "error: --sim-backend gpu needs a CUDA render device (--render-backend gpu), which "
            "does not exist on the lab server (lavapipe only). Use --sim-backend cpu (default)."
        )

    policy = None
    env_kwargs = {}
    if args.checkpoint is not None:
        # The checkpoint decides how the env is built: control mode, partner_obs, reward mode and
        # episode length are those of the training run.
        policy = CheckpointPolicy(args.checkpoint, deterministic=not args.stochastic)
        cfg = policy.cfg
        args.policy = f"checkpoint:{policy.run_name}"
        control_mode = cfg.control_mode
        env_kwargs = {
            "partner_obs": cfg.partner_obs,
            "reward_mode": cfg.reward_mode,
            "reconfiguration_freq": 0,  # as in training
        }
        if cfg.max_episode_steps is not None:
            env_kwargs["max_episode_steps"] = cfg.max_episode_steps
        default_name = f"{policy.run_name}_policy"
    else:
        control_mode = "pd_joint_pos" if args.policy == "scripted" else "pd_joint_delta_pos"
        default_name = f"face_turn_{args.policy}"

    name = args.name or default_name
    out_path = args.output_dir / f"{name}.mp4"
    max_steps = args.max_steps
    if max_steps is None and args.policy == "random":
        max_steps = DEFAULT_RANDOM_STEPS

    env = make_render_env(args, control_mode, **env_kwargs)
    base = env.unwrapped
    fps = args.fps or base.control_freq

    writer = None
    if not args.dry_run:
        import imageio.v2 as imageio

        args.output_dir.mkdir(parents=True, exist_ok=True)
        # macro_block_size=1: the frame sides are already even, do not let imageio resize.
        writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1)
    recorder = FrameRecorder(env, args, writer, max_steps)

    obs, reset_info = env.reset(seed=args.seed)
    print(
        f"policy={args.policy} control_mode={control_mode} sim_backend={args.sim_backend} "
        f"render_backend={args.render_backend} wrist {args.width}x{args.height} "
        f"scene {2 * args.width}x{2 * args.height}, control freq {base.control_freq} Hz"
    )
    success = None
    summary = EpisodeSummary()
    try:
        if policy is not None:
            episode_steps = int(gym_utils.find_max_episode_steps_value(env))
            bind_policy(policy, env, obs)
            summary.observe(reset_info)
            recorder.status = summary.status()
            print(
                f"{policy.cfg.env_id}: episode length {episode_steps} steps, "
                f"{'sampled' if args.stochastic else 'deterministic (actor mean)'} actions, "
                f"{'full episode' if args.full_episode else 'stop on success'}"
            )
        recorder.record(obs)  # step 0: the freshly reset state
        if policy is not None:
            info = play_policy(env, policy, obs, recorder, summary, args.full_episode)
        elif args.policy == "scripted":
            rig = Rig(env, ArmModel(), on_step=recorder.record)
            run_face_turn(rig, lambda phase: print(f"  [step {recorder.num_frames - 1}] {phase}"))
            info = rig.last_info
        else:
            rng = np.random.default_rng(args.seed)
            info = {}
            while True:
                obs, _, _, _, info = env.step(random_actions(base, rng))
                recorder.record(obs)
        success = bool(info["success"].all()) if "success" in info else None
    except _StopEpisode:
        print(f"stopped after {recorder.num_frames - 1} steps (--max-steps)")
        if policy is not None:
            success = summary.success
    finally:
        if writer is not None:
            writer.close()
        env.close()

    height, width = recorder.frame_shape[:2]
    duration = recorder.num_frames / fps
    print(
        f"{recorder.num_frames} frames of {width}x{height} "
        f"(scene {2 * args.width}x{2 * args.height} | holder / rotator wrist "
        f"{args.width}x{args.height} each, footer {FOOTER_HEIGHT} px), "
        f"{duration:.1f} s at {fps:g} fps, success={success}"
    )
    if policy is not None:
        print(f"episode: {summary.text()}")
    if args.dry_run:
        print("dry run: nothing rendered, nothing written")
    else:
        print(f"wrote {out_path} in {time.time() - recorder.start:.0f} s")


if __name__ == "__main__":
    main()
