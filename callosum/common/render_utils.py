"""Render and encode helpers for the CubeWrench-v0 demo videos (`render_wrench_demo.py`).

The reduced env (`callosum.experiments.cube_wrench`) renders nothing on its own, so the demo
builds a **render-only copy** of the same MJCF scene through `mujoco.MjSpec`: the copied
model gets colours, markers, a skybox and lights, and the physics model of the env is never
touched. The state of the env is copied into the copy's `qpos`/`qvel` before every frame,
so the visuals are exactly the env's physics.

Why the markers: the bare layer is a square plate of exactly the core's footprint, so a
90-degree turn of a symmetric square would be invisible, and the core is a symmetric cube,
so its yaw drift would be invisible too. The render-only copy therefore overhangs the layer
plate by 4 mm and adds a red tab on the layer plus two coloured face markers on the core;
these geoms exist in the render copy only (no physics uses them).

Encoding avoids new dependencies: MP4 is written by `cv2.VideoWriter` when OpenCV is
installed, otherwise by an `ffmpeg` subprocess (H.264/yuv420p; present on the RunPod pod and
on the macOS dev laptop); GIF by Pillow's `save(..., save_all=True)`. Frames are decorated
with Pillow (`ImageDraw`), not OpenCV, so both encoder paths work without extra fonts.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# --- look of the render-only scene copy --------------------------------------------------
CORE_RGBA = (0.20, 0.30, 0.47, 1.0)
LAYER_RGBA = (0.96, 0.56, 0.15, 1.0)
LAYER_TAB_RGBA = (0.86, 0.11, 0.11, 1.0)
CORE_MARKER_A_RGBA = (0.90, 0.94, 0.98, 1.0)
CORE_MARKER_B_RGBA = (0.15, 0.82, 0.78, 1.0)
SKY_RGB1 = (0.035, 0.045, 0.075)
SKY_RGB2 = (0.17, 0.23, 0.36)

CAM_LOOKAT = (0.0, 0.0, 0.018)
"""Camera target: the centre of the floating cube."""
CAM_DISTANCE = 0.165
"""Camera distance (m) for a ~5.7 cm cube in a 45-degree frame."""
CAM_AZIMUTH = 142.0
"""Camera azimuth (deg): a three-quarter view that shows two core faces and the layer."""
CAM_ELEVATION = -26.0
"""Camera elevation (deg); negative looks down on the cube from above."""

SCENE_XML = Path("callosum") / "envs" / "wrench_cube" / "assets" / "cube_wrench.xml"
"""The env's MJCF, relative to the repository root."""

TITLE_COLOR = (255, 255, 255, 255)
SUBTITLE_COLOR = (198, 210, 228, 255)
HUD_COLOR = (232, 238, 248, 255)
PANEL_RGBA = (10, 14, 22, 165)
GOAL_PANEL_RGBA = (8, 12, 20, 205)
"""Panel of the top goal band (the GOAL / SUCCESS lines shown on every frame)."""
CARD_SCRIM = 0.30
"""Brightness factor of the dimmed scene frame behind an intro/outro card."""
CARD_PANEL_RGBA = (10, 15, 24, 232)
"""Panel of an intro/outro card."""
ACCENT_RGBA = (242, 140, 40, 235)
SUCCESS_COLOR = (110, 230, 140, 255)
FAILURE_COLOR = (255, 105, 105, 255)
MUTED_COLOR = (150, 162, 180, 255)
HOLDER_COLOR = (96, 214, 138, 255)
"""Green: the holder's "hold the core" action (HUD block, arrows and legend)."""
TURN_COLOR = (242, 140, 40, 255)
"""Orange: the rotator's "turn the layer" action (HUD block, arrow and legend)."""
ARROW_OUTLINE_COLOR = (10, 12, 18, 205)
"""Dark outline under the arrows, so they stay readable on the yellow layer / blue core."""

_FONT_CANDIDATES = {
    False: (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ),
    True: (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ),
}
"""TrueType candidates, Linux (pod) first, macOS second; Pillow's built-in font is the
fallback and is scalable in Pillow >= 10."""

_FONT_CACHE: dict[tuple[int, bool], Any] = {}


def mujoco_module() -> Any:
    """Import and return `mujoco` lazily (so `--help` works without an OpenGL setup)."""
    import mujoco

    return mujoco


def repo_root() -> Path:
    """The repository root (the parent of `scripts/`)."""
    return Path(__file__).resolve().parent.parent


def load_font(size: int, *, bold: bool = False) -> Any:
    """Load a TrueType font of `size` px, falling back to Pillow's built-in scalable font."""
    key = (int(size), bool(bold))
    if key not in _FONT_CACHE:
        font: Any = None
        for candidate in _FONT_CANDIDATES[bold]:
            path = Path(candidate)
            if not path.exists():
                continue
            try:
                font = ImageFont.truetype(str(path), size=key[0])
                break
            except OSError:
                continue
        if font is None:
            try:
                font = ImageFont.load_default(size=key[0])
            except TypeError:  # Pillow < 10.1 has no scalable default font
                font = ImageFont.load_default()
        _FONT_CACHE[key] = font
    return _FONT_CACHE[key]


def build_render_model(width: int, height: int) -> tuple[Any, Any, Any]:
    """Build the render-only model copy, its `MjData` and the demo `MjvCamera`.

    Args:
        width: Offscreen framebuffer width (px) the copy must support.
        height: Offscreen framebuffer height (px) the copy must support.

    Returns:
        `(model, data, camera)`: a compiled copy of the env's MJCF with the demo colours,
        markers, skybox and lights, an `MjData` for it and a configured free camera. The
        copy's joint layout is identical to the env's, so the env state can be copied into
        `data.qpos`/`data.qvel` verbatim.
    """
    mujoco = mujoco_module()
    spec = mujoco.MjSpec.from_file(str(repo_root() / SCENE_XML))

    core = spec.body("core")
    layer = spec.body("layer")
    spec.geom("core_geom").rgba = list(CORE_RGBA)
    layer_geom = spec.geom("layer_geom")
    layer_geom.rgba = list(LAYER_RGBA)
    layer_geom.size = [0.0325, 0.0325, 0.002]  # 4 mm overhang: the turn becomes visible
    layer.add_geom(
        name="layer_tab",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.008, 0.004, 0.0018],
        pos=[0.0245, 0.0, 0.0038],
        rgba=list(LAYER_TAB_RGBA),
        contype=0,
        conaffinity=0,
    )
    # Flush painted stripes on the two core faces the camera sees: they make the core's
    # yaw drift of the no-partner scene visible (the cube alone is symmetric).
    core.add_geom(
        name="core_marker_a",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.0040, 0.0006, 0.0100],
        pos=[0.0285, 0.0, 0.0],
        rgba=list(CORE_MARKER_A_RGBA),
        contype=0,
        conaffinity=0,
    )
    core.add_geom(
        name="core_marker_b",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.0040, 0.0006, 0.0100],
        pos=[0.0, -0.0285, 0.0],
        rgba=list(CORE_MARKER_B_RGBA),
        contype=0,
        conaffinity=0,
    )

    spec.add_texture(
        name="demo_sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=list(SKY_RGB1),
        rgb2=list(SKY_RGB2),
        width=256,
        height=512,
    )
    world = spec.worldbody
    world.add_light(
        name="key",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[0.25, -0.20, 0.45],
        dir=[-0.45, 0.35, -0.82],
        diffuse=[0.95, 0.93, 0.90],
        specular=[0.45, 0.45, 0.45],
        castshadow=True,
    )
    world.add_light(
        name="fill",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[-0.30, 0.30, 0.20],
        dir=[0.60, -0.50, -0.40],
        diffuse=[0.35, 0.42, 0.58],
        castshadow=False,
    )
    world.add_light(
        name="rim",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[0.0, 0.40, 0.10],
        dir=[0.0, -1.0, -0.25],
        diffuse=[0.30, 0.26, 0.22],
        castshadow=False,
    )
    world.add_light(
        name="front_fill",  # from the camera side: keeps the visible faces readable
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[0.0, 0.0, 0.2],
        dir=[-0.55, 0.70, -0.45],
        diffuse=[0.34, 0.37, 0.44],
        castshadow=False,
    )
    world.add_light(
        name="ambient",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[0.0, 0.0, 1.0],
        dir=[0.0, 0.0, -1.0],
        diffuse=[0.15, 0.15, 0.15],
        castshadow=False,
    )

    # MuJoCo's default offscreen framebuffer is 640x480; enlarge it for any requested size.
    spec.visual.global_.offwidth = max(int(width), int(spec.visual.global_.offwidth))
    spec.visual.global_.offheight = max(int(height), int(spec.visual.global_.offheight))
    model = spec.compile()
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = CAM_LOOKAT
    camera.distance = CAM_DISTANCE
    camera.azimuth = CAM_AZIMUTH
    camera.elevation = CAM_ELEVATION
    return model, data, camera


class DemoRenderer:
    """Offscreen renderer of the render-only scene copy, driven by the env's state."""

    def __init__(self, width: int, height: int) -> None:
        """Create the render copy and the offscreen `mujoco.Renderer` at `width` x `height`."""
        mujoco = mujoco_module()
        self._mujoco = mujoco
        self._model, self._data, self._camera = build_render_model(width, height)
        self._renderer = mujoco.Renderer(self._model, height, width)
        self.width = int(width)
        self.height = int(height)

    def render_state(self, state: dict[str, Any]) -> np.ndarray:
        """Render one RGB frame (uint8, `height` x `width` x 3) of the given env state.

        Args:
            state: The env's `state()` dict (`core_pos`, `core_quat`, `core_lin_vel`,
                `core_ang_vel`, `layer_angle`, `layer_ang_vel`).
        """
        qpos = np.concatenate(
            [
                np.asarray(state["core_pos"], dtype=np.float64),
                np.asarray(state["core_quat"], dtype=np.float64),
                [float(state["layer_angle"])],
            ]
        )
        qvel = np.concatenate(
            [
                np.asarray(state["core_lin_vel"], dtype=np.float64),
                np.asarray(state["core_ang_vel"], dtype=np.float64),
                [float(state["layer_ang_vel"])],
            ]
        )
        if qpos.shape != self._data.qpos.shape or qvel.shape != self._data.qvel.shape:
            raise ValueError(
                f"state does not fit the render model: qpos {qpos.shape} vs "
                f"{self._data.qpos.shape}, qvel {qvel.shape} vs {self._data.qvel.shape}"
            )
        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel
        self._mujoco.mj_forward(self._model, self._data)
        self._renderer.update_scene(self._data, camera=self._camera)
        return np.asarray(self._renderer.render(), dtype=np.uint8)

    def close(self) -> None:
        """Release the offscreen OpenGL context."""
        self._renderer.close()


# --- frame decoration --------------------------------------------------------------------
def text_size(draw: ImageDraw.ImageDraw, text: str, font: Any) -> tuple[int, int]:
    """Pixel `(width, height)` of `text` in `font`."""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return int(right - left), int(bottom - top)


def fit_font(
    draw: ImageDraw.ImageDraw, text: str, max_width: int, size: int, *, bold: bool = False
) -> Any:
    """The largest font of at most `size` px that keeps `text` within `max_width` px."""
    size = max(8, int(size))
    while size > 8 and text_size(draw, text, load_font(size, bold=bold))[0] > max_width:
        size -= 1
    return load_font(size, bold=bold)


@dataclass(frozen=True)
class SceneAnchors:
    """Screen-space anchors of the demo scene, found from the rendered pixels.

    The projection is taken from the image itself instead of the camera matrices: the layer
    is the only yellow region, the tab on it is the only red region, and the cube is the
    bright region. This keeps the arrows glued to the cube under perspective and drift,
    without duplicating MuJoCo's free-camera conventions.
    """

    layer_center: tuple[float, float]
    layer_radius: tuple[float, float]
    tab_angle: float
    core_center: tuple[float, float]


def scene_anchors(frame: np.ndarray) -> SceneAnchors | None:
    """Locate the layer (centre, screen radii, red-tab angle) and the core centre in a frame."""
    image = np.asarray(frame, dtype=np.int16)
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    layer_mask = (red > 150) & (green > 110) & (blue < 120)
    if not layer_mask.any():
        return None
    rows, cols = np.nonzero(layer_mask)
    layer_cx, layer_cy = float(cols.mean()), float(rows.mean())
    radius_x = max(4.0, float(cols.max() - cols.min()) / 2.0)
    radius_y = max(3.0, float(rows.max() - rows.min()) / 2.0)
    tab_mask = (red > 140) & (green < 90) & (blue < 90)
    tab_angle = 0.0
    if tab_mask.any():
        tab_rows, tab_cols = np.nonzero(tab_mask)
        tab_angle = math.atan2(
            (float(tab_rows.mean()) - layer_cy) / radius_y,
            (float(tab_cols.mean()) - layer_cx) / radius_x,
        )
    bright = np.maximum(np.maximum(red, green), blue) > 95
    cube_rows, cube_cols = np.nonzero(bright)
    if not len(cube_rows):
        return None
    top, bottom = int(cube_rows.min()), int(cube_rows.max())
    core_center = (
        (float(cube_cols.min()) + float(cube_cols.max())) / 2.0,
        top + 0.62 * (bottom - top),
    )
    return SceneAnchors(
        layer_center=(layer_cx, layer_cy),
        layer_radius=(radius_x, radius_y),
        tab_angle=tab_angle,
        core_center=core_center,
    )


def _arrow_head(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    line_width: int,
    color: tuple[int, int, int, int],
) -> None:
    """Draw a V-shaped arrow head at `end`, pointing along `start -> end`."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return
    ux, uy = dx / norm, dy / norm
    size = max(7.0, 2.6 * line_width)
    px, py = -uy, ux
    left = (end[0] - ux * size + px * size * 0.62, end[1] - uy * size + py * size * 0.62)
    right = (end[0] - ux * size - px * size * 0.62, end[1] - uy * size - py * size * 0.62)
    draw.line([end, left], fill=ARROW_OUTLINE_COLOR, width=line_width + 2)
    draw.line([end, right], fill=ARROW_OUTLINE_COLOR, width=line_width + 2)
    draw.line([end, left], fill=color, width=line_width)
    draw.line([end, right], fill=color, width=line_width)


def draw_action_arrows(
    frame: np.ndarray,
    anchors: SceneAnchors,
    *,
    holder_force_ratio: float,
    rotator_torque_ratio: float,
) -> np.ndarray:
    """Draw the wrench arrows of both agents on top of a rendered frame.

    Green arrows start at the core centre and point outward/down (the holder's "keep the core
    still" action); their length is linear in |F| / the display reference. The orange arc runs
    around the top layer from the red tab and sweeps in the direction of the hinge torque; its
    sweep length and thickness are linear in |tau_z| / the rotator torque authority.

    Args:
        frame: RGB uint8 frame of the scene (without the HUD).
        anchors: Screen anchors of the layer/core (see `scene_anchors`).
        holder_force_ratio: `|F|` divided by the holder's display reference (clipped to [0,1]).
        rotator_torque_ratio: `tau_z` divided by the rotator torque authority (signed, [-1,1]).
    """
    height, width = frame.shape[:2]
    scale = height / 640.0
    image = Image.fromarray(np.ascontiguousarray(frame)).convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    holder_ratio = float(np.clip(holder_force_ratio, 0.0, 1.0))
    if holder_ratio > 0.02:
        length = (24.0 + 70.0 * holder_ratio) * scale
        line_width = max(3, round((4.0 + 5.0 * holder_ratio) * scale))
        start = anchors.core_center
        for angle_deg in (55.0, 90.0, 125.0):
            angle = math.radians(angle_deg)
            end = (start[0] + length * math.cos(angle), start[1] + length * math.sin(angle))
            draw.line([start, end], fill=ARROW_OUTLINE_COLOR, width=line_width + 3)
            draw.line([start, end], fill=HOLDER_COLOR, width=line_width)
            _arrow_head(draw, start, end, line_width, HOLDER_COLOR)

    turn_ratio = float(np.clip(rotator_torque_ratio, -1.0, 1.0))
    if abs(turn_ratio) > 0.02:
        centre_x, centre_y = anchors.layer_center
        radius_x = anchors.layer_radius[0] * 0.92
        radius_y = anchors.layer_radius[1] * 0.92
        # Positive hinge torque increases the layer angle, which moves the red tab towards
        # decreasing screen angle in this camera; negative torque sweeps the other way.
        sweep = (0.45 + 1.7 * abs(turn_ratio)) * (-1.0 if turn_ratio > 0 else 1.0)
        steps = 22
        points = [
            (
                centre_x + radius_x * math.cos(anchors.tab_angle + sweep * index / steps),
                centre_y + radius_y * math.sin(anchors.tab_angle + sweep * index / steps),
            )
            for index in range(steps + 1)
        ]
        line_width = max(3, round((3.5 + 6.0 * abs(turn_ratio)) * scale))
        draw.line(points, fill=ARROW_OUTLINE_COLOR, width=line_width + 3, joint="curve")
        draw.line(points, fill=TURN_COLOR, width=line_width, joint="curve")
        _arrow_head(draw, points[-2], points[-1], line_width, TURN_COLOR)

    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"), dtype=np.uint8)


def annotate_frame(
    frame: np.ndarray,
    *,
    title: str,
    subtitle: str,
    hud_lines: Sequence[str],
    agent_blocks: Sequence[tuple[str, Sequence[str], tuple[int, int, int, int]]] = (),
    legend: Sequence[tuple[tuple[int, int, int, int], str]] = (),
    angle_fraction: float | None = None,
    status: str | None = None,
    status_color: tuple[int, int, int, int] = MUTED_COLOR,
    status_detail: str | None = None,
    goal_lines: Sequence[str] = (),
) -> np.ndarray:
    """Draw the demo overlay (goal band, title, agent action blocks, HUD, legend, status, bar).

    Args:
        frame: RGB uint8 frame from `DemoRenderer` (the wrench arrows are already drawn).
        title: Scene caption, e.g. "hold + turn (both agents)".
        subtitle: One-line explanation under the title.
        hud_lines: Bottom-left live numbers (time, angle, drift), one line each.
        agent_blocks: Top-right action blocks, `(title, lines, colour)` each -- the holder's
            stabilise-wrench block and the rotator's turn-torque block.
        legend: Bottom legend items `(colour, label)` -- green "hold", orange "turn".
        angle_fraction: `layer_angle / goal_angle` in `[0, 1]` for the bottom progress bar.
        status: Optional big status word ("SUCCESS", "FAILED") in the bottom-right corner.
        status_color: Colour of the status word.
        status_detail: Optional small line under the status word (the failure reason).
        goal_lines: Optional top goal band, one line each (the coop demo's GOAL / SUCCESS
            lines); the first line is drawn in the title colour, the rest in the success
            colour. The band spans the full frame width, so the title panel below it is
            pushed down by its height.
    """
    height, width = frame.shape[:2]
    image = Image.fromarray(np.ascontiguousarray(frame)).convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    radius = max(3, height // 110)

    margin = max(8, height // 40)
    pad = max(6, height // 80)
    available = width - 2 * margin - 2 * pad
    title_font = fit_font(draw, title, available, max(14, round(height / 22)), bold=True)
    subtitle_font = fit_font(draw, subtitle, available, max(11, round(height / 36)))
    hud_font = load_font(max(10, round(height / 34)))
    status_font = load_font(max(13, round(height / 26)), bold=True)

    # Top goal band: the explicit goal / success condition, on every frame. Drawn first and
    # full-width, then everything else starts below it.
    content_top = margin
    if goal_lines:
        goal_font = fit_font(
            draw,
            max(goal_lines, key=len),
            available,
            max(11, round(height / 32)),
            bold=True,
        )
        goal_bar_w = max(4, height // 90)
        goal_line_h = max(text_size(draw, line, goal_font)[1] for line in goal_lines)
        goal_h = len(goal_lines) * goal_line_h + (len(goal_lines) + 1) * pad
        draw.rounded_rectangle(
            [margin, margin, width - margin, margin + goal_h],
            radius=radius,
            fill=GOAL_PANEL_RGBA,
        )
        text_x = margin + pad + goal_bar_w + pad // 2
        draw.rounded_rectangle(
            [
                margin + pad // 2,
                margin + pad,
                margin + pad // 2 + goal_bar_w,
                margin + goal_h - pad,
            ],
            radius=goal_bar_w // 2,
            fill=ACCENT_RGBA,
        )
        for index, line in enumerate(goal_lines):
            draw.text(
                (text_x, margin + pad + index * goal_line_h),
                line,
                font=goal_font,
                fill=TITLE_COLOR if index == 0 else SUCCESS_COLOR,
                stroke_width=2,
                stroke_fill=(0, 0, 0, 210),
            )
        content_top = margin + goal_h + pad

    # Top-left title panel: caption + one-line explanation.
    title_w, title_h = text_size(draw, title, title_font)
    subtitle_w, subtitle_h = text_size(draw, subtitle, subtitle_font)
    panel_w = max(title_w, subtitle_w) + 2 * pad
    panel_h = title_h + subtitle_h + 3 * pad
    draw.rounded_rectangle(
        [margin, content_top, margin + panel_w, content_top + panel_h],
        radius=radius,
        fill=PANEL_RGBA,
    )
    draw.text(
        (margin + pad, content_top + pad),
        title,
        font=title_font,
        fill=TITLE_COLOR,
        stroke_width=2,
        stroke_fill=(0, 0, 0, 210),
    )
    draw.text(
        (margin + pad, content_top + 2 * pad + title_h),
        subtitle,
        font=subtitle_font,
        fill=SUBTITLE_COLOR,
        stroke_width=1,
        stroke_fill=(0, 0, 0, 180),
    )

    # Top-right per-agent action blocks ("who does what"), stacked below the title panel.
    if agent_blocks:
        block_title_font = load_font(max(9, round(height / 38)), bold=True)
        block_line_font = load_font(max(9, round(height / 44)))
        bar_w = max(3, height // 90)
        block_top = content_top + panel_h + 2 * pad
        block_x1 = width - margin
        for block_title, block_lines, block_color in agent_blocks:
            block_title_w, block_title_h = text_size(draw, block_title, block_title_font)
            line_h = max(
                (text_size(draw, line, block_line_font)[1] for line in block_lines), default=0
            )
            line_w = max(
                (text_size(draw, line, block_line_font)[0] for line in block_lines), default=0
            )
            block_w = max(block_title_w, line_w) + 3 * pad + bar_w
            block_h = block_title_h + len(block_lines) * line_h + 3 * pad
            x0 = block_x1 - block_w
            draw.rounded_rectangle(
                [x0, block_top, block_x1, block_top + block_h], radius=radius, fill=PANEL_RGBA
            )
            draw.rounded_rectangle(
                [x0 + pad // 2, block_top + pad, x0 + pad // 2 + bar_w, block_top + block_h - pad],
                radius=bar_w // 2,
                fill=block_color,
            )
            text_x = x0 + pad + bar_w + pad // 2
            draw.text(
                (text_x, block_top + pad),
                block_title,
                font=block_title_font,
                fill=block_color,
                stroke_width=2,
                stroke_fill=(0, 0, 0, 200),
            )
            for index, line in enumerate(block_lines):
                draw.text(
                    (text_x, block_top + 2 * pad + block_title_h + index * line_h),
                    line,
                    font=block_line_font,
                    fill=HUD_COLOR,
                    stroke_width=1,
                    stroke_fill=(0, 0, 0, 170),
                )
            block_top += block_h + pad

    # Bottom: angle progress bar (0 -> goal 90 degrees).
    bar_height = max(5, height // 90)
    bar_y = height - margin - bar_height
    bar_x0 = margin
    bar_x1 = width - margin
    draw.rounded_rectangle(
        [bar_x0, bar_y, bar_x1, bar_y + bar_height],
        radius=bar_height // 2,
        fill=(255, 255, 255, 45),
    )
    if angle_fraction is not None:
        fraction = float(np.clip(angle_fraction, 0.0, 1.0))
        if fraction > 0.0:
            draw.rounded_rectangle(
                [bar_x0, bar_y, bar_x0 + fraction * (bar_x1 - bar_x0), bar_y + bar_height],
                radius=bar_height // 2,
                fill=ACCENT_RGBA,
            )

    # Bottom legend strip (green = hold, orange = turn), centred above the progress bar.
    legend_h = 0
    if legend:
        legend_font = load_font(max(9, round(height / 46)))
        text_h = max(text_size(draw, label, legend_font)[1] for _, label in legend)
        swatch = max(8, height // 48)
        item_widths = [
            swatch + pad // 2 + text_size(draw, label, legend_font)[0] for _, label in legend
        ]
        legend_w = sum(item_widths) + (len(legend) + 1) * pad
        legend_h = text_h + 2 * pad
        x0 = max(margin, (width - legend_w) // 2)
        y0 = bar_y - pad - legend_h
        draw.rounded_rectangle(
            [x0, y0, x0 + legend_w, y0 + legend_h], radius=radius, fill=PANEL_RGBA
        )
        cursor = x0 + pad
        for (color, label), item_w in zip(legend, item_widths):
            draw.rounded_rectangle(
                [
                    cursor,
                    y0 + (legend_h - swatch) // 2,
                    cursor + swatch,
                    y0 + (legend_h + swatch) // 2,
                ],
                radius=max(2, swatch // 4),
                fill=color,
            )
            draw.text(
                (cursor + swatch + pad // 2, y0 + (legend_h - text_h) // 2),
                label,
                font=legend_font,
                fill=HUD_COLOR,
            )
            cursor += item_w + pad

    bottom_limit = bar_y - pad - (legend_h + pad if legend_h else 0)

    # Bottom-left HUD panel with the live numbers.
    if hud_lines:
        line_h = max(text_size(draw, line, hud_font)[1] for line in hud_lines)
        hud_w = max(text_size(draw, line, hud_font)[0] for line in hud_lines)
        hud_h = len(hud_lines) * line_h + (len(hud_lines) + 1) * pad
        hud_y1 = bottom_limit
        hud_y0 = hud_y1 - hud_h
        draw.rounded_rectangle(
            [margin, hud_y0, margin + hud_w + 2 * pad, hud_y1],
            radius=radius,
            fill=PANEL_RGBA,
        )
        for index, line in enumerate(hud_lines):
            draw.text(
                (margin + pad, hud_y0 + pad + index * line_h),
                line,
                font=hud_font,
                fill=HUD_COLOR,
                stroke_width=1,
                stroke_fill=(0, 0, 0, 170),
            )

    # Bottom-right status word (only once the episode is over) plus the failure reason.
    if status:
        detail_font = load_font(max(9, round(height / 44)))
        status_w, status_h = text_size(draw, status, status_font)
        detail = status_detail or ""
        detail_w, detail_h = text_size(draw, detail, detail_font) if detail else (0, 0)
        block_w = max(status_w, detail_w) + 2 * pad
        block_h = status_h + (detail_h + pad // 2 if detail else 0) + 2 * pad
        x1 = width - margin
        y1 = bottom_limit
        x0 = x1 - block_w
        y0 = y1 - block_h
        draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=PANEL_RGBA)
        draw.text(
            (x0 + pad, y0 + pad),
            status,
            font=status_font,
            fill=status_color,
            stroke_width=2,
            stroke_fill=(0, 0, 0, 200),
        )
        if detail:
            draw.text(
                (x0 + pad, y0 + 2 * pad + status_h),
                detail,
                font=detail_font,
                fill=MUTED_COLOR,
            )

    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"), dtype=np.uint8)


# --- intro / outro cards -------------------------------------------------------------------
def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: Any, max_width: int) -> list[str]:
    """Greedy word wrap of `text` to at most `max_width` px in `font`.

    Empty text yields one empty line (a vertical spacer); a word wider than the limit stays
    whole, so the caller can still shrink the font when even one word does not fit.
    """
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if text_size(draw, candidate, font)[0] <= int(max_width):
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


@dataclass(frozen=True)
class CardLine:
    """One text line of an intro/outro card (`card_frame`)."""

    text: str
    color: tuple[int, int, int, int] = HUD_COLOR
    bold: bool = False


@dataclass(frozen=True)
class CardText:
    """The content of one intro or outro card, drawn by `card_frame`."""

    eyebrow: str
    """Small accent line above the title (`TASK GOAL`, `ATTEMPT 1/3`, `RESULT`)."""
    title: str
    """The card's headline (the task name or the final status word)."""
    lines: tuple[CardLine, ...] = ()
    """Body lines: goal, success condition, controls, achieved metrics."""
    footer: str = ""
    """One muted line at the bottom (checkpoint / physics / duration)."""
    title_color: tuple[int, int, int, int] = TITLE_COLOR
    """Colour of the headline (SUCCESS/FAILED colour for a result card)."""


def dim_frame(frame: np.ndarray, *, factor: float = CARD_SCRIM, blur: float = 3.0) -> np.ndarray:
    """Darkened, blurred copy of a rendered frame (the backdrop of an intro/outro card).

    A card is never a black frame: the task scene stays visible behind the panel, just dimmed
    and out of focus so the card text reads on top of it.
    """
    image = Image.fromarray(np.ascontiguousarray(frame)).convert("RGB")
    if blur > 0:
        image = image.filter(ImageFilter.GaussianBlur(float(blur)))
    pixels = np.asarray(image, dtype=np.float32) * float(np.clip(factor, 0.0, 1.0))
    return np.asarray(np.clip(pixels, 0.0, 255.0), dtype=np.uint8)


def card_frame(background: np.ndarray, card: CardText) -> np.ndarray:
    """Draw an intro/outro card over the dimmed `background` frame.

    The card is a HUD-style centred panel: the eyebrow in the accent colour, a large headline
    in `card.title_color`, a separator and the body lines, plus the muted footer. The panel is
    sized from the frame height, so a 640x480 clip and a 1280x480 side-by-side clip get
    readable cards of the same text size.

    Args:
        background: RGB uint8 scene frame (the episode's first or last frame); it is dimmed
            and blurred to become the card backdrop.
        card: The card text (`CardText`).

    Returns:
        The decorated RGB uint8 frame.
    """
    height, width = background.shape[:2]
    image = Image.fromarray(dim_frame(background)).convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    radius = max(5, height // 70)
    margin = max(10, height // 26)
    pad = max(8, height // 40)
    bar_w = max(4, height // 90)
    panel_w = min(width - 2 * margin, max(360, round(width * 0.80)))
    inner_w = panel_w - 2 * pad - bar_w - pad // 2
    eyebrow_font = fit_font(draw, card.eyebrow, inner_w, max(11, round(height / 34)), bold=True)
    title_font = fit_font(draw, card.title, inner_w, max(20, round(height / 13)), bold=True)
    _, eyebrow_h = text_size(draw, card.eyebrow, eyebrow_font)
    _, title_h = text_size(draw, card.title, title_font)

    # Body lines: one readable font size for all lines; long lines wrap over several rows.
    line_size = max(10, round(height / 30))
    line_fonts = [load_font(line_size, bold=line.bold) for line in card.lines]
    while line_size > 9 and any(
        line.text and max(text_size(draw, word, font)[0] for word in line.text.split()) > inner_w
        for line, font in zip(card.lines, line_fonts)
    ):
        line_size -= 1
        line_fonts = [load_font(line_size, bold=line.bold) for line in card.lines]
    wrapped: list[tuple[str, tuple[int, int, int, int], Any]] = []
    for line, font in zip(card.lines, line_fonts):
        for chunk in wrap_text(draw, line.text, font, inner_w):
            wrapped.append((chunk, line.color, font))
    line_heights = [text_size(draw, text, font)[1] if text else 0 for text, _color, font in wrapped]
    line_h = max(line_heights, default=0)
    footer_font = load_font(max(9, round(height / 46)))
    footer_lines: list[str] = []
    if card.footer:
        footer_font = fit_font(draw, card.footer, inner_w, max(9, round(height / 46)))
        footer_lines = wrap_text(draw, card.footer, footer_font, inner_w)
    footer_line_h = max((text_size(draw, line, footer_font)[1] for line in footer_lines), default=0)
    footer_h = footer_line_h * len(footer_lines)

    separator_h = max(1, height // 320)
    panel_h = (
        3 * pad
        + eyebrow_h
        + pad // 2
        + title_h
        + pad
        + separator_h
        + pad // 2
        + len(wrapped) * line_h
        + (pad + footer_h if card.footer else 0)
    )
    panel_h = min(panel_h, height - 2 * margin)
    x0 = (width - panel_w) // 2
    y0 = max(margin, (height - panel_h) // 2)
    x1, y1 = x0 + panel_w, y0 + panel_h
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=CARD_PANEL_RGBA)
    draw.rounded_rectangle(
        [x0 + pad // 2, y0 + pad, x0 + pad // 2 + bar_w, y1 - pad],
        radius=bar_w // 2,
        fill=ACCENT_RGBA,
    )
    text_x = x0 + pad + bar_w + pad // 2
    cursor = y0 + pad
    draw.text(
        (text_x, cursor),
        card.eyebrow,
        font=eyebrow_font,
        fill=ACCENT_RGBA if card.title_color == TITLE_COLOR else card.title_color,
        stroke_width=1,
        stroke_fill=(0, 0, 0, 180),
    )
    cursor += eyebrow_h + pad // 2
    draw.text(
        (text_x, cursor),
        card.title,
        font=title_font,
        fill=card.title_color,
        stroke_width=2,
        stroke_fill=(0, 0, 0, 210),
    )
    cursor += title_h + pad
    draw.line(
        [(text_x, cursor), (x1 - pad, cursor)],
        fill=(255, 255, 255, 45),
        width=separator_h,
    )
    cursor += separator_h + pad // 2
    for text, color, font in wrapped:
        if text:
            draw.text(
                (text_x, cursor),
                text,
                font=font,
                fill=color,
                stroke_width=1,
                stroke_fill=(0, 0, 0, 170),
            )
        cursor += line_h
    for index, line in enumerate(footer_lines):
        draw.text(
            (text_x, y1 - pad - footer_h + index * footer_line_h),
            line,
            font=footer_font,
            fill=MUTED_COLOR,
        )
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"), dtype=np.uint8)


def hstack_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Horizontally stack two RGB frames of equal size with a thin divider between them."""
    if left.shape != right.shape:
        raise ValueError(f"frames must match, got {left.shape} and {right.shape}")
    joined = np.concatenate([left, right], axis=1)
    middle = left.shape[1]
    joined[:, middle - 1 : middle + 1] = np.asarray((36, 48, 68), dtype=np.uint8)
    return joined


# --- encoding ----------------------------------------------------------------------------
def _start_frames(frames: Iterable[np.ndarray]) -> tuple[np.ndarray, Iterator[np.ndarray]]:
    """Split an iterable of frames into its first frame and the iterator over the rest."""
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("no frames to encode") from exc
    return np.ascontiguousarray(first, dtype=np.uint8), iterator


def _frame_size(frame: np.ndarray) -> tuple[int, int]:
    """`(width, height)` of an RGB frame; the yuv420 encoders need even dimensions."""
    height, width = frame.shape[:2]
    if width % 2 or height % 2:
        raise ValueError(f"encoder needs even frame sizes, got {width}x{height}")
    return int(width), int(height)


def _write_mp4_cv2(path: Path, frames: Iterable[np.ndarray], fps: float) -> str:
    """Encode with `cv2.VideoWriter` (`mp4v`); returns "cv2"."""
    import cv2

    first, rest = _start_frames(frames)
    width, height = _frame_size(first)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {path}")
    try:
        for frame in chain((first,), rest):
            writer.write(np.ascontiguousarray(frame[:, :, ::-1], dtype=np.uint8))
    finally:
        writer.release()
    return "cv2"


def _write_mp4_ffmpeg(path: Path, frames: Iterable[np.ndarray], fps: float) -> str:
    """Encode with an `ffmpeg` subprocess (H.264/yuv420p); returns "ffmpeg"."""
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise RuntimeError("neither cv2 nor ffmpeg is available to encode MP4")
    first, rest = _start_frames(frames)
    width, height = _frame_size(first)
    command = [
        binary,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdin is None:  # pragma: no cover -- Popen with PIPE always provides stdin
        raise RuntimeError("ffmpeg stdin is not available")
    try:
        for frame in chain((first,), rest):
            process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    finally:
        process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {stderr.strip()}")
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no data for {path}")
    return "ffmpeg"


def write_mp4(path: Path, frames: Iterable[np.ndarray], fps: float) -> str:
    """Encode RGB frames (streamed) to an MP4 file; returns the encoder that was used.

    Prefers `cv2.VideoWriter` (`mp4v`) when OpenCV is installed and falls back to the
    `ffmpeg` binary (H.264) otherwise. The pod's venv has no OpenCV, so the fallback is the
    path used for the demo videos; nothing is installed either way.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # noqa: F401 -- availability probe only
    except ImportError:
        return _write_mp4_ffmpeg(path, frames, fps)
    return _write_mp4_cv2(path, frames, fps)


def write_gif(
    path: Path,
    frames: Iterable[np.ndarray],
    *,
    frame_duration_ms: int,
    width: int,
) -> int:
    """Write a looping GIF preview from `frames`, downscaled to `width` px; returns frames."""
    images: list[Image.Image] = []
    for frame in frames:
        image = Image.fromarray(np.ascontiguousarray(frame)).convert("RGB")
        target_width = min(int(width), image.width)
        target_height = max(1, round(image.height * target_width / image.width))
        images.append(image.resize((target_width, target_height), Image.LANCZOS))
    if not images:
        raise ValueError("no frames to encode")
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=max(20, int(frame_duration_ms)),
        loop=0,
        optimize=True,
    )
    return len(images)


def probe_video(path: Path) -> dict[str, Any]:
    """Best-effort video probe (frame count, size, duration, fps) via `ffprobe`."""
    binary = shutil.which("ffprobe")
    if binary is None or not path.exists():
        return {}
    command = [
        binary,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,width,height,r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {}
    try:
        stream = json.loads(completed.stdout)["streams"][0]
    except (KeyError, IndexError, json.JSONDecodeError):
        return {}
    numerator, _, denominator = str(stream.get("r_frame_rate", "0/1")).partition("/")
    fps = float(numerator) / float(denominator) if float(denominator or 0) else 0.0
    frames = int(stream.get("nb_read_frames") or 0)
    duration = float(stream.get("duration") or 0.0)
    if duration <= 0 and frames and fps > 0:
        duration = frames / fps
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "frames": frames,
        "fps": round(fps, 3),
        "duration_s": round(duration, 3),
    }


def mean_luma(frame: np.ndarray) -> float:
    """Mean brightness of an RGB frame (a black frame would score ~0)."""
    return float(np.asarray(frame, dtype=np.float64).mean())


def save_png_stills(
    out_dir: Path, stem: str, frames: Sequence[np.ndarray], count: int = 3
) -> list[Path]:
    """Save the first `count` frames as `{stem}_frame{i}.png`; returns the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, frame in enumerate(frames[:count]):
        path = out_dir / f"{stem}_frame{index}.png"
        Image.fromarray(np.ascontiguousarray(frame)).save(path)
        paths.append(path)
    return paths


def degrees(radians: float) -> float:
    """Radians to degrees (kept here so the demo scripts share one conversion)."""
    return float(np.degrees(float(radians)))
