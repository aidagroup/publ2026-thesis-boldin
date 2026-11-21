"""Plot the CubeWrench-v0 learning curves (greedy eval success vs PPO update).

Reads the trainer's stdout logs (``eval @ update N: ... success_rate=...`` lines) and
draws one line per run with Pillow (the repo's dev venv has Pillow, not matplotlib, and
this script must stay runnable on the mac and in CI). The x-axis is the PPO update; the
y-axis is the greedy success rate on the trainer's 32-episode eval.

    python scripts/plot_wrench_curves.py runs/*.log -o demo/learning_curves.png
    python scripts/plot_wrench_curves.py runs/wrench_warp_*.log --group-mode
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_EVAL_RE = re.compile(r"eval @ update (\d+):.*?success_rate=([0-9.]+)")
_MODE_RE = re.compile(r"wrench_(?:warp_)?(oracle|none|predicted)")
_COLORS = {
    "oracle": (46, 160, 67),
    "none": (56, 106, 200),
    "predicted": (200, 96, 40),
}
_W, _H = 1000, 560
_MARGIN = (80, 40, 40, 70)  # left, top, right, bottom


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Helvetica.ttc", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def parse_curves(paths: list[Path]) -> dict[str, list[tuple[int, float]]]:
    """Map a run label to its sorted `[(update, success_rate), ...]` curve."""
    curves: dict[str, list[tuple[int, float]]] = {}
    for path in paths:
        label = path.stem
        mode_match = _MODE_RE.search(label)
        if mode_match is not None:
            label = f"{mode_match.group(1)} ({path.stem.split('_s')[0].split('wrench_')[-1]})"
        points: list[tuple[int, float]] = []
        for line in path.read_text(errors="ignore").splitlines():
            match = _EVAL_RE.search(line)
            if match is not None:
                points.append((int(match.group(1)), float(match.group(2))))
        if points:
            curves[label] = sorted(points)
    return curves


def draw(curves: dict[str, list[tuple[int, float]]], out: Path, title: str) -> None:
    """Render the curves to a PNG (kept dependency-free: axes are drawn by hand)."""
    img = Image.new("RGB", (_W, _H), (250, 250, 248))
    draw = ImageDraw.Draw(img)
    font = _load_font(20)
    small = _load_font(16)
    left, top, right, bottom = _MARGIN
    plot_w = _W - left - right
    plot_h = _H - top - bottom
    max_x = max((pts[-1][0] for pts in curves.values()), default=1)
    max_x = max(max_x, 1)

    def px(x: float) -> float:
        return left + plot_w * x / max_x

    def py(y: float) -> float:
        return top + plot_h * (1.0 - y)

    draw.rectangle(
        [left, top, left + plot_w, top + plot_h], fill=(255, 255, 255), outline=(120, 120, 120)
    )
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = py(frac)
        draw.line([(left, y), (left + plot_w, y)], fill=(225, 225, 225))
        draw.text((left - 52, y - 10), f"{frac:.2f}", fill=(80, 80, 80), font=small)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = px(frac * max_x)
        draw.line([(x, top + plot_h), (x, top + plot_h + 6)], fill=(120, 120, 120))
        draw.text(
            (x - 18, top + plot_h + 12), str(int(frac * max_x)), fill=(80, 80, 80), font=small
        )
    draw.text((left, 16), title, fill=(20, 20, 20), font=font)
    draw.text((left + plot_w - 170, top + plot_h + 36), "PPO update", fill=(80, 80, 80), font=small)
    draw.text((8, top - 2), "success", fill=(80, 80, 80), font=small)

    legend_y = top + 6
    for label, pts in sorted(curves.items()):
        mode = next((m for m in _COLORS if m in label), "none")
        color = _COLORS[mode]
        if len(pts) > 1:
            draw.line([(px(x), py(y)) for x, y in pts], fill=color, width=3)
        for x, y in pts:
            draw.ellipse([px(x) - 4, py(y) - 4, px(x) + 4, py(y) + 4], fill=color)
        draw.rectangle(
            [left + plot_w - 250, legend_y, left + plot_w - 234, legend_y + 14], fill=color
        )
        draw.text((left + plot_w - 228, legend_y - 3), label, fill=(30, 30, 30), font=small)
        legend_y += 22
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="trainer stdout logs")
    parser.add_argument("-o", "--out", type=Path, default=Path("demo/learning_curves.png"))
    parser.add_argument("--title", default="CubeWrench-v0: greedy success vs PPO updates")
    parser.add_argument("--group-mode", action="store_true", help="average runs by partner mode")
    args = parser.parse_args(argv)
    curves = parse_curves(args.logs)
    if not curves:
        print("no eval lines found", file=sys.stderr)
        return 1
    if args.group_mode:
        grouped: dict[str, list[tuple[int, float]]] = {}
        for label, pts in curves.items():
            mode = next((m for m in _COLORS if m in label), "none")
            grouped.setdefault(mode, []).extend(pts)
        curves = {
            mode: sorted(
                (
                    update,
                    sum(y for u, y in pts if u == update) / sum(1 for u, _ in pts if u == update),
                )
                for update in {u for u, _ in pts}
            )
            for mode, pts in grouped.items()
        }
    draw(curves, args.out, args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
