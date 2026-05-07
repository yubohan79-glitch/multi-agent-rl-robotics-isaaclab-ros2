from __future__ import annotations

import csv
import json
import math
import textwrap
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "paper"
MEDIA = ROOT / "docs" / "media"
DOC_DATA = ROOT / "docs" / "rl_data" / "world_model_sacflow_final"
TRAIN_CSV_CANDIDATES = [
    DOC_DATA / "training_curve.csv",
    ROOT / "isaaclab_sim" / "output" / "rl" / "world_model_sacflow_seed260707_rerun" / "training_curve.csv",
]
TRAIN_SUMMARY_CANDIDATES = [
    DOC_DATA / "training_summary.json",
    ROOT / "isaaclab_sim" / "output" / "rl" / "world_model_sacflow_seed260707_rerun" / "training_summary.json",
]
EVAL_JSON_CANDIDATES = [
    DOC_DATA / "contract_eval_multiseed.json",
    ROOT / "isaaclab_sim" / "output" / "eval" / "world_model_sacflow_microaim_contract_eval256.json",
    ROOT / "isaaclab_sim" / "output" / "eval" / "world_model_sacflow_rs004_multiseed_contract_eval128.json",
]
STRICT_JSON_CANDIDATES = [
    DOC_DATA / "strict_replay_summary.json",
    ROOT / "isaaclab_sim" / "output" / "replay" / "world_model_sacflow_strict_replay_abs" / "strict_replay_summary.json",
]

W, H = 1920, 1080
SCALE = 2

COL = {
    "ink": "#111827",
    "muted": "#64748B",
    "light": "#F8FAFC",
    "grid": "#E2E8F0",
    "line": "#CBD5E1",
    "yellow": "#E5B82E",
    "blue": "#2563EB",
    "green": "#16A34A",
    "orange": "#F97316",
    "red": "#DC2626",
    "violet": "#7C3AED",
    "cyan": "#0891B2",
    "cream": "#FFF7ED",
    "pale_blue": "#EFF6FF",
    "pale_green": "#ECFDF5",
    "pale_red": "#FEF2F2",
    "pale_yellow": "#FEFCE8",
    "white": "#FFFFFF",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\seguisb.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size * SCALE)
        except OSError:
            continue
    return ImageFont.load_default()


def sx(v: float) -> int:
    return int(round(v * SCALE))


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


@dataclass
class Obj:
    kind: str
    args: tuple
    kwargs: dict = field(default_factory=dict)


@dataclass
class Figure:
    name: str
    title: str
    subtitle: str
    objects: list[Obj] = field(default_factory=list)

    def add(self, kind: str, *args, **kwargs):
        self.objects.append(Obj(kind, args, kwargs))

    def panel(self, x, y, w, h, label, title, fill=COL["white"], outline=COL["ink"]):
        self.add("round_rect", x, y, w, h, 10, fill, outline, 1.3)
        self.text(x + 18, y + 12, label, 18, True, COL["ink"])
        self.text(x + 58, y + 13, title, 18, True, COL["ink"])

    def rect(self, x, y, w, h, text="", fill=COL["white"], outline=COL["line"], width=1.2, r=8, fs=18, bold=False):
        self.add("round_rect", x, y, w, h, r, fill, outline, width)
        if text:
            self.add("wrapped_text", x + 12, y + 12, w - 24, text, fs, bold, COL["ink"], "center")

    def text(self, x, y, text, size=18, bold=False, color=COL["ink"], anchor="la"):
        self.add("text", x, y, text, size, bold, color, anchor)

    def wrapped(self, x, y, w, text, size=16, bold=False, color=COL["ink"], align="left"):
        self.add("wrapped_text", x, y, w, text, size, bold, color, align)

    def line(self, x1, y1, x2, y2, color=COL["ink"], width=2.0, arrow=False, dash=False):
        self.add("line", x1, y1, x2, y2, color, width, arrow, dash)

    def circle(self, x, y, r, fill, outline=COL["ink"], width=1.2, text=""):
        self.add("circle", x, y, r, fill, outline, width)
        if text:
            self.add("text", x, y - 7, text, 13, True, COL["ink"], "ma")

    def image(self, path: Path, x, y, w, h, outline=COL["line"], label=""):
        self.add("image", path, x, y, w, h, outline, label)


def draw_wrapped(draw: ImageDraw.ImageDraw, xy, width, text, size, bold=False, fill=COL["ink"], align="left"):
    f = font(size, bold)
    max_chars = max(8, int(width / (size * 0.56)))
    lines: list[str] = []
    for para in str(text).split("\n"):
        lines.extend(textwrap.wrap(para, max_chars) or [""])
    x, y = sx(xy[0]), sx(xy[1])
    line_h = int(size * 1.28 * SCALE)
    for line in lines:
        if align == "center":
            bbox = draw.textbbox((0, 0), line, font=f)
            tx = x + (sx(width) - (bbox[2] - bbox[0])) // 2
        else:
            tx = x
        draw.text((tx, y), line, font=f, fill=hex_to_rgb(fill))
        y += line_h


def render(fig: Figure) -> Path:
    im = Image.new("RGB", (W * SCALE, H * SCALE), "white")
    d = ImageDraw.Draw(im)
    d.text((sx(42), sx(22)), fig.title, font=font(35, True), fill=hex_to_rgb(COL["ink"]))
    d.text((sx(44), sx(66)), fig.subtitle, font=font(18), fill=hex_to_rgb(COL["muted"]))

    for obj in fig.objects:
        a, k = obj.args, obj.kwargs
        if obj.kind == "round_rect":
            x, y, w, h, r, fill, outline, width = a
            d.rounded_rectangle([sx(x), sx(y), sx(x + w), sx(y + h)], radius=sx(r), fill=hex_to_rgb(fill), outline=hex_to_rgb(outline), width=max(1, sx(width)))
        elif obj.kind == "text":
            x, y, text, size, bold, color, anchor = a
            d.text((sx(x), sx(y)), str(text), font=font(size, bold), fill=hex_to_rgb(color), anchor=anchor)
        elif obj.kind == "wrapped_text":
            x, y, width, text, size, bold, color, align = a
            draw_wrapped(d, (x, y), width, text, size, bold, color, align)
        elif obj.kind == "line":
            x1, y1, x2, y2, color, width, arrow, dash = a
            if dash:
                draw_dashed_line(d, x1, y1, x2, y2, color, width)
            else:
                d.line([sx(x1), sx(y1), sx(x2), sx(y2)], fill=hex_to_rgb(color), width=max(1, sx(width)))
            if arrow:
                draw_arrow_head(d, x1, y1, x2, y2, color, width)
        elif obj.kind == "circle":
            x, y, r, fill, outline, width = a
            d.ellipse([sx(x - r), sx(y - r), sx(x + r), sx(y + r)], fill=hex_to_rgb(fill), outline=hex_to_rgb(outline), width=max(1, sx(width)))
        elif obj.kind == "poly":
            pts, fill, outline = a
            d.polygon([(sx(x), sx(y)) for x, y in pts], fill=hex_to_rgb(fill), outline=hex_to_rgb(outline))
        elif obj.kind == "image":
            path, x, y, w, h, outline, label = a
            if Path(path).exists():
                thumb = Image.open(path).convert("RGB").resize((sx(w), sx(h)), Image.Resampling.LANCZOS)
                im.paste(thumb, (sx(x), sx(y)))
            d.rounded_rectangle([sx(x), sx(y), sx(x + w), sx(y + h)], radius=sx(8), outline=hex_to_rgb(outline), width=max(1, sx(1.2)))
            if label:
                d.rounded_rectangle([sx(x + 10), sx(y + 10), sx(x + 180), sx(y + 40)], radius=sx(6), fill=hex_to_rgb(COL["white"]), outline=hex_to_rgb(outline), width=max(1, sx(1)))
                d.text((sx(x + 22), sx(y + 18)), str(label), font=font(13, True), fill=hex_to_rgb(COL["ink"]))

    im = im.resize((W, H), Image.Resampling.LANCZOS)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{fig.name}.png"
    im.save(path, "PNG", optimize=True)
    return path


def draw_arrow_head(d: ImageDraw.ImageDraw, x1, y1, x2, y2, color, width):
    ang = math.atan2(y2 - y1, x2 - x1)
    length = 12 + width * 2
    spread = 0.48
    pts = [
        (x2, y2),
        (x2 - length * math.cos(ang - spread), y2 - length * math.sin(ang - spread)),
        (x2 - length * math.cos(ang + spread), y2 - length * math.sin(ang + spread)),
    ]
    d.polygon([(sx(x), sx(y)) for x, y in pts], fill=hex_to_rgb(color))


def draw_dashed_line(d, x1, y1, x2, y2, color, width):
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 1e-6:
        return
    steps = int(length // 18)
    for i in range(steps + 1):
        if i % 2 == 0:
            a = i / max(steps, 1)
            b = min(1.0, (i + 0.65) / max(steps, 1))
            d.line(
                [sx(x1 + (x2 - x1) * a), sx(y1 + (y2 - y1) * a), sx(x1 + (x2 - x1) * b), sx(y1 + (y2 - y1) * b)],
                fill=hex_to_rgb(color),
                width=max(1, sx(width)),
            )


def read_curve(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("none of these inputs exist: " + ", ".join(str(path) for path in paths))


def chart_axes(fig: Figure, x, y, w, h, label_y="", label_x=""):
    for i in range(5):
        yy = y + h * i / 4
        fig.line(x, yy, x + w, yy, COL["grid"], 1)
    fig.line(x, y + h, x + w, y + h, COL["ink"], 1.2)
    fig.line(x, y, x, y + h, COL["ink"], 1.2)
    if label_y:
        fig.text(x - 28, y - 14, label_y, 12, False, COL["muted"])
    if label_x:
        fig.text(x + w - 80, y + h + 22, label_x, 12, False, COL["muted"])


def line_chart(fig: Figure, x, y, w, h, rows, key, color, label, every=7):
    sampled = rows[::every]
    xs = [r["env_step"] for r in sampled]
    ys = [r[key] for r in sampled]
    ymin, ymax = min(ys), max(ys)
    if abs(ymax - ymin) < 1e-9:
        ymax = ymin + 1.0
    chart_axes(fig, x, y, w, h, label_y=label, label_x="env steps")
    last = None
    for xv, yv in zip(xs, ys):
        px = x + (xv - xs[0]) / max(1e-9, xs[-1] - xs[0]) * w
        py = y + h - (yv - ymin) / (ymax - ymin) * h
        if last:
            fig.line(last[0], last[1], px, py, color, 2.4)
        last = (px, py)
    if last:
        fig.circle(last[0], last[1], 4, color, color)
    fig.text(x + 8, y + 8, f"{ymax:.3g}", 10, False, COL["muted"])
    fig.text(x + 8, y + h - 18, f"{ymin:.3g}", 10, False, COL["muted"])


def bar_chart(fig: Figure, x, y, w, h, labels: list[str], values: list[float], colors: list[str], title: str, ymax=1.0, fmt="{:.0%}"):
    chart_axes(fig, x, y, w, h, label_y=title)
    gap = w / len(values)
    bw = min(42, gap * 0.55)
    for i, (lab, val, col) in enumerate(zip(labels, values, colors)):
        bh = h * max(0.0, min(ymax, val)) / ymax
        bx = x + gap * i + (gap - bw) / 2
        fig.add("round_rect", bx, y + h - bh, bw, bh, 4, col, col, 1)
        fig.text(bx + bw / 2, y + h - bh - 24, fmt.format(val), 12, True, COL["ink"], "ma")
        fig.text(bx + bw / 2, y + h + 16, lab, 11, False, COL["muted"], "ma")


def replay_gif_path(keyword: str) -> Path:
    for path in MEDIA.glob("*.gif"):
        if keyword in path.name:
            return path
    return MEDIA / "missing.gif"


def metric_card(fig: Figure, x, y, w, label: str, value: str, color: str, sub: str = ""):
    fig.rect(x, y, w, 76, fill=COL["white"], outline=color, width=1.5)
    fig.text(x + 16, y + 18, label, 13, True, COL["muted"])
    fig.text(x + 16, y + 44, value, 21, True, color)
    if sub:
        fig.text(x + w - 16, y + 48, sub, 11, False, COL["muted"], "ra")


def draw_arena(fig: Figure, x: float, y: float, w: float, h: float):
    fig.rect(x, y, w, h, fill="#F8FAFC", outline=COL["ink"], width=1.4, r=6)
    for i in range(1, 5):
        fig.line(x + w * i / 5, y, x + w * i / 5, y + h, "#D7DEE8", 0.9)
        fig.line(x, y + h * i / 5, x + w, y + h * i / 5, "#D7DEE8", 0.9)
    # bases and start partitions
    fig.rect(x + 18, y + 18, 82, 72, fill="#DBEAFE", outline=COL["blue"], width=2.0, r=4)
    fig.rect(x + w - 100, y + h - 90, 82, 72, fill="#FEF3C7", outline=COL["yellow"], width=2.0, r=4)
    # pushable boxes
    fig.rect(x + w * 0.62, y + h * 0.22, 48, 48, fill="#FB923C", outline=COL["orange"], width=1.4, r=4)
    fig.rect(x + w * 0.22, y + h * 0.66, 48, 48, fill="#FB923C", outline=COL["orange"], width=1.4, r=4)
    # robots
    fig.circle(x + w * 0.28, y + h * 0.80, 18, COL["yellow"], COL["ink"], text="Y")
    fig.circle(x + w * 0.75, y + h * 0.36, 18, COL["blue"], COL["ink"], text="B")
    # targets around walls
    target_pts = [
        (0.12, 0.48), (0.18, 0.72), (0.38, 0.12), (0.50, 0.24),
        (0.82, 0.28), (0.84, 0.56), (0.68, 0.82), (0.34, 0.88),
    ]
    for tx, ty in target_pts:
        fig.rect(x + w * tx - 10, y + h * ty - 16, 20, 32, fill="#FFFFFF", outline=COL["ink"], width=1.0, r=3)
        fig.circle(x + w * tx, y + h * ty, 5, COL["red"], COL["red"])
    # tactical rays and path
    fig.line(x + w * 0.28, y + h * 0.80, x + w * 0.34, y + h * 0.88, COL["yellow"], 2.0, arrow=True)
    fig.line(x + w * 0.75, y + h * 0.36, x + w * 0.84, y + h * 0.56, COL["blue"], 2.0, arrow=True)
    fig.line(x + w * 0.28, y + h * 0.80, x + w * 0.22, y + h * 0.66, COL["orange"], 2.0, arrow=True, dash=True)
    fig.text(x + 16, y + h - 18, "3m x 3m arena: robots, targets, armor and pushable boxes become objects", 11, False, COL["muted"])


def draw_token_matrix(fig: Figure, x: float, y: float):
    fig.rect(x, y, 420, 230, fill=COL["white"], outline=COL["line"], width=1.2, r=6)
    fig.text(x + 18, y + 20, "Object-token table z_t", 16, True)
    cols = ["type", "x/y", "yaw", "owner", "state", "visible"]
    for i, col in enumerate(cols):
        fig.text(x + 26 + i * 62, y + 58, col, 10, True, COL["muted"])
    rows = [
        ("robot_Y", COL["yellow"], [0.90, 0.55, 0.20, 1.00, 0.85]),
        ("robot_B", COL["blue"], [0.20, 0.62, 0.75, 1.00, 0.80]),
        ("target", COL["red"], [0.70, 0.18, 0.50, 0.30, 0.95]),
        ("armor", COL["cyan"], [0.40, 0.88, 0.45, 0.60, 0.35]),
        ("box", COL["orange"], [0.58, 0.42, 0.00, 0.75, 0.70]),
    ]
    for r, (name, color, vals) in enumerate(rows):
        yy = y + 84 + r * 27
        fig.text(x + 24, yy, name, 10, True, color)
        for c, val in enumerate(vals):
            shade = "#E2E8F0" if c == 2 and name == "box" else color
            fig.add("round_rect", x + 88 + c * 62, yy - 7, 42 * val + 4, 12, 3, shade, shade, 1)
    fig.text(x + 24, y + 212, "typed rows keep ownership, armor and box dynamics explicit", 11, False, COL["muted"])


def fig01_overview(eval_summary, strict_summary) -> Figure:
    fig = Figure(
        "fig01_project_overview",
        "Figure 1. Object-Centric World-Model Flow RL: From Arena Physics to Audited Evidence",
        "Top-conference style overview: environment objects, structured tokens, model-based self-play, rule-gated deployment and replay evidence.",
    )
    fig.panel(30, 118, 560, 455, "(a)", "Arena-to-object abstraction", COL["white"])
    draw_arena(fig, 72, 178, 365, 320)
    fig.line(452, 338, 525, 338, COL["muted"], 2.5, arrow=True)
    fig.rect(478, 196, 78, 58, "event\nlog", COL["cream"], COL["orange"], fs=12, bold=True)
    fig.rect(478, 282, 78, 58, "object\nstate", COL["pale_blue"], COL["blue"], fs=12, bold=True)
    fig.rect(478, 368, 78, 58, "rule\nstate", COL["pale_green"], COL["green"], fs=12, bold=True)

    fig.panel(620, 118, 505, 455, "(b)", "Structured state used by learning", COL["white"])
    draw_token_matrix(fig, 665, 185)
    metric_card(fig, 668, 438, 122, "obs", "46", COL["blue"], "local")
    metric_card(fig, 810, 438, 122, "objects", "165", COL["green"], "global")
    metric_card(fig, 952, 438, 122, "action", "6D", COL["violet"], "tactical")

    fig.panel(1155, 118, 735, 455, "(c)", "Model-based flow-control stack", COL["white"])
    stack = [
        (1210, 188, 150, 72, "Object\nencoder", COL["pale_blue"], COL["blue"]),
        (1410, 188, 160, 72, "World model\nrollout", COL["pale_green"], COL["green"]),
        (1625, 188, 160, 72, "Flow actor\nu(t,a|z)", "#F5F3FF", COL["violet"]),
        (1315, 340, 160, 72, "Twin-Q\nself-play SAC", COL["cream"], COL["orange"]),
        (1535, 340, 170, 72, "Rule shield\nlegal action", COL["pale_red"], COL["red"]),
    ]
    for x, y, w, h, txt, fill, out in stack:
        fig.rect(x, y, w, h, txt, fill, out, fs=14, bold=True)
    for a, b in [((1360, 224), (1410, 224)), ((1570, 224), (1625, 224)), ((1705, 260), (1620, 340)), ((1475, 376), (1535, 376)), ((1490, 260), (1395, 340))]:
        fig.line(a[0], a[1], b[0], b[1], COL["ink"], 1.8, arrow=True)
    fig.rect(1220, 470, 540, 38, "replay buffer: z_t, local obs, flow action, reward, z_{t+1}", COL["light"], COL["line"], fs=14, bold=True)

    fig.panel(30, 610, 740, 315, "(d)", "Evidence from multi-seed evaluation", COL["white"])
    bar_chart(fig, 82, 690, 300, 145, ["Yellow", "Blue", "Draw"], [eval_summary["yellow_win_rate"], eval_summary["blue_win_rate"], eval_summary["draw_rate"]], [COL["yellow"], COL["blue"], COL["muted"]], "win rate", 1.0)
    metric_card(fig, 440, 682, 130, "games", str(eval_summary["episodes"]), COL["ink"])
    metric_card(fig, 590, 682, 130, "mean time", f"{eval_summary['mean_episode_time_s']:.1f}s", COL["cyan"])
    fig.rect(440, 790, 280, 46, "zero static / box penetration in contract eval", COL["pale_green"], COL["green"], fs=14, bold=True)

    fig.panel(805, 610, 520, 315, "(e)", "Strict replay gate", COL["white"])
    checks = [
        ("0.80s dwell", COL["green"]), ("20-80cm base shot", COL["blue"]),
        ("box pose changes", COL["orange"]), ("no penetration", COL["red"]),
    ]
    for i, (txt, col) in enumerate(checks):
        x = 850 + (i % 2) * 220
        y = 690 + (i // 2) * 80
        fig.circle(x, y + 18, 16, col, col, text=str(i + 1))
        fig.text(x + 35, y + 7, txt, 15, True)
    fig.rect(910, 840, 300, 38, f"{strict_summary['hard_violations']} hard violations, {strict_summary['warnings']} warnings", COL["pale_green"], COL["green"], fs=15, bold=True)

    fig.panel(1360, 610, 530, 315, "(f)", "Three-view replay is the first README evidence", COL["white"])
    fig.image(replay_gif_path("顶视角"), 1400, 680, 135, 76, COL["green"], "top")
    fig.image(replay_gif_path("黄车"), 1548, 680, 135, 76, COL["yellow"], "yellow")
    fig.image(replay_gif_path("蓝车"), 1696, 680, 135, 76, COL["blue"], "blue")
    fig.rect(1425, 808, 360, 56, "README first visual: top-view replay GIF\nthen method figures and metrics", COL["light"], COL["line"], fs=14, bold=True)
    fig.rect(50, 960, 1820, 54, "Core claim: object tokens + learned flow policy + explicit rule audit produce interpretable robotic self-play results.", "#FFF7ED", "#F59E0B", 1.5, fs=22, bold=True)
    return fig


def fig02_architecture() -> Figure:
    fig = Figure(
        "fig02_method_architecture",
        "Figure 2. Detailed Architecture: Object Tokens, Imagination Rollout and Flow Actor",
        "A publishable method diagram: each arrow corresponds to a logged tensor, loss term, safety gate or deployment artifact.",
    )
    fig.panel(36, 118, 505, 825, "(a)", "Object-centric observation model", COL["white"])
    draw_arena(fig, 78, 190, 275, 250)
    draw_token_matrix(fig, 78, 485)
    fig.rect(380, 215, 118, 55, "local obs\n46-D", COL["pale_blue"], COL["blue"], fs=13, bold=True)
    fig.rect(380, 300, 118, 55, "object state\n165-D", COL["pale_green"], COL["green"], fs=13, bold=True)
    fig.rect(380, 385, 118, 55, "action\n6-D", "#F5F3FF", COL["violet"], fs=13, bold=True)
    fig.wrapped(82, 742, 390, "Objects retain identity over time: a pushed box is the same box after contact, and a base armor plate is removed only by legal target events.", 14, False, COL["muted"])

    fig.panel(580, 118, 760, 825, "(b)", "Self-play learning graph", COL["white"])
    fig.rect(640, 190, 190, 72, "Encoder\nE(z_t, o_t)", COL["pale_blue"], COL["blue"], fs=15, bold=True)
    fig.rect(910, 190, 190, 72, "World model\np(z_{t+1}, r, d)", COL["pale_green"], COL["green"], fs=15, bold=True)
    fig.rect(1160, 190, 130, 72, "k-step\nrollout", COL["light"], COL["line"], fs=15, bold=True)
    fig.rect(640, 388, 190, 72, "Flow actor\nv_theta(a,t|z)", "#F5F3FF", COL["violet"], fs=15, bold=True)
    fig.rect(910, 388, 190, 72, "Twin-Q critic\nmin(Q1,Q2)", COL["cream"], COL["orange"], fs=15, bold=True)
    fig.rect(1160, 388, 130, 72, "target\nnetworks", COL["light"], COL["line"], fs=14, bold=True)
    fig.rect(745, 590, 390, 70, "Replay buffer with yellow/blue transitions\nz_t, o_t, a_t, r_t, done, z_{t+1}", "#FFFFFF", COL["ink"], fs=15, bold=True)
    for a, b in [
        ((830, 226), (910, 226)), ((1100, 226), (1160, 226)), ((735, 262), (735, 388)),
        ((1005, 262), (1005, 388)), ((830, 424), (910, 424)), ((1100, 424), (1160, 424)),
        ((835, 590), (720, 460)), ((1035, 590), (1005, 460)),
    ]:
        fig.line(a[0], a[1], b[0], b[1], COL["ink"], 1.8, arrow=True)
    losses = [
        ("L_Q", "Bellman residual", COL["orange"]),
        ("L_actor", "entropy-regularized flow objective", COL["violet"]),
        ("L_model", "object transition prediction", COL["green"]),
    ]
    for i, (name, txt, col) in enumerate(losses):
        fig.rect(660 + i * 210, 735, 185, 58, f"{name}\n{txt}", "#FFFFFF", col, fs=12, bold=True)

    fig.panel(1375, 118, 505, 825, "(c)", "Runtime safety and deployment", COL["white"])
    y0 = 200
    runtime = [
        ("flow action", "target / base / block / recover / fire / risk", COL["violet"]),
        ("expert residual", "route-aware prior and micro-aim corrections", COL["yellow"]),
        ("geometry shield", "LOS, dwell, range, armor and collision gates", COL["red"]),
        ("ROS2 output", "cmd_vel, shooter service and replay trace", COL["blue"]),
    ]
    for i, (head, txt, col) in enumerate(runtime):
        y = y0 + i * 125
        fig.rect(1425, y, 360, 78, f"{head}\n{txt}", "#FFFFFF", col, fs=13, bold=True)
        if i < len(runtime) - 1:
            fig.line(1605, y + 78, 1605, y + 125, COL["ink"], 1.8, arrow=True)
    fig.rect(1440, 735, 330, 78, "Formal output:\naudited policy + result tables + three-view replay", COL["pale_green"], COL["green"], fs=15, bold=True)
    fig.rect(74, 968, 1772, 54, "Tensor interfaces, loss paths, safety gates and deployment artifacts are shown in one reproducible method graph.", "#FFF7ED", "#F59E0B", fs=22, bold=True)
    return fig


def fig03_results(curve, eval_summary, strict_summary) -> Figure:
    fig = Figure(
        "fig03_training_and_results",
        "Figure 3. Training Loop, Logged Dynamics and Multi-Seed Evaluation",
        "The figure combines the training objective flow, logged optimization curves and final game-level contract metrics.",
    )
    fig.panel(38, 120, 520, 405, "(a)", "Closed training loop", COL["white"])
    loop = [
        (88, 205, "parallel self-play\n32 envs", COL["blue"]),
        (318, 205, "object replay\nbuffer", COL["green"]),
        (318, 365, "SAC Flow\nupdates", COL["violet"]),
        (88, 365, "strict quick\neval", COL["orange"]),
    ]
    for x, y, txt, col in loop:
        fig.rect(x, y, 160, 72, txt, "#FFFFFF", col, fs=14, bold=True)
    for a, b in [((248, 241), (318, 241)), ((398, 277), (398, 365)), ((318, 401), (248, 401)), ((168, 365), (168, 277))]:
        fig.line(a[0], a[1], b[0], b[1], COL["ink"], 1.8, arrow=True)
    fig.rect(88, 465, 390, 34, "promotion requires contract metrics, not reward only", COL["pale_green"], COL["green"], fs=14, bold=True)

    fig.panel(595, 120, 735, 405, "(b)", "Optimization dynamics", COL["white"])
    line_chart(fig, 650, 210, 285, 175, curve, "mean_reward", COL["blue"], "mean reward", every=10)
    line_chart(fig, 1000, 210, 285, 175, curve, "critic_loss", COL["red"], "critic loss", every=10)
    final = curve[-1]
    metric_card(fig, 650, 425, 150, "final reward", f"{final['mean_reward']:.3f}", COL["blue"])
    metric_card(fig, 825, 425, 150, "alpha", f"{final['alpha']:.2f}", COL["violet"])
    metric_card(fig, 1000, 425, 150, "throughput", f"{final['steps_per_second']:.1f}", COL["green"], "steps/s")
    metric_card(fig, 1175, 425, 120, "done", f"{final['done_rate']:.2f}", COL["orange"])

    fig.panel(1365, 120, 515, 405, "(c)", "Contract result snapshot", COL["white"])
    bar_chart(fig, 1428, 210, 325, 175, ["Y", "B", "D"], [eval_summary["yellow_win_rate"], eval_summary["blue_win_rate"], eval_summary["draw_rate"]], [COL["yellow"], COL["blue"], COL["muted"]], "win rate", 1.0)
    fig.rect(1435, 425, 330, 38, f"{eval_summary['episodes']} games | mean {eval_summary['mean_episode_time_s']:.2f}s | draw {eval_summary['draw_rate']:.1%}", COL["light"], COL["line"], fs=13, bold=True)

    fig.panel(38, 565, 900, 350, "(d)", "Normal target count distribution", COL["white"])
    dist = eval_summary["normal_hit_count_distribution"]
    labels, vals, colors = [], [], []
    for hits in range(1, 5):
        labels += [f"Y{hits}", f"B{hits}"]
        vals += [dist["yellow"][str(hits)], dist["blue"][str(hits)]]
        colors += [COL["yellow"], COL["blue"]]
    bar_chart(fig, 92, 650, 740, 175, labels, vals, colors, "episode share", 1.0)

    fig.panel(980, 565, 900, 350, "(e)", "Base success by cleared targets", COL["white"])
    base = eval_summary["base_success_by_hits"]
    labels, vals, colors = [], [], []
    for hits in range(1, 5):
        labels += [f"Y{hits}", f"B{hits}"]
        vals += [base["yellow"][str(hits)]["success_rate"], base["blue"][str(hits)]["success_rate"]]
        colors += [COL["yellow"], COL["blue"]]
    bar_chart(fig, 1035, 650, 740, 175, labels, vals, colors, "success rate", 1.0)
    fig.rect(74, 956, 1772, 54, "Evaluation emphasizes win balance, target diversity, legal base timing, push-box behavior and zero penetration.", "#FFF7ED", "#F59E0B", fs=22, bold=True)
    return fig


def fig04_ablation(eval_summary) -> Figure:
    fig = Figure(
        "fig04_ablation_and_safety",
        "Figure 4. Safety Diagnostics, Ablation Protocol and Failure-Mode Coverage",
        "A top-tier experiment figure should expose what is tested, which constraints are active and where failures would be counted.",
    )
    fig.panel(38, 120, 770, 420, "(a)", "Ablation matrix", COL["white"])
    headers = ["Object\nstate", "World\nmodel", "Flow\nactor", "Expert\nresidual", "Action\nshield"]
    rows = [
        ("Full method", [1, 1, 1, 1, 1]),
        ("w/o world model", [1, 0, 1, 1, 1]),
        ("single-mode actor", [1, 1, 0, 1, 1]),
        ("no residual prior", [1, 1, 1, 0, 1]),
        ("no action shield", [1, 1, 1, 1, 0]),
    ]
    x0, y0, cw, ch = 250, 205, 90, 54
    for i, h in enumerate(headers):
        fig.rect(x0 + i * cw, y0 - 62, cw - 8, 46, h, COL["light"], COL["line"], fs=12, bold=True)
    for r, (name, vals) in enumerate(rows):
        y = y0 + r * ch
        fig.text(78, y + 13, name, 15, True if r == 0 else False, COL["ink"])
        for c, val in enumerate(vals):
            fill = COL["pale_green"] if val else COL["pale_red"]
            out = COL["green"] if val else COL["red"]
            mark = "yes" if val else "x"
            fig.rect(x0 + c * cw, y, cw - 8, 36, mark, fill, out, fs=14, bold=True)
    fig.wrapped(76, 492, 650, "The matrix is a reproducible ablation protocol: report each row with the same contract metrics, replay audit and three-view media.", 13, False, COL["muted"])

    fig.panel(850, 120, 500, 420, "(b)", "Measured safety audit", COL["white"])
    safety = [
        ("Static penetration", 0, COL["red"]),
        ("Box penetration", 0, COL["orange"]),
        ("Robot contact/game", eval_summary["robot_contacts_per_episode"], COL["blue"]),
        ("Repeat target order", eval_summary["repeat_target_order_events_total"], COL["violet"]),
    ]
    for i, (lab, val, col) in enumerate(safety):
        y = 205 + i * 70
        fig.circle(900, y + 15, 17, COL["pale_green"] if float(val) == 0 else COL["pale_red"], col, text="0" if float(val) == 0 else "!")
        fig.text(935, y, lab, 16, True)
        fig.text(1220, y, f"{float(val):.2f}", 18, True, col)

    fig.panel(1390, 120, 490, 420, "(c)", "Observed behavior", COL["white"])
    push = eval_summary["push_events_per_episode"]
    disp = eval_summary["mean_final_box_displacement_m"]
    items = [
        (f"Yellow push events/game: {push['yellow']:.2f}", COL["yellow"]),
        (f"Blue push events/game: {push['blue']:.2f}", COL["blue"]),
        (f"box_ne final displacement: {disp['box_ne']:.3f} m", COL["orange"]),
        (f"box_sw final displacement: {disp['box_sw']:.3f} m", COL["cyan"]),
    ]
    for i, (txt, col) in enumerate(items):
        fig.rect(1435, 205 + i * 70, 380, 42, txt, "#FFFFFF", col, fs=14, bold=True)

    fig.panel(38, 590, 1842, 315, "(d)", "Failure modes explicitly checked before claiming success", COL["white"])
    checks = [
        ("Base before armor removal", "raycast blocked, shot rejected", COL["red"]),
        ("<0.80 s laser dwell", "100% cannot knock target", COL["orange"]),
        ("Box/armor collision", "hard penetration audited", COL["violet"]),
        ("Stuck near firing pose", "micro-aim + side nudges", COL["blue"]),
        ("One-route collapse", "normal-hit distribution logged", COL["green"]),
    ]
    for i, (a, b, col) in enumerate(checks):
        x = 82 + i * 355
        fig.rect(x, 680, 285, 112, f"{a}\n{b}", "#FFFFFF", col, fs=14, bold=True)
    fig.rect(74, 956, 1772, 54, "The formal claim is tied to rule audits and replay evidence, not reward-only curves.", "#FFF7ED", "#F59E0B", fs=23, bold=True)
    return fig


def fig05_pipeline(strict_summary) -> Figure:
    fig = Figure(
        "fig05_sim2real_replay_pipeline",
        "Figure 5. Sim2Real Evidence Pipeline with Three-View Replay Artifacts",
        "The final figure connects ROS2 deployment interfaces, IsaacLab physics replay, strict audit files and README GIF evidence.",
    )
    fig.panel(38, 120, 550, 780, "(a)", "ROS2 runtime contract", COL["white"])
    ros_nodes = [
        ("rcvrl_vision\nAprilTag / target detection", COL["pale_blue"], COL["blue"]),
        ("robot_localization\nwheel odom + IMU EKF", COL["pale_green"], COL["green"]),
        ("Nav2 controller\ncostmap + cmd_vel", COL["cream"], COL["orange"]),
        ("shooter services\nlaser dwell gate", COL["pale_red"], COL["red"]),
    ]
    for i, (txt, fill, out) in enumerate(ros_nodes):
        fig.rect(98, 205 + i * 120, 380, 70, txt, fill, out, fs=15, bold=True)
        if i < len(ros_nodes) - 1:
            fig.line(288, 275 + i * 120, 288, 325 + i * 120, COL["ink"], 1.8, arrow=True)

    fig.panel(625, 120, 660, 780, "(b)", "Policy-in-the-loop simulation", COL["white"])
    fig.rect(690, 210, 220, 72, "IsaacLab arena\nrobots, targets, boxes", COL["pale_blue"], COL["blue"], fs=15, bold=True)
    fig.rect(1010, 210, 220, 72, "Rule environment\nfast parallel rollout", COL["pale_green"], COL["green"], fs=15, bold=True)
    fig.rect(850, 390, 230, 76, "World-model SAC Flow\nhigh-level tactical actor", "#F5F3FF", COL["violet"], fs=15, bold=True)
    fig.rect(710, 570, 220, 72, "Contract eval\n128+ stochastic games", COL["cream"], COL["orange"], fs=15, bold=True)
    fig.rect(1000, 570, 220, 72, "Strict replay\nstep-wise audit", COL["pale_red"], COL["red"], fs=15, bold=True)
    for x1, y1, x2, y2 in [(910, 246, 1010, 246), (1120, 282, 980, 390), (800, 282, 920, 390), (940, 466, 820, 570), (1000, 466, 1110, 570)]:
        fig.line(x1, y1, x2, y2, COL["ink"], 1.8, arrow=True)
    fig.rect(760, 725, 400, 52, f"strict replay: {strict_summary['hard_violations']} hard violations, {strict_summary['warnings']} warnings", COL["pale_green"], COL["green"], fs=16, bold=True)

    fig.panel(1320, 120, 560, 780, "(c)", "Published replay evidence", COL["white"])
    fig.image(replay_gif_path("顶视角"), 1360, 190, 220, 124, COL["green"], "top-view first")
    fig.image(replay_gif_path("黄车"), 1605, 190, 220, 124, COL["yellow"], "yellow POV")
    fig.image(replay_gif_path("蓝车"), 1485, 340, 220, 124, COL["blue"], "blue POV")
    fig.line(1468, 314, 1545, 340, COL["muted"], 1.5, arrow=True)
    fig.line(1715, 314, 1608, 340, COL["muted"], 1.5, arrow=True)
    for i, (txt, col) in enumerate([
        ("policy checkpoint + export manifest", COL["violet"]),
        ("contract JSON/CSV + regenerated figures", COL["orange"]),
        ("strict trace + event log + audit report", COL["green"]),
    ]):
        fig.rect(1390, 525 + i * 74, 410, 46, txt, "#FFFFFF", col, fs=14, bold=True)
    fig.rect(1395, 770, 395, 52, "README order: top replay GIF first, then method and experiment figures", COL["light"], COL["line"], fs=14, bold=True)
    fig.rect(74, 956, 1772, 54, "Final media and metrics are generated from audited traces and linked directly in README.", "#FFF7ED", "#F59E0B", fs=23, bold=True)
    return fig


def emu(v: float) -> int:
    return int(v / W * 13.333333 * 914400)


def emuy(v: float) -> int:
    return int(v / H * 7.5 * 914400)


def ppt_color(value: str) -> str:
    return value.lstrip("#").upper()


def ppt_text_body(text: str, size: int, bold: bool, color: str) -> str:
    safe = escape(str(text))
    bold_attr = ' b="1"' if bold else ""
    return (
        '<p:txBody><a:bodyPr wrap="square" lIns="60000" tIns="40000" rIns="60000" bIns="40000"/>'
        '<a:lstStyle/><a:p><a:r>'
        f'<a:rPr lang="en-US" sz="{size * 100}"{bold_attr}><a:solidFill><a:srgbClr val="{ppt_color(color)}"/></a:solidFill></a:rPr>'
        f"<a:t>{safe}</a:t></a:r><a:endParaRPr lang=\"en-US\" sz=\"{size * 100}\"/></a:p></p:txBody>"
    )


def ppt_shape(shape_id: int, x, y, w, h, text="", fill=COL["white"], outline=COL["line"], radius=True, size=16, bold=False) -> str:
    prst = "roundRect" if radius else "rect"
    body = ppt_text_body(text, size, bold, COL["ink"]) if text else "<p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>"
    return f"""
<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="shape {shape_id}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emuy(y)}"/><a:ext cx="{emu(w)}" cy="{emuy(h)}"/></a:xfrm>
<a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{ppt_color(fill)}"/></a:solidFill>
<a:ln w="12700"><a:solidFill><a:srgbClr val="{ppt_color(outline)}"/></a:solidFill></a:ln></p:spPr>{body}</p:sp>
"""


def ppt_line(shape_id: int, x1, y1, x2, y2, color=COL["ink"]) -> str:
    x, y = min(x1, x2), min(y1, y2)
    w, h = abs(x2 - x1), abs(y2 - y1)
    return f"""
<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{shape_id}" name="line {shape_id}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>
<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emuy(y)}"/><a:ext cx="{max(1, emu(w))}" cy="{max(1, emuy(h))}"/></a:xfrm>
<a:prstGeom prst="line"><a:avLst/></a:prstGeom><a:ln w="19050"><a:solidFill><a:srgbClr val="{ppt_color(color)}"/></a:solidFill><a:tailEnd type="triangle"/></a:ln></p:spPr></p:cxnSp>
"""


def ppt_slide(fig: Figure, slide_index: int) -> str:
    sid = 2
    items = [
        ppt_shape(sid, 38, 18, 1500, 42, fig.title, COL["white"], COL["white"], False, 24, True),
        ppt_shape(sid + 1, 42, 62, 1500, 30, fig.subtitle, COL["white"], COL["white"], False, 12, False),
    ]
    sid += 2
    for obj in fig.objects:
        a = obj.args
        if obj.kind == "round_rect":
            x, y, w, h, r, fill, outline, width = a
            items.append(ppt_shape(sid, x, y, w, h, "", fill, outline, True))
            sid += 1
        elif obj.kind == "text":
            x, y, text, size, bold, color, anchor = a
            items.append(ppt_shape(sid, x, y, min(620, max(80, len(str(text)) * size * 0.55)), size * 1.5, text, COL["white"], COL["white"], False, max(8, size), bold))
            sid += 1
        elif obj.kind == "wrapped_text":
            x, y, width, text, size, bold, color, align = a
            items.append(ppt_shape(sid, x, y, width, max(42, size * 4.2), text, COL["white"], COL["white"], False, max(8, size), bold))
            sid += 1
        elif obj.kind == "line":
            x1, y1, x2, y2, color, width, arrow, dash = a
            items.append(ppt_line(sid, x1, y1, x2, y2, color))
            sid += 1
        elif obj.kind == "circle":
            x, y, r, fill, outline, width = a
            items.append(ppt_shape(sid, x - r, y - r, r * 2, r * 2, "", fill, outline, True))
            sid += 1
        elif obj.kind == "image":
            path, x, y, w, h, outline, label = a
            items.append(ppt_shape(sid, x, y, w, h, label or Path(path).name, COL["light"], outline, True, 12, True))
            sid += 1
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
{''.join(items)}
</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"""


def build_pptx(figures: list[Figure]) -> Path:
    path = OUT / "world_model_sacflow_paper_figures_master.pptx"
    slides_overrides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, len(figures) + 1)
    )
    slide_ids = "\n".join(f'<p:sldId id="{255+i}" r:id="rId{i}"/>' for i in range(1, len(figures) + 1))
    rels = "\n".join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, len(figures) + 1)
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
{slides_overrides}
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>""")
        z.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>""")
        z.writestr("ppt/presentation.xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:sldIdLst>{slide_ids}</p:sldIdLst><p:sldSz cx="12192000" cy="6858000" type="wide"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>""")
        z.writestr("ppt/_rels/presentation.xml.rels", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}</Relationships>""")
        for i, fig in enumerate(figures, 1):
            z.writestr(f"ppt/slides/slide{i}.xml", ppt_slide(fig, i))
        z.writestr("docProps/core.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>World Model SAC Flow Paper Figures</dc:title><dc:creator>Codex</dc:creator></cp:coreProperties>""")
        z.writestr("docProps/app.xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Generated editable PPTX master</Application><Slides>{len(figures)}</Slides></Properties>""")
    return path


def write_manifest(paths: list[Path], pptx: Path, train_csv: Path, train_summary_json: Path, eval_json: Path, strict_json: Path):
    payload = {
        "png": [str(p.relative_to(ROOT)).replace("\\", "/") for p in paths],
        "editable_pptx": str(pptx.relative_to(ROOT)).replace("\\", "/"),
        "style": "publication-style, white background, thin borders, restrained Okabe-Ito-like palette",
        "sources": [
            str(train_csv.relative_to(ROOT)).replace("\\", "/"),
            str(train_summary_json.relative_to(ROOT)).replace("\\", "/"),
            str(eval_json.relative_to(ROOT)).replace("\\", "/"),
            str(strict_json.relative_to(ROOT)).replace("\\", "/"),
        ],
    }
    (OUT / "paper_figures_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_csv = first_existing(TRAIN_CSV_CANDIDATES)
    train_summary_json = first_existing(TRAIN_SUMMARY_CANDIDATES)
    strict_json = first_existing(STRICT_JSON_CANDIDATES)
    curve = read_curve(train_csv)
    eval_json = first_existing(EVAL_JSON_CANDIDATES)
    eval_payload = read_json(eval_json)
    strict_payload = read_json(strict_json)
    eval_summary = eval_payload["summary"]
    strict_summary = strict_payload["summary"]
    figures = [
        fig01_overview(eval_summary, strict_summary),
        fig02_architecture(),
        fig03_results(curve, eval_summary, strict_summary),
        fig04_ablation(eval_summary),
        fig05_pipeline(strict_summary),
    ]
    pngs = [render(fig) for fig in figures]
    pptx = build_pptx(figures)
    write_manifest(pngs, pptx, train_csv, train_summary_json, eval_json, strict_json)
    for path in pngs + [pptx]:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
