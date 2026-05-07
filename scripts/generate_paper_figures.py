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


def fig01_overview(eval_summary, strict_summary) -> Figure:
    fig = Figure(
        "fig01_project_overview",
        "Object-Centric World-Model Flow RL for Multi-Agent Robotic Competition",
        "From ROS2/IsaacLab events to object tokens, flow-policy self-play, strict rule audit and three-view replay.",
    )
    fig.panel(28, 116, 595, 370, "(a)", "From arena events to object-centric state", COL["white"])
    fig.rect(60, 170, 238, 238, fill=COL["cream"], outline="#F2B46D")
    fig.text(86, 196, "Match event table", 18, True)
    cols = ["t", "robot", "target", "box", "score"]
    for i, c in enumerate(cols):
        fig.text(78 + i * 42, 236, c, 12, True, COL["muted"])
    rows = [("0.0", "Y", "T01", "NE", "0"), ("7.4", "B", "T08", "SW", "5"), ("18.2", "Y", "Base", "NE", "70"), ("30.8", "B", "Base", "SW", "70")]
    for r, row in enumerate(rows):
        yy = 270 + r * 36
        fig.line(72, yy - 12, 268, yy - 12, COL["line"], 1)
        for i, c in enumerate(row):
            fig.text(78 + i * 42, yy, c, 12, False, COL["ink"] if c != "Base" else COL["red"])
    fig.line(312, 286, 380, 286, COL["muted"], 3, arrow=True)
    for x, y, label, color in [
        (420, 190, "Y", COL["yellow"]), (498, 190, "B", COL["blue"]), (452, 282, "T", "#FCA5A5"), (536, 290, "Box", "#FDE68A"), (430, 374, "Armor", "#BFDBFE")
    ]:
        fig.circle(x, y, 28, color, COL["ink"], 1.2, label)
    for a, b, col in [((420, 190), (452, 282), COL["ink"]), ((498, 190), (452, 282), COL["ink"]), ((452, 282), (536, 290), COL["orange"]), ((498, 190), (430, 374), COL["blue"])]:
        fig.line(a[0], a[1], b[0], b[1], col, 2, arrow=True)
    fig.text(380, 428, "object graph / tokens", 15, False, COL["muted"])

    fig.panel(650, 116, 735, 370, "(b)", "Proposed world-model SAC Flow policy", COL["white"])
    boxes = [
        (690, 178, 170, 102, "Object encoder\nrobots / targets / boxes", COL["pale_blue"], COL["blue"]),
        (910, 178, 170, 102, "World model\nnext state / reward / done", COL["pale_green"], COL["green"]),
        (1130, 178, 170, 102, "Flow actor\ncontinuous tactical action", "#F5F3FF", COL["violet"]),
        (790, 335, 205, 88, "Centralized twin-Q\nself-play value learning", COL["cream"], COL["orange"]),
        (1070, 335, 205, 88, "Action shield + expert residual\nlegal fire / recovery / push", COL["pale_red"], COL["red"]),
    ]
    for x, y, w, h, t, fill, out in boxes:
        fig.rect(x, y, w, h, t, fill, out, 1.5, fs=15, bold=True)
    for x1, y1, x2, y2 in [(860, 229, 910, 229), (1080, 229, 1130, 229), (1215, 280, 1175, 335), (995, 379, 1070, 379), (1015, 280, 890, 335)]:
        fig.line(x1, y1, x2, y2, COL["ink"], 1.8, arrow=True)
    fig.rect(710, 445, 590, 26, "self-play replay buffer: object_state, obs, action, reward, done", COL["light"], COL["line"], fs=13)

    fig.panel(1415, 116, 475, 370, "(c)", "Rule and replay contract", COL["white"])
    checks = [
        ("0.80s laser dwell", COL["green"]), ("base range 20-80 cm", COL["blue"]), ("pushable boxes move", COL["orange"]),
        ("no wall / box penetration", COL["red"]), ("3-view IsaacLab replay", COL["violet"]),
    ]
    for i, (txt, col) in enumerate(checks):
        y = 178 + i * 52
        fig.circle(1460, y + 12, 13, col, col, text=str(i + 1))
        fig.text(1490, y, txt, 17, True, COL["ink"])
    fig.rect(1460, 430, 360, 34, f"strict audit: {strict_summary['hard_violations']} hard violations", COL["pale_green"], COL["green"], fs=15, bold=True)

    fig.panel(28, 520, 595, 360, "(d)", "Evaluation snapshot", COL["white"])
    bar_chart(fig, 90, 610, 450, 185, ["Yellow", "Blue", "Draw"], [eval_summary["yellow_win_rate"], eval_summary["blue_win_rate"], eval_summary["draw_rate"]], [COL["yellow"], COL["blue"], COL["muted"]], "win rate", 1.0)
    fig.rect(95, 822, 410, 34, f"{eval_summary['episodes']} games, mean time {eval_summary['mean_episode_time_s']:.2f}s", COL["light"], COL["line"], fs=15)

    fig.panel(650, 520, 735, 360, "(e)", "Final artifacts", COL["white"])
    for i, (name, col) in enumerate([("policy.pt", COL["violet"]), ("contract_eval.json/csv", COL["blue"]), ("strict trace", COL["green"]), ("top/yellow/blue replay", COL["orange"])]):
        fig.rect(720 + (i % 2) * 300, 610 + (i // 2) * 105, 235, 64, name, "#FFFFFF", col, 1.4, fs=15, bold=True)
    fig.line(955, 642, 1020, 642, COL["muted"], 1.6, arrow=True)
    fig.line(955, 747, 1020, 747, COL["muted"], 1.6, arrow=True)
    fig.wrapped(720, 838, 560, "All formal metrics and media are regenerated from the world-model SAC Flow checkpoint, not renamed legacy PPO/MAPPO results.", 15, False, COL["muted"])

    fig.panel(1415, 520, 475, 360, "(f)", "Main takeaway", COL["white"])
    fig.rect(1465, 612, 350, 76, "Object-centric state makes long-horizon target/base timing explicit.", COL["pale_blue"], COL["blue"], fs=16, bold=True)
    fig.rect(1465, 710, 350, 76, "Flow actor keeps multiple tactical modes instead of one fixed route.", COL["pale_green"], COL["green"], fs=16, bold=True)
    fig.rect(1465, 808, 350, 50, "Strict replay closes the sim-to-result loop.", COL["cream"], COL["orange"], fs=16, bold=True)
    fig.rect(48, 925, 1822, 72, "World model + flow policy + rule audit jointly improve multi-agent robotic strategy under physical constraints.", "#FFF7ED", "#F59E0B", 1.5, fs=24, bold=True)
    return fig


def fig02_architecture() -> Figure:
    fig = Figure(
        "fig02_method_architecture",
        "Method Architecture: Object Tokens, World Model and SAC Flow Self-Play",
        "Editable pipeline diagram for the proposed multi-agent robotic RL method.",
    )
    fig.panel(38, 120, 430, 810, "(a)", "Inputs and tokens", COL["white"])
    token_y = [205, 305, 405, 505, 605]
    tokens = [("Robot states", COL["yellow"]), ("Opponent belief", COL["blue"]), ("Targets / armor", "#FCA5A5"), ("Pushable boxes", "#FDE68A"), ("Sensor fusion", "#C4B5FD")]
    for y, (txt, col) in zip(token_y, tokens):
        fig.rect(88, y, 280, 58, txt, fill=col, outline=COL["ink"], fs=16, bold=True)
        fig.line(368, y + 29, 430, 525, COL["muted"], 1.3, arrow=True)
    fig.rect(118, 720, 250, 82, "Object state z_t\n165 dimensions", COL["pale_blue"], COL["blue"], fs=17, bold=True)

    fig.panel(505, 120, 690, 810, "(b)", "Learning core", COL["white"])
    fig.rect(560, 210, 230, 92, "Object encoder\nshared MLP/token mixer", COL["pale_blue"], COL["blue"], fs=16, bold=True)
    fig.rect(850, 210, 250, 92, "Auxiliary world model\nz_{t+1}, r_t, done_t", COL["pale_green"], COL["green"], fs=16, bold=True)
    fig.rect(560, 390, 230, 92, "SAC Flow actor\nvelocity field over actions", "#F5F3FF", COL["violet"], fs=16, bold=True)
    fig.rect(850, 390, 250, 92, "Centralized twin-Q\nQ1/Q2 over both robots", COL["cream"], COL["orange"], fs=16, bold=True)
    fig.rect(665, 585, 300, 92, "Replay buffer\n(z, obs, action, reward, next z)", "#FFFFFF", COL["ink"], fs=16, bold=True)
    for x1, y1, x2, y2 in [(790, 256, 850, 256), (675, 302, 675, 390), (975, 302, 975, 390), (790, 436, 850, 436), (710, 585, 650, 482), (940, 585, 980, 482)]:
        fig.line(x1, y1, x2, y2, COL["ink"], 1.8, arrow=True)
    fig.wrapped(560, 720, 560, "The auxiliary world model regularizes object-centric representation and prepares a clean interface for TD-MPC2/Dreamer-style imagination rollouts.", 15, False, COL["muted"])

    fig.panel(1232, 120, 650, 810, "(c)", "Deployment and safety layer", COL["white"])
    fig.rect(1295, 205, 230, 74, "Raw flow action\n6 tactical controls", "#F5F3FF", COL["violet"], fs=16, bold=True)
    fig.rect(1295, 350, 230, 74, "Expert residual\nroute / base / push prior", COL["pale_yellow"], COL["yellow"], fs=16, bold=True)
    fig.rect(1295, 495, 230, 74, "Geometry action shield\nline-of-sight, dwell, blockers", COL["pale_red"], COL["red"], fs=15, bold=True)
    fig.rect(1600, 350, 210, 74, "ROS2/Nav2 contract\ncmd_vel + shooter services", COL["pale_blue"], COL["blue"], fs=15, bold=True)
    for y in (242, 387, 532):
        fig.line(1525, y, 1600, 387, COL["ink"], 1.7, arrow=True)
    fig.rect(1340, 675, 410, 118, "Legal match action\nattack target / push box / base rush / recover / fire", COL["pale_green"], COL["green"], fs=17, bold=True)
    fig.line(1705, 424, 1705, 675, COL["ink"], 1.7, arrow=True)
    fig.rect(74, 962, 1772, 54, "Key design: learned high-level strategy stays inside explicit physical and rule contracts.", "#FFF7ED", "#F59E0B", fs=23, bold=True)
    return fig


def fig03_results(curve, eval_summary, strict_summary) -> Figure:
    fig = Figure(
        "fig03_training_and_results",
        "Training Dynamics and Multi-Seed Evaluation",
        "Actual logged metrics from the selected world-model SAC Flow checkpoint and strict replay audit.",
    )
    fig.panel(38, 120, 880, 390, "(a)", "Training curves", COL["white"])
    line_chart(fig, 92, 210, 350, 190, curve, "mean_reward", COL["blue"], "mean reward", every=10)
    line_chart(fig, 520, 210, 330, 190, curve, "actor_loss", COL["violet"], "actor loss", every=10)
    fig.rect(100, 438, 735, 38, f"200k env steps | 32 envs | batch 1024 | gradient steps 2 | RTX 4090", COL["light"], COL["line"], fs=15, bold=True)

    fig.panel(960, 120, 430, 390, "(b)", "Win balance", COL["white"])
    bar_chart(fig, 1035, 215, 285, 190, ["Y", "B", "D"], [eval_summary["yellow_win_rate"], eval_summary["blue_win_rate"], eval_summary["draw_rate"]], [COL["yellow"], COL["blue"], COL["muted"]], "win rate", 1.0)
    fig.rect(1028, 438, 300, 38, f"{eval_summary['episodes']} stochastic games", COL["pale_green"], COL["green"], fs=15, bold=True)

    fig.panel(1430, 120, 450, 390, "(c)", "Strict audit", COL["white"])
    audit_vals = [strict_summary["hard_violations"], strict_summary["warnings"], strict_summary["own_target_penalties_per_episode"], strict_summary["robot_contacts_per_episode"]]
    bar_chart(fig, 1490, 215, 315, 190, ["hard", "warn", "own", "contact"], [float(v) for v in audit_vals], [COL["red"], COL["orange"], COL["violet"], COL["blue"]], "count / game", max(1.0, max(float(v) for v in audit_vals) + 0.5), "{:.1f}")
    fig.rect(1505, 438, 300, 38, f"{strict_summary['episodes']} episodes, base wins/game {strict_summary['base_wins_per_episode']:.2f}", COL["pale_green"], COL["green"], fs=14, bold=True)

    fig.panel(38, 550, 900, 360, "(d)", "Normal target count distribution", COL["white"])
    dist = eval_summary["normal_hit_count_distribution"]
    labels, vals, colors = [], [], []
    for hits in range(1, 5):
        labels += [f"Y{hits}", f"B{hits}"]
        vals += [dist["yellow"][str(hits)], dist["blue"][str(hits)]]
        colors += [COL["yellow"], COL["blue"]]
    bar_chart(fig, 92, 635, 740, 185, labels, vals, colors, "episode share", 1.0)

    fig.panel(980, 550, 900, 360, "(e)", "Base success by cleared targets", COL["white"])
    base = eval_summary["base_success_by_hits"]
    labels, vals, colors = [], [], []
    for hits in range(1, 5):
        labels += [f"Y{hits}", f"B{hits}"]
        vals += [base["yellow"][str(hits)]["success_rate"], base["blue"][str(hits)]["success_rate"]]
        colors += [COL["yellow"], COL["blue"]]
    bar_chart(fig, 1035, 635, 740, 185, labels, vals, colors, "success rate", 1.0)
    fig.rect(74, 956, 1772, 54, "Evaluation emphasizes win balance, target diversity, legal base timing, push-box behavior and zero penetration.", "#FFF7ED", "#F59E0B", fs=22, bold=True)
    return fig


def fig04_ablation(eval_summary) -> Figure:
    fig = Figure(
        "fig04_ablation_and_safety",
        "Ablation Design and Safety Diagnostics",
        "Component-wise evaluation plan plus measured safety and behavior indicators from the selected run.",
    )
    fig.panel(38, 120, 770, 420, "(a)", "Ablation matrix", COL["white"])
    headers = ["Object\nstate", "World\nmodel", "Flow\nactor", "Expert\nresidual", "Action\nshield"]
    rows = [
        ("Full method", [1, 1, 1, 1, 1]),
        ("w/o world model", [1, 0, 1, 1, 1]),
        ("Gaussian actor", [1, 1, 0, 1, 1]),
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
    fig.wrapped(76, 492, 650, "No fabricated ablation scores: this panel defines the exact component toggles for reproducible follow-up experiments.", 13, False, COL["muted"])

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
        "ROS2 + IsaacLab + RL Closed Loop and Replay Evidence",
        "Deployment-facing diagram: learned high-level policy remains inside ROS2/Nav2 and strict replay contracts.",
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

    fig.panel(1320, 120, 560, 780, "(c)", "Published evidence", COL["white"])
    for i, (txt, col) in enumerate([
        ("final replay top view", COL["green"]),
        ("yellow first-person replay", COL["yellow"]),
        ("blue first-person replay", COL["blue"]),
        ("checkpoint + export manifest", COL["violet"]),
        ("contract JSON/CSV + figures", COL["orange"]),
    ]):
        fig.rect(1375, 210 + i * 105, 400, 55, txt, "#FFFFFF", col, fs=15, bold=True)
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
