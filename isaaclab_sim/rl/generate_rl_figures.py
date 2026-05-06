from __future__ import annotations

import csv
import html
import json
import math
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "docs" / "figures" / "rl"
PORTFOLIO_DIR = ROOT / "docs" / "figures" / "portfolio"
README_ASSETS = ROOT / "assets" / "readme"

FULL = ROOT / "docs" / "rl_data" / "mappo_selfplay_full_gpu"
SHIELD = ROOT / "docs" / "rl_data" / "drshield_recessed_base_shared"
BOXMOVE = ROOT / "docs" / "rl_data" / "recessed_base_boxmove_final_shared"
MOTION = ROOT / "docs" / "rl_data" / "ros2_motion_drift_live" / "motion_drift_live_log.csv"
GEOMETRY = ROOT / "docs" / "rl_data" / "rule_geometry_audit.csv"


PALETTE = {
    "ink": "#1B1F23",
    "muted": "#5B6472",
    "line": "#C9D1D9",
    "grid": "#E6EBF0",
    "panel": "#F7F9FB",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#E69F00",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#7A869A",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def read_csv(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, object] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def style() -> str:
    return f"""
    <style>
      text{{font-family:Inter,Arial,Helvetica,sans-serif}}
      .title{{font-size:34px;font-weight:700;fill:{PALETTE['ink']}}}
      .subtitle{{font-size:15px;font-weight:400;fill:{PALETTE['muted']}}}
      .label{{font-size:15px;font-weight:650;fill:{PALETTE['ink']}}}
      .body{{font-size:13px;font-weight:400;fill:{PALETTE['muted']}}}
      .small{{font-size:11px;font-weight:400;fill:{PALETTE['muted']}}}
      .tiny{{font-size:9px;font-weight:500;fill:{PALETTE['muted']}}}
      .panel{{fill:#fff;stroke:{PALETTE['line']};stroke-width:1.2}}
      .soft{{fill:{PALETTE['panel']};stroke:{PALETTE['line']};stroke-width:1}}
      .grid{{stroke:{PALETTE['grid']};stroke-width:1}}
      .axis{{stroke:{PALETTE['ink']};stroke-width:1.1}}
      .wire{{fill:none;stroke:{PALETTE['gray']};stroke-width:1.8;marker-end:url(#arrowGray)}}
      .wireBlue{{fill:none;stroke:{PALETTE['blue']};stroke-width:2;marker-end:url(#arrowBlue)}}
      .wireGreen{{fill:none;stroke:{PALETTE['green']};stroke-width:2;marker-end:url(#arrowGreen)}}
      .dash{{stroke-dasharray:6 6}}
    </style>
    """


def svg(width: int, height: int, title: str, subtitle: str, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <marker id="arrowGray" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0 L9 4.5 L0 9 Z" fill="{PALETTE['gray']}"/></marker>
    <marker id="arrowBlue" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0 L9 4.5 L0 9 Z" fill="{PALETTE['blue']}"/></marker>
    <marker id="arrowGreen" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0 L9 4.5 L0 9 Z" fill="{PALETTE['green']}"/></marker>
    {style()}
  </defs>
  <rect width="{width}" height="{height}" fill="#FFFFFF"/>
  <text class="title" x="64" y="58">{esc(title)}</text>
  <text class="subtitle" x="66" y="86">{esc(subtitle)}</text>
{body}
</svg>
"""


def write_svg(path: Path, width: int, height: int, title: str, subtitle: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = svg(width, height, title, subtitle, body)
    normalized = "\n".join(line.rstrip() for line in content.splitlines()) + "\n"
    path.write_text(normalized, encoding="utf-8")


def rect(x: float, y: float, w: float, h: float, cls: str = "panel", rx: float = 6, fill: str | None = None) -> str:
    fill_attr = f' fill="{fill}"' if fill else ""
    return f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}"{fill_attr}/>'


def label(x: float, y: float, text: str, cls: str = "label", anchor: str = "start") -> str:
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{esc(text)}</text>'


def multi_text(x: float, y: float, lines: list[str], dy: float = 22, cls: str = "body") -> str:
    return "\n".join(label(x, y + i * dy, line, cls) for i, line in enumerate(lines))


def arrow(x1: float, y1: float, x2: float, y2: float, cls: str = "wire") -> str:
    return f'<path class="{cls}" d="M{x1:.1f} {y1:.1f} L{x2:.1f} {y2:.1f}"/>'


def path_arrow(d: str, cls: str = "wire") -> str:
    return f'<path class="{cls}" d="{d}"/>'


def chart(
    x: float,
    y: float,
    w: float,
    h: float,
    rows: list[dict[str, object]],
    x_key: str,
    y_key: str,
    color: str,
    title: str,
    y_min: float | None = None,
    y_max: float | None = None,
    ticks: int = 4,
) -> str:
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row[y_key]) for row in rows]
    if not xs:
        return ""
    x0, x1 = min(xs), max(xs)
    y0 = min(ys) if y_min is None else y_min
    y1 = max(ys) if y_max is None else y_max
    if abs(x1 - x0) < 1e-12:
        x1 += 1
    if abs(y1 - y0) < 1e-12:
        y1 += 1

    def sx(v: float) -> float:
        return x + (v - x0) / (x1 - x0) * w

    def sy(v: float) -> float:
        return y + h - (v - y0) / (y1 - y0) * h

    grid = []
    for i in range(ticks + 1):
        gy = y + h * i / ticks
        grid.append(f'<line class="grid" x1="{x:.1f}" y1="{gy:.1f}" x2="{x+w:.1f}" y2="{gy:.1f}"/>')
    points = " ".join(f"{sx(float(row[x_key])):.1f},{sy(float(row[y_key])):.1f}" for row in rows)
    return "\n".join(
        [
            label(x, y - 20, title, "label"),
            *grid,
            f'<line class="axis" x1="{x:.1f}" y1="{y+h:.1f}" x2="{x+w:.1f}" y2="{y+h:.1f}"/>',
            f'<line class="axis" x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y+h:.1f}"/>',
            f'<polyline fill="none" stroke="{color}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" points="{points}"/>',
            f'<circle cx="{sx(xs[-1]):.1f}" cy="{sy(ys[-1]):.1f}" r="4.2" fill="{color}"/>',
            label(x, y + h + 20, f"{int(x0):,}", "tiny"),
            label(x + w, y + h + 20, f"{int(x1):,}", "tiny", "end"),
            label(x + 6, y + 12, f"{y1:.3g}", "tiny"),
            label(x + 6, y + h - 6, f"{y0:.3g}", "tiny"),
        ]
    )


def bar_group(
    x: float,
    y: float,
    w: float,
    h: float,
    labels: list[str],
    values: list[float],
    colors: list[str],
    max_value: float | None = None,
    value_fmt: str | None = None,
) -> str:
    max_v = max(values) if max_value is None else max_value
    max_v = max(max_v, 1e-9)
    band = w / len(values)
    parts = []
    for i, (name, value, color) in enumerate(zip(labels, values, colors)):
        bh = h * max(0.0, value) / max_v
        bx = x + i * band + band * 0.22
        bw = band * 0.56
        parts.append(f'<rect x="{bx:.1f}" y="{y+h-bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{color}"/>')
        parts.append(label(bx + bw / 2, y + h - bh - 8, format_value(value, value_fmt), "tiny", "middle"))
        parts.append(label(bx + bw / 2, y + h + 18, name, "tiny", "middle"))
    return "\n".join(parts)


def format_value(value: float, value_fmt: str | None = None) -> str:
    if value_fmt is not None:
        return value_fmt.format(value)
    if abs(value) >= 1000:
        return f"{value / 1000:.1f}k"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.2f}"


def fig_overview() -> Path:
    body = f"""
  <rect class="soft" x="52" y="122" width="1496" height="690" rx="10"/>
  <text class="label" x="92" y="168">End-to-end research stack</text>
  <g>
    {rect(92, 215, 330, 170)}
    {label(122, 255, "1. IsaacLab environments")}
    {multi_text(122, 294, ["two robots, rigid boxes", "targets, armor, blockers", "strict replay traces"])}
    {rect(92, 455, 330, 170)}
    {label(122, 495, "2. Multi-agent RL")}
    {multi_text(122, 534, ["MAPPO / CTDE training", "yellow and blue actors", "residual expert priors"])}
    {arrow(422, 300, 560, 300, "wireBlue")}
    {arrow(422, 540, 560, 540, "wireBlue")}
  </g>
  <g>
    {rect(560, 215, 390, 410)}
    {label(600, 255, "3. Rule-aware policy contract")}
    {multi_text(600, 300, ["opponent-target fire gate", "line-of-sight blockers", "0.80 s dwell requirement", "20-80 cm recessed-base range", "pushable obstacle state"])}
    <line x1="600" y1="455" x2="910" y2="455" stroke="{PALETTE['line']}" stroke-width="1"/>
    {label(600, 500, "outputs")}
    {multi_text(600, 538, ["target choice", "base-rush timing", "block / recover / fire gates"])}
    {arrow(950, 420, 1080, 420, "wireGreen")}
  </g>
  <g>
    {rect(1080, 215, 390, 410)}
    {label(1120, 255, "4. ROS2 / Nav2 deployment")}
    {multi_text(1120, 300, ["behavior state machine", "Nav2 goal execution", "AprilTag alignment", "EKF + lidar + IMU fusion", "shooter services"])}
    <line x1="1120" y1="455" x2="1430" y2="455" stroke="{PALETTE['line']}" stroke-width="1"/>
    {label(1120, 500, "evidence")}
    {multi_text(1120, 538, ["pytest rule checks", "64-episode evaluation", "three-view IsaacLab replay"])}
  </g>
  <path class="wire dash" d="M1280 625 C1280 720 250 720 250 625"/>
  <text class="small" x="620" y="735">sim-to-real feedback: drift logs, replay audits and rule geometry checks refine the training environment</text>
"""
    out = PORTFOLIO_DIR / "overview.svg"
    write_svg(out, 1600, 900, "Multi-Agent Robot RL System", "Publication-style summary of the IsaacLab + ROS2 + Sim2Real stack.", body)
    return out


def world_to_svg(xy: tuple[float, float], x: float, y: float, size: float) -> tuple[float, float]:
    return x + (xy[0] + 1.5) / 3.0 * size, y + (1.5 - xy[1]) / 3.0 * size


def fig_arena() -> Path:
    audit = read_csv(GEOMETRY)
    x, y, size = 96, 146, 700
    targets = []
    for row in audit:
        px, py = world_to_svg((float(row["x"]), float(row["y"])), x, y, size)
        color = PALETTE["blue"] if row["owner"] == "blue" else PALETTE["yellow"]
        radius = 10 if row["kind"] == "normal" else 15
        targets.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius}" fill="#fff" stroke="{color}" stroke-width="2.5"/>')
        yaw = math.radians(float(row["yaw_deg"]))
        tx = px + math.cos(yaw) * 32
        ty = py - math.sin(yaw) * 32
        targets.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="{color}" stroke-width="2"/>')
    body = f"""
  {rect(62, 122, 1476, 690, "soft", 10)}
  <rect x="{x}" y="{y}" width="{size}" height="{size}" fill="#FFFFFF" stroke="{PALETTE['ink']}" stroke-width="2.2"/>
  <line x1="{x}" y1="{y+size/2}" x2="{x+size}" y2="{y+size/2}" stroke="{PALETTE['grid']}" stroke-width="4"/>
  <line x1="{x+size/2}" y1="{y}" x2="{x+size/2}" y2="{y+size}" stroke="{PALETTE['grid']}" stroke-width="1.2" stroke-dasharray="6 6"/>
  <rect x="{x+size*0.05}" y="{y+size*0.05}" width="{size*0.18}" height="{size*0.18}" fill="#EAF4FB" stroke="{PALETTE['blue']}" stroke-width="2"/>
  <rect x="{x+size*0.77}" y="{y+size*0.77}" width="{size*0.18}" height="{size*0.18}" fill="#FFF5DA" stroke="{PALETTE['yellow']}" stroke-width="2"/>
  <rect x="{x+size*0.61}" y="{y+size*0.42}" width="{size*0.08}" height="{size*0.16}" fill="#FDEAD7" stroke="{PALETTE['red']}" stroke-width="2"/>
  <rect x="{x+size*0.31}" y="{y+size*0.42}" width="{size*0.08}" height="{size*0.16}" fill="#FDEAD7" stroke="{PALETTE['red']}" stroke-width="2"/>
  {"".join(targets)}
  {rect(880, 172, 520, 490)}
  {label(920, 220, "Rule geometry audit")}
  {multi_text(920, 270, ["10 targets detected: 8 normal + 2 base", "target faces oriented at 45 degrees", "base targets are recessed behind armor", "red boxes are pushable rigid bodies", "laser line of sight includes blockers"])}
  <line x1="920" y1="430" x2="1350" y2="430" stroke="{PALETTE['line']}"/>
  {label(920, 482, "Visual encoding")}
  <circle cx="940" cy="525" r="10" fill="#fff" stroke="{PALETTE['blue']}" stroke-width="2.5"/>{label(965, 530, "blue-owned target", "body")}
  <circle cx="940" cy="565" r="10" fill="#fff" stroke="{PALETTE['yellow']}" stroke-width="2.5"/>{label(965, 570, "yellow-owned target", "body")}
  <rect x="930" y="600" width="22" height="22" fill="#FDEAD7" stroke="{PALETTE['red']}" stroke-width="2"/>{label(965, 618, "pushable obstacle", "body")}
"""
    out = PORTFOLIO_DIR / "arena_rule_scene.svg"
    write_svg(out, 1600, 900, "Rule-Checked Arena Geometry", "All target positions, target yaw probes, base blockers and pushable obstacles are audited before training.", body)
    return out


def fig_robot_sensor_layout() -> Path:
    body = f"""
  {rect(64, 126, 1468, 650, "soft", 10)}
  <g transform="translate(320 210)">
    <rect x="210" y="120" width="390" height="250" rx="36" fill="#FFFFFF" stroke="{PALETTE['ink']}" stroke-width="2.5"/>
    <rect x="245" y="154" width="320" height="182" rx="24" fill="#F7F9FB" stroke="{PALETTE['line']}" stroke-width="1.5"/>
    <circle cx="290" cy="370" r="46" fill="#FFFFFF" stroke="{PALETTE['ink']}" stroke-width="2"/>
    <circle cx="520" cy="370" r="46" fill="#FFFFFF" stroke="{PALETTE['ink']}" stroke-width="2"/>
    <rect x="385" y="86" width="44" height="70" rx="8" fill="{PALETTE['blue']}"/>
    <line x1="407" y1="86" x2="407" y2="20" stroke="{PALETTE['blue']}" stroke-width="2"/>
    <path d="M280 178 L122 80" class="wireBlue"/>
    <path d="M510 178 L724 86" class="wireBlue"/>
    <path d="M565 260 L785 260" class="wireGreen"/>
    <path d="M290 320 L118 476" class="wire"/>
    <path d="M520 320 L726 476" class="wire"/>
    <path d="M405 210 L405 32" stroke="{PALETTE['sky']}" stroke-width="2" stroke-dasharray="5 5"/>
  </g>
  {rect(95, 210, 260, 108)}
  {label(125, 250, "RGB camera + AprilTag")}
  {multi_text(125, 282, ["visual target detection", "front alignment"])}
  {rect(1110, 210, 280, 108)}
  {label(1140, 250, "2D lidar / costmap")}
  {multi_text(1140, 282, ["clearance", "obstacle avoidance"])}
  {rect(1180, 420, 260, 108)}
  {label(1210, 460, "ToF + bumper")}
  {multi_text(1210, 492, ["contact state", "pushable boxes"])}
  {rect(92, 600, 300, 108)}
  {label(122, 640, "Wheel odom + IMU")}
  {multi_text(122, 672, ["motion consistency", "drift-risk estimate"])}
  {rect(1088, 600, 340, 108)}
  {label(1118, 640, "Fixed laser module")}
  {multi_text(1118, 672, ["range-gated firing", "0.80 s dwell"])}
"""
    out = PORTFOLIO_DIR / "robot_sensor_layout.svg"
    write_svg(out, 1600, 900, "Robot Sensor and Actuation Layout", "Deployment observes only real ROS2 sensor signals; no privileged simulator state is required.", body)
    return out


def fig_ros2_runtime_graph() -> Path:
    nodes = [
        ("camera", 110, 210, "Camera\n/image_raw"),
        ("vision", 360, 210, "rcvrl_vision\n/target_detection"),
        ("behavior", 680, 320, "rcvrl_behavior\ncompetition FSM"),
        ("nav", 1030, 210, "Nav2\nNavigateToPose"),
        ("shooter", 1030, 460, "rcvrl_shooter\n/fire service"),
        ("fusion", 360, 460, "robot_localization\nEKF"),
        ("motion", 110, 460, "wheel/IMU/lidar\nsensor stack"),
        ("logger", 680, 610, "rcvrl_motion\ntelemetry recorder"),
    ]
    body = f"{rect(64, 128, 1468, 640, 'soft', 10)}"
    for _key, x, y, text_block in nodes:
        lines = text_block.split("\n")
        body += rect(x, y, 230, 118)
        body += label(x + 22, y + 43, lines[0])
        body += label(x + 22, y + 76, lines[1], "body")
    body += "\n".join(
        [
            arrow(340, 269, 360, 269, "wireBlue"),
            arrow(590, 269, 680, 335, "wireBlue"),
            arrow(910, 335, 1030, 269, "wireGreen"),
            arrow(910, 390, 1030, 515, "wireGreen"),
            arrow(340, 519, 360, 519, "wire"),
            arrow(590, 519, 680, 390, "wire"),
            arrow(590, 519, 680, 645, "wire"),
            arrow(910, 645, 1030, 560, "wire"),
            path_arrow("M1180 328 C1180 720 720 760 300 580", "wire dash"),
        ]
    )
    body += multi_text(108, 720, ["Runtime contract: perception, localization, navigation and shooter services are explicit ROS2 interfaces for Sim2Real testing."], 20, "body")
    out = PORTFOLIO_DIR / "ros2_runtime_graph.svg"
    write_svg(out, 1600, 900, "ROS2 Runtime Graph", "A small, inspectable deployment graph links perception, EKF, Nav2, behavior logic and shooter services.", body)
    return out


def fig_hierarchical_policy() -> Path:
    body = f"""
  {rect(64, 126, 1468, 650, "soft", 10)}
  {rect(120, 230, 280, 340)}
  {label(150, 270, "Local observation")}
  {multi_text(150, 315, ["pose and heading", "opponent bearing", "armor / score / time", "target visibility", "sensor-fusion state"])}
  {arrow(400, 400, 540, 400, "wireBlue")}
  {rect(540, 190, 360, 420)}
  {label(580, 235, "Decentralized actor")}
  <circle cx="630" cy="320" r="14" fill="#fff" stroke="{PALETTE['blue']}" stroke-width="2"/>
  <circle cx="715" cy="285" r="14" fill="#fff" stroke="{PALETTE['blue']}" stroke-width="2"/>
  <circle cx="715" cy="355" r="14" fill="#fff" stroke="{PALETTE['blue']}" stroke-width="2"/>
  <circle cx="800" cy="320" r="14" fill="#fff" stroke="{PALETTE['blue']}" stroke-width="2"/>
  <path d="M644 320 L701 285 M644 320 L701 355 M729 285 L786 320 M729 355 L786 320" stroke="{PALETTE['line']}" stroke-width="1.7"/>
  {multi_text(580, 440, ["6-D tactical output", "target selector", "base-rush / block / recover / fire"])}
  {arrow(900, 400, 1040, 400, "wireBlue")}
  {rect(1040, 230, 350, 340)}
  {label(1080, 270, "Rule and safety shield")}
  {multi_text(1080, 315, ["opponent target only", "line-of-sight blockers", "dwell and range gate", "contact hull guard"])}
  <path d="M1198 435 l56 24 v70 c0 40 -30 68 -56 82 c-26 -14 -56 -42 -56 -82 v-70 z" fill="#fff" stroke="{PALETTE['red']}" stroke-width="2"/>
  {arrow(1215, 570, 1215, 680, "wireGreen")}
  {label(1120, 715, "ROS2 behavior / Nav2 / shooter execution", "label")}
  <path class="wire dash" d="M720 190 C760 105 980 105 1160 230"/>
  {label(830, 145, "central critic is used only during training", "body")}
"""
    out = FIG_DIR / "rl_hierarchical_policy.svg"
    write_svg(out, 1600, 900, "Hierarchical MAPPO Strategy", "Local actors choose tactics, safety gates enforce rules, and ROS2 controllers execute low-level motion.", body)
    return out


def fig_selfplay_training() -> Path:
    body = f"""
  {rect(64, 126, 1468, 650, "soft", 10)}
  {rect(100, 220, 340, 380)}
  {label(130, 260, "Vectorized rule environments")}
  {multi_text(130, 305, ["32 parallel matches", "fast Python rule model", "domain randomization", "legal-shot masking"])}
  <g transform="translate(130 430)">
    <rect x="0" y="0" width="52" height="52" fill="#fff" stroke="{PALETTE['line']}"/>
    <rect x="66" y="0" width="52" height="52" fill="#fff" stroke="{PALETTE['line']}"/>
    <rect x="132" y="0" width="52" height="52" fill="#fff" stroke="{PALETTE['line']}"/>
    <text class="small" x="210" y="32">...</text>
  </g>
  {arrow(440, 410, 590, 410, "wireBlue")}
  {rect(590, 220, 360, 380)}
  {label(630, 260, "MAPPO update")}
  {multi_text(630, 305, ["centralized critic", "decentralized actors", "GAE advantage estimate", "clipped PPO objective"])}
  <rect x="640" y="455" width="260" height="70" rx="6" fill="#fff" stroke="{PALETTE['blue']}"/>
  {label(670, 498, "actor + critic gradients", "body")}
  {arrow(950, 410, 1100, 410, "wireGreen")}
  {rect(1100, 220, 340, 380)}
  {label(1140, 260, "Audited replay")}
  {multi_text(1140, 305, ["strict geometry checks", "trajectory trace", "three-view IsaacLab videos", "rule-contract reports"])}
  <path class="wire dash" d="M1270 600 C1270 720 270 720 270 600"/>
  {label(610, 725, "failed audits feed back into geometry, reward and action-shield revisions", "body")}
"""
    out = FIG_DIR / "rl_selfplay_training.svg"
    write_svg(out, 1600, 900, "Parallel MAPPO Self-Play Training", "A reproducible loop connects vectorized training, checkpoint selection, strict replay and geometry audits.", body)
    return out


def fig_sim2real_pipeline() -> Path:
    stages = [
        ("IsaacLab scene", "rigid bodies, targets, sensors"),
        ("Rule env", "fast MAPPO rollout"),
        ("Policy export", "TorchScript / ONNX actor"),
        ("ROS2 runtime", "Nav2, vision, shooter"),
        ("Field logs", "bags, drift, replay audit"),
    ]
    body = f"{rect(64, 126, 1468, 650, 'soft', 10)}"
    x0 = 115
    for i, (name, desc) in enumerate(stages):
        x = x0 + i * 285
        body += rect(x, 295, 220, 150)
        body += label(x + 24, 345, name)
        body += label(x + 24, 382, desc, "body")
        if i < len(stages) - 1:
            body += arrow(x + 220, 370, x + 280, 370, "wireBlue")
    body += path_arrow("M1260 445 C1260 610 250 610 250 445", "wire dash")
    body += label(595, 650, "calibration feedback: sensor noise, contact margins and action shields", "body")
    body += rect(430, 170, 730, 68)
    body += label(470, 214, "No privileged simulator state crosses into deployment", "label")
    out = FIG_DIR / "rl_sim2real_pipeline.svg"
    write_svg(out, 1600, 900, "IsaacLab-to-ROS2 Sim2Real Pipeline", "Training, export, ROS2 deployment and evidence collection are separated by explicit contracts.", body)
    return out


def fig_training_curve(
    rows: list[dict[str, object]],
    summary: dict[str, object],
    out: Path,
    title: str,
    subtitle: str,
) -> Path:
    final = summary["final_curve_row"]
    body = f"""
  {rect(58, 120, 1484, 670, "soft", 10)}
  {chart(110, 210, 620, 190, rows, "global_step", "mean_reward", PALETTE["blue"], "A. Mean reward")}
  {chart(870, 210, 560, 190, rows, "global_step", "done_rate", PALETTE["green"], "B. Terminal rate", 0.0, max(0.01, max(float(r["done_rate"]) for r in rows)))}
  {chart(110, 535, 620, 160, rows, "global_step", "explained_variance", PALETTE["purple"], "C. Critic explained variance")}
  {chart(870, 535, 560, 160, rows, "global_step", "approx_kl", PALETTE["red"], "D. Approximate KL")}
  {rect(110, 735, 1320, 38, "panel", 4)}
  {label(135, 760, f"final reward {float(final['mean_reward']):.4f} | speed {float(final['steps_per_second']):.0f} steps/s | device {summary.get('cuda_device', summary.get('device'))}", "body")}
"""
    write_svg(out, 1600, 850, title, subtitle, body)
    return out


def fig_strategy_metrics(det: dict[str, object], stoch: dict[str, object], baseline: dict[str, object]) -> Path:
    names = ["det", "stoch", "script"]
    colors = [PALETTE["blue"], PALETTE["purple"], PALETTE["green"]]
    summaries = [det["summary"], stoch["summary"], baseline["summary"]]
    metric_keys = [
        ("normal_hits_per_episode", "Normal hits"),
        ("base_hit_wins_per_episode", "Base wins"),
        ("robot_contacts_per_episode", "Contacts"),
        ("block_steps_per_episode", "Block steps"),
        ("base_rush_steps_per_episode", "Rush steps"),
    ]
    body = f"{rect(58, 120, 1484, 670, 'soft', 10)}"
    start_x = 112
    for i, (key, name) in enumerate(metric_keys):
        x = start_x + i * 285
        values = [float(s.get(key, 0.0)) for s in summaries]
        body += label(x, 190, name, "label")
        body += bar_group(x, 240, 220, 320, names, values, colors)
    body += rect(120, 650, 1260, 80)
    body += label(150, 688, f"Stochastic MAPPO: blue win {float(stoch['summary']['blue_win_rate'])*100:.1f}% | draw/timeout {float(stoch['summary']['draw_or_timeout_rate'])*100:.1f}% | own-target penalties {float(stoch['summary']['own_target_penalties_per_episode']):.1f}", "body")
    out = FIG_DIR / "rl_strategy_event_metrics.svg"
    write_svg(out, 1600, 850, "Learned Strategy Event Metrics", "Evaluation over 64 episodes, comparing deterministic MAPPO, stochastic MAPPO and scripted rules baseline.", body)
    return out


def fig_eval_metrics(payload: dict[str, object], strict: dict[str, object], out: Path, title: str, subtitle: str) -> Path:
    summary = payload["summary"]
    values = [
        float(summary.get("yellow_win_rate", 0.0)),
        float(summary.get("blue_win_rate", 0.0)),
        float(summary.get("draw_or_timeout_rate", 0.0)),
    ]
    labels = ["Yellow", "Blue", "Draw"]
    body = f"""
  {rect(58, 120, 1484, 670, "soft", 10)}
  {label(110, 190, "A. Win balance")}
  {bar_group(110, 245, 520, 330, labels, values, [PALETTE["yellow"], PALETTE["blue"], PALETTE["gray"]], 1.0, "{:.1%}")}
  {label(790, 190, "B. Contract metrics")}
  {rect(790, 245, 610, 330)}
  {multi_text(830, 300, [
      f"normal hits / episode: {float(summary.get('normal_hits_per_episode', 0.0)):.3f}",
      f"base wins / episode: {float(summary.get('base_hit_wins_per_episode', summary.get('base_wins_per_episode', 0.0))):.3f}",
      f"own-target penalties / episode: {float(summary.get('own_target_penalties_per_episode', 0.0)):.3f}",
      f"hard replay violations: {int(strict['summary'].get('hard_violations', 0))}",
      f"strict replay warnings: {int(strict['summary'].get('warnings', 0))}",
  ], 42, "body")}
  {rect(160, 675, 1180, 62)}
  {label(190, 715, "Audited figures use committed CSV/JSON snapshots under docs/rl_data, not local runtime output.", "body")}
"""
    write_svg(out, 1600, 850, title, subtitle, body)
    return out


def fig_policy_trace(stoch: dict[str, object]) -> Path:
    episode = next((ep for ep in stoch["episodes"] if ep.get("trace") and ep.get("winner") in ("yellow", "blue")), stoch["episodes"][0])
    trace = episode.get("trace", [])
    x, y, size = 88, 138, 640
    yellow = [world_to_svg((float(t["yellow_pose"][0]), float(t["yellow_pose"][1])), x, y, size) for t in trace]
    blue = [world_to_svg((float(t["blue_pose"][0]), float(t["blue_pose"][1])), x, y, size) for t in trace]
    def poly(points: list[tuple[float, float]]) -> str:
        return " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    events = []
    for item in trace:
        for team in ("yellow", "blue"):
            info = item.get(f"{team}_info", {})
            if "hit" in info or "winner" in info:
                events.append((float(item["elapsed_s"]), team, info.get("hit") or info.get("winner")))
    body = f"""
  {rect(58, 120, 1484, 670, "soft", 10)}
  <rect x="{x}" y="{y}" width="{size}" height="{size}" fill="#fff" stroke="{PALETTE['ink']}" stroke-width="2"/>
  <line x1="{x}" y1="{y+size/2}" x2="{x+size}" y2="{y+size/2}" stroke="{PALETTE['grid']}" stroke-width="4"/>
  <line x1="{x+size/2}" y1="{y}" x2="{x+size/2}" y2="{y+size}" stroke="{PALETTE['grid']}" stroke-dasharray="5 5"/>
  <polyline fill="none" stroke="{PALETTE['yellow']}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{poly(yellow)}"/>
  <polyline fill="none" stroke="{PALETTE['blue']}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{poly(blue)}"/>
  <circle cx="{yellow[0][0]:.1f}" cy="{yellow[0][1]:.1f}" r="7" fill="{PALETTE['yellow']}"/>
  <circle cx="{blue[0][0]:.1f}" cy="{blue[0][1]:.1f}" r="7" fill="{PALETTE['blue']}"/>
  {rect(830, 160, 620, 540)}
  {label(870, 208, "Event timeline")}
"""
    for i, (time_s, team, event) in enumerate(events[:11]):
        cy = 260 + i * 38
        color = PALETTE["yellow"] if team == "yellow" else PALETTE["blue"]
        body += f'<circle cx="882" cy="{cy-5}" r="6" fill="{color}"/>'
        body += label(902, cy, f"{time_s:05.1f}s {team}: {event}", "body")
    body += rect(870, 630, 510, 44)
    body += label(895, 658, f"winner: {episode['winner']} | score Y {episode['scores']['yellow']} / B {episode['scores']['blue']}", "body")
    out = FIG_DIR / "rl_policy_episode_trace.svg"
    write_svg(out, 1600, 850, "Policy Rollout Trace", "One stochastic episode projected onto the audited arena coordinate frame with event timing.", body)
    return out


def fig_tactical_contract(summary: dict[str, object], stoch: dict[str, object]) -> Path:
    body = f"""
  {rect(58, 120, 1484, 670, "soft", 10)}
  {rect(120, 250, 300, 250)}
  {label(150, 290, f"Local observation [{summary['obs_dim']}]")}
  {multi_text(150, 330, ["pose, target flags", "sensor-fusion state", "opponent relation"])}
  {arrow(420, 375, 560, 375, "wireBlue")}
  {rect(560, 220, 360, 310)}
  {label(600, 265, "Dual tactical actors")}
  {multi_text(600, 310, ["yellow_expert residual", "blue_expert residual", "6-D action in [-1, 1]"])}
  {arrow(920, 375, 1060, 375, "wireBlue")}
  {rect(1060, 250, 330, 250)}
  {label(1090, 290, "ROS2 behavior contract")}
  {multi_text(1090, 330, ["Nav2 target pose", "AprilTag align", "shooter dwell gate"])}
  <path class="wire dash" d="M740 220 C790 135 980 135 1130 250"/>
  {label(835, 170, f"central critic [{summary['central_obs_dim']}] is training-only", "body")}
  {rect(260, 610, 1040, 64)}
  {label(295, 650, f"Stochastic eval base wins / episode: {float(stoch['summary']['base_hit_wins_per_episode']):.3f}; own-target penalties / episode: {float(stoch['summary']['own_target_penalties_per_episode']):.1f}", "body")}
"""
    out = FIG_DIR / "rl_tactical_contract.svg"
    write_svg(out, 1600, 850, "Tactical Deployment Contract", "The actor selects high-level tactics; safety, navigation and shooting remain explicit deployable interfaces.", body)
    return out


def fig_sensorfusion_architecture() -> Path:
    body = f"""
  {rect(58, 120, 1484, 670, "soft", 10)}
  {rect(110, 230, 320, 320)}
  {label(145, 275, "Physical sensors")}
  {multi_text(145, 320, ["wheel odom", "IMU yaw rate", "2D lidar", "camera / AprilTag", "ToF + bumper"])}
  {arrow(430, 390, 570, 390, "wireBlue")}
  {rect(570, 210, 360, 360)}
  {label(610, 255, "Fusion features")}
  {multi_text(610, 302, ["motion consistency", "scan clearance", "front contact", "target visibility", "EKF confidence"])}
  {arrow(930, 390, 1070, 390, "wireGreen")}
  {rect(1070, 230, 340, 320)}
  {label(1110, 275, "Policy observation")}
  {multi_text(1110, 320, ["46-D local vector", "no privileged state", "domain-randomized noise", "action shield records"])}
  {rect(265, 635, 980, 56)}
  {label(300, 670, "Design goal: same fields are observable in simulation, ROS2 dry-run and real-robot logging.", "body")}
"""
    out = FIG_DIR / "ros2_isaaclab_sensorfusion_architecture.svg"
    write_svg(out, 1600, 850, "ROS2 + IsaacLab Sensor-Fusion Interface", "A compact observation contract keeps the learned policy deployable outside the simulator.", body)
    return out


def fig_motion_drift() -> Path:
    rows = read_csv(MOTION)
    if len(rows) > 240:
        stride = max(1, len(rows) // 240)
        rows = rows[::stride]
    t0 = float(rows[0]["time_s"])
    for row in rows:
        row["t_rel"] = float(row["time_s"]) - t0
    body = f"""
  {rect(58, 120, 1484, 670, "soft", 10)}
  {chart(110, 220, 620, 190, rows, "t_rel", "odom_xy_error_m", PALETTE["blue"], "A. Odom XY error (m)")}
  {chart(870, 220, 560, 190, rows, "t_rel", "front_scan_min_m", PALETTE["green"], "B. Front scan min (m)")}
  {chart(110, 545, 620, 160, rows, "t_rel", "drift_risk", PALETTE["red"], "C. Drift-risk score")}
  {chart(870, 545, 560, 160, rows, "t_rel", "cmd_linear_x", PALETTE["purple"], "D. Commanded linear velocity")}
"""
    out = FIG_DIR / "ros2_motion_drift_live.svg"
    write_svg(out, 1600, 850, "ROS2 Motion Drift Telemetry", "Live topic collection visualizes drift risk, scan clearance and commanded motion for Sim2Real debugging.", body)
    return out


def generate_all_svgs() -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    README_ASSETS.mkdir(parents=True, exist_ok=True)

    full_curve = read_csv(FULL / "training_curve.csv")
    full_summary = read_json(FULL / "training_summary.json")
    full_det = read_json(FULL / "mappo_full_gpu_eval.json")
    full_stoch = read_json(FULL / "mappo_full_gpu_eval_stochastic.json")
    full_baseline = read_json(FULL / "scripted_rules_baseline_eval.json")

    shield_curve = read_csv(SHIELD / "training_curve.csv")
    shield_summary = read_json(SHIELD / "training_summary.json")
    shield_eval = read_json(SHIELD / "eval_seed3100_stochastic.json")
    shield_strict = read_json(SHIELD / "strict_replay_summary.json")

    box_curve = read_csv(BOXMOVE / "training_curve.csv")
    box_summary = read_json(BOXMOVE / "training_summary.json")
    box_eval = read_json(BOXMOVE / "eval_seed2600_stochastic.json")
    box_strict = read_json(BOXMOVE / "strict_replay_summary.json")

    paths = [
        fig_overview(),
        fig_arena(),
        fig_robot_sensor_layout(),
        fig_ros2_runtime_graph(),
        fig_hierarchical_policy(),
        fig_selfplay_training(),
        fig_sim2real_pipeline(),
        fig_training_curve(full_curve, full_summary, FIG_DIR / "rl_training_curve_gpu.svg", "GPU MAPPO Training Curve", "Committed training snapshot: 507,904 agent steps on RTX 4090."),
        fig_strategy_metrics(full_det, full_stoch, full_baseline),
        fig_policy_trace(full_stoch),
        fig_tactical_contract(full_summary, full_stoch),
        fig_training_curve(shield_curve, shield_summary, FIG_DIR / "rl_sensorfusion_training_curve.svg", "Sensor-Fusion MAPPO Training", "Residual expert with domain randomization and action shielding; observation dim 46."),
        fig_eval_metrics(shield_eval, shield_strict, FIG_DIR / "rl_sensorfusion_eval_metrics.svg", "Sensor-Fusion Evaluation Metrics", "64 stochastic evaluation episodes plus strict replay legality audit."),
        fig_training_curve(box_curve, box_summary, FIG_DIR / "rl_offaxis_base_training_curve.svg", "Recessed-Base Training Curve", "Archived moving-box, recessed-base training snapshot used for replay documentation."),
        fig_eval_metrics(box_eval, box_strict, FIG_DIR / "rl_offaxis_base_eval_metrics.svg", "Recessed-Base Evaluation Metrics", "Evaluation snapshot for moving boxes and recessed-base target rules."),
        fig_sensorfusion_architecture(),
        fig_motion_drift(),
    ]

    manifest = {
        "style": "publication_vector_v2",
        "palette": "Okabe-Ito inspired, colorblind-safe",
        "figures": [str(path.relative_to(ROOT)).replace("\\", "/") for path in paths],
        "source_data": [
            "docs/rl_data/mappo_selfplay_full_gpu/",
            "docs/rl_data/drshield_recessed_base_shared/",
            "docs/rl_data/recessed_base_boxmove_final_shared/",
            "docs/rl_data/ros2_motion_drift_live/motion_drift_live_log.csv",
            "docs/rl_data/rule_geometry_audit.csv",
        ],
    }
    (FIG_DIR / "rl_result_figures_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (PORTFOLIO_DIR / "publication_figures_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths


def export_png(svg_path: Path, png_path: Path, width: int = 1600) -> bool:
    inkscape = shutil.which("inkscape") or shutil.which("inkscape.com")
    if not inkscape:
        return False
    png_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            inkscape,
            str(svg_path),
            "--export-type=png",
            f"--export-filename={png_path}",
            f"--export-width={width}",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return True


def export_readme_pngs() -> list[str]:
    mapping = {
        PORTFOLIO_DIR / "overview.svg": README_ASSETS / "overview.png",
        PORTFOLIO_DIR / "arena_rule_scene.svg": README_ASSETS / "arena_rule_scene.png",
        PORTFOLIO_DIR / "robot_sensor_layout.svg": README_ASSETS / "robot_sensor_layout.png",
        PORTFOLIO_DIR / "ros2_runtime_graph.svg": README_ASSETS / "ros2_runtime_graph.png",
        FIG_DIR / "rl_hierarchical_policy.svg": README_ASSETS / "rl_hierarchical_policy.png",
        FIG_DIR / "rl_selfplay_training.svg": README_ASSETS / "rl_selfplay_training.png",
        FIG_DIR / "rl_sim2real_pipeline.svg": README_ASSETS / "rl_sim2real_pipeline.png",
    }
    exported = []
    for src, dst in mapping.items():
        if export_png(src, dst):
            exported.append(str(dst.relative_to(ROOT)).replace("\\", "/"))
    return exported


def main() -> None:
    paths = generate_all_svgs()
    exported = export_readme_pngs()
    result = {
        "svg_count": len(paths),
        "exported_pngs": exported,
        "figure_dir": str(FIG_DIR.relative_to(ROOT)).replace("\\", "/"),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
