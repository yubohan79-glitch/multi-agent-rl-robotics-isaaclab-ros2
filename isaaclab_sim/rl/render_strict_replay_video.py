from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robocup_visionrl_gym_env import (
    BLUE_BASE_TARGET_YAW,
    BLUE_BASE_TARGET_XY,
    NORTH_MIDDLE_TARGET_X,
    SIDE_GATE_TARGET_Y,
    SOUTH_MIDDLE_TARGET_X,
    TARGET_WALL_INSET,
    YELLOW_BASE_TARGET_YAW,
    YELLOW_BASE_TARGET_XY,
    inward_45deg_target_yaws,
)


ROOT = Path(__file__).resolve().parents[2]
TARGET_EDGE = 1.50 - TARGET_WALL_INSET
TARGET_YAWS = inward_45deg_target_yaws()

TARGETS = {
    "T01_NorthMiddle": {"xy": (NORTH_MIDDLE_TARGET_X, TARGET_EDGE), "yaw": TARGET_YAWS["T01_NorthMiddle"], "owner": "blue", "kind": "normal"},
    "T02_NorthEast": {"xy": (TARGET_EDGE, TARGET_EDGE), "yaw": TARGET_YAWS["T02_NorthEast"], "owner": "blue", "kind": "normal"},
    "T03_WestAboveGate": {"xy": (-TARGET_EDGE, SIDE_GATE_TARGET_Y), "yaw": TARGET_YAWS["T03_WestAboveGate"], "owner": "blue", "kind": "normal"},
    "T04_WestBelowGate": {"xy": (-TARGET_EDGE, -SIDE_GATE_TARGET_Y), "yaw": TARGET_YAWS["T04_WestBelowGate"], "owner": "yellow", "kind": "normal"},
    "T05_EastAboveGate": {"xy": (TARGET_EDGE, SIDE_GATE_TARGET_Y), "yaw": TARGET_YAWS["T05_EastAboveGate"], "owner": "blue", "kind": "normal"},
    "T06_EastBelowGate": {"xy": (TARGET_EDGE, -SIDE_GATE_TARGET_Y), "yaw": TARGET_YAWS["T06_EastBelowGate"], "owner": "yellow", "kind": "normal"},
    "T07_SouthWest": {"xy": (-TARGET_EDGE, -TARGET_EDGE), "yaw": TARGET_YAWS["T07_SouthWest"], "owner": "yellow", "kind": "normal"},
    "T08_SouthMiddle": {"xy": (SOUTH_MIDDLE_TARGET_X, -TARGET_EDGE), "yaw": TARGET_YAWS["T08_SouthMiddle"], "owner": "yellow", "kind": "normal"},
    "BlueBaseTarget": {"xy": tuple(float(v) for v in BLUE_BASE_TARGET_XY), "yaw": BLUE_BASE_TARGET_YAW, "owner": "blue", "kind": "base"},
    "YellowBaseTarget": {"xy": tuple(float(v) for v in YELLOW_BASE_TARGET_XY), "yaw": YELLOW_BASE_TARGET_YAW, "owner": "yellow", "kind": "base"},
}

STATIC_BLOCKERS = [
    ((-1.52, 0.0), (0.04, 3.08)),
    ((1.52, 0.0), (0.04, 3.08)),
    ((0.0, -1.52), (3.08, 0.04)),
    ((0.0, 1.52), (3.08, 0.04)),
    ((-1.00, 0.0), (1.00, 0.04)),
    ((1.00, 0.0), (1.00, 0.04)),
    ((0.00, 1.25), (0.04, 0.50)),
    ((0.00, -1.25), (0.04, 0.50)),
]

BASE_ARMOR_BLOCKERS = {
    "blue": [
        ((-1.025, 1.375), (0.050, 0.250)),
        ((-1.375, 1.025), (0.250, 0.050)),
        ((-1.025, 1.125), (0.050, 0.250)),
        ((-1.125, 1.025), (0.250, 0.050)),
    ],
    "yellow": [
        ((1.025, -1.375), (0.050, 0.250)),
        ((1.375, -1.025), (0.250, 0.050)),
        ((1.025, -1.125), (0.050, 0.250)),
        ((1.125, -1.025), (0.250, 0.050)),
    ],
}

PUSHABLE_BOX_COLUMNS = {
    "box_ne": ("box_ne_x", "box_ne_y"),
    "box_sw": ("box_sw_x", "box_sw_y"),
}


def parse_int_field(row: dict[str, object], key: str, default: int) -> int:
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return int(float(value))


def active_armor_for_row(row: dict[str, dict[str, object]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    yellow_row = row["yellow"]
    armor_counts = {
        "yellow": parse_int_field(yellow_row, "armor_yellow", 4),
        "blue": parse_int_field(yellow_row, "armor_blue", 4),
    }
    blockers: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for team, specs in BASE_ARMOR_BLOCKERS.items():
        remaining = max(0, min(4, armor_counts[team]))
        blockers.extend(specs[4 - remaining :])
    return blockers


def pushable_boxes_for_row(row: dict[str, dict[str, object]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    yellow_row = row["yellow"]
    boxes: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for x_key, y_key in PUSHABLE_BOX_COLUMNS.values():
        try:
            center = (float(yellow_row[x_key]), float(yellow_row[y_key]))
        except (KeyError, TypeError, ValueError):
            continue
        boxes.append((center, (0.30, 0.30)))
    return boxes

COLORS = {
    "ink": (17, 24, 39),
    "muted": (71, 85, 105),
    "grid": (226, 232, 240),
    "panel": (248, 250, 252),
    "wall": (148, 163, 184),
    "wall_dark": (100, 116, 139),
    "yellow": (242, 201, 76),
    "blue": (37, 99, 235),
    "red": (225, 29, 72),
    "green": (22, 163, 74),
    "orange": (249, 115, 22),
    "white": (255, 255, 255),
}


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def world_to_px(x: float, y: float, origin: tuple[int, int], size: int) -> tuple[int, int]:
    ox, oy = origin
    return int(ox + (x + 1.5) / 3.0 * size), int(oy + (1.5 - y) / 3.0 * size)


def rect_world(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    span: tuple[float, float],
    origin: tuple[int, int],
    size: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
):
    cx, cy = center
    sx, sy = span
    x0, y0 = world_to_px(cx - sx * 0.5, cy + sy * 0.5, origin, size)
    x1, y1 = world_to_px(cx + sx * 0.5, cy - sy * 0.5, origin, size)
    draw.rectangle([x0, y0, x1, y1], fill=fill, outline=outline)


def load_trace(path: Path, episode: int) -> dict[int, dict[str, dict[str, object]]]:
    by_step: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["episode"]) != episode:
                continue
            step = int(row["step"])
            team = row["team"]
            by_step[step][team] = row
    return dict(sorted(by_step.items()))


def load_events(path: Path, episode: int) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if int(item["episode"]) == episode:
                events.append(item)
    return events


def load_summary(path: Path, episode: int) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    for item in payload["episodes"]:
        if int(item["episode"]) == episode:
            return {"summary": payload["summary"], "episode": item}
    raise ValueError(f"Episode {episode} not found in {path}")


def event_target_names(event: dict[str, object]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for team in ("yellow", "blue"):
        info = event.get(f"{team}_info", {})
        if not isinstance(info, dict):
            continue
        if isinstance(info.get("hit"), str):
            out.append((team, str(info["hit"])))
        if isinstance(info.get("winner"), str):
            out.append((team, "YellowBaseTarget" if info["winner"] == "blue" else "BlueBaseTarget"))
    return out


def draw_polyline(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color: tuple[int, int, int], width: int):
    if len(points) > 1:
        draw.line(points, fill=color, width=width, joint="curve")


def draw_robot(
    draw: ImageDraw.ImageDraw,
    pose: tuple[float, float, float],
    team: str,
    origin: tuple[int, int],
    size: int,
):
    x, y, yaw = pose
    px, py = world_to_px(x, y, origin, size)
    color = COLORS["yellow"] if team == "yellow" else COLORS["blue"]
    radius = 18
    draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=color, outline=COLORS["ink"], width=2)
    hx = px + int(math.cos(yaw) * 28)
    hy = py - int(math.sin(yaw) * 28)
    draw.line([px, py, hx, hy], fill=COLORS["ink"], width=4)
    draw.ellipse([hx - 4, hy - 4, hx + 4, hy + 4], fill=COLORS["ink"])


def draw_target(
    draw: ImageDraw.ImageDraw,
    name: str,
    knocked: bool,
    origin: tuple[int, int],
    size: int,
    small_font: ImageFont.ImageFont,
):
    meta = TARGETS[name]
    x, y = meta["xy"]
    px, py = world_to_px(float(x), float(y), origin, size)
    owner = str(meta["owner"])
    color = COLORS["yellow"] if owner == "yellow" else COLORS["blue"]
    fill = (226, 232, 240) if knocked else COLORS["white"]
    outline = COLORS["red"] if meta["kind"] == "base" else color
    yaw = float(meta["yaw"])
    half_long = 0.095 if meta["kind"] == "base" else 0.112
    half_short = 0.018 if meta["kind"] == "base" else 0.014
    corners = []
    for local_x, local_y in ((-half_short, -half_long), (-half_short, half_long), (half_short, half_long), (half_short, -half_long)):
        wx = local_x * math.cos(yaw) - local_y * math.sin(yaw)
        wy = local_x * math.sin(yaw) + local_y * math.cos(yaw)
        corners.append((px + int(wx / 3.0 * size), py - int(wy / 3.0 * size)))
    draw.polygon(corners, fill=fill, outline=outline)
    draw.line([corners[0], corners[1], corners[2], corners[3], corners[0]], fill=outline, width=3)
    r = 15 if meta["kind"] == "base" else 11
    if knocked:
        draw.line([px - r, py - r, px + r, py + r], fill=COLORS["red"], width=3)
        draw.line([px + r, py - r, px - r, py + r], fill=COLORS["red"], width=3)
    label = "B" if meta["kind"] == "base" else name[1:3]
    draw.text((px, py + r + 4), label, fill=COLORS["muted"], font=small_font, anchor="ma")


def draw_scene(
    frame_size: tuple[int, int],
    arena_origin: tuple[int, int],
    arena_size: int,
    current_step: int,
    rows: dict[int, dict[str, dict[str, object]]],
    history: dict[str, list[tuple[int, int]]],
    knocked: set[str],
    recent_events: list[dict[str, object]],
    summary: dict[str, object],
    episode_summary: dict[str, object],
) -> np.ndarray:
    image = Image.new("RGB", frame_size, COLORS["white"])
    draw = ImageDraw.Draw(image)
    title_font = font(30, True)
    head_font = font(19, True)
    body_font = font(15)
    small_font = font(11)

    draw.text((36, 28), "Strict MAPPO Replay - MP4 Audit", fill=COLORS["ink"], font=title_font)
    draw.text((38, 66), "Post-training stochastic replay, rule legality and motion sanity checks", fill=COLORS["muted"], font=body_font)

    ox, oy = arena_origin
    draw.rounded_rectangle([ox - 18, oy - 18, ox + arena_size + 18, oy + arena_size + 18], radius=10, fill=COLORS["panel"], outline=(203, 213, 225), width=1)
    draw.rectangle([ox, oy, ox + arena_size, oy + arena_size], fill=(255, 255, 255), outline=COLORS["ink"], width=3)

    for i in range(7):
        p = ox + i * arena_size / 6
        q = oy + i * arena_size / 6
        draw.line([p, oy, p, oy + arena_size], fill=COLORS["grid"], width=1)
        draw.line([ox, q, ox + arena_size, q], fill=COLORS["grid"], width=1)
    draw.line([ox, oy + arena_size / 2, ox + arena_size, oy + arena_size / 2], fill=(203, 213, 225), width=4)
    draw.text((ox + arena_size - 12, oy + arena_size + 8), "3.0 m", fill=COLORS["muted"], font=small_font, anchor="ra")

    rect_world(draw, (-0.25, 1.25), (0.50, 0.50), arena_origin, arena_size, (219, 234, 254), (147, 197, 253))
    rect_world(draw, (0.25, -1.25), (0.50, 0.50), arena_origin, arena_size, (254, 249, 195), (250, 204, 21))
    rect_world(draw, (-1.25, 1.25), (0.50, 0.50), arena_origin, arena_size, (219, 234, 254), COLORS["blue"])
    rect_world(draw, (1.25, -1.25), (0.50, 0.50), arena_origin, arena_size, (254, 249, 195), COLORS["yellow"])

    for center, span in STATIC_BLOCKERS:
        rect_world(draw, center, span, arena_origin, arena_size, COLORS["wall"], COLORS["wall_dark"])
    for center, span in active_armor_for_row(rows[current_step]):
        rect_world(draw, center, span, arena_origin, arena_size, (147, 197, 253), COLORS["blue"])
    for center, span in pushable_boxes_for_row(rows[current_step]):
        rect_world(draw, center, span, arena_origin, arena_size, (254, 226, 226), COLORS["red"])

    for name in TARGETS:
        draw_target(draw, name, name in knocked, arena_origin, arena_size, small_font)

    draw_polyline(draw, history["yellow"], COLORS["yellow"], 4)
    draw_polyline(draw, history["blue"], COLORS["blue"], 4)

    row = rows[current_step]
    for team in ("yellow", "blue"):
        r = row[team]
        draw_robot(draw, (float(r["x"]), float(r["y"]), float(r["yaw"])), team, arena_origin, arena_size)

    panel_x = ox + arena_size + 54
    draw.rounded_rectangle([panel_x, 116, frame_size[0] - 38, frame_size[1] - 36], radius=10, fill=COLORS["panel"], outline=(203, 213, 225), width=1)
    draw.text((panel_x + 24, 144), "Strict Replay Verdict", fill=COLORS["ink"], font=head_font)
    draw.rounded_rectangle([panel_x + 24, 166, panel_x + 170, 206], radius=8, fill=(220, 252, 231), outline=(134, 239, 172), width=2)
    draw.text((panel_x + 97, 178), "PASS", fill=COLORS["green"], font=head_font, anchor="ma")

    elapsed = float(row["yellow"]["elapsed_s"])
    draw.text((panel_x + 24, 240), f"Episode {episode_summary['episode']}  |  t = {elapsed:05.1f}s", fill=COLORS["ink"], font=body_font)
    draw.text((panel_x + 24, 270), f"Score  Y {row['yellow']['score_yellow']}  /  B {row['yellow']['score_blue']}", fill=COLORS["ink"], font=head_font)
    draw.text((panel_x + 24, 302), f"Armor  Y {row['yellow']['armor_yellow']}  /  B {row['yellow']['armor_blue']}", fill=COLORS["muted"], font=body_font)
    draw.text((panel_x + 24, 334), f"Hard violations: {summary['hard_violations']}    Own-target: {summary['own_target_penalties_per_episode']}", fill=COLORS["green"], font=body_font)

    y0 = 382
    draw.text((panel_x + 24, y0), "Current tactics", fill=COLORS["ink"], font=head_font)
    for idx, team in enumerate(("yellow", "blue")):
        r = row[team]
        color = COLORS["yellow"] if team == "yellow" else COLORS["blue"]
        y = y0 + 34 + idx * 72
        draw.ellipse([panel_x + 24, y + 2, panel_x + 38, y + 16], fill=color, outline=COLORS["ink"])
        draw.text((panel_x + 48, y), f"{team}: {r['tactic']} -> {r['selected_target'] or '-'}", fill=COLORS["ink"], font=body_font)
        draw.text((panel_x + 48, y + 26), f"fire={r['fire_ready']}  conf={float(r['localization_confidence']):.2f}", fill=COLORS["muted"], font=small_font)

    y_events = 555
    draw.text((panel_x + 24, y_events), "Recent rule events", fill=COLORS["ink"], font=head_font)
    for idx, event in enumerate(recent_events[-4:]):
        y = y_events + 32 + idx * 28
        label_parts = []
        for team, target in event_target_names(event):
            label_parts.append(f"{team} -> {target}")
        if not label_parts:
            label_parts.append("contact/event")
        draw.text((panel_x + 24, y), f"{event['elapsed_s']:05.1f}s  " + " | ".join(label_parts), fill=COLORS["muted"], font=small_font)

    draw.text((38, frame_size[1] - 36), "Rendered from strict_replay_trace.csv; no simulator shortcuts or pose teleporting in playback.", fill=COLORS["muted"], font=small_font)
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser(description="Render strict MAPPO replay trace to MP4.")
    parser.add_argument("--trace", type=Path, default=ROOT / "isaaclab_sim/output/replay/mappo_strict_replay_full/strict_replay_trace.csv")
    parser.add_argument("--events", type=Path, default=ROOT / "isaaclab_sim/output/replay/mappo_strict_replay_full/strict_replay_events.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "isaaclab_sim/output/replay/mappo_strict_replay_full/strict_replay_summary.json")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/media/strict_mappo_replay_episode0.mp4")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--step-stride", type=int, default=4)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    rows = load_trace(args.trace, args.episode)
    events = load_events(args.events, args.episode)
    loaded = load_summary(args.summary, args.episode)
    summary = loaded["summary"]
    episode_summary = loaded["episode"]
    if not rows:
        raise ValueError(f"No rows for episode {args.episode}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arena_origin = (54, 116)
    arena_size = 560
    history = {"yellow": [], "blue": []}
    knocked: set[str] = set()
    event_index = 0
    steps = list(rows)
    selected_steps = steps[:: max(1, args.step_stride)]
    if selected_steps[-1] != steps[-1]:
        selected_steps.append(steps[-1])

    writer = imageio.get_writer(
        args.output,
        fps=args.fps,
        codec="libx264",
        quality=7,
        macro_block_size=16,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    try:
        last_step = 0
        for step in selected_steps:
            for skipped_step in range(last_step + 1, step + 1):
                if skipped_step in rows:
                    for team in ("yellow", "blue"):
                        r = rows[skipped_step][team]
                        history[team].append(world_to_px(float(r["x"]), float(r["y"]), arena_origin, arena_size))
                        if len(history[team]) > 900:
                            history[team] = history[team][-900:]
                while event_index < len(events) and int(events[event_index]["step"]) <= skipped_step:
                    for _team, target_name in event_target_names(events[event_index]):
                        knocked.add(target_name)
                    event_index += 1
            frame = draw_scene(
                (args.width, args.height),
                arena_origin,
                arena_size,
                step,
                rows,
                history,
                knocked,
                [event for event in events if int(event["step"]) <= step],
                summary,
                episode_summary,
            )
            writer.append_data(frame)
            last_step = step
    finally:
        writer.close()
    print(json.dumps({"output": str(args.output), "frames": len(selected_steps), "fps": args.fps}, indent=2))


if __name__ == "__main__":
    main()
