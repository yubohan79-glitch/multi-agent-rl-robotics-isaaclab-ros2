from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from robocup_visionrl_gym_env import (
    BASE_TARGET_CONTACT_RADIUS,
    HALF_ARENA,
    NORMAL_TARGET_CONTACT_RADIUS,
    PUSHABLE_OBSTACLE_HALF,
    PUSHABLE_OBSTACLE_STARTS,
    ROBOT_RADIUS,
    TARGET_WALL_INSET,
    RoboCupVisionRLGymEnv,
    active_base_armor_blockers,
    segment_intersects_aabb,
)
from robocup_visionrl_selfplay_env import AGENTS, RoboCupVisionRLSelfPlayEnv


ROOT = Path(__file__).resolve().parents[2]


def target_visual_parts(kind: str, xy: tuple[float, float], yaw: float) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    if kind.startswith("base_"):
        support_offset = 0.034
        foot_offset = 0.045
        board_span = (0.012, 0.095)
        support_span = (0.018, 0.018)
        foot_span = (0.075, 0.115)
    else:
        support_offset = -0.034
        foot_offset = -0.045
        board_span = (0.012, 0.180)
        support_span = (0.018, 0.018)
        foot_span = (0.110, 0.205)
    front = (math.cos(yaw), math.sin(yaw))
    return [
        ("board", xy, board_span),
        ("support", (xy[0] + support_offset * front[0], xy[1] + support_offset * front[1]), support_span),
        ("base_plate", (xy[0] + foot_offset * front[0], xy[1] + foot_offset * front[1]), foot_span),
    ]


def oriented_extent(yaw: float, span: tuple[float, float]) -> tuple[float, float]:
    span_x, span_y = span
    return (
        abs(math.cos(yaw)) * span_x * 0.5 + abs(math.sin(yaw)) * span_y * 0.5,
        abs(math.sin(yaw)) * span_x * 0.5 + abs(math.cos(yaw)) * span_y * 0.5,
    )


def overlap_depth(
    center_a: tuple[float, float],
    extent_a: tuple[float, float],
    center_b: tuple[float, float],
    half_b: tuple[float, float],
) -> tuple[float, float]:
    return (
        half_b[0] + extent_a[0] - abs(center_a[0] - center_b[0]),
        half_b[1] + extent_a[1] - abs(center_a[1] - center_b[1]),
    )


def target_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    env = RoboCupVisionRLGymEnv()
    targets = env._make_targets()
    static_blockers = env._make_blockers(inflated=False)
    armor_blockers = active_base_armor_blockers({"yellow": 4, "blue": 4}, inflated=False)
    blockers = [(f"static_{index}", center, half) for index, (center, half) in enumerate(static_blockers)]
    blockers += [(f"armor_{index}", center, half) for index, (center, half) in enumerate(armor_blockers)]
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for target in targets:
        target_xy = np.asarray(target.xy, dtype=np.float32)
        front = np.asarray([math.cos(target.yaw), math.sin(target.yaw)], dtype=np.float32)
        probe_distance = 0.44 if target.kind.startswith("base_") else 0.30
        probe = target_xy + front * probe_distance
        footprint_parts = target_visual_parts(target.kind, tuple(float(v) for v in target.xy), float(target.yaw))
        center_overlaps: list[str] = []
        line_blocks: list[str] = []
        visual_overlaps: list[str] = []
        min_clearance = math.inf
        for label, center, half in blockers:
            clearance_x = max(0.0, abs(float(target.xy[0]) - center[0]) - half[0])
            clearance_y = max(0.0, abs(float(target.xy[1]) - center[1]) - half[1])
            min_clearance = min(min_clearance, math.hypot(clearance_x, clearance_y))
            if (
                abs(float(target.xy[0]) - center[0]) <= half[0] + 0.035
                and abs(float(target.xy[1]) - center[1]) <= half[1] + 0.035
            ):
                center_overlaps.append(label)
            if segment_intersects_aabb((float(probe[0]), float(probe[1])), target.xy, center, half):
                line_blocks.append(label)
            for part_name, part_center, span in footprint_parts:
                depth = overlap_depth(part_center, oriented_extent(target.yaw, span), center, half)
                if depth[0] >= 0.0 and depth[1] >= 0.0:
                    visual_overlaps.append(f"{label}:{part_name}:{depth[0]:.3f}/{depth[1]:.3f}")

        plane_yaw = (float(target.yaw) + math.pi * 0.5) % math.pi
        angle_to_x_wall = min(plane_yaw, math.pi - plane_yaw)
        angle_to_y_wall = abs(math.pi * 0.5 - plane_yaw)
        target_radius = BASE_TARGET_CONTACT_RADIUS if target.kind.startswith("base_") else NORMAL_TARGET_CONTACT_RADIUS
        arena_clearance = HALF_ARENA - max(abs(float(target.xy[0])), abs(float(target.xy[1]))) - target_radius
        row = {
            "name": target.name,
            "owner": target.owner,
            "kind": target.kind,
            "x": round(float(target.xy[0]), 4),
            "y": round(float(target.xy[1]), 4),
            "yaw_deg": round(math.degrees(float(target.yaw)), 3),
            "front_probe_x": round(float(probe[0]), 4),
            "front_probe_y": round(float(probe[1]), 4),
            "plane_angle_x_deg": round(math.degrees(angle_to_x_wall), 3),
            "plane_angle_y_deg": round(math.degrees(angle_to_y_wall), 3),
            "min_blocker_clearance_m": round(float(min_clearance), 4),
            "arena_clearance_m": round(float(arena_clearance), 4),
            "line_blocked_by": ";".join(line_blocks),
            "center_overlaps": ";".join(center_overlaps),
            "visual_overlaps": ";".join(visual_overlaps),
        }
        rows.append(row)
        unexpected_line_blocks = list(line_blocks)
        if target.kind.startswith("base_"):
            # Recessed base targets are intentionally hidden by intact armor.
            # They should fail only if walls or non-armor blockers occlude the
            # target, or if the target geometry overlaps a blocker.
            unexpected_line_blocks = [label for label in line_blocks if not label.startswith("armor_")]
        if unexpected_line_blocks or center_overlaps or visual_overlaps:
            failures.append(
                {
                    "target": target.name,
                    "issue": "blocked_or_overlapped",
                    "unexpected_line_blocks": unexpected_line_blocks,
                    "row": row,
                }
            )
        if abs(angle_to_x_wall - math.pi * 0.25) > math.radians(2.0):
            failures.append({"target": target.name, "issue": "plane_not_45deg_to_x_wall", "row": row})
        if abs(angle_to_y_wall - math.pi * 0.25) > math.radians(2.0):
            failures.append({"target": target.name, "issue": "plane_not_45deg_to_y_wall", "row": row})
        if arena_clearance < 0.0:
            failures.append({"target": target.name, "issue": "outside_arena_clearance", "row": row})

    normal_count = sum(1 for target in targets if target.kind == "normal")
    base_count = sum(1 for target in targets if target.kind.startswith("base_"))
    if normal_count != 8 or base_count != 2 or len(targets) != 10:
        failures.append({"target": "layout", "issue": "wrong_target_count", "normal_count": normal_count, "base_count": base_count})
    if TARGET_WALL_INSET < 0.14:
        failures.append({"target": "layout", "issue": "target_wall_inset_too_small", "value": TARGET_WALL_INSET})
    return rows, failures


def physics_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=271)

    obstacle_pass = (
        np.allclose(PUSHABLE_OBSTACLE_STARTS["box_ne"], np.array([0.80, 0.80], dtype=np.float32), atol=1e-6)
        and np.allclose(PUSHABLE_OBSTACLE_STARTS["box_sw"], np.array([-0.80, -0.80], dtype=np.float32), atol=1e-6)
        and abs(PUSHABLE_OBSTACLE_HALF * 2.0 - 0.30) < 1e-6
    )
    rows.append(
        {
            "check": "pushable_obstacle_rule_reference_positions",
            "pass": obstacle_pass,
            "box_ne": tuple(round(float(v), 3) for v in PUSHABLE_OBSTACLE_STARTS["box_ne"]),
            "box_sw": tuple(round(float(v), 3) for v in PUSHABLE_OBSTACLE_STARTS["box_sw"]),
            "size_m": round(PUSHABLE_OBSTACLE_HALF * 2.0, 3),
        }
    )
    if not obstacle_pass:
        failures.append({"check": "pushable_obstacle_rule_reference_positions", "issue": "obstacle_reference_mismatch"})

    for team in AGENTS:
        base_target = next(target for target in env.targets if target.kind == f"base_{env._opponent(team)}")
        blocked_full = env._best_fire_pose(team, base_target, risk=0.70) is None
        rows.append({"check": f"{team}_full_armor_base_fire_pose_blocked", "pass": blocked_full, "value": "None" if blocked_full else "available"})
        if not blocked_full:
            failures.append({"check": f"{team}_full_armor_base_fire_pose_blocked", "issue": "base_visible_before_armor_removed"})

    env.reset(seed=272)
    start = np.array([0.68, 0.42], dtype=np.float32)
    env.pushable_obstacles["box_ne"] = start.copy()
    env.poses["yellow"] = np.array([0.29, 0.42, 0.0], dtype=np.float32)
    blocked = env._integrate_command("yellow", 0.35, 0.0, allow_push=True)
    pushed = env.pushable_obstacles["box_ne"].copy()
    displacement = float(np.linalg.norm(pushed - start))
    env.poses["yellow"] = np.array([0.05, 0.05, 0.0], dtype=np.float32)
    env._integrate_command("yellow", 0.0, 0.0, allow_push=False)
    persisted = float(np.linalg.norm(env.pushable_obstacles["box_ne"] - pushed)) < 1e-6
    push_pass = (not blocked) and displacement > 0.035 and persisted
    rows.append(
        {
            "check": "pushable_box_moves_and_persists",
            "pass": push_pass,
            "blocked": blocked,
            "displacement_m": round(displacement, 4),
            "persisted": persisted,
            "box_half_m": PUSHABLE_OBSTACLE_HALF,
            "robot_radius_m": round(float(ROBOT_RADIUS), 4),
        }
    )
    if not push_pass:
        failures.append({"check": "pushable_box_moves_and_persists", "issue": "box_did_not_move_or_persist"})
    return rows, failures


def write_outputs(
    target_table: list[dict[str, object]],
    physics_table: list[dict[str, object]],
    failures: list[dict[str, object]],
    *,
    csv_path: Path,
    md_path: Path,
    json_path: Path,
):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(target_table[0].keys()))
        writer.writeheader()
        writer.writerows(target_table)
    payload = {"target_table": target_table, "physics_table": physics_table, "failures": failures}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Rule Geometry Audit",
        "",
        f"Verdict: **{'PASS' if not failures else 'FAIL'}**",
        "",
        "## Target Table",
        "",
        "| name | owner | kind | xy | yaw deg | front probe | plane angles | line blocks | center overlaps | visual overlaps |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in target_table:
        lines.append(
            "| {name} | {owner} | {kind} | ({x}, {y}) | {yaw_deg} | ({front_probe_x}, {front_probe_y}) | "
            "{plane_angle_x_deg}/{plane_angle_y_deg} | {line_blocked_by} | {center_overlaps} | {visual_overlaps} |".format(
                **{key: (value if value != "" else "-") for key, value in row.items()}
            )
        )
    lines.extend(["", "## Physics Checks", "", "| check | pass | details |", "| --- | ---: | --- |"])
    for row in physics_table:
        detail = ", ".join(f"{key}={value}" for key, value in row.items() if key not in ("check", "pass"))
        lines.append(f"| {row['check']} | {row['pass']} | {detail} |")
    lines.extend(["", "## Failures", ""])
    if failures:
        for failure in failures:
            lines.append(f"- `{failure}`")
    else:
        lines.append("- None")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Audit target layout, armor blocking, and pushable-box physics contracts.")
    parser.add_argument("--csv", type=Path, default=ROOT / "docs/rl_data/rule_geometry_audit.csv")
    parser.add_argument("--json", type=Path, default=ROOT / "docs/rl_data/rule_geometry_audit.json")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/rl_rule_geometry_audit.md")
    args = parser.parse_args()

    targets, target_failures = target_rows()
    physics, physics_failures = physics_rows()
    failures = target_failures + physics_failures
    write_outputs(targets, physics, failures, csv_path=args.csv, md_path=args.report, json_path=args.json)
    print(json.dumps({"target_count": len(targets), "physics_checks": len(physics), "failures": len(failures)}, indent=2))
    print(f"[INFO] wrote {args.report}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
