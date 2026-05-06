#!/usr/bin/env bash
set -eo pipefail

OUT="${1:-docs/rl_data/ros2_motion_drift_recheck_20260505}"
DURATION_S="${2:-18.0}"
mkdir -p "$OUT"

source /opt/ros/humble/setup.bash
source crc_robocup_vision_ws/install/setup.bash
set -u

CSV="$OUT/motion_drift_live_log.csv"
LOG="$OUT/launch.log"

ros2 daemon stop >/dev/null 2>&1 || true
ros2 daemon start >/dev/null 2>&1 || true

ros2 launch rcvrl_motion motion_drift_sim_collection.launch.py \
  output_csv:="$CSV" \
  duration_s:="$DURATION_S" \
  >"$LOG" 2>&1 &
LAUNCH_PID=$!

cleanup() {
  kill "$LAUNCH_PID" >/dev/null 2>&1 || true
  wait "$LAUNCH_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

sleep 4
ros2 topic list --no-daemon -t >"$OUT/topic_list.txt" 2>&1 || true
ros2 topic info /cmd_vel >"$OUT/cmd_vel_info.txt" 2>&1 || true
ros2 topic info /scan >"$OUT/scan_info.txt" 2>&1 || true
timeout 5s ros2 topic hz /cmd_vel --window 8 >"$OUT/cmd_vel_hz.txt" 2>&1 || true
timeout 5s ros2 topic hz /scan --window 8 >"$OUT/scan_hz.txt" 2>&1 || true
timeout 4s ros2 topic echo /cmd_vel --once >"$OUT/cmd_vel_once.txt" 2>&1 || true
timeout 4s ros2 topic echo /wheel/odom --once >"$OUT/wheel_odom_once.txt" 2>&1 || true
timeout 4s ros2 topic echo /odometry/filtered --once >"$OUT/filtered_odom_once.txt" 2>&1 || true

sleep "$(python3 - <<PY
duration = float("$DURATION_S")
print(max(2.0, duration - 8.0))
PY
)"
cleanup
trap - EXIT

python3 - "$OUT" "$CSV" <<'PY'
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

out = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
rows = []
if csv_path.exists():
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))


def values(key: str) -> list[float]:
    result = []
    for row in rows:
        raw = row.get(key, "")
        if raw == "":
            continue
        try:
            result.append(float(raw))
        except ValueError:
            pass
    return result


def mean(items: list[float]) -> float:
    return sum(items) / len(items) if items else 0.0


def p95(items: list[float]) -> float:
    if not items:
        return 0.0
    ordered = sorted(items)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


cmd_lin = values("cmd_linear_x")
cmd_ang = values("cmd_angular_z")
lin_acc = values("linear_accel")
ang_acc = values("angular_accel")
xy = values("odom_xy_error_m")
yaw = values("odom_yaw_error_rad")
scan = values("front_scan_min_m")
risk = values("drift_risk")
high_risk = [
    float(row["drift_risk"])
    for row in rows
    if row.get("drift_risk", "")
    and (float(row.get("linear_accel") or 0.0) > 1.05 or float(row.get("angular_accel") or 0.0) > 4.20)
]
low_risk = [
    float(row["drift_risk"])
    for row in rows
    if row.get("drift_risk", "")
    and (float(row.get("linear_accel") or 0.0) <= 1.05 and float(row.get("angular_accel") or 0.0) <= 4.20)
]

duration = 0.0
if len(rows) > 1:
    duration = float(rows[-1]["time_s"]) - float(rows[0]["time_s"])

summary = {
    "date": "2026-05-05",
    "ros_distro": "humble",
    "source": "WSL2 ROS2 live topics from rcvrl_motion motion_drift_sim_collection.launch.py",
    "csv": str(csv_path),
    "samples": len(rows),
    "duration_s": round(duration, 3),
    "cmd_linear_x_max_abs": round(max([abs(item) for item in cmd_lin], default=0.0), 5),
    "cmd_angular_z_max_abs": round(max([abs(item) for item in cmd_ang], default=0.0), 5),
    "linear_accel_max_mps2": round(max(lin_acc, default=0.0), 5),
    "angular_accel_max_radps2": round(max(ang_acc, default=0.0), 5),
    "odom_xy_error_mean_m": round(mean(xy), 5),
    "odom_xy_error_p95_m": round(p95(xy), 5),
    "odom_yaw_error_mean_rad": round(mean(yaw), 5),
    "front_scan_min_mean_m": round(mean(scan), 5),
    "front_scan_min_min_m": round(min(scan) if scan else 0.0, 5),
    "drift_risk_mean": round(mean(risk), 5),
    "drift_risk_p95": round(p95(risk), 5),
    "drift_risk_max": round(max(risk, default=0.0), 5),
    "high_accel_drift_risk_mean": round(mean(high_risk), 5),
    "low_accel_drift_risk_mean": round(mean(low_risk), 5),
    "topic_checks": {
        "topic_list": str(out / "topic_list.txt"),
        "cmd_vel_hz": str(out / "cmd_vel_hz.txt"),
        "scan_hz": str(out / "scan_hz.txt"),
        "wheel_odom_once": str(out / "wheel_odom_once.txt"),
    },
}

(out / "motion_drift_live_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
