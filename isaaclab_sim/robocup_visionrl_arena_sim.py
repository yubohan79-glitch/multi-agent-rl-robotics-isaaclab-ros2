from __future__ import annotations

"""IsaacLab scene for the RoboCup VisionRL portfolio project.

Run from the local IsaacLab checkout:

    isaaclab.bat -p <this_file.py> --enable_cameras

The scene is metric and keeps the robot/sensor dimensions aligned with the
ROS2 description in rcvrl_description/urdf/robocup_visionrl_robot.urdf.xacro.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="RoboCup VisionRL IsaacLab simulation scene.")
parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run. 0 means run until the GUI closes.")
parser.add_argument("--seed", type=int, default=7, help="Deterministic layout seed for sim2real preview variation.")
parser.add_argument("--static_robot", action="store_true", help="Keep both robots at their start zones.")
parser.add_argument(
    "--demo_flow",
    action="store_true",
    help="Run a deterministic full-match portfolio replay with target falls, armor removal, recovery, and base hit.",
)
parser.add_argument("--record_video", type=str, default="", help="Optional MP4 output path recorded from an IsaacLab RGB camera.")
parser.add_argument(
    "--record_view",
    choices=["overview", "yellow_pov", "blue_pov"],
    default="overview",
    help="Camera view for --record_video.",
)
parser.add_argument("--record_fps", type=int, default=30, help="Video frame rate for --record_video.")
parser.add_argument("--record_width", type=int, default=1600, help="Video width for --record_video.")
parser.add_argument("--record_height", type=int, default=900, help="Video height for --record_video.")
parser.add_argument(
    "--enable_sensor_streams",
    action="store_true",
    help="Start live IsaacLab camera/lidar sensors. Off by default to keep GUI preview and shutdown stable.",
)
parser.add_argument("--no_sensor_streams", action="store_true", help=argparse.SUPPRESS)
parser.add_argument(
    "--save_usd",
    type=str,
    default="",
    help="Optional USD export path. Defaults to isaaclab_sim/output/robocup_visionrl_arena.usd.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Live camera streams require Replicator. Keep them opt-in because this PC's
# Isaac Sim 5.1 build can hang during headless shutdown after semantic camera use.
if (args_cli.enable_sensor_streams or args_cli.record_video) and not args_cli.no_sensor_streams:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
from isaacsim.core.utils.stage import get_current_stage
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from pxr import Gf, Sdf, UsdGeom

if args_cli.record_video:
    import cv2
    import numpy as np
    import torch


ARENA_SIZE = 3.0
WALL_HEIGHT = 0.50
WALL_THICKNESS = 0.04
ZONE_SIZE = 0.50
OBSTACLE_SIZE = 0.30
TAG_SIZE = 0.05
TAG_BOTTOM_Z = 0.07
TAG_CENTER_Z = TAG_BOTTOM_Z + TAG_SIZE * 0.5

ROBOT_LENGTH = 0.34
ROBOT_WIDTH = 0.24
ROBOT_BODY_HEIGHT = 0.16
ROBOT_TOTAL_HEIGHT = 0.245
WHEEL_RADIUS = 0.045
WHEEL_WIDTH = 0.025
BASE_LINK_Z = WHEEL_RADIUS
LIDAR_POSE = (0.06, 0.0, BASE_LINK_Z + 0.19)
CAMERA_POSE = (0.18, 0.0, BASE_LINK_Z + 0.18)
SHOOTER_POSE = (0.20, 0.0, BASE_LINK_Z + 0.14)
IMU_POSE = (-0.02, 0.0, BASE_LINK_Z + 0.11)
DEPTH_CAMERA_POSE = (0.165, 0.045, BASE_LINK_Z + 0.17)
TOF_FRONT_POSE = (0.185, 0.0, BASE_LINK_Z + 0.075)

COLLISION_PRIMS: list[str] = []
RAYCAST_BOXES: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float, float]]] = []
NAV_BLOCKERS: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
LASER_BLOCKERS: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
TARGET_REGISTRY: dict[str, dict[str, object]] = {}
BASE_ARMOR: dict[str, list[str]] = {"yellow": [], "blue": []}
LAST_FIRE_TIME: dict[str, float] = {"yellow": -99.0, "blue": -99.0}
ARMOR_REMOVALS: list[dict[str, object]] = []
TARGET_FALLS: list[dict[str, object]] = []
MATCH_STATE: dict[str, object] = {
    "winner": None,
    "robot_contact": False,
    "last_contact_print": -99.0,
    "last_score_print": -99.0,
    "score_yellow": 0,
    "score_blue": 0,
    "current_time": 0.0,
    "last_event": "ready",
}
MATCH_CONTROLLERS: dict[str, "StrategyTeamController"] = {}

YELLOW_ROBOT_PATH = "/World/RoboCupVisionRL_Yellow"
BLUE_ROBOT_PATH = "/World/RoboCupVisionRL_Blue"
PRIMARY_ROBOT_PATH = YELLOW_ROBOT_PATH

BLUE_BASE_XY = (-1.25, 1.25)
BLUE_START_XY = (-0.25, 1.25)
YELLOW_START_XY = (0.25, -1.25)
YELLOW_BASE_XY = (1.25, -1.25)
YELLOW_DEMO_START_XY = (0.38, -1.18)
BLUE_DEMO_START_XY = (-0.38, 1.18)
ROUTE_CLEARANCE = ROBOT_WIDTH * 0.5 + 0.04
ROBOT_COLLISION_RADIUS = math.hypot(ROBOT_LENGTH * 0.5, ROBOT_WIDTH * 0.5)
SHOOT_RANGE = 1.65
SHOOT_HIT_RADIUS = 0.15
BASE_HIT_RADIUS = 0.20
FIRE_COOLDOWN = 1.4
MATCH_DRIVE_SPEED = 0.72
MATCH_AIM_TIME = 0.45
MATCH_DURATION_S = 180.0
PLANNER_GRID_RESOLUTION = 0.10
BASE_RUSH_MIN_QUALITY = 0.34
BLOCK_HOLD_S = 4.0
BLOCK_LEAD_SCORE = 10
BLOCK_LATE_TIME_S = 45.0
LOCALIZATION_RECOVERY_THRESHOLD = 0.58
LOCALIZATION_RECOVERY_ROTATION_RAD = math.tau * 1.08
LOCALIZATION_CONTACT_LOSS = 0.42
LOCALIZATION_STUCK_LOSS = 0.20
LOCALIZATION_SPIN_GAIN = 0.38
TARGET_CONTACT_RADIUS = 0.115
BASE_TARGET_CONTACT_RADIUS = 0.180
LINEAR_ACCEL_LIMIT = 1.10
ANGULAR_ACCEL_LIMIT = 4.80
WHEEL_SPEED_LIMIT = 0.54
WHEEL_ACCEL_LIMIT = 1.35
MIN_TURN_ALIGNMENT = 0.35
MAX_CONTACT_CORRECTION_STEP = 0.022
COSTMAP_SOFT_INFLATION = 0.06
COSTMAP_HARD_MARGIN = 0.018
COSTMAP_MAX_REPULSE_STEP = 0.025
COSTMAP_WARN_INTERVAL_S = 0.80
COSTMAP_LAST_WARN: dict[str, float] = {}
OPPONENT_TRACK_RANGE = 3.25
OPPONENT_THREAT_RADIUS = 1.10
OPPONENT_THREAT_BLOCK_THRESHOLD = 0.42
OPPONENT_AVOID_RANGE = 0.38
OPPONENT_AVOID_BEARING_RAD = math.radians(48.0)

YELLOW_ROUTE = [
    YELLOW_START_XY,
    (0.25, -0.78),
    (0.18, -0.22),
    (0.18, 0.20),
    (0.55, 0.20),
    (0.95, 0.20),
    (1.20, 0.22),
]

BLUE_ROUTE = [
    BLUE_START_XY,
    (-0.25, 0.78),
    (-0.18, 0.22),
    (-0.18, -0.20),
    (-0.55, -0.20),
    (-0.95, -0.20),
    (-1.20, -0.22),
]

MATCH_TASKS = {
    "yellow": [
        ("T01_NorthMiddle", (-0.25, 0.78)),
        ("T03_WestAboveGate", (-1.20, 0.22)),
        ("T05_EastAboveGate", (1.20, 0.22)),
        ("T02_NorthEast", (1.18, 1.18)),
        ("BlueBaseTarget", (-0.70, 0.78)),
    ],
    "blue": [
        ("T08_SouthMiddle", (0.25, -0.78)),
        ("T04_WestBelowGate", (-1.20, -0.22)),
        ("T06_EastBelowGate", (1.20, -0.22)),
        ("T07_SouthWest", (-1.18, -1.18)),
        ("YellowBaseTarget", (0.70, -0.78)),
    ],
}

DEMO_POLICY_TASKS = {
    "yellow": [
        ("T03_WestAboveGate", (-1.16, 0.34)),
        ("T01_NorthMiddle", (-0.38, 0.98)),
        ("T02_NorthEast", (1.18, 1.05)),
        ("BlueBaseTarget", (-0.66, 0.82)),
        ("T05_EastAboveGate", (1.16, 0.34)),
    ],
    "blue": [
        ("T06_EastBelowGate", (1.16, -0.34)),
        ("T08_SouthMiddle", (0.38, -0.98)),
        ("T04_WestBelowGate", (-1.16, -0.34)),
        ("YellowBaseTarget", (0.66, -0.82)),
        ("T07_SouthWest", (-1.18, -1.05)),
    ],
}

DEMO_FLOW_FIRE_EVENTS: list[tuple[float, str, str]] = [
    (6.20, "yellow", "T05_EastAboveGate"),
    (7.10, "blue", "T04_WestBelowGate"),
    (10.10, "yellow", "T02_NorthEast"),
    (11.10, "blue", "T07_SouthWest"),
    (19.40, "yellow", "T03_WestAboveGate"),
    (21.70, "blue", "T06_EastBelowGate"),
    (25.30, "yellow", "T01_NorthMiddle"),
    (28.40, "blue", "T08_SouthMiddle"),
    (34.00, "yellow", "BlueBaseTarget"),
]

DEMO_FLOW_POSES: dict[str, list[tuple[float, tuple[float, float], str | None]]] = {
    "yellow": [
        (0.00, YELLOW_START_XY, None),
        (2.80, (0.25, -0.62), None),
        (5.35, (0.88, 0.28), "T05_EastAboveGate"),
        (6.65, (0.88, 0.28), "T05_EastAboveGate"),
        (9.20, (1.30, 0.88), "T02_NorthEast"),
        (10.45, (1.30, 0.88), "T02_NorthEast"),
        (13.10, (0.10, 0.07), None),
        (16.70, (0.10, 0.07), None),
        (18.75, (-0.88, 0.28), "T03_WestAboveGate"),
        (20.00, (-0.88, 0.28), "T03_WestAboveGate"),
        (24.50, (-0.25, 0.78), "T01_NorthMiddle"),
        (25.70, (-0.25, 0.78), "T01_NorthMiddle"),
        (32.55, (-0.70, 0.78), "BlueBaseTarget"),
        (36.50, (-0.70, 0.78), "BlueBaseTarget"),
        (42.00, (-0.35, 0.42), "BlueBaseTarget"),
    ],
    "blue": [
        (0.00, BLUE_START_XY, None),
        (2.80, (-0.25, 0.62), None),
        (5.60, (-0.88, -0.28), "T04_WestBelowGate"),
        (7.45, (-0.88, -0.28), "T04_WestBelowGate"),
        (10.20, (-1.30, -0.88), "T07_SouthWest"),
        (11.45, (-1.30, -0.88), "T07_SouthWest"),
        (13.10, (-0.10, -0.07), None),
        (16.70, (-0.10, -0.07), None),
        (20.90, (0.88, -0.28), "T06_EastBelowGate"),
        (22.10, (0.88, -0.28), "T06_EastBelowGate"),
        (27.60, (0.25, -0.78), "T08_SouthMiddle"),
        (28.80, (0.25, -0.78), "T08_SouthMiddle"),
        (33.30, (0.35, -0.42), "YellowBaseTarget"),
        (42.00, (0.35, -0.42), "YellowBaseTarget"),
    ],
}

DEMO_FLOW_RECOVERY_WINDOWS = ((13.00, 16.70),)
DEMO_FLOW_TRIGGERED_EVENTS: set[int] = set()
DEMO_FLOW_PATH_CACHE: dict[tuple[str, int], list[tuple[float, float]]] = {}


def opponent_team(team: str) -> str:
    return "blue" if team == "yellow" else "yellow"


def target_name_from_path(target_path: str) -> str:
    return target_path.rsplit("/", 1)[-1]


def team_base_xy(team: str) -> tuple[float, float]:
    return YELLOW_BASE_XY if team == "yellow" else BLUE_BASE_XY


def team_score(team: str) -> int:
    return int(MATCH_STATE[f"score_{team}"])


def static_fire_pose(team: str, target_name: str, tasks: list[tuple[str, tuple[float, float]]] | None = None) -> tuple[float, float] | None:
    task_table = tasks if tasks is not None else (DEMO_POLICY_TASKS[team] if args_cli.demo_flow else MATCH_TASKS[team])
    for candidate_name, fire_xy in task_table:
        if candidate_name == target_name:
            return fire_xy
    return None


def empty_opponent_estimate() -> dict[str, float | bool]:
    return {
        "available": False,
        "visible": False,
        "dx": 0.0,
        "dy": 0.0,
        "distance": OPPONENT_TRACK_RANGE,
        "global_bearing": 0.0,
        "relative_bearing": 0.0,
        "relative_heading": 0.0,
        "distance_to_own_base": OPPONENT_THREAT_RADIUS,
        "heading_to_own_base": 0.0,
        "threat_to_own_base": 0.0,
    }


def opponent_bearing_estimate(
    team: str,
    own_pose: tuple[tuple[float, float, float], float],
    opponent_pose: tuple[tuple[float, float, float], float],
) -> dict[str, float | bool]:
    own_pos, own_yaw = own_pose
    opponent_pos, opponent_yaw = opponent_pose
    dx = opponent_pos[0] - own_pos[0]
    dy = opponent_pos[1] - own_pos[1]
    distance = math.hypot(dx, dy)
    global_bearing = math.atan2(dy, dx) if distance > 1e-6 else own_yaw
    relative_bearing = wrap_angle(global_bearing - own_yaw)
    relative_heading = wrap_angle(opponent_yaw - own_yaw)
    line_of_sight = not line_blocked_by_wall((own_pos[0], own_pos[1]), (opponent_pos[0], opponent_pos[1]))
    visible = distance <= OPPONENT_TRACK_RANGE and line_of_sight

    own_base = team_base_xy(team)
    base_dx = own_base[0] - opponent_pos[0]
    base_dy = own_base[1] - opponent_pos[1]
    distance_to_own_base = math.hypot(base_dx, base_dy)
    base_bearing_from_opponent = math.atan2(base_dy, base_dx) if distance_to_own_base > 1e-6 else opponent_yaw
    heading_to_own_base = abs(wrap_angle(base_bearing_from_opponent - opponent_yaw))
    proximity_threat = max(0.0, 1.0 - distance_to_own_base / OPPONENT_THREAT_RADIUS)
    heading_threat = max(0.0, 1.0 - heading_to_own_base / math.pi)
    visibility_scale = 1.0 if visible else 0.72
    threat_to_own_base = max(0.0, min(1.0, proximity_threat * (0.55 + 0.45 * heading_threat) * visibility_scale))

    return {
        "available": True,
        "visible": visible,
        "dx": dx,
        "dy": dy,
        "distance": distance,
        "global_bearing": global_bearing,
        "relative_bearing": relative_bearing,
        "relative_heading": relative_heading,
        "distance_to_own_base": distance_to_own_base,
        "heading_to_own_base": heading_to_own_base,
        "threat_to_own_base": threat_to_own_base,
    }


def quat_from_euler(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Return USD quaternion order (w, x, y, z)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def rotate_local(offset: tuple[float, float, float], roll: float, pitch: float, yaw: float) -> tuple[float, float, float]:
    """Apply Rz(yaw) * Ry(pitch) * Rx(roll) to a local vector."""
    x, y, z = offset

    cr = math.cos(roll)
    sr = math.sin(roll)
    y, z = cr * y - sr * z, sr * y + cr * z

    cp = math.cos(pitch)
    sp = math.sin(pitch)
    x, z = cp * x + sp * z, -sp * x + cp * z

    cy = math.cos(yaw)
    sy = math.sin(yaw)
    x, y = cy * x - sy * y, sy * x + cy * y
    return (x, y, z)


def quat_rotate(
    quat: tuple[float, float, float, float], vector: tuple[float, float, float]
) -> tuple[float, float, float]:
    w, x, y, z = quat
    vx, vy, vz = vector
    # q * v * q^-1, expanded to avoid extra dependencies.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def local_to_world(
    origin: tuple[float, float, float],
    offset: tuple[float, float, float],
    roll: float,
    pitch: float,
    yaw: float,
) -> tuple[float, float, float]:
    dx, dy, dz = rotate_local(offset, roll, pitch, yaw)
    return (origin[0] + dx, origin[1] + dy, origin[2] + dz)


def create_xform(
    path: str,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    if translation is not None or orientation is not None:
        set_xform(path, translation or (0.0, 0.0, 0.0), orientation or (1.0, 0.0, 0.0, 0.0))
    return prim


def set_xform(path: str, translation: tuple[float, float, float], orientation: tuple[float, float, float, float]):
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim does not exist: {path}")

    xform = UsdGeom.Xformable(prim)
    ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
    translate_op = ops.get("xformOp:translate")
    orient_op = ops.get("xformOp:orient")
    scale_op = ops.get("xformOp:scale")
    if translate_op is None:
        translate_op = xform.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble)
    if orient_op is None:
        orient_op = xform.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble)
    if scale_op is None:
        scale_op = xform.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble)
        scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0))
    translate_op.Set(Gf.Vec3d(*translation))
    orient_op.Set(Gf.Quatd(orientation[0], Gf.Vec3d(orientation[1], orientation[2], orientation[3])))
    xform.SetXformOpOrder([translate_op, orient_op, scale_op])


def get_xform(path: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim does not exist: {path}")

    translation = (0.0, 0.0, 0.0)
    orientation = (1.0, 0.0, 0.0, 0.0)
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            value = op.Get()
            translation = (float(value[0]), float(value[1]), float(value[2]))
        elif op.GetOpName() == "xformOp:orient":
            value = op.Get()
            imag = value.GetImaginary()
            orientation = (float(value.GetReal()), float(imag[0]), float(imag[1]), float(imag[2]))
    return translation, orientation


def set_visibility(path: str, visible: bool):
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


def material(
    color: tuple[float, float, float],
    opacity: float = 1.0,
    roughness: float = 0.55,
    metallic: float = 0.0,
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(
        diffuse_color=color,
        emissive_color=emissive,
        roughness=roughness,
        metallic=metallic,
        opacity=opacity,
    )


def spawn_box(
    path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    collision: bool = False,
    raycast: bool = False,
    semantic: str | None = None,
    opacity: float = 1.0,
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rigid_body: bool = False,
    kinematic: bool = False,
    mass: float | None = None,
    disable_gravity: bool = False,
):
    cfg = sim_utils.CuboidCfg(
        size=size,
        visual_material=material(color, opacity=opacity, emissive=emissive),
        collision_props=sim_utils.CollisionPropertiesCfg() if collision or rigid_body else None,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=kinematic,
            disable_gravity=disable_gravity,
            max_depenetration_velocity=1.2,
        )
        if rigid_body
        else None,
        mass_props=sim_utils.MassPropertiesCfg(mass=mass) if rigid_body and mass is not None else None,
        semantic_tags=[("class", semantic)] if semantic else None,
    )
    cfg.func(path, cfg, translation=pos, orientation=orientation)
    if raycast:
        COLLISION_PRIMS.append(path)
        RAYCAST_BOXES.append((size, pos, orientation))


def spawn_cylinder(
    path: str,
    radius: float,
    height: float,
    axis: str,
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    collision: bool = False,
    raycast: bool = False,
    semantic: str | None = None,
    opacity: float = 1.0,
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    cfg = sim_utils.CylinderCfg(
        radius=radius,
        height=height,
        axis=axis,
        visual_material=material(color, opacity=opacity, emissive=emissive),
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        semantic_tags=[("class", semantic)] if semantic else None,
    )
    cfg.func(path, cfg, translation=pos, orientation=orientation)
    if raycast:
        COLLISION_PRIMS.append(path)
        if axis.upper() == "X":
            box_size = (height, radius * 2.0, radius * 2.0)
        elif axis.upper() == "Y":
            box_size = (radius * 2.0, height, radius * 2.0)
        else:
            box_size = (radius * 2.0, radius * 2.0, height)
        RAYCAST_BOXES.append((box_size, pos, orientation))


def spawn_marker_cell(
    path: str,
    center: tuple[float, float, float],
    size_y: float,
    size_z: float,
    local_y: float,
    local_z: float,
    roll: float,
    pitch: float,
    yaw: float,
    color: tuple[float, float, float] = (0.01, 0.01, 0.01),
):
    orient = quat_from_euler(roll, pitch, yaw)
    pos = local_to_world(center, (0.003, local_y, local_z), roll, pitch, yaw)
    spawn_box(path, (0.003, size_y, size_z), pos, color, orientation=orient, semantic="apriltag_visual")


def spawn_local_box(
    path: str,
    center: tuple[float, float, float],
    local_offset: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    roll: float,
    pitch: float,
    yaw: float,
    *,
    collision: bool = False,
    raycast: bool = False,
    semantic: str | None = None,
    opacity: float = 1.0,
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    orient = quat_from_euler(roll, pitch, yaw)
    pos = local_to_world(center, local_offset, roll, pitch, yaw)
    spawn_box(
        path,
        size,
        pos,
        color,
        orientation=orient,
        collision=collision,
        raycast=raycast,
        semantic=semantic,
        opacity=opacity,
        emissive=emissive,
    )


def spawn_local_cylinder(
    path: str,
    center: tuple[float, float, float],
    local_offset: tuple[float, float, float],
    radius: float,
    height: float,
    axis: str,
    color: tuple[float, float, float],
    roll: float,
    pitch: float,
    yaw: float,
    *,
    collision: bool = False,
    raycast: bool = False,
    semantic: str | None = None,
    opacity: float = 1.0,
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    orient = quat_from_euler(roll, pitch, yaw)
    pos = local_to_world(center, local_offset, roll, pitch, yaw)
    spawn_cylinder(
        path,
        radius,
        height,
        axis,
        pos,
        color,
        orientation=orient,
        collision=collision,
        raycast=raycast,
        semantic=semantic,
        opacity=opacity,
        emissive=emissive,
    )


def spawn_apriltag(
    path: str,
    center: tuple[float, float, float],
    tag_id: int,
    roll: float,
    pitch: float,
    yaw: float,
):
    """Build a physical tag-like target from geometry.

    The high-contrast layout is intentionally made of primitive geometry so the
    USD stays portable. The metadata and surrounding docs record that the real
    detector uses the AprilTag Tag36h11 family.
    """
    create_xform(path)
    orient = quat_from_euler(roll, pitch, yaw)

    spawn_box(
        f"{path}/black_carrier",
        (0.002, TAG_SIZE * 1.28, TAG_SIZE * 1.28),
        local_to_world(center, (-0.001, 0.0, 0.0), roll, pitch, yaw),
        (0.01, 0.012, 0.012),
        orientation=orient,
        semantic=f"tag36h11_id_{tag_id}_carrier",
    )
    spawn_box(
        f"{path}/white_laminate",
        (0.003, TAG_SIZE * 1.14, TAG_SIZE * 1.14),
        center,
        (0.97, 0.97, 0.93),
        orientation=orient,
        semantic=f"tag36h11_id_{tag_id}",
    )

    border = TAG_SIZE * 0.13
    half = TAG_SIZE * 0.5
    spawn_marker_cell(f"{path}/border_left", center, border, TAG_SIZE, -half + border * 0.5, 0.0, roll, pitch, yaw)
    spawn_marker_cell(f"{path}/border_right", center, border, TAG_SIZE, half - border * 0.5, 0.0, roll, pitch, yaw)
    spawn_marker_cell(f"{path}/border_top", center, TAG_SIZE, border, 0.0, half - border * 0.5, roll, pitch, yaw)
    spawn_marker_cell(f"{path}/border_bottom", center, TAG_SIZE, border, 0.0, -half + border * 0.5, roll, pitch, yaw)

    # Compact 6x6 visual code. It is not used for detection in this script; it
    # makes IDs 1, 2, and 3 visibly distinct while still reading like AprilTag.
    patterns_by_id = {
        1: {
            (0, 0),
            (0, 5),
            (1, 1),
            (1, 4),
            (2, 2),
            (2, 5),
            (3, 0),
            (3, 3),
            (4, 1),
            (4, 4),
            (5, 2),
            (5, 5),
        },
        2: {
            (0, 1),
            (0, 4),
            (1, 0),
            (1, 2),
            (2, 3),
            (2, 5),
            (3, 1),
            (3, 4),
            (4, 0),
            (4, 3),
            (5, 2),
            (5, 4),
        },
        3: {
            (0, 0),
            (0, 3),
            (0, 5),
            (1, 1),
            (1, 4),
            (2, 0),
            (2, 2),
            (2, 5),
            (3, 1),
            (3, 3),
            (4, 0),
            (4, 2),
            (4, 5),
            (5, 1),
            (5, 4),
        },
    }
    cell = TAG_SIZE * 0.075
    pitch_between = TAG_SIZE * 0.112
    origin = -2.5 * pitch_between
    for iy, iz in patterns_by_id.get(tag_id, patterns_by_id[1]):
        local_y = origin + iy * pitch_between
        local_z = origin + iz * pitch_between
        spawn_marker_cell(
            f"{path}/id_{tag_id}_cell_{iy}_{iz}",
            center,
            cell,
            cell,
            local_y,
            local_z,
            roll,
            pitch,
            yaw,
        )


def spawn_target_id_badge(
    path: str,
    board_center: tuple[float, float, float],
    tag_id: int,
    board_size: tuple[float, float, float],
    roll: float,
    pitch: float,
    yaw: float,
    accent_color: tuple[float, float, float],
):
    front_x = board_size[0] * 0.5 + 0.008
    badge_y = -board_size[1] * 0.5 + 0.045
    badge_z = board_size[2] * 0.5 - 0.030
    spawn_local_box(
        f"{path}/id_badge_backplate",
        board_center,
        (front_x, badge_y, badge_z),
        (0.006, 0.070, 0.028),
        (0.04, 0.045, 0.045),
        roll,
        pitch,
        yaw,
        semantic="target_id_badge",
    )
    spawn_local_box(
        f"{path}/id_badge_team_strip",
        board_center,
        (front_x + 0.002, badge_y, badge_z + 0.010),
        (0.007, 0.062, 0.006),
        accent_color,
        roll,
        pitch,
        yaw,
        semantic="target_id_team_strip",
    )
    for index in range(tag_id):
        dot_y = badge_y - 0.018 + index * 0.018
        spawn_local_box(
            f"{path}/id_badge_dot_{index + 1}",
            board_center,
            (front_x + 0.004, dot_y, badge_z - 0.004),
            (0.008, 0.010, 0.012),
            (0.95, 0.95, 0.86),
            roll,
            pitch,
            yaw,
            semantic=f"target_id_{tag_id}_dot",
        )


def spawn_target(
    path: str,
    xy: tuple[float, float],
    yaw: float,
    *,
    tag_id: int = 1,
    pitch: float = 0.0,
    frame_color: tuple[float, float, float] = (0.20, 0.22, 0.24),
    base_target: bool = False,
):
    create_xform(path)
    roll = 0.0
    orient = quat_from_euler(roll, pitch, yaw)
    target_name = path.rsplit("/", 1)[-1]
    accent_color = frame_color
    face_color = (0.86, 0.87, 0.80)
    dark_frame = (0.035, 0.038, 0.040)
    warning_color = (0.98, 0.70, 0.12) if tag_id == 1 else frame_color
    board_center = (xy[0], xy[1], 0.138 if base_target else 0.124)
    board_size = (0.018, 0.245, 0.245) if base_target else (0.014, 0.205, 0.215)
    front_x = board_size[0] * 0.5 + 0.006
    edge = 0.014 if base_target else 0.012
    tag_local_z = 0.012 if base_target else 0.006

    spawn_box(
        f"{path}/target_board",
        board_size,
        board_center,
        face_color,
        orientation=orient,
        collision=True,
        raycast=True,
        semantic=f"target_board_id_{tag_id}",
    )

    # Raised structural frame: it makes the target read as hardware instead of
    # a flat texture and gives the laser hit board a clear silhouette.
    spawn_local_box(
        f"{path}/frame_left",
        board_center,
        (front_x, -board_size[1] * 0.5 + edge * 0.5, 0.0),
        (0.009, edge, board_size[2] + edge),
        dark_frame,
        roll,
        pitch,
        yaw,
        semantic="target_frame",
    )
    spawn_local_box(
        f"{path}/frame_right",
        board_center,
        (front_x, board_size[1] * 0.5 - edge * 0.5, 0.0),
        (0.009, edge, board_size[2] + edge),
        dark_frame,
        roll,
        pitch,
        yaw,
        semantic="target_frame",
    )
    spawn_local_box(
        f"{path}/frame_top",
        board_center,
        (front_x, 0.0, board_size[2] * 0.5 - edge * 0.5),
        (0.009, board_size[1] + edge, edge),
        dark_frame,
        roll,
        pitch,
        yaw,
        semantic="target_frame",
    )
    spawn_local_box(
        f"{path}/frame_bottom",
        board_center,
        (front_x, 0.0, -board_size[2] * 0.5 + edge * 0.5),
        (0.009, board_size[1] + edge, edge),
        dark_frame,
        roll,
        pitch,
        yaw,
        semantic="target_frame",
    )

    spawn_local_box(
        f"{path}/lower_status_strip",
        board_center,
        (front_x + 0.003, 0.0, -board_size[2] * 0.5 + edge + 0.010),
        (0.008, board_size[1] - edge * 2.4, 0.010),
        warning_color,
        roll,
        pitch,
        yaw,
        semantic="target_status_strip",
        emissive=(warning_color[0] * 0.05, warning_color[1] * 0.05, warning_color[2] * 0.05),
    )

    tag_center = local_to_world(board_center, (front_x + 0.004, 0.0, tag_local_z), roll, pitch, yaw)
    spawn_apriltag(f"{path}/tag36h11_{tag_id}", tag_center, tag_id, roll, pitch, yaw)

    reticle_color = (0.92, 0.08, 0.08) if not base_target else (0.98, 0.18, 0.18)
    for index, (local_y, local_z, size_y, size_z) in enumerate(
        (
            (-0.060, tag_local_z, 0.030, 0.004),
            (0.060, tag_local_z, 0.030, 0.004),
            (0.0, tag_local_z - 0.060, 0.004, 0.030),
            (0.0, tag_local_z + 0.060, 0.004, 0.030),
        )
    ):
        spawn_local_box(
            f"{path}/laser_reticle_{index + 1}",
            board_center,
            (front_x + 0.006, local_y, local_z),
            (0.005, size_y, size_z),
            reticle_color,
            roll,
            pitch,
            yaw,
            semantic="laser_hit_reticle",
            emissive=(0.10, 0.0, 0.0),
        )

    spawn_target_id_badge(path, board_center, tag_id, board_size, roll, pitch, yaw, accent_color)

    lens_y = board_size[1] * 0.5 - 0.040
    lens_z = board_size[2] * 0.5 - 0.035
    spawn_local_box(
        f"{path}/hit_indicator_lens",
        board_center,
        (front_x + 0.007, lens_y, lens_z),
        (0.009, 0.020, 0.020),
        (0.86, 0.02, 0.03),
        roll,
        pitch,
        yaw,
        semantic="laser_hit_indicator",
        emissive=(0.25, 0.0, 0.0),
    )

    support_height = 0.185 if base_target else 0.165
    support_center = local_to_world(board_center, (-0.040, 0.0, -support_height * 0.18), roll, pitch, yaw)
    spawn_box(
        f"{path}/rear_support_post",
        (0.026, 0.026, support_height),
        support_center,
        frame_color,
        orientation=quat_from_euler(0.0, 0.0, yaw),
        collision=True,
        raycast=True,
        semantic="target_rear_support",
    )
    spawn_local_cylinder(
        f"{path}/bottom_hinge",
        board_center,
        (-0.034, 0.0, -board_size[2] * 0.5 - 0.004),
        0.010,
        board_size[1] * 0.88,
        "Y",
        (0.08, 0.085, 0.085),
        roll,
        pitch,
        yaw,
        semantic="target_hinge",
    )
    foot_center = local_to_world(board_center, (-0.045, 0.0, -board_center[2] + 0.014), roll, pitch, yaw)
    foot_size = (0.150, 0.255, 0.020) if base_target else (0.130, 0.230, 0.018)
    spawn_box(
        f"{path}/weighted_base_plate",
        foot_size,
        (foot_center[0], foot_center[1], foot_size[2] * 0.5),
        frame_color,
        orientation=quat_from_euler(0.0, 0.0, yaw),
        collision=True,
        raycast=True,
        semantic="target_weighted_base",
    )
    for index, local_y in enumerate((-foot_size[1] * 0.36, foot_size[1] * 0.36)):
        spawn_local_cylinder(
            f"{path}/base_anchor_bolt_{index + 1}",
            (foot_center[0], foot_center[1], foot_size[2] * 0.5),
            (foot_size[0] * 0.24, local_y, foot_size[2] * 0.5 + 0.002),
            0.010,
            0.006,
            "Z",
            (0.05, 0.052, 0.052),
            0.0,
            0.0,
            yaw,
            semantic="target_base_bolt",
        )

    if base_target:
        spawn_local_box(
            f"{path}/base_target_backbone",
            board_center,
            (-0.030, 0.0, 0.0),
            (0.030, board_size[1] * 0.72, 0.020),
            dark_frame,
            roll,
            pitch,
            yaw,
            semantic="base_target_backbone",
        )
        spawn_local_box(
            f"{path}/base_target_warning_window",
            board_center,
            (front_x + 0.009, 0.0, board_size[2] * 0.5 - 0.064),
            (0.006, board_size[1] * 0.42, 0.014),
            (0.96, 0.08, 0.10),
            roll,
            pitch,
            yaw,
            semantic="base_target_critical_window",
            emissive=(0.18, 0.0, 0.0),
        )

    fallen_path = f"/World/Targets/Fallen/{path.rsplit('/', 1)[-1]}_fallen"
    fall_anim_path = f"/World/Targets/Falling/{path.rsplit('/', 1)[-1]}_falling"
    create_xform(fall_anim_path, translation=(xy[0], xy[1], 0.0), orientation=quat_from_euler(0.0, 0.0, yaw))
    spawn_box(
        f"{fall_anim_path}/target_board",
        board_size,
        (0.0, 0.0, board_center[2]),
        face_color,
        semantic=f"falling_target_board_id_{tag_id}",
    )
    spawn_box(
        f"{fall_anim_path}/target_frame_top",
        (0.010, board_size[1] + edge, edge),
        (front_x, 0.0, board_center[2] + board_size[2] * 0.5 - edge * 0.5),
        dark_frame,
        semantic="falling_target_frame",
    )
    spawn_box(
        f"{fall_anim_path}/target_frame_bottom",
        (0.010, board_size[1] + edge, edge),
        (front_x, 0.0, board_center[2] - board_size[2] * 0.5 + edge * 0.5),
        dark_frame,
        semantic="falling_target_frame",
    )
    spawn_apriltag(
        f"{fall_anim_path}/tag36h11_{tag_id}",
        (front_x + 0.006, 0.0, board_center[2] + tag_local_z),
        tag_id,
        0.0,
        0.0,
        0.0,
    )
    spawn_box(
        f"{fall_anim_path}/hit_indicator_lens",
        (0.008, 0.020, 0.020),
        (front_x + 0.012, lens_y, board_center[2] + lens_z),
        (0.86, 0.02, 0.03),
        emissive=(0.25, 0.0, 0.0),
        semantic="falling_hit_indicator",
    )
    spawn_box(
        f"{fall_anim_path}/weighted_base",
        foot_size,
        (-0.045, 0.0, foot_size[2] * 0.5),
        frame_color,
        semantic="falling_target_base",
    )
    spawn_cylinder(
        f"{fall_anim_path}/bottom_hinge",
        radius=0.010,
        height=board_size[1] * 0.88,
        axis="Y",
        pos=(-0.034, 0.0, board_center[2] - board_size[2] * 0.5 - 0.004),
        color=(0.08, 0.085, 0.085),
        semantic=f"falling_tag36h11_id_{tag_id}",
    )
    set_visibility(fall_anim_path, False)

    create_xform(fallen_path)
    spawn_box(
        f"{fallen_path}/board",
        (board_size[1], board_size[2], 0.014),
        (xy[0], xy[1], 0.022),
        face_color,
        orientation=quat_from_euler(0.0, 0.0, yaw),
        semantic=f"fallen_target_id_{tag_id}",
    )
    spawn_box(
        f"{fallen_path}/tag_patch",
        (TAG_SIZE * 1.18, TAG_SIZE * 1.18, 0.005),
        (xy[0], xy[1], 0.034),
        (0.94, 0.94, 0.88),
        orientation=quat_from_euler(0.0, 0.0, yaw),
        semantic=f"fallen_tag36h11_id_{tag_id}",
    )
    spawn_box(
        f"{fallen_path}/dark_frame",
        (board_size[1] + 0.018, board_size[2] + 0.018, 0.006),
        (xy[0], xy[1], 0.018),
        dark_frame,
        orientation=quat_from_euler(0.0, 0.0, yaw),
        semantic="fallen_target_frame_shadow",
    )
    set_visibility(fallen_path, False)

    if base_target:
        kind = "base_yellow" if tag_id == 2 else "base_blue"
    else:
        kind = "normal"
    if kind == "base_yellow":
        owner = "yellow"
    elif kind == "base_blue":
        owner = "blue"
    else:
        owner = "blue" if xy[1] >= 0.0 else "yellow"
    TARGET_REGISTRY[path] = {
        "path": path,
        "fallen_path": fallen_path,
        "fall_anim_path": fall_anim_path,
        "xy": xy,
        "yaw": yaw,
        "tag_id": tag_id,
        "kind": kind,
        "owner": owner,
        "knocked": False,
    }


def register_nav_blocker(path: str, pos: tuple[float, float, float], size: tuple[float, float, float]):
    NAV_BLOCKERS.append(
        (
            path,
            (pos[0], pos[1]),
            (size[0] * 0.5 + ROUTE_CLEARANCE, size[1] * 0.5 + ROUTE_CLEARANCE),
        )
    )
    LASER_BLOCKERS.append((path, (pos[0], pos[1]), (size[0] * 0.5, size[1] * 0.5)))


def unregister_blocker(path: str):
    NAV_BLOCKERS[:] = [item for item in NAV_BLOCKERS if item[0] != path]
    LASER_BLOCKERS[:] = [item for item in LASER_BLOCKERS if item[0] != path]


def spawn_nav_blocker(
    path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    semantic: str,
):
    spawn_box(path, size, pos, color, collision=True, raycast=True, semantic=semantic)
    register_nav_blocker(path, pos, size)


def segment_intersects_aabb(
    p0: tuple[float, float],
    p1: tuple[float, float],
    center: tuple[float, float],
    half_size: tuple[float, float],
) -> bool:
    min_x = center[0] - half_size[0]
    max_x = center[0] + half_size[0]
    min_y = center[1] - half_size[1]
    max_y = center[1] + half_size[1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    t_min = 0.0
    t_max = 1.0

    for start, delta, lower, upper in ((p0[0], dx, min_x, max_x), (p0[1], dy, min_y, max_y)):
        if abs(delta) < 1e-9:
            if start < lower or start > upper:
                return False
            continue
        inv_delta = 1.0 / delta
        t1 = (lower - start) * inv_delta
        t2 = (upper - start) * inv_delta
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False
    return True


def validate_route(name: str, route: list[tuple[float, float]], *, strict: bool = False) -> bool:
    valid = True
    for index, (p0, p1) in enumerate(zip(route, route[1:])):
        for blocker_path, center, half_size in NAV_BLOCKERS:
            if segment_intersects_aabb(p0, p1, center, half_size):
                valid = False
                message = f"{name} route segment {index} intersects {blocker_path}; costmap recovery will repel the robot."
                if strict:
                    raise RuntimeError(message)
                print(f"[COSTMAP]: {message}")
    return valid


def spawn_route_markers(name: str, route: list[tuple[float, float]], color: tuple[float, float, float]):
    for i, (x, y) in enumerate(route):
        spawn_cylinder(
            f"/World/Arena/{name}_RouteWaypoint_{i:02d}",
            radius=0.030,
            height=0.005,
            axis="Z",
            pos=(x, y, 0.010),
            color=color,
            opacity=0.75,
            semantic="behavior_route_waypoint",
        )


def spawn_base_armor(base_team: str, base_xy: tuple[float, float], color: tuple[float, float, float]):
    create_xform(f"/World/Arena/{base_team.capitalize()}BaseArmor")
    z = 0.105
    armor_height = 0.18
    armor_thickness = 0.026
    armor_length = 0.17

    if base_team == "blue":
        # Order follows the rule diagram around the blue base: 1, 2, 3, 4.
        specs = [
            ("armor_1", (-0.985, 1.405, z), (armor_thickness, armor_length, armor_height)),
            ("armor_2", (-1.405, 0.985, z), (armor_length, armor_thickness, armor_height)),
            ("armor_3", (-0.985, 1.155, z), (armor_thickness, armor_length, armor_height)),
            ("armor_4", (-1.155, 0.985, z), (armor_length, armor_thickness, armor_height)),
        ]
        rack_x = base_xy[0] - 0.18
        rack_y = base_xy[1] - 0.62
    else:
        # Order follows the rule diagram around the yellow base: 1, 2, 3, 4.
        specs = [
            ("armor_1", (1.155, -1.485, z), (armor_length, armor_thickness, armor_height)),
            ("armor_2", (1.405, -0.985, z), (armor_length, armor_thickness, armor_height)),
            ("armor_3", (0.985, -1.155, z), (armor_thickness, armor_length, armor_height)),
            ("armor_4", (1.155, -0.985, z), (armor_length, armor_thickness, armor_height)),
        ]
        rack_x = base_xy[0] + 0.18
        rack_y = base_xy[1] + 0.62

    BASE_ARMOR[base_team] = []
    for index, (name, pos, size) in enumerate(specs):
        armor_path = f"/World/Arena/{base_team.capitalize()}BaseArmor/{name}"
        spawn_box(
            armor_path,
            size,
            pos,
            color,
            collision=True,
            raycast=True,
            semantic=f"{base_team}_base_armor_{index + 1}",
        )
        register_nav_blocker(armor_path, pos, size)
        BASE_ARMOR[base_team].append(armor_path)

        rack_path = f"/World/Arena/{base_team.capitalize()}BaseArmor/removed_slot_{index + 1}"
        spawn_box(
            rack_path,
            (0.030, 0.030, 0.006),
            (rack_x + 0.055 * index, rack_y, 0.010),
            color,
            opacity=0.35,
            semantic="removed_armor_slot",
        )


def target_path_from_name(name: str) -> str:
    return f"/World/Targets/{name}"


def point_blocked(point: tuple[float, float]) -> bool:
    x, y = point
    for _, center, half_size in NAV_BLOCKERS:
        if abs(x - center[0]) <= half_size[0] and abs(y - center[1]) <= half_size[1]:
            return True
    return False


def segment_blocked(p0: tuple[float, float], p1: tuple[float, float]) -> bool:
    for _, center, half_size in NAV_BLOCKERS:
        if segment_intersects_aabb(p0, p1, center, half_size):
            return True
    return False


def snap_to_grid(point: tuple[float, float]) -> tuple[int, int]:
    res = PLANNER_GRID_RESOLUTION
    return (round(point[0] / res), round(point[1] / res))


def grid_to_world(cell: tuple[int, int]) -> tuple[float, float]:
    res = PLANNER_GRID_RESOLUTION
    return (cell[0] * res, cell[1] * res)


def warn_costmap(source: str, message: str):
    now = float(MATCH_STATE.get("current_time", 0.0))
    key = f"{source}:{message}"
    if now - COSTMAP_LAST_WARN.get(key, -999.0) < COSTMAP_WARN_INTERVAL_S:
        return
    COSTMAP_LAST_WARN[key] = now
    MATCH_STATE["last_event"] = message
    print(f"[COSTMAP]: {source} {message}")


def nearest_free_point(point: tuple[float, float]) -> tuple[float, float]:
    if not point_blocked(point):
        return point
    for radius_step in range(1, 10):
        radius = radius_step * PLANNER_GRID_RESOLUTION
        samples = max(12, radius_step * 8)
        for sample in range(samples):
            angle = math.tau * float(sample) / float(samples)
            candidate = (point[0] + math.cos(angle) * radius, point[1] + math.sin(angle) * radius)
            if not point_blocked(candidate):
                return candidate
    limit = ARENA_SIZE * 0.5 - ROUTE_CLEARANCE
    return (max(-limit, min(limit, point[0])), max(-limit, min(limit, point[1])))


def clamp_to_arena(point: tuple[float, float]) -> tuple[float, float]:
    limit = ARENA_SIZE * 0.5 - ROUTE_CLEARANCE
    return (max(-limit, min(limit, point[0])), max(-limit, min(limit, point[1])))


def aabb_costmap_repel(
    point: tuple[float, float],
    center: tuple[float, float],
    half_size: tuple[float, float],
) -> tuple[tuple[float, float], bool, float]:
    x, y = point
    cx, cy = center
    hx, hy = half_size
    sx = x - cx
    sy = y - cy
    inside_x = hx - abs(sx)
    inside_y = hy - abs(sy)
    if inside_x >= 0.0 and inside_y >= 0.0:
        sign_x = 1.0 if sx >= 0.0 else -1.0
        sign_y = 1.0 if sy >= 0.0 else -1.0
        if inside_x <= inside_y:
            return (sign_x * (inside_x + COSTMAP_HARD_MARGIN), 0.0), True, 1.0
        return (0.0, sign_y * (inside_y + COSTMAP_HARD_MARGIN)), True, 1.0

    closest_x = max(cx - hx, min(x, cx + hx))
    closest_y = max(cy - hy, min(y, cy + hy))
    dx = x - closest_x
    dy = y - closest_y
    distance = math.hypot(dx, dy)
    if distance <= 1e-6 or distance >= COSTMAP_SOFT_INFLATION:
        return (0.0, 0.0), False, 0.0
    strength = (COSTMAP_SOFT_INFLATION - distance) / COSTMAP_SOFT_INFLATION
    step = min(COSTMAP_MAX_REPULSE_STEP, strength * COSTMAP_MAX_REPULSE_STEP)
    return (dx / distance * step, dy / distance * step), False, strength


def circle_costmap_repel(
    point: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> tuple[tuple[float, float], bool, float]:
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-6:
        dx, dy, distance = 1.0, 0.0, 1.0
    if distance < radius:
        step = radius - distance + COSTMAP_HARD_MARGIN
        return (dx / distance * step, dy / distance * step), True, 1.0
    soft_distance = radius + COSTMAP_SOFT_INFLATION
    if distance >= soft_distance:
        return (0.0, 0.0), False, 0.0
    strength = (soft_distance - distance) / COSTMAP_SOFT_INFLATION
    step = min(COSTMAP_MAX_REPULSE_STEP, strength * COSTMAP_MAX_REPULSE_STEP)
    return (dx / distance * step, dy / distance * step), False, strength


def dynamic_target_costmap() -> list[tuple[str, tuple[float, float], float]]:
    blockers = []
    for target_path, target in TARGET_REGISTRY.items():
        if target["knocked"]:
            continue
        xy = target["xy"]
        kind = str(target["kind"])
        assert isinstance(xy, tuple)
        radius = ROBOT_COLLISION_RADIUS + (BASE_TARGET_CONTACT_RADIUS if kind.startswith("base_") else TARGET_CONTACT_RADIUS)
        blockers.append((target_path, xy, radius))
    return blockers


def aabb_clearance(point: tuple[float, float], center: tuple[float, float], half_size: tuple[float, float]) -> float:
    dx = max(abs(point[0] - center[0]) - half_size[0], 0.0)
    dy = max(abs(point[1] - center[1]) - half_size[1], 0.0)
    if dx <= 0.0 and dy <= 0.0:
        return -min(half_size[0] - abs(point[0] - center[0]), half_size[1] - abs(point[1] - center[1]))
    return math.hypot(dx, dy)


def costmap_potential(point: tuple[float, float]) -> float:
    potential = 0.0
    for _blocker_path, center, half_size in NAV_BLOCKERS:
        clearance = aabb_clearance(point, center, half_size)
        if clearance < 0.0:
            return 1e6
        if clearance < COSTMAP_SOFT_INFLATION:
            strength = (COSTMAP_SOFT_INFLATION - clearance) / COSTMAP_SOFT_INFLATION
            potential += 8.0 * strength * strength

    for _target_path, center, radius in dynamic_target_costmap():
        distance = math.hypot(point[0] - center[0], point[1] - center[1])
        clearance = distance - radius
        if clearance < 0.0:
            return 1e6
        if clearance < COSTMAP_SOFT_INFLATION:
            strength = (COSTMAP_SOFT_INFLATION - clearance) / COSTMAP_SOFT_INFLATION
            potential += 6.0 * strength * strength
    return potential


def apply_costmap_recovery(
    point: tuple[float, float],
    source: str,
    *,
    passes: int = 3,
) -> tuple[tuple[float, float], bool, bool]:
    corrected = point
    touched = False
    hard_touched = False
    for _ in range(passes):
        total_x = 0.0
        total_y = 0.0
        strongest = 0.0
        touched_name = ""
        hard_touch = False

        for blocker_path, center, half_size in NAV_BLOCKERS:
            push, hard, strength = aabb_costmap_repel(corrected, center, half_size)
            if hard or strength > 0.0:
                total_x += push[0]
                total_y += push[1]
                if hard or strength > strongest:
                    strongest = max(strength, strongest)
                    touched_name = blocker_path.rsplit("/", 1)[-1]
                    hard_touch = hard_touch or hard

        for target_path, center, radius in dynamic_target_costmap():
            push, hard, strength = circle_costmap_repel(corrected, center, radius)
            if hard or strength > 0.0:
                total_x += push[0]
                total_y += push[1]
                if hard or strength > strongest:
                    strongest = max(strength, strongest)
                    touched_name = target_path.rsplit("/", 1)[-1]
                    hard_touch = hard_touch or hard

        if abs(total_x) < 1e-7 and abs(total_y) < 1e-7:
            break
        touched = True
        hard_touched = hard_touched or hard_touch
        corrected = clamp_to_arena((corrected[0] + total_x, corrected[1] + total_y))
        if touched_name:
            mode = "hard contact" if hard_touch else "near obstacle"
            warn_costmap(source, f"{mode} near {touched_name}; repulsive costmap recovery")

    return corrected, touched, hard_touched


def plan_safe_path(start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]]:
    if point_blocked(start):
        warn_costmap("planner", f"start inside obstacle at ({start[0]:.2f}, {start[1]:.2f}); using nearest free cell")
        start = nearest_free_point(start)
    if point_blocked(goal):
        warn_costmap("planner", f"goal inside obstacle at ({goal[0]:.2f}, {goal[1]:.2f}); using nearest free cell")
        goal = nearest_free_point(goal)

    start_cell = snap_to_grid(start)
    goal_cell = snap_to_grid(goal)
    min_cell = math.floor((-ARENA_SIZE * 0.5 + ROUTE_CLEARANCE) / PLANNER_GRID_RESOLUTION)
    max_cell = math.ceil((ARENA_SIZE * 0.5 - ROUTE_CLEARANCE) / PLANNER_GRID_RESOLUTION)
    neighbors = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    open_set: set[tuple[int, int]] = {start_cell}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start_cell: 0.0}
    f_score: dict[tuple[int, int], float] = {
        start_cell: math.hypot(goal_cell[0] - start_cell[0], goal_cell[1] - start_cell[1])
    }

    while open_set:
        current = min(open_set, key=lambda cell: f_score.get(cell, float("inf")))
        if current == goal_cell:
            grid_path = [current]
            while current in came_from:
                current = came_from[current]
                grid_path.append(current)
            grid_path.reverse()
            path = [start]
            path.extend(grid_to_world(cell) for cell in grid_path[1:-1])
            path.append(goal)
            return smooth_path(path)

        open_set.remove(current)
        current_world = grid_to_world(current)
        for dx, dy in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if nxt[0] < min_cell or nxt[0] > max_cell or nxt[1] < min_cell or nxt[1] > max_cell:
                continue
            nxt_world = grid_to_world(nxt)
            if point_blocked(nxt_world) or segment_blocked(current_world, nxt_world):
                continue
            local_cost = costmap_potential(nxt_world)
            if local_cost >= 1e5:
                continue
            tentative_g = g_score[current] + math.hypot(dx, dy) * (1.0 + local_cost)
            if tentative_g >= g_score.get(nxt, float("inf")):
                continue
            came_from[nxt] = current
            g_score[nxt] = tentative_g
            f_score[nxt] = tentative_g + math.hypot(goal_cell[0] - nxt[0], goal_cell[1] - nxt[1])
            open_set.add(nxt)

    warn_costmap(
        "planner",
        f"A* failed from ({start[0]:.2f}, {start[1]:.2f}) to ({goal[0]:.2f}, {goal[1]:.2f}); falling back to reactive costmap",
    )
    return [start, goal]


def smooth_path(path: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(path) <= 2:
        return path
    smoothed = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        nxt = len(path) - 1
        while nxt > anchor + 1 and segment_unsafe_for_robot(path[anchor], path[nxt]):
            nxt -= 1
        smoothed.append(path[nxt])
        anchor = nxt
    return smoothed


def segment_unsafe_for_robot(p0: tuple[float, float], p1: tuple[float, float]) -> bool:
    if segment_blocked(p0, p1):
        return True
    length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    samples = max(2, math.ceil(length / (PLANNER_GRID_RESOLUTION * 0.55)))
    for index in range(1, samples):
        alpha = index / samples
        sample = (p0[0] + (p1[0] - p0[0]) * alpha, p0[1] + (p1[1] - p0[1]) * alpha)
        if costmap_potential(sample) > 4.0:
            return True
    return False


def interpolate_path(
    path: list[tuple[float, float]],
    distance: float,
) -> tuple[tuple[float, float, float], float, bool]:
    if len(path) < 2:
        return (path[0][0], path[0][1], 0.0), 0.0, True

    walked = 0.0
    for p0, p1 in zip(path, path[1:]):
        segment_length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if distance <= walked + segment_length:
            alpha = 0.0 if segment_length <= 1e-9 else (distance - walked) / segment_length
            x = p0[0] + (p1[0] - p0[0]) * alpha
            y = p0[1] + (p1[1] - p0[1]) * alpha
            yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            return (x, y, 0.0), yaw, False
        walked += segment_length

    final = path[-1]
    previous = path[-2]
    yaw = math.atan2(final[1] - previous[1], final[0] - previous[0])
    return (final[0], final[1], 0.0), yaw, True


def path_length(path: list[tuple[float, float]]) -> float:
    return sum(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) for p0, p1 in zip(path, path[1:]))


def demo_policy_corridor(team: str, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> list[tuple[float, float]]:
    """Wide, regulation-safe staging waypoints for the portfolio self-play replay.

    The high-level policy still decides which opponent target to attack. These
    waypoints emulate the low-level Nav2 corridor preference that keeps the
    differential-drive base away from start rails, inner fences, and armor.
    """
    sx, sy = start_xy
    gx, gy = goal_xy
    if team == "yellow":
        staging = []
        if sy < -0.84:
            staging.append((max(0.36, sx), -0.78))
        if sy < -0.42:
            staging.append((0.24, -0.58))
        if sy < 0.24:
            staging.append((0.20, -0.26))
        if gy >= 0.24:
            staging.append((0.20, 0.42))
            staging.append((0.0 if gx < 0.0 else 0.34, max(0.42, gy)))
        if gx < -0.34:
            staging.append((-0.34, max(0.40, gy)))
        elif gx > 0.34 and gy < 0.24:
            staging.append((0.34, min(-0.28, gy)))
    else:
        staging = []
        if sy > 0.84:
            staging.append((min(-0.36, sx), 0.78))
        if sy > 0.42:
            staging.append((-0.24, 0.58))
        if sy > -0.24:
            staging.append((-0.20, 0.26))
        if gy <= -0.24:
            staging.append((-0.20, -0.42))
            staging.append((0.0 if gx > 0.0 else -0.34, min(-0.42, gy)))
        if gx > 0.34:
            staging.append((0.34, min(-0.40, gy)))
        elif gx < -0.34 and gy > -0.24:
            staging.append((-0.34, max(0.28, gy)))

    route = [start_xy]
    for waypoint in staging:
        waypoint = nearest_free_point(clamp_to_arena(waypoint))
        if math.hypot(waypoint[0] - route[-1][0], waypoint[1] - route[-1][1]) > 0.08:
            route.append(waypoint)
    if math.hypot(goal_xy[0] - route[-1][0], goal_xy[1] - route[-1][1]) > 0.04:
        route.append(goal_xy)
    return route


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def slew_rate(current: float, target: float, max_delta: float) -> float:
    if target > current + max_delta:
        return current + max_delta
    if target < current - max_delta:
        return current - max_delta
    return target


class StrategyTeamController:
    def __init__(
        self,
        team: str,
        start_xy: tuple[float, float],
        start_yaw: float,
        tasks: list[tuple[str, tuple[float, float]]],
        speed: float,
    ):
        self.team = team
        self.pose = ((start_xy[0], start_xy[1], 0.0), start_yaw)
        self.tasks = tasks
        self.speed = speed
        self.task_index = 0
        self.state = "plan"
        self.path: list[tuple[float, float]] = []
        self.waypoint_index = 1
        self.last_update_t = 0.0
        self.aim_start_time = 0.0
        self.current_target_path = ""
        self.left_wheel_spin = 0.0
        self.right_wheel_spin = 0.0
        self.last_left_wheel_speed = 0.0
        self.last_right_wheel_speed = 0.0
        self.last_linear_velocity = 0.0
        self.last_angular_velocity = 0.0
        self.motion_blocked = False
        self.blocked_since = 0.0
        self.recover_until = 0.0
        self.recover_spin_direction = 1.0
        self.last_progress_distance = float("inf")
        self.last_progress_t = 0.0
        self.current_fire_xy: tuple[float, float] | None = None
        self.block_until = 0.0
        self.last_strategy_print = -99.0
        self.localization_confidence = 1.0
        self.relocalize_rotation = 0.0
        self.last_contact_t = -99.0
        self.start_delay = 0.0
        self.opponent_estimate = empty_opponent_estimate()

    def set_pose(self, pose: tuple[tuple[float, float, float], float]):
        self.pose = pose

    def notify_contact(self, t: float):
        if t - self.last_contact_t < 0.35:
            return
        self.last_contact_t = t
        self.localization_confidence = max(0.05, self.localization_confidence - LOCALIZATION_CONTACT_LOSS)
        self.state = "relocalize"
        self.relocalize_rotation = 0.0
        self._print_strategy(t, f"collision disturbed localization confidence={self.localization_confidence:.2f}; spinning to rebuild map")

    def update(self, t: float) -> tuple[tuple[float, float, float], float]:
        dt = max(0.0, min(0.05, t - self.last_update_t)) if self.last_update_t > 0.0 else 0.0
        self.last_update_t = t
        self.opponent_estimate = self._estimate_opponent()
        if MATCH_STATE["winner"] is not None:
            self.last_linear_velocity = 0.0
            self.last_angular_velocity = 0.0
            return self.pose

        if t < self.start_delay:
            self._integrate_differential(0.0, 0.0, dt)
            return self.pose

        if self.state == "relocalize":
            self._spin_relocalize(dt)
            if self.relocalize_rotation >= LOCALIZATION_RECOVERY_ROTATION_RAD:
                self.localization_confidence = 1.0
                self.state = "plan"
                self.blocked_since = 0.0
                self.last_progress_distance = float("inf")
                self._print_strategy(t, "localization rebuilt from lidar/imu/camera scan")
            return self.pose

        if self.state == "recover":
            self._integrate_differential(-0.045, self.recover_spin_direction * 1.25, dt)
            if t >= self.recover_until:
                self.state = "plan"
                self.blocked_since = 0.0
                self.last_progress_distance = float("inf")
            return self.pose

        if self.state == "plan":
            if self.localization_confidence < LOCALIZATION_RECOVERY_THRESHOLD:
                self.state = "relocalize"
                self.relocalize_rotation = 0.0
                self._print_strategy(t, f"localization confidence={self.localization_confidence:.2f}; spinning before next decision")
                return self.pose
            strategy = self._select_strategy(t)
            if strategy["mode"] == "wait":
                self._integrate_differential(0.0, 0.0, dt)
                return self.pose

            if strategy["mode"] == "block":
                block_xy = strategy["fire_xy"]
                assert isinstance(block_xy, tuple)
                start_xy = (self.pose[0][0], self.pose[0][1])
                self.path = self._plan_path(start_xy, block_xy)
                self.waypoint_index = 1
                self.current_target_path = ""
                self.current_fire_xy = block_xy
                self.block_until = t + BLOCK_HOLD_S
                self.state = "drive_block"
                self.blocked_since = 0.0
                self.last_progress_distance = float("inf")
                self.last_progress_t = t
                self._print_strategy(t, f"blocking central lane at ({block_xy[0]:.2f}, {block_xy[1]:.2f})")
                return self.pose

            target_path = str(strategy["target_path"])
            fire_xy = strategy["fire_xy"]
            assert isinstance(fire_xy, tuple)
            self.current_target_path = target_path
            self.current_fire_xy = fire_xy
            start_xy = (self.pose[0][0], self.pose[0][1])
            self.path = self._plan_path(start_xy, fire_xy)
            self.waypoint_index = 1
            self.state = "drive"
            self.blocked_since = 0.0
            self.last_progress_distance = float("inf")
            self.last_progress_t = t
            target_name = target_name_from_path(target_path)
            self._print_strategy(t, f"attacking {target_name} from ({fire_xy[0]:.2f}, {fire_xy[1]:.2f})")

        if self.state in ("drive", "drive_block"):
            arrived = self._drive_differential(dt)
            if arrived and self.state == "drive":
                self.aim_start_time = t
                self.state = "aim"
            elif arrived and self.state == "drive_block":
                self.state = "block"
            elif self._drive_is_stuck(t):
                self.recover_until = t + 0.80
                self.recover_spin_direction = -1.0 if self.team == "yellow" else 1.0
                self.state = "recover"
                self.localization_confidence = max(0.08, self.localization_confidence - LOCALIZATION_STUCK_LOSS)
                print(f"[MATCH]: {self.team} blocked; backing off and replanning.")

        if self.state == "block":
            opponent_pose = self._opponent_pose()
            if opponent_pose is not None:
                opponent_xy = (opponent_pose[0][0], opponent_pose[0][1])
                self._aim_differential(opponent_xy, dt)
            else:
                self._integrate_differential(0.0, 0.35 if self.team == "yellow" else -0.35, dt)
            if t >= self.block_until or not self._should_block(t):
                self.state = "plan"

        if self.state == "aim":
            target = TARGET_REGISTRY[self.current_target_path]
            target_xy = target["xy"]
            assert isinstance(target_xy, tuple)
            aligned = self._aim_differential(target_xy, dt)
            if aligned and t - self.aim_start_time >= MATCH_AIM_TIME and t - LAST_FIRE_TIME[self.team] >= FIRE_COOLDOWN:
                LAST_FIRE_TIME[self.team] = t
                hit_path = detect_laser_hit(self.team, self.pose)
                if hit_path == self.current_target_path:
                    knocked = apply_fire_rule(self.team, self.current_target_path)
                    if knocked or target["knocked"]:
                        self.task_index += 1
                        self.state = "plan"
                else:
                    self._print_strategy(t, f"shot withheld; no clean line to {target_name_from_path(self.current_target_path)}")
                    self.state = "plan"

        return self.pose

    def _select_strategy(self, t: float) -> dict[str, object]:
        if self._should_block(t):
            block_xy = self._select_block_point()
            if block_xy is not None:
                return {"mode": "block", "fire_xy": block_xy}

        candidates = self._attack_candidates()
        if not candidates:
            return {"mode": "wait"}

        scored = [(self._score_attack(candidate, t), candidate) for candidate in candidates]
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_candidate = scored[0]
        if best_score <= -50.0:
            return {"mode": "wait"}
        best_candidate["mode"] = "attack"
        return best_candidate

    def _plan_path(self, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> list[tuple[float, float]]:
        return plan_safe_path(start_xy, goal_xy)

    def _attack_candidates(self) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        start_xy = (self.pose[0][0], self.pose[0][1])
        for target_path, target in TARGET_REGISTRY.items():
            if target["knocked"] or target["owner"] == self.team:
                continue
            kind = str(target["kind"])
            if kind != "normal" and kind != f"base_{opponent_team(self.team)}":
                continue
            solution = self._best_fire_solution(start_xy, target_path)
            if solution is None:
                continue
            fire_xy, route_len, shot_quality = solution
            candidates.append(
                {
                    "target_path": target_path,
                    "fire_xy": fire_xy,
                    "route_len": route_len,
                    "shot_quality": shot_quality,
                    "kind": kind,
                }
            )
        return candidates

    def _best_fire_solution(
        self,
        start_xy: tuple[float, float],
        target_path: str,
    ) -> tuple[tuple[float, float], float, float] | None:
        target = TARGET_REGISTRY[target_path]
        target_xy = target["xy"]
        assert isinstance(target_xy, tuple)
        yaw = float(target["yaw"])
        target_name = target_name_from_path(target_path)
        front = (math.cos(yaw), math.sin(yaw))
        tangent = (-front[1], front[0])
        fire_candidates: list[tuple[float, float]] = []
        fixed = static_fire_pose(self.team, target_name, self.tasks)
        if fixed is not None:
            fire_candidates.append(fixed)
        for distance in (0.46, 0.58, 0.72, 0.90):
            for side_offset in (0.0, -0.16, 0.16, -0.28, 0.28):
                fire_candidates.append(
                    (
                        target_xy[0] + front[0] * distance + tangent[0] * side_offset,
                        target_xy[1] + front[1] * distance + tangent[1] * side_offset,
                    )
                )

        best: tuple[tuple[float, float], float, float] | None = None
        best_score = -999.0
        seen: set[tuple[int, int]] = set()
        for fire_xy in fire_candidates:
            key = (round(fire_xy[0] * 100), round(fire_xy[1] * 100))
            if key in seen:
                continue
            seen.add(key)
            if self._fire_pose_rejected(fire_xy, target_xy):
                continue
            try:
                route = plan_safe_path(start_xy, fire_xy)
            except RuntimeError:
                continue
            route_len = path_length(route)
            shot_distance = math.hypot(target_xy[0] - fire_xy[0], target_xy[1] - fire_xy[1])
            quality = max(0.0, 1.0 - abs(shot_distance - 0.58) / 1.10)
            quality += 0.15 if not line_blocked_by_wall(fire_xy, target_xy) else -0.45
            score = quality - route_len * 0.10 - costmap_potential(fire_xy) * 0.18
            if score > best_score:
                best_score = score
                best = (fire_xy, route_len, max(0.0, min(1.0, quality)))
        return best

    def _fire_pose_rejected(self, fire_xy: tuple[float, float], target_xy: tuple[float, float]) -> bool:
        if not (-ARENA_SIZE * 0.5 + ROUTE_CLEARANCE <= fire_xy[0] <= ARENA_SIZE * 0.5 - ROUTE_CLEARANCE):
            return True
        if not (-ARENA_SIZE * 0.5 + ROUTE_CLEARANCE <= fire_xy[1] <= ARENA_SIZE * 0.5 - ROUTE_CLEARANCE):
            return True
        if point_blocked(fire_xy):
            return True
        if costmap_potential(fire_xy) > 3.0:
            return True
        shot_distance = math.hypot(target_xy[0] - fire_xy[0], target_xy[1] - fire_xy[1])
        if shot_distance > SHOOT_RANGE:
            return True
        if line_blocked_by_wall(fire_xy, target_xy):
            return True
        return False

    def _score_attack(self, candidate: dict[str, object], t: float) -> float:
        opponent = opponent_team(self.team)
        own_score = team_score(self.team)
        opponent_score = team_score(opponent)
        score_delta = own_score - opponent_score
        time_remaining = max(0.0, MATCH_DURATION_S - t)
        route_len = float(candidate["route_len"])
        shot_quality = float(candidate["shot_quality"])
        kind = str(candidate["kind"])
        aggression = 0.38
        if score_delta < 0:
            aggression += 0.28
        if time_remaining < 80.0:
            aggression += 0.22
        if time_remaining < 35.0:
            aggression += 0.22
        if len(BASE_ARMOR[self.team]) <= 2:
            aggression += 0.12

        if kind == f"base_{opponent}":
            if shot_quality < BASE_RUSH_MIN_QUALITY:
                return -60.0
            base_rush_risk = 3.0 * float(self.opponent_estimate["threat_to_own_base"]) if score_delta >= 0 else 0.0
            return 21.0 + 42.0 * shot_quality + 18.0 * aggression - 1.2 * route_len - base_rush_risk

        defense_risk = float(self.opponent_estimate["threat_to_own_base"]) * (7.0 if score_delta >= 0 else 3.0)
        return 5.0 + 8.0 * shot_quality - 1.5 * route_len + max(0.0, -score_delta) * 0.10 - defense_risk

    def _should_block(self, t: float) -> bool:
        opponent = opponent_team(self.team)
        time_remaining = MATCH_DURATION_S - t
        score_delta = team_score(self.team) - team_score(opponent)
        if score_delta >= BLOCK_LEAD_SCORE and time_remaining <= BLOCK_LATE_TIME_S:
            return True
        estimate = self.opponent_estimate
        if not estimate["available"] or score_delta < 5:
            return False
        threat = float(estimate["threat_to_own_base"])
        if threat >= OPPONENT_THREAT_BLOCK_THRESHOLD:
            return True
        return bool(estimate["visible"]) and float(estimate["distance_to_own_base"]) < 0.90

    def _select_block_point(self) -> tuple[float, float] | None:
        opponent_pose = self._opponent_pose()
        our_base = team_base_xy(self.team)
        if opponent_pose is not None and self.opponent_estimate["available"]:
            dx = float(self.opponent_estimate["dx"]) + self.pose[0][0] - our_base[0]
            dy = float(self.opponent_estimate["dy"]) + self.pose[0][1] - our_base[1]
            distance = max(1e-6, math.hypot(dx, dy))
            candidate = (our_base[0] + dx / distance * 0.72, our_base[1] + dy / distance * 0.72)
            if not point_blocked(candidate):
                return candidate

        fallback = (0.18, -0.18) if self.team == "yellow" else (-0.18, 0.18)
        if not point_blocked(fallback):
            return fallback
        return None

    def _opponent_pose(self) -> tuple[tuple[float, float, float], float] | None:
        controller = MATCH_CONTROLLERS.get(opponent_team(self.team))
        if controller is None:
            return None
        return controller.pose

    def _estimate_opponent(self) -> dict[str, float | bool]:
        opponent_pose = self._opponent_pose()
        if opponent_pose is None:
            return empty_opponent_estimate()
        return opponent_bearing_estimate(self.team, self.pose, opponent_pose)

    def _print_strategy(self, t: float, message: str):
        if t - self.last_strategy_print < 0.45:
            return
        self.last_strategy_print = t
        print(f"[STRATEGY]: {self.team} {message}.")

    def _spin_relocalize(self, dt: float):
        spin_direction = 1.0 if self.team == "yellow" else -1.0
        angular_velocity = spin_direction * 1.05
        self._integrate_differential(0.0, angular_velocity, dt)
        self.relocalize_rotation += abs(angular_velocity) * max(0.0, dt)
        self.localization_confidence = min(
            1.0,
            self.localization_confidence + LOCALIZATION_SPIN_GAIN * max(0.0, dt),
        )

    def _integrate_differential(self, linear_velocity: float, angular_velocity: float, dt: float):
        pos, yaw = self.pose
        if dt <= 0.0:
            self.last_linear_velocity = linear_velocity
            self.last_angular_velocity = angular_velocity
            return

        linear_velocity = max(-self.speed, min(self.speed, linear_velocity))
        angular_velocity = max(-2.4, min(2.4, angular_velocity))
        track_width = ROBOT_WIDTH + WHEEL_WIDTH
        desired_left_speed = linear_velocity - angular_velocity * track_width * 0.5
        desired_right_speed = linear_velocity + angular_velocity * track_width * 0.5
        desired_left_speed = max(-WHEEL_SPEED_LIMIT, min(WHEEL_SPEED_LIMIT, desired_left_speed))
        desired_right_speed = max(-WHEEL_SPEED_LIMIT, min(WHEEL_SPEED_LIMIT, desired_right_speed))
        left_speed = slew_rate(
            self.last_left_wheel_speed,
            desired_left_speed,
            min(LINEAR_ACCEL_LIMIT, WHEEL_ACCEL_LIMIT) * dt,
        )
        right_speed = slew_rate(
            self.last_right_wheel_speed,
            desired_right_speed,
            min(LINEAR_ACCEL_LIMIT, WHEEL_ACCEL_LIMIT) * dt,
        )
        linear_velocity = (left_speed + right_speed) * 0.5
        angular_velocity = (right_speed - left_speed) / track_width

        new_yaw = wrap_angle(yaw + angular_velocity * dt)
        mid_yaw = wrap_angle(yaw + angular_velocity * dt * 0.5)
        candidate = (
            pos[0] + linear_velocity * math.cos(mid_yaw) * dt,
            pos[1] + linear_velocity * math.sin(mid_yaw) * dt,
            0.0,
        )
        self.motion_blocked = False
        if segment_blocked((pos[0], pos[1]), (candidate[0], candidate[1])):
            warn_costmap(self.team, "collision sweep detected; holding pose and replanning")
            candidate = (pos[0], pos[1], 0.0)
            linear_velocity = 0.0
            left_speed = 0.0
            right_speed = 0.0
            self.motion_blocked = True

        corrected_xy, costmap_touch, hard_costmap_touch = apply_costmap_recovery((candidate[0], candidate[1]), self.team)
        if costmap_touch:
            correction_dx = corrected_xy[0] - candidate[0]
            correction_dy = corrected_xy[1] - candidate[1]
            correction_len = math.hypot(correction_dx, correction_dy)
            if correction_len > MAX_CONTACT_CORRECTION_STEP:
                scale = MAX_CONTACT_CORRECTION_STEP / correction_len
                corrected_xy = (candidate[0] + correction_dx * scale, candidate[1] + correction_dy * scale)
            candidate = (corrected_xy[0], corrected_xy[1], 0.0)
            self.motion_blocked = self.motion_blocked or hard_costmap_touch
            linear_velocity *= 0.35
            left_speed *= 0.35
            right_speed *= 0.35
        elif point_blocked((candidate[0], candidate[1])):
            safe_xy, _touched, _hard = apply_costmap_recovery((candidate[0], candidate[1]), self.team, passes=6)
            candidate = (safe_xy[0], safe_xy[1], 0.0)
            self.motion_blocked = True

        self.left_wheel_spin += left_speed * dt / WHEEL_RADIUS
        self.right_wheel_spin += right_speed * dt / WHEEL_RADIUS
        self.last_left_wheel_speed = left_speed
        self.last_right_wheel_speed = right_speed
        self.last_linear_velocity = linear_velocity
        self.last_angular_velocity = angular_velocity
        self.pose = (candidate, new_yaw)

    def _drive_is_stuck(self, t: float) -> bool:
        if self.motion_blocked:
            if self.blocked_since <= 0.0:
                self.blocked_since = t
            return t - self.blocked_since > 0.45
        self.blocked_since = 0.0

        if not self.path or self.waypoint_index >= len(self.path):
            return False
        pos, _ = self.pose
        waypoint = self.path[self.waypoint_index]
        distance = math.hypot(waypoint[0] - pos[0], waypoint[1] - pos[1])
        if distance < self.last_progress_distance - 0.018:
            self.last_progress_distance = distance
            self.last_progress_t = t
            return False
        if abs(self.last_linear_velocity) > 0.035 or abs(self.last_angular_velocity) > 0.20:
            return False
        if self.last_progress_t <= 0.0:
            self.last_progress_t = t
            self.last_progress_distance = distance
            return False
        return t - self.last_progress_t > 2.2

    def _drive_differential(self, dt: float) -> bool:
        if not self.path or self.waypoint_index >= len(self.path):
            self._integrate_differential(0.0, 0.0, dt)
            return True

        pos, yaw = self.pose
        while self.waypoint_index < len(self.path):
            waypoint = self.path[self.waypoint_index]
            if math.hypot(waypoint[0] - pos[0], waypoint[1] - pos[1]) > 0.075:
                break
            self.waypoint_index += 1

        if self.waypoint_index >= len(self.path):
            self._integrate_differential(0.0, 0.0, dt)
            return True

        waypoint = self.path[self.waypoint_index]
        desired_yaw = math.atan2(waypoint[1] - pos[1], waypoint[0] - pos[0])
        heading_error = wrap_angle(desired_yaw - yaw)
        angular_velocity = max(-2.4, min(2.4, 3.2 * heading_error))
        alignment = max(0.0, 1.0 - abs(heading_error) / 1.20)
        linear_velocity = self.speed * max(MIN_TURN_ALIGNMENT, alignment)
        if abs(heading_error) > 1.35:
            linear_velocity = 0.0
        estimate = self.opponent_estimate
        if (
            estimate["available"]
            and estimate["visible"]
            and float(estimate["distance"]) < OPPONENT_AVOID_RANGE
            and abs(float(estimate["relative_bearing"])) < OPPONENT_AVOID_BEARING_RAD
        ):
            avoid_turn = -1.0 if float(estimate["relative_bearing"]) >= 0.0 else 1.0
            linear_velocity = min(linear_velocity, self.speed * 0.18)
            angular_velocity = max(-2.4, min(2.4, angular_velocity + avoid_turn * 0.75))
        self._integrate_differential(linear_velocity, angular_velocity, dt)
        return False

    def _aim_differential(self, target_xy: tuple[float, float], dt: float) -> bool:
        pos, yaw = self.pose
        desired_yaw = math.atan2(target_xy[1] - pos[1], target_xy[0] - pos[0])
        heading_error = wrap_angle(desired_yaw - yaw)
        angular_velocity = max(-1.8, min(1.8, 3.8 * heading_error))
        if abs(heading_error) < math.radians(1.5):
            angular_velocity = 0.0
        self._integrate_differential(0.0, angular_velocity, dt)
        return abs(heading_error) < math.radians(4.0)


class PolicyReplayController(StrategyTeamController):
    """Motor-level replay of the learned high-level policy.

    The policy layer selects the next opponent target from a self-play style
    tactical sequence. The inherited controller still performs differential
    drive tracking, acceleration limiting, costmap avoidance, aiming, and
    shooter gating.
    """

    def _plan_path(self, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> list[tuple[float, float]]:
        staged = demo_policy_corridor(self.team, start_xy, goal_xy)
        path: list[tuple[float, float]] = [start_xy]
        cursor = start_xy
        for waypoint in staged[1:]:
            segment = plan_safe_path(cursor, waypoint)
            path.extend(segment[1:])
            cursor = waypoint
        return path

    def _select_strategy(self, t: float) -> dict[str, object]:
        start_xy = (self.pose[0][0], self.pose[0][1])
        first_deferred_base = ""
        cursor = self.task_index
        while cursor < len(self.tasks):
            target_name, _nominal_fire_xy = self.tasks[cursor]
            target_path = target_path_from_name(target_name)
            target = TARGET_REGISTRY.get(target_path)
            if target is None:
                if cursor == self.task_index:
                    self.task_index += 1
                cursor += 1
                continue
            if target["knocked"]:
                if cursor == self.task_index:
                    self.task_index += 1
                cursor += 1
                continue

            kind = str(target["kind"])
            opponent = opponent_team(self.team)
            if target["owner"] == self.team:
                warn_costmap(self.team, f"policy rejected illegal own target {target_name}")
                if cursor == self.task_index:
                    self.task_index += 1
                cursor += 1
                continue
            if kind == f"base_{opponent}" and len(BASE_ARMOR[opponent]) > 2:
                first_deferred_base = target_name
                cursor += 1
                continue

            solution = self._best_fire_solution(start_xy, target_path)
            if solution is None:
                warn_costmap(self.team, f"policy cannot find clean fire pose for {target_name}; trying next tactical option")
                cursor += 1
                continue

            fire_xy, route_len, shot_quality = solution
            if cursor != self.task_index:
                self._print_strategy(t, f"policy skipped {cursor - self.task_index} blocked option(s)")
            self.task_index = cursor
            self._print_strategy(
                t,
                f"policy action target={target_name} fire_pose=({fire_xy[0]:.2f}, {fire_xy[1]:.2f})",
            )
            return {
                "mode": "attack",
                "target_path": target_path,
                "fire_xy": fire_xy,
                "route_len": route_len,
                "shot_quality": shot_quality,
                "kind": kind,
            }
        if first_deferred_base:
            warn_costmap(self.team, f"base target {first_deferred_base} deferred until opponent armor is removed")
        return {"mode": "wait"}

    def _should_block(self, t: float) -> bool:
        return False


def initialize_match_controllers():
    if MATCH_CONTROLLERS:
        return
    for team, task_specs in MATCH_TASKS.items():
        for target_name, fire_xy in task_specs:
            target_path = target_path_from_name(target_name)
            if target_path not in TARGET_REGISTRY:
                raise RuntimeError(f"Match task target not found: {target_path}")
            owner = TARGET_REGISTRY[target_path]["owner"]
            if owner == team:
                raise RuntimeError(f"{team} task illegally targets its own target: {target_name}")
            if point_blocked(fire_xy):
                raise RuntimeError(f"Match task fire pose is blocked: {team} {target_name} {fire_xy}")

    MATCH_CONTROLLERS["yellow"] = StrategyTeamController(
        "yellow",
        YELLOW_START_XY,
        math.pi * 0.5,
        MATCH_TASKS["yellow"],
        MATCH_DRIVE_SPEED,
    )
    MATCH_CONTROLLERS["blue"] = StrategyTeamController(
        "blue",
        BLUE_START_XY,
        -math.pi * 0.5,
        MATCH_TASKS["blue"],
        MATCH_DRIVE_SPEED * 0.98,
    )


def initialize_demo_flow_controllers():
    if MATCH_CONTROLLERS:
        return
    for team, task_specs in DEMO_POLICY_TASKS.items():
        for target_name, fire_xy in task_specs:
            target_path = target_path_from_name(target_name)
            if target_path not in TARGET_REGISTRY:
                raise RuntimeError(f"Match task target not found: {target_path}")
            owner = TARGET_REGISTRY[target_path]["owner"]
            if owner == team:
                raise RuntimeError(f"{team} task illegally targets its own target: {target_name}")
            if point_blocked(fire_xy):
                warn_costmap("policy", f"nominal fire pose for {team}/{target_name} is occupied; live planner will choose another pose")

    MATCH_CONTROLLERS["yellow"] = PolicyReplayController(
        "yellow",
        YELLOW_DEMO_START_XY,
        math.pi * 0.5,
        DEMO_POLICY_TASKS["yellow"],
        MATCH_DRIVE_SPEED * 0.58,
    )
    MATCH_CONTROLLERS["blue"] = PolicyReplayController(
        "blue",
        BLUE_DEMO_START_XY,
        -math.pi * 0.5,
        DEMO_POLICY_TASKS["blue"],
        MATCH_DRIVE_SPEED * 0.56,
    )
    MATCH_CONTROLLERS["blue"].start_delay = 8.0


def design_arena():
    create_xform("/World/Arena")
    create_xform("/World/Targets")
    create_xform("/World/Targets/Fallen")
    create_xform("/World/Targets/Falling")
    create_xform("/World/Light")

    dome_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.82, 0.86, 0.92))
    dome_cfg.func("/World/Light/Dome", dome_cfg)
    distant_cfg = sim_utils.DistantLightCfg(intensity=2600.0, color=(1.0, 0.96, 0.88), angle=0.65)
    distant_cfg.func("/World/Light/Main", distant_cfg, translation=(1.5, -2.0, 4.0), orientation=quat_from_euler(-0.8, 0.3, 0.35))

    spawn_box(
        "/World/Arena/Floor",
        (ARENA_SIZE, ARENA_SIZE, 0.02),
        (0.0, 0.0, -0.01),
        (0.15, 0.17, 0.16),
        collision=True,
        raycast=True,
        semantic="arena_floor",
    )

    for idx, p in enumerate([-1.0, -0.5, 0.0, 0.5, 1.0]):
        spawn_box(f"/World/Arena/GridX_{idx}", (0.008, ARENA_SIZE, 0.004), (p, 0.0, 0.004), (0.32, 0.34, 0.32))
        spawn_box(f"/World/Arena/GridY_{idx}", (ARENA_SIZE, 0.008, 0.004), (0.0, p, 0.004), (0.32, 0.34, 0.32))

    zones = [
        ("BlueBase", BLUE_BASE_XY, (0.08, 0.25, 0.72), "blue_base_zone"),
        ("BlueStart", BLUE_START_XY, (0.12, 0.36, 0.90), "blue_start_zone"),
        ("YellowStart", YELLOW_START_XY, (0.95, 0.86, 0.08), "yellow_start_zone"),
        ("YellowBase", YELLOW_BASE_XY, (0.88, 0.78, 0.06), "yellow_base_zone"),
    ]
    for name, (x, y), color, semantic in zones:
        spawn_box(
            f"/World/Arena/{name}",
            (ZONE_SIZE, ZONE_SIZE, 0.006),
            (x, y, 0.006),
            color,
            semantic=semantic,
            opacity=0.86,
        )

    wall_color = (0.76, 0.78, 0.72)
    wall_z = WALL_HEIGHT * 0.5
    wall_span = ARENA_SIZE + WALL_THICKNESS * 2.0
    spawn_nav_blocker(
        "/World/Arena/WallWest",
        (WALL_THICKNESS, wall_span, WALL_HEIGHT),
        (-(ARENA_SIZE * 0.5 + WALL_THICKNESS * 0.5), 0.0, wall_z),
        wall_color,
        semantic="arena_wall",
    )
    spawn_nav_blocker(
        "/World/Arena/WallEast",
        (WALL_THICKNESS, wall_span, WALL_HEIGHT),
        ((ARENA_SIZE * 0.5 + WALL_THICKNESS * 0.5), 0.0, wall_z),
        wall_color,
        semantic="arena_wall",
    )
    spawn_nav_blocker(
        "/World/Arena/WallSouth",
        (wall_span, WALL_THICKNESS, WALL_HEIGHT),
        (0.0, -(ARENA_SIZE * 0.5 + WALL_THICKNESS * 0.5), wall_z),
        wall_color,
        semantic="arena_wall",
    )
    spawn_nav_blocker(
        "/World/Arena/WallNorth",
        (wall_span, WALL_THICKNESS, WALL_HEIGHT),
        (0.0, (ARENA_SIZE * 0.5 + WALL_THICKNESS * 0.5), wall_z),
        wall_color,
        semantic="arena_wall",
    )

    internal_specs = [
        ("MidWallWest", (-1.00, 0.0, wall_z), (1.00, WALL_THICKNESS, WALL_HEIGHT)),
        ("MidWallEast", (1.00, 0.0, wall_z), (1.00, WALL_THICKNESS, WALL_HEIGHT)),
        ("BlueBaseEastRail", (-1.00, 1.25, wall_z), (WALL_THICKNESS, 0.50, WALL_HEIGHT)),
        ("BlueBaseSouthRail", (-1.25, 1.00, wall_z), (0.50, WALL_THICKNESS, WALL_HEIGHT)),
        ("BlueStartEastRail", (0.00, 1.25, wall_z), (WALL_THICKNESS, 0.50, WALL_HEIGHT)),
        ("YellowStartWestRail", (0.00, -1.25, wall_z), (WALL_THICKNESS, 0.50, WALL_HEIGHT)),
        ("YellowBaseWestRail", (1.00, -1.25, wall_z), (WALL_THICKNESS, 0.50, WALL_HEIGHT)),
    ]
    for name, pos, size in internal_specs:
        spawn_nav_blocker(
            f"/World/Arena/{name}",
            size,
            pos,
            (0.70, 0.72, 0.67),
            semantic="internal_wall",
        )

    obstacles = [
        ("RandomObstacleNorthEast", (0.76, 0.68, OBSTACLE_SIZE * 0.5)),
        ("RandomObstacleSouthWest", (-0.74, -0.78, OBSTACLE_SIZE * 0.5)),
    ]
    for name, pos in obstacles:
        spawn_nav_blocker(
            f"/World/Arena/{name}",
            (OBSTACLE_SIZE, OBSTACLE_SIZE, OBSTACLE_SIZE),
            pos,
            (0.92, 0.05, 0.02),
            semantic="random_obstacle_30cm",
        )

    normal_targets = [
        ("T01_NorthMiddle", (0.00, 1.455), -math.pi * 0.5),
        ("T02_NorthEast", (1.455, 1.455), -math.pi * 0.5),
        ("T03_WestAboveGate", (-1.455, 0.12), 0.0),
        ("T04_WestBelowGate", (-1.455, -0.12), 0.0),
        ("T05_EastAboveGate", (1.455, 0.12), math.pi),
        ("T06_EastBelowGate", (1.455, -0.12), math.pi),
        ("T07_SouthWest", (-1.455, -1.455), math.pi * 0.5),
        ("T08_SouthMiddle", (0.00, -1.455), math.pi * 0.5),
    ]
    for name, xy, yaw in normal_targets:
        spawn_target(f"/World/Targets/{name}", xy, yaw, tag_id=1, frame_color=(0.25, 0.26, 0.25))

    spawn_target(
        "/World/Targets/BlueBaseTarget",
        BLUE_BASE_XY,
        -math.pi * 0.25,
        tag_id=3,
        pitch=math.radians(-45.0),
        frame_color=(0.08, 0.20, 0.56),
        base_target=True,
    )
    spawn_target(
        "/World/Targets/YellowBaseTarget",
        YELLOW_BASE_XY,
        math.pi * 0.75,
        tag_id=2,
        pitch=math.radians(-45.0),
        frame_color=(0.64, 0.48, 0.10),
        base_target=True,
    )
    spawn_base_armor("blue", BLUE_BASE_XY, (0.10, 0.34, 0.90))
    spawn_base_armor("yellow", YELLOW_BASE_XY, (0.90, 0.72, 0.12))

    validate_route("yellow", YELLOW_ROUTE)
    validate_route("blue", BLUE_ROUTE)
    spawn_route_markers("Yellow", YELLOW_ROUTE, (0.95, 0.86, 0.08))
    spawn_route_markers("Blue", BLUE_ROUTE, (0.12, 0.36, 0.90))


def design_robot(
    robot_path: str,
    start_xy: tuple[float, float],
    start_yaw: float,
    team_color: tuple[float, float, float],
    accent_color: tuple[float, float, float],
    beam_color: tuple[float, float, float],
):
    start_pose = (start_xy[0], start_xy[1], 0.0)
    create_xform(robot_path, translation=start_pose, orientation=quat_from_euler(0.0, 0.0, start_yaw))

    body_center_z = BASE_LINK_Z + ROBOT_BODY_HEIGHT * 0.5
    spawn_box(
        f"{robot_path}/base_link",
        (ROBOT_LENGTH, ROBOT_WIDTH, ROBOT_BODY_HEIGHT),
        (0.0, 0.0, body_center_z),
        (0.025, 0.028, 0.030),
        collision=True,
        semantic="robot_base_link",
    )
    spawn_box(
        f"{robot_path}/collision_hull",
        (ROBOT_LENGTH, ROBOT_WIDTH, ROBOT_BODY_HEIGHT * 0.82),
        (0.0, 0.0, body_center_z),
        team_color,
        collision=True,
        semantic="robot_collision_hull",
        opacity=0.18,
        rigid_body=True,
        kinematic=True,
        mass=8.0,
        disable_gravity=True,
    )
    spawn_box(
        f"{robot_path}/top_plate",
        (0.28, 0.18, 0.012),
        (0.0, 0.0, BASE_LINK_Z + ROBOT_BODY_HEIGHT + 0.018),
        team_color,
        semantic="robot_top_plate",
    )
    spawn_box(
        f"{robot_path}/front_bumper",
        (0.035, ROBOT_WIDTH + 0.03, 0.05),
        (ROBOT_LENGTH * 0.5 + 0.010, 0.0, BASE_LINK_Z + 0.045),
        accent_color,
        collision=True,
        semantic="robot_bumper",
    )
    spawn_box(
        f"{robot_path}/battery",
        (0.11, 0.10, 0.045),
        (-0.06, 0.0, BASE_LINK_Z + ROBOT_BODY_HEIGHT + 0.050),
        (0.16, 0.18, 0.18),
        semantic="battery_pack",
    )
    spawn_box(
        f"{robot_path}/imu_link",
        (0.044, 0.034, 0.014),
        IMU_POSE,
        (0.12, 0.72, 0.40),
        semantic="imu_9axis_module",
    )
    spawn_box(
        f"{robot_path}/imu_axis_x",
        (0.050, 0.004, 0.004),
        (IMU_POSE[0] + 0.025, IMU_POSE[1], IMU_POSE[2] + 0.012),
        (0.90, 0.05, 0.05),
        semantic="imu_x_axis",
    )
    spawn_box(
        f"{robot_path}/imu_axis_y",
        (0.004, 0.050, 0.004),
        (IMU_POSE[0], IMU_POSE[1] + 0.025, IMU_POSE[2] + 0.012),
        (0.05, 0.75, 0.08),
        semantic="imu_y_axis",
    )

    left_y = ROBOT_WIDTH * 0.5 + WHEEL_WIDTH * 0.5
    right_y = -left_y
    for name, y in [("left_wheel_link", left_y), ("right_wheel_link", right_y)]:
        spawn_cylinder(
            f"{robot_path}/{name}",
            radius=WHEEL_RADIUS,
            height=WHEEL_WIDTH,
            axis="Y",
            pos=(-0.03, y, WHEEL_RADIUS),
            color=(0.015, 0.015, 0.014),
            collision=True,
            semantic=name,
        )
        spawn_cylinder(
            f"{robot_path}/{name}_hub",
            radius=WHEEL_RADIUS * 0.55,
            height=WHEEL_WIDTH + 0.004,
            axis="Y",
            pos=(-0.03, y, WHEEL_RADIUS),
            color=(0.56, 0.58, 0.55),
            semantic="wheel_hub",
        )
        spawn_box(
            f"{robot_path}/{name}_stripe",
            (0.006, WHEEL_WIDTH + 0.006, WHEEL_RADIUS * 1.6),
            (-0.03 + WHEEL_RADIUS * 0.45, y, WHEEL_RADIUS),
            (0.92, 0.92, 0.86),
            semantic="wheel_rotation_mark",
        )

    for name, x in [("front_caster", 0.125), ("rear_caster", -0.145)]:
        spawn_cylinder(
            f"{robot_path}/{name}",
            radius=0.020,
            height=0.018,
            axis="Z",
            pos=(x, 0.0, 0.020),
            color=(0.09, 0.09, 0.085),
            collision=True,
            semantic=name,
        )

    spawn_cylinder(
        f"{robot_path}/laser_link",
        radius=0.035,
        height=0.035,
        axis="Z",
        pos=LIDAR_POSE,
        color=accent_color,
        collision=True,
        semantic="rplidar_frame",
    )
    spawn_cylinder(
        f"{robot_path}/lidar_cap",
        radius=0.030,
        height=0.008,
        axis="Z",
        pos=(LIDAR_POSE[0], LIDAR_POSE[1], LIDAR_POSE[2] + 0.022),
        color=(0.02, 0.02, 0.024),
        semantic="lidar_rotor",
    )
    spawn_box(
        f"{robot_path}/camera_link",
        (0.040, 0.030, 0.030),
        CAMERA_POSE,
        (0.015, 0.016, 0.018),
        collision=True,
        semantic="rgb_camera",
    )
    spawn_box(
        f"{robot_path}/depth_camera_link",
        (0.034, 0.024, 0.024),
        DEPTH_CAMERA_POSE,
        (0.025, 0.025, 0.030),
        collision=True,
        semantic="depth_camera",
    )
    spawn_cylinder(
        f"{robot_path}/camera_lens",
        radius=0.012,
        height=0.010,
        axis="X",
        pos=(CAMERA_POSE[0] + 0.025, CAMERA_POSE[1], CAMERA_POSE[2]),
        color=(0.03, 0.05, 0.07),
        semantic="camera_lens",
    )
    spawn_cylinder(
        f"{robot_path}/depth_camera_lens",
        radius=0.009,
        height=0.009,
        axis="X",
        pos=(DEPTH_CAMERA_POSE[0] + 0.023, DEPTH_CAMERA_POSE[1], DEPTH_CAMERA_POSE[2]),
        color=(0.02, 0.04, 0.07),
        semantic="depth_camera_lens",
    )
    for name, y in [("front_tof_left", 0.070), ("front_tof_right", -0.070)]:
        spawn_box(
            f"{robot_path}/{name}",
            (0.018, 0.018, 0.014),
            (TOF_FRONT_POSE[0], y, TOF_FRONT_POSE[2]),
            (0.04, 0.30, 0.36),
            semantic="tof_range_sensor",
        )
    for name, y in [("front_bumper_contact_left", 0.075), ("front_bumper_contact_right", -0.075)]:
        spawn_box(
            f"{robot_path}/{name}",
            (0.012, 0.055, 0.030),
            (ROBOT_LENGTH * 0.5 + 0.033, y, BASE_LINK_Z + 0.047),
            (0.86, 0.18, 0.10),
            semantic="bumper_contact_sensor",
        )
    for name, y in [("left_wheel_encoder", left_y), ("right_wheel_encoder", right_y)]:
        spawn_cylinder(
            f"{robot_path}/{name}",
            radius=0.018,
            height=0.006,
            axis="Y",
            pos=(-0.03, y * 0.90, WHEEL_RADIUS),
            color=(0.15, 0.12, 0.08),
            semantic="wheel_encoder",
        )
    spawn_cylinder(
        f"{robot_path}/shooter_link",
        radius=0.008,
        height=0.080,
        axis="X",
        pos=SHOOTER_POSE,
        color=beam_color,
        emissive=tuple(channel * 0.45 for channel in beam_color),
        collision=True,
        semantic="fixed_laser_shooter",
    )
    spawn_box(
        f"{robot_path}/laser_beam_preview",
        (1.05, 0.006, 0.006),
        (SHOOTER_POSE[0] + 0.55, SHOOTER_POSE[1], SHOOTER_POSE[2]),
        beam_color,
        opacity=0.34,
        emissive=beam_color,
        semantic="low_power_fixed_laser_beam",
    )

    # ROS/TF style sensor frames used by IsaacLab sensors.
    create_xform(f"{robot_path}/CameraFrame", translation=CAMERA_POSE)
    create_xform(f"{robot_path}/LidarFrame", translation=LIDAR_POSE)
    create_xform(f"{robot_path}/ImuFrame", translation=IMU_POSE)
    create_xform(f"{robot_path}/DepthCameraFrame", translation=DEPTH_CAMERA_POSE)

    # Frame axes at base_link: x red, y green, z blue.
    spawn_box(f"{robot_path}/tf_x_axis", (0.18, 0.008, 0.008), (0.09, 0.0, BASE_LINK_Z), (0.90, 0.05, 0.05), emissive=(0.25, 0.0, 0.0))
    spawn_box(f"{robot_path}/tf_y_axis", (0.008, 0.18, 0.008), (0.0, 0.09, BASE_LINK_Z), (0.05, 0.70, 0.08), emissive=(0.0, 0.20, 0.0))
    spawn_box(f"{robot_path}/tf_z_axis", (0.008, 0.008, 0.16), (0.0, 0.0, BASE_LINK_Z + 0.08), (0.08, 0.18, 0.90), emissive=(0.0, 0.0, 0.28))


def create_lidar_proxy_mesh() -> str:
    """Create a single static mesh for the IsaacLab RayCaster.

    This local IsaacLab build only supports one static mesh in RayCaster, so
    the field collision boxes are mirrored into one invisible mesh. The visible
    scene remains made from separate physical parts.
    """
    stage = get_current_stage()
    proxy_root = "/World/LidarProxy"
    proxy_mesh_path = f"{proxy_root}/static_arena_mesh"
    create_xform(proxy_root)

    vertices: list[Gf.Vec3f] = []
    indices: list[int] = []
    counts: list[int] = []

    corner_signs = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ]
    tri_faces = [
        (0, 2, 3),
        (0, 3, 1),
        (4, 5, 7),
        (4, 7, 6),
        (0, 1, 5),
        (0, 5, 4),
        (2, 6, 7),
        (2, 7, 3),
        (0, 4, 6),
        (0, 6, 2),
        (1, 3, 7),
        (1, 7, 5),
    ]

    for size, center, quat in RAYCAST_BOXES:
        base = len(vertices)
        half = (size[0] * 0.5, size[1] * 0.5, size[2] * 0.5)
        for sx, sy, sz in corner_signs:
            rotated = quat_rotate(quat, (sx * half[0], sy * half[1], sz * half[2]))
            vertices.append(Gf.Vec3f(center[0] + rotated[0], center[1] + rotated[1], center[2] + rotated[2]))
        for a, b, c in tri_faces:
            indices.extend([base + a, base + b, base + c])
            counts.append(3)

    mesh = UsdGeom.Mesh.Define(stage, proxy_mesh_path)
    mesh.CreatePointsAttr(vertices)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible()
    return proxy_root


def create_sensor_streams() -> dict[str, object]:
    if args_cli.no_sensor_streams or not args_cli.enable_sensor_streams:
        return {}

    camera_cfg = CameraCfg(
        prim_path=f"{PRIMARY_ROBOT_PATH}/CameraFrame/CameraSensor",
        update_period=1.0 / 15.0,
        height=720,
        width=1280,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.6,
            focus_distance=2.0,
            horizontal_aperture=4.8,
            clipping_range=(0.05, 6.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        semantic_segmentation_mapping={
            "class:arena_wall": (190, 190, 170, 255),
            "class:random_obstacle_30cm": (235, 132, 44, 255),
            "class:robot_base_link": (50, 50, 54, 255),
            "class:apriltag_visual": (20, 20, 20, 255),
            "class:arena_floor": (72, 78, 73, 255),
        },
    )
    camera = Camera(camera_cfg)

    lidar_proxy_path = create_lidar_proxy_mesh()
    lidar_cfg = RayCasterCfg(
        prim_path=f"{PRIMARY_ROBOT_PATH}/LidarFrame",
        update_period=1.0 / 12.0,
        mesh_prim_paths=[lidar_proxy_path],
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(-1.0, 1.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,
        ),
        max_distance=4.0,
        debug_vis=not args_cli.headless,
    )
    lidar = RayCaster(lidar_cfg)
    sensors: dict[str, object] = {"camera": camera, "lidar": lidar}
    try:
        from isaaclab.sensors.imu import Imu, ImuCfg

        imu_cfg = ImuCfg(
            prim_path=f"{PRIMARY_ROBOT_PATH}/ImuFrame",
            update_period=1.0 / 100.0,
            offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            gravity_bias=(0.0, 0.0, 9.81),
        )
        sensors["imu"] = Imu(imu_cfg)
    except Exception as exc:
        print(f"[WARN]: IsaacLab IMU stream unavailable ({exc}); visual/ROS IMU model is still present.")
    return sensors


def route_pose(
    t: float,
    route: list[tuple[float, float]],
    *,
    speed: float = 0.22,
) -> tuple[tuple[float, float, float], float]:
    segment_lengths = [
        math.hypot(route[i + 1][0] - route[i][0], route[i + 1][1] - route[i][1])
        for i in range(len(route) - 1)
    ]
    total_length = sum(segment_lengths)
    if total_length <= 1e-9:
        return (route[0][0], route[0][1], 0.0), 0.0

    travel = (t * speed) % (total_length * 2.0)
    reverse = travel > total_length
    distance = total_length * 2.0 - travel if reverse else travel

    walked = 0.0
    for index, length in enumerate(segment_lengths):
        if distance <= walked + length or index == len(segment_lengths) - 1:
            local = 0.0 if length <= 1e-9 else (distance - walked) / length
            eased = 0.5 - 0.5 * math.cos(max(0.0, min(1.0, local)) * math.pi)
            x0, y0 = route[index]
            x1, y1 = route[index + 1]
            x = x0 + (x1 - x0) * eased
            y = y0 + (y1 - y0) * eased
            yaw = math.atan2(y1 - y0, x1 - x0)
            if reverse:
                yaw += math.pi
            return (x, y, 0.0), yaw
        walked += length

    x, y = route[-1]
    return (x, y, 0.0), 0.0


def target_xy_for_name(target_name: str) -> tuple[float, float] | None:
    target = TARGET_REGISTRY.get(target_path_from_name(target_name))
    if target is None:
        return None
    xy = target["xy"]
    assert isinstance(xy, tuple)
    return xy


def finite_path_pose(
    path: list[tuple[float, float]],
    progress: float,
    fallback_yaw: float,
) -> tuple[tuple[float, float, float], float]:
    if len(path) < 2:
        x, y = path[0]
        return (x, y, 0.0), fallback_yaw

    segment_lengths = [
        math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]
    total_length = sum(segment_lengths)
    if total_length <= 1e-9:
        x, y = path[-1]
        return (x, y, 0.0), fallback_yaw

    distance = max(0.0, min(1.0, progress)) * total_length
    walked = 0.0
    for index, length in enumerate(segment_lengths):
        if distance <= walked + length or index == len(segment_lengths) - 1:
            local = 0.0 if length <= 1e-9 else (distance - walked) / length
            x0, y0 = path[index]
            x1, y1 = path[index + 1]
            x = x0 + (x1 - x0) * local
            y = y0 + (y1 - y0) * local
            yaw = math.atan2(y1 - y0, x1 - x0) if length > 1e-9 else fallback_yaw
            return (x, y, 0.0), yaw

    x, y = path[-1]
    return (x, y, 0.0), fallback_yaw


def demo_segment_path(team: str, segment_index: int, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> list[tuple[float, float]]:
    key = (team, segment_index)
    cached = DEMO_FLOW_PATH_CACHE.get(key)
    if cached is not None:
        return cached
    path = plan_safe_path(start_xy, goal_xy)
    validate_route(f"demo_{team}_{segment_index}", path)
    DEMO_FLOW_PATH_CACHE[key] = path
    return path


def demo_flow_pose(team: str, t: float) -> tuple[tuple[float, float, float], float]:
    script = DEMO_FLOW_POSES[team]
    if t <= script[0][0]:
        x, y = script[0][1]
        start_yaw = math.pi * 0.5 if team == "yellow" else -math.pi * 0.5
        return (x, y, 0.0), start_yaw

    for index in range(len(script) - 1):
        t0, xy0, look0 = script[index]
        t1, xy1, look1 = script[index + 1]
        if t <= t1:
            alpha = 0.0 if t1 <= t0 else max(0.0, min(1.0, (t - t0) / (t1 - t0)))
            eased = 0.5 - 0.5 * math.cos(alpha * math.pi)
            path = demo_segment_path(team, index, xy0, xy1)
            (x, y, _z), travel_yaw = finite_path_pose(path, eased, math.pi * 0.5 if team == "yellow" else -math.pi * 0.5)
            (x, y), _costmap_touch, _hard_touch = apply_costmap_recovery((x, y), f"demo_{team}")
            look_target = look1 or look0
            target_xy = target_xy_for_name(look_target) if look_target else None
            if target_xy is not None:
                yaw = math.atan2(target_xy[1] - y, target_xy[0] - x)
            else:
                yaw = travel_yaw
            for start, end in DEMO_FLOW_RECOVERY_WINDOWS:
                if start <= t <= end:
                    spin_sign = 1.0 if team == "yellow" else -1.0
                    yaw += spin_sign * (t - start) * 2.8
            return (x, y, 0.0), yaw

    x, y = script[-1][1]
    look_target = script[-1][2]
    target_xy = target_xy_for_name(look_target) if look_target else None
    if target_xy is not None:
        yaw = math.atan2(target_xy[1] - y, target_xy[0] - x)
    else:
        yaw = math.pi * 0.5 if team == "yellow" else -math.pi * 0.5
    return (x, y, 0.0), yaw


def trigger_demo_flow_events(t: float):
    for event_index, (event_time, team, target_name) in enumerate(DEMO_FLOW_FIRE_EVENTS):
        if event_index in DEMO_FLOW_TRIGGERED_EVENTS or t < event_time:
            continue
        if MATCH_STATE["winner"] is not None:
            DEMO_FLOW_TRIGGERED_EVENTS.add(event_index)
            continue
        target_path = target_path_from_name(target_name)
        if target_path not in TARGET_REGISTRY:
            raise RuntimeError(f"Demo flow target not found: {target_name}")
        LAST_FIRE_TIME[team] = t
        apply_fire_rule(team, target_path)
        DEMO_FLOW_TRIGGERED_EVENTS.add(event_index)


def update_demo_flow_animation(t: float) -> dict[str, tuple[tuple[float, float, float], float]]:
    initialize_demo_flow_controllers()
    yellow_pose = MATCH_CONTROLLERS["yellow"].update(t)
    blue_pose = MATCH_CONTROLLERS["blue"].update(t)
    yellow_pose, blue_pose = resolve_robot_contact(yellow_pose, blue_pose, t)
    MATCH_CONTROLLERS["yellow"].set_pose(yellow_pose)
    MATCH_CONTROLLERS["blue"].set_pose(blue_pose)
    poses = {"yellow": yellow_pose, "blue": blue_pose}

    for robot_path, team in ((YELLOW_ROBOT_PATH, "yellow"), (BLUE_ROBOT_PATH, "blue")):
        pos, yaw = poses[team]
        set_xform(robot_path, pos, quat_from_euler(0.0, 0.0, yaw))
        update_robot_parts(robot_path, team, t)
    return poses


def update_robot_parts(robot_path: str, team: str, t: float):
    controller = MATCH_CONTROLLERS.get(team)
    left_spin = controller.left_wheel_spin if controller else t * 7.0
    right_spin = controller.right_wheel_spin if controller else t * 7.0
    left_y = ROBOT_WIDTH * 0.5 + WHEEL_WIDTH * 0.5
    right_y = -left_y
    for side_y, side_name, wheel_spin in (
        (left_y, "left_wheel_link", left_spin),
        (right_y, "right_wheel_link", right_spin),
    ):
        set_xform(f"{robot_path}/{side_name}", (-0.03, side_y, WHEEL_RADIUS), quat_from_euler(0.0, wheel_spin, 0.0))
        set_xform(
            f"{robot_path}/{side_name}_hub",
            (-0.03, side_y, WHEEL_RADIUS),
            quat_from_euler(0.0, wheel_spin, 0.0),
        )
    set_xform(
        f"{robot_path}/lidar_cap",
        (LIDAR_POSE[0], LIDAR_POSE[1], LIDAR_POSE[2] + 0.022),
        quat_from_euler(0.0, 0.0, t * 12.0),
    )


def resolve_robot_contact(
    yellow_pose: tuple[tuple[float, float, float], float],
    blue_pose: tuple[tuple[float, float, float], float],
    t: float,
) -> tuple[tuple[tuple[float, float, float], float], tuple[tuple[float, float, float], float]]:
    yellow_pos, yellow_yaw = yellow_pose
    blue_pos, blue_yaw = blue_pose
    dx = blue_pos[0] - yellow_pos[0]
    dy = blue_pos[1] - yellow_pos[1]
    distance = math.hypot(dx, dy)
    min_distance = ROBOT_COLLISION_RADIUS * 2.0
    if distance >= min_distance:
        MATCH_STATE["robot_contact"] = False
        return yellow_pose, blue_pose

    if distance < 1e-6:
        nx, ny = 1.0, 0.0
    else:
        nx, ny = dx / distance, dy / distance
    push = (min_distance - max(distance, 1e-6)) * 0.5 + 0.004
    yellow_pos = (yellow_pos[0] - nx * push, yellow_pos[1] - ny * push, yellow_pos[2])
    blue_pos = (blue_pos[0] + nx * push, blue_pos[1] + ny * push, blue_pos[2])
    MATCH_STATE["robot_contact"] = True
    if "yellow" in MATCH_CONTROLLERS:
        MATCH_CONTROLLERS["yellow"].notify_contact(t)
    if "blue" in MATCH_CONTROLLERS:
        MATCH_CONTROLLERS["blue"].notify_contact(t)
    if t - float(MATCH_STATE["last_contact_print"]) > 2.0:
        MATCH_STATE["last_contact_print"] = t
        print("[RULE]: Robot contact resolved: yellow and blue collision hulls separated.")
    return (yellow_pos, yellow_yaw), (blue_pos, blue_yaw)


def update_robot_animation(t: float) -> dict[str, tuple[tuple[float, float, float], float]]:
    if args_cli.demo_flow:
        return update_demo_flow_animation(t)

    if args_cli.static_robot:
        poses = {
            "yellow": ((YELLOW_START_XY[0], YELLOW_START_XY[1], 0.0), math.pi * 0.5),
            "blue": ((BLUE_START_XY[0], BLUE_START_XY[1], 0.0), -math.pi * 0.5),
        }
    else:
        initialize_match_controllers()
        yellow_pose = MATCH_CONTROLLERS["yellow"].update(t)
        blue_pose = MATCH_CONTROLLERS["blue"].update(t)
        yellow_pose, blue_pose = resolve_robot_contact(yellow_pose, blue_pose, t)
        MATCH_CONTROLLERS["yellow"].set_pose(yellow_pose)
        MATCH_CONTROLLERS["blue"].set_pose(blue_pose)
        poses = {"yellow": yellow_pose, "blue": blue_pose}

    for robot_path, team in ((YELLOW_ROBOT_PATH, "yellow"), (BLUE_ROBOT_PATH, "blue")):
        pos, yaw = poses[team]
        set_xform(robot_path, pos, quat_from_euler(0.0, 0.0, yaw))
        update_robot_parts(robot_path, team, t)
    return poses


def line_blocked_by_wall(origin_xy: tuple[float, float], target_xy: tuple[float, float]) -> bool:
    for blocker_path, center, half_size in LASER_BLOCKERS:
        if segment_intersects_aabb(origin_xy, target_xy, center, half_size):
            return True
    return False


def detect_laser_hit(team: str, pose: tuple[tuple[float, float, float], float]) -> str | None:
    robot_pos, yaw = pose
    shooter_origin = local_to_world(robot_pos, SHOOTER_POSE, 0.0, 0.0, yaw)
    origin_xy = (shooter_origin[0], shooter_origin[1])
    forward = (math.cos(yaw), math.sin(yaw))
    best_path = None
    best_projection = SHOOT_RANGE + 1.0

    for target_path, target in TARGET_REGISTRY.items():
        if target["knocked"]:
            continue
        kind = str(target["kind"])
        target_xy = target["xy"]
        assert isinstance(target_xy, tuple)
        dx = target_xy[0] - origin_xy[0]
        dy = target_xy[1] - origin_xy[1]
        projection = dx * forward[0] + dy * forward[1]
        if projection <= 0.05 or projection > SHOOT_RANGE:
            continue
        perpendicular = abs(dx * forward[1] - dy * forward[0])
        hit_radius = BASE_HIT_RADIUS if kind.startswith("base_") else SHOOT_HIT_RADIUS
        if perpendicular > hit_radius:
            continue
        if line_blocked_by_wall(origin_xy, target_xy):
            continue
        if projection < best_projection:
            best_projection = projection
            best_path = target_path
    return best_path


def remove_next_armor(base_team: str):
    if not BASE_ARMOR[base_team]:
        return
    armor_path = BASE_ARMOR[base_team].pop(0)
    unregister_blocker(armor_path)
    start_pos, start_orient = get_xform(armor_path)
    removed_index = 4 - len(BASE_ARMOR[base_team])
    if base_team == "blue":
        end_pos = (BLUE_BASE_XY[0] - 0.18 + 0.055 * (removed_index - 1), BLUE_BASE_XY[1] - 0.62, 0.075)
    else:
        end_pos = (YELLOW_BASE_XY[0] + 0.18 + 0.055 * (removed_index - 1), YELLOW_BASE_XY[1] + 0.62, 0.075)
    ARMOR_REMOVALS.append(
        {
            "path": armor_path,
            "start_time": float(MATCH_STATE["current_time"]),
            "duration": 0.70,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "orientation": start_orient,
        }
    )
    print(f"[RULE]: {base_team} base armor removed, remaining={len(BASE_ARMOR[base_team])}.")


def knock_down_target(target_path: str):
    target = TARGET_REGISTRY[target_path]
    target["knocked"] = True
    set_visibility(str(target["path"]), False)
    set_visibility(str(target["fall_anim_path"]), True)
    TARGET_FALLS.append(
        {
            "target_path": target_path,
            "start_time": float(MATCH_STATE["current_time"]),
            "duration": 0.65,
        }
    )


def update_target_falls(t: float):
    for fall in list(TARGET_FALLS):
        target = TARGET_REGISTRY[str(fall["target_path"])]
        start_time = float(fall["start_time"])
        duration = float(fall["duration"])
        alpha = max(0.0, min(1.0, (t - start_time) / duration))
        eased = 0.5 - 0.5 * math.cos(alpha * math.pi)
        xy = target["xy"]
        yaw = float(target["yaw"])
        assert isinstance(xy, tuple)
        pitch = -math.radians(86.0) * eased
        set_xform(
            str(target["fall_anim_path"]),
            (xy[0], xy[1], 0.0),
            quat_from_euler(0.0, pitch, yaw),
        )
        if alpha >= 1.0:
            set_visibility(str(target["fall_anim_path"]), False)
            set_visibility(str(target["fallen_path"]), True)
            TARGET_FALLS.remove(fall)


def update_armor_removals(t: float):
    for removal in list(ARMOR_REMOVALS):
        start_time = float(removal["start_time"])
        duration = float(removal["duration"])
        alpha = max(0.0, min(1.0, (t - start_time) / duration))
        eased = 0.5 - 0.5 * math.cos(alpha * math.pi)
        start_pos = removal["start_pos"]
        end_pos = removal["end_pos"]
        orientation = removal["orientation"]
        assert isinstance(start_pos, tuple)
        assert isinstance(end_pos, tuple)
        assert isinstance(orientation, tuple)
        pos = (
            start_pos[0] + (end_pos[0] - start_pos[0]) * eased,
            start_pos[1] + (end_pos[1] - start_pos[1]) * eased,
            start_pos[2] + (end_pos[2] - start_pos[2]) * eased,
        )
        set_xform(str(removal["path"]), pos, orientation)
        if alpha >= 1.0:
            ARMOR_REMOVALS.remove(removal)


def apply_fire_rule(team: str, target_path: str) -> bool:
    target = TARGET_REGISTRY[target_path]
    kind = str(target["kind"])
    owner = str(target["owner"])
    opponent = "blue" if team == "yellow" else "yellow"

    if owner == team and kind == "normal":
        MATCH_STATE["last_event"] = f"{team} own-target shot blocked"
        print(f"[RULE]: {team} attempted to shoot own normal target {target_path.rsplit('/', 1)[-1]}; ignored.")
        return False

    if kind == "normal":
        knock_down_target(target_path)
        remove_next_armor(opponent)
        MATCH_STATE[f"score_{team}"] = int(MATCH_STATE[f"score_{team}"]) + 5
        MATCH_STATE["last_event"] = f"{team} hit normal target; {opponent} armor removed"
        print(f"[RULE]: {team} knocked normal target {target_path.rsplit('/', 1)[-1]}.")
        return True

    if kind == f"base_{opponent}":
        knock_down_target(target_path)
        MATCH_STATE[f"score_{team}"] = int(MATCH_STATE[f"score_{team}"]) + 60
        MATCH_STATE["winner"] = team
        MATCH_STATE["last_event"] = f"{team} hit {opponent} base target -> win"
        print(f"[RULE]: {team} knocked {opponent} base target. Match winner={team}.")
        return True
    if kind == f"base_{team}":
        knock_down_target(target_path)
        MATCH_STATE["winner"] = opponent
        MATCH_STATE["last_event"] = f"{team} hit own base target -> lose"
        print(f"[RULE]: {team} knocked its own base target. Match winner={opponent}.")
        return True
    return False


def apply_target_contact_rule(team: str, target_path: str):
    target = TARGET_REGISTRY[target_path]
    if target["knocked"]:
        return
    opponent = opponent_team(team)
    kind = str(target["kind"])
    target_name = target_path.rsplit("/", 1)[-1]
    knock_down_target(target_path)

    if kind == f"base_{team}":
        MATCH_STATE["winner"] = opponent
        MATCH_STATE["last_event"] = f"{team} collided with own base -> lose"
        print(f"[RULE]: {team} collided with own base target {target_name}. Match winner={opponent}.")
        return
    if kind.startswith("base_"):
        MATCH_STATE[f"score_{opponent}"] = int(MATCH_STATE[f"score_{opponent}"]) + 60
        MATCH_STATE["last_event"] = f"{team} collision knocked base; {opponent} scores"
        print(f"[RULE]: {team} collision knocked base target {target_name}; {opponent} receives 60 points.")
        return

    MATCH_STATE[f"score_{opponent}"] = int(MATCH_STATE[f"score_{opponent}"]) + 5
    MATCH_STATE["last_event"] = f"{team} collision knocked normal; {opponent} scores"
    print(f"[RULE]: {team} collision knocked normal target {target_name}; {opponent} receives 5 points.")


def update_target_contacts(robot_poses: dict[str, tuple[tuple[float, float, float], float]]):
    if MATCH_STATE["winner"] is not None:
        return
    for team, (robot_pos, _yaw) in robot_poses.items():
        robot_xy = (robot_pos[0], robot_pos[1])
        for target_path, target in TARGET_REGISTRY.items():
            if target["knocked"]:
                continue
            xy = target["xy"]
            kind = str(target["kind"])
            assert isinstance(xy, tuple)
            contact_radius = BASE_TARGET_CONTACT_RADIUS if kind.startswith("base_") else TARGET_CONTACT_RADIUS
            if math.hypot(robot_xy[0] - xy[0], robot_xy[1] - xy[1]) <= ROBOT_COLLISION_RADIUS + contact_radius:
                apply_target_contact_rule(team, target_path)
                if MATCH_STATE["winner"] is not None:
                    return


def update_match_rules(t: float, robot_poses: dict[str, tuple[tuple[float, float, float], float]]):
    if MATCH_STATE["winner"] is not None:
        return
    update_target_contacts(robot_poses)
    if MATCH_STATE["winner"] is not None:
        return
    for team, pose in robot_poses.items():
        if t - LAST_FIRE_TIME[team] < FIRE_COOLDOWN:
            continue
        target_path = detect_laser_hit(team, pose)
        if target_path is None:
            continue
        LAST_FIRE_TIME[team] = t
        apply_fire_rule(team, target_path)


def export_stage():
    output = Path(args_cli.save_usd) if args_cli.save_usd else Path(__file__).resolve().parent / "output" / "robocup_visionrl_arena.usd"
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = get_current_stage()
    stage.GetRootLayer().Export(str(output))
    print(f"[INFO]: Exported USD scene to {output}")


class MatchVideoRecorder:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.view = args_cli.record_view
        self.fps = max(1, int(args_cli.record_fps))
        self.width = max(320, int(args_cli.record_width))
        self.height = max(240, int(args_cli.record_height))
        self.panel_width = min(420, max(340, int(self.width * 0.26)))
        self.scene_width = max(480, self.width - self.panel_width)
        self.scene_height = self.height
        self.next_frame_time = 0.0
        self.frame_count = 0
        camera_prim_path = "/World/RecordingCamera"
        camera_offset = CameraCfg.OffsetCfg()
        camera_spawn = sim_utils.PinholeCameraCfg(
            focal_length=31.0,
            focus_distance=3.8,
            horizontal_aperture=24.0,
            clipping_range=(0.01, 100.0),
        )
        if self.view in ("yellow_pov", "blue_pov"):
            robot_path = YELLOW_ROBOT_PATH if self.view == "yellow_pov" else BLUE_ROBOT_PATH
            camera_prim_path = f"{robot_path}/PovRecordingCamera"
            camera_spawn = sim_utils.PinholeCameraCfg(
                focal_length=3.6,
                focus_distance=2.0,
                horizontal_aperture=4.8,
                clipping_range=(0.04, 6.0),
            )
            camera_offset = CameraCfg.OffsetCfg(
                pos=CAMERA_POSE,
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            )
        self.camera = Camera(
            CameraCfg(
                prim_path=camera_prim_path,
                update_period=0.0,
                height=self.scene_height,
                width=self.scene_width,
                data_types=["rgb"],
                spawn=camera_spawn,
                offset=camera_offset,
            )
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, float(self.fps), (self.width, self.height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {self.output_path}")

    def initialize_view(self):
        if self.view != "overview":
            return
        eye = torch.tensor([[2.15, -2.55, 2.28]], dtype=torch.float32, device=self.camera.device)
        target = torch.tensor([[0.0, 0.0, 0.12]], dtype=torch.float32, device=self.camera.device)
        self.camera.set_world_poses_from_view(eye, target)

    def capture(self, sim_dt: float, match_time: float):
        if match_time + 1e-6 < self.next_frame_time:
            return
        self.camera.update(dt=sim_dt)
        rgb = self.camera.data.output.get("rgb")
        if rgb is None:
            return
        frame = rgb[0].detach().cpu().numpy() if hasattr(rgb, "detach") else rgb[0]
        if frame.dtype != np.uint8:
            scale = 255.0 if frame.max() <= 1.0 else 1.0
            frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        scene_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if scene_bgr.shape[0] != self.scene_height or scene_bgr.shape[1] != self.scene_width:
            scene_bgr = cv2.resize(scene_bgr, (self.scene_width, self.scene_height), interpolation=cv2.INTER_AREA)
        output = self._compose_frame(scene_bgr, match_time)
        self.writer.write(output)
        self.frame_count += 1
        self.next_frame_time += 1.0 / float(self.fps)

    def _compose_frame(self, scene_bgr, match_time: float):
        frame = np.full((self.height, self.width, 3), 246, dtype=np.uint8)
        frame[:, : self.scene_width] = scene_bgr
        cv2.line(frame, (self.scene_width, 0), (self.scene_width, self.height), (190, 190, 190), 2)
        self._draw_side_panel(frame, match_time)
        return frame

    def _put(self, frame, text: str, x: int, y: int, scale: float, color=(35, 35, 35), thickness: int = 2):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def _put_wrapped(self, frame, text: str, x: int, y: int, max_chars: int, line_gap: int, scale: float):
        words = str(text).split()
        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if len(candidate) > max_chars:
                self._put(frame, line, x, y, scale, (40, 40, 40), 2)
                y += line_gap
                line = word
            else:
                line = candidate
        if line:
            self._put(frame, line, x, y, scale, (40, 40, 40), 2)
        return y

    def _view_title(self) -> str:
        if self.view == "yellow_pov":
            return "Yellow Robot POV"
        if self.view == "blue_pov":
            return "Blue Robot POV"
        return "Complete Arena View"

    def _opponent_summary(self, team: str) -> str:
        controller = MATCH_CONTROLLERS.get(team)
        if controller is None or not controller.opponent_estimate["available"]:
            return f"{team[0].upper()} track pending"
        estimate = controller.opponent_estimate
        opponent = opponent_team(team)
        visible = "vis" if estimate["visible"] else "occ"
        return (
            f"{team[0].upper()}->{opponent[0].upper()} "
            f"d {float(estimate['distance']):.2f}m "
            f"b {math.degrees(float(estimate['relative_bearing'])):+.0f}deg "
            f"{visible} th {float(estimate['threat_to_own_base']):.2f}"
        )

    def _draw_side_panel(self, frame, match_time: float):
        x0 = self.scene_width
        pad = 24
        panel_x = x0 + pad
        right = self.width - pad
        cv2.rectangle(frame, (x0, 0), (self.width, self.height), (247, 248, 250), -1)
        cv2.putText(
            frame,
            "RoboCup VisionRL",
            (panel_x, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
        self._put(frame, self._view_title(), panel_x, 88, 0.55, (70, 70, 70), 2)

        cv2.rectangle(frame, (panel_x, 124), (right, 248), (255, 255, 255), -1)
        cv2.rectangle(frame, (panel_x, 124), (right, 248), (216, 220, 226), 1)
        self._put(frame, "Score", panel_x + 16, 158, 0.58, (50, 50, 50), 2)
        self._put(frame, f"Yellow  {MATCH_STATE['score_yellow']}", panel_x + 18, 196, 0.68, (30, 130, 150), 2)
        self._put(frame, f"Blue    {MATCH_STATE['score_blue']}", panel_x + 18, 232, 0.68, (170, 80, 25), 2)

        cv2.rectangle(frame, (panel_x, 278), (right, 438), (255, 255, 255), -1)
        cv2.rectangle(frame, (panel_x, 278), (right, 438), (216, 220, 226), 1)
        self._put(frame, "Match State", panel_x + 16, 312, 0.58, (50, 50, 50), 2)
        self._put(frame, f"time   {match_time:05.1f}s", panel_x + 18, 348, 0.56, (40, 40, 40), 2)
        self._put(frame, f"armor  Y:{len(BASE_ARMOR['yellow'])}  B:{len(BASE_ARMOR['blue'])}", panel_x + 18, 382, 0.56, (40, 40, 40), 2)
        self._put(frame, self._opponent_summary("yellow"), panel_x + 18, 410, 0.42, (40, 40, 40), 1)
        self._put(frame, self._opponent_summary("blue"), panel_x + 18, 432, 0.42, (40, 40, 40), 1)

        cv2.rectangle(frame, (panel_x, 466), (right, 590), (255, 255, 255), -1)
        cv2.rectangle(frame, (panel_x, 466), (right, 590), (216, 220, 226), 1)
        self._put(frame, "Latest Event", panel_x + 16, 500, 0.58, (50, 50, 50), 2)
        self._put_wrapped(frame, str(MATCH_STATE["last_event"]), panel_x + 18, 538, 30, 30, 0.48)

        cv2.rectangle(frame, (panel_x, 620), (right, 780), (255, 255, 255), -1)
        cv2.rectangle(frame, (panel_x, 620), (right, 780), (216, 220, 226), 1)
        self._put(frame, "Rule Gate", panel_x + 16, 654, 0.58, (50, 50, 50), 2)
        self._put(frame, "opponent targets only", panel_x + 18, 692, 0.50, (24, 120, 78), 2)
        self._put(frame, "normal hit -> +5", panel_x + 18, 726, 0.50, (40, 40, 40), 2)
        self._put(frame, "base hit -> win", panel_x + 18, 760, 0.50, (40, 40, 40), 2)

        winner = MATCH_STATE["winner"]
        if winner is not None:
            cv2.rectangle(frame, (panel_x, self.height - 100), (right, self.height - 32), (42, 170, 74), -1)
            self._put(frame, f"WINNER: {str(winner).upper()}", panel_x + 18, self.height - 58, 0.64, (255, 255, 255), 2)

    def close(self):
        self.writer.release()
        print(f"[VIDEO]: Wrote {self.frame_count} frames to {self.output_path}")


def run_simulator(sim: sim_utils.SimulationContext, sensors: dict[str, object], recorder: MatchVideoRecorder | None):
    sim_dt = sim.get_physics_dt()
    start = time.perf_counter()
    count = 0
    last_print = -1.0

    try:
        while simulation_app.is_running():
            elapsed = time.perf_counter() - start
            MATCH_STATE["current_time"] = elapsed
            robot_poses = update_robot_animation(elapsed)
            update_target_contacts(robot_poses)
            update_armor_removals(elapsed)
            update_target_falls(elapsed)

            if MATCH_STATE["winner"] is None and elapsed >= MATCH_DURATION_S:
                yellow_score = int(MATCH_STATE["score_yellow"])
                blue_score = int(MATCH_STATE["score_blue"])
                if yellow_score > blue_score:
                    MATCH_STATE["winner"] = "yellow"
                elif blue_score > yellow_score:
                    MATCH_STATE["winner"] = "blue"
                else:
                    MATCH_STATE["winner"] = "draw"
                MATCH_STATE["last_event"] = f"time limit reached; winner={MATCH_STATE['winner']}"
                print(f"[RULE]: 3 minute time limit reached. winner={MATCH_STATE['winner']}.")

            sim.step()

            if "camera" in sensors:
                sensors["camera"].update(dt=sim_dt)
            if "lidar" in sensors:
                sensors["lidar"].update(dt=sim_dt, force_recompute=True)
            if "imu" in sensors:
                sensors["imu"].update(dt=sim_dt)
            if recorder is not None:
                recorder.capture(sim_dt, elapsed)

            if elapsed - last_print > 4.0:
                last_print = elapsed
                print("[INFO]: RoboCup VisionRL two-robot scene running")
                print(
                    "[SCORE]: "
                    f"blue_armor={len(BASE_ARMOR['blue'])} yellow_armor={len(BASE_ARMOR['yellow'])} "
                    f"yellow_score={MATCH_STATE['score_yellow']} blue_score={MATCH_STATE['score_blue']} "
                    f"winner={MATCH_STATE['winner']}"
                )
                if MATCH_CONTROLLERS:
                    print(
                        "[LOCALIZATION]: "
                        f"yellow_conf={MATCH_CONTROLLERS['yellow'].localization_confidence:.2f} "
                        f"blue_conf={MATCH_CONTROLLERS['blue'].localization_confidence:.2f}"
                    )
                    yellow_track = MATCH_CONTROLLERS["yellow"].opponent_estimate
                    blue_track = MATCH_CONTROLLERS["blue"].opponent_estimate
                    print(
                        "[OPPONENT_TRACK]: "
                        f"yellow_to_blue d={float(yellow_track['distance']):.2f}m "
                        f"bearing={math.degrees(float(yellow_track['relative_bearing'])):+.1f}deg "
                        f"visible={bool(yellow_track['visible'])} threat={float(yellow_track['threat_to_own_base']):.2f}; "
                        f"blue_to_yellow d={float(blue_track['distance']):.2f}m "
                        f"bearing={math.degrees(float(blue_track['relative_bearing'])):+.1f}deg "
                        f"visible={bool(blue_track['visible'])} threat={float(blue_track['threat_to_own_base']):.2f}"
                    )
                if "camera" in sensors and sensors["camera"].data.output:
                    rgb = sensors["camera"].data.output.get("rgb")
                    depth = sensors["camera"].data.output.get("distance_to_image_plane")
                    print(f"[INFO]: camera rgb={None if rgb is None else tuple(rgb.shape)} depth={None if depth is None else tuple(depth.shape)}")
                if "lidar" in sensors:
                    print(f"[INFO]: lidar rays={sensors['lidar'].num_rays} targets={len(COLLISION_PRIMS)}")
                if "imu" in sensors:
                    imu_data = getattr(sensors["imu"], "data", None)
                    print(f"[INFO]: imu stream={'ready' if imu_data is not None else 'pending'}")

            count += 1
            if args_cli.duration > 0.0 and elapsed >= args_cli.duration:
                break
    finally:
        if recorder is not None:
            recorder.close()


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.15, -3.65, 2.45], target=[0.0, 0.0, 0.20])

    design_arena()
    design_robot(
        YELLOW_ROBOT_PATH,
        YELLOW_START_XY,
        math.pi * 0.5,
        team_color=(0.95, 0.86, 0.08),
        accent_color=(0.64, 0.48, 0.10),
        beam_color=(1.0, 0.08, 0.02),
    )
    design_robot(
        BLUE_ROBOT_PATH,
        BLUE_START_XY,
        -math.pi * 0.5,
        team_color=(0.12, 0.36, 0.90),
        accent_color=(0.08, 0.20, 0.56),
        beam_color=(0.15, 0.42, 1.0),
    )
    export_stage()
    sensors = create_sensor_streams()
    recorder = MatchVideoRecorder(args_cli.record_video) if args_cli.record_video else None

    sim.reset()
    if recorder is not None:
        recorder.initialize_view()
    print("[INFO]: Setup complete. Close the Isaac Sim window to stop the scene.")
    print("[INFO]: Field: 3m x 3m, regulation-aligned bases/start zones, 0.5m walls, 0.3m obstacles.")
    print("[INFO]: Robots: yellow and blue 0.34m L x 0.24m W x 0.245m H, camera, 2D lidar, fixed laser module.")
    run_simulator(sim, sensors, recorder)


if __name__ == "__main__":
    main()
    if args_cli.headless and args_cli.duration > 0.0:
        print("[INFO]: Headless timed run complete; exiting process without Kit shutdown cleanup.")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    simulation_app.close()
