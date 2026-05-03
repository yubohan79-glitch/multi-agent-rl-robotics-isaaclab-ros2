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
if args_cli.enable_sensor_streams and not args_cli.no_sensor_streams:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
from isaacsim.core.utils.stage import get_current_stage
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from pxr import Gf, Sdf, UsdGeom


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
}
MATCH_CONTROLLERS: dict[str, "StrategyTeamController"] = {}

YELLOW_ROBOT_PATH = "/World/RoboCupVisionRL_Yellow"
BLUE_ROBOT_PATH = "/World/RoboCupVisionRL_Blue"
PRIMARY_ROBOT_PATH = YELLOW_ROBOT_PATH

BLUE_BASE_XY = (-1.25, 1.25)
BLUE_START_XY = (-0.25, 1.25)
YELLOW_START_XY = (0.25, -1.25)
YELLOW_BASE_XY = (1.25, -1.25)
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


def opponent_team(team: str) -> str:
    return "blue" if team == "yellow" else "yellow"


def target_name_from_path(target_path: str) -> str:
    return target_path.rsplit("/", 1)[-1]


def team_base_xy(team: str) -> tuple[float, float]:
    return YELLOW_BASE_XY if team == "yellow" else BLUE_BASE_XY


def team_score(team: str) -> int:
    return int(MATCH_STATE[f"score_{team}"])


def static_fire_pose(team: str, target_name: str) -> tuple[float, float] | None:
    for candidate_name, fire_xy in MATCH_TASKS[team]:
        if candidate_name == target_name:
            return fire_xy
    return None


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
        f"{path}/white_substrate",
        (0.003, TAG_SIZE * 1.08, TAG_SIZE * 1.08),
        center,
        (0.96, 0.96, 0.90),
        orientation=orient,
        semantic=f"tag36h11_id_{tag_id}",
    )

    border = TAG_SIZE * 0.14
    half = TAG_SIZE * 0.5
    spawn_marker_cell(f"{path}/border_left", center, border, TAG_SIZE, -half + border * 0.5, 0.0, roll, pitch, yaw)
    spawn_marker_cell(f"{path}/border_right", center, border, TAG_SIZE, half - border * 0.5, 0.0, roll, pitch, yaw)
    spawn_marker_cell(f"{path}/border_top", center, TAG_SIZE, border, 0.0, half - border * 0.5, roll, pitch, yaw)
    spawn_marker_cell(f"{path}/border_bottom", center, TAG_SIZE, border, 0.0, -half + border * 0.5, roll, pitch, yaw)

    # Compact 4x4 visual code. It is not used for detection in this script; it
    # makes IDs 1, 2, and 3 visibly distinct in the rendered arena.
    patterns_by_id = {
        1: {(0, 0), (1, 1), (2, 2), (3, 3), (0, 3), (3, 0)},
        2: {(0, 1), (0, 2), (1, 0), (2, 3), (3, 1), (3, 2)},
        3: {(0, 0), (0, 3), (1, 1), (1, 2), (2, 1), (2, 2), (3, 0), (3, 3)},
    }
    cell = TAG_SIZE * 0.13
    pitch_between = TAG_SIZE * 0.16
    origin = -1.5 * pitch_between
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
    board_center = (xy[0], xy[1], 0.115)
    board_size = (0.012, 0.18, 0.19)

    spawn_box(
        f"{path}/target_board",
        board_size,
        board_center,
        (0.82, 0.84, 0.78),
        orientation=orient,
        collision=True,
        raycast=True,
        semantic=f"target_board_id_{tag_id}",
    )

    tag_center = local_to_world(board_center, (board_size[0] * 0.5 + 0.004, 0.0, TAG_CENTER_Z - board_center[2]), roll, pitch, yaw)
    spawn_apriltag(f"{path}/tag36h11_{tag_id}", tag_center, tag_id, roll, pitch, yaw)

    # A small base and rear support make each target a physical object instead
    # of a paper-like decal.
    support_center = local_to_world(board_center, (-0.035, 0.0, -0.09), roll, pitch, yaw)
    spawn_box(
        f"{path}/support_post",
        (0.025, 0.025, 0.18),
        support_center,
        frame_color,
        orientation=quat_from_euler(0.0, 0.0, yaw),
        collision=True,
        raycast=True,
        semantic="target_support",
    )
    foot_center = local_to_world(board_center, (-0.040, 0.0, -0.19), roll, pitch, yaw)
    spawn_box(
        f"{path}/target_foot",
        (0.12, 0.22, 0.018),
        (foot_center[0], foot_center[1], 0.009),
        frame_color,
        orientation=quat_from_euler(0.0, 0.0, yaw),
        collision=True,
        raycast=True,
        semantic="target_foot",
    )

    fallen_path = f"/World/Targets/Fallen/{path.rsplit('/', 1)[-1]}_fallen"
    fall_anim_path = f"/World/Targets/Falling/{path.rsplit('/', 1)[-1]}_falling"
    create_xform(fall_anim_path, translation=(xy[0], xy[1], 0.0), orientation=quat_from_euler(0.0, 0.0, yaw))
    spawn_box(
        f"{fall_anim_path}/target_board",
        board_size,
        (0.0, 0.0, 0.115),
        (0.86, 0.88, 0.80),
        semantic=f"falling_target_board_id_{tag_id}",
    )
    spawn_box(
        f"{fall_anim_path}/tag_patch",
        (0.006, 0.070, 0.070),
        (0.014, 0.0, TAG_CENTER_Z),
        (0.96, 0.96, 0.90),
        semantic=f"falling_tag36h11_id_{tag_id}",
    )
    spawn_box(
        f"{fall_anim_path}/sensor_box",
        (0.11, 0.16, 0.045),
        (-0.040, 0.0, 0.028),
        frame_color,
        semantic="falling_laser_target_base",
    )
    set_visibility(fall_anim_path, False)

    create_xform(fallen_path)
    spawn_box(
        f"{fallen_path}/board",
        (0.20, 0.16, 0.012),
        (xy[0], xy[1], 0.022),
        (0.72, 0.74, 0.68),
        orientation=quat_from_euler(0.0, 0.0, yaw),
        semantic=f"fallen_target_id_{tag_id}",
    )
    spawn_box(
        f"{fallen_path}/tag_patch",
        (0.070, 0.070, 0.004),
        (xy[0], xy[1], 0.032),
        (0.94, 0.94, 0.88),
        orientation=quat_from_euler(0.0, 0.0, yaw),
        semantic=f"fallen_tag36h11_id_{tag_id}",
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


def validate_route(name: str, route: list[tuple[float, float]]):
    for index, (p0, p1) in enumerate(zip(route, route[1:])):
        for blocker_path, center, half_size in NAV_BLOCKERS:
            if segment_intersects_aabb(p0, p1, center, half_size):
                raise RuntimeError(f"{name} route segment {index} intersects {blocker_path}")


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


def plan_safe_path(start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]]:
    if point_blocked(start):
        raise RuntimeError(f"Start point is blocked: {start}")
    if point_blocked(goal):
        raise RuntimeError(f"Goal point is blocked: {goal}")

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
            tentative_g = g_score[current] + math.hypot(dx, dy)
            if tentative_g >= g_score.get(nxt, float("inf")):
                continue
            came_from[nxt] = current
            g_score[nxt] = tentative_g
            f_score[nxt] = tentative_g + math.hypot(goal_cell[0] - nxt[0], goal_cell[1] - nxt[1])
            open_set.add(nxt)

    raise RuntimeError(f"No safe route found from {start} to {goal}")


def smooth_path(path: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(path) <= 2:
        return path
    smoothed = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        nxt = len(path) - 1
        while nxt > anchor + 1 and segment_blocked(path[anchor], path[nxt]):
            nxt -= 1
        smoothed.append(path[nxt])
        anchor = nxt
    return smoothed


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
        if MATCH_STATE["winner"] is not None:
            self.last_linear_velocity = 0.0
            self.last_angular_velocity = 0.0
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
                self.path = plan_safe_path(start_xy, block_xy)
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
            self.path = plan_safe_path(start_xy, fire_xy)
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
        fixed = static_fire_pose(self.team, target_name)
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
            score = quality - route_len * 0.10
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
            return 21.0 + 42.0 * shot_quality + 18.0 * aggression - 1.2 * route_len

        return 5.0 + 8.0 * shot_quality - 1.5 * route_len + max(0.0, -score_delta) * 0.10

    def _should_block(self, t: float) -> bool:
        opponent = opponent_team(self.team)
        time_remaining = MATCH_DURATION_S - t
        score_delta = team_score(self.team) - team_score(opponent)
        if score_delta >= BLOCK_LEAD_SCORE and time_remaining <= BLOCK_LATE_TIME_S:
            return True
        opponent_pose = self._opponent_pose()
        if opponent_pose is None or score_delta < 5:
            return False
        our_base = team_base_xy(self.team)
        opponent_xy = (opponent_pose[0][0], opponent_pose[0][1])
        return math.hypot(opponent_xy[0] - our_base[0], opponent_xy[1] - our_base[1]) < 0.90

    def _select_block_point(self) -> tuple[float, float] | None:
        opponent_pose = self._opponent_pose()
        our_base = team_base_xy(self.team)
        if opponent_pose is not None:
            opponent_xy = (opponent_pose[0][0], opponent_pose[0][1])
            dx = opponent_xy[0] - our_base[0]
            dy = opponent_xy[1] - our_base[1]
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

        linear_velocity = slew_rate(
            self.last_linear_velocity,
            linear_velocity,
            LINEAR_ACCEL_LIMIT * dt,
        )
        angular_velocity = slew_rate(
            self.last_angular_velocity,
            angular_velocity,
            ANGULAR_ACCEL_LIMIT * dt,
        )

        new_yaw = wrap_angle(yaw + angular_velocity * dt)
        mid_yaw = wrap_angle(yaw + angular_velocity * dt * 0.5)
        candidate = (
            pos[0] + linear_velocity * math.cos(mid_yaw) * dt,
            pos[1] + linear_velocity * math.sin(mid_yaw) * dt,
            0.0,
        )
        self.motion_blocked = False
        if point_blocked((candidate[0], candidate[1])):
            linear_velocity = 0.0
            candidate = pos
            new_yaw = wrap_angle(yaw + angular_velocity * dt)
            self.motion_blocked = True

        track_width = ROBOT_WIDTH + WHEEL_WIDTH
        left_speed = linear_velocity - angular_velocity * track_width * 0.5
        right_speed = linear_velocity + angular_velocity * track_width * 0.5
        self.left_wheel_spin += left_speed * dt / WHEEL_RADIUS
        self.right_wheel_spin += right_speed * dt / WHEEL_RADIUS
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
        linear_velocity = self.speed * (0.18 + 0.82 * alignment)
        if abs(heading_error) > 1.35:
            linear_velocity = 0.0
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
        print(f"[RULE]: {team} attempted to shoot own normal target {target_path.rsplit('/', 1)[-1]}; ignored.")
        return False

    if kind == "normal":
        knock_down_target(target_path)
        remove_next_armor(opponent)
        MATCH_STATE[f"score_{team}"] = int(MATCH_STATE[f"score_{team}"]) + 5
        print(f"[RULE]: {team} knocked normal target {target_path.rsplit('/', 1)[-1]}.")
        return True

    if kind == f"base_{opponent}":
        knock_down_target(target_path)
        MATCH_STATE[f"score_{team}"] = int(MATCH_STATE[f"score_{team}"]) + 60
        MATCH_STATE["winner"] = team
        print(f"[RULE]: {team} knocked {opponent} base target. Match winner={team}.")
        return True
    if kind == f"base_{team}":
        knock_down_target(target_path)
        MATCH_STATE["winner"] = opponent
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
        print(f"[RULE]: {team} collided with own base target {target_name}. Match winner={opponent}.")
        return
    if kind.startswith("base_"):
        MATCH_STATE[f"score_{opponent}"] = int(MATCH_STATE[f"score_{opponent}"]) + 60
        print(f"[RULE]: {team} collision knocked base target {target_name}; {opponent} receives 60 points.")
        return

    MATCH_STATE[f"score_{opponent}"] = int(MATCH_STATE[f"score_{opponent}"]) + 5
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


def run_simulator(sim: sim_utils.SimulationContext, sensors: dict[str, object]):
    sim_dt = sim.get_physics_dt()
    start = time.perf_counter()
    count = 0
    last_print = -1.0

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
            print(f"[RULE]: 3 minute time limit reached. winner={MATCH_STATE['winner']}.")

        sim.step()

        if "camera" in sensors:
            sensors["camera"].update(dt=sim_dt)
        if "lidar" in sensors:
            sensors["lidar"].update(dt=sim_dt, force_recompute=True)
        if "imu" in sensors:
            sensors["imu"].update(dt=sim_dt)

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

    sim.reset()
    print("[INFO]: Setup complete. Close the Isaac Sim window to stop the scene.")
    print("[INFO]: Field: 3m x 3m, regulation-aligned bases/start zones, 0.5m walls, 0.3m obstacles.")
    print("[INFO]: Robots: yellow and blue 0.34m L x 0.24m W x 0.245m H, camera, 2D lidar, fixed laser module.")
    run_simulator(sim, sensors)


if __name__ == "__main__":
    main()
    if args_cli.headless and args_cli.duration > 0.0:
        print("[INFO]: Headless timed run complete; exiting process without Kit shutdown cleanup.")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    simulation_app.close()
