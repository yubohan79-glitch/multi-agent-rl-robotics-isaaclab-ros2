from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:  # pragma: no cover - keeps rule-env smoke tests dependency-light.
    class _Env:
        def reset(self, seed: int | None = None):
            self.np_random = np.random.default_rng(seed)

    class _Box:
        def __init__(self, low, high, dtype=None):
            self.low = np.asarray(low, dtype=dtype)
            self.high = np.asarray(high, dtype=dtype)
            self.dtype = dtype
            self.shape = self.low.shape

        def sample(self):
            return np.zeros(self.shape, dtype=self.dtype or np.float32)

    class _Spaces:
        Box = _Box

    class _Gym:
        Env = _Env

    gym = _Gym()
    spaces = _Spaces()


ARENA_SIZE = 3.0
HALF_ARENA = ARENA_SIZE * 0.5
WALL_THICKNESS = 0.04
ZONE_SIZE = 0.50
OBSTACLE_SIZE = 0.30
PUSHABLE_OBSTACLE_HALF = OBSTACLE_SIZE * 0.5
# The rules state that two 30 cm cube obstacles are randomly placed.  The
# default deterministic layout follows the red obstacle centers measured from
# the national-rule field diagram; training can then jitter these references.
PUSHABLE_OBSTACLE_STARTS = {
    "box_ne": np.array([0.80, 0.80], dtype=np.float32),
    "box_sw": np.array([-0.80, -0.80], dtype=np.float32),
}
PUSHABLE_OBSTACLE_RANDOM_JITTER = 0.08
PUSHABLE_STEP_M = 0.060
PUSHABLE_CLEARANCE_MARGIN = 0.025
ROBOT_LENGTH = 0.34
ROBOT_WIDTH = 0.24
ROBOT_RADIUS = math.hypot(ROBOT_LENGTH * 0.5, ROBOT_WIDTH * 0.5)
ROBOT_PUSHABLE_CLEARANCE_RADIUS = ROBOT_RADIUS + 0.030
# Conservative visual/contact hull used for pushable red boxes.  It includes
# the rendered wheel/body footprint so videos and strict audits agree.
ROBOT_PUSHABLE_VISUAL_HALF_EXTENTS = (ROBOT_LENGTH * 0.5 + 0.110, ROBOT_WIDTH * 0.5 + 0.087)
ROUTE_CLEARANCE = ROBOT_WIDTH * 0.5 + 0.04
# Real-laser contract used by the RL rule environments and the IsaacLab replay.
# Distances are measured from the fixed shooter outlet, not from base_link.
# Normal targets remain a close 5-50 cm shot. Base targets are physically
# recessed behind armor, so the valid outlet-to-target range is wider but still
# bounded; line-of-sight through remaining armor is always checked separately.
NORMAL_SHOOT_MIN_RANGE = 0.05
NORMAL_SHOOT_RANGE = 0.50
NORMAL_SHOOT_IDEAL_DISTANCE = 0.30
BASE_SHOOT_MIN_RANGE = 0.20
BASE_SHOOT_RANGE = 0.80
BASE_SHOOT_IDEAL_DISTANCE = 0.48
SHOOT_MIN_RANGE = NORMAL_SHOOT_MIN_RANGE
SHOOT_RANGE = NORMAL_SHOOT_RANGE
SHOOT_IDEAL_DISTANCE = NORMAL_SHOOT_IDEAL_DISTANCE
SHOOTER_FORWARD_OFFSET = 0.20
SHOOT_HIT_RADIUS = 0.028
BASE_HIT_RADIUS = 0.018
NORMAL_TARGET_CONTACT_RADIUS = 0.035
BASE_TARGET_CONTACT_RADIUS = 0.045
LASER_DWELL_REQUIRED_S = 0.80
LASER_DWELL_FULL_CONFIDENCE_S = 2.00
LASER_FIRE_COOLDOWN_S = 1.0
TARGET_WALL_INSET = 0.240
TARGET_WALL_ANGLE_RAD = math.radians(45.0)
NORTH_MIDDLE_TARGET_X = 0.18
SOUTH_MIDDLE_TARGET_X = -0.18
SIDE_GATE_TARGET_Y = 0.24

YELLOW_START = np.array([0.25, -1.25, math.pi * 0.5], dtype=np.float32)
BLUE_START = np.array([-0.25, 1.25, -math.pi * 0.5], dtype=np.float32)
BLUE_BASE_XY = np.array([-1.25, 1.25], dtype=np.float32)
YELLOW_BASE_XY = np.array([1.25, -1.25], dtype=np.float32)
BLUE_BASE_TARGET_XY = np.array([-1.36, 1.36], dtype=np.float32)
YELLOW_BASE_TARGET_XY = np.array([1.36, -1.36], dtype=np.float32)
BLUE_BASE_TARGET_YAW = -math.pi / 4.0
YELLOW_BASE_TARGET_YAW = 3.0 * math.pi / 4.0
BASE_HIT_SUCCESS_BY_NORMAL_HITS = {
    0: 0.0,
    1: 0.40,
    2: 0.55,
    3: 0.80,
    4: 0.95,
}

BASE_ARMOR_SIZE = {
    "thickness": 0.050,
    "length": 0.250,
}
BASE_ARMOR_SPECS = {
    "blue": [
        ((-1.025, 1.375), (BASE_ARMOR_SIZE["thickness"], BASE_ARMOR_SIZE["length"])),
        ((-1.375, 1.025), (BASE_ARMOR_SIZE["length"], BASE_ARMOR_SIZE["thickness"])),
        ((-1.025, 1.125), (BASE_ARMOR_SIZE["thickness"], BASE_ARMOR_SIZE["length"])),
        ((-1.125, 1.025), (BASE_ARMOR_SIZE["length"], BASE_ARMOR_SIZE["thickness"])),
    ],
    "yellow": [
        ((1.025, -1.375), (BASE_ARMOR_SIZE["thickness"], BASE_ARMOR_SIZE["length"])),
        ((1.375, -1.025), (BASE_ARMOR_SIZE["length"], BASE_ARMOR_SIZE["thickness"])),
        ((1.025, -1.125), (BASE_ARMOR_SIZE["thickness"], BASE_ARMOR_SIZE["length"])),
        ((1.125, -1.025), (BASE_ARMOR_SIZE["length"], BASE_ARMOR_SIZE["thickness"])),
    ],
}

BLUE_ROUTE = [
    (-0.25, 1.25),
    (-0.25, 0.78),
    (-0.18, 0.22),
    (-0.18, -0.20),
    (-0.55, -0.20),
    (-0.95, -0.20),
    (-1.20, -0.22),
]


@dataclass
class Target:
    name: str
    xy: tuple[float, float]
    yaw: float
    kind: str
    owner: str
    knocked: bool = False


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def oriented_rect_aabb_collision(
    rect_center: tuple[float, float],
    yaw: float,
    rect_half: tuple[float, float],
    box_center: tuple[float, float],
    box_half: tuple[float, float],
) -> tuple[bool, tuple[float, float], float]:
    ux = (math.cos(yaw), math.sin(yaw))
    uy = (-math.sin(yaw), math.cos(yaw))
    delta = (rect_center[0] - box_center[0], rect_center[1] - box_center[1])
    axes = (ux, uy, (1.0, 0.0), (0.0, 1.0))
    best_axis = (1.0, 0.0)
    best_overlap = math.inf
    for axis in axes:
        rect_radius = rect_half[0] * abs(ux[0] * axis[0] + ux[1] * axis[1]) + rect_half[1] * abs(
            uy[0] * axis[0] + uy[1] * axis[1]
        )
        box_radius = box_half[0] * abs(axis[0]) + box_half[1] * abs(axis[1])
        distance = abs(delta[0] * axis[0] + delta[1] * axis[1])
        overlap = rect_radius + box_radius - distance
        if overlap <= 0.0:
            return False, (0.0, 0.0), 0.0
        if overlap < best_overlap:
            best_overlap = overlap
            sign = 1.0 if delta[0] * axis[0] + delta[1] * axis[1] >= 0.0 else -1.0
            best_axis = (axis[0] * sign, axis[1] * sign)
    norm = math.hypot(best_axis[0], best_axis[1])
    if norm <= 1e-8:
        return True, (1.0, 0.0), float(best_overlap)
    return True, (best_axis[0] / norm, best_axis[1] / norm), float(best_overlap)


def robot_pushable_collision(
    pose: np.ndarray,
    box_center: tuple[float, float],
    box_half: tuple[float, float] = (PUSHABLE_OBSTACLE_HALF, PUSHABLE_OBSTACLE_HALF),
) -> tuple[bool, tuple[float, float], float]:
    yaw = float(pose[2]) if pose.shape[0] >= 3 else 0.0
    return oriented_rect_aabb_collision(
        (float(pose[0]), float(pose[1])),
        yaw,
        ROBOT_PUSHABLE_VISUAL_HALF_EXTENTS,
        box_center,
        box_half,
    )


def angled_wall_target_yaw(wall_normal_yaw: float, sign: float) -> float:
    return wrap_angle(wall_normal_yaw + sign * TARGET_WALL_ANGLE_RAD)


def inward_45deg_target_yaws() -> dict[str, float]:
    # yaw is the target face normal. The target plane itself is yaw + 90 deg,
    # so each corner panel cuts the two wall planes at 45 deg.
    return {
        "T01_NorthMiddle": -math.pi / 4.0,
        "T02_NorthEast": -3.0 * math.pi / 4.0,
        "T03_WestAboveGate": math.pi / 4.0,
        "T04_WestBelowGate": -math.pi / 4.0,
        "T05_EastAboveGate": 3.0 * math.pi / 4.0,
        "T06_EastBelowGate": -3.0 * math.pi / 4.0,
        "T07_SouthWest": math.pi / 4.0,
        "T08_SouthMiddle": 3.0 * math.pi / 4.0,
    }


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


def route_pose(t: float, route: list[tuple[float, float]], speed: float = 0.22) -> np.ndarray:
    segment_lengths = [
        math.hypot(route[i + 1][0] - route[i][0], route[i + 1][1] - route[i][1])
        for i in range(len(route) - 1)
    ]
    total_length = sum(segment_lengths)
    travel = (t * speed) % (total_length * 2.0)
    reverse = travel > total_length
    distance = total_length * 2.0 - travel if reverse else travel
    walked = 0.0
    for index, length in enumerate(segment_lengths):
        if distance <= walked + length or index == len(segment_lengths) - 1:
            alpha = 0.0 if length <= 1e-9 else (distance - walked) / length
            eased = 0.5 - 0.5 * math.cos(max(0.0, min(1.0, alpha)) * math.pi)
            x0, y0 = route[index]
            x1, y1 = route[index + 1]
            yaw = math.atan2(y1 - y0, x1 - x0)
            if reverse:
                yaw += math.pi
            return np.array([x0 + (x1 - x0) * eased, y0 + (y1 - y0) * eased, wrap_angle(yaw)], dtype=np.float32)
        walked += length
    return np.array([route[-1][0], route[-1][1], 0.0], dtype=np.float32)


def laser_origin_from_pose(pose: np.ndarray) -> tuple[float, float]:
    yaw = float(pose[2])
    return (
        float(pose[0]) + SHOOTER_FORWARD_OFFSET * math.cos(yaw),
        float(pose[1]) + SHOOTER_FORWARD_OFFSET * math.sin(yaw),
    )


def shooting_range_limits(base_target: bool) -> tuple[float, float]:
    if base_target:
        return BASE_SHOOT_MIN_RANGE, BASE_SHOOT_RANGE
    return NORMAL_SHOOT_MIN_RANGE, NORMAL_SHOOT_RANGE


def ideal_shoot_distance(base_target: bool) -> float:
    return BASE_SHOOT_IDEAL_DISTANCE if base_target else NORMAL_SHOOT_IDEAL_DISTANCE


def laser_accuracy_from_geometry(distance: float, lateral_error: float, base_target: bool) -> float:
    min_range, max_range = shooting_range_limits(base_target)
    if distance < min_range or distance > max_range:
        return 0.0
    hit_radius = BASE_HIT_RADIUS if base_target else SHOOT_HIT_RADIUS
    if lateral_error > hit_radius:
        return 0.0
    distance_quality = (max_range - distance) / max(1e-6, max_range - min_range)
    lateral_quality = 1.0 - lateral_error / max(hit_radius, 1e-6)
    accuracy = 0.18 + 0.64 * distance_quality + 0.18 * lateral_quality
    if base_target:
        accuracy -= 0.10
    return float(np.clip(accuracy, 0.05, 0.98))


def laser_dwell_success_probability(dwell_s: float) -> float:
    if dwell_s + 1e-9 < LASER_DWELL_REQUIRED_S:
        return 0.0
    alpha = min(1.0, max(0.0, (dwell_s - LASER_DWELL_REQUIRED_S) / (LASER_DWELL_FULL_CONFIDENCE_S - LASER_DWELL_REQUIRED_S)))
    not_fall = 0.20 - 0.10 * alpha
    return float(np.clip(1.0 - not_fall, 0.0, 0.90))


def normalized_laser_dwell_factor(dwell_s: float) -> float:
    return laser_dwell_success_probability(dwell_s) / 0.90


def base_hit_success_cap(normal_hits: int) -> float:
    key = max(0, min(4, int(normal_hits)))
    return float(BASE_HIT_SUCCESS_BY_NORMAL_HITS[key])


def base_removed_side_lane_quality(normal_hits: int, base_xy: np.ndarray, xy: np.ndarray) -> float:
    """Score whether a base shot is taken from the side whose armor was removed.

    The four armor plates open the base progressively. A one-target early rush
    may only shoot through the first removed side; after two normal hits the
    second side is also allowed. This prevents far or arbitrary line-of-sight
    shots from counting as a legal base attack.
    """

    hits = max(0, min(4, int(normal_hits)))
    if hits <= 0:
        return 0.0
    if hits >= 4:
        return 1.0
    base = np.asarray(base_xy, dtype=np.float32)
    point = np.asarray(xy, dtype=np.float32)
    rel = point - base
    distance = float(np.linalg.norm(rel))
    if distance < 0.20:
        return 0.0
    unit = rel / max(distance, 1e-6)
    if float(base[0]) < 0.0:
        opened_dirs = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, -1.0], dtype=np.float32),
            np.array([1.0, -1.0], dtype=np.float32) / math.sqrt(2.0),
        ]
    else:
        opened_dirs = [
            np.array([-1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([-1.0, 1.0], dtype=np.float32) / math.sqrt(2.0),
        ]
    allowed = opened_dirs[:1] if hits == 1 else opened_dirs[:2] if hits == 2 else opened_dirs
    best_alignment = max(float(np.dot(unit, direction)) for direction in allowed)
    threshold = {1: 0.90, 2: 0.84, 3: 0.58}[hits]
    if best_alignment < threshold:
        return 0.0
    return float(np.clip(0.25 + 0.75 * (best_alignment - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0))


def base_attack_pose_quality(normal_hits: int, target_xy: tuple[float, float], target_yaw: float, base_xy: np.ndarray, xy: np.ndarray) -> float:
    hits = max(0, min(4, int(normal_hits)))
    if hits <= 0:
        return 0.0
    side_quality = base_removed_side_lane_quality(hits, base_xy, xy)
    if side_quality <= 0.0:
        return 0.0
    approach_yaw = math.atan2(float(xy[1]) - target_xy[1], float(xy[0]) - target_xy[0])
    off_axis = abs(wrap_angle(approach_yaw - float(target_yaw)))
    min_off_axis = {1: 0.62, 2: 0.42, 3: 0.18, 4: 0.0}[hits]
    max_off_axis = 2.55
    if off_axis < min_off_axis or off_axis > max_off_axis:
        return 0.0
    base_distance = float(np.linalg.norm(np.asarray(base_xy, dtype=np.float32) - np.asarray(xy, dtype=np.float32)))
    corner_radius = {1: 0.95, 2: 1.05, 3: 1.22, 4: 1.45}[hits]
    if base_distance > corner_radius:
        return 0.0
    angle_quality = (off_axis - min_off_axis) / max(max_off_axis - min_off_axis, 1e-6)
    corner_quality = 1.0 - base_distance / max(corner_radius, 1e-6)
    return float(np.clip((0.38 + 0.37 * angle_quality + 0.25 * corner_quality) * side_quality, 0.0, 1.0))


def active_base_armor_blockers(
    armor_remaining: dict[str, int],
    *,
    inflated: bool = False,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    margin = ROUTE_CLEARANCE + 0.045 if inflated else 0.0
    blockers: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for team, specs in BASE_ARMOR_SPECS.items():
        remaining = max(0, min(4, int(armor_remaining.get(team, 4))))
        for center, size in specs[4 - remaining :]:
            blockers.append((center, (size[0] * 0.5 + margin, size[1] * 0.5 + margin)))
    return blockers


class RoboCupVisionRLGymEnv(gym.Env):
    """Fast 2D rule environment for PPO before moving policies into IsaacLab.

    Action: [linear_velocity, angular_velocity, fire_gate], each in [-1, 1].
    Observation: normalized robot states, armor counts, target flags, nearest target vector, base vector.
    """

    metadata = {"render_modes": []}

    def __init__(self, dt: float = 0.10, max_time_s: float = 180.0):
        super().__init__()
        self.dt = dt
        self.max_time_s = max_time_s
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.full(31, -np.inf, dtype=np.float32),
            high=np.full(31, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.nav_blockers = self._make_blockers(inflated=True)
        self.laser_blockers = self._make_blockers(inflated=False)
        self.reset()

    def _make_blockers(self, *, inflated: bool) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        margin = ROUTE_CLEARANCE if inflated else 0.0
        wall_span = ARENA_SIZE + WALL_THICKNESS * 2.0
        raw = [
            ((-(HALF_ARENA + WALL_THICKNESS * 0.5), 0.0), (WALL_THICKNESS, wall_span)),
            (((HALF_ARENA + WALL_THICKNESS * 0.5), 0.0), (WALL_THICKNESS, wall_span)),
            ((0.0, -(HALF_ARENA + WALL_THICKNESS * 0.5)), (wall_span, WALL_THICKNESS)),
            ((0.0, (HALF_ARENA + WALL_THICKNESS * 0.5)), (wall_span, WALL_THICKNESS)),
            ((-1.00, 0.0), (1.00, WALL_THICKNESS)),
            ((1.00, 0.0), (1.00, WALL_THICKNESS)),
            ((0.00, 1.25), (WALL_THICKNESS, 0.50)),
            ((0.00, -1.25), (WALL_THICKNESS, 0.50)),
        ]
        return [(center, (size[0] * 0.5 + margin, size[1] * 0.5 + margin)) for center, size in raw]

    def _make_targets(self) -> list[Target]:
        n = HALF_ARENA - TARGET_WALL_INSET
        target_yaws = inward_45deg_target_yaws()
        return [
            Target("T01_NorthMiddle", (NORTH_MIDDLE_TARGET_X, n), target_yaws["T01_NorthMiddle"], "normal", "blue"),
            Target("T02_NorthEast", (n, n), target_yaws["T02_NorthEast"], "normal", "blue"),
            Target("T03_WestAboveGate", (-n, SIDE_GATE_TARGET_Y), target_yaws["T03_WestAboveGate"], "normal", "blue"),
            Target("T04_WestBelowGate", (-n, -SIDE_GATE_TARGET_Y), target_yaws["T04_WestBelowGate"], "normal", "yellow"),
            Target("T05_EastAboveGate", (n, SIDE_GATE_TARGET_Y), target_yaws["T05_EastAboveGate"], "normal", "blue"),
            Target("T06_EastBelowGate", (n, -SIDE_GATE_TARGET_Y), target_yaws["T06_EastBelowGate"], "normal", "yellow"),
            Target("T07_SouthWest", (-n, -n), target_yaws["T07_SouthWest"], "normal", "yellow"),
            Target("T08_SouthMiddle", (SOUTH_MIDDLE_TARGET_X, -n), target_yaws["T08_SouthMiddle"], "normal", "yellow"),
            Target("BlueBaseTarget", tuple(BLUE_BASE_TARGET_XY), BLUE_BASE_TARGET_YAW, "base_blue", "blue"),
            Target("YellowBaseTarget", tuple(YELLOW_BASE_TARGET_XY), YELLOW_BASE_TARGET_YAW, "base_yellow", "yellow"),
        ]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.yellow = YELLOW_START.copy()
        self.blue = BLUE_START.copy()
        self.targets = self._make_targets()
        self.pushable_obstacles = {
            name: value.copy() for name, value in PUSHABLE_OBSTACLE_STARTS.items()
        }
        self.armor = {"yellow": 4, "blue": 4}
        self.elapsed = 0.0
        self.last_fire = {"yellow": -99.0, "blue": -99.0}
        self.laser_locks = {
            "yellow": {"target": "", "start": -99.0},
            "blue": {"target": "", "start": -99.0},
        }
        self.winner: str | None = None
        self.last_contact = False
        self.last_shot_attempt: dict[str, dict[str, object]] = {"yellow": {}, "blue": {}}
        self.localization_confidence = 1.0
        self.rng = np.random.default_rng(seed)
        self._previous_blue_base_distance = float(np.linalg.norm(self.yellow[:2] - BLUE_BASE_XY))
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.elapsed += self.dt
        reward = -0.01
        info: dict[str, object] = {}

        previous_distance = float(np.linalg.norm(self.yellow[:2] - BLUE_BASE_XY))
        blocked = self._apply_yellow_action(action)
        if blocked:
            reward -= 0.4
            self.localization_confidence = max(0.05, self.localization_confidence - 0.16)

        self.blue = route_pose(self.elapsed, BLUE_ROUTE)
        if self._resolve_robot_contact():
            reward -= 1.0
            self.localization_confidence = max(0.05, self.localization_confidence - 0.38)
            info["robot_contact"] = True

        if self.localization_confidence < 0.62:
            spinning_in_place = abs(float(action[1])) > 0.62 and abs(float(action[0])) < 0.18
            if spinning_in_place:
                self.localization_confidence = min(1.0, self.localization_confidence + 0.14)
                reward += 0.35
                info["relocalizing"] = True
            else:
                reward -= 0.08

        if action[2] > 0.25 and self.elapsed - self.last_fire["yellow"] > LASER_FIRE_COOLDOWN_S:
            reward += self._apply_fire_rule("yellow", info)
        else:
            self._reset_laser_lock("yellow")

        if self.elapsed - self.last_fire["blue"] > 1.4:
            self._apply_fire_rule("blue", info)

        new_distance = float(np.linalg.norm(self.yellow[:2] - BLUE_BASE_XY))
        reward += 0.15 * (previous_distance - new_distance)
        terminated = self.winner is not None
        truncated = self.elapsed >= self.max_time_s
        if self.winner == "yellow":
            reward += 60.0
        elif self.winner == "blue":
            reward -= 45.0
        return self._get_obs(), float(reward), terminated, truncated, info

    def _apply_yellow_action(self, action: np.ndarray) -> bool:
        candidate = self.yellow.copy()
        linear_speed = 0.45 * float(action[0])
        angular_speed = 1.8 * float(action[1])
        candidate[2] = wrap_angle(float(candidate[2] + angular_speed * self.dt))
        candidate[0] += linear_speed * math.cos(float(candidate[2])) * self.dt
        candidate[1] += linear_speed * math.sin(float(candidate[2])) * self.dt
        if self._static_pose_blocked(candidate):
            return True
        if self._target_collision_name(candidate) is not None:
            return True
        obstacle_name = self._pushable_collision_name(candidate)
        if obstacle_name is not None:
            motion_yaw = float(candidate[2]) if linear_speed > 0.0 else wrap_angle(float(candidate[2]) + math.pi)
            if abs(linear_speed) <= 0.03 or not self._push_obstacle(obstacle_name, motion_yaw, candidate[:2]):
                return True
            if self._pushable_collision_name(candidate) is not None:
                return True
        self.yellow = candidate
        return False

    def _pose_blocked(self, pose: np.ndarray) -> bool:
        return (
            self._static_pose_blocked(pose)
            or self._pushable_collision_name(pose) is not None
            or self._target_collision_name(pose) is not None
        )

    def _static_pose_blocked(self, pose: np.ndarray) -> bool:
        x, y = float(pose[0]), float(pose[1])
        for center, half_size in self.nav_blockers:
            if abs(x - center[0]) <= half_size[0] and abs(y - center[1]) <= half_size[1]:
                return True
        return False

    def _pushable_collision_name(self, pose: np.ndarray) -> str | None:
        for name, center in self.pushable_obstacles.items():
            collided, _normal, _penetration = robot_pushable_collision(
                pose,
                (float(center[0]), float(center[1])),
            )
            if collided:
                return name
        return None

    def _pushable_position_valid(self, obstacle_name: str, xy: np.ndarray, robot_pose: np.ndarray) -> bool:
        limit = HALF_ARENA - PUSHABLE_OBSTACLE_HALF - PUSHABLE_CLEARANCE_MARGIN
        if abs(float(xy[0])) > limit or abs(float(xy[1])) > limit:
            return False
        inflated = PUSHABLE_OBSTACLE_HALF + PUSHABLE_CLEARANCE_MARGIN
        for center, half_size in self.nav_blockers:
            if abs(float(xy[0]) - center[0]) <= half_size[0] + inflated and abs(float(xy[1]) - center[1]) <= half_size[1] + inflated:
                return False
        for name, center in self.pushable_obstacles.items():
            if name == obstacle_name:
                continue
            if float(np.linalg.norm(xy - center)) < PUSHABLE_OBSTACLE_HALF * 2.0 + PUSHABLE_CLEARANCE_MARGIN:
                return False
        for target in self.targets:
            if target.knocked:
                continue
            radius = BASE_TARGET_CONTACT_RADIUS if target.kind.startswith("base_") else NORMAL_TARGET_CONTACT_RADIUS
            if float(np.linalg.norm(xy - np.asarray(target.xy, dtype=np.float32))) < inflated + radius:
                return False
        pose = robot_pose if robot_pose.shape[0] >= 3 else np.array([robot_pose[0], robot_pose[1], 0.0], dtype=np.float32)
        collided, _normal, _penetration = robot_pushable_collision(
            pose,
            (float(xy[0]), float(xy[1])),
        )
        return not collided

    def _target_collision_name(self, pose: np.ndarray) -> str | None:
        x, y = float(pose[0]), float(pose[1])
        for target in self.targets:
            if target.knocked:
                continue
            radius = BASE_TARGET_CONTACT_RADIUS if target.kind.startswith("base_") else NORMAL_TARGET_CONTACT_RADIUS
            if (target.xy[0] - x) ** 2 + (target.xy[1] - y) ** 2 <= (ROBOT_RADIUS + radius) ** 2:
                return target.name
        return None

    def _push_obstacle(self, obstacle_name: str, motion_yaw: float, robot_xy: np.ndarray) -> bool:
        current = self.pushable_obstacles[obstacle_name]
        direction = np.array([math.cos(motion_yaw), math.sin(motion_yaw)], dtype=np.float32)
        limit = HALF_ARENA - PUSHABLE_OBSTACLE_HALF - PUSHABLE_CLEARANCE_MARGIN
        accepted = None
        robot_pose = np.array([float(robot_xy[0]), float(robot_xy[1]), motion_yaw], dtype=np.float32)
        for multiplier in (1.0, 1.7, 2.4, 3.1, 4.0):
            candidate = current + direction * (PUSHABLE_STEP_M * multiplier)
            candidate = np.array(
                [
                    float(np.clip(candidate[0], -limit, limit)),
                    float(np.clip(candidate[1], -limit, limit)),
                ],
                dtype=np.float32,
            )
            if self._pushable_position_valid(obstacle_name, candidate, robot_pose):
                accepted = candidate
                break
        if accepted is None:
            return False
        self.pushable_obstacles[obstacle_name] = accepted
        return True

    def _resolve_robot_contact(self) -> bool:
        delta = self.blue[:2] - self.yellow[:2]
        distance = float(np.linalg.norm(delta))
        min_distance = ROBOT_RADIUS * 2.0
        if distance >= min_distance:
            self.last_contact = False
            return False
        normal = np.array([1.0, 0.0], dtype=np.float32) if distance < 1e-6 else delta / distance
        push = (min_distance - max(distance, 1e-6)) * 0.5 + 0.004
        self.yellow[:2] -= normal * push
        self.blue[:2] += normal * push
        self.last_contact = True
        return True

    def _detect_laser_hit(self, team: str) -> Target | None:
        pose = self.yellow if team == "yellow" else self.blue
        origin = laser_origin_from_pose(pose)
        forward = (math.cos(float(pose[2])), math.sin(float(pose[2])))
        best_target = None
        best_projection = max(SHOOT_RANGE, BASE_SHOOT_RANGE) + 1.0
        best_accuracy = 0.0
        best_lateral_error = 0.0
        own_candidate_projection = max(SHOOT_RANGE, BASE_SHOOT_RANGE) + 1.0
        for target in self.targets:
            if target.knocked:
                continue
            dx = target.xy[0] - origin[0]
            dy = target.xy[1] - origin[1]
            projection = dx * forward[0] + dy * forward[1]
            min_range, max_range = shooting_range_limits(target.kind.startswith("base_"))
            if projection < min_range or projection > max_range:
                continue
            hit_radius = BASE_HIT_RADIUS if target.kind.startswith("base_") else SHOOT_HIT_RADIUS
            perpendicular = abs(dx * forward[1] - dy * forward[0])
            if perpendicular > hit_radius:
                continue
            if self._line_blocked(origin, target.xy):
                continue
            if target.owner == team:
                own_candidate_projection = min(own_candidate_projection, projection)
                continue
            accuracy = laser_accuracy_from_geometry(projection, perpendicular, target.kind.startswith("base_"))
            if target.kind.startswith("base_"):
                opponent = "blue" if team == "yellow" else "yellow"
                normal_hits = max(0, 4 - int(self.armor[opponent]))
                base_xy = BLUE_BASE_XY if target.kind == "base_blue" else YELLOW_BASE_XY
                pose_quality = base_attack_pose_quality(normal_hits, target.xy, target.yaw, base_xy, pose[:2])
                if pose_quality <= 0.0:
                    continue
                accuracy = min(base_hit_success_cap(normal_hits), accuracy * pose_quality)
            if projection < best_projection:
                best_projection = projection
                best_target = target
                best_accuracy = accuracy
                best_lateral_error = perpendicular
        if own_candidate_projection <= best_projection:
            self._reset_laser_lock(team)
            self.last_shot_attempt[team] = {"hit": False, "reason": "own_target_safety_gate"}
            return None
        if best_target is None:
            self._reset_laser_lock(team)
            self.last_shot_attempt[team] = {"hit": False, "reason": "no_geometry"}
            return None
        dwell_s = self._update_laser_lock(team, best_target.name)
        if dwell_s + 1e-9 < LASER_DWELL_REQUIRED_S:
            self.last_shot_attempt[team] = {
                "hit": False,
                "reason": "dwell",
                "target": best_target.name,
                "dwell_s": round(float(dwell_s), 3),
                "required_s": LASER_DWELL_REQUIRED_S,
                "distance_m": round(float(best_projection), 4),
                "lateral_error_m": round(float(best_lateral_error), 4),
                "accuracy": round(float(best_accuracy), 4),
            }
            return None
        dwell_factor = normalized_laser_dwell_factor(dwell_s)
        final_accuracy = float(np.clip(best_accuracy * dwell_factor, 0.0, 0.95))
        hit = bool(self.rng.random() <= final_accuracy)
        self.last_shot_attempt[team] = {
            "hit": hit,
            "reason": "" if hit else "probabilistic_miss",
            "target": best_target.name,
            "dwell_s": round(float(dwell_s), 3),
            "distance_m": round(float(best_projection), 4),
            "lateral_error_m": round(float(best_lateral_error), 4),
            "geometry_accuracy": round(float(best_accuracy), 4),
            "dwell_factor": round(float(dwell_factor), 4),
            "accuracy": round(float(final_accuracy), 4),
        }
        self._reset_laser_lock(team)
        return best_target if hit else None

    def _update_laser_lock(self, team: str, target_name: str) -> float:
        lock = self.laser_locks[team]
        if lock["target"] != target_name:
            lock["target"] = target_name
            lock["start"] = self.elapsed
            return 0.0
        return max(0.0, self.elapsed - float(lock["start"]))

    def _reset_laser_lock(self, team: str):
        self.laser_locks[team]["target"] = ""
        self.laser_locks[team]["start"] = -99.0

    def _line_blocked(self, origin: tuple[float, float], target_xy: tuple[float, float]) -> bool:
        for center, half_size in self.laser_blockers:
            if segment_intersects_aabb(origin, target_xy, center, half_size):
                return True
        for center, half_size in active_base_armor_blockers(self.armor, inflated=False):
            if segment_intersects_aabb(origin, target_xy, center, half_size):
                return True
        for center in self.pushable_obstacles.values():
            if segment_intersects_aabb(
                origin,
                target_xy,
                (float(center[0]), float(center[1])),
                (PUSHABLE_OBSTACLE_HALF, PUSHABLE_OBSTACLE_HALF),
            ):
                return True
        return False

    def _opponent_tracking_features(self) -> np.ndarray:
        delta = self.blue[:2] - self.yellow[:2]
        distance = float(np.linalg.norm(delta))
        bearing = math.atan2(float(delta[1]), float(delta[0])) if distance > 1e-6 else float(self.yellow[2])
        relative_bearing = wrap_angle(bearing - float(self.yellow[2]))
        visible = 0.0 if self._line_blocked((float(self.yellow[0]), float(self.yellow[1])), (float(self.blue[0]), float(self.blue[1]))) else 1.0

        base_delta = YELLOW_BASE_XY - self.blue[:2]
        base_distance = float(np.linalg.norm(base_delta))
        base_bearing = math.atan2(float(base_delta[1]), float(base_delta[0])) if base_distance > 1e-6 else float(self.blue[2])
        heading_to_yellow_base = abs(wrap_angle(base_bearing - float(self.blue[2])))
        proximity_threat = max(0.0, 1.0 - base_distance / 1.10)
        heading_threat = max(0.0, 1.0 - heading_to_yellow_base / math.pi)
        threat = max(0.0, min(1.0, proximity_threat * (0.55 + 0.45 * heading_threat) * (1.0 if visible else 0.72)))

        return np.array(
            [
                distance / ARENA_SIZE,
                math.cos(relative_bearing),
                math.sin(relative_bearing),
                visible,
                threat,
            ],
            dtype=np.float32,
        )

    def _apply_fire_rule(self, team: str, info: dict[str, object]) -> float:
        target = self._detect_laser_hit(team)
        info[f"{team}_shot_attempt"] = self.last_shot_attempt.get(team, {})
        if target is None:
            reason = str(self.last_shot_attempt.get(team, {}).get("reason", ""))
            if reason == "dwell":
                return 0.015 if team == "yellow" else 0.0
            if reason == "probabilistic_miss":
                self.last_fire[team] = self.elapsed
            return -0.05 if team == "yellow" else 0.0
        self.last_fire[team] = self.elapsed
        opponent = "blue" if team == "yellow" else "yellow"
        if target.kind == f"base_{team}":
            info[f"{team}_own_base_blocked"] = target.name
            return -1.0 if team == "yellow" else 0.0
        if target.owner == team:
            info[f"{team}_own_target_blocked"] = target.name
            return -1.0 if team == "yellow" else 0.0
        if target.kind == "normal":
            target.knocked = True
            self.armor[opponent] = max(0, self.armor[opponent] - 1)
            info[f"{team}_hit"] = target.name
            return 6.0 if team == "yellow" else 0.0
        if target.kind == f"base_{opponent}":
            target.knocked = True
            self.winner = team
            info["winner"] = team
            return 70.0 if team == "yellow" else 0.0
        return -0.1 if team == "yellow" else 0.0

    def _get_obs(self) -> np.ndarray:
        normal_targets = [target for target in self.targets if target.kind == "normal"]
        active_normals = [target for target in normal_targets if not target.knocked and target.owner != "yellow"]
        if active_normals:
            nearest = min(active_normals, key=lambda t: np.linalg.norm(np.array(t.xy, dtype=np.float32) - self.yellow[:2]))
            nearest_vec = (np.array(nearest.xy, dtype=np.float32) - self.yellow[:2]) / ARENA_SIZE
        else:
            nearest_vec = np.zeros(2, dtype=np.float32)
        blue_base_vec = (BLUE_BASE_XY - self.yellow[:2]) / ARENA_SIZE
        knocked_flags = np.array([1.0 if target.knocked else 0.0 for target in normal_targets], dtype=np.float32)
        opponent_track = self._opponent_tracking_features()
        obs = np.concatenate(
            [
                np.array([self.yellow[0] / HALF_ARENA, self.yellow[1] / HALF_ARENA, math.cos(self.yellow[2]), math.sin(self.yellow[2])]),
                np.array([self.blue[0] / HALF_ARENA, self.blue[1] / HALF_ARENA, math.cos(self.blue[2]), math.sin(self.blue[2])]),
                opponent_track,
                np.array([self.armor["blue"] / 4.0, self.armor["yellow"] / 4.0, self.elapsed / self.max_time_s, float(self.last_contact)]),
                np.array([self.localization_confidence]),
                knocked_flags,
                nearest_vec,
                blue_base_vec,
                np.array([1.0 if self.winner == "yellow" else -1.0 if self.winner == "blue" else 0.0]),
            ]
        ).astype(np.float32)
        return obs


if __name__ == "__main__":
    env = RoboCupVisionRLGymEnv()
    obs, _ = env.reset(seed=7)
    for _ in range(16):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(f"reward={reward:.3f} done={terminated or truncated} info={info}")
        if terminated or truncated:
            break
