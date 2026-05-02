from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


ARENA_SIZE = 3.0
HALF_ARENA = ARENA_SIZE * 0.5
WALL_THICKNESS = 0.04
ZONE_SIZE = 0.50
OBSTACLE_SIZE = 0.30
ROBOT_LENGTH = 0.34
ROBOT_WIDTH = 0.24
ROBOT_RADIUS = math.hypot(ROBOT_LENGTH * 0.5, ROBOT_WIDTH * 0.5)
ROUTE_CLEARANCE = ROBOT_WIDTH * 0.5 + 0.04
SHOOT_RANGE = 1.65
SHOOT_HIT_RADIUS = 0.15
BASE_HIT_RADIUS = 0.20

YELLOW_START = np.array([0.25, -1.25, math.pi * 0.5], dtype=np.float32)
BLUE_START = np.array([-0.25, 1.25, -math.pi * 0.5], dtype=np.float32)
BLUE_BASE_XY = np.array([-1.25, 1.25], dtype=np.float32)
YELLOW_BASE_XY = np.array([1.25, -1.25], dtype=np.float32)

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
            low=np.full(26, -np.inf, dtype=np.float32),
            high=np.full(26, np.inf, dtype=np.float32),
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
            ((-1.00, 1.25), (WALL_THICKNESS, 0.50)),
            ((-1.25, 1.00), (0.50, WALL_THICKNESS)),
            ((0.00, 1.25), (WALL_THICKNESS, 0.50)),
            ((0.00, -1.25), (WALL_THICKNESS, 0.50)),
            ((1.00, -1.25), (WALL_THICKNESS, 0.50)),
            ((0.76, 0.68), (OBSTACLE_SIZE, OBSTACLE_SIZE)),
            ((-0.74, -0.78), (OBSTACLE_SIZE, OBSTACLE_SIZE)),
        ]
        return [(center, (size[0] * 0.5 + margin, size[1] * 0.5 + margin)) for center, size in raw]

    def _make_targets(self) -> list[Target]:
        return [
            Target("T01_NorthMiddle", (0.00, 1.455), -math.pi * 0.5, "normal", "blue"),
            Target("T02_NorthEast", (1.455, 1.455), -math.pi * 0.5, "normal", "blue"),
            Target("T03_WestAboveGate", (-1.455, 0.12), 0.0, "normal", "blue"),
            Target("T04_WestBelowGate", (-1.455, -0.12), 0.0, "normal", "yellow"),
            Target("T05_EastAboveGate", (1.455, 0.12), math.pi, "normal", "blue"),
            Target("T06_EastBelowGate", (1.455, -0.12), math.pi, "normal", "yellow"),
            Target("T07_SouthWest", (-1.455, -1.455), math.pi * 0.5, "normal", "yellow"),
            Target("T08_SouthMiddle", (0.00, -1.455), math.pi * 0.5, "normal", "yellow"),
            Target("BlueBaseTarget", tuple(BLUE_BASE_XY), -math.pi * 0.25, "base_blue", "blue"),
            Target("YellowBaseTarget", tuple(YELLOW_BASE_XY), math.pi * 0.75, "base_yellow", "yellow"),
        ]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.yellow = YELLOW_START.copy()
        self.blue = BLUE_START.copy()
        self.targets = self._make_targets()
        self.armor = {"yellow": 4, "blue": 4}
        self.elapsed = 0.0
        self.last_fire = {"yellow": -99.0, "blue": -99.0}
        self.winner: str | None = None
        self.last_contact = False
        self.localization_confidence = 1.0
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

        if action[2] > 0.25 and self.elapsed - self.last_fire["yellow"] > 1.0:
            self.last_fire["yellow"] = self.elapsed
            reward += self._apply_fire_rule("yellow", info)

        if self.elapsed - self.last_fire["blue"] > 1.4:
            target = self._detect_laser_hit("blue")
            if target is not None:
                self.last_fire["blue"] = self.elapsed
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
        if self._pose_blocked(candidate):
            return True
        self.yellow = candidate
        return False

    def _pose_blocked(self, pose: np.ndarray) -> bool:
        x, y = float(pose[0]), float(pose[1])
        for center, half_size in self.nav_blockers:
            if abs(x - center[0]) <= half_size[0] and abs(y - center[1]) <= half_size[1]:
                return True
        return False

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
        origin = (float(pose[0]), float(pose[1]))
        forward = (math.cos(float(pose[2])), math.sin(float(pose[2])))
        best_target = None
        best_projection = SHOOT_RANGE + 1.0
        for target in self.targets:
            if target.knocked:
                continue
            dx = target.xy[0] - origin[0]
            dy = target.xy[1] - origin[1]
            projection = dx * forward[0] + dy * forward[1]
            if projection <= 0.05 or projection > SHOOT_RANGE:
                continue
            hit_radius = BASE_HIT_RADIUS if target.kind.startswith("base_") else SHOOT_HIT_RADIUS
            perpendicular = abs(dx * forward[1] - dy * forward[0])
            if perpendicular > hit_radius:
                continue
            if self._line_blocked(origin, target.xy):
                continue
            if projection < best_projection:
                best_projection = projection
                best_target = target
        return best_target

    def _line_blocked(self, origin: tuple[float, float], target_xy: tuple[float, float]) -> bool:
        for center, half_size in self.laser_blockers:
            if segment_intersects_aabb(origin, target_xy, center, half_size):
                return True
        return False

    def _apply_fire_rule(self, team: str, info: dict[str, object]) -> float:
        target = self._detect_laser_hit(team)
        if target is None:
            return -0.05 if team == "yellow" else 0.0
        opponent = "blue" if team == "yellow" else "yellow"
        if target.kind == f"base_{team}":
            target.knocked = True
            self.winner = opponent
            info[f"{team}_own_base_hit"] = target.name
            return -80.0 if team == "yellow" else 0.0
        if target.owner == team:
            target.knocked = True
            info[f"{team}_own_target_hit"] = target.name
            return -8.0 if team == "yellow" else 0.0
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
        obs = np.concatenate(
            [
                np.array([self.yellow[0] / HALF_ARENA, self.yellow[1] / HALF_ARENA, math.cos(self.yellow[2]), math.sin(self.yellow[2])]),
                np.array([self.blue[0] / HALF_ARENA, self.blue[1] / HALF_ARENA, math.cos(self.blue[2]), math.sin(self.blue[2])]),
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
