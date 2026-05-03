from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from robocup_visionrl_gym_env import (
    ARENA_SIZE,
    BASE_HIT_RADIUS,
    BLUE_BASE_XY,
    BLUE_START,
    HALF_ARENA,
    ROBOT_RADIUS,
    SHOOT_HIT_RADIUS,
    SHOOT_RANGE,
    RoboCupVisionRLGymEnv,
    YELLOW_BASE_XY,
    YELLOW_START,
    Target,
    segment_intersects_aabb,
    wrap_angle,
)


AGENTS = ("yellow", "blue")


@dataclass
class ShotResult:
    shooter: str
    target_name: str
    target_owner: str
    kind: str


class RoboCupVisionRLSelfPlayEnv:
    """Two-agent rule environment for MAPPO/self-play strategy training.

    The interface is intentionally lightweight: each step receives one action
    per team and returns dicts keyed by `yellow` and `blue`. A MAPPO trainer can
    wrap this class with its own vectorization/rollout adapter.
    """

    metadata = {"render_modes": []}

    def __init__(self, dt: float = 0.10, max_time_s: float = 180.0):
        self.dt = dt
        self.max_time_s = max_time_s
        self.action_spaces = {
            team: spaces.Box(
                low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            for team in AGENTS
        }
        self.observation_spaces = {
            team: spaces.Box(
                low=np.full(34, -np.inf, dtype=np.float32),
                high=np.full(34, np.inf, dtype=np.float32),
                dtype=np.float32,
            )
            for team in AGENTS
        }
        helper = RoboCupVisionRLGymEnv(dt=dt, max_time_s=max_time_s)
        self.nav_blockers = helper.nav_blockers
        self.laser_blockers = helper.laser_blockers
        self.reset()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.rng = np.random.default_rng(seed)
        self.poses = {
            "yellow": YELLOW_START.copy(),
            "blue": BLUE_START.copy(),
        }
        self.targets = RoboCupVisionRLGymEnv()._make_targets()
        self.armor = {"yellow": 4, "blue": 4}
        self.scores = {"yellow": 0, "blue": 0}
        self.elapsed = 0.0
        self.last_fire = {"yellow": -99.0, "blue": -99.0}
        self.last_contact = False
        self.localization_confidence = {"yellow": 1.0, "blue": 1.0}
        self.winner: str | None = None
        self.previous_base_distance = {
            "yellow": float(np.linalg.norm(self.poses["yellow"][:2] - BLUE_BASE_XY)),
            "blue": float(np.linalg.norm(self.poses["blue"][:2] - YELLOW_BASE_XY)),
        }
        return {team: self._obs(team) for team in AGENTS}, {team: {} for team in AGENTS}

    def step(self, actions: dict[str, np.ndarray]):
        rewards = {team: -0.01 for team in AGENTS}
        infos: dict[str, dict[str, object]] = {team: {} for team in AGENTS}
        self.elapsed += self.dt

        for team in AGENTS:
            previous = self._base_distance(team)
            blocked = self._apply_action(team, actions.get(team, np.zeros(3, dtype=np.float32)))
            rewards[team] += 0.15 * (previous - self._base_distance(team))
            if blocked:
                rewards[team] -= 0.35
                self.localization_confidence[team] = max(0.05, self.localization_confidence[team] - 0.16)
                infos[team]["blocked"] = True

        if self._resolve_contact():
            for team in AGENTS:
                contact_reward = self._contact_reward(team)
                rewards[team] += contact_reward
                self.localization_confidence[team] = max(0.05, self.localization_confidence[team] - 0.38)
                infos[team]["robot_contact"] = True

        for team in AGENTS:
            self._resolve_target_contacts(team, rewards, infos)

        for team in AGENTS:
            action = np.clip(np.asarray(actions.get(team, np.zeros(3, dtype=np.float32)), dtype=np.float32), -1.0, 1.0)
            if self.localization_confidence[team] < 0.62:
                spinning_in_place = abs(float(action[1])) > 0.62 and abs(float(action[0])) < 0.18
                if spinning_in_place:
                    self.localization_confidence[team] = min(1.0, self.localization_confidence[team] + 0.14)
                    rewards[team] += 0.35
                    infos[team]["relocalizing"] = True
                else:
                    rewards[team] -= 0.08

        for team in AGENTS:
            action = np.clip(np.asarray(actions.get(team, np.zeros(3, dtype=np.float32)), dtype=np.float32), -1.0, 1.0)
            if action[2] > 0.25 and self.elapsed - self.last_fire[team] > 1.0:
                self.last_fire[team] = self.elapsed
                result = self._apply_fire(team)
                if result is not None:
                    self._score_shot(result, rewards, infos)
                else:
                    rewards[team] -= 0.08

        terminated = self.winner is not None
        truncated = self.elapsed >= self.max_time_s
        if truncated and self.winner is None:
            if self.scores["yellow"] > self.scores["blue"]:
                self.winner = "yellow"
            elif self.scores["blue"] > self.scores["yellow"]:
                self.winner = "blue"
            else:
                self.winner = "draw"

        if self.winner in AGENTS:
            loser = self._opponent(self.winner)
            rewards[self.winner] += 60.0
            rewards[loser] -= 45.0
        observations = {team: self._obs(team) for team in AGENTS}
        terminations = {team: terminated for team in AGENTS}
        truncations = {team: truncated for team in AGENTS}
        return observations, rewards, terminations, truncations, infos

    def _apply_action(self, team: str, action: np.ndarray) -> bool:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        pose = self.poses[team].copy()
        linear_speed = 0.42 * float(action[0])
        angular_speed = 1.65 * float(action[1])
        pose[2] = wrap_angle(float(pose[2] + angular_speed * self.dt))
        pose[0] += linear_speed * math.cos(float(pose[2])) * self.dt
        pose[1] += linear_speed * math.sin(float(pose[2])) * self.dt
        if self._pose_blocked(pose):
            return True
        self.poses[team] = pose
        return False

    def _pose_blocked(self, pose: np.ndarray) -> bool:
        x, y = float(pose[0]), float(pose[1])
        for center, half_size in self.nav_blockers:
            if abs(x - center[0]) <= half_size[0] and abs(y - center[1]) <= half_size[1]:
                return True
        return False

    def _resolve_contact(self) -> bool:
        delta = self.poses["blue"][:2] - self.poses["yellow"][:2]
        distance = float(np.linalg.norm(delta))
        min_distance = ROBOT_RADIUS * 2.0
        if distance >= min_distance:
            self.last_contact = False
            return False
        normal = np.array([1.0, 0.0], dtype=np.float32) if distance < 1e-6 else delta / distance
        push = (min_distance - max(distance, 1e-6)) * 0.5 + 0.004
        self.poses["yellow"][:2] -= normal * push
        self.poses["blue"][:2] += normal * push
        self.last_contact = True
        return True

    def _contact_reward(self, team: str) -> float:
        opponent = self._opponent(team)
        own_base = YELLOW_BASE_XY if team == "yellow" else BLUE_BASE_XY
        opponent_distance_to_own_base = float(np.linalg.norm(self.poses[opponent][:2] - own_base))
        own_distance_to_own_base = float(np.linalg.norm(self.poses[team][:2] - own_base))
        if opponent_distance_to_own_base < 0.85 and own_distance_to_own_base > 0.45:
            return 0.18
        if own_distance_to_own_base < 0.55:
            return -0.45
        return -0.04

    def _resolve_target_contacts(
        self,
        team: str,
        rewards: dict[str, float],
        infos: dict[str, dict[str, object]],
    ):
        if self.winner is not None:
            return
        opponent = self._opponent(team)
        pose_xy = self.poses[team][:2]
        for target in self.targets:
            if target.knocked:
                continue
            target_radius = 0.18 if target.kind.startswith("base_") else 0.115
            distance = float(np.linalg.norm(np.array(target.xy, dtype=np.float32) - pose_xy))
            if distance > ROBOT_RADIUS + target_radius:
                continue
            target.knocked = True
            infos[team]["target_collision"] = target.name
            if target.kind == f"base_{team}":
                self.winner = opponent
                rewards[team] -= 70.0
                rewards[opponent] += 60.0
                infos[team]["own_base_collision"] = True
                return
            if target.kind.startswith("base_"):
                self.scores[opponent] += 60
                rewards[team] -= 20.0
                rewards[opponent] += 20.0
                return
            self.scores[opponent] += 5
            rewards[team] -= 6.0
            rewards[opponent] += 4.0

    def _apply_fire(self, team: str) -> ShotResult | None:
        target = self._detect_laser_hit(team)
        if target is None:
            return None
        return ShotResult(team, target.name, target.owner, target.kind)

    def _detect_laser_hit(self, team: str) -> Target | None:
        pose = self.poses[team]
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

    def _opponent_tracking_features(self, team: str, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        own_base = YELLOW_BASE_XY if team == "yellow" else BLUE_BASE_XY
        delta = other[:2] - own[:2]
        distance = float(np.linalg.norm(delta))
        bearing = math.atan2(float(delta[1]), float(delta[0])) if distance > 1e-6 else float(own[2])
        relative_bearing = wrap_angle(bearing - float(own[2]))
        visible = 0.0 if self._line_blocked((float(own[0]), float(own[1])), (float(other[0]), float(other[1]))) else 1.0

        base_delta = own_base - other[:2]
        base_distance = float(np.linalg.norm(base_delta))
        base_bearing = math.atan2(float(base_delta[1]), float(base_delta[0])) if base_distance > 1e-6 else float(other[2])
        heading_to_own_base = abs(wrap_angle(base_bearing - float(other[2])))
        proximity_threat = max(0.0, 1.0 - base_distance / 1.10)
        heading_threat = max(0.0, 1.0 - heading_to_own_base / math.pi)
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

    def _score_shot(
        self,
        result: ShotResult,
        rewards: dict[str, float],
        infos: dict[str, dict[str, object]],
    ):
        shooter = result.shooter
        opponent = self._opponent(shooter)
        target = next(t for t in self.targets if t.name == result.target_name)
        target.knocked = True

        if result.target_owner == shooter and result.kind == f"base_{shooter}":
            self.winner = opponent
            rewards[shooter] -= 80.0
            rewards[opponent] += 60.0
            infos[shooter]["own_base_hit"] = True
            return
        if result.target_owner == shooter:
            rewards[shooter] -= 8.0
            rewards[opponent] += 4.0
            infos[shooter]["own_target_hit"] = result.target_name
            return
        if result.kind == "normal":
            self.armor[opponent] = max(0, self.armor[opponent] - 1)
            self.scores[shooter] += 5
            rewards[shooter] += 6.0
            rewards[opponent] -= 2.0
            infos[shooter]["hit"] = result.target_name
            return
        if result.kind == f"base_{opponent}":
            self.scores[shooter] += 60
            self.winner = shooter
            infos[shooter]["winner"] = shooter

    def _obs(self, team: str) -> np.ndarray:
        opponent = self._opponent(team)
        own = self.poses[team]
        other = self.poses[opponent]
        opponent_base = BLUE_BASE_XY if team == "yellow" else YELLOW_BASE_XY
        own_base = YELLOW_BASE_XY if team == "yellow" else BLUE_BASE_XY
        normal_targets = [target for target in self.targets if target.kind == "normal"]
        active_opponent_normals = [
            target for target in normal_targets if not target.knocked and target.owner == opponent
        ]
        if active_opponent_normals:
            nearest = min(
                active_opponent_normals,
                key=lambda target: np.linalg.norm(np.array(target.xy, dtype=np.float32) - own[:2]),
            )
            nearest_vec = (np.array(nearest.xy, dtype=np.float32) - own[:2]) / ARENA_SIZE
        else:
            nearest_vec = np.zeros(2, dtype=np.float32)
        knocked_flags = np.array([1.0 if target.knocked else 0.0 for target in normal_targets], dtype=np.float32)
        rel_opponent = (other[:2] - own[:2]) / ARENA_SIZE
        opponent_track = self._opponent_tracking_features(team, own, other)
        obs = np.concatenate(
            [
                np.array([own[0] / HALF_ARENA, own[1] / HALF_ARENA, math.cos(own[2]), math.sin(own[2])]),
                rel_opponent,
                opponent_track,
                np.array([math.cos(other[2]), math.sin(other[2])]),
                np.array([self.armor[opponent] / 4.0, self.armor[team] / 4.0]),
                np.array([(self.scores[team] - self.scores[opponent]) / 60.0, self.elapsed / self.max_time_s]),
                np.array([float(self.last_contact), self.localization_confidence[team]]),
                knocked_flags,
                nearest_vec,
                (opponent_base - own[:2]) / ARENA_SIZE,
                (own_base - own[:2]) / ARENA_SIZE,
                np.array([1.0 if self.winner == team else -1.0 if self.winner == opponent else 0.0]),
            ]
        ).astype(np.float32)
        return obs

    def _base_distance(self, team: str) -> float:
        opponent_base = BLUE_BASE_XY if team == "yellow" else YELLOW_BASE_XY
        return float(np.linalg.norm(self.poses[team][:2] - opponent_base))

    @staticmethod
    def _opponent(team: str) -> str:
        return "blue" if team == "yellow" else "yellow"


if __name__ == "__main__":
    env = RoboCupVisionRLSelfPlayEnv()
    obs, _ = env.reset(seed=11)
    for _ in range(16):
        actions = {team: env.action_spaces[team].sample() for team in AGENTS}
        obs, rewards, terminated, truncated, info = env.step(actions)
        print(f"rewards={rewards} done={terminated} truncated={truncated} info={info}")
        if any(terminated.values()) or any(truncated.values()):
            break
