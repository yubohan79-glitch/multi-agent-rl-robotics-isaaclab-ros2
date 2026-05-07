from __future__ import annotations

import math
import heapq
from dataclasses import dataclass

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:  # pragma: no cover - mirrors the single-agent fallback.
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
        pass

    gym = _Gym()
    spaces = _Spaces()

from robocup_visionrl_gym_env import (
    ARENA_SIZE,
    BASE_HIT_RADIUS,
    BASE_HIT_SUCCESS_BY_NORMAL_HITS,
    BASE_SHOOT_IDEAL_DISTANCE,
    BASE_SHOOT_MIN_RANGE,
    BASE_SHOOT_RANGE,
    BLUE_BASE_XY,
    BLUE_BASE_TARGET_XY,
    BLUE_START,
    HALF_ARENA,
    PUSHABLE_OBSTACLE_HALF,
    PUSHABLE_OBSTACLE_RANDOM_JITTER,
    PUSHABLE_OBSTACLE_STARTS,
    ROBOT_PUSHABLE_CLEARANCE_RADIUS,
    ROBOT_RADIUS,
    ROBOT_LENGTH,
    ROBOT_WIDTH,
    LASER_DWELL_REQUIRED_S,
    LASER_DWELL_FULL_CONFIDENCE_S,
    LASER_FIRE_COOLDOWN_S,
    SHOOT_HIT_RADIUS,
    SHOOT_IDEAL_DISTANCE,
    SHOOT_MIN_RANGE,
    SHOOT_RANGE,
    SHOOTER_FORWARD_OFFSET,
    RoboCupVisionRLGymEnv,
    YELLOW_BASE_TARGET_XY,
    YELLOW_BASE_XY,
    YELLOW_START,
    Target,
    active_base_armor_blockers,
    base_attack_pose_quality,
    base_hit_success_cap,
    base_removed_side_lane_quality,
    ideal_shoot_distance,
    normalized_laser_dwell_factor,
    segment_intersects_aabb,
    shooting_range_limits,
    wrap_angle,
)


AGENTS = ("yellow", "blue")
TACTICAL_ACTION_DIM = 6
TACTICAL_ACTION_LABELS = (
    "target_selector",
    "base_rush_gate",
    "block_interference_gate",
    "recovery_gate",
    "fire_gate",
    "risk_preference",
)
SENSOR_FUSION_FEATURE_LABELS = (
    "robot_contact",
    "fused_localization_confidence",
    "wheel_imu_consistency",
    "scan_clearance",
    "tof_front_left_clearance",
    "tof_front_right_clearance",
    "bumper_or_hard_contact",
    "camera_target_visible",
    "pushable_contact",
)
SENSOR_FUSION_FEATURE_DIM = len(SENSOR_FUSION_FEATURE_LABELS)
SELFPLAY_OBSERVATION_DIM = 46
RECOVERY_CONFIDENCE_THRESHOLD = 0.28
RECOVERY_COOLDOWN_S = 2.50
BASE_RUSH_ARMOR_GATE = 3
BASE_RUSH_EARLY_NORMAL_HITS = 1
BASE_RUSH_BALANCED_NORMAL_HITS = 2
BASE_RUSH_PREFERRED_NORMAL_HITS = 3
TACTICAL_STANDOFF_MIN = SHOOTER_FORWARD_OFFSET + 0.24
TACTICAL_STANDOFF_MAX = SHOOTER_FORWARD_OFFSET + SHOOT_RANGE - 0.02
IDEAL_SHOOT_DISTANCE = SHOOT_IDEAL_DISTANCE
BASE_IDEAL_CENTER_STANDOFF = SHOOTER_FORWARD_OFFSET + BASE_SHOOT_IDEAL_DISTANCE
IDEAL_CENTER_STANDOFF = SHOOTER_FORWARD_OFFSET + SHOOT_IDEAL_DISTANCE
MIN_SHOOT_DISTANCE = SHOOT_MIN_RANGE
FIRE_GOAL_READY_RADIUS = 0.12
FIRE_YAW_TOLERANCE_RAD = 0.095
NORMAL_FIRE_HOLD_RADIUS = 0.180
BASE_FIRE_HOLD_RADIUS = 0.040
FIRE_POSE_BLOCKED_STEP_LIMIT = 6
NORMAL_ATTACK_STALE_STEP_LIMIT = 110
BASE_ATTACK_STALE_STEP_LIMIT = 80
SHOT_CLOSE_DISTANCE = SHOOTER_FORWARD_OFFSET + 0.14
BASE_SHOT_CLOSE_DISTANCE = SHOOTER_FORWARD_OFFSET + BASE_SHOOT_MIN_RANGE + 0.02
NORMAL_AIM_MICRO_SCAN_RAD = 0.014
BASE_AIM_MICRO_SCAN_RAD = 0.012
BASE_AIM_SEEK_SCAN_RAD = 0.018
BASE_FIRE_REPLAN_NUDGE_M = 0.008
SHOT_TIME_COST_SCALE = 0.035
PUSH_INTENT_THRESHOLD = 0.48
PUSH_STEP_M = 0.060
PUSH_ROBOT_RECOIL_M = 0.014
PUSH_CLEARANCE_MARGIN = 0.025
# Match the IsaacLab rendered wheel/body footprint so strict replay catches the
# same red-box overlap that is visible in video.
ROBOT_PUSHABLE_VISUAL_HALF_EXTENTS = (ROBOT_LENGTH * 0.5 + 0.110, ROBOT_WIDTH * 0.5 + 0.087)
DRAW_TIMEOUT_PENALTY = 14.0
NORMAL_TARGET_CONTACT_RADIUS = 0.035
BASE_TARGET_CONTACT_RADIUS = 0.045
PUSHABLE_CONTACT_RADIUS_SCALE = ROBOT_PUSHABLE_CLEARANCE_RADIUS / ROBOT_RADIUS
TOF_SENSOR_RANGE_M = 0.72
TOF_SENSOR_LATERAL_OFFSET_M = 0.070
FUSION_CONFIDENCE_RECOVERY_GAIN = 0.030
FUSION_CONFIDENCE_DRIFT_LOSS = 0.004
FUSION_HARD_CONTACT_LOSS = 0.025
FUSION_JAMMED_PUSH_LOSS = 0.020
ACCEL_DRIFT_LINEAR_THRESHOLD = 1.05
ACCEL_DRIFT_ANGULAR_THRESHOLD = 4.20
ACCEL_DRIFT_LOSS_SCALE = 0.020
CAMERA_MEMORY_FOV_RAD = math.radians(78.0)
CAMERA_MEMORY_RANGE_M = 1.15
POST_HIT_RETREAT_S = 0.42
POST_HIT_RETREAT_SPEED = -0.18


@dataclass
class ShotResult:
    shooter: str
    target_name: str
    target_owner: str
    kind: str


@dataclass
class DomainRandomizationParams:
    drive_scale: float = 1.0
    turn_scale: float = 1.0
    push_step_scale: float = 1.0
    shot_accuracy_scale: float = 1.0
    drift_loss_scale: float = 1.0
    sensor_noise_scale: float = 0.0


def laser_origin_from_pose(pose: np.ndarray) -> tuple[float, float]:
    yaw = float(pose[2])
    return (
        float(pose[0]) + SHOOTER_FORWARD_OFFSET * math.cos(yaw),
        float(pose[1]) + SHOOTER_FORWARD_OFFSET * math.sin(yaw),
    )


def circle_aabb_collision(
    point: tuple[float, float],
    center: tuple[float, float],
    half_size: tuple[float, float],
    radius: float,
) -> tuple[bool, tuple[float, float], float]:
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    closest_x = max(center[0] - half_size[0], min(point[0], center[0] + half_size[0]))
    closest_y = max(center[1] - half_size[1], min(point[1], center[1] + half_size[1]))
    vx = point[0] - closest_x
    vy = point[1] - closest_y
    distance = math.hypot(vx, vy)
    if distance > 1e-8:
        penetration = radius - distance
        if penetration <= 0.0:
            return False, (0.0, 0.0), 0.0
        return True, (vx / distance, vy / distance), penetration

    inside_x = half_size[0] - abs(dx)
    inside_y = half_size[1] - abs(dy)
    if inside_x < 0.0 or inside_y < 0.0:
        return False, (0.0, 0.0), 0.0
    if inside_x <= inside_y:
        normal = (1.0 if dx >= 0.0 else -1.0, 0.0)
        penetration = inside_x + radius
    else:
        normal = (0.0, 1.0 if dy >= 0.0 else -1.0)
        penetration = inside_y + radius
    return True, normal, penetration


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
    return oriented_rect_aabb_collision(
        (float(pose[0]), float(pose[1])),
        float(pose[2]),
        ROBOT_PUSHABLE_VISUAL_HALF_EXTENTS,
        box_center,
        box_half,
    )


def team_frame_sign(team: str) -> float:
    return 1.0 if team == "yellow" else -1.0


class RoboCupVisionRLSelfPlayEnv:
    """Two-agent rule environment for world-model SAC Flow self-play training.

    The interface is intentionally lightweight: each step receives one action
    per team and returns dicts keyed by `yellow` and `blue`. The SAC Flow trainer
    wraps this class with its vectorization/rollout adapter.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 0.10,
        max_time_s: float = 180.0,
        *,
        domain_randomization: bool = False,
        action_shield: bool = True,
    ):
        self.dt = dt
        self.max_time_s = max_time_s
        self.domain_randomization = bool(domain_randomization)
        self.action_shield = bool(action_shield)
        self.domain_params = DomainRandomizationParams()
        self.action_spaces = {
            team: spaces.Box(
                low=np.full(TACTICAL_ACTION_DIM, -1.0, dtype=np.float32),
                high=np.full(TACTICAL_ACTION_DIM, 1.0, dtype=np.float32),
                dtype=np.float32,
            )
            for team in AGENTS
        }
        self.observation_spaces = {
            team: spaces.Box(
                low=np.full(SELFPLAY_OBSERVATION_DIM, -np.inf, dtype=np.float32),
                high=np.full(SELFPLAY_OBSERVATION_DIM, np.inf, dtype=np.float32),
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
        self.domain_params = self._sample_domain_randomization()
        noise_seed = int(seed if seed is not None else self.rng.integers(0, 2**31 - 1))
        first_noise = noise_seed * 2 + 101
        second_noise = noise_seed * 2 + 202
        first_base_noise = noise_seed * 2 + 303
        second_base_noise = noise_seed * 2 + 404
        if noise_seed % 2 == 0:
            self.shot_rng = {
                "yellow": np.random.default_rng(first_noise),
                "blue": np.random.default_rng(second_noise),
            }
            self.base_cap_rng = {
                "yellow": np.random.default_rng(first_base_noise),
                "blue": np.random.default_rng(second_base_noise),
            }
        else:
            self.shot_rng = {
                "yellow": np.random.default_rng(second_noise),
                "blue": np.random.default_rng(first_noise),
            }
            self.base_cap_rng = {
                "yellow": np.random.default_rng(second_base_noise),
                "blue": np.random.default_rng(first_base_noise),
            }
        self.base_rush_priority_team = AGENTS[noise_seed % len(AGENTS)]
        self.poses = {
            "yellow": YELLOW_START.copy(),
            "blue": BLUE_START.copy(),
        }
        self.targets = RoboCupVisionRLGymEnv()._make_targets()
        self.pushable_obstacles = self._sample_pushable_obstacle_starts()
        self.armor = {"yellow": 4, "blue": 4}
        self.scores = {"yellow": 0, "blue": 0}
        self.elapsed = 0.0
        self.last_fire = {"yellow": -99.0, "blue": -99.0}
        self.laser_locks = {
            team: {"target": "", "start": -99.0}
            for team in AGENTS
        }
        self.last_relocalization_time = {team: -999.0 for team in AGENTS}
        self.last_contact = False
        self.localization_confidence = {"yellow": 1.0, "blue": 1.0}
        self.sensor_fusion = {
            team: self._default_sensor_fusion_state()
            for team in AGENTS
        }
        self.winner: str | None = None
        self.pending_fire = {team: False for team in AGENTS}
        self.last_push_event = {team: "" for team in AGENTS}
        self.last_push_impulse = {team: {} for team in AGENTS}
        self.last_target_contact_time = {team: -99.0 for team in AGENTS}
        self.last_shot_attempt: dict[str, dict[str, float | str | bool]] = {team: {} for team in AGENTS}
        self.base_rush_lottery: dict[str, dict[tuple[str, int], bool]] = {team: {} for team in AGENTS}
        self.base_retry_min_normal_hits = {team: 0 for team in AGENTS}
        self.selected_target_name: dict[str, str | None] = {team: None for team in AGENTS}
        self.last_motion_command = {team: (0.0, 0.0) for team in AGENTS}
        self.target_order: dict[str, list[str]] = {team: [] for team in AGENTS}
        self.target_fail_counts: dict[str, dict[str, int]] = {team: {} for team in AGENTS}
        self.target_cooldowns: dict[str, dict[str, float]] = {team: {} for team in AGENTS}
        self.lost_targets: dict[str, set[str]] = {team: set() for team in AGENTS}
        self.fire_pose_blocked_steps: dict[str, dict[str, int]] = {team: {} for team in AGENTS}
        self.normal_attack_stale_steps: dict[str, dict[str, int]] = {team: {} for team in AGENTS}
        self.base_attack_stale_steps: dict[str, dict[str, int]] = {team: {} for team in AGENTS}
        self.post_hit_retreat_until = {team: -99.0 for team in AGENTS}
        self._fire_pose_cache: dict[tuple[str, int], list[tuple[np.ndarray, float, float, float]]] = {}
        self._path_cache: dict[tuple[tuple[int, int], tuple[int, int], tuple[tuple[int, int], ...]], list[np.ndarray]] = {}
        self._route_distance_cache: dict[tuple[tuple[int, int], tuple[int, int], tuple[tuple[int, int], ...]], float] = {}
        self.strategy_counts = {
            team: {
                "attack_steps": 0,
                "base_rush_steps": 0,
                "block_steps": 0,
                "interference_steps": 0,
                "recovery_steps": 0,
                "normal_hits": 0,
                "base_hits": 0,
            }
            for team in AGENTS
        }
        self.previous_base_distance = {
            "yellow": float(np.linalg.norm(self.poses["yellow"][:2] - BLUE_BASE_XY)),
            "blue": float(np.linalg.norm(self.poses["blue"][:2] - YELLOW_BASE_XY)),
        }
        self.previous_attack_distance = {
            team: self._nearest_opponent_target_distance(team)
            for team in AGENTS
        }
        return {team: self._obs(team) for team in AGENTS}, {team: {} for team in AGENTS}

    def step(self, actions: dict[str, np.ndarray]):
        rewards = {team: -0.018 for team in AGENTS}
        infos: dict[str, dict[str, object]] = {team: {} for team in AGENTS}
        action_values = {
            team: self._coerce_action(actions.get(team, np.zeros(TACTICAL_ACTION_DIM, dtype=np.float32)))
            for team in AGENTS
        }
        team_order = list(AGENTS)
        if bool(self.rng.integers(0, 2)):
            team_order.reverse()
        self.elapsed += self.dt

        for team in team_order:
            previous = self._base_distance(team)
            previous_attack_distance = self._nearest_opponent_target_distance(team)
            action = action_values[team]
            blocked, decision_info = self._apply_action(team, action)
            decision_info["step_order"] = team_order.index(team)
            infos[team].update(decision_info)
            rewards[team] += 0.10 * (previous - self._base_distance(team))
            rewards[team] += 0.13 * (previous_attack_distance - self._nearest_opponent_target_distance(team))
            if decision_info.get("tactic") == "block":
                threat = float(decision_info.get("opponent_threat", 0.0))
                rewards[team] += -0.045 + 0.08 * threat
            if decision_info.get("tactic") == "recover":
                useful = self.localization_confidence[team] < RECOVERY_CONFIDENCE_THRESHOLD
                rewards[team] += 0.055 if useful else -0.08
            if decision_info.get("base_rush") and self.armor[self._opponent(team)] > BASE_RUSH_ARMOR_GATE:
                rewards[team] -= 0.10
            if decision_info.get("base_rush"):
                normal_hits = self._normal_hits_against(team)
                if normal_hits <= 1:
                    rewards[team] -= 0.18
                elif normal_hits == BASE_RUSH_BALANCED_NORMAL_HITS:
                    rewards[team] -= 0.055
                elif normal_hits >= BASE_RUSH_PREFERRED_NORMAL_HITS:
                    rewards[team] += 0.095
            if decision_info.get("tactic") == "attack":
                goal_distance = float(decision_info.get("goal_distance_m", 1.0))
                rewards[team] += 0.06 * max(0.0, 1.0 - goal_distance / 0.75)
                shot_distance = float(decision_info.get("shot_distance_m", SHOOT_RANGE))
                close_quality = self._shot_accuracy_from_geometry(
                    shot_distance,
                    float(decision_info.get("shot_lateral_error_m", SHOOT_HIT_RADIUS)),
                    bool(decision_info.get("base_rush", False)),
                )
                rewards[team] += 0.09 * close_quality
                rewards[team] -= SHOT_TIME_COST_SCALE * max(0.0, 0.38 - shot_distance)
                fire_requested = float(action[4]) > 0.55
                if decision_info.get("fire_ready"):
                    base_bonus = 0.28 if decision_info.get("base_rush") else 0.0
                    rewards[team] += 0.24 + 0.30 * close_quality + base_bonus
                elif fire_requested:
                    distance = float(decision_info.get("shot_distance_m", 9.0))
                    yaw_error = float(decision_info.get("shot_yaw_error_rad", math.pi))
                    miss_penalty = 0.06
                    min_range, max_range = shooting_range_limits(bool(decision_info.get("base_rush", False)))
                    if distance > max_range or distance < min_range:
                        miss_penalty += 0.08
                    if yaw_error > FIRE_YAW_TOLERANCE_RAD:
                        miss_penalty += 0.05
                    rewards[team] -= miss_penalty
            if decision_info.get("pushed_obstacle"):
                useful_push = decision_info.get("tactic") in ("attack", "push_clear")
                rewards[team] += 0.18 if useful_push else -0.05
            if blocked:
                rewards[team] -= 0.10
                infos[team]["blocked"] = True

        if self._resolve_contact():
            for team in AGENTS:
                contact_reward = self._contact_reward(team, infos[team])
                rewards[team] += contact_reward
                infos[team]["robot_contact"] = True
                infos[team]["tactical_contact"] = contact_reward > 0.0

        for team in AGENTS:
            self._resolve_target_contacts(team, rewards, infos)

        for team in AGENTS:
            action = action_values[team]
            if self.localization_confidence[team] < RECOVERY_CONFIDENCE_THRESHOLD:
                recovery_requested = bool(infos[team].get("tactic") == "recover") or float(action[3]) > 0.35
                if recovery_requested and self._can_relocalize(team):
                    self.localization_confidence[team] = min(1.0, self.localization_confidence[team] + 0.34)
                    self.last_relocalization_time[team] = self.elapsed
                    self._boost_sensor_fusion_recovery(team)
                    rewards[team] += 0.08
                    infos[team]["relocalizing"] = True
                elif recovery_requested:
                    rewards[team] -= 0.02
                    infos[team]["relocalization_cooldown"] = True
                else:
                    rewards[team] -= 0.14

        shot_results: list[ShotResult] = []
        missed_shots: list[str] = []
        for team in team_order:
            if self.pending_fire[team] and self.elapsed - self.last_fire[team] > LASER_FIRE_COOLDOWN_S:
                result = self._apply_fire(team)
                if result is not None:
                    self.last_fire[team] = self.elapsed
                    shot_results.append(result)
                else:
                    reason = str(self.last_shot_attempt.get(team, {}).get("reason", ""))
                    if reason == "dwell":
                        rewards[team] += 0.018
                    elif reason == "probabilistic_miss":
                        self.last_fire[team] = self.elapsed
                        infos[team]["shot_attempt"] = dict(self.last_shot_attempt[team])
                        rewards[team] -= 0.025
                    else:
                        missed_shots.append(team)
            else:
                self._reset_laser_lock(team)

        base_win_results = [
            result for result in shot_results
            if result.kind == f"base_{self._opponent(result.shooter)}"
        ]
        if len(base_win_results) == 2:
            for result in shot_results:
                self._score_shot(result, rewards, infos, terminal_override=False)
            self.winner = "draw"
            for result in base_win_results:
                infos[result.shooter]["simultaneous_base_hit"] = True
        else:
            for result in shot_results:
                self._score_shot(result, rewards, infos)
        for team in missed_shots:
            target_name = self.selected_target_name.get(team)
            shot_attempt = self.last_shot_attempt.get(team, {})
            if target_name and str(target_name).endswith("BaseTarget") and shot_attempt.get("reason") == "base_cap_failed":
                # A failed base cap lottery means this normal-hit bucket cannot
                # legally win an early base in this episode. Force one- and
                # two-hit rushes to improve the success window, but allow a
                # three-hit attack to retry after cooldown because it is already
                # the intended high-probability tempo.
                normal_hits = self._normal_hits_against(team)
                required_hits = min(4, normal_hits + 1) if normal_hits < BASE_RUSH_PREFERRED_NORMAL_HITS else normal_hits
                self.base_retry_min_normal_hits[team] = max(
                    int(self.base_retry_min_normal_hits.get(team, 0)),
                    required_hits,
                )
                self._fire_pose_cache.clear()
            if target_name:
                self._mark_target_failed(team, target_name)
            infos[team]["shot_attempt"] = dict(self.last_shot_attempt[team])
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
            rewards[self.winner] += 80.0
            rewards[loser] -= 55.0
        elif self.winner == "draw":
            for team in AGENTS:
                rewards[team] -= DRAW_TIMEOUT_PENALTY
        observations = {team: self._obs(team) for team in AGENTS}
        terminations = {team: terminated for team in AGENTS}
        truncations = {team: truncated for team in AGENTS}
        return observations, rewards, terminations, truncations, infos

    def _coerce_action(self, action: np.ndarray | None) -> np.ndarray:
        values = np.zeros(TACTICAL_ACTION_DIM, dtype=np.float32)
        if action is not None:
            raw = np.asarray(action, dtype=np.float32).reshape(-1)
            count = min(raw.shape[0], TACTICAL_ACTION_DIM)
            values[:count] = raw[:count]
        return np.clip(values, -1.0, 1.0)

    def _sample_domain_randomization(self) -> DomainRandomizationParams:
        if not self.domain_randomization:
            return DomainRandomizationParams()
        return DomainRandomizationParams(
            drive_scale=float(self.rng.uniform(0.92, 1.08)),
            turn_scale=float(self.rng.uniform(0.90, 1.10)),
            push_step_scale=float(self.rng.uniform(0.72, 1.18)),
            shot_accuracy_scale=float(self.rng.uniform(0.82, 1.05)),
            drift_loss_scale=float(self.rng.uniform(0.85, 1.40)),
            sensor_noise_scale=float(self.rng.uniform(0.0, 0.035)),
        )

    def _sample_pushable_obstacle_starts(self) -> dict[str, np.ndarray]:
        starts = {name: value.copy() for name, value in PUSHABLE_OBSTACLE_STARTS.items()}
        if not self.domain_randomization:
            return starts
        for name, base_xy in starts.items():
            jitter = self.rng.uniform(-PUSHABLE_OBSTACLE_RANDOM_JITTER, PUSHABLE_OBSTACLE_RANDOM_JITTER, size=2)
            candidate = base_xy + jitter.astype(np.float32)
            sign = np.sign(base_xy)
            lower = np.array([0.58, 0.58], dtype=np.float32) * sign
            upper = np.array([0.96, 0.96], dtype=np.float32) * sign
            starts[name] = np.minimum(np.maximum(candidate, np.minimum(lower, upper)), np.maximum(lower, upper))
        return starts

    def _shield_contact_action(self, team: str, action: np.ndarray) -> tuple[np.ndarray, bool]:
        if not self.action_shield:
            return action, False
        if not self._near_own_critical_assets(team):
            return action, False
        shielded = action.copy()
        changed = False
        if float(shielded[2]) > 0.10:
            shielded[2] = min(float(shielded[2]), -0.25)
            changed = True
        if float(shielded[5]) > 0.35:
            shielded[5] = min(float(shielded[5]), 0.20)
            changed = True
        return shielded, changed

    def _apply_action(self, team: str, action: np.ndarray | None) -> tuple[bool, dict[str, object]]:
        action = self._coerce_action(action)
        action, contact_shielded = self._shield_contact_action(team, action)
        self.pending_fire[team] = False
        self.selected_target_name[team] = None
        info: dict[str, object] = {
            "action_labels": TACTICAL_ACTION_LABELS,
        }
        if contact_shielded:
            info["action_shield_contact"] = True

        if (
            self.localization_confidence[team] < RECOVERY_CONFIDENCE_THRESHOLD
            and float(action[3]) > 0.35
            and self._can_relocalize(team)
        ):
            spin = 0.72 if team == "yellow" else -0.72
            blocked = self._integrate_command(team, 0.0, spin)
            self._reset_laser_lock(team)
            self.strategy_counts[team]["recovery_steps"] += 1
            info.update({"tactic": "recover", "recovery_gate": float(action[3])})
            return blocked, info

        if self.elapsed < self.post_hit_retreat_until[team]:
            blocked = self._integrate_command(team, POST_HIT_RETREAT_SPEED, 0.0, allow_push=True)
            self._reset_laser_lock(team)
            info.update(
                {
                    "tactic": "push_clear",
                    "post_hit_retreat": True,
                    "pushed_obstacle": self.last_push_event[team],
                    "push_impulse": self.last_push_impulse[team],
                }
            )
            return blocked, info

        if self._should_block(team, action):
            risk = (float(action[5]) + 1.0) * 0.5
            goal = self._block_goal(team, risk)
            opponent = self._opponent(team)
            blocked = self._drive_to_goal(team, goal, risk, face_xy=self.poses[opponent][:2])
            self.strategy_counts[team]["block_steps"] += 1
            if risk > 0.72:
                self.strategy_counts[team]["interference_steps"] += 1
            self._reset_laser_lock(team)
            info.update(
                {
                    "tactic": "block",
                    "goal_xy": tuple(round(float(v), 3) for v in goal),
                    "opponent_threat": self._opponent_threat(team),
                    "interference": risk > 0.72,
                    "pushed_obstacle": self.last_push_event[team],
                    "push_impulse": self.last_push_impulse[team],
                }
            )
            return blocked, info

        target = self._select_tactical_target(team, action)
        if target is None:
            info["tactic"] = "wait"
            self._reset_laser_lock(team)
            return self._integrate_command(team, 0.0, 0.0), info

        risk = (float(action[5]) + 1.0) * 0.5
        goal = self._fire_standoff_goal(team, target, risk)
        pre_goal_distance = float(np.linalg.norm(goal - self.poses[team][:2]))
        geometry_snapshot = self._fire_geometry_snapshot(team, target, risk)
        center_distance = float(geometry_snapshot["center_distance"])
        line_clear = bool(geometry_snapshot["line_clear"])
        yaw_error = float(geometry_snapshot["yaw_error"])
        transient_base_occlusion = (
            target.kind.startswith("base_")
            and yaw_error > 0.25
            and self._center_aim_line_clear(team, target)
        )
        base_target = target.kind.startswith("base_")
        hold_radius = BASE_FIRE_HOLD_RADIUS if base_target else NORMAL_FIRE_HOLD_RADIUS
        min_shot_range, max_shot_range = shooting_range_limits(base_target)
        shot_distance_now = float(geometry_snapshot["shot_distance"])
        line_ok_for_hold = (
            line_clear
            or transient_base_occlusion
            or (base_target and pre_goal_distance < hold_radius and bool(geometry_snapshot["base_pose_ok"]))
        )
        near_fire_window = (
            pre_goal_distance < hold_radius
            and line_ok_for_hold
            and center_distance >= self._target_contact_clearance(target)
            and shot_distance_now >= min_shot_range + 0.006
            and shot_distance_now <= max_shot_range
        )
        holding_fire_pose = bool(geometry_snapshot["geometry_ready"]) or near_fire_window
        geometry_ready = bool(geometry_snapshot["geometry_ready"])
        if holding_fire_pose and base_target and not geometry_ready and pre_goal_distance < hold_radius:
            refined = self._best_fire_pose(team, target, risk, route_aware=True)
            if refined is not None and float(np.linalg.norm(refined[0] - self.poses[team][:2])) > BASE_FIRE_REPLAN_NUDGE_M:
                goal = refined[0]
                blocked = self._drive_to_goal(team, goal, risk, face_xy=np.asarray(target.xy, dtype=np.float32))
                holding_fire_pose = False
            else:
                blocked = self._hold_fire_pose(team, target, risk)
        elif holding_fire_pose:
            blocked = self._hold_fire_pose(team, target, risk)
            if blocked and not geometry_ready:
                face_target_xy = None if not line_clear else np.asarray(target.xy, dtype=np.float32)
                blocked = self._drive_to_goal(team, goal, risk, face_xy=face_target_xy)
                holding_fire_pose = False
        else:
            face_target_xy = np.asarray(target.xy, dtype=np.float32)
            if target.kind.startswith("base_") and pre_goal_distance > 0.075:
                face_target_xy = None
            if not bool(geometry_snapshot["line_clear"]) and pre_goal_distance > 0.030:
                face_target_xy = None
            blocked = self._drive_to_goal(team, goal, risk, face_xy=face_target_xy)
        fire_info = self._update_fire_gate(team, target, action, risk)
        fire_pose_replanned = False
        if holding_fire_pose and blocked and not self.pending_fire[team]:
            blocked_steps = self.fire_pose_blocked_steps[team].get(target.name, 0) + 1
            self.fire_pose_blocked_steps[team][target.name] = blocked_steps
            if blocked_steps >= FIRE_POSE_BLOCKED_STEP_LIMIT:
                self._mark_target_failed(team, target.name)
                self._reset_laser_lock(team)
                self.fire_pose_blocked_steps[team].pop(target.name, None)
                self._fire_pose_cache.clear()
                fire_pose_replanned = True
        else:
            self.fire_pose_blocked_steps[team].pop(target.name, None)
        normal_attack_replanned = False
        if target.kind == "normal" and not self.pending_fire[team]:
            normal_stale_steps = self.normal_attack_stale_steps[team].get(target.name, 0) + 1
            self.normal_attack_stale_steps[team][target.name] = normal_stale_steps
            if normal_stale_steps >= NORMAL_ATTACK_STALE_STEP_LIMIT:
                self._mark_target_failed(team, target.name)
                self._reset_laser_lock(team)
                self.normal_attack_stale_steps[team].pop(target.name, None)
                self._fire_pose_cache.clear()
                normal_attack_replanned = True
        else:
            self.normal_attack_stale_steps[team].pop(target.name, None)
        base_attack_replanned = False
        if target.kind.startswith("base_") and not self.pending_fire[team] and self._normal_hits_against(team) < 4:
            stale_steps = self.base_attack_stale_steps[team].get(target.name, 0) + 1
            self.base_attack_stale_steps[team][target.name] = stale_steps
            if stale_steps >= BASE_ATTACK_STALE_STEP_LIMIT:
                normal_hits = self._normal_hits_against(team)
                required_hits = min(4, normal_hits + 1) if normal_hits < BASE_RUSH_PREFERRED_NORMAL_HITS else normal_hits
                self.base_retry_min_normal_hits[team] = max(
                    int(self.base_retry_min_normal_hits.get(team, 0)),
                    required_hits,
                )
                self._mark_target_failed(team, target.name)
                self._reset_laser_lock(team)
                self.base_attack_stale_steps[team].pop(target.name, None)
                self._fire_pose_cache.clear()
                base_attack_replanned = True
        else:
            self.base_attack_stale_steps[team].pop(target.name, None)
        self.selected_target_name[team] = target.name
        self.strategy_counts[team]["attack_steps"] += 1
        if target.kind.startswith("base_"):
            self.strategy_counts[team]["base_rush_steps"] += 1
        info.update(
            {
                "tactic": "attack",
                "selected_target": target.name,
                "base_rush": target.kind.startswith("base_"),
                "goal_xy": tuple(round(float(v), 3) for v in goal),
                "goal_distance_m": round(float(np.linalg.norm(goal - self.poses[team][:2])), 4),
                "fire_ready": self.pending_fire[team],
                "holding_fire_pose": holding_fire_pose,
                "fire_pose_replanned": fire_pose_replanned,
                "normal_attack_replanned": normal_attack_replanned,
                "base_attack_replanned": base_attack_replanned,
                "pushed_obstacle": self.last_push_event[team],
                "push_impulse": self.last_push_impulse[team],
            }
        )
        info.update(fire_info)
        return blocked, info

    def _integrate_command(
        self,
        team: str,
        linear_speed: float,
        angular_speed: float,
        *,
        allow_push: bool = False,
    ) -> bool:
        self.last_push_event[team] = ""
        self.last_push_impulse[team] = {}
        before = self.poses[team].copy()
        separated_before = self._separated_pose_from_all_pushables(before)
        if not np.allclose(separated_before, before, atol=1e-6):
            self.poses[team] = separated_before
            before = separated_before.copy()
        linear_speed = float(linear_speed) * self.domain_params.drive_scale
        angular_speed = float(angular_speed) * self.domain_params.turn_scale
        pose = before.copy()
        pose[2] = wrap_angle(float(pose[2] + angular_speed * self.dt))
        pose[0] += linear_speed * math.cos(float(pose[2])) * self.dt
        pose[1] += linear_speed * math.sin(float(pose[2])) * self.dt
        if self._static_pose_blocked(pose):
            if abs(linear_speed) <= 1e-6 and not self._footprint_outside_arena(pose):
                self.poses[team] = pose
                self._record_motion_sensor_fusion(
                    team,
                    before,
                    pose,
                    linear_speed,
                    angular_speed,
                    blocked=False,
                )
                return False
            self._record_motion_sensor_fusion(
                team,
                before,
                before,
                linear_speed,
                angular_speed,
                blocked=True,
                hard_contact=True,
            )
            return True
        if self._target_collision_name(pose) is not None:
            self._record_motion_sensor_fusion(
                team,
                before,
                before,
                linear_speed,
                angular_speed,
                blocked=True,
                hard_contact=True,
            )
            return True
        obstacle_name = self._pushable_collision_name(pose)
        if obstacle_name is not None:
            if abs(linear_speed) <= 0.03:
                separated = self._separated_pose_from_pushable(pose, obstacle_name)
                if self._pushable_collision_name(separated) is None:
                    self.poses[team] = separated
                    self._record_motion_sensor_fusion(
                        team,
                        before,
                        separated,
                        linear_speed,
                        angular_speed,
                        blocked=False,
                        push_contact=True,
                    )
                    return False
            if not allow_push or abs(linear_speed) <= 0.03:
                self._record_motion_sensor_fusion(
                    team,
                    before,
                    before,
                    linear_speed,
                    angular_speed,
                    blocked=True,
                    push_contact=True,
                    jammed_push=True,
                )
                return True
            motion_yaw = float(pose[2]) if linear_speed > 0.0 else wrap_angle(float(pose[2]) + math.pi)
            if not self._push_obstacle(team, obstacle_name, motion_yaw, pose[:2]):
                self._record_motion_sensor_fusion(
                    team,
                    before,
                    before,
                    linear_speed,
                    angular_speed,
                    blocked=True,
                    push_contact=True,
                    jammed_push=True,
                )
                return True
            pose = self._apply_push_recoil_pose(pose, motion_yaw, linear_speed)
            pose = self._separated_pose_from_pushable(pose, obstacle_name)
            if self._pushable_collision_name(pose) is not None:
                self._record_motion_sensor_fusion(
                    team,
                    before,
                    before,
                    linear_speed,
                    angular_speed,
                    blocked=True,
                    push_contact=True,
                    jammed_push=True,
                )
                return True
        self.poses[team] = pose
        self._record_motion_sensor_fusion(
            team,
            before,
            pose,
            linear_speed,
            angular_speed,
            blocked=False,
            push_contact=obstacle_name is not None,
        )
        return False

    def _drive_to_goal(self, team: str, goal_xy: np.ndarray, risk: float, face_xy: np.ndarray | None = None) -> bool:
        pose = self.poses[team]
        subgoal_xy, final_leg = self._planned_subgoal(pose[:2], goal_xy)
        for _ in range(2):
            if self._local_blocker_cost(pose[:2]) > 0.25:
                break
            if final_leg or float(np.linalg.norm(subgoal_xy - pose[:2])) >= 0.10:
                break
            next_subgoal, next_final = self._planned_subgoal(subgoal_xy, goal_xy)
            if np.allclose(next_subgoal, subgoal_xy, atol=1e-4):
                break
            subgoal_xy, final_leg = next_subgoal, next_final
        dx = float(subgoal_xy[0] - pose[0])
        dy = float(subgoal_xy[1] - pose[1])
        distance = math.hypot(dx, dy)
        if final_leg and face_xy is not None and distance < 0.003:
            desired_yaw = math.atan2(float(face_xy[1] - pose[1]), float(face_xy[0] - pose[0]))
        else:
            desired_yaw = math.atan2(dy, dx) if distance > 1e-6 else float(pose[2])
        yaw_error = wrap_angle(desired_yaw - float(pose[2]))
        angular_speed = float(np.clip(3.05 * yaw_error, -2.65, 2.65))
        alignment = max(0.0, 1.0 - abs(yaw_error) / 1.20)
        max_speed = 0.22 + 0.22 * risk
        linear_speed = max_speed * max(0.10, alignment)
        stop_radius = 0.002 if final_leg else 0.035
        if distance < stop_radius:
            linear_speed = 0.0
        elif final_leg:
            linear_speed = min(linear_speed, max(0.0, (distance - stop_radius) * 0.70 / max(self.dt, 1e-6)))
        near_blocker = self._local_blocker_cost(pose[:2]) > 0.25
        cautious_turning = near_blocker or distance < 0.24
        hard_turn_limit = 0.78 if cautious_turning else 1.35
        slow_turn_limit = 0.45 if cautious_turning else 0.95
        if abs(yaw_error) > hard_turn_limit:
            linear_speed = 0.0
            escape_speed = self._boundary_escape_linear_speed(pose)
            if abs(escape_speed) > 1e-6:
                linear_speed = escape_speed
        elif abs(yaw_error) > slow_turn_limit:
            linear_speed *= 0.25
        blocked = self._integrate_command(team, linear_speed, angular_speed, allow_push=risk >= PUSH_INTENT_THRESHOLD)
        if blocked and linear_speed <= 0.02 and distance > stop_radius:
            # Differential-drive escape: when a corner/armor footprint rejects
            # in-place rotation, back out slowly instead of staying locked.
            escape_speed = -0.11 * max(0.45, 1.0 - self._arena_footprint_margin(pose) / 0.16)
            escape_turn = 0.55 * angular_speed
            escaped = self._integrate_command(team, escape_speed, escape_turn, allow_push=False)
            if not escaped:
                return False
        if blocked and linear_speed > 0.0:
            for scale in (0.45, 0.30, 0.20, 0.13):
                cautious_speed = max(0.045, linear_speed * scale)
                cautious = self._integrate_command(
                    team,
                    cautious_speed,
                    angular_speed,
                    allow_push=risk >= PUSH_INTENT_THRESHOLD,
                )
                if not cautious:
                    return False
            return self._integrate_command(team, 0.0, angular_speed, allow_push=False)
        return blocked

    def _planned_subgoal(self, current_xy: np.ndarray, goal_xy: np.ndarray) -> tuple[np.ndarray, bool]:
        corridor_xy, corridor_final = self._central_lane_subgoal(current_xy, goal_xy)
        if not corridor_final:
            if not self._segment_blocked_for_nav(current_xy, corridor_xy):
                return corridor_xy, False
            path_to_corridor = self._astar_path(current_xy, corridor_xy)
            if len(path_to_corridor) >= 3:
                return path_to_corridor[2], False
            if len(path_to_corridor) == 2:
                return path_to_corridor[1], False
        if not self._segment_blocked_for_nav(current_xy, goal_xy):
            return goal_xy, True
        path = self._astar_path(current_xy, goal_xy)
        if len(path) >= 3:
            return path[2], False
        if len(path) == 2:
            return path[1], False
        return self._corridor_subgoal(current_xy, goal_xy)

    def _central_lane_subgoal(self, current_xy: np.ndarray, goal_xy: np.ndarray) -> tuple[np.ndarray, bool]:
        current_y = float(current_xy[1])
        goal_y = float(goal_xy[1])
        if current_y < -0.06 and goal_y > 0.06:
            gate_x = 0.28 if float(current_xy[0]) >= 0.0 else -0.28
            if abs(float(current_xy[0]) - gate_x) > 0.10 or current_y < -0.35:
                return np.array([gate_x, -0.24], dtype=np.float32), False
            return np.array([gate_x, 0.24], dtype=np.float32), False
        if (
            -0.06 <= current_y < 0.30
            and abs(float(current_xy[0])) < 0.45
            and goal_y > 0.06
            and abs(float(goal_xy[0])) > 0.55
        ):
            gate_x = 0.28 if float(current_xy[0]) >= 0.0 else -0.28
            return np.array([gate_x, 0.36], dtype=np.float32), False
        if current_y > 0.06 and goal_y < -0.06:
            gate_x = -0.28 if float(current_xy[0]) <= 0.0 else 0.28
            if abs(float(current_xy[0]) - gate_x) > 0.10 or current_y > 0.35:
                return np.array([gate_x, 0.24], dtype=np.float32), False
            return np.array([gate_x, -0.24], dtype=np.float32), False
        if (
            0.06 >= current_y > -0.30
            and abs(float(current_xy[0])) < 0.45
            and goal_y < -0.06
            and abs(float(goal_xy[0])) > 0.55
        ):
            gate_x = -0.28 if float(current_xy[0]) <= 0.0 else 0.28
            return np.array([gate_x, -0.36], dtype=np.float32), False
        return goal_xy, True

    def _segment_blocked_for_nav(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> bool:
        distance = float(np.linalg.norm(goal_xy - start_xy))
        samples = max(2, int(math.ceil(distance / 0.06)))
        for index in range(1, samples + 1):
            alpha = index / samples
            point = start_xy * (1.0 - alpha) + goal_xy * alpha
            if self._point_blocked_for_nav(float(point[0]), float(point[1])):
                return True
        origin = (float(start_xy[0]), float(start_xy[1]))
        target = (float(goal_xy[0]), float(goal_xy[1]))
        push_half = PUSHABLE_OBSTACLE_HALF + ROBOT_PUSHABLE_CLEARANCE_RADIUS
        for center in self.pushable_obstacles.values():
            if segment_intersects_aabb(origin, target, (float(center[0]), float(center[1])), (push_half, push_half)):
                return True
        return False

    def _segment_likely_blocked_for_nav(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> bool:
        origin = (float(start_xy[0]), float(start_xy[1]))
        target = (float(goal_xy[0]), float(goal_xy[1]))
        for center, half_size in self.nav_blockers:
            if segment_intersects_aabb(origin, target, center, half_size):
                return True
        for center, half_size in active_base_armor_blockers(self.armor, inflated=True):
            if segment_intersects_aabb(origin, target, center, half_size):
                return True
        push_half = PUSHABLE_OBSTACLE_HALF + ROBOT_PUSHABLE_CLEARANCE_RADIUS
        for center in self.pushable_obstacles.values():
            if segment_intersects_aabb(origin, target, (float(center[0]), float(center[1])), (push_half, push_half)):
                return True
        return False

    def _astar_path(self, current_xy: np.ndarray, goal_xy: np.ndarray) -> list[np.ndarray]:
        resolution = 0.18
        limit = HALF_ARENA - ROBOT_RADIUS - 0.02

        def to_key(point: np.ndarray) -> tuple[int, int]:
            clamped = np.array(
                [
                    float(np.clip(point[0], -limit, limit)),
                    float(np.clip(point[1], -limit, limit)),
                ],
                dtype=np.float32,
            )
            return (int(round(float(clamped[0]) / resolution)), int(round(float(clamped[1]) / resolution)))

        def to_point(key: tuple[int, int]) -> np.ndarray:
            return np.array(
                [
                    float(np.clip(key[0] * resolution, -limit, limit)),
                    float(np.clip(key[1] * resolution, -limit, limit)),
                ],
                dtype=np.float32,
            )

        def free(key: tuple[int, int]) -> bool:
            point = to_point(key)
            return not self._point_blocked_for_nav(float(point[0]), float(point[1]))

        start_key = to_key(self._nearest_free_xy(current_xy))
        goal_key = to_key(self._nearest_free_xy(goal_xy))
        obstacle_signature = tuple(
            sorted((round(float(v[0]) / resolution), round(float(v[1]) / resolution)) for v in self.pushable_obstacles.values())
        )
        cache_key = (start_key, goal_key, obstacle_signature)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        min_cell = int(math.floor(-limit / resolution))
        max_cell = int(math.ceil(limit / resolution))
        open_heap: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, start_key))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        cost_so_far: dict[tuple[int, int], float] = {start_key: 0.0}
        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, 1.42),
            (-1, 1, 1.42),
            (1, -1, 1.42),
            (1, 1, 1.42),
        ]

        while open_heap:
            _priority, current = heapq.heappop(open_heap)
            if current == goal_key:
                break
            for dx, dy, step_cost in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if nxt[0] < min_cell or nxt[0] > max_cell or nxt[1] < min_cell or nxt[1] > max_cell:
                    continue
                if not free(nxt):
                    continue
                if dx != 0 and dy != 0 and (not free((current[0] + dx, current[1])) or not free((current[0], current[1] + dy))):
                    continue
                point = to_point(nxt)
                new_cost = cost_so_far[current] + step_cost + 0.04 * self._local_blocker_cost(point)
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    heuristic = math.hypot(goal_key[0] - nxt[0], goal_key[1] - nxt[1])
                    heapq.heappush(open_heap, (new_cost + heuristic, nxt))
                    came_from[nxt] = current

        if goal_key not in came_from and goal_key != start_key:
            self._path_cache[cache_key] = []
            return []

        keys = [goal_key]
        while keys[-1] != start_key:
            keys.append(came_from[keys[-1]])
        keys.reverse()
        path = [to_point(key) for key in keys]
        if len(path) > 0:
            path[-1] = self._nearest_free_xy(goal_xy)
        self._path_cache[cache_key] = path
        if len(self._path_cache) > 512:
            self._path_cache.clear()
        return path

    def _corridor_subgoal(self, current_xy: np.ndarray, goal_xy: np.ndarray) -> tuple[np.ndarray, bool]:
        current_y = float(current_xy[1])
        goal_y = float(goal_xy[1])
        if current_y < -0.06 and goal_y > 0.06:
            gate_x = 0.28 if float(current_xy[0]) >= 0.0 else -0.28
            if abs(float(current_xy[0]) - gate_x) > 0.10 or current_y < -0.35:
                return np.array([gate_x, -0.24], dtype=np.float32), False
            return np.array([gate_x, 0.24], dtype=np.float32), False
        if (
            -0.06 <= current_y < 0.30
            and abs(float(current_xy[0])) < 0.45
            and goal_y > 0.06
            and abs(float(goal_xy[0])) > 0.55
        ):
            gate_x = 0.28 if float(current_xy[0]) >= 0.0 else -0.28
            return np.array([gate_x, 0.36], dtype=np.float32), False
        if current_y > 0.06 and goal_y < -0.06:
            gate_x = -0.28 if float(current_xy[0]) <= 0.0 else 0.28
            if abs(float(current_xy[0]) - gate_x) > 0.10 or current_y > 0.35:
                return np.array([gate_x, 0.24], dtype=np.float32), False
            return np.array([gate_x, -0.24], dtype=np.float32), False
        if (
            0.06 >= current_y > -0.30
            and abs(float(current_xy[0])) < 0.45
            and goal_y < -0.06
            and abs(float(goal_xy[0])) > 0.55
        ):
            gate_x = -0.28 if float(current_xy[0]) <= 0.0 else 0.28
            return np.array([gate_x, -0.36], dtype=np.float32), False
        if abs(current_y) < 0.18 and abs(float(goal_xy[0])) > 0.34 and abs(float(current_xy[0])) < 0.38:
            return np.array([float(current_xy[0]), 0.24 if goal_y >= 0.0 else -0.24], dtype=np.float32), False
        if goal_y > 0.80 and float(goal_xy[0]) < -0.55 and (
            float(current_xy[0]) > -0.58 or current_y > 0.36
        ):
            if current_y > 0.36:
                return np.array([float(current_xy[0]), 0.30], dtype=np.float32), False
            return np.array([-0.62, 0.30], dtype=np.float32), False
        if goal_y < -0.80 and float(goal_xy[0]) > 0.55 and (
            float(current_xy[0]) < 0.58 or current_y < -0.36
        ):
            if current_y < -0.36:
                return np.array([float(current_xy[0]), -0.30], dtype=np.float32), False
            return np.array([0.62, -0.30], dtype=np.float32), False
        if float(goal_xy[0]) > 0.90 and goal_y > 0.50 and current_y < 1.04:
            if float(current_xy[0]) < 1.14 or current_y < 0.26:
                return np.array([1.18, 0.30], dtype=np.float32), False
            return np.array([1.18, 1.08], dtype=np.float32), False
        if float(goal_xy[0]) < -0.90 and goal_y > 0.50 and current_y < 1.04:
            if float(current_xy[0]) > -1.14 or current_y < 0.26:
                return np.array([-1.18, 0.30], dtype=np.float32), False
            return np.array([-1.18, 1.08], dtype=np.float32), False
        if float(goal_xy[0]) > 0.90 and goal_y < -0.50 and current_y > -1.04:
            if float(current_xy[0]) < 1.14 or current_y > -0.26:
                return np.array([1.18, -0.30], dtype=np.float32), False
            return np.array([1.18, -1.08], dtype=np.float32), False
        if float(goal_xy[0]) < -0.90 and goal_y < -0.50 and current_y > -1.04:
            if float(current_xy[0]) > -1.14 or current_y > -0.26:
                return np.array([-1.18, -0.30], dtype=np.float32), False
            return np.array([-1.18, -1.08], dtype=np.float32), False
        return goal_xy, True

    def _should_block(self, team: str, action: np.ndarray) -> bool:
        opponent = self._opponent(team)
        time_remaining = max(0.0, self.max_time_s - self.elapsed)
        score_delta = self.scores[team] - self.scores[opponent]
        threat = self._opponent_threat(team)
        block_gate = float(action[2])
        risk = (float(action[5]) + 1.0) * 0.5
        if block_gate <= 0.25:
            return False
        if threat > 0.68 and block_gate > 0.45:
            return True
        if score_delta >= 10 and time_remaining < 35.0 and block_gate > 0.55:
            return True
        opponent_distance = float(np.linalg.norm(self.poses[opponent][:2] - self.poses[team][:2]))
        if block_gate > 0.72 and risk > 0.62 and opponent_distance < 1.05:
            return not self._near_own_critical_assets(team)
        return block_gate > 0.92 and risk > 0.86 and not self._near_own_critical_assets(team)

    def _block_goal(self, team: str, risk: float) -> np.ndarray:
        opponent = self._opponent(team)
        own_base = YELLOW_BASE_XY if team == "yellow" else BLUE_BASE_XY
        opponent_xy = self.poses[opponent][:2]
        toward_opponent = opponent_xy - own_base
        distance = float(np.linalg.norm(toward_opponent))
        if distance < 1e-6:
            toward_opponent = np.array([-1.0, 1.0], dtype=np.float32) if team == "yellow" else np.array([1.0, -1.0], dtype=np.float32)
            distance = float(np.linalg.norm(toward_opponent))
        unit = toward_opponent / max(distance, 1e-6)
        if risk > 0.82 and not self._near_own_critical_assets(team):
            return self._nearest_free_xy(opponent_xy)
        lane_distance = 0.54 + 0.26 * risk
        if risk > 0.72:
            lane_distance = min(distance - 0.08, 0.86)
        goal = own_base + unit * max(0.35, lane_distance)
        return self._nearest_free_xy(goal)

    def _select_tactical_target(self, team: str, action: np.ndarray) -> Target | None:
        opponent = self._opponent(team)
        self._refresh_target_visibility_memory(team)
        normal_targets = [
            target for target in self.targets
            if target.kind == "normal"
            and target.owner == opponent
            and not target.knocked
            and not self._target_on_cooldown(team, target.name)
        ]
        base_targets = [
            target for target in self.targets
            if target.kind == f"base_{opponent}"
            and not target.knocked
            and not self._target_on_cooldown(team, target.name)
        ]
        normal_hits = self._normal_hits_against(team)
        risk = (float(action[5]) + 1.0) * 0.5
        time_remaining = max(0.0, self.max_time_s - self.elapsed)
        low_time = time_remaining < 45.0
        early_base_commit = (
            normal_hits >= BASE_RUSH_PREFERRED_NORMAL_HITS
            or (
                normal_hits == BASE_RUSH_BALANCED_NORMAL_HITS
                and (float(action[1]) > 0.45 or low_time)
                and risk > 0.62
            )
            or (
                normal_hits == BASE_RUSH_EARLY_NORMAL_HITS
                and (float(action[1]) > 0.72 or low_time)
                and risk > 0.80
            )
        )
        if self._base_rush_open(team, action) and base_targets and float(action[1]) > -0.35 and early_base_commit:
            reachable_base_targets = [
                target for target in base_targets
                if self._best_fire_pose(team, target, (float(action[5]) + 1.0) * 0.5, route_aware=True) is not None
            ]
            if reachable_base_targets:
                return reachable_base_targets[0]
        base_gate = float(action[1])
        allow_base = (
            (
                self._normal_hits_against(team) >= BASE_RUSH_BALANCED_NORMAL_HITS
                and base_gate > 0.22
                and risk > 0.50
            )
            or self._base_rush_open(team, action)
            or low_time
        )
        candidates = list(normal_targets)
        if allow_base or not candidates:
            candidates.extend(base_targets)
        if not candidates:
            return None

        scored_candidates = [
            (self._target_priority(team, target, action, risk, route_aware=True), target)
            for target in candidates
        ]
        ranked = [
            target for priority, target in sorted(scored_candidates, key=lambda item: item[0], reverse=True)
            if priority > -50.0
        ]
        if not ranked:
            return None
        if len(ranked) == 1:
            return ranked[0]
        selector = (float(action[0]) + 1.0) * 0.5
        near_window = min(3, len(ranked))
        index = int(round(selector * (near_window - 1)))
        return ranked[max(0, min(near_window - 1, index))]

    def _target_on_cooldown(self, team: str, target_name: str) -> bool:
        if target_name.endswith("BaseTarget"):
            min_hits = int(self.base_retry_min_normal_hits.get(team, 0))
            if self._normal_hits_against(team) < min_hits:
                if (
                    self._normal_hits_against(team) >= BASE_RUSH_BALANCED_NORMAL_HITS
                    and not self._has_available_normal_retry_target(team)
                ):
                    return False
                return True
            self.lost_targets[team].discard(target_name)
            if self.target_cooldowns[team].get(target_name, -99.0) > self.max_time_s:
                self.target_cooldowns[team].pop(target_name, None)
        if target_name in self.lost_targets[team] and self.elapsed >= self.target_cooldowns[team].get(target_name, -99.0):
            self.lost_targets[team].discard(target_name)
        return target_name in self.lost_targets[team] or self.elapsed < self.target_cooldowns[team].get(target_name, -99.0)

    def _has_available_normal_retry_target(self, team: str) -> bool:
        opponent = self._opponent(team)
        for target in self.targets:
            if target.kind != "normal" or target.owner != opponent or target.knocked:
                continue
            if target.name in self.lost_targets[team] and self.elapsed >= self.target_cooldowns[team].get(target.name, -99.0):
                self.lost_targets[team].discard(target.name)
            if target.name in self.lost_targets[team] or self.elapsed < self.target_cooldowns[team].get(target.name, -99.0):
                continue
            if self._best_fire_pose(team, target, risk=0.82, route_aware=True) is not None:
                return True
        return False

    def _mark_target_failed(self, team: str, target_name: str):
        count = self.target_fail_counts[team].get(target_name, 0) + 1
        self.target_fail_counts[team][target_name] = count
        target = next((item for item in self.targets if item.name == target_name), None)
        if target_name.endswith("BaseTarget") or (target is not None and target.kind.startswith("base_")):
            self.lost_targets[team].discard(target_name)
            self.target_cooldowns[team][target_name] = self.elapsed + 4.0 + 1.5 * min(count, 4)
            return
        if target is not None and not target.knocked and not self._target_visible_in_camera(team, target):
            self.lost_targets[team].add(target_name)
            self.target_cooldowns[team][target_name] = self.elapsed + 8.0 + 3.0 * min(count, 4)
            return
        self.target_cooldowns[team][target_name] = self.elapsed + 6.0 + 2.5 * min(count, 4)

    def _refresh_target_visibility_memory(self, team: str):
        opponent = self._opponent(team)
        for target in self.targets:
            if target.owner != opponent or target.kind != "normal" or target.knocked:
                continue
            if self.target_fail_counts[team].get(target.name, 0) <= 0:
                continue
            if not self._target_visible_in_camera(team, target):
                self.lost_targets[team].add(target.name)
                self.target_cooldowns[team][target.name] = max(
                    self.target_cooldowns[team].get(target.name, -99.0),
                    self.elapsed + 3.0,
                )

    def _target_visible_in_camera(self, team: str, target: Target) -> bool:
        pose = self.poses[team]
        delta = np.asarray(target.xy, dtype=np.float32) - pose[:2]
        distance = float(np.linalg.norm(delta))
        if distance > CAMERA_MEMORY_RANGE_M or distance < 1e-6:
            return False
        bearing = math.atan2(float(delta[1]), float(delta[0]))
        if abs(wrap_angle(bearing - float(pose[2]))) > CAMERA_MEMORY_FOV_RAD * 0.5:
            return False
        return not self._line_blocked((float(pose[0]), float(pose[1])), target.xy)

    def _normal_hits_against(self, team: str) -> int:
        opponent = self._opponent(team)
        return max(0, 4 - int(self.armor[opponent]))

    def _base_rush_open(self, team: str, action: np.ndarray | None = None) -> bool:
        hits = self._normal_hits_against(team)
        if hits >= BASE_RUSH_PREFERRED_NORMAL_HITS:
            return True
        if hits < BASE_RUSH_EARLY_NORMAL_HITS:
            return False
        if action is None:
            return False
        risk = (float(action[5]) + 1.0) * 0.5
        time_remaining = max(0.0, self.max_time_s - self.elapsed)
        if hits >= BASE_RUSH_BALANCED_NORMAL_HITS:
            return (float(action[1]) > 0.42 and risk > 0.58) or time_remaining < 38.0
        return (float(action[1]) > 0.72 and risk > 0.80) or time_remaining < 30.0

    def _target_priority(
        self,
        team: str,
        target: Target,
        action: np.ndarray,
        risk: float,
        *,
        route_aware: bool,
    ) -> float:
        opponent = self._opponent(team)
        solution = self._best_fire_pose(team, target, risk, route_aware=route_aware)
        if solution is None:
            return -999.0
        _fire_xy, route_distance, shot_distance, shot_quality = solution
        priority = 1.35 - route_distance / (ARENA_SIZE * 0.90)
        priority += 0.65 * shot_quality
        priority += 0.18 * max(0.0, 1.0 - abs(shot_distance - IDEAL_SHOOT_DISTANCE) / 0.22)
        normal_hits = self._normal_hits_against(team)
        if target.kind == "normal":
            priority += 0.42 + 0.14 * min(normal_hits, BASE_RUSH_BALANCED_NORMAL_HITS)
            if normal_hits == 0 and target.name in {"T03_WestAboveGate", "T06_EastBelowGate"}:
                priority += 0.64
            if normal_hits == 1 and target.name in {"T03_WestAboveGate", "T06_EastBelowGate"}:
                priority += 0.46
            if normal_hits >= BASE_RUSH_PREFERRED_NORMAL_HITS:
                priority -= 0.28
            if normal_hits >= 4:
                priority -= 1.15
        if target.kind == f"base_{opponent}":
            cap = self._base_hit_cap_for_team(team)
            priority += 0.42 * risk + 0.40 * float(action[1]) + 0.82 * cap
            if normal_hits == BASE_RUSH_BALANCED_NORMAL_HITS:
                priority_team = getattr(self, "base_rush_priority_team", None)
                if priority_team in AGENTS:
                    priority += 0.36 if team == priority_team else -0.58
            if normal_hits >= 4:
                priority += 1.08
            elif normal_hits >= BASE_RUSH_PREFERRED_NORMAL_HITS:
                priority += 0.72
            elif normal_hits >= BASE_RUSH_BALANCED_NORMAL_HITS:
                priority -= 0.12
            elif normal_hits >= BASE_RUSH_EARLY_NORMAL_HITS:
                priority -= 0.82
            else:
                priority -= 1.20
        return priority

    def _fire_standoff_goal(self, team: str, target: Target, risk: float) -> np.ndarray:
        if self._geometry_fire_ready(team, target, risk):
            return self.poses[team][:2].copy()

        solution = self._best_fire_pose(team, target, risk, route_aware=True)
        if solution is not None:
            return solution[0]

        target_xy = np.asarray(target.xy, dtype=np.float32)
        front = np.array([math.cos(target.yaw), math.sin(target.yaw)], dtype=np.float32)
        return self._nearest_free_xy(target_xy + front * IDEAL_CENTER_STANDOFF)

    def _candidate_fire_poses(self, team: str, target: Target, risk: float) -> list[np.ndarray]:
        target_xy = np.asarray(target.xy, dtype=np.float32)
        if target.kind.startswith("base_"):
            return self._candidate_base_fire_poses(team, target, risk)

        standoff = TACTICAL_STANDOFF_MAX - (TACTICAL_STANDOFF_MAX - TACTICAL_STANDOFF_MIN) * risk

        front = np.array([math.cos(target.yaw), math.sin(target.yaw)], dtype=np.float32)
        tangent = np.array([-front[1], front[0]], dtype=np.float32)
        candidates: list[np.ndarray] = []
        for distance in (
            SHOT_CLOSE_DISTANCE,
            SHOOTER_FORWARD_OFFSET + 0.24,
            IDEAL_CENTER_STANDOFF,
            standoff,
            SHOOTER_FORWARD_OFFSET + 0.42,
            SHOOTER_FORWARD_OFFSET + SHOOT_RANGE - 0.03,
        ):
            for lateral in (0.0, -0.08, 0.08):
                candidates.append(target_xy + front * distance + tangent * lateral)

        return candidates

    def _candidate_base_fire_poses(self, team: str, target: Target, risk: float) -> list[np.ndarray]:
        target_xy = np.asarray(target.xy, dtype=np.float32)
        front = np.array([math.cos(target.yaw), math.sin(target.yaw)], dtype=np.float32)
        tangent = np.array([-front[1], front[0]], dtype=np.float32)
        wall_limit = HALF_ARENA - ROBOT_RADIUS - 0.045
        edge_offsets = (-0.16, -0.08, 0.0, 0.08, 0.16)
        candidates: list[np.ndarray] = []
        hits = max(1, min(4, self._normal_hits_against(team)))
        base_xy = BLUE_BASE_XY if target.kind == "base_blue" else YELLOW_BASE_XY
        if target.kind == "base_blue":
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
        allowed_dirs = opened_dirs[:1] if hits == 1 else [opened_dirs[1], opened_dirs[0]] if hits == 2 else opened_dirs
        side_radii = (0.62,) if hits == 1 else (0.48, 0.62, 0.78) if hits == 2 else (0.48, 0.62, 0.78, 0.94)
        side_laterals = (0.0,) if hits == 1 else (-0.06, 0.0, 0.06) if hits == 2 else (-0.10, -0.04, 0.0, 0.04, 0.10)
        fine_radial_offsets = (0.0, -0.055, -0.035, -0.018, 0.018, 0.035, 0.055) if hits >= 2 else (0.0,)
        fine_lateral_offsets = (0.0, -0.055, -0.035, -0.015, 0.015, 0.035, 0.055) if hits >= 2 else (0.0,)
        for direction in allowed_dirs:
            side_tangent = np.array([-direction[1], direction[0]], dtype=np.float32)
            for radius in side_radii:
                for lateral in side_laterals:
                    anchor = np.asarray(base_xy, dtype=np.float32) + direction * radius + side_tangent * lateral
                    candidates.append(anchor)
                    for radial_delta in fine_radial_offsets:
                        for lateral_delta in fine_lateral_offsets:
                            if abs(radial_delta) <= 1e-6 and abs(lateral_delta) <= 1e-6:
                                continue
                            candidates.append(anchor + direction * radial_delta + side_tangent * lateral_delta)

        if hits >= 2:
            if target.kind == "base_blue":
                for offset in edge_offsets:
                    candidates.append(np.array([-wall_limit, float(target_xy[1] + offset)], dtype=np.float32))
                    candidates.append(np.array([float(target_xy[0] + offset), wall_limit], dtype=np.float32))
            else:
                for offset in edge_offsets:
                    candidates.append(np.array([wall_limit, float(target_xy[1] + offset)], dtype=np.float32))
                    candidates.append(np.array([float(target_xy[0] + offset), -wall_limit], dtype=np.float32))

        # Add diagonal center-facing options for late base attacks after more
        # armor has been removed; the validation step still rejects them during
        # one- or two-target rushes.
        if hits >= 2:
            angle_offsets = {
                2: (-0.96, 0.96, -0.70, 0.70),
                3: (-0.78, 0.78, -0.44, 0.44, -0.26, 0.26),
                4: (-0.62, 0.62, -0.34, 0.34, -0.16, 0.16, 0.0),
            }[hits]
            for distance in (
                BASE_SHOT_CLOSE_DISTANCE,
                SHOOTER_FORWARD_OFFSET + 0.34,
                BASE_IDEAL_CENTER_STANDOFF,
                SHOOTER_FORWARD_OFFSET + BASE_SHOOT_RANGE - 0.04,
            ):
                for offset in angle_offsets:
                    direction = np.array(
                        [
                            math.cos(float(target.yaw) + offset),
                            math.sin(float(target.yaw) + offset),
                        ],
                        dtype=np.float32,
                    )
                    candidates.append(target_xy + direction * distance)
            for distance in (
                BASE_SHOT_CLOSE_DISTANCE,
                SHOOTER_FORWARD_OFFSET + 0.36,
                BASE_IDEAL_CENTER_STANDOFF,
                SHOOTER_FORWARD_OFFSET + BASE_SHOOT_RANGE - 0.03,
            ):
                for lateral in (0.0, -0.04, 0.04, -0.10, 0.10, -0.18, 0.18):
                    candidates.append(target_xy + front * distance + tangent * lateral)
        return candidates

    def _laser_origin_for_fire_pose(self, candidate: np.ndarray, target_xy: np.ndarray) -> tuple[float, float]:
        delta = target_xy - candidate
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-6:
            return (float(candidate[0]), float(candidate[1]))
        forward = delta / distance
        origin = candidate + forward * SHOOTER_FORWARD_OFFSET
        return (float(origin[0]), float(origin[1]))

    def _base_line_clear_with_yaw_margin(self, team: str, candidate: np.ndarray, target: Target) -> bool:
        target_xy = np.asarray(target.xy, dtype=np.float32)
        base_yaw = math.atan2(float(target_xy[1] - candidate[1]), float(target_xy[0] - candidate[0]))
        for yaw_delta in (-0.035, 0.0, 0.035):
            yaw = base_yaw + yaw_delta
            origin = (
                float(candidate[0]) + SHOOTER_FORWARD_OFFSET * math.cos(yaw),
                float(candidate[1]) + SHOOTER_FORWARD_OFFSET * math.sin(yaw),
            )
            if not self._target_line_clear(team, origin, target):
                return False
        return True

    def _valid_fire_pose_candidates(
        self,
        team: str,
        target: Target,
        risk: float,
    ) -> list[tuple[np.ndarray, float, float, float]]:
        if target.kind.startswith("base_"):
            normal_hits = self._normal_hits_against(team)
            if normal_hits < BASE_RUSH_EARLY_NORMAL_HITS:
                return []
            if normal_hits < int(self.base_retry_min_normal_hits.get(team, 0)):
                if normal_hits < BASE_RUSH_BALANCED_NORMAL_HITS or self._has_available_normal_retry_target(team):
                    return []
        risk_bucket = int(round(float(np.clip(risk, 0.0, 1.0)) * 4.0))
        cache_key = (target.name, risk_bucket, int(self.armor[target.owner]))
        cached = self._fire_pose_cache.get(cache_key)
        if cached is not None:
            return cached

        bucket_risk = risk_bucket / 4.0
        target_xy = np.asarray(target.xy, dtype=np.float32)
        valid: list[tuple[np.ndarray, float, float, float]] = []
        seen: set[tuple[int, int]] = set()
        for raw_candidate in self._candidate_fire_poses(team, target, bucket_risk):
            candidate = np.asarray(raw_candidate, dtype=np.float32)
            firing_yaw = math.atan2(float(target_xy[1] - candidate[1]), float(target_xy[0] - candidate[0]))
            pose_candidate = np.array([candidate[0], candidate[1], firing_yaw], dtype=np.float32)
            if self._pose_blocked(pose_candidate):
                candidate = self._nearest_free_xy(candidate)
                firing_yaw = math.atan2(float(target_xy[1] - candidate[1]), float(target_xy[0] - candidate[0]))
                pose_candidate = np.array([candidate[0], candidate[1], firing_yaw], dtype=np.float32)
            key = (round(float(candidate[0]) * 100), round(float(candidate[1]) * 100))
            if key in seen:
                continue
            seen.add(key)
            if self._pose_blocked(pose_candidate):
                continue
            if target.kind.startswith("base_") and self._base_attack_pose_quality(team, target, candidate) <= 0.0:
                continue
            laser_origin = self._laser_origin_for_fire_pose(candidate, target_xy)
            if not self._target_line_clear(team, laser_origin, target):
                continue
            if target.kind.startswith("base_") and not self._base_line_clear_with_yaw_margin(team, candidate, target):
                continue
            center_distance = float(np.linalg.norm(candidate - target_xy))
            shot_distance = max(0.0, center_distance - SHOOTER_FORWARD_OFFSET)
            contact_min = self._target_contact_clearance(target)
            min_range, max_range = shooting_range_limits(target.kind.startswith("base_"))
            if center_distance < contact_min or shot_distance < min_range or shot_distance > max_range:
                continue
            shot_quality = self._shot_accuracy_from_geometry(shot_distance, 0.0, target.kind.startswith("base_"))
            if target.kind.startswith("base_"):
                shot_quality *= self._base_attack_pose_quality(team, target, candidate)
            blocker_cost = self._local_blocker_cost(candidate)
            valid.append((candidate, shot_distance, shot_quality, blocker_cost))
        self._fire_pose_cache[cache_key] = valid
        return valid

    def _fire_geometry_snapshot(self, team: str, target: Target, risk: float) -> dict[str, object]:
        pose = self.poses[team]
        center_dx = target.xy[0] - float(pose[0])
        center_dy = target.xy[1] - float(pose[1])
        center_distance = math.hypot(center_dx, center_dy)
        origin = laser_origin_from_pose(pose)
        dx = target.xy[0] - origin[0]
        dy = target.xy[1] - origin[1]
        forward = (math.cos(float(pose[2])), math.sin(float(pose[2])))
        distance = dx * forward[0] + dy * forward[1]
        lateral_error = abs(dx * forward[1] - dy * forward[0])
        bearing = math.atan2(dy, dx)
        yaw_error = abs(wrap_angle(bearing - float(pose[2])))
        angle_threshold = FIRE_YAW_TOLERANCE_RAD + 0.035 * risk
        if target.kind.startswith("base_"):
            angle_threshold = min(0.24 + 0.04 * risk, 0.17 + 0.04 * self._normal_hits_against(team))
        line_clear = self._target_line_clear(team, origin, target)
        base_pose_ok = True
        if target.kind.startswith("base_"):
            base_pose_ok = self._base_attack_pose_quality(team, target, pose[:2]) > 0.0
        hit_radius = BASE_HIT_RADIUS if target.kind.startswith("base_") else SHOOT_HIT_RADIUS
        min_range, max_range = shooting_range_limits(target.kind.startswith("base_"))
        geometry_ready = bool(
            target.owner != team
            and yaw_error < angle_threshold
            and lateral_error <= hit_radius
            and center_distance >= self._target_contact_clearance(target)
            and distance >= min_range
            and distance <= max_range
            and line_clear
            and base_pose_ok
        )
        return {
            "center_distance": center_distance,
            "shot_distance": distance,
            "lateral_error": lateral_error,
            "yaw_error": yaw_error,
            "angle_threshold": angle_threshold,
            "hit_radius": hit_radius,
            "line_clear": line_clear,
            "base_pose_ok": base_pose_ok,
            "geometry_ready": geometry_ready,
        }

    def _geometry_fire_ready(self, team: str, target: Target, risk: float) -> bool:
        return bool(self._fire_geometry_snapshot(team, target, risk)["geometry_ready"])

    def _hold_fire_pose(self, team: str, target: Target, risk: float) -> bool:
        pose = self.poses[team]
        desired_yaw = math.atan2(target.xy[1] - float(pose[1]), target.xy[0] - float(pose[0]))
        geometry = self._fire_geometry_snapshot(team, target, risk)
        base_target = target.kind.startswith("base_")
        if bool(geometry["geometry_ready"]):
            shot_distance = max(float(geometry["shot_distance"]), 1e-6)
            hit_radius = float(geometry["hit_radius"])
            lateral_error = float(geometry["lateral_error"])
            margin_rad = max(0.0, hit_radius - lateral_error) / shot_distance
            max_scan = BASE_AIM_MICRO_SCAN_RAD if base_target else NORMAL_AIM_MICRO_SCAN_RAD
            scan_amp = min(max_scan, 0.45 * margin_rad)
            if scan_amp > 0.002:
                phase_offset = 0.0 if team == "yellow" else math.pi * 0.5
                frequency_hz = 0.42 if base_target else 0.58
                desired_yaw = wrap_angle(
                    desired_yaw + scan_amp * math.sin(math.tau * frequency_hz * self.elapsed + phase_offset)
                )
        elif base_target and bool(geometry["line_clear"]) and bool(geometry["base_pose_ok"]):
            min_range, max_range = shooting_range_limits(True)
            shot_distance = max(float(geometry["shot_distance"]), 1e-6)
            if min_range <= shot_distance <= max_range:
                hit_radius = float(geometry["hit_radius"])
                lateral_error = float(geometry["lateral_error"])
                # At a legal base fire pose a centimeter of pose error can leave
                # the laser just outside the small base hit radius. Keep a slow
                # deterministic search alive instead of freezing at a single yaw.
                seek_amp = min(BASE_AIM_SEEK_SCAN_RAD, max(BASE_AIM_MICRO_SCAN_RAD, 0.30 * lateral_error / shot_distance))
                if lateral_error > 0.50 * hit_radius and seek_amp > 0.002:
                    phase_offset = math.pi * 0.25 if team == "yellow" else math.pi * 0.75
                    desired_yaw = wrap_angle(
                        desired_yaw + seek_amp * math.sin(math.tau * 0.30 * self.elapsed + phase_offset)
                    )
        yaw_error = wrap_angle(desired_yaw - float(pose[2]))
        settling_deadband = 0.004 if bool(geometry["geometry_ready"]) else 0.012
        angular_speed = 0.0 if abs(yaw_error) < settling_deadband else float(np.clip(2.25 * yaw_error, -0.72, 0.72))
        return self._integrate_command(team, 0.0, angular_speed, allow_push=False)

    def _target_line_clear(self, team: str, origin: tuple[float, float], target: Target) -> bool:
        return not self._line_blocked(origin, target.xy)

    def _center_aim_line_clear(self, team: str, target: Target) -> bool:
        pose = self.poses[team].copy()
        pose[2] = math.atan2(target.xy[1] - float(pose[1]), target.xy[0] - float(pose[0]))
        return self._target_line_clear(team, laser_origin_from_pose(pose), target)

    def _best_fire_pose(
        self,
        team: str,
        target: Target,
        risk: float,
        *,
        route_aware: bool = True,
    ) -> tuple[np.ndarray, float, float, float] | None:
        pose = self.poses[team]
        scored: list[tuple[float, np.ndarray, float, float, float]] = []
        blocked_candidates: list[tuple[float, np.ndarray, float, float, float]] = []
        blocker_weight = 0.42 if target.kind.startswith("base_") else 0.20
        for candidate, shot_distance, shot_quality, blocker_cost in self._valid_fire_pose_candidates(team, target, risk):
            direct_distance = float(np.linalg.norm(candidate - pose[:2]))
            direct_blocked = self._segment_likely_blocked_for_nav(pose[:2], candidate)
            if not route_aware:
                route_distance = direct_distance + (1.65 if direct_blocked else 0.0)
                score = shot_quality - 0.16 * route_distance - blocker_weight * blocker_cost
                scored.append((score, candidate, route_distance, shot_distance, shot_quality))
                continue
            if direct_blocked:
                estimated_distance = direct_distance + 0.72 + 0.20 * blocker_cost
                quick_score = shot_quality - 0.16 * estimated_distance - blocker_weight * blocker_cost
                blocked_candidates.append((quick_score, candidate, estimated_distance, shot_distance, shot_quality))
                continue
            route_distance = direct_distance
            score = shot_quality - 0.16 * route_distance - blocker_weight * blocker_cost
            scored.append((score, candidate, route_distance, shot_distance, shot_quality))
        if route_aware and blocked_candidates:
            best_direct = max((item[0] for item in scored), default=-math.inf)
            best_blocked = max(item[0] for item in blocked_candidates)
            if not scored or best_blocked > best_direct - 0.06:
                blocked_candidates.sort(key=lambda item: item[0], reverse=True)
                for _quick_score, candidate, _estimated_distance, shot_distance, shot_quality in blocked_candidates[:6]:
                    route_distance = self._route_distance_to(pose[:2], candidate)
                    if not math.isfinite(route_distance):
                        continue
                    blocker_cost = self._local_blocker_cost(candidate)
                    score = shot_quality - 0.16 * route_distance - blocker_weight * blocker_cost
                    scored.append((score, candidate, route_distance, shot_distance, shot_quality))
        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            _score, fire_xy, route_distance, shot_distance, shot_quality = scored[0]
            return fire_xy, route_distance, shot_distance, shot_quality
        return None

    def _route_distance_to(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> float:
        resolution = 0.05

        def to_key(point: np.ndarray) -> tuple[int, int]:
            return (int(round(float(point[0]) / resolution)), int(round(float(point[1]) / resolution)))

        obstacle_signature = tuple(
            sorted((round(float(v[0]) / resolution), round(float(v[1]) / resolution)) for v in self.pushable_obstacles.values())
        )
        cache_key = (to_key(start_xy), to_key(goal_xy), obstacle_signature)
        cached = self._route_distance_cache.get(cache_key)
        if cached is not None:
            return cached

        direct = float(np.linalg.norm(goal_xy - start_xy))
        if not self._segment_blocked_for_nav(start_xy, goal_xy):
            distance = direct
        else:
            path = self._astar_path(start_xy, goal_xy)
            if len(path) < 2:
                distance = math.inf
            else:
                distance = float(np.linalg.norm(path[0] - start_xy))
                for index in range(1, len(path)):
                    distance += float(np.linalg.norm(path[index] - path[index - 1]))
        self._route_distance_cache[cache_key] = distance
        if len(self._route_distance_cache) > 2048:
            self._route_distance_cache.clear()
        return distance

    def _update_fire_gate(self, team: str, target: Target, action: np.ndarray, risk: float) -> dict[str, object]:
        geometry = self._fire_geometry_snapshot(team, target, risk)
        center_distance = float(geometry["center_distance"])
        distance = float(geometry["shot_distance"])
        lateral_error = float(geometry["lateral_error"])
        yaw_error = float(geometry["yaw_error"])
        angle_threshold = float(geometry["angle_threshold"])
        line_clear = bool(geometry["line_clear"])
        fire_gate = float(action[4])
        action_shield_fire = bool(self.action_shield and fire_gate > -0.25 and not bool(geometry["geometry_ready"]))
        if action_shield_fire:
            fire_gate = -1.0
        self.pending_fire[team] = (
            fire_gate > -0.25
            and bool(geometry["geometry_ready"])
        )
        if not self.pending_fire[team]:
            self._reset_laser_lock(team)
        return {
            "shot_distance_m": round(float(distance), 4),
            "center_target_distance_m": round(float(center_distance), 4),
            "shot_yaw_error_rad": round(float(yaw_error), 4),
            "shot_lateral_error_m": round(float(lateral_error), 4),
            "shot_accuracy_estimate": round(
                self._shot_accuracy_from_geometry(
                    distance,
                    lateral_error,
                    target.kind.startswith("base_"),
                ),
                4,
            ),
            "line_clear": line_clear,
            "fire_yaw_threshold_rad": round(float(angle_threshold), 4),
            "action_shield_fire": action_shield_fire,
        }

    def _target_contact_clearance(self, target: Target) -> float:
        target_radius = BASE_TARGET_CONTACT_RADIUS if target.kind.startswith("base_") else NORMAL_TARGET_CONTACT_RADIUS
        return ROBOT_RADIUS + target_radius + 0.004

    def _base_attack_pose_quality(self, team: str, target: Target, xy: np.ndarray) -> float:
        if not target.kind.startswith("base_"):
            return 1.0
        hits = self._normal_hits_against(team)
        if hits < BASE_RUSH_EARLY_NORMAL_HITS:
            return 0.0
        base_xy = BLUE_BASE_XY if target.kind == "base_blue" else YELLOW_BASE_XY
        return base_attack_pose_quality(hits, target.xy, target.yaw, base_xy, xy)

    def _base_hit_cap_for_team(self, team: str) -> float:
        return base_hit_success_cap(self._normal_hits_against(team))

    def _shot_accuracy_from_geometry(self, distance: float, lateral_error: float, base_target: bool) -> float:
        min_range, max_range = shooting_range_limits(base_target)
        if distance < min_range or distance > max_range:
            return 0.0
        max_lateral = BASE_HIT_RADIUS if base_target else SHOOT_HIT_RADIUS
        if lateral_error > max_lateral:
            return 0.0
        distance_quality = (max_range - distance) / max(1e-6, max_range - min_range)
        lateral_quality = 1.0 - lateral_error / max(max_lateral, 1e-6)
        # Close, centered shots are reliable; far-edge shots are intentionally
        # uncertain so the policy learns the time-vs-accuracy tradeoff.
        accuracy = 0.18 + 0.64 * distance_quality + 0.18 * lateral_quality
        if base_target:
            accuracy -= 0.10
        return float(np.clip(accuracy, 0.05, 0.98))

    def _local_blocker_cost(self, point: np.ndarray) -> float:
        x, y = float(point[0]), float(point[1])
        cost = 0.0
        for center, half_size in self.nav_blockers:
            dx = max(0.0, abs(x - center[0]) - half_size[0])
            dy = max(0.0, abs(y - center[1]) - half_size[1])
            distance = math.hypot(dx, dy)
            cost += max(0.0, 0.18 - distance) / 0.18
        for center, half_size in active_base_armor_blockers(self.armor, inflated=False):
            dx = max(0.0, abs(x - center[0]) - half_size[0])
            dy = max(0.0, abs(y - center[1]) - half_size[1])
            distance = math.hypot(dx, dy)
            cost += 1.15 * max(0.0, 0.20 - distance) / 0.20
        for center in self.pushable_obstacles.values():
            dx = max(0.0, abs(x - float(center[0])) - PUSHABLE_OBSTACLE_HALF)
            dy = max(0.0, abs(y - float(center[1])) - PUSHABLE_OBSTACLE_HALF)
            distance = math.hypot(dx, dy)
            cost += 0.55 * max(0.0, 0.18 - distance) / 0.18
        return cost

    def _nearest_free_xy(self, point: np.ndarray) -> np.ndarray:
        limit = HALF_ARENA - ROBOT_RADIUS - 0.02
        clamped = np.array(
            [
                float(np.clip(point[0], -limit, limit)),
                float(np.clip(point[1], -limit, limit)),
            ],
            dtype=np.float32,
        )
        if not self._pose_blocked(np.array([clamped[0], clamped[1], 0.0], dtype=np.float32)):
            return clamped
        for radius in (0.08, 0.14, 0.20, 0.28):
            for index in range(16):
                angle = math.tau * index / 16.0
                candidate = clamped + np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * radius
                candidate[0] = float(np.clip(candidate[0], -limit, limit))
                candidate[1] = float(np.clip(candidate[1], -limit, limit))
                if not self._pose_blocked(np.array([candidate[0], candidate[1], 0.0], dtype=np.float32)):
                    return candidate
        return clamped

    def _pose_blocked(self, pose: np.ndarray) -> bool:
        return (
            self._static_pose_blocked(pose)
            or self._pushable_collision_name(pose) is not None
            or self._target_collision_name(pose) is not None
        )

    def _point_blocked_for_nav(self, x: float, y: float) -> bool:
        if abs(x) + ROBOT_RADIUS >= HALF_ARENA or abs(y) + ROBOT_RADIUS >= HALF_ARENA:
            return True
        eps = 1e-4
        for center, half_size in self.nav_blockers:
            if abs(x - center[0]) < half_size[0] - eps and abs(y - center[1]) < half_size[1] - eps:
                return True
        for center, half_size in active_base_armor_blockers(self.armor, inflated=True):
            if abs(x - center[0]) < half_size[0] - eps and abs(y - center[1]) < half_size[1] - eps:
                return True
        # Pushable boxes are rigid contacts, not static walls. The global route
        # planner may pass through them so the local integrator can solve a
        # persistent push instead of declaring the rest of the arena unreachable.
        for target in self.targets:
            if target.knocked:
                continue
            radius = BASE_TARGET_CONTACT_RADIUS if target.kind.startswith("base_") else NORMAL_TARGET_CONTACT_RADIUS
            dx = target.xy[0] - x
            dy = target.xy[1] - y
            if dx * dx + dy * dy <= (ROBOT_RADIUS + radius) ** 2:
                return True
        return False

    def _static_pose_blocked(self, pose: np.ndarray) -> bool:
        if self._footprint_outside_arena(pose):
            return True
        x, y = float(pose[0]), float(pose[1])
        eps = 1e-4
        for center, half_size in self.nav_blockers:
            if abs(x - center[0]) < half_size[0] - eps and abs(y - center[1]) < half_size[1] - eps:
                return True
        for center, half_size in active_base_armor_blockers(self.armor, inflated=True):
            if abs(x - center[0]) < half_size[0] - eps and abs(y - center[1]) < half_size[1] - eps:
                return True
        for center, half_size in active_base_armor_blockers(self.armor, inflated=False):
            if self._pose_overlaps_aabb(pose, center, half_size, margin=0.010):
                return True
        return False

    def _footprint_outside_arena(self, pose: np.ndarray) -> bool:
        return self._arena_footprint_margin(pose) < 0.0

    def _arena_footprint_margin(self, pose: np.ndarray) -> float:
        yaw = float(pose[2])
        half_x = abs(math.cos(yaw)) * ROBOT_LENGTH * 0.5 + abs(math.sin(yaw)) * ROBOT_WIDTH * 0.5
        half_y = abs(math.sin(yaw)) * ROBOT_LENGTH * 0.5 + abs(math.cos(yaw)) * ROBOT_WIDTH * 0.5
        return min(HALF_ARENA - (abs(float(pose[0])) + half_x), HALF_ARENA - (abs(float(pose[1])) + half_y))

    def _pose_overlaps_aabb(
        self,
        pose: np.ndarray,
        center: tuple[float, float],
        half_size: tuple[float, float],
        *,
        margin: float = 0.0,
    ) -> bool:
        yaw = float(pose[2])
        half_x = abs(math.cos(yaw)) * ROBOT_LENGTH * 0.5 + abs(math.sin(yaw)) * ROBOT_WIDTH * 0.5
        half_y = abs(math.sin(yaw)) * ROBOT_LENGTH * 0.5 + abs(math.cos(yaw)) * ROBOT_WIDTH * 0.5
        return (
            abs(float(pose[0]) - center[0]) <= half_size[0] + half_x + margin
            and abs(float(pose[1]) - center[1]) <= half_size[1] + half_y + margin
        )

    def _boundary_escape_linear_speed(self, pose: np.ndarray) -> float:
        if self._arena_footprint_margin(pose) > 0.075 and self._local_blocker_cost(pose[:2]) < 0.55:
            return 0.0
        yaw = float(pose[2])
        heading = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        inward = np.array(
            [
                -math.copysign(1.0, float(pose[0])) if abs(float(pose[0])) > 1.08 else 0.0,
                -math.copysign(1.0, float(pose[1])) if abs(float(pose[1])) > 1.08 else 0.0,
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(inward))
        if norm <= 1e-6:
            return 0.0
        inward /= norm
        forward_score = float(np.dot(heading, inward))
        if abs(forward_score) < 0.18:
            return 0.0
        return 0.10 if forward_score > 0.0 else -0.10

    def _pushable_collision_name(self, pose: np.ndarray) -> str | None:
        for name, center in self.pushable_obstacles.items():
            collided, _normal, _penetration = robot_pushable_collision(
                pose,
                (float(center[0]), float(center[1])),
            )
            if collided:
                return name
        return None

    def _target_collision_name(self, pose: np.ndarray) -> str | None:
        x = float(pose[0])
        y = float(pose[1])
        for target in self.targets:
            if target.knocked:
                continue
            radius = BASE_TARGET_CONTACT_RADIUS if target.kind.startswith("base_") else NORMAL_TARGET_CONTACT_RADIUS
            dx = target.xy[0] - x
            dy = target.xy[1] - y
            if dx * dx + dy * dy <= (ROBOT_RADIUS + radius) ** 2:
                return target.name
        return None

    def _push_obstacle(self, team: str, obstacle_name: str, robot_yaw: float, robot_xy: np.ndarray) -> bool:
        heading = np.array([math.cos(robot_yaw), math.sin(robot_yaw)], dtype=np.float32)
        current = self.pushable_obstacles[obstacle_name]
        contact_normal = current - np.asarray(robot_xy, dtype=np.float32)
        contact_norm = float(np.linalg.norm(contact_normal))
        if contact_norm > 1e-6:
            contact_normal /= contact_norm
            if float(np.dot(heading, contact_normal)) < -0.10:
                return False
            direction = heading * 0.72 + contact_normal * 0.28
            direction_norm = float(np.linalg.norm(direction))
            direction = heading if direction_norm <= 1e-6 else direction / direction_norm
        else:
            direction = heading
        limit = HALF_ARENA - PUSHABLE_OBSTACLE_HALF - PUSH_CLEARANCE_MARGIN
        inflated = PUSHABLE_OBSTACLE_HALF + PUSH_CLEARANCE_MARGIN
        accepted = None
        for multiplier in (1.0, 1.7, 2.4, 3.1, 4.0):
            candidate = current + direction * (PUSH_STEP_M * self.domain_params.push_step_scale * multiplier)
            candidate = np.array(
                [
                    float(np.clip(candidate[0], -limit, limit)),
                    float(np.clip(candidate[1], -limit, limit)),
                ],
                dtype=np.float32,
            )
            robot_pose = np.array([float(robot_xy[0]), float(robot_xy[1]), float(robot_yaw)], dtype=np.float32)
            still_colliding, _normal, _penetration = robot_pushable_collision(
                robot_pose,
                (float(candidate[0]), float(candidate[1])),
                (PUSHABLE_OBSTACLE_HALF, PUSHABLE_OBSTACLE_HALF),
            )
            if still_colliding:
                continue
            blocked = False
            for center, half_size in self.nav_blockers:
                if (
                    abs(float(candidate[0]) - center[0]) <= half_size[0] + inflated
                    and abs(float(candidate[1]) - center[1]) <= half_size[1] + inflated
                ):
                    blocked = True
                    break
            if blocked:
                continue
            for target in self.targets:
                if target.knocked:
                    continue
                target_radius = BASE_TARGET_CONTACT_RADIUS if target.kind.startswith("base_") else NORMAL_TARGET_CONTACT_RADIUS
                if float(np.linalg.norm(candidate - np.asarray(target.xy, dtype=np.float32))) < inflated + target_radius:
                    blocked = True
                    break
            if blocked:
                continue
            for name, center in self.pushable_obstacles.items():
                if name == obstacle_name:
                    continue
                if float(np.linalg.norm(candidate - center)) < PUSHABLE_OBSTACLE_HALF * 2.0 + PUSH_CLEARANCE_MARGIN:
                    blocked = True
                    break
            if blocked:
                continue
            accepted = candidate
            break
        if accepted is None:
            return False
        self.pushable_obstacles[obstacle_name] = accepted
        self._fire_pose_cache.clear()
        self._path_cache.clear()
        self._route_distance_cache.clear()
        self.last_push_event[team] = obstacle_name
        self.last_push_impulse[team] = {
            "box": obstacle_name,
            "box_displacement_m": round(float(np.linalg.norm(accepted - current)), 4),
            "robot_recoil_m": PUSH_ROBOT_RECOIL_M,
        }
        return True

    def _apply_push_recoil_pose(self, pose: np.ndarray, motion_yaw: float, linear_speed: float) -> np.ndarray:
        recoil = PUSH_ROBOT_RECOIL_M * max(0.45, min(1.0, abs(float(linear_speed)) / 0.32))
        candidate = pose.copy()
        candidate[0] -= math.cos(float(motion_yaw)) * recoil
        candidate[1] -= math.sin(float(motion_yaw)) * recoil
        if self._static_pose_blocked(candidate) or self._target_collision_name(candidate) is not None:
            return pose
        return candidate

    def _separated_pose_from_pushable(self, pose: np.ndarray, obstacle_name: str) -> np.ndarray:
        center = self.pushable_obstacles[obstacle_name]
        collided, normal, penetration = robot_pushable_collision(
            pose,
            (float(center[0]), float(center[1])),
            (PUSHABLE_OBSTACLE_HALF, PUSHABLE_OBSTACLE_HALF),
        )
        if not collided:
            return pose
        corrected = pose.copy()
        corrected[0] += normal[0] * (penetration + 0.008)
        corrected[1] += normal[1] * (penetration + 0.008)
        limit = HALF_ARENA - ROBOT_RADIUS - 0.012
        corrected[0] = float(np.clip(corrected[0], -limit, limit))
        corrected[1] = float(np.clip(corrected[1], -limit, limit))
        if self._static_pose_blocked(corrected) or self._target_collision_name(corrected) is not None:
            return pose
        return corrected

    def _separated_pose_from_all_pushables(self, pose: np.ndarray) -> np.ndarray:
        corrected = pose.copy()
        for _ in range(4):
            changed = False
            for _name, center in self.pushable_obstacles.items():
                collided, normal, penetration = robot_pushable_collision(
                    corrected,
                    (float(center[0]), float(center[1])),
                    (PUSHABLE_OBSTACLE_HALF, PUSHABLE_OBSTACLE_HALF),
                )
                if not collided:
                    continue
                candidate = corrected.copy()
                candidate[0] += normal[0] * (penetration + 0.012)
                candidate[1] += normal[1] * (penetration + 0.012)
                limit = HALF_ARENA - ROBOT_RADIUS - 0.012
                candidate[0] = float(np.clip(candidate[0], -limit, limit))
                candidate[1] = float(np.clip(candidate[1], -limit, limit))
                if self._static_pose_blocked(candidate) or self._target_collision_name(candidate) is not None:
                    continue
                corrected = candidate
                changed = True
            if not changed:
                break
        return corrected

    def _resolve_contact(self) -> bool:
        delta = self.poses["blue"][:2] - self.poses["yellow"][:2]
        distance = float(np.linalg.norm(delta))
        min_distance = ROBOT_RADIUS * 2.0
        if distance >= min_distance:
            self.last_contact = False
            return False
        normal = np.array([1.0, 0.0], dtype=np.float32) if distance < 1e-6 else delta / distance
        push = (min_distance - max(distance, 1e-6)) * 0.5 + 0.004
        before = {team: self.poses[team].copy() for team in AGENTS}
        yellow_candidate = before["yellow"].copy()
        blue_candidate = before["blue"].copy()
        yellow_candidate[:2] -= normal * push
        blue_candidate[:2] += normal * push
        self.poses["yellow"] = self._safe_contact_separation_pose(before["yellow"], yellow_candidate)
        self.poses["blue"] = self._safe_contact_separation_pose(before["blue"], blue_candidate)
        self.last_contact = True
        return True

    def _safe_contact_separation_pose(self, before: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        if not self._pose_blocked(candidate):
            return candidate
        repaired = candidate.copy()
        repaired[:2] = self._nearest_free_xy(candidate[:2])
        if not self._pose_blocked(repaired):
            return repaired
        return before

    def _contact_reward(self, team: str, info: dict[str, object]) -> float:
        opponent = self._opponent(team)
        own_base = YELLOW_BASE_XY if team == "yellow" else BLUE_BASE_XY
        opponent_distance_to_own_base = float(np.linalg.norm(self.poses[opponent][:2] - own_base))
        own_distance_to_own_base = float(np.linalg.norm(self.poses[team][:2] - own_base))
        tactical_intent = info.get("tactic") == "block" or bool(info.get("interference"))
        if tactical_intent and not self._near_own_critical_assets(team):
            threat = self._opponent_threat(team)
            return 0.09 + 0.16 * threat
        if opponent_distance_to_own_base < 0.85 and own_distance_to_own_base > 0.45:
            return 0.08
        if own_distance_to_own_base < 0.55:
            return -0.55
        return -0.035

    def _near_own_critical_assets(self, team: str) -> bool:
        own_base = YELLOW_BASE_XY if team == "yellow" else BLUE_BASE_XY
        if float(np.linalg.norm(self.poses[team][:2] - own_base)) < 0.72:
            return True
        for target in self.targets:
            if target.owner != team or target.knocked:
                continue
            critical_radius = 0.34 if target.kind.startswith("base_") else 0.24
            if float(np.linalg.norm(self.poses[team][:2] - np.asarray(target.xy, dtype=np.float32))) < critical_radius:
                return True
        return False

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
            target_radius = BASE_TARGET_CONTACT_RADIUS if target.kind.startswith("base_") else NORMAL_TARGET_CONTACT_RADIUS
            distance = float(np.linalg.norm(np.array(target.xy, dtype=np.float32) - pose_xy))
            if distance > ROBOT_RADIUS + target_radius:
                continue
            infos[team]["target_collision"] = target.name
            if self.elapsed - self.last_target_contact_time[team] < 0.75:
                return
            self.last_target_contact_time[team] = self.elapsed
            self.post_hit_retreat_until[team] = self.elapsed + POST_HIT_RETREAT_S
            self._mark_target_failed(team, target.name)
            rewards[team] -= 1.2
            self.localization_confidence[team] = max(0.05, self.localization_confidence[team] - 0.03)
            if target.kind == f"base_{team}":
                infos[team]["own_base_collision"] = True
                rewards[team] -= 8.0
                return
            if target.kind.startswith("base_"):
                rewards[team] -= 3.0
                return
            rewards[team] -= 1.5

    def _apply_fire(self, team: str) -> ShotResult | None:
        target = self._detect_laser_hit(team)
        if target is None:
            return None
        return ShotResult(team, target.name, target.owner, target.kind)

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

    def _detect_laser_hit(self, team: str) -> Target | None:
        pose = self.poses[team]
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
            if (
                target.kind.startswith("base_")
                and target.owner != team
                and self._normal_hits_against(team) < BASE_RUSH_EARLY_NORMAL_HITS
            ):
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
            if not self._target_line_clear(team, origin, target):
                continue
            if target.owner == team:
                own_candidate_projection = min(own_candidate_projection, projection)
                continue
            accuracy = self._shot_accuracy_from_geometry(projection, perpendicular, target.kind.startswith("base_"))
            if target.kind.startswith("base_"):
                pose_quality = self._base_attack_pose_quality(team, target, pose[:2])
                if pose_quality <= 0.0:
                    continue
                accuracy *= pose_quality
            if projection < best_projection:
                best_projection = projection
                best_target = target
                best_accuracy = accuracy
                best_lateral_error = perpendicular
        if own_candidate_projection <= max(SHOOT_RANGE, BASE_SHOOT_RANGE) and own_candidate_projection <= best_projection:
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
        final_accuracy = float(
            np.clip(best_accuracy * dwell_factor * self.domain_params.shot_accuracy_scale, 0.0, 0.95)
        )
        if best_target.kind.startswith("base_"):
            normal_hits = max(0, min(4, self._normal_hits_against(team)))
            if normal_hits < BASE_RUSH_PREFERRED_NORMAL_HITS:
                lottery_key = (best_target.name, normal_hits)
                if lottery_key not in self.base_rush_lottery[team]:
                    self.base_rush_lottery[team][lottery_key] = bool(
                        self.base_cap_rng[team].random() <= self._base_hit_cap_for_team(team)
                    )
                base_cap_passed = self.base_rush_lottery[team][lottery_key]
            else:
                base_cap_passed = bool(self.base_cap_rng[team].random() <= self._base_hit_cap_for_team(team))
            if not base_cap_passed:
                self.target_cooldowns[team][best_target.name] = max(
                    self.target_cooldowns[team].get(best_target.name, -99.0),
                    self.elapsed + 14.0,
                )
                self.last_shot_attempt[team] = {
                    "hit": False,
                    "reason": "base_cap_failed",
                    "target": best_target.name,
                    "dwell_s": round(float(dwell_s), 3),
                    "distance_m": round(float(best_projection), 4),
                    "lateral_error_m": round(float(best_lateral_error), 4),
                    "geometry_accuracy": round(float(best_accuracy), 4),
                    "dwell_factor": round(float(dwell_factor), 4),
                    "accuracy": 0.0,
                    "base_hit_cap": round(float(self._base_hit_cap_for_team(team)), 4),
                }
                self._reset_laser_lock(team)
                return None
        hit = bool(self.shot_rng[team].random() <= final_accuracy)
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
            "base_hit_cap": round(float(self._base_hit_cap_for_team(team)), 4)
            if best_target.kind.startswith("base_")
            else "",
        }
        if hit:
            self._reset_laser_lock(team)
            return best_target
        return None

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
        *,
        terminal_override: bool = True,
    ):
        shooter = result.shooter
        opponent = self._opponent(shooter)
        target = next(t for t in self.targets if t.name == result.target_name)

        if result.target_owner == shooter and result.kind == f"base_{shooter}":
            rewards[shooter] -= 1.0
            infos[shooter]["own_base_blocked"] = True
            return
        if result.target_owner == shooter:
            rewards[shooter] -= 1.0
            infos[shooter]["own_target_blocked"] = result.target_name
            return
        target.knocked = True
        if result.kind == "normal":
            self.armor[opponent] = max(0, self.armor[opponent] - 1)
            self._fire_pose_cache.clear()
            self._path_cache.clear()
            self._route_distance_cache.clear()
            opponent_base_name = "BlueBaseTarget" if opponent == "blue" else "YellowBaseTarget"
            if self._normal_hits_against(shooter) >= int(self.base_retry_min_normal_hits.get(shooter, 0)):
                self.lost_targets[shooter].discard(opponent_base_name)
                self.target_cooldowns[shooter].pop(opponent_base_name, None)
            self.scores[shooter] += 5
            normal_hits_after = self._normal_hits_against(shooter)
            armor_break_bonus = (
                5.0
                if normal_hits_after == BASE_RUSH_PREFERRED_NORMAL_HITS
                else 3.2
                if normal_hits_after == BASE_RUSH_BALANCED_NORMAL_HITS
                else 1.8
                if normal_hits_after == BASE_RUSH_EARLY_NORMAL_HITS
                else 0.0
            )
            over_clear_penalty = 3.5 if normal_hits_after > BASE_RUSH_PREFERRED_NORMAL_HITS else 0.0
            rewards[shooter] += 10.0 + armor_break_bonus - over_clear_penalty
            rewards[opponent] -= 3.5
            self.strategy_counts[shooter]["normal_hits"] += 1
            self.target_order[shooter].append(target.name)
            self.post_hit_retreat_until[shooter] = self.elapsed + POST_HIT_RETREAT_S
            infos[shooter]["hit"] = result.target_name
            infos[shooter]["target_order"] = list(self.target_order[shooter])
            infos[shooter]["normal_hit_count"] = self.strategy_counts[shooter]["normal_hits"]
            return
        if result.kind == f"base_{opponent}":
            self.scores[shooter] += 60
            if terminal_override:
                self.winner = shooter
            self.strategy_counts[shooter]["base_hits"] += 1
            self.target_order[shooter].append(target.name)
            rewards[shooter] += 100.0
            rewards[opponent] -= 70.0
            if terminal_override:
                infos[shooter]["winner"] = shooter
            infos[shooter]["target_order"] = list(self.target_order[shooter])

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
        fire_solutions = [
            self._best_fire_pose(team, target, risk=0.35, route_aware=False)
            for target in active_opponent_normals
        ]
        reachable_fire_xy = [solution[0] for solution in fire_solutions if solution is not None]
        if reachable_fire_xy:
            nearest_fire_xy = min(reachable_fire_xy, key=lambda xy: np.linalg.norm(xy - own[:2]))
            nearest_vec = self._team_vector(team, nearest_fire_xy - own[:2]) / ARENA_SIZE
        else:
            nearest_vec = np.zeros(2, dtype=np.float32)
        knocked_flags = self._canonical_target_flags(team, normal_targets)
        rel_opponent = self._team_vector(team, other[:2] - own[:2]) / ARENA_SIZE
        opponent_track = self._opponent_tracking_features(team, own, other)
        pushable_items = sorted(
            self.pushable_obstacles.items(),
            key=lambda item: tuple(round(float(v), 4) for v in self._team_vector(team, item[1] - own[:2])),
        )
        pushable_vectors = np.concatenate(
            [
                self._team_vector(team, center - own[:2]) / ARENA_SIZE
                for _name, center in pushable_items
            ]
        ).astype(np.float32)
        own_xy = self._team_point(team, own[:2])
        own_yaw = self._team_yaw(team, float(own[2]))
        other_yaw = self._team_yaw(team, float(other[2]))
        obs = np.concatenate(
            [
                np.array([own_xy[0] / HALF_ARENA, own_xy[1] / HALF_ARENA, math.cos(own_yaw), math.sin(own_yaw)]),
                rel_opponent,
                opponent_track,
                np.array([math.cos(other_yaw), math.sin(other_yaw)]),
                np.array([self.armor[opponent] / 4.0, self.armor[team] / 4.0]),
                np.array([(self.scores[team] - self.scores[opponent]) / 60.0, self.elapsed / self.max_time_s]),
                self._sensor_fusion_features(team),
                knocked_flags,
                nearest_vec,
                pushable_vectors,
                self._team_vector(team, opponent_base - own[:2]) / ARENA_SIZE,
                self._team_vector(team, own_base - own[:2]) / ARENA_SIZE,
                np.array([1.0 if self.winner == team else -1.0 if self.winner == opponent else 0.0]),
                np.array([1.0 if team == "yellow" else -1.0]),
            ]
        ).astype(np.float32)
        return obs

    def _default_sensor_fusion_state(self) -> dict[str, float]:
        return {
            "wheel_imu_consistency": 1.0,
            "scan_clearance": 1.0,
            "tof_front_left_clearance": 1.0,
            "tof_front_right_clearance": 1.0,
            "bumper_or_hard_contact": 0.0,
            "camera_target_visible": 1.0,
            "pushable_contact": 0.0,
        }

    def _sensor_fusion_features(self, team: str) -> np.ndarray:
        fusion = self.sensor_fusion[team]
        if self.domain_params.sensor_noise_scale > 0.0:
            noise = self.domain_params.sensor_noise_scale
            fusion = {
                key: (
                    float(np.clip(value + self.rng.normal(0.0, noise), 0.0, 1.0))
                    if key not in ("bumper_or_hard_contact", "pushable_contact")
                    else value
                )
                for key, value in fusion.items()
            }
        return np.array(
            [
                float(self.last_contact),
                float(self.localization_confidence[team]),
                float(fusion["wheel_imu_consistency"]),
                float(fusion["scan_clearance"]),
                float(fusion["tof_front_left_clearance"]),
                float(fusion["tof_front_right_clearance"]),
                float(fusion["bumper_or_hard_contact"]),
                float(fusion["camera_target_visible"]),
                float(fusion["pushable_contact"]),
            ],
            dtype=np.float32,
        )

    def _record_motion_sensor_fusion(
        self,
        team: str,
        before: np.ndarray,
        after: np.ndarray,
        linear_speed: float,
        angular_speed: float,
        *,
        blocked: bool,
        hard_contact: bool = False,
        push_contact: bool = False,
        jammed_push: bool = False,
    ):
        expected_distance = abs(float(linear_speed)) * self.dt
        actual_distance = float(np.linalg.norm(after[:2] - before[:2]))
        expected_yaw = abs(float(angular_speed)) * self.dt
        actual_yaw = abs(wrap_angle(float(after[2] - before[2])))
        translation_error = abs(expected_distance - actual_distance) / max(0.035, expected_distance + 0.02)
        yaw_error = abs(expected_yaw - actual_yaw) / max(0.060, expected_yaw + 0.03)
        consistency = float(np.clip(1.0 - 0.62 * translation_error - 0.38 * yaw_error, 0.0, 1.0))
        scan_clearance = self._scan_clearance_score(after)
        tof_left = self._tof_clearance_score(after, TOF_SENSOR_LATERAL_OFFSET_M)
        tof_right = self._tof_clearance_score(after, -TOF_SENSOR_LATERAL_OFFSET_M)
        camera_visible = self._camera_visible_opponent_target_score(team)

        fusion = self.sensor_fusion[team]
        alpha = 0.72
        fusion["wheel_imu_consistency"] = alpha * fusion["wheel_imu_consistency"] + (1.0 - alpha) * consistency
        fusion["scan_clearance"] = alpha * fusion["scan_clearance"] + (1.0 - alpha) * scan_clearance
        fusion["tof_front_left_clearance"] = alpha * fusion["tof_front_left_clearance"] + (1.0 - alpha) * tof_left
        fusion["tof_front_right_clearance"] = alpha * fusion["tof_front_right_clearance"] + (1.0 - alpha) * tof_right
        fusion["camera_target_visible"] = alpha * fusion["camera_target_visible"] + (1.0 - alpha) * camera_visible
        fusion["bumper_or_hard_contact"] = 1.0 if hard_contact else 0.0
        fusion["pushable_contact"] = 1.0 if push_contact else 0.0

        previous_linear, previous_angular = self.last_motion_command.get(team, (0.0, 0.0))
        if self.dt > 1e-6:
            linear_accel = abs(float(linear_speed) - previous_linear) / self.dt
            angular_accel = abs(float(angular_speed) - previous_angular) / self.dt
        else:
            linear_accel = 0.0
            angular_accel = 0.0
        self.last_motion_command[team] = (float(linear_speed), float(angular_speed))
        linear_excess = max(0.0, linear_accel - ACCEL_DRIFT_LINEAR_THRESHOLD) / ACCEL_DRIFT_LINEAR_THRESHOLD
        angular_excess = max(0.0, angular_accel - ACCEL_DRIFT_ANGULAR_THRESHOLD) / ACCEL_DRIFT_ANGULAR_THRESHOLD
        accel_drift_loss = (
            ACCEL_DRIFT_LOSS_SCALE
            * self.domain_params.drift_loss_scale
            * (0.62 * linear_excess + 0.38 * angular_excess)
        )
        if accel_drift_loss > 0.0:
            fusion["wheel_imu_consistency"] = max(0.0, fusion["wheel_imu_consistency"] - 0.35 * accel_drift_loss)

        if self.last_contact:
            return
        fused_quality = (
            0.40 * fusion["wheel_imu_consistency"] +
            0.25 * fusion["scan_clearance"] +
            0.20 * min(fusion["tof_front_left_clearance"], fusion["tof_front_right_clearance"]) +
            0.15 * fusion["camera_target_visible"]
        )
        confidence_delta = FUSION_CONFIDENCE_RECOVERY_GAIN * max(0.0, fused_quality - 0.45)
        if fused_quality < 0.24:
            confidence_delta -= FUSION_CONFIDENCE_DRIFT_LOSS * (0.24 - fused_quality) / 0.24
        if blocked or hard_contact:
            confidence_delta -= FUSION_HARD_CONTACT_LOSS
        if jammed_push:
            confidence_delta -= FUSION_JAMMED_PUSH_LOSS
        confidence_delta -= accel_drift_loss
        self.localization_confidence[team] = float(
            np.clip(self.localization_confidence[team] + confidence_delta, 0.05, 1.0)
        )

    def _boost_sensor_fusion_recovery(self, team: str):
        fusion = self.sensor_fusion[team]
        for key in ("wheel_imu_consistency", "scan_clearance", "tof_front_left_clearance", "tof_front_right_clearance"):
            fusion[key] = min(1.0, 0.60 * fusion[key] + 0.40)
        fusion["bumper_or_hard_contact"] = 0.0

    def _can_relocalize(self, team: str) -> bool:
        return self.elapsed - float(self.last_relocalization_time[team]) >= RECOVERY_COOLDOWN_S

    def _scan_clearance_score(self, pose: np.ndarray) -> float:
        margin = max(0.0, self._arena_footprint_margin(pose))
        nearest = min(margin, 0.45)
        for center, half_size in self.nav_blockers:
            dx = max(0.0, abs(float(pose[0]) - center[0]) - half_size[0])
            dy = max(0.0, abs(float(pose[1]) - center[1]) - half_size[1])
            nearest = min(nearest, math.hypot(dx, dy))
        for center, half_size in active_base_armor_blockers(self.armor, inflated=False):
            dx = max(0.0, abs(float(pose[0]) - center[0]) - half_size[0])
            dy = max(0.0, abs(float(pose[1]) - center[1]) - half_size[1])
            nearest = min(nearest, math.hypot(dx, dy))
        for center in self.pushable_obstacles.values():
            dx = max(0.0, abs(float(pose[0]) - float(center[0])) - PUSHABLE_OBSTACLE_HALF)
            dy = max(0.0, abs(float(pose[1]) - float(center[1])) - PUSHABLE_OBSTACLE_HALF)
            nearest = min(nearest, math.hypot(dx, dy))
        return float(np.clip(nearest / 0.45, 0.0, 1.0))

    def _tof_clearance_score(self, pose: np.ndarray, lateral_offset: float) -> float:
        yaw = float(pose[2])
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        lateral = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
        origin = pose[:2] + lateral * lateral_offset + forward * (ROBOT_LENGTH * 0.38)
        samples = 8
        for index in range(1, samples + 1):
            distance = TOF_SENSOR_RANGE_M * index / samples
            point = origin + forward * distance
            candidate = np.array([point[0], point[1], yaw], dtype=np.float32)
            if self._pose_blocked(candidate):
                return float(np.clip(distance / TOF_SENSOR_RANGE_M, 0.0, 1.0))
        return 1.0

    def _camera_visible_opponent_target_score(self, team: str) -> float:
        opponent = self._opponent(team)
        candidates = [
            target for target in self.targets
            if target.owner == opponent and not target.knocked
        ]
        if not candidates:
            return 0.0
        return 1.0 if any(self._target_visible_in_camera(team, target) for target in candidates) else 0.0

    def _team_point(self, team: str, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, dtype=np.float32) * team_frame_sign(team)

    def _team_vector(self, team: str, vector: np.ndarray) -> np.ndarray:
        return np.asarray(vector, dtype=np.float32) * team_frame_sign(team)

    def _team_yaw(self, team: str, yaw: float) -> float:
        return wrap_angle(yaw if team == "yellow" else yaw + math.pi)

    def _canonical_target_flags(self, team: str, normal_targets: list[Target]) -> np.ndarray:
        opponent = self._opponent(team)

        def key(target: Target) -> tuple[int, float, float]:
            xy = self._team_point(team, np.asarray(target.xy, dtype=np.float32))
            owner_group = 0 if target.owner == opponent else 1
            return owner_group, round(float(xy[0]), 4), round(float(xy[1]), 4)

        ordered = sorted(normal_targets, key=key)
        return np.array([1.0 if target.knocked else 0.0 for target in ordered], dtype=np.float32)

    def _base_distance(self, team: str) -> float:
        opponent_base = BLUE_BASE_XY if team == "yellow" else YELLOW_BASE_XY
        return float(np.linalg.norm(self.poses[team][:2] - opponent_base))

    def _nearest_opponent_target_distance(self, team: str) -> float:
        opponent = self._opponent(team)
        pose = self.poses[team]
        candidates = [
            target for target in self.targets
            if target.owner == opponent and not target.knocked
        ]
        if not candidates:
            return 0.0
        return min(float(np.linalg.norm(np.asarray(target.xy, dtype=np.float32) - pose[:2])) for target in candidates)

    def _opponent_threat(self, team: str) -> float:
        opponent = self._opponent(team)
        own_base = YELLOW_BASE_XY if team == "yellow" else BLUE_BASE_XY
        other = self.poses[opponent]
        base_delta = own_base - other[:2]
        base_distance = float(np.linalg.norm(base_delta))
        base_bearing = math.atan2(float(base_delta[1]), float(base_delta[0])) if base_distance > 1e-6 else float(other[2])
        heading_error = abs(wrap_angle(base_bearing - float(other[2])))
        proximity = max(0.0, 1.0 - base_distance / 1.10)
        heading = max(0.0, 1.0 - heading_error / math.pi)
        visible = not self._line_blocked((float(self.poses[team][0]), float(self.poses[team][1])), (float(other[0]), float(other[1])))
        return max(0.0, min(1.0, proximity * (0.55 + 0.45 * heading) * (1.0 if visible else 0.72)))

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
