from __future__ import annotations

import math

import numpy as np

from expert_policy import blue_expert_action
from robocup_visionrl_gym_env import (
    BASE_ARMOR_SPECS,
    BASE_ARMOR_SIZE,
    BASE_HIT_RADIUS,
    BASE_HIT_SUCCESS_BY_NORMAL_HITS,
    BASE_SHOOT_MIN_RANGE,
    BASE_SHOOT_RANGE,
    BLUE_BASE_TARGET_YAW,
    BLUE_BASE_TARGET_XY,
    BLUE_BASE_XY,
    BLUE_START,
    LASER_DWELL_REQUIRED_S,
    NORTH_MIDDLE_TARGET_X,
    NORMAL_TARGET_CONTACT_RADIUS,
    PUSHABLE_OBSTACLE_HALF,
    PUSHABLE_OBSTACLE_RANDOM_JITTER,
    PUSHABLE_OBSTACLE_STARTS,
    ROBOT_PUSHABLE_CLEARANCE_RADIUS,
    ROBOT_RADIUS,
    SIDE_GATE_TARGET_Y,
    SHOOT_HIT_RADIUS,
    SHOOTER_FORWARD_OFFSET,
    SHOOT_IDEAL_DISTANCE,
    SOUTH_MIDDLE_TARGET_X,
    TARGET_WALL_INSET,
    YELLOW_BASE_TARGET_YAW,
    YELLOW_BASE_TARGET_XY,
    YELLOW_BASE_XY,
    YELLOW_START,
    RoboCupVisionRLGymEnv,
    active_base_armor_blockers,
    base_hit_success_cap,
    base_removed_side_lane_quality,
    base_attack_pose_quality,
    segment_intersects_aabb,
)
from robocup_visionrl_selfplay_env import (
    AGENTS,
    ROBOT_PUSHABLE_VISUAL_HALF_EXTENTS,
    TACTICAL_ACTION_DIM,
    RoboCupVisionRLSelfPlayEnv,
    oriented_rect_aabb_collision,
)
from robocup_visionrl_gym_env import Target
from robocup_visionrl_selfplay_vec import RoboCupVisionRLSelfPlayVector


def test_single_agent_rule_env_step():
    env = RoboCupVisionRLGymEnv()
    obs, info = env.reset(seed=7)

    assert obs.shape == env.observation_space.shape
    assert info == {}

    obs, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))

    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_selfplay_env_step_contract():
    env = RoboCupVisionRLSelfPlayEnv()
    observations, infos = env.reset(seed=11)

    assert set(observations) == set(AGENTS)
    assert set(infos) == set(AGENTS)

    actions = {team: np.zeros(TACTICAL_ACTION_DIM, dtype=np.float32) for team in AGENTS}
    observations, rewards, terminations, truncations, infos = env.step(actions)

    assert set(observations) == set(AGENTS)
    assert set(rewards) == set(AGENTS)
    assert set(terminations) == set(AGENTS)
    assert set(truncations) == set(AGENTS)
    assert set(infos) == set(AGENTS)


def test_selfplay_env_keeps_legacy_action_compatibility():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=13)

    actions = {team: np.zeros(3, dtype=np.float32) for team in AGENTS}
    observations, rewards, terminations, truncations, infos = env.step(actions)

    assert set(observations) == set(AGENTS)
    assert set(rewards) == set(AGENTS)
    assert set(terminations) == set(AGENTS)
    assert set(truncations) == set(AGENTS)
    assert all(info["action_labels"] for info in infos.values())


def test_vectorized_selfplay_smoke():
    vec = RoboCupVisionRLSelfPlayVector(num_envs=4, seed=3)
    observations, infos = vec.reset()

    assert len(observations) == 4
    assert len(infos) == 4

    actions = [
        {team: vec.envs[index].action_spaces[team].sample() for team in AGENTS}
        for index in range(vec.num_envs)
    ]
    outputs = vec.step(actions)
    assert all(len(item) == 4 for item in outputs)


def test_selfplay_observation_exposes_opponent_bearing():
    env = RoboCupVisionRLSelfPlayEnv()
    observations, _ = env.reset(seed=11)

    yellow_obs = observations["yellow"]
    blue_obs = observations["blue"]

    np.testing.assert_allclose(yellow_obs[:34], blue_obs[:34], atol=1e-2)
    np.testing.assert_allclose(yellow_obs[36:-1], blue_obs[36:-1], atol=1e-2)
    assert yellow_obs[-1] == 1.0
    assert blue_obs[-1] == -1.0
    assert np.isfinite(yellow_obs[6:11]).all()
    assert yellow_obs[6] > 0.0
    assert -1.0 <= yellow_obs[7] <= 1.0
    assert -1.0 <= yellow_obs[8] <= 1.0
    assert yellow_obs[9] in (0.0, 1.0)
    assert 0.0 <= yellow_obs[10] <= 1.0
    assert yellow_obs.shape[0] == 46


def test_selfplay_observation_exposes_multisensor_fusion_state():
    env = RoboCupVisionRLSelfPlayEnv()
    observations, _ = env.reset(seed=12)

    yellow_obs = observations["yellow"]
    fusion = yellow_obs[19:28]

    assert fusion.shape[0] == 9
    assert 0.0 <= fusion[1] <= 1.0
    assert 0.0 <= fusion[2] <= 1.0
    assert 0.0 <= fusion[3] <= 1.0
    assert 0.0 <= fusion[4] <= 1.0
    assert 0.0 <= fusion[5] <= 1.0

    env.poses["yellow"] = np.array([1.48, 0.0, 0.0], dtype=np.float32)
    before = env.localization_confidence["yellow"]
    blocked = env._integrate_command("yellow", 0.30, 0.0)

    assert blocked is True
    assert env.sensor_fusion["yellow"]["bumper_or_hard_contact"] == 1.0
    assert env.localization_confidence["yellow"] < before


def test_selfplay_initial_attack_geometry_is_side_symmetric():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=19)
    env.pushable_obstacles = {}
    env._fire_pose_cache.clear()
    env._path_cache.clear()
    env._route_distance_cache.clear()

    def summaries(team: str):
        opponent = env._opponent(team)
        rows = []
        for target in env.targets:
            if target.kind != "normal" or target.owner != opponent:
                continue
            solution = env._best_fire_pose(team, target, risk=0.35)
            rows.append(None if solution is None else (round(float(solution[1]), 3), round(float(solution[2]), 3)))
        return sorted(row for row in rows if row is not None)

    yellow_rows = summaries("yellow")
    blue_rows = summaries("blue")
    assert len(yellow_rows) == len(blue_rows)
    for yellow_row, blue_row in zip(yellow_rows, blue_rows):
        np.testing.assert_allclose(yellow_row, blue_row, atol=0.03)

    yellow_base = next(target for target in env.targets if target.kind == "base_blue")
    blue_base = next(target for target in env.targets if target.kind == "base_yellow")
    assert (env._best_fire_pose("yellow", yellow_base, risk=0.35) is None) == (
        env._best_fire_pose("blue", blue_base, risk=0.35) is None
    )
    assert env._best_fire_pose("yellow", yellow_base, risk=0.35) is None

    env.armor["blue"] = 2
    env.armor["yellow"] = 2
    env._fire_pose_cache.clear()
    assert env._best_fire_pose("yellow", yellow_base, risk=0.35) is not None
    assert env._best_fire_pose("blue", blue_base, risk=0.35) is not None


def test_base_targets_are_small_and_recessed_behind_armor():
    blue_inset = np.linalg.norm(BLUE_BASE_XY - BLUE_BASE_TARGET_XY)
    yellow_inset = np.linalg.norm(YELLOW_BASE_XY - YELLOW_BASE_TARGET_XY)

    np.testing.assert_allclose(blue_inset, yellow_inset, atol=1e-5)
    assert 0.14 <= blue_inset <= 0.18
    assert BASE_HIT_RADIUS < SHOOT_HIT_RADIUS

    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=19)
    yellow_base = next(target for target in env.targets if target.kind == "base_blue")
    blue_base = next(target for target in env.targets if target.kind == "base_yellow")

    for remaining in (4, 3):
        env.armor["blue"] = remaining
        env.armor["yellow"] = remaining
        env._fire_pose_cache.clear()
        yellow_valid = env._valid_fire_pose_candidates("yellow", yellow_base, risk=0.35)
        blue_valid = env._valid_fire_pose_candidates("blue", blue_base, risk=0.35)
        if remaining == 4:
            assert yellow_valid == []
            assert blue_valid == []
        else:
            assert 1 <= len(yellow_valid) <= 3
            assert 1 <= len(blue_valid) <= 3
            assert all(base_removed_side_lane_quality(1, BLUE_BASE_XY, item[0]) > 0.0 for item in yellow_valid)
            assert all(base_removed_side_lane_quality(1, YELLOW_BASE_XY, item[0]) > 0.0 for item in blue_valid)


def test_base_shot_range_and_rush_caps_are_rule_hardened():
    assert BASE_SHOOT_MIN_RANGE == 0.20
    assert BASE_SHOOT_RANGE == 0.80
    assert BASE_HIT_SUCCESS_BY_NORMAL_HITS[1] <= 0.40
    assert BASE_HIT_SUCCESS_BY_NORMAL_HITS[2] <= 0.55
    assert base_hit_success_cap(1) <= 0.40
    assert base_hit_success_cap(2) <= 0.55


def test_base_rush_requires_explicit_gate_not_only_score_deficit():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=22)
    env.armor["blue"] = 2
    env.scores["yellow"] = 0
    env.scores["blue"] = 15

    desperate_but_not_committed = np.array([0.0, 0.34, 0.0, -0.5, 0.8, 0.58], dtype=np.float32)
    committed_two_hit_rush = np.array([0.0, 0.52, 0.0, -0.5, 0.8, 0.30], dtype=np.float32)

    assert env._base_rush_open("yellow", desperate_but_not_committed) is False
    assert env._base_rush_open("yellow", committed_two_hit_rush) is True

    env.armor["blue"] = 3
    committed_one_hit_rush = np.array([0.0, 0.80, 0.0, -0.5, 0.8, 0.70], dtype=np.float32)
    assert env._base_rush_open("yellow", desperate_but_not_committed) is False
    assert env._base_rush_open("yellow", committed_one_hit_rush) is True


def test_score_deficit_does_not_insert_base_into_target_candidates_too_early():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=23)
    env.armor["blue"] = 2
    env.scores["yellow"] = 0
    env.scores["blue"] = 15
    action = np.array([-0.92, 0.12, -0.28, -0.64, -0.72, 0.58], dtype=np.float32)

    target = env._select_tactical_target("yellow", action)

    assert target is not None
    assert target.kind == "normal"


def test_early_base_attack_must_use_removed_armor_side():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=25)
    base = next(target for target in env.targets if target.kind == "base_blue")

    east_open_pose = np.array([-0.55, 1.25], dtype=np.float32)
    south_unopened_pose = np.array([-1.25, 0.55], dtype=np.float32)

    assert base_removed_side_lane_quality(1, BLUE_BASE_XY, east_open_pose) > 0.0
    assert base_removed_side_lane_quality(1, BLUE_BASE_XY, south_unopened_pose) == 0.0
    assert base_attack_pose_quality(1, base.xy, base.yaw, BLUE_BASE_XY, east_open_pose) > 0.0
    assert base_attack_pose_quality(1, base.xy, base.yaw, BLUE_BASE_XY, south_unopened_pose) == 0.0

    env.armor["blue"] = 3
    env._fire_pose_cache.clear()
    valid = env._valid_fire_pose_candidates("yellow", base, risk=1.0)
    assert all(base_removed_side_lane_quality(1, BLUE_BASE_XY, item[0]) > 0.0 for item in valid)


def test_early_base_laser_from_unopened_side_cannot_score():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=26)
    base = next(target for target in env.targets if target.kind == "base_blue")
    for target in env.targets:
        if target is not base:
            target.knocked = True
    env.armor["blue"] = 3
    env.elapsed = LASER_DWELL_REQUIRED_S + 0.20

    south_unopened_pose = np.array([-1.25, 0.55], dtype=np.float32)
    yaw = math.atan2(base.xy[1] - float(south_unopened_pose[1]), base.xy[0] - float(south_unopened_pose[0]))
    env.poses["yellow"] = np.array([south_unopened_pose[0], south_unopened_pose[1], yaw], dtype=np.float32)
    env.laser_locks["yellow"] = {"target": base.name, "start": 0.0}

    assert env._detect_laser_hit("yellow") is None
    assert env.last_shot_attempt["yellow"]["reason"] == "no_geometry"
    assert base.knocked is False


def test_base_hit_probability_remains_capped_after_domain_randomization():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=27)
    base = next(target for target in env.targets if target.kind == "base_blue")
    for target in env.targets:
        if target is not base:
            target.knocked = True
    env.armor["blue"] = 3
    env.domain_params.shot_accuracy_scale = 1.50
    env._fire_pose_cache.clear()
    solution = env._best_fire_pose("yellow", base, risk=1.0)
    assert solution is not None
    fire_xy = solution[0]
    yaw = math.atan2(base.xy[1] - float(fire_xy[1]), base.xy[0] - float(fire_xy[0]))
    env.poses["yellow"] = np.array([fire_xy[0], fire_xy[1], yaw], dtype=np.float32)
    env.elapsed = LASER_DWELL_REQUIRED_S + 0.30
    env.laser_locks["yellow"] = {"target": base.name, "start": 0.0}

    env._detect_laser_hit("yellow")

    assert float(env.last_shot_attempt["yellow"]["accuracy"]) <= BASE_HIT_SUCCESS_BY_NORMAL_HITS[1]


def test_targets_are_angled_about_45_degrees_from_walls():
    env = RoboCupVisionRLGymEnv()
    targets = {target.name: target for target in env._make_targets()}
    assert len(targets) == 10
    assert sum(target.kind == "normal" for target in targets.values()) == 8
    assert sum(target.kind.startswith("base_") for target in targets.values()) == 2
    expected_yaws = {
        "T01_NorthMiddle": -math.pi / 4.0,
        "T02_NorthEast": -3.0 * math.pi / 4.0,
        "T03_WestAboveGate": math.pi / 4.0,
        "T04_WestBelowGate": -math.pi / 4.0,
        "T05_EastAboveGate": 3.0 * math.pi / 4.0,
        "T06_EastBelowGate": -3.0 * math.pi / 4.0,
        "T07_SouthWest": math.pi / 4.0,
        "T08_SouthMiddle": 3.0 * math.pi / 4.0,
        "BlueBaseTarget": BLUE_BASE_TARGET_YAW,
        "YellowBaseTarget": YELLOW_BASE_TARGET_YAW,
    }

    for name, expected in expected_yaws.items():
        error = (targets[name].yaw - expected + math.pi) % (2.0 * math.pi) - math.pi
        assert abs(error) <= math.radians(2.0)
        plane_yaw = (targets[name].yaw + math.pi / 2.0) % math.pi
        angle_to_x_wall = min(plane_yaw, math.pi - plane_yaw)
        angle_to_y_wall = abs(math.pi / 2.0 - plane_yaw)
        assert abs(angle_to_x_wall - math.pi / 4.0) <= math.radians(2.0)
        assert abs(angle_to_y_wall - math.pi / 4.0) <= math.radians(2.0)
    assert TARGET_WALL_INSET >= 0.14
    for target in targets.values():
        if target.kind == "normal":
            assert max(abs(target.xy[0]), abs(target.xy[1])) <= 1.36


def test_all_targets_match_rule_layout_and_clear_static_walls():
    env = RoboCupVisionRLGymEnv()
    targets = {target.name: target for target in env._make_targets()}
    n = 1.50 - TARGET_WALL_INSET
    expected = {
        "T01_NorthMiddle": ((NORTH_MIDDLE_TARGET_X, n), "blue", "normal"),
        "T02_NorthEast": ((n, n), "blue", "normal"),
        "T03_WestAboveGate": ((-n, SIDE_GATE_TARGET_Y), "blue", "normal"),
        "T04_WestBelowGate": ((-n, -SIDE_GATE_TARGET_Y), "yellow", "normal"),
        "T05_EastAboveGate": ((n, SIDE_GATE_TARGET_Y), "blue", "normal"),
        "T06_EastBelowGate": ((n, -SIDE_GATE_TARGET_Y), "yellow", "normal"),
        "T07_SouthWest": ((-n, -n), "yellow", "normal"),
        "T08_SouthMiddle": ((SOUTH_MIDDLE_TARGET_X, -n), "yellow", "normal"),
        "BlueBaseTarget": (tuple(float(v) for v in BLUE_BASE_TARGET_XY), "blue", "base_blue"),
        "YellowBaseTarget": (tuple(float(v) for v in YELLOW_BASE_TARGET_XY), "yellow", "base_yellow"),
    }

    assert set(targets) == set(expected)
    blockers = env._make_blockers(inflated=False)
    armor_blockers = active_base_armor_blockers({"blue": 4, "yellow": 4}, inflated=False)
    for name, (xy, owner, kind) in expected.items():
        target = targets[name]
        np.testing.assert_allclose(target.xy, xy, atol=1e-5)
        assert target.owner == owner
        assert target.kind == kind
        for center, half_size in blockers:
            assert not (
                abs(target.xy[0] - center[0]) <= half_size[0] + 0.035
                and abs(target.xy[1] - center[1]) <= half_size[1] + 0.035
            ), f"{name} overlaps static wall/blocker at {center}"
        for center, half_size in armor_blockers:
            assert not (
                abs(target.xy[0] - center[0]) <= half_size[0] + 0.035
                and abs(target.xy[1] - center[1]) <= half_size[1] + 0.035
            ), f"{name} overlaps grounded base armor at {center}"
        for start_name, start_pose in {"yellow_start": YELLOW_START, "blue_start": BLUE_START}.items():
            start_xy = np.asarray(start_pose[:2], dtype=np.float32)
            clearance = float(np.linalg.norm(np.asarray(target.xy, dtype=np.float32) - start_xy))
            assert clearance > ROBOT_RADIUS + 0.055, f"{name} overlaps {start_name}"


def test_base_armor_matches_rule_open_edges():
    assert BASE_ARMOR_SIZE["length"] == 0.250

    expected_blue = [
        ((-1.025, 1.375), (0.050, 0.250)),
        ((-1.375, 1.025), (0.250, 0.050)),
        ((-1.025, 1.125), (0.050, 0.250)),
        ((-1.125, 1.025), (0.250, 0.050)),
    ]
    expected_yellow = [((-cx, -cy), size) for (cx, cy), size in expected_blue]

    assert BASE_ARMOR_SPECS["blue"] == expected_blue
    assert BASE_ARMOR_SPECS["yellow"] == expected_yellow

    for center, size in BASE_ARMOR_SPECS["blue"]:
        if size[0] < size[1]:
            assert math.isclose(center[0] + size[0] * 0.5, -1.0, abs_tol=1e-6)
        else:
            assert math.isclose(center[1] - size[1] * 0.5, 1.0, abs_tol=1e-6)
    for center, size in BASE_ARMOR_SPECS["yellow"]:
        if size[0] < size[1]:
            assert math.isclose(center[0] - size[0] * 0.5, 1.0, abs_tol=1e-6)
        else:
            assert math.isclose(center[1] + size[1] * 0.5, -1.0, abs_tol=1e-6)


def test_pushable_obstacle_defaults_follow_rule_diagram_reference():
    np.testing.assert_allclose(PUSHABLE_OBSTACLE_STARTS["box_ne"], np.array([0.80, 0.80], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(PUSHABLE_OBSTACLE_STARTS["box_sw"], np.array([-0.80, -0.80], dtype=np.float32), atol=1e-6)
    assert math.isclose(PUSHABLE_OBSTACLE_HALF * 2.0, 0.30, abs_tol=1e-6)

    env = RoboCupVisionRLSelfPlayEnv(domain_randomization=True)
    for seed in range(10, 18):
        env.reset(seed=seed)
        for name, default_xy in PUSHABLE_OBSTACLE_STARTS.items():
            offset = env.pushable_obstacles[name] - default_xy
            assert np.all(np.abs(offset) <= PUSHABLE_OBSTACLE_RANDOM_JITTER + 1e-6)
            assert np.all(np.abs(env.pushable_obstacles[name]) >= 0.58 - 1e-6)
            assert np.all(np.abs(env.pushable_obstacles[name]) <= 0.96 + 1e-6)


def test_target_visual_footprints_do_not_clip_internal_walls():
    env = RoboCupVisionRLGymEnv()
    blockers = env._make_blockers(inflated=False)

    for target in env._make_targets():
        if target.kind.startswith("base_"):
            board_span = (0.012, 0.095)
            support_span = (0.018, 0.018)
            foot_span = (0.075, 0.115)
            support_offset = 0.034
            foot_offset = 0.045
        else:
            board_span = (0.012, 0.180)
            support_span = (0.018, 0.018)
            foot_span = (0.110, 0.205)
            support_offset = -0.034
            foot_offset = -0.045
        front = (math.cos(target.yaw), math.sin(target.yaw))
        footprint_parts = [
            (target.xy[0], target.xy[1], board_span),
            (
                target.xy[0] + support_offset * front[0],
                target.xy[1] + support_offset * front[1],
                support_span,
            ),
            (
                target.xy[0] + foot_offset * front[0],
                target.xy[1] + foot_offset * front[1],
                foot_span,
            ),
        ]

        for cx, cy, (span_x, span_y) in footprint_parts:
            extent_x = abs(math.cos(target.yaw)) * span_x * 0.5 + abs(math.sin(target.yaw)) * span_y * 0.5
            extent_y = abs(math.sin(target.yaw)) * span_x * 0.5 + abs(math.cos(target.yaw)) * span_y * 0.5
            for center, half_size in blockers:
                overlap_x = half_size[0] + extent_x - abs(cx - center[0])
                overlap_y = half_size[1] + extent_y - abs(cy - center[1])
                assert overlap_x < 0.0 or overlap_y < 0.0, (
                    f"{target.name} visual footprint clips blocker at {center}; "
                    f"overlap=({overlap_x:.3f}, {overlap_y:.3f})"
                )


def test_each_target_has_clear_front_face_and_not_wall_slot():
    env = RoboCupVisionRLGymEnv()
    blockers = env._make_blockers(inflated=False)

    for target in env._make_targets():
        front = np.array([math.cos(target.yaw), math.sin(target.yaw)], dtype=np.float32)
        target_xy = np.asarray(target.xy, dtype=np.float32)
        probe_distance = 0.44 if target.kind.startswith("base_") else 0.30
        probe = target_xy + front * probe_distance

        assert -1.42 <= float(probe[0]) <= 1.42, f"{target.name} front probe outside arena"
        assert -1.42 <= float(probe[1]) <= 1.42, f"{target.name} front probe outside arena"
        for center, half_size in blockers:
            assert not (
                abs(float(probe[0]) - center[0]) <= half_size[0] + ROBOT_RADIUS * 0.15
                and abs(float(probe[1]) - center[1]) <= half_size[1] + ROBOT_RADIUS * 0.15
            ), f"{target.name} front face points into wall/blocker at {center}"
            assert not segment_intersects_aabb(
                (float(probe[0]), float(probe[1])),
                (float(target_xy[0]), float(target_xy[1])),
                center,
                half_size,
            ), f"{target.name} front line of sight is blocked by {center}"


def test_selfplay_laser_requires_dwell_before_knockdown():
    env = RoboCupVisionRLSelfPlayEnv(dt=0.10)
    env.reset(seed=23)
    target = Target("UnitTarget", (0.42, 0.0), 0.0, "normal", "blue")
    env.targets = [target]
    env.poses["yellow"] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    env.shot_rng["yellow"] = np.random.default_rng(1)

    assert env._detect_laser_hit("yellow") is None
    assert env.last_shot_attempt["yellow"]["reason"] == "dwell"
    assert target.knocked is False

    env.elapsed = LASER_DWELL_REQUIRED_S - 0.10
    assert env._detect_laser_hit("yellow") is None
    assert env.last_shot_attempt["yellow"]["reason"] == "dwell"

    env.elapsed = LASER_DWELL_REQUIRED_S + 0.02
    assert env._detect_laser_hit("yellow") is target


def test_selfplay_own_target_safety_gate_blocks_laser():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=31)
    target = Target("OwnTarget", (0.42, 0.0), 0.0, "normal", "yellow")
    env.targets = [target]
    env.poses["yellow"] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    env.elapsed = LASER_DWELL_REQUIRED_S + 0.20

    assert env._detect_laser_hit("yellow") is None
    assert env.last_shot_attempt["yellow"]["reason"] == "own_target_safety_gate"
    assert target.knocked is False


def test_selfplay_distinguishes_pushable_obstacles_from_static_barriers():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=17)

    env.pushable_obstacles["box_ne"] = np.array([0.68, 0.42], dtype=np.float32)
    env.poses["yellow"] = np.array([0.29, 0.42, 0.0], dtype=np.float32)
    before = env.pushable_obstacles["box_ne"].copy()
    blocked = env._integrate_command("yellow", 0.35, 0.0, allow_push=True)

    assert blocked is False
    assert env.last_push_event["yellow"] == "box_ne"
    assert env.pushable_obstacles["box_ne"][0] > before[0]
    assert env._pushable_collision_name(env.poses["yellow"]) is None

    wall_pose = np.array([1.48, 0.0, 0.0], dtype=np.float32)
    assert env._static_pose_blocked(wall_pose) is True


def test_selfplay_pushable_collision_uses_visual_hull_clearance():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=1711)

    box_xy = np.array([0.68, 0.42], dtype=np.float32)
    env.pushable_obstacles["box_ne"] = box_xy.copy()
    center_x = float(box_xy[0]) - PUSHABLE_OBSTACLE_HALF - ROBOT_PUSHABLE_CLEARANCE_RADIUS + 0.006
    env.poses["yellow"] = np.array([center_x, float(box_xy[1]), 0.0], dtype=np.float32)

    assert env._pushable_collision_name(env.poses["yellow"]) == "box_ne"


def test_oriented_visual_hull_catches_corner_box_penetration():
    collided, normal, penetration = oriented_rect_aabb_collision(
        (0.42, 0.42),
        math.radians(45.0),
        ROBOT_PUSHABLE_VISUAL_HALF_EXTENTS,
        (0.68, 0.42),
        (PUSHABLE_OBSTACLE_HALF, PUSHABLE_OBSTACLE_HALF),
    )

    assert collided is True
    assert penetration > 0.02
    assert normal[0] < 0.0


def test_selfplay_push_keeps_robot_outside_visual_box_hull():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=1712)

    env.pushable_obstacles["box_ne"] = np.array([0.68, 0.42], dtype=np.float32)
    env.poses["yellow"] = np.array([0.27, 0.42, 0.0], dtype=np.float32)

    blocked = env._integrate_command("yellow", 0.35, 0.0, allow_push=True)

    assert blocked is False
    assert env.last_push_event["yellow"] == "box_ne"
    assert env._pushable_collision_name(env.poses["yellow"]) is None


def test_blue_south_middle_fire_pose_does_not_deadlock_on_southwest_box():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=10700)
    env.poses["blue"] = np.array([-0.37902, -0.92001, -1.121], dtype=np.float32)

    blocked, info = env._apply_action("blue", blue_expert_action(env))

    assert info["selected_target"] == "T08_SouthMiddle"
    assert blocked is False
    assert env._pushable_collision_name(env.poses["blue"]) is None
    assert float(info["shot_yaw_error_rad"]) < 0.12


def test_drive_to_goal_downshifts_near_pushable_instead_of_freezing():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=10702)
    env.pushable_obstacles["box_ne"] = np.array([1.121077, 0.66485965], dtype=np.float32)
    env.poses["yellow"] = np.array([0.642041, 0.5758572, -0.7383547], dtype=np.float32)
    goal = np.array([0.9690008, 0.2782986], dtype=np.float32)
    before = env.poses["yellow"].copy()

    blocked = env._drive_to_goal("yellow", goal, risk=0.905)

    assert blocked is False
    assert float(np.linalg.norm(env.poses["yellow"][:2] - before[:2])) > 0.01
    assert env._pushable_collision_name(env.poses["yellow"]) is None


def test_normal_target_stale_attack_replans_instead_of_deadlock():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=10702)
    target = next(item for item in env.targets if item.name == "T05_EastAboveGate")
    env.pushable_obstacles["box_ne"] = np.array([1.121077, 0.66485965], dtype=np.float32)
    env.poses["yellow"] = np.array([0.6558842, 0.5638315, -0.73935413], dtype=np.float32)
    env.armor["blue"] = 2
    env.base_retry_min_normal_hits["yellow"] = 3
    env.normal_attack_stale_steps["yellow"][target.name] = 109

    blocked, info = env._apply_action(
        "yellow",
        np.array([-1.0, 0.68, -0.28, -0.64, -0.72, 0.81], dtype=np.float32),
    )

    assert blocked is False
    assert info["selected_target"] == target.name
    assert info["normal_attack_replanned"] is True
    assert env._target_on_cooldown("yellow", target.name) is True


def test_selfplay_pushable_box_displacement_persists_after_contact():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=171)

    env.pushable_obstacles["box_ne"] = np.array([0.68, 0.42], dtype=np.float32)
    start = env.pushable_obstacles["box_ne"].copy()
    env.poses["yellow"] = np.array([0.29, 0.42, 0.0], dtype=np.float32)

    assert env._integrate_command("yellow", 0.35, 0.0, allow_push=True) is False
    pushed = env.pushable_obstacles["box_ne"].copy()
    assert float(np.linalg.norm(pushed - start)) > 0.035

    env.poses["yellow"] = np.array([0.05, 0.05, 0.0], dtype=np.float32)
    assert env._integrate_command("yellow", 0.0, 0.0, allow_push=False) is False
    np.testing.assert_allclose(env.pushable_obstacles["box_ne"], pushed, atol=1e-6)


def test_single_agent_pushable_box_displacement_persists_after_contact():
    env = RoboCupVisionRLGymEnv()
    env.reset(seed=172)

    env.pushable_obstacles["box_ne"] = np.array([0.68, 0.42], dtype=np.float32)
    env.yellow = np.array([0.29, 0.42, 0.0], dtype=np.float32)
    start = env.pushable_obstacles["box_ne"].copy()

    assert env._apply_yellow_action(np.array([1.0, 0.0, 0.0], dtype=np.float32)) is False
    pushed = env.pushable_obstacles["box_ne"].copy()
    assert float(np.linalg.norm(pushed - start)) > 0.035

    env.yellow = np.array([0.05, 0.05, 0.0], dtype=np.float32)
    assert env._apply_yellow_action(np.array([0.0, 0.0, 0.0], dtype=np.float32)) is False
    np.testing.assert_allclose(env.pushable_obstacles["box_ne"], pushed, atol=1e-6)


def test_single_agent_robot_and_box_cannot_pass_through_targets():
    env = RoboCupVisionRLGymEnv()
    env.reset(seed=173)

    target = Target("UnitTarget", (0.46, 0.0), 0.0, "normal", "blue")
    env.targets = [target]
    env.pushable_obstacles = {}
    env.yellow = np.array([0.20, 0.0, 0.0], dtype=np.float32)

    assert env._target_collision_name(np.array([0.25, 0.0, 0.0], dtype=np.float32)) == "UnitTarget"
    assert env._apply_yellow_action(np.array([1.0, 0.0, 0.0], dtype=np.float32)) is True
    np.testing.assert_allclose(env.yellow, np.array([0.20, 0.0, 0.0], dtype=np.float32))

    env.pushable_obstacles = {"box_ne": np.array([0.23, 0.0], dtype=np.float32)}
    blocked_xy = np.array([0.46 - NORMAL_TARGET_CONTACT_RADIUS - 0.15, 0.0], dtype=np.float32)
    assert env._pushable_position_valid("box_ne", blocked_xy, np.array([-1.0, 0.0], dtype=np.float32)) is False


def test_base_armor_blocks_robot_footprint_not_only_center():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=18)
    center, half_size = active_base_armor_blockers(env.armor, inflated=False)[0]
    pose = np.array(
        [
            center[0] + half_size[0] + 0.16,
            center[1],
            0.0,
        ],
        dtype=np.float32,
    )

    assert abs(float(pose[0]) - center[0]) > half_size[0]
    assert env._static_pose_blocked(pose) is True


def test_selfplay_pushable_box_blocks_robot_when_jammed():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=18)
    env.pushable_obstacles["box_ne"] = np.array([1.30, 0.68], dtype=np.float32)
    env.poses["yellow"] = np.array([1.02, 0.68, 0.0], dtype=np.float32)
    before_pose = env.poses["yellow"].copy()

    blocked = env._integrate_command("yellow", 0.35, 0.0, allow_push=True)

    assert blocked is True
    assert env.poses["yellow"][0] < before_pose[0]
    assert env._pushable_collision_name(env.poses["yellow"]) is None
    assert env.pushable_obstacles["box_ne"][0] <= 1.325


def test_selfplay_shooter_range_uses_laser_outlet_distance():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=23)
    target = Target("UnitTarget", (0.42, 0.0), 0.0, "normal", "blue")
    env.poses["yellow"] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    info = env._update_fire_gate("yellow", target, np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32), 0.0)

    assert env.pending_fire["yellow"] is True
    np.testing.assert_allclose(info["shot_distance_m"], 0.22, atol=1e-3)

    far_target = Target("FarTarget", (0.74, 0.0), 0.0, "normal", "blue")
    info = env._update_fire_gate("yellow", far_target, np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32), 0.0)

    assert env.pending_fire["yellow"] is False
    assert info["shot_distance_m"] > 0.50


def test_base_shooter_uses_20_to_80_cm_outlet_range():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=24)
    base = next(target for target in env.targets if target.kind == "base_blue")
    env.armor["blue"] = 0
    env._fire_pose_cache.clear()
    direction = np.array([math.cos(base.yaw + 0.55), math.sin(base.yaw + 0.55)], dtype=np.float32)
    fire_xy = np.asarray(base.xy, dtype=np.float32) + direction * (SHOOTER_FORWARD_OFFSET + 0.76)
    yaw = math.atan2(base.xy[1] - float(fire_xy[1]), base.xy[0] - float(fire_xy[0]))
    env.poses["yellow"] = np.array([fire_xy[0], fire_xy[1], yaw], dtype=np.float32)

    info = env._update_fire_gate("yellow", base, np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32), 1.0)

    assert info["shot_distance_m"] > 0.70
    assert env.pending_fire["yellow"] is True

    fire_xy = np.asarray(base.xy, dtype=np.float32) + direction * (SHOOTER_FORWARD_OFFSET + 0.15)
    yaw = math.atan2(base.xy[1] - float(fire_xy[1]), base.xy[0] - float(fire_xy[0]))
    env.poses["yellow"] = np.array([fire_xy[0], fire_xy[1], yaw], dtype=np.float32)
    info = env._update_fire_gate("yellow", base, np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32), 1.0)

    assert info["shot_distance_m"] < BASE_SHOOT_MIN_RANGE
    assert env.pending_fire["yellow"] is False


def test_selfplay_holds_position_inside_legal_fire_window():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=41)
    target = next(target for target in env.targets if target.name == "T03_WestAboveGate")
    for other in env.targets:
        if other.name != target.name:
            other.knocked = True
    front = np.array([math.cos(target.yaw), math.sin(target.yaw)], dtype=np.float32)
    fire_xy = np.asarray(target.xy, dtype=np.float32) + front * (SHOOTER_FORWARD_OFFSET + SHOOT_IDEAL_DISTANCE)
    yaw = math.atan2(target.xy[1] - float(fire_xy[1]), target.xy[0] - float(fire_xy[0]))
    env.poses["yellow"] = np.array([fire_xy[0], fire_xy[1], yaw], dtype=np.float32)
    before_xy = env.poses["yellow"][:2].copy()

    blocked, info = env._apply_action(
        "yellow",
        np.array([-1.0, -0.8, -1.0, -1.0, 1.0, 0.0], dtype=np.float32),
    )

    assert blocked is False
    assert info["holding_fire_pose"] is True
    assert env.pending_fire["yellow"] is True
    np.testing.assert_allclose(env.poses["yellow"][:2], before_xy, atol=1e-5)


def test_selfplay_close_standoff_aims_from_laser_outlet():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=42)
    env.pushable_obstacles = {}
    env._fire_pose_cache.clear()
    env._path_cache.clear()
    env._route_distance_cache.clear()
    target = next(target for target in env.targets if target.name == "T05_EastAboveGate")
    for other in env.targets:
        if other.name != target.name:
            other.knocked = True
    start_xy = np.array([1.00, 0.42], dtype=np.float32)
    start_yaw = math.atan2(target.xy[1] - float(start_xy[1]), target.xy[0] - float(start_xy[0])) + 0.35
    env.poses["yellow"] = np.array([start_xy[0], start_xy[1], start_yaw], dtype=np.float32)
    action = np.array([-1.0, -0.8, -1.0, -1.0, 1.0, 0.0], dtype=np.float32)

    initial_yaw_error = float(env._fire_geometry_snapshot("yellow", target, risk=0.5)["yaw_error"])
    for _ in range(45):
        blocked, info = env._apply_action("yellow", action)
        if env.pending_fire["yellow"]:
            break

    final_yaw_error = float(env._fire_geometry_snapshot("yellow", target, risk=0.5)["yaw_error"])
    assert blocked is False
    assert info["holding_fire_pose"] is True
    assert final_yaw_error < initial_yaw_error * 0.35
    assert env.pending_fire["yellow"] is True


def test_selfplay_recessed_base_window_requires_precise_final_alignment():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=43)
    base = next(target for target in env.targets if target.kind == "base_blue")
    for target in env.targets:
        if target is not base:
            target.knocked = True
    env.armor["blue"] = 3
    env.armor["yellow"] = 3
    env._fire_pose_cache.clear()
    early_solution = env._best_fire_pose("yellow", base, risk=0.70)
    assert early_solution is not None
    assert early_solution[3] <= BASE_HIT_SUCCESS_BY_NORMAL_HITS[1]
    assert base_removed_side_lane_quality(1, BLUE_BASE_XY, early_solution[0]) > 0.0

    env.armor["blue"] = 2
    env._fire_pose_cache.clear()
    solution = env._best_fire_pose("yellow", base, risk=0.70)
    assert solution is not None
    aim_yaw = math.atan2(base.xy[1] - float(solution[0][1]), base.xy[0] - float(solution[0][0]))
    env.poses["yellow"] = np.array([solution[0][0], solution[0][1], aim_yaw], dtype=np.float32)
    action = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0], dtype=np.float32)

    for _ in range(55):
        blocked, info = env._apply_action("yellow", action)
        if env.pending_fire["yellow"]:
            break

    assert blocked is False
    assert info["selected_target"] == "BlueBaseTarget"
    assert info["base_rush"] is True
    assert env.pending_fire["yellow"] is True
    assert info["line_clear"] is True
    assert info["goal_distance_m"] <= 0.020


def test_failed_base_cap_forces_more_normal_hits_before_retry():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=44)
    base = next(target for target in env.targets if target.name == "BlueBaseTarget")
    env.armor["blue"] = 2
    env.base_retry_min_normal_hits["yellow"] = 3
    env._fire_pose_cache.clear()

    assert env._target_on_cooldown("yellow", base.name) is True
    assert env._best_fire_pose("yellow", base, risk=0.80) is None

    env.armor["blue"] = 1
    env._fire_pose_cache.clear()

    assert env._target_on_cooldown("yellow", base.name) is False
    assert env._best_fire_pose("yellow", base, risk=0.80) is not None


def test_base_stale_replan_does_not_permanently_lose_base_target():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=10702)
    base = next(target for target in env.targets if target.name == "YellowBaseTarget")
    env.armor["yellow"] = 2
    env.base_retry_min_normal_hits["blue"] = 3
    env.elapsed = 22.2

    env._mark_target_failed("blue", base.name)

    assert base.name not in env.lost_targets["blue"]
    assert env._target_on_cooldown("blue", base.name) is True

    env.armor["yellow"] = 1
    env.target_cooldowns["blue"][base.name] = env.max_time_s + 99.0

    assert env._target_on_cooldown("blue", base.name) is False
    assert base.name not in env.lost_targets["blue"]


def test_three_hit_base_stale_replan_can_retry_base_after_cooldown():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=10702)
    base = next(target for target in env.targets if target.name == "YellowBaseTarget")
    env.armor["yellow"] = 1
    env.elapsed = 34.0

    env.base_attack_stale_steps["blue"][base.name] = 79
    env.selected_target_name["blue"] = base.name
    blocked, info = env._apply_action("blue", np.array([1.0, 1.0, -1.0, -1.0, -1.0, 1.0], dtype=np.float32))

    assert blocked is False
    assert info["base_attack_replanned"] is True
    assert env.base_retry_min_normal_hits["blue"] == 3
    env.elapsed = env.target_cooldowns["blue"][base.name] + 0.1

    assert env._target_on_cooldown("blue", base.name) is False


def test_two_hit_base_retry_allowed_when_no_normal_retry_is_available():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=10702)
    base = next(target for target in env.targets if target.name == "BlueBaseTarget")
    env.armor["blue"] = 2
    env.base_retry_min_normal_hits["yellow"] = 3
    for target in env.targets:
        if target.kind == "normal" and target.owner == "blue" and not target.knocked:
            env.target_cooldowns["yellow"][target.name] = env.elapsed + 20.0

    assert env._normal_hits_against("yellow") == 2
    assert env._has_available_normal_retry_target("yellow") is False
    assert env._target_on_cooldown("yellow", base.name) is False
    assert env._best_fire_pose("yellow", base, risk=0.82) is not None


def test_base_fire_ready_requires_laser_inside_small_hit_radius():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=45)
    base = next(target for target in env.targets if target.kind == "base_blue")
    env.armor["blue"] = 2
    env._fire_pose_cache.clear()
    solution = env._best_fire_pose("yellow", base, risk=0.70)
    assert solution is not None
    fire_xy = solution[0]
    aim_yaw = math.atan2(base.xy[1] - float(fire_xy[1]), base.xy[0] - float(fire_xy[0]))
    env.poses["yellow"] = np.array([fire_xy[0], fire_xy[1], aim_yaw + 0.05], dtype=np.float32)

    off_axis = env._fire_geometry_snapshot("yellow", base, risk=0.70)
    assert off_axis["line_clear"] is True
    assert off_axis["lateral_error"] > off_axis["hit_radius"]
    assert off_axis["geometry_ready"] is False

    env.poses["yellow"][2] = aim_yaw
    centered = env._fire_geometry_snapshot("yellow", base, risk=0.70)
    assert centered["lateral_error"] <= centered["hit_radius"]
    assert centered["geometry_ready"] is True


def test_selfplay_shot_accuracy_increases_when_closer():
    env = RoboCupVisionRLSelfPlayEnv()

    assert env._shot_accuracy_from_geometry(0.04, 0.0, False) == 0.0
    close = env._shot_accuracy_from_geometry(0.08, 0.0, False)
    mid = env._shot_accuracy_from_geometry(0.30, 0.0, False)
    far = env._shot_accuracy_from_geometry(0.50, 0.0, False)

    assert close > mid > far > 0.0


def test_selfplay_target_contact_does_not_knock_down_target():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=29)
    target = next(item for item in env.targets if item.kind == "normal")
    env.poses["yellow"] = np.array([target.xy[0], target.xy[1], 0.0], dtype=np.float32)
    rewards = {team: 0.0 for team in AGENTS}
    infos = {team: {} for team in AGENTS}

    env._resolve_target_contacts("yellow", rewards, infos)

    assert target.knocked is False
    assert infos["yellow"]["target_collision"] == target.name
    assert rewards["yellow"] < 0.0


def test_robot_contact_does_not_trigger_relocalization():
    env = RoboCupVisionRLSelfPlayEnv(dt=0.0)
    env.reset(seed=33)
    env.poses["yellow"] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    env.poses["blue"] = np.array([0.05, 0.0, math.pi], dtype=np.float32)
    before = dict(env.localization_confidence)

    actions = {team: np.zeros(TACTICAL_ACTION_DIM, dtype=np.float32) for team in AGENTS}
    _obs, rewards, _terms, _truncs, infos = env.step(actions)

    assert infos["yellow"]["robot_contact"] is True
    assert infos["blue"]["robot_contact"] is True
    assert env.localization_confidence == before
    assert rewards["yellow"] > -0.25
    assert rewards["blue"] > -0.25


def test_robot_contact_separation_does_not_push_robot_into_static_blocker():
    env = RoboCupVisionRLSelfPlayEnv(dt=0.0)
    env.reset(seed=36)
    env.poses["yellow"] = np.array([1.25, 1.28, 0.0], dtype=np.float32)
    env.poses["blue"] = np.array([1.25, 1.30, math.pi], dtype=np.float32)

    assert env._resolve_contact() is True

    for team in AGENTS:
        assert env._static_pose_blocked(env.poses[team]) is False
        assert env._pushable_collision_name(env.poses[team]) is None


def test_failed_target_gets_cooldown_instead_of_repeat_attack():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=35)
    action = np.array([-1.0, -0.8, -1.0, -1.0, 1.0, 0.0], dtype=np.float32)
    first = env._select_tactical_target("yellow", action)
    assert first is not None

    env._mark_target_failed("yellow", first.name)
    second = env._select_tactical_target("yellow", action)

    assert second is not None
    assert second.name != first.name


def test_unreachable_fire_pose_is_not_ranked_as_nearest_target():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=44)
    env.poses["yellow"] = np.array([0.900, 0.322, 0.160], dtype=np.float32)
    blocked_target = next(target for target in env.targets if target.name == "T05_EastAboveGate")
    reachable_target = next(target for target in env.targets if target.name == "T02_NorthEast")

    best_fire_pose = env._best_fire_pose("yellow", blocked_target, 0.67)
    assert best_fire_pose is not None
    assert env._route_distance_to(env.poses["yellow"][:2], best_fire_pose[0]) > 0.45
    assert env._best_fire_pose("yellow", reachable_target, 0.67) is not None

    action = np.array([-1.0, -0.8, -1.0, -1.0, 1.0, 0.34], dtype=np.float32)
    selected = env._select_tactical_target("yellow", action)

    assert selected is not None
    assert selected.owner == "blue"


def test_single_agent_observation_exposes_blue_bearing():
    env = RoboCupVisionRLGymEnv()
    obs, _ = env.reset(seed=7)

    opponent_track = obs[8:13]
    assert np.isfinite(opponent_track).all()
    assert opponent_track[0] > 0.0
    assert -1.0 <= opponent_track[1] <= 1.0
    assert -1.0 <= opponent_track[2] <= 1.0
    assert opponent_track[3] in (0.0, 1.0)
    assert 0.0 <= opponent_track[4] <= 1.0
