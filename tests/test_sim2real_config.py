from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_arena_rules_capture_required_sensors_and_match_limits():
    rules = yaml.safe_load((ROOT / "config" / "arena_rules.yaml").read_text(encoding="utf-8"))

    assert rules["arena"]["size_m"] == [3.0, 3.0]
    assert rules["match"]["duration_s"] == 180
    assert rules["match"]["own_target_fire_allowed"] is False
    assert "imu" in rules["sensors"]["required"]
    assert "lidar_2d" in rules["sensors"]["required"]
    assert "rgb_camera" in rules["sensors"]["required"]


def test_sim2real_yaml_uses_robot_envelope_and_sensor_frames():
    sim2real = yaml.safe_load(
        (
            ROOT
            / "crc_robocup_vision_ws"
            / "src"
            / "rcvrl_bringup"
            / "config"
            / "sim2real.yaml"
        ).read_text(encoding="utf-8")
    )

    text = str(sim2real)
    assert "0.34" in text
    assert "0.24" in text
    assert "imu" in text.lower()
    assert "lidar" in text.lower()


def test_ros2_launch_wires_ekf_filtered_odometry_to_behavior():
    bringup = ROOT / "crc_robocup_vision_ws" / "src" / "rcvrl_bringup"
    behavior = ROOT / "crc_robocup_vision_ws" / "src" / "rcvrl_behavior"
    launch_text = (bringup / "launch" / "competition.launch.py").read_text(encoding="utf-8")
    fusion = yaml.safe_load((bringup / "config" / "sensor_fusion.yaml").read_text(encoding="utf-8"))
    behavior_yaml = yaml.safe_load((behavior / "config" / "behavior.yaml").read_text(encoding="utf-8"))

    assert "robot_localization" in launch_text
    assert "ekf_node" in launch_text
    assert 'DeclareLaunchArgument("target_file", default_value="auto")' in launch_text
    assert "OpaqueFunction(function=behavior_node_for_team)" in launch_text
    assert "targets.elimination.blue.yaml" in launch_text
    assert "targets.elimination.yellow.yaml" in launch_text
    params = fusion["ekf_filter_node"]["ros__parameters"]
    assert params["odom0"] == "/wheel/odom"
    assert params["imu0"] == "/imu/data_raw"
    assert params["two_d_mode"] is True
    assert behavior_yaml["competition_behavior"]["ros__parameters"]["filtered_odom_topic"] == "/odometry/filtered"


def test_ros2_contact_impulses_do_not_force_relocalization_when_ekf_is_good():
    source = (
        ROOT
        / "crc_robocup_vision_ws"
        / "src"
        / "rcvrl_behavior"
        / "src"
        / "competition_behavior.cpp"
    ).read_text(encoding="utf-8")

    assert "void handle_contact_impulse" in source
    assert 'handle_contact_impulse("imu collision impulse")' in source
    assert 'handle_contact_impulse("left bumper contact")' in source
    assert 'handle_contact_impulse("right bumper contact")' in source
    assert 'request_localization_recovery("imu collision impulse")' not in source
    assert "filtered_odom_confidence_ < filtered_odom_min_confidence_" in source
    assert "Contact handled without full relocalization" in source
