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
