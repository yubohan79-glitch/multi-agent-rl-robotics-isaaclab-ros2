from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR_CPP = ROOT / "crc_robocup_vision_ws" / "src" / "rcvrl_behavior" / "src" / "competition_behavior.cpp"


def test_behavior_node_has_opponent_target_gate():
    source = BEHAVIOR_CPP.read_text(encoding="utf-8")

    assert "is_own_detection" in source
    assert "target_owner" in source
    assert "team_color" in source
    assert "shooter/fire" in source


def test_own_detection_is_rejected_before_fire():
    source = BEHAVIOR_CPP.read_text(encoding="utf-8")
    fire_start = source.index("case State::FIRE:\n        stop_robot();")
    fire_end = source.index("case State::NEXT_TARGET", fire_start)
    fire_case = source[fire_start:fire_end]

    assert "fire_is_allowed()" in fire_case
    assert "call_trigger(fire_client_" in fire_case
    assert fire_case.index("fire_is_allowed()") < fire_case.index("call_trigger(fire_client_")
