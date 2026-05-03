from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_public_target_layout_matches_rule_contract():
    layout = yaml.safe_load((ROOT / "config" / "target_layout.yaml").read_text(encoding="utf-8"))
    normals = layout["targets"]["normal"]["positions"]
    base = layout["targets"]["base"]

    assert layout["targets"]["normal"]["tag_id"] == 1
    assert len(normals) == 8
    assert base["yellow"]["tag_id"] == 2
    assert base["blue"]["tag_id"] == 3
    assert {target["owner"] for target in normals} == {"yellow", "blue"}


def test_elimination_target_files_are_team_separated():
    yellow_config = yaml.safe_load((
        ROOT
        / "crc_robocup_vision_ws"
        / "src"
        / "rcvrl_navigation"
        / "config"
        / "targets.elimination.yellow.yaml"
    ).read_text(encoding="utf-8"))
    blue_config = yaml.safe_load((
        ROOT
        / "crc_robocup_vision_ws"
        / "src"
        / "rcvrl_navigation"
        / "config"
        / "targets.elimination.blue.yaml"
    ).read_text(encoding="utf-8"))

    yellow_owners = yellow_config["competition_behavior"]["ros__parameters"]["target_owner"]
    blue_owners = blue_config["competition_behavior"]["ros__parameters"]["target_owner"]

    assert set(yellow_owners) == {"blue"}
    assert set(blue_owners) == {"yellow"}
