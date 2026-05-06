from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from robocup_visionrl_selfplay_env import AGENTS, RoboCupVisionRLSelfPlayEnv
from robocup_visionrl_selfplay_env import TACTICAL_ACTION_DIM
from expert_policy import scripted_action as expert_scripted_action


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def opponent(team: str) -> str:
    return "blue" if team == "yellow" else "yellow"


def select_target(env: RoboCupVisionRLSelfPlayEnv, team: str):
    other = opponent(team)
    pose = env.poses[team]
    active_normals = [
        target
        for target in env.targets
        if target.kind == "normal" and target.owner == other and not target.knocked
    ]
    base_targets = [
        target
        for target in env.targets
        if target.kind == f"base_{other}" and not target.knocked
    ]
    if env.armor[other] <= 2 and base_targets:
        return base_targets[0]
    candidates = active_normals + base_targets
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda target: float(np.linalg.norm(np.asarray(target.xy, dtype=np.float32) - pose[:2])),
    )


def scripted_action(env: RoboCupVisionRLSelfPlayEnv, team: str) -> np.ndarray:
    pose = env.poses[team]
    if env.localization_confidence[team] < 0.35:
        return np.array([0.0, -0.5, -0.8, 1.0, -1.0, -0.2], dtype=np.float32)

    target = select_target(env, team)
    if target is None:
        return np.zeros(TACTICAL_ACTION_DIM, dtype=np.float32)

    dx = target.xy[0] - float(pose[0])
    dy = target.xy[1] - float(pose[1])
    bearing = math.atan2(dy, dx)
    yaw_error = wrap_angle(bearing - float(pose[2]))
    distance = math.hypot(dx, dy)

    other = opponent(team)
    score_delta = env.scores[team] - env.scores[other]
    own_base = np.array([1.25, -1.25], dtype=np.float32) if team == "yellow" else np.array([-1.25, 1.25], dtype=np.float32)
    opponent_to_own_base = float(np.linalg.norm(env.poses[other][:2] - own_base))
    target_selector = -0.75 if target.kind == "normal" else 0.90
    base_rush_gate = 0.85 if target.kind.startswith("base_") or env.armor[other] <= 2 or score_delta < 0 else -0.35
    block_gate = 0.75 if (score_delta >= 5 and env.elapsed > env.max_time_s * 0.58) or opponent_to_own_base < 0.85 else -0.65
    recovery_gate = -0.55
    fire_gate = 0.90 if abs(yaw_error) < 0.12 and 0.32 < distance < 0.72 else 0.20
    risk = 0.45 if score_delta < 0 or env.elapsed > env.max_time_s * 0.72 else 0.05
    return np.array([target_selector, base_rush_gate, block_gate, recovery_gate, fire_gate, risk], dtype=np.float32)


def run_episode(seed: int, max_steps: int):
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=seed)
    rewards_total = {team: 0.0 for team in AGENTS}
    event_counts = {
        "normal_hits": 0,
        "base_hit_wins": 0,
        "own_target_penalties": 0,
        "collision_recovery_events": 0,
        "robot_contacts": 0,
        "block_steps": 0,
        "base_rush_steps": 0,
    }

    steps = 0
    for steps in range(1, max_steps + 1):
        actions = {team: expert_scripted_action(env, team) for team in AGENTS}
        _obs, rewards, terminations, truncations, infos = env.step(actions)
        for team in AGENTS:
            rewards_total[team] += float(rewards[team])
            info = infos[team]
            if "hit" in info:
                event_counts["normal_hits"] += 1
            if "winner" in info:
                event_counts["base_hit_wins"] += 1
            if any(
                key in info
                for key in ("own_target_hit", "own_base_hit", "own_target_blocked", "own_base_blocked", "own_base_collision")
            ):
                event_counts["own_target_penalties"] += 1
            if info.get("relocalizing"):
                event_counts["collision_recovery_events"] += 1
            if info.get("robot_contact"):
                event_counts["robot_contacts"] += 1
            if info.get("tactic") == "block":
                event_counts["block_steps"] += 1
            if info.get("base_rush"):
                event_counts["base_rush_steps"] += 1
        if any(terminations.values()) or any(truncations.values()):
            break

    return {
        "winner": env.winner or "timeout",
        "elapsed_s": round(float(env.elapsed), 3),
        "steps": steps,
        "scores": dict(env.scores),
        "armor": dict(env.armor),
        "rewards": {team: round(value, 3) for team, value in rewards_total.items()},
        "events": event_counts,
        "target_order": {team: list(env.target_order[team]) for team in AGENTS},
        "strategy_counts": {team: dict(env.strategy_counts[team]) for team in AGENTS},
    }


def summarize(episodes: list[dict], wall_time_s: float):
    count = len(episodes)
    winners = [episode["winner"] for episode in episodes]
    total_steps = sum(int(episode["steps"]) for episode in episodes)
    totals = {
        key: sum(int(episode["events"][key]) for episode in episodes)
        for key in episodes[0]["events"]
    }
    return {
        "episodes": count,
        "policy": "scripted_line_of_sight",
        "yellow_win_rate": round(winners.count("yellow") / count, 3),
        "blue_win_rate": round(winners.count("blue") / count, 3),
        "draw_or_timeout_rate": round((winners.count("draw") + winners.count("timeout")) / count, 3),
        "mean_episode_time_s": round(sum(float(ep["elapsed_s"]) for ep in episodes) / count, 3),
        "normal_hits_per_episode": round(totals["normal_hits"] / count, 3),
        "base_hit_wins_per_episode": round(totals["base_hit_wins"] / count, 3),
        "own_target_penalties_per_episode": round(totals["own_target_penalties"] / count, 3),
        "collision_recovery_events_per_episode": round(totals["collision_recovery_events"] / count, 3),
        "robot_contacts_per_episode": round(totals["robot_contacts"] / count, 3),
        "block_steps_per_episode": round(totals["block_steps"] / count, 3),
        "base_rush_steps_per_episode": round(totals["base_rush_steps"] / count, 3),
        "simulated_steps_per_second": round(total_steps / max(wall_time_s, 1e-9), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RoboCup VisionRL self-play rule environment.")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--output", type=Path, default=Path("../output/eval/selfplay_summary.json"))
    args = parser.parse_args()

    started = time.perf_counter()
    episodes = [run_episode(args.seed + index, args.max_steps) for index in range(args.episodes)]
    wall_time_s = time.perf_counter() - started
    summary = summarize(episodes, wall_time_s)
    payload = {"summary": summary, "episodes": episodes}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[INFO] wrote {args.output}")


if __name__ == "__main__":
    main()
