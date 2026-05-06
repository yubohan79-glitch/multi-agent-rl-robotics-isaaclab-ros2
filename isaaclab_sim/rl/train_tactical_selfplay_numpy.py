from __future__ import annotations

"""Dependency-light tactical self-play training.

This script optimizes the high-level action contract with a small CEM policy
search. It is intentionally kept NumPy-only so the rule-level strategy can be
trained and smoke-tested on machines that do not yet have PyTorch installed.
For long runs, `train_mappo_selfplay_parallel_torch.py` remains the MAPPO path.
"""

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from robocup_visionrl_selfplay_env import AGENTS, RoboCupVisionRLSelfPlayEnv


@dataclass
class TrainConfig:
    iterations: int
    population: int
    elite_fraction: float
    episodes_per_candidate: int
    max_steps: int
    sigma_init: float
    sigma_min: float
    seed: int


def unpack_policy(theta: np.ndarray, obs_dim: int, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    weight_size = obs_dim * action_dim
    weights = theta[:weight_size].reshape(obs_dim, action_dim)
    bias = theta[weight_size: weight_size + action_dim]
    return weights, bias


def policy_action(theta: np.ndarray, obs: np.ndarray, obs_dim: int, action_dim: int) -> np.ndarray:
    weights, bias = unpack_policy(theta, obs_dim, action_dim)
    return np.tanh(np.asarray(obs, dtype=np.float32) @ weights + bias).astype(np.float32)


def evaluate_candidate(
    theta: np.ndarray,
    *,
    obs_dim: int,
    action_dim: int,
    seed: int,
    episodes: int,
    max_steps: int,
) -> dict[str, object]:
    returns = []
    winners = []
    normal_hits = 0
    base_wins = 0
    own_target_penalties = 0
    block_steps = 0
    recovery_steps = 0
    target_orders: list[list[str]] = []

    for episode in range(episodes):
        env = RoboCupVisionRLSelfPlayEnv()
        observations, _infos = env.reset(seed=seed + episode)
        total_reward = {team: 0.0 for team in AGENTS}
        for _step in range(max_steps):
            actions = {
                team: policy_action(theta, observations[team], obs_dim, action_dim)
                for team in AGENTS
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            for team in AGENTS:
                total_reward[team] += float(rewards[team])
                info = infos[team]
                if "hit" in info:
                    normal_hits += 1
                if "winner" in info:
                    base_wins += 1
                if any(
                    key in info
                    for key in ("own_target_hit", "own_base_hit", "own_target_blocked", "own_base_blocked", "own_base_collision")
                ):
                    own_target_penalties += 1
                if info.get("tactic") == "block":
                    block_steps += 1
                if info.get("relocalizing"):
                    recovery_steps += 1
            if any(terminations.values()) or any(truncations.values()):
                break

        returns.append(sum(total_reward.values()) / len(AGENTS))
        winners.append(env.winner or "timeout")
        for team in AGENTS:
            if env.target_order[team]:
                target_orders.append(list(env.target_order[team]))

    return {
        "score": float(np.mean(returns)),
        "mean_return": float(np.mean(returns)),
        "yellow_win_rate": winners.count("yellow") / max(1, len(winners)),
        "blue_win_rate": winners.count("blue") / max(1, len(winners)),
        "timeout_rate": (winners.count("draw") + winners.count("timeout")) / max(1, len(winners)),
        "normal_hits_per_episode": normal_hits / max(1, episodes),
        "base_wins_per_episode": base_wins / max(1, episodes),
        "own_target_penalties_per_episode": own_target_penalties / max(1, episodes),
        "block_steps_per_episode": block_steps / max(1, episodes),
        "recovery_steps_per_episode": recovery_steps / max(1, episodes),
        "sample_target_orders": target_orders[:8],
    }


def main():
    parser = argparse.ArgumentParser(description="Train a tactical self-play policy with NumPy CEM.")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--population", type=int, default=18)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--episodes-per-candidate", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--sigma-init", type=float, default=0.55)
    parser.add_argument("--sigma-min", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("../output/rl/tactical_numpy"))
    args = parser.parse_args()

    cfg = TrainConfig(
        iterations=args.iterations,
        population=args.population,
        elite_fraction=args.elite_fraction,
        episodes_per_candidate=args.episodes_per_candidate,
        max_steps=args.max_steps,
        sigma_init=args.sigma_init,
        sigma_min=args.sigma_min,
        seed=args.seed,
    )
    rng = np.random.default_rng(cfg.seed)
    probe = RoboCupVisionRLSelfPlayEnv()
    obs_dim = probe.observation_spaces["yellow"].shape[0]
    action_dim = probe.action_spaces["yellow"].shape[0]
    param_dim = obs_dim * action_dim + action_dim
    mean = np.zeros(param_dim, dtype=np.float32)
    sigma = np.full(param_dim, cfg.sigma_init, dtype=np.float32)
    elite_count = max(2, int(round(cfg.population * cfg.elite_fraction)))

    output_dir = (Path(__file__).resolve().parent / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output_dir / "training_curve.csv"
    history: list[dict[str, object]] = []
    started = time.perf_counter()

    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iteration",
                "best_score",
                "mean_score",
                "normal_hits_per_episode",
                "base_wins_per_episode",
                "own_target_penalties_per_episode",
                "block_steps_per_episode",
                "sigma_mean",
            ],
        )
        writer.writeheader()
        best_theta = mean.copy()
        best_metrics: dict[str, object] | None = None

        for iteration in range(1, cfg.iterations + 1):
            candidates = mean + rng.normal(size=(cfg.population, param_dim)).astype(np.float32) * sigma
            evaluated = []
            for index, theta in enumerate(candidates):
                metrics = evaluate_candidate(
                    theta,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    seed=cfg.seed + iteration * 1000 + index * 37,
                    episodes=cfg.episodes_per_candidate,
                    max_steps=cfg.max_steps,
                )
                evaluated.append((float(metrics["score"]), theta, metrics))
            evaluated.sort(key=lambda item: item[0], reverse=True)
            elites = np.stack([item[1] for item in evaluated[:elite_count]])
            mean = elites.mean(axis=0).astype(np.float32)
            sigma = np.maximum(elites.std(axis=0).astype(np.float32), cfg.sigma_min)
            if best_metrics is None or evaluated[0][0] > float(best_metrics["score"]):
                best_theta = evaluated[0][1].astype(np.float32)
                best_metrics = evaluated[0][2]

            row = {
                "iteration": iteration,
                "best_score": round(evaluated[0][0], 4),
                "mean_score": round(float(np.mean([item[0] for item in evaluated])), 4),
                "normal_hits_per_episode": round(float(evaluated[0][2]["normal_hits_per_episode"]), 4),
                "base_wins_per_episode": round(float(evaluated[0][2]["base_wins_per_episode"]), 4),
                "own_target_penalties_per_episode": round(float(evaluated[0][2]["own_target_penalties_per_episode"]), 4),
                "block_steps_per_episode": round(float(evaluated[0][2]["block_steps_per_episode"]), 4),
                "sigma_mean": round(float(sigma.mean()), 4),
            }
            writer.writerow(row)
            history.append({**row, "best_metrics": evaluated[0][2]})
            print(
                "[TACTICAL-RL]: "
                f"iter={iteration} best={row['best_score']} mean={row['mean_score']} "
                f"hits/ep={row['normal_hits_per_episode']} base/ep={row['base_wins_per_episode']} "
                f"block/ep={row['block_steps_per_episode']}"
            )

    final_metrics = evaluate_candidate(
        best_theta,
        obs_dim=obs_dim,
        action_dim=action_dim,
        seed=cfg.seed + 90_000,
        episodes=max(8, cfg.episodes_per_candidate * 2),
        max_steps=cfg.max_steps,
    )
    np.savez(output_dir / "linear_tactical_policy.npz", theta=best_theta, obs_dim=obs_dim, action_dim=action_dim)
    summary = {
        "algorithm": "CEM policy search for high-level tactical self-play",
        "config": asdict(cfg),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "action_contract": [
            "target_selector",
            "base_rush_gate",
            "block_interference_gate",
            "recovery_gate",
            "fire_gate",
            "risk_preference",
        ],
        "wall_time_s": round(time.perf_counter() - started, 3),
        "final_metrics": final_metrics,
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["final_metrics"], indent=2))
    print(f"[INFO] wrote {output_dir}")


if __name__ == "__main__":
    main()
