from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_mappo_policy import json_safe, load_policy, run_episode, summarize


def numeric_summary(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float))})
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        out[key] = {
            "mean": round(float(statistics.fmean(values)), 6),
            "std": round(float(statistics.pstdev(values)), 6) if len(values) > 1 else 0.0,
            "min": round(float(min(values)), 6),
            "max": round(float(max(values)), 6),
        }
    return out


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    episodes: int,
    seed: int,
    max_steps: int,
    device: torch.device,
    stochastic: bool,
    policy_mode_arg: str,
    residual_scale_arg: float | None,
) -> dict[str, object]:
    model, checkpoint_payload = load_policy(checkpoint, device)
    train_config = checkpoint_payload.get("config", {})
    policy_mode = str(train_config.get("policy_mode", "direct")) if policy_mode_arg == "auto" else policy_mode_arg
    residual_scale = (
        float(train_config.get("residual_scale", 0.28))
        if residual_scale_arg is None
        else float(residual_scale_arg)
    )
    started = time.perf_counter()
    episodes_payload = [
        run_episode(
            model,
            seed=seed + index,
            max_steps=max_steps,
            device=device,
            deterministic=not stochastic,
            capture_trace=False,
            policy_mode=policy_mode,
            residual_scale=residual_scale,
        )
        for index in range(episodes)
    ]
    summary = summarize(episodes_payload, time.perf_counter() - started)
    summary["win_rate_gap_abs"] = round(abs(float(summary["yellow_win_rate"]) - float(summary["blue_win_rate"])), 4)
    summary["mean_score_gap_abs"] = round(
        abs(float(summary["mean_yellow_score"]) - float(summary["mean_blue_score"])),
        4,
    )
    return {
        "checkpoint": str(checkpoint),
        "training_seed": int(train_config.get("seed", -1)),
        "policy_mode": policy_mode,
        "residual_scale": residual_scale,
        "deterministic": not stochastic,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate and aggregate multiple fair MAPPO seeds.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument("--checkpoint-template", type=str, default="isaaclab_sim/output/rl/mappo_fair_seed_{seed}_005_050_gpu/policy.pt")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=1200)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--policy-mode", choices=("auto", "direct", "expert", "residual_expert"), default="auto")
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("isaaclab_sim/output/eval/mappo_fair_multiseed_eval.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false.")

    per_seed = []
    for index, seed in enumerate(args.seeds):
        checkpoint = Path(args.checkpoint_template.format(seed=seed))
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        per_seed.append(
            evaluate_checkpoint(
                checkpoint,
                episodes=args.episodes,
                seed=args.eval_seed + index * 10_000,
                max_steps=args.max_steps,
                device=device,
                stochastic=args.stochastic,
                policy_mode_arg=args.policy_mode,
                residual_scale_arg=args.residual_scale,
            )
        )

    summaries = [item["summary"] for item in per_seed]
    aggregate = numeric_summary(summaries)
    payload = {
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes,
        "total_episodes": args.episodes * len(args.seeds),
        "device": str(device),
        "stochastic": args.stochastic,
        "policy_mode": args.policy_mode,
        "residual_scale": args.residual_scale,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json_safe(payload)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"[INFO] wrote {args.output}")


if __name__ == "__main__":
    main()
