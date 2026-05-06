from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def clone_state(state: dict[str, object]) -> dict[str, object]:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in state.items()}


def actor_suffixes(state: dict[str, torch.Tensor]) -> list[str]:
    return sorted(key[len("yellow_actor.") :] for key in state if key.startswith("yellow_actor."))


def average_tensors(values: list[torch.Tensor]) -> torch.Tensor:
    return sum(values) / float(len(values))


def build_balanced_checkpoint(
    checkpoints: list[Path],
    *,
    log_std: float | None,
    zero_team_id: bool,
    blend_alpha: float,
) -> dict[str, object]:
    payloads = [torch.load(path, map_location="cpu") for path in checkpoints]
    base = payloads[0]
    base_state = base["model_state_dict"]
    if str(base.get("actor_mode", base.get("config", {}).get("actor_mode", "shared"))) != "dual":
        raise ValueError("balanced actor export requires dual-actor checkpoints")

    state = clone_state(base_state)
    suffixes = actor_suffixes(base_state)
    for suffix in suffixes:
        values = []
        for payload in payloads:
            model_state = payload["model_state_dict"]
            values.append(model_state[f"yellow_actor.{suffix}"])
            values.append(model_state[f"blue_actor.{suffix}"])
        averaged = average_tensors(values)
        if zero_team_id and suffix == "0.weight":
            averaged = averaged.clone()
            averaged[:, -1] = 0.0
        alpha = float(max(0.0, min(1.0, blend_alpha)))
        state[f"yellow_actor.{suffix}"] = (1.0 - alpha) * base_state[f"yellow_actor.{suffix}"] + alpha * averaged
        state[f"blue_actor.{suffix}"] = (1.0 - alpha) * base_state[f"blue_actor.{suffix}"] + alpha * averaged

    for key in list(state.keys()):
        if key.startswith("critic.") or key == "log_std":
            values = [payload["model_state_dict"][key] for payload in payloads]
            state[key] = average_tensors(values)
    if log_std is not None:
        state["log_std"] = torch.full_like(state["log_std"], float(log_std))

    checkpoint = {key: value for key, value in base.items() if key != "model_state_dict"}
    checkpoint["model_state_dict"] = state
    checkpoint["actor_mode"] = "dual"
    checkpoint["config"] = dict(checkpoint.get("config", {}))
    checkpoint["config"]["actor_mode"] = "dual"
    checkpoint["balanced_policy"] = {
        "type": "symmetrized_canonical_dual_actor",
        "source_checkpoints": [str(path) for path in checkpoints],
        "team_id_first_layer_zeroed": bool(zero_team_id),
        "blend_alpha": float(max(0.0, min(1.0, blend_alpha))),
        "log_std_override": log_std,
        "notes": "Yellow and blue decentralized actors are partially blended toward one canonical-frame residual policy for fair self-play deployment while keeping team expert priors separate.",
    }
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a fair symmetrized dual-actor MAPPO checkpoint.")
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-std", type=float, default=None)
    parser.add_argument("--keep-team-id", action="store_true")
    parser.add_argument("--blend-alpha", type=float, default=1.0)
    args = parser.parse_args()

    checkpoint = build_balanced_checkpoint(
        args.checkpoints,
        log_std=args.log_std,
        zero_team_id=not args.keep_team_id,
        blend_alpha=args.blend_alpha,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    policy_path = args.output / "policy.pt"
    torch.save(checkpoint, policy_path)
    manifest = {
        "policy_path": str(policy_path),
        "balanced_policy": checkpoint["balanced_policy"],
        "obs_dim": checkpoint.get("obs_dim"),
        "central_obs_dim": checkpoint.get("central_obs_dim"),
        "action_dim": checkpoint.get("action_dim"),
        "actor_mode": checkpoint.get("actor_mode"),
    }
    (args.output / "balanced_policy_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
