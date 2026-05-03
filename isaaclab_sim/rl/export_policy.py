from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Record a deployable-policy export manifest.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("../output/policy_export/manifest.json"))
    parser.add_argument("--format", choices=["torchscript", "onnx", "checkpoint"], default="checkpoint")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    manifest = {
        "project": "RoboCup VisionRL",
        "source_checkpoint": str(args.checkpoint),
        "export_format": args.format,
        "deployment_contract": {
            "inputs": [
                "local_robot_observation",
                "target_status",
                "armor_status",
                "time_remaining",
                "localization_confidence",
            ],
            "outputs": ["target_selection", "route_mode", "recover_gate", "fire_gate"],
            "ros2_runtime": "rcvrl_behavior + Nav2 + rcvrl_shooter",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[INFO] wrote {args.output}")


if __name__ == "__main__":
    main()
