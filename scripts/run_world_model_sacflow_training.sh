#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="isaaclab_sim/output/rl/world_model_sacflow_seed260707_rerun"
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/train.log"

{
  echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') starting world-model SAC Flow training"
  python3 isaaclab_sim/rl/train_world_model_sacflow_selfplay.py \
    --config isaaclab_sim/rl/configs/world_model_flow.yaml \
    --timesteps 200000 \
    --num-envs 32 \
    --batch-size 1024 \
    --learning-starts 4096 \
    --gradient-steps 2 \
    --hidden-dim 256 \
    --device cuda \
    --seed 260707 \
    --output ../output/rl/world_model_sacflow_seed260707_rerun
  echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') training finished"
} 2>&1 | tee -a "${LOG}"
