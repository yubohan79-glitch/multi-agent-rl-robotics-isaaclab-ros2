#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TIMESTEPS="${TIMESTEPS:-1000000}"
NUM_ENVS="${NUM_ENVS:-32}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-256}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-7 17 29}"
RUN_TAG="${RUN_TAG:-basewindow}"
ENT_COEF="${ENT_COEF:-0.001}"
POLICY_MODE="${POLICY_MODE:-residual_expert}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.04}"
RESIDUAL_L2_COEF="${RESIDUAL_L2_COEF:-0.0015}"
ACTOR_MODE="${ACTOR_MODE:-dual}"

echo "[FAIR-MULTISEED] repo=$REPO_ROOT"
echo "[FAIR-MULTISEED] seeds=$SEEDS timesteps=$TIMESTEPS num_envs=$NUM_ENVS device=$DEVICE tag=$RUN_TAG ent_coef=$ENT_COEF policy_mode=$POLICY_MODE actor_mode=$ACTOR_MODE residual_scale=$RESIDUAL_SCALE"

for seed in $SEEDS; do
  output="../output/rl/mappo_fair_${RUN_TAG}_seed_${seed}_005_050_gpu"
  echo "[FAIR-MULTISEED] start seed=$seed output=$output"
  python3 isaaclab_sim/rl/train_mappo_selfplay_parallel_torch.py \
    --config isaaclab_sim/rl/configs/mappo_selfplay.yaml \
    --seed "$seed" \
    --timesteps "$TIMESTEPS" \
    --num-envs "$NUM_ENVS" \
    --rollout-steps "$ROLLOUT_STEPS" \
    --device "$DEVICE" \
    --ent-coef "$ENT_COEF" \
    --policy-mode "$POLICY_MODE" \
    --residual-scale "$RESIDUAL_SCALE" \
    --residual-l2-coef "$RESIDUAL_L2_COEF" \
    --actor-mode "$ACTOR_MODE" \
    --output "$output"
  echo "[FAIR-MULTISEED] done seed=$seed"
done

echo "[FAIR-MULTISEED] all seeds done"
