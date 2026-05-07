# Reproducibility

This page keeps the current, minimal commands for rebuilding the public repository state after the cleanup pass. Runtime outputs are intentionally generated under `isaaclab_sim/output/` and are not committed.

## Python Tests

```bash
python -m pip install -r isaaclab_sim/rl/requirements.txt
python -m pytest tests -q
```

## World-Model SAC Flow Training

Use WSL/Linux Python with PyTorch CUDA:

```bash
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
  --output ../output/rl/world_model_sacflow_seed260707
```

Expected checkpoint:

```text
isaaclab_sim/output/rl/world_model_sacflow_seed260707/policy.pt
```

## Evaluation

```bash
python3 isaaclab_sim/rl/evaluate_world_model_sacflow_policy.py \
  --checkpoint isaaclab_sim/output/rl/world_model_sacflow_seed260707/policy.pt \
  --episodes 64 \
  --stochastic \
  --output isaaclab_sim/output/eval/world_model_sacflow_eval64.json

python3 isaaclab_sim/rl/evaluate_strategy_contract.py \
  --checkpoint isaaclab_sim/output/rl/world_model_sacflow_seed260707/policy.pt \
  --episodes 64 \
  --stochastic \
  --output-json isaaclab_sim/output/eval/world_model_sacflow_contract_eval64.json \
  --output-csv isaaclab_sim/output/eval/world_model_sacflow_contract_eval64.csv
```

## Export

```bash
python3 isaaclab_sim/rl/export_world_model_sacflow_policy.py \
  --checkpoint isaaclab_sim/output/rl/world_model_sacflow_seed260707/policy.pt \
  --format torchscript \
  --output-dir isaaclab_sim/output/policy_export/world_model_sacflow_seed260707
```

## IsaacLab Replay

The retained public media is limited to:

```text
docs/media/最终回放_顶视角.mp4
docs/media/最终回放_黄车第一视角.mp4
docs/media/最终回放_蓝车第一视角.mp4
```

The strict replay source for these files is `docs/rl_expert_base_cap_rng_physical_boxes_strict8.md`.
