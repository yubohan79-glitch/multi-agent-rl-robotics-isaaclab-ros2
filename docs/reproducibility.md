# Reproducibility

This page lists the minimal commands and artifact locations needed to reproduce or audit the public repository evidence.

Generated outputs should stay under `isaaclab_sim/output/`. Only selected final evidence is committed under `docs/`.

## 1. Python Tests

```bash
python -m pip install -r isaaclab_sim/rl/requirements.txt
python -m pytest tests -q
```

Expected result:

- rule contracts pass;
- target layout and scoring checks pass;
- strategy and Sim2Real configuration checks pass.

## 2. World-Model SAC Flow Training

Recommended environment:

- Linux or WSL with CUDA PyTorch
- Python 3.10+
- dependencies from `isaaclab_sim/rl/requirements.txt`

Reference command:

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
  --output isaaclab_sim/output/rl/world_model_sacflow_seed260707
```

Expected checkpoint:

```text
isaaclab_sim/output/rl/world_model_sacflow_seed260707/policy.pt
```

## 3. Evaluation

Stochastic policy evaluation:

```bash
python3 isaaclab_sim/rl/evaluate_world_model_sacflow_policy.py \
  --checkpoint isaaclab_sim/output/rl/world_model_sacflow_seed260707/policy.pt \
  --episodes 64 \
  --stochastic \
  --output isaaclab_sim/output/eval/world_model_sacflow_eval64.json
```

Rule-contract evaluation:

```bash
python3 isaaclab_sim/rl/evaluate_strategy_contract.py \
  --checkpoint isaaclab_sim/output/rl/world_model_sacflow_seed260707/policy.pt \
  --episodes 64 \
  --stochastic \
  --output-json isaaclab_sim/output/eval/world_model_sacflow_contract_eval64.json \
  --output-csv isaaclab_sim/output/eval/world_model_sacflow_contract_eval64.csv
```

Published reference artifacts:

```text
docs/rl_data/world_model_sacflow_final/training_summary.json
docs/rl_data/world_model_sacflow_final/contract_eval_multiseed.json
docs/rl_data/world_model_sacflow_final/contract_eval_multiseed.csv
docs/rl_data/world_model_sacflow_final/strict_replay_summary.json
docs/rl_data/world_model_sacflow_final/strict_replay_audit.md
```

## 4. Policy Export

```bash
python3 isaaclab_sim/rl/export_world_model_sacflow_policy.py \
  --checkpoint isaaclab_sim/output/rl/world_model_sacflow_seed260707/policy.pt \
  --format torchscript \
  --output-dir isaaclab_sim/output/policy_export/world_model_sacflow_seed260707
```

Record the exported policy path in any report or application material if it is used for deployment.

## 5. IsaacLab Replay

Published final three-view media:

```text
docs/media/最终回放_顶视角.mp4
docs/media/最终回放_黄车第一视角.mp4
docs/media/最终回放_蓝车第一视角.mp4
docs/media/最终回放_顶视角.gif
docs/media/最终回放_黄车第一视角.gif
docs/media/最终回放_蓝车第一视角.gif
docs/media/最终回放_三视角同步拼接版.mp4
```

Windows IsaacLab wrapper:

```powershell
.\scripts\run_isaaclab_project.ps1 -Headless -DemoFlow -Duration 120
```

Process inspection:

```powershell
.\scripts\stop_project_isaaclab.ps1 -WhatIfOnly
```

## 6. 50v50 Simulation-Stage Benchmark

The 50v50 benchmark is a rule-level large-scale extension with IsaacLab tactical replay. It is not claimed as full 100-robot hardware deployment.

Primary artifacts:

```text
docs/large_scale_50v50_plan.md
docs/large_scale_50v50_curriculum_plan.md
docs/large_scale_50v50_report.md
docs/rl_data/large_scale_50v50/
docs/media/large_scale_50v50_isaaclab_replay.mp4
docs/media/large_scale_50v50_isaaclab_replay.gif
docs/figures/large_scale_50v50/
```

## 7. Claim Audit Checklist

Before reporting a result, verify:

- metrics come from JSON/CSV artifacts, not only screenshots;
- replay videos correspond to the evaluated checkpoint or trace;
- collision, penetration, target legality and base-blocker checks are reported;
- 1v1 real-robot evidence is not overstated beyond packaged public logs;
- 50v50 is described as simulation-stage rule-level evidence.
