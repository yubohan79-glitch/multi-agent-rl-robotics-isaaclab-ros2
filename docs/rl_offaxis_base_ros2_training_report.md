# Off-Axis Base-Rush ROS2 + IsaacLab RL Report

Date: 2026-05-04

## What Was Fixed

This run addresses the latest rule corrections:

- The base shooting pose is no longer forced to touch the wall.
- If the robot attacks the base after only 1 or 2 normal targets, the shot must come from a narrow off-axis angle, so it is possible but difficult.
- Base attack success caps are kept at `1 -> 60%`, `2 -> 70%`, `3 -> 85%`, `4 -> 95%`.
- Laser target fall still requires `0.80 s` dwell; below `0.80 s` the target cannot fall.
- The grounded base armor remains a true laser blocker.
- Shooting range is still restricted to `0.05 m` to `0.50 m` from the shooter outlet.
- Pushable boxes remain dynamic rigid obstacles in the RL trace and strict replay audit.
- Robot-to-robot contact is allowed as a tactical event, but does not trigger relocalization by itself.
- Own-target and own-base shots remain blocked and audited.

## ROS2 Realtime Data

Realtime ROS2 topics were collected before this RL run and are documented in:

- `docs/ros2_realtime_topic_collection_report.md`
- `docs/rl_data/ros2_motion_drift_live/motion_drift_live_log.csv`
- `docs/rl_data/ros2_motion_drift_live/motion_drift_live_summary.json`

Key measurement:

| Metric | Value |
|---|---:|
| Samples | 228 |
| `/cmd_vel` rate | about 30 Hz |
| `/scan` rate | about 30 Hz |
| Mean odom XY residual | 0.10177 m |
| P95 odom XY residual | 0.19650 m |
| Mean drift risk | 0.19078 |
| High-accel mean drift risk | 0.54432 |
| Low-accel mean drift risk | 0.18286 |

This supports the training model where high acceleration increases localization drift risk.

## Training

Command:

```bash
python3 isaaclab_sim/rl/train_mappo_selfplay_parallel_torch.py \
  --config isaaclab_sim/rl/configs/mappo_selfplay.yaml \
  --timesteps 200000 \
  --num-envs 16 \
  --rollout-steps 128 \
  --update-epochs 3 \
  --minibatch-size 1024 \
  --hidden-dim 192 \
  --device cuda \
  --seed 121 \
  --policy-mode residual_expert \
  --residual-scale 0.36 \
  --residual-l2-coef 0.0015 \
  --actor-mode dual \
  --output ../output/rl/mappo_offaxis_base_ros2_gpu_seed121
```

Output:

- checkpoint: `isaaclab_sim/output/rl/mappo_offaxis_base_ros2_gpu_seed121/policy.pt`
- training curve: `isaaclab_sim/output/rl/mappo_offaxis_base_ros2_gpu_seed121/training_curve.csv`

## Evaluation

Stochastic evaluation, 64 episodes:

| Metric | Value |
|---|---:|
| Yellow win rate | 0.4844 |
| Blue win rate | 0.4375 |
| Draw / timeout | 0.0781 |
| Mean episode time | 47.0187 s |
| Normal hits / episode | 1.5156 |
| Base-hit wins / episode | 0.9219 |
| Own-target penalties / episode | 0.0 |
| Robot contacts / episode | 0.0 |
| Block steps / episode | 3.5781 |
| Base-rush steps / episode | 182.1406 |
| Interference steps / episode | 3.4531 |

Strict replay audit, 16 episodes:

| Metric | Value |
|---|---:|
| Verdict | PASS |
| Yellow win rate | 0.4375 |
| Blue win rate | 0.5000 |
| Draw / timeout | 0.0625 |
| Hard violations | 0 |
| Warnings | 0 |
| Normal hits / episode | 1.6250 |
| Base wins / episode | 0.9375 |
| Own-target penalties / episode | 0.0 |
| Blocked steps / episode | 0.0 |
| Target contact events / episode | 0.0 |
| Robot contacts / episode | 0.0 |

The policy is not mathematically perfect, but it is now in the intended band: both sides remain competitive, base wins are frequent, the average strategy attacks the base after about 1-2 normal targets, and strict safety violations are zero.

## Generated Figures

Architecture:

- `docs/figures/rl/ros2_isaaclab_sensorfusion_architecture.svg`

Experiment figures with real data:

- `docs/figures/rl/ros2_motion_drift_live.svg`
- `docs/figures/rl/rl_offaxis_base_training_curve.svg`
- `docs/figures/rl/rl_offaxis_base_eval_metrics.svg`

Recommended table:

| Strategy bucket | What to report |
|---|---|
| 1 normal target then base | fastest, hardest angle, 60% base cap |
| 2 normal targets then base | preferred real-match balance, 70% base cap |
| 3 normal targets then base | safer base shot, 85% base cap |
| 4 normal targets then base | highest base reliability, slowest, 95% base cap |

Use the evaluation JSON `target_order` fields to compute the final bucket counts for the paper/table.
