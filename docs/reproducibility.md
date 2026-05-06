# Reproducibility

These commands reproduce the public smoke tests and dry-run paths without requiring private competition files.

## Python RL Smoke Tests

```bash
python -m pip install -r isaaclab_sim/rl/requirements.txt
python -m pytest tests -q
```

Run a deterministic rule-environment evaluation:

```bash
cd isaaclab_sim/rl
python evaluate_selfplay.py --episodes 16 --output ../output/eval/selfplay_summary.json
```

Run the vectorized self-play rollout check:

```bash
cd isaaclab_sim/rl
python robocup_visionrl_selfplay_vec.py
```

## Full GPU MAPPO Run

The completed experiment used CUDA PyTorch on an NVIDIA GeForce RTX 4090:

```bash
python3 isaaclab_sim/rl/train_mappo_selfplay_parallel_torch.py --config configs/mappo_selfplay.yaml
```

Equivalent explicit command:

```bash
python3 isaaclab_sim/rl/train_mappo_selfplay_parallel_torch.py \
  --timesteps 500000 \
  --num-envs 32 \
  --rollout-steps 256 \
  --update-epochs 4 \
  --minibatch-size 2048 \
  --hidden-dim 256 \
  --device cuda \
  --output ../output/rl/mappo_selfplay_full_gpu
```

Evaluate the saved policy:

```bash
python3 isaaclab_sim/rl/evaluate_mappo_policy.py \
  --checkpoint isaaclab_sim/output/rl/mappo_selfplay_full_gpu/policy.pt \
  --episodes 64 \
  --device cuda \
  --output isaaclab_sim/output/eval/mappo_full_gpu_eval.json

python3 isaaclab_sim/rl/evaluate_mappo_policy.py \
  --checkpoint isaaclab_sim/output/rl/mappo_selfplay_full_gpu/policy.pt \
  --episodes 64 \
  --device cuda \
  --stochastic \
  --output isaaclab_sim/output/eval/mappo_full_gpu_eval_stochastic.json
```

Regenerate the SVG figures from the archived CSV/JSON data:

```bash
python isaaclab_sim/rl/generate_rl_figures.py
```

Export the decentralized tactical actor for ROS2-side integration:

```bash
python3 isaaclab_sim/rl/export_policy.py \
  --checkpoint isaaclab_sim/output/rl/mappo_selfplay_full_gpu/policy.pt \
  --format torchscript \
  --device cpu \
  --output-dir isaaclab_sim/output/policy_export/mappo_selfplay_full_gpu
```

Run strict post-training replay audit:

```bash
python3 isaaclab_sim/rl/replay_mappo_policy_strict.py \
  --checkpoint isaaclab_sim/output/rl/mappo_selfplay_full_gpu/policy.pt \
  --episodes 32 \
  --seed 901 \
  --device cpu \
  --stochastic \
  --output-dir ../output/replay/mappo_strict_replay_full \
  --report ../../docs/rl_strict_replay_audit.md
```

Render the strict replay as an MP4:

```bash
python3 isaaclab_sim/rl/render_strict_replay_video.py \
  --episode 0 \
  --step-stride 4 \
  --fps 12 \
  --output docs/media/strict_mappo_replay_episode0.mp4
```

The completed run report is `docs/rl_full_strategy_report.md`.

## Current Dual-Expert Contact-Hull Run

This is the current replayed policy. It uses recessed base targets, grounded blue armor as robot/laser blockers, early-base side gating, 20-80 cm base-target outlet range, 0.8 s dwell, probabilistic base fall, pushable rigid boxes, a conservative robot-box contact hull and multi-sensor drift features from the ROS2 contract.

```bash
PYTHONPATH=isaaclab_sim/rl python3 isaaclab_sim/rl/train_mappo_selfplay_parallel_torch.py \
  --config isaaclab_sim/rl/configs/mappo_selfplay.yaml \
  --timesteps 16384 \
  --num-envs 32 \
  --rollout-steps 64 \
  --update-epochs 2 \
  --minibatch-size 2048 \
  --hidden-dim 256 \
  --device cuda \
  --seed 260507 \
  --policy-mode residual_expert \
  --residual-scale 0.04 \
  --residual-l2-coef 0.0018 \
  --actor-mode dual \
  --domain-randomization \
  --resume isaaclab_sim/output/rl/mappo_dual_experts_recovery_cooldown_blend052_seed260505/policy.pt \
  --output isaaclab_sim/output/rl/mappo_dual_experts_contact_hull_seed260507
```

Evaluate the selected checkpoint:

```bash
PYTHONPATH=isaaclab_sim/rl python3 isaaclab_sim/rl/evaluate_mappo_policy.py \
  --checkpoint isaaclab_sim/output/rl/mappo_dual_experts_contact_hull_seed260507/policy.pt \
  --episodes 64 \
  --seed 930000 \
  --max-steps 1800 \
  --device cuda \
  --stochastic \
  --output isaaclab_sim/output/eval/mappo_dual_experts_contact_hull_seed260507_eval64.json
```

Run the strict audit used by the final videos:

```bash
PYTHONPATH=isaaclab_sim/rl python3 isaaclab_sim/rl/replay_mappo_policy_strict.py \
  --checkpoint isaaclab_sim/output/rl/mappo_dual_experts_contact_hull_seed260507/policy.pt \
  --episodes 8 \
  --seed 940000 \
  --max-steps 1800 \
  --device cuda \
  --stochastic \
  --output-dir isaaclab_sim/output/replay/mappo_dual_experts_contact_hull_seed260507_strict8 \
  --report docs/rl_dual_experts_contact_hull_seed260507_strict8.md
```

Record the three IsaacLab MP4 views from contact-hull strict episode 5 on Windows:

```powershell
cmd /c """C:\Users\Administrator\IsaacLab\isaaclab.bat"" -p isaaclab_sim\robocup_visionrl_arena_sim.py --headless --duration 32 --replay_trace isaaclab_sim\output\replay\mappo_dual_experts_contact_hull_seed260507_strict8\strict_replay_trace.csv --replay_events isaaclab_sim\output\replay\mappo_dual_experts_contact_hull_seed260507_strict8\strict_replay_events.jsonl --replay_episode 5 --record_video docs\media\isaaclab_contact_hull_top.mp4 --record_view top --record_fps 12 --record_width 1280 --record_height 720
cmd /c """C:\Users\Administrator\IsaacLab\isaaclab.bat"" -p isaaclab_sim\robocup_visionrl_arena_sim.py --headless --duration 32 --replay_trace isaaclab_sim\output\replay\mappo_dual_experts_contact_hull_seed260507_strict8\strict_replay_trace.csv --replay_events isaaclab_sim\output\replay\mappo_dual_experts_contact_hull_seed260507_strict8\strict_replay_events.jsonl --replay_episode 5 --record_video docs\media\isaaclab_contact_hull_yellow_pov.mp4 --record_view yellow_pov --record_fps 12 --record_width 1280 --record_height 720
cmd /c """C:\Users\Administrator\IsaacLab\isaaclab.bat"" -p isaaclab_sim\robocup_visionrl_arena_sim.py --headless --duration 32 --replay_trace isaaclab_sim\output\replay\mappo_dual_experts_contact_hull_seed260507_strict8\strict_replay_trace.csv --replay_events isaaclab_sim\output\replay\mappo_dual_experts_contact_hull_seed260507_strict8\strict_replay_events.jsonl --replay_episode 5 --record_video docs\media\isaaclab_contact_hull_blue_pov.mp4 --record_view blue_pov --record_fps 12 --record_width 1280 --record_height 720
```

## Precision Shooting Run

The current precision rule model uses shooter-outlet distance rather than robot-center distance:

- valid normal-target laser range: `0.05 m` to `0.50 m`
- valid recessed-base laser range: `0.20 m` to `0.80 m`
- closer, centered shots have higher probability
- close shots can cost more time because the robot must drive to a tighter firing pose
- target contact does not knock down targets
- 0.3 m cube obstacles are pushable; fences, armor, walls and targets are static blockers

Run the updated MAPPO training:

```bash
python3 isaaclab_sim/rl/train_mappo_selfplay_parallel_torch.py \
  --config isaaclab_sim/rl/configs/mappo_selfplay.yaml \
  --timesteps 1000000 \
  --num-envs 32 \
  --rollout-steps 256 \
  --device cuda \
  --output ../output/rl/mappo_precision_005_050_outlet_gpu
```

The implementation note is `docs/rl_precision_shooting_model.md`.

## Rule-Accurate 45-Degree Target and Dwell-Fire Run

This stricter run models opponent-only shooting, 45-degree target placement, a 5-50 cm normal-target shooter-outlet range, a 20-80 cm recessed-base range, 0.8 s laser dwell, and pushable 0.3 m obstacle boxes:

```bash
python3 isaaclab_sim/rl/train_mappo_selfplay_parallel_torch.py \
  --config configs/mappo_selfplay.yaml \
  --timesteps 500000 \
  --num-envs 32 \
  --rollout-steps 256 \
  --device cuda \
  --policy-mode residual_expert \
  --residual-scale 0.22 \
  --residual-l2-coef 0.004 \
  --actor-mode dual \
  --output ../output/rl/mappo_rule45_dwell_push_residual_gpu
```

Build the fair deployment checkpoint and run strict stochastic replay:

```bash
python3 isaaclab_sim/rl/build_balanced_dual_actor_policy.py \
  --checkpoints isaaclab_sim/output/rl/mappo_rule45_dwell_push_residual_gpu/policy.pt \
  --output isaaclab_sim/output/rl/mappo_rule45_dwell_push_residual_balanced \
  --log-std -1.7

python3 isaaclab_sim/rl/replay_mappo_policy_strict.py \
  --checkpoint isaaclab_sim/output/rl/mappo_rule45_dwell_push_residual_balanced/policy.pt \
  --episodes 64 \
  --seed 5000 \
  --device cuda \
  --stochastic \
  --policy-mode residual_expert \
  --residual-scale 0.08 \
  --output-dir ../output/replay/mappo_rule45_dwell_push_balanced_strict64 \
  --report ../../docs/rl_rule45_dwell_push_strict_audit.md
```

Render the audited replay:

```bash
python3 isaaclab_sim/rl/render_strict_replay_video.py \
  --trace isaaclab_sim/output/replay/mappo_rule45_dwell_push_balanced_strict64/strict_replay_trace.csv \
  --events isaaclab_sim/output/replay/mappo_rule45_dwell_push_balanced_strict64/strict_replay_events.jsonl \
  --summary isaaclab_sim/output/replay/mappo_rule45_dwell_push_balanced_strict64/strict_replay_summary.json \
  --episode 0 \
  --step-stride 4 \
  --fps 12 \
  --output docs/media/rule45_dwell_push_strict_replay_episode0.mp4
```

The update report is `docs/rl_rule45_dwell_push_update.md`.

## Tactical Contact Run

This run lets the high-level policy decide whether blocking/contact is worth the risk. Robot-robot contact is treated as a tactical event and does not trigger relocalization; wall, armor, target, and jammed-obstacle contact remain recovery-relevant.

```bash
python3 isaaclab_sim/rl/train_mappo_selfplay_parallel_torch.py \
  --config configs/mappo_selfplay.yaml \
  --timesteps 250000 \
  --num-envs 32 \
  --rollout-steps 256 \
  --device cuda \
  --policy-mode residual_expert \
  --residual-scale 0.55 \
  --residual-l2-coef 0.002 \
  --actor-mode dual \
  --output ../output/rl/mappo_tactical_contact_open_gate_gpu
```

Evaluate and audit:

```bash
python3 isaaclab_sim/rl/evaluate_mappo_policy.py \
  --checkpoint isaaclab_sim/output/rl/mappo_tactical_contact_open_gate_gpu/policy.pt \
  --episodes 64 \
  --seed 7200 \
  --device cuda \
  --stochastic \
  --policy-mode residual_expert \
  --residual-scale 0.55 \
  --output isaaclab_sim/output/eval/mappo_tactical_contact_open_gate_eval64_stochastic.json

python3 isaaclab_sim/rl/replay_mappo_policy_strict.py \
  --checkpoint isaaclab_sim/output/rl/mappo_tactical_contact_open_gate_gpu/policy.pt \
  --episodes 16 \
  --seed 7600 \
  --device cuda \
  --stochastic \
  --policy-mode residual_expert \
  --residual-scale 0.55 \
  --output-dir ../output/replay/mappo_tactical_contact_open_gate_strict16 \
  --report ../../docs/rl_tactical_contact_strict_audit.md
```

The update report is `docs/rl_tactical_contact_update.md`.

## Final Domain-Randomized Shielded Recessed-Base Run

This is the latest rule-accurate embodied run used for the README MP4s. It keeps base targets smaller and recessed behind ground-touching blue armor blockers, preserves 45-degree normal target placement, records dynamic pushable box poses, uses a shared actor to reduce yellow/blue bias, randomizes Sim2Real dynamics/sensor parameters per episode, and enables the geometry-aware action shield during training.

Train:

```bash
python3 isaaclab_sim/rl/train_mappo_selfplay_parallel_torch.py \
  --config isaaclab_sim/rl/configs/mappo_selfplay.yaml \
  --timesteps 220000 \
  --num-envs 24 \
  --rollout-steps 128 \
  --update-epochs 3 \
  --minibatch-size 2048 \
  --hidden-dim 192 \
  --device cuda \
  --seed 419 \
  --policy-mode residual_expert \
  --residual-scale 0.40 \
  --residual-l2-coef 0.0015 \
  --actor-mode shared \
  --domain-randomization \
  --output ../output/rl/mappo_drshield_recessed_base_shared_gpu_seed419
```

Evaluate two stochastic seeds:

```bash
python3 isaaclab_sim/rl/evaluate_mappo_policy.py \
  --checkpoint isaaclab_sim/output/rl/mappo_drshield_recessed_base_shared_gpu_seed419/policy.pt \
  --episodes 64 \
  --seed 3100 \
  --max-steps 1800 \
  --device cuda \
  --stochastic \
  --trace-episodes 8 \
  --output isaaclab_sim/output/eval/mappo_drshield_recessed_base_seed419_eval64_seed3100_stochastic.json

python3 isaaclab_sim/rl/evaluate_mappo_policy.py \
  --checkpoint isaaclab_sim/output/rl/mappo_drshield_recessed_base_shared_gpu_seed419/policy.pt \
  --episodes 64 \
  --seed 3200 \
  --max-steps 1800 \
  --device cuda \
  --stochastic \
  --trace-episodes 8 \
  --output isaaclab_sim/output/eval/mappo_drshield_recessed_base_seed419_eval64_seed3200_stochastic.json
```

Run strict audit:

```bash
python3 isaaclab_sim/rl/replay_mappo_policy_strict.py \
  --checkpoint isaaclab_sim/output/rl/mappo_drshield_recessed_base_shared_gpu_seed419/policy.pt \
  --episodes 16 \
  --seed 3300 \
  --max-steps 1800 \
  --device cuda \
  --stochastic \
  --output-dir ../output/replay/mappo_drshield_recessed_base_seed419_strict16 \
  --report ../../docs/rl_drshield_recessed_base_strict_audit.md
```

Render the audited IsaacLab videos from the current contact-hull strict episode 5:

```powershell
& "C:\Users\Administrator\IsaacLab\isaaclab.bat" -p "<repo-root>\isaaclab_sim\robocup_visionrl_arena_sim.py" `
  --headless --duration 32 `
  --replay_trace "<repo-root>\isaaclab_sim\output\replay\mappo_dual_experts_contact_hull_seed260507_strict8\strict_replay_trace.csv" `
  --replay_events "<repo-root>\isaaclab_sim\output\replay\mappo_dual_experts_contact_hull_seed260507_strict8\strict_replay_events.jsonl" `
  --replay_episode 5 `
  --record_video "<repo-root>\docs\media\isaaclab_contact_hull_top.mp4" `
  --record_view top --record_fps 12 --record_width 1280 --record_height 720
```

Repeat the render command with `--record_view yellow_pov` / `isaaclab_contact_hull_yellow_pov.mp4` and `--record_view blue_pov` / `isaaclab_contact_hull_blue_pov.mp4` for the two robot POV videos.

## ROS2 Build

Use Ubuntu 24.04 with ROS2 Jazzy:

```bash
cd crc_robocup_vision_ws
rosdep install --from-paths src --ignore-src -r -y --rosdistro jazzy
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --event-handlers console_direct+
```

## ROS2 Dry Run

```bash
source crc_robocup_vision_ws/install/setup.bash
ros2 launch rcvrl_bringup competition.launch.py start_navigation:=false shooter_dry_run:=true auto_start:=false
```

Yellow-side elimination route:

```bash
ros2 launch rcvrl_bringup competition.launch.py \
  team_color:=yellow \
  target_file:=$(ros2 pkg prefix rcvrl_navigation)/share/rcvrl_navigation/config/targets.elimination.yellow.yaml
```

Blue-side elimination route:

```bash
ros2 launch rcvrl_bringup competition.launch.py \
  team_color:=blue \
  target_file:=$(ros2 pkg prefix rcvrl_navigation)/share/rcvrl_navigation/config/targets.elimination.blue.yaml
```

## IsaacLab Preview

Run from a local IsaacLab checkout:

```powershell
.\isaaclab.bat -p <repo-root>\isaaclab_sim\robocup_visionrl_arena_sim.py
```

Headless smoke:

```powershell
.\isaaclab.bat -p <repo-root>\isaaclab_sim\robocup_visionrl_arena_sim.py --headless --duration 5
```

## Replacing Hardware Parameters

Edit these files for the real robot:

- `crc_robocup_vision_ws/src/rcvrl_bringup/config/sim2real.yaml`
- `crc_robocup_vision_ws/src/rcvrl_bringup/config/sensor_fusion.yaml`
- `crc_robocup_vision_ws/src/rcvrl_shooter/config/shooter.yaml`
- `crc_robocup_vision_ws/src/rcvrl_vision/config/vision.yaml`
- `crc_robocup_vision_ws/src/rcvrl_description/urdf/robocup_visionrl_robot.urdf.xacro`

Record every real run with rosbag2:

```bash
ros2 bag record /tf /tf_static /scan /imu/data_raw /wheel/odom /odometry/filtered \
  /camera/image_raw /camera/camera_info /target_detection /cmd_vel
```
