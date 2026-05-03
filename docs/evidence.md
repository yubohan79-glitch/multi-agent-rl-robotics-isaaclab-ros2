# Evidence Pack

This file lists the visual and log evidence expected for a strong GitHub/portfolio submission.

## Existing Repository Assets

| Evidence | File |
| --- | --- |
| Project overview | `docs/figures/portfolio/portfolio_overview_robocup.png` |
| Field and rule scene | `docs/figures/portfolio/arena_rule_scene.png` |
| Robot sensor layout | `docs/figures/portfolio/robot_sensor_layout.png` |
| ROS2 runtime graph | `docs/figures/portfolio/ros2_runtime_graph.png` |
| Hierarchical MAPPO policy | `docs/figures/rl/rl_hierarchical_policy.png` |
| Parallel self-play training | `docs/figures/rl/rl_selfplay_training.png` |
| Sim2Real pipeline | `docs/figures/rl/rl_sim2real_pipeline.png` |

## Evidence To Capture From Runtime

| Item | Capture method | Why it matters |
| --- | --- | --- |
| IsaacLab full arena | GUI screenshot | Shows two robots, targets, armor and obstacles |
| Target falling | short video or GIF | Proves hit-to-rule-event simulation |
| Armor removal | short video or GIF | Proves ordinary target score logic |
| RViz map and Nav2 path | screenshot | Proves ROS2 navigation integration |
| AprilTag detection | camera screenshot | Proves visual target perception |
| RL training curve | TensorBoard/CSV screenshot | Proves learning pipeline beyond scripting |
| rosbag2 replay | terminal and RViz screenshot | Proves real-run reproducibility |

## Suggested File Names

```text
docs/evidence/isaaclab_arena.png
docs/evidence/target_fall.gif
docs/evidence/armor_removal.gif
docs/evidence/rviz_nav2_path.png
docs/evidence/apriltag_detection.png
docs/evidence/mappo_training_curve.png
docs/evidence/rosbag2_replay.png
```

The `docs/evidence/` directory is intentionally not required for CI. Add runtime screenshots only after they are captured from your machine or real robot.
