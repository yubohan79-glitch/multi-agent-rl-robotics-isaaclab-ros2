# Results

This page separates measured smoke-test evidence from final hardware metrics. Values marked as "hardware run required" must be filled from real-robot rosbag2 logs before a formal competition report is submitted.

## Current Measured Evidence

| Layer | Test | Metric | Current result |
| --- | --- | --- | --- |
| Python RL contract | `python -m pytest tests -q` | RL env, rule config and Sim2Real config checks | Passing locally |
| Self-play rule env | `evaluate_selfplay.py --episodes 16` | Simulated steps per second | 5233.7 steps/s |
| Self-play rule env | scripted line-of-sight baseline | Own-target penalties | 0.0 per episode |
| Self-play rule env | scripted line-of-sight baseline | Mean episode horizon | 180.0 s |
| GitHub CI | `.github/workflows/ros2-ci.yml` | ROS2 Jazzy build + Python smoke tests | Configured |

The scripted baseline is intentionally conservative and is used as a regression harness, not as the final policy. It verifies environment stepping, rule-event logging, and own-target safety before MAPPO training.

## Final Hardware Metrics To Fill

| Test | Metric | Source |
| --- | --- | --- |
| AprilTag detection | ID 1/2/3 recognition rate and false-positive rate | `/camera/image_raw`, `/target_detection` |
| Single normal target | Mean search-align-fire time | rosbag2 event timestamps |
| Base target attack | Base-hit success rate under clear line of sight | target fall video + shooter service log |
| Navigation route | Reached target poses / requested poses | Nav2 feedback |
| Collision recovery | Time from bumper/contact to stable pose | `/bumper/*`, `/imu/data_raw`, `/odometry/filtered` |
| Full 180s match | Score, target hits, armor removals, winner | rule-event log + video |

## PPO vs MAPPO Evaluation Plan

| Policy | Purpose | Expected comparison |
| --- | --- | --- |
| Scripted baseline | Rule regression and CI smoke | Stable, interpretable, low tactical skill |
| PPO single-agent | Fast yellow-side attack baseline | Improves route/fire timing but ignores opponent adaptation |
| MAPPO self-play | Final tactical layer | Learns target choice, base rush timing, blocking and recovery under opponent pressure |

Recommended report metrics:

- ordinary target average time
- base target hit rate
- collision recovery time
- parallel training throughput
- own-target fire count, expected to remain zero
- PPO baseline score vs MAPPO self-play score

## Evidence Checklist

- IsaacLab GUI screenshot with two robots, targets, armor and obstacles
- IsaacLab video clip showing target fall and armor removal
- RViz screenshot with map, TF, Nav2 path and robot footprint
- AprilTag camera screenshot with detected ID and pose
- TensorBoard or CSV training curve for PPO and MAPPO
- rosbag2 replay command and final event table
