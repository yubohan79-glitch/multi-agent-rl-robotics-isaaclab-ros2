# Sensor-Fusion Embodied RL Update

This update turns the strategy layer into a clearer embodied-intelligence setup: the high-level policy still decides target selection, base rush, interference, recovery, fire gating and risk, but the observation now includes a compact multi-sensor fusion state instead of a single abstract localization scalar.

## What Changed

- ROS2 behavior now subscribes to `/odometry/filtered` and estimates EKF confidence from pose covariance.
- `robot_localization/ekf_node` remains launched from `rcvrl_bringup` with `/wheel/odom` and `/imu/data_raw`.
- RL observation dimension increased from 39 to 46.
- New fusion features:
  - robot contact flag
  - fused localization confidence
  - wheel/IMU motion consistency
  - scan/costmap clearance
  - front-left and front-right ToF clearance
  - bumper or hard-contact flag
  - camera-visible opponent target flag
  - pushable obstacle contact flag
- Recovery is no longer triggered by robot-robot contact. Tactical contact is allowed when useful.
- Shooting uses a 5-50 cm laser-outlet range, distance-dependent accuracy and 0.80 s dwell before target knockdown.
- The target layout keeps boards inset from the wall and angled about 45 degrees.
- Base targets are smaller and recessed behind ground-touching blue armor blockers; the base target is not visible or hittable from outside until armor is removed.
- Pushable boxes are recorded in strict replay as dynamic state (`box_ne_x/y`, `box_sw_x/y`) and applied to IsaacLab playback.
- The latest policy adds reset-time Sim2Real domain randomization over drive/turn scale, push response, shot accuracy, drift loss and sensor noise.
- The latest policy also enables a geometry-aware action shield that suppresses unsafe contact/fire commands and logs the intervention.

## Training Run

| Item | Value |
|---|---:|
| Device | NVIDIA GeForce RTX 4090 |
| Torch | 2.11.0+cu128 |
| Timesteps | 221,184 |
| Parallel envs | 24 |
| Rollout steps | 128 |
| Policy mode | residual expert |
| Actor mode | shared actor for yellow/blue fairness |
| Observation dim | 46 |
| Central critic dim | 92 |
| Domain randomization | enabled |
| Action shield | enabled |
| Training wall time | 905.130 s |
| Final training throughput | 240.20 steps/s |

## Evaluation

Final stochastic evaluation:

| Seed | Episodes | Yellow win | Blue win | Draw / timeout | Base-hit wins / episode | Own-target penalties / episode |
|---:|---:|---:|---:|---:|---:|---:|
| 3100 | 64 | 50.00% | 48.44% | 1.56% | 0.9844 | 0.0 |
| 3200 | 64 | 50.00% | 48.44% | 1.56% | 0.9844 | 0.0 |
| Combined | 128 | 50.00% | 48.44% | 1.56% | 0.9844 | 0.0 |

Strict replay audit:

| Metric | Value |
|---|---:|
| Episodes | 16 |
| Yellow win rate | 62.50% |
| Blue win rate | 37.50% |
| Hard violations | 0 |
| Warnings | 0 |
| Own-target penalties / episode | 0.0 |
| Blocked steps / episode | 0.0 |
| Target contact events / episode | 0.0 |
| Robot contacts / episode | 28.25 |
| Base wins / episode | 1.0 |

## Videos

- `docs/media/final_training_replay_overview.mp4`
- `docs/media/final_training_replay_yellow_pov.mp4`
- `docs/media/final_training_replay_blue_pov.mp4`

The selected IsaacLab replay is strict trace episode 1. It shows two robots starting together, legal opponent-side target hits, 0.80 s dwell-gated knockdown events, pushable rigid boxes moving in the map frame, and final base target defeat after the recessed armor window opens. The rendered MP4s are synchronized 28.04 s / 24 fps recordings.

## Figures To Draw

Architecture figures:

1. Multi-Sensor Fusion Runtime Graph  
   Show wheel odometry, IMU, lidar scan/costmap, ToF, bumper and camera target observations feeding EKF/fusion confidence, then behavior/RL action gating.

2. Embodied Strategy Stack  
   CTDE MAPPO at the top, competition behavior state machine in the middle, Nav2/differential drive/AprilTag/shooter timing at the bottom.

3. Rule-Gated Shooting Model  
   Show laser outlet range 5-50 cm, 0.80 s dwell timer, distance-dependent accuracy and opponent-target safety gate.

Experimental figures using real data:

1. Training Curve  
   Use `docs/rl_data/drshield_recessed_base_shared/training_curve.csv`.

2. Win Rate and Safety Summary  
   Use `docs/rl_data/drshield_recessed_base_shared/eval_seed3100_stochastic.json`, `eval_seed3200_stochastic.json` and `strict_replay_summary.json`.

3. Normal Targets Before Base Distribution  
   Same evaluation figure includes the count distribution. This supports the observation that one to two normal targets before base rush is the preferred strategy.

4. Strict Replay Audit Table  
   Plot hard violations, warnings, own-target penalties, target contact events and blocked steps. All strict safety counts are zero in the selected audit.

5. Sim2Real Sensor Contract Table  
   Use a table/diagram linking `/wheel/odom`, `/imu/data_raw`, `/scan`, ToF, bumper, `/target_detection`, `/odometry/filtered`, `/cmd_vel` and `/shooter/fire`.

## Data Sources

- `docs/rl_data/drshield_recessed_base_shared/training_curve.csv`
- `docs/rl_data/drshield_recessed_base_shared/training_summary.json`
- `docs/rl_data/drshield_recessed_base_shared/eval_seed3100_stochastic.json`
- `docs/rl_data/drshield_recessed_base_shared/eval_seed3200_stochastic.json`
- `docs/rl_data/drshield_recessed_base_shared/strict_replay_summary.json`
- `isaaclab_sim/output/replay/mappo_drshield_recessed_base_seed419_strict16/strict_replay_trace.csv`
