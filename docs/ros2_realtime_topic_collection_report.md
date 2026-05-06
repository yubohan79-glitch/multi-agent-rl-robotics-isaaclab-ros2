# ROS2 Realtime Topic Collection Report

Date: 2026-05-04

## Result

The missing live-topic problem was reproduced and fixed. Before starting any project publisher, WSL2 ROS2 only exposed:

```text
/parameter_events [rcl_interfaces/msg/ParameterEvent]
/rosout [rcl_interfaces/msg/Log]
```

After launching the project ROS2 motion source and recorder, the following realtime topics were visible and sampled:

```text
/cmd_vel [geometry_msgs/msg/Twist]
/imu/data_raw [sensor_msgs/msg/Imu]
/odometry/filtered [nav_msgs/msg/Odometry]
/scan [sensor_msgs/msg/LaserScan]
/wheel/odom [nav_msgs/msg/Odometry]
```

This confirms the recorder was not reading a static file. It subscribed to active ROS2 topics and wrote a fresh CSV log.

## Command Used

```bash
cd /mnt/c/Users/Administrator/Desktop/作品集/RoboCupVisionRL_IsaacLab_ROS2
source /opt/ros/humble/setup.bash
source crc_robocup_vision_ws/install/setup.bash

ros2 launch rcvrl_motion motion_drift_sim_collection.launch.py \
  output_csv:=docs/rl_data/ros2_motion_drift_live/motion_drift_live_log.csv \
  duration_s:=20.0
```

Topic checks were run while the launch was alive:

```bash
ros2 topic list --no-daemon -t
ros2 topic info /cmd_vel
ros2 topic info /scan
ros2 topic hz /cmd_vel --window 8
ros2 topic hz /scan --window 8
ros2 topic echo /cmd_vel --once
ros2 topic echo /wheel/odom --once
```

Only project-owned collection processes were stopped after sampling.

## Collected Files

| File | Purpose |
|---|---|
| `docs/rl_data/ros2_motion_drift_live/motion_drift_live_log.csv` | Fresh realtime ROS2 topic recording |
| `docs/rl_data/ros2_motion_drift_live/motion_drift_live_summary.json` | Statistics generated from the fresh CSV |
| `docs/rl_data/ros2_motion_drift_live/topic_list.txt` | Topic list captured while publishers were live |
| `docs/rl_data/ros2_motion_drift_live/cmd_vel_hz.txt` | `/cmd_vel` frequency check |
| `docs/rl_data/ros2_motion_drift_live/scan_hz.txt` | `/scan` frequency check |
| `docs/rl_data/ros2_motion_drift_live/wheel_odom_once.txt` | One live odometry sample |
| `docs/rl_data/rules_pdf_extract.txt` | Text extracted from the local 2025 visual challenge rule PDF |

## Data Summary

| Metric | Value |
|---|---:|
| Samples | 228 |
| Duration | 22.700 s |
| `/cmd_vel` rate | about 30 Hz |
| `/scan` rate | about 30 Hz |
| Max command linear speed | 0.44503 m/s |
| Max command angular speed | 1.15000 rad/s |
| Max linear acceleration | 1.20299 m/s^2 |
| Max angular acceleration | 18.45786 rad/s^2 |
| Mean wheel-vs-filtered odom XY residual | 0.10177 m |
| 95th percentile odom XY residual | 0.19650 m |
| Mean odom yaw residual | 0.08104 rad |
| Mean front scan clearance | 0.60984 m |
| Minimum front scan clearance | 0.30401 m |
| Mean drift risk | 0.19078 |
| 95th percentile drift risk | 0.32000 |
| Max drift risk | 1.00000 |
| High-acceleration mean drift risk | 0.54432 |
| Low-acceleration mean drift risk | 0.18286 |

The high-acceleration samples produce substantially higher drift risk, so the RL environment now has a data-backed reason to penalize abrupt acceleration and to model localization confidence as a multi-sensor fusion variable.

## ROS1 To ROS2 Integration

The old workspaces were reviewed and mapped into the current ROS2 stack.

| Source | ROS1 contract found | ROS2 integration |
|---|---|---|
| `shoot_robot（mini）` | `/cmd_vel`, `move_base`, `tag_detections`, `/shoot`, `/close` | Nav2 `navigate_to_pose`, `/target_detection`, `/shooter/enable`, `/shooter/fire`, `/shooter/disable` |
| `shoot_service.cpp` | serial `/dev/arm`, 9600 baud, open byte `0xA3`, close byte `0xA0` | `rcvrl_shooter/src/shooter_controller.cpp` keeps `/dev/arm`, 9600, configurable `[163]` enable/fire and `[160]` disable |
| `apriltag_detect.cpp` | AprilTag36h11 target alignment, target distance around `0.52 m` | `rcvrl_vision/src/apriltag_detector.cpp` publishes `rcvrl_interfaces/TargetDetection`; behavior uses `target_distance_m: 0.52` |
| `robot_ws.zip` Mini base stack | wheel odom, IMU, RPLidar `/scan`, EKF odometry | `rcvrl_motion` recorder and simulated source use `/wheel/odom`, `/imu/data_raw`, `/scan`, `/odometry/filtered` |

The competition launch now wires behavior, shooter, vision, EKF, and optional drift recording through ROS2.

## Rule Notes From Local PDF

The extracted 2025 visual-challenge rule PDF confirms:

- 8 normal targets are placed in the arena.
- Each base has one base target and 4 armor plates.
- Normal target AprilTag ID is `1`; yellow base target ID is `2`; blue base target ID is `3`.
- Normal targets can be set to 30, 45, or 60 degrees in elimination; base targets are tilted 45 degrees.
- Knocking a normal target removes one opponent base armor plate in order.
- After knocking one normal target, the robot may continue normal targets or directly attack the opponent base target.
- Normal targets score 5; base target scores 60 or wins in elimination.
- Collision-caused target falls are penalized.
- The laser module must be statically fixed on the robot and must be 5 V, no more than 0.25 W.

The official PDF text does not state a numeric continuous laser dwell time. The project therefore keeps the `0.80 s` dwell threshold as a practice-calibrated sim2real safety rule from field testing, and models target fall probability between `0.80 s` and `2.00 s`.

## IsaacLab/RL Rule Sync

Code changes made for the next training run:

- Base shots no longer require the robot fire pose to be against the wall.
- One- or two-normal-target base rushes require a narrow off-axis shooting angle, so they are possible but harder.
- Base hit caps are hardened for early-rush realism: 1 normal hit `40%`, 2 hits `55%`, 3 hits `80%`, 4 hits `95%`.
- Grounded base armor remains a laser blocker.
- Normal-target laser range remains `0.05 m` to `0.50 m` from the shooter outlet; recessed base-target range is `0.20 m` to `0.80 m`.
- Laser dwell remains required before target fall.
- Pushable boxes remain dynamic state in the RL trace so IsaacLab replay can move them instead of letting the robot pass through them.

## Figures To Draw

Architecture figures:

1. ROS2-to-IsaacLab embodied stack: wheel odom, IMU, lidar scan, camera AprilTag, shooter services, Nav2, and MAPPO policy.
2. Multi-sensor fusion contract: `/wheel/odom`, `/imu/data_raw`, `/scan`, `/odometry/filtered`, camera target visibility, bumper/contact, and drift risk.
3. Rule-gated shooting model: 5-50 cm outlet range, off-axis base attack gate, grounded armor laser blocking, 0.80 s dwell, probabilistic fall.

Experimental figures using real collected data:

1. Acceleration vs drift risk from `motion_drift_live_log.csv`.
2. Odom XY residual over time from `motion_drift_live_log.csv`.
3. Front scan clearance over time from `motion_drift_live_log.csv`.
4. Win-rate and base-rush hit-count distribution after the next multi-seed training.
5. Strict replay audit table: hard penetration, own-target shot, target contact, box displacement, and base win.
