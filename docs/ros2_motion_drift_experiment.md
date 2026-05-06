# ROS2 Motion Drift Experiment

This project now includes a ROS2 port of the useful motion/sensor contracts from the original Mini ROS1 stacks:

- `shoot_robot(mini)`: `/cmd_vel` navigation commands, AprilTag alignment, and serial laser open/close semantics.
- `robot_ws.zip`: wheel odometry, IMU, RPLidar `/scan`, and EKF-style fused odometry.

The ROS2 integration lives in `crc_robocup_vision_ws/src/rcvrl_motion`.

## Recorder

Run the recorder together with the competition stack:

```bash
ros2 launch rcvrl_bringup competition.launch.py \
  start_motion_drift_recorder:=true \
  motion_drift_output:=docs/rl_data/ros2_motion_drift/motion_drift_log.csv
```

Or run it by itself:

```bash
ros2 launch rcvrl_motion motion_drift_experiment.launch.py \
  output_csv:=docs/rl_data/ros2_motion_drift/motion_drift_log.csv
```

The recorder subscribes to `/cmd_vel`, `/wheel/odom`, `/odometry/filtered`, `/imu/data_raw`, and `/scan`.
It writes command acceleration, wheel-vs-EKF residuals, IMU yaw-rate residual, front scan clearance, and a normalized `drift_risk`.

## Simulated ROS2 Topic Source

When no real robot, Gazebo bridge, rosbag2 replay, or IsaacLab ROS bridge is publishing topics, run the project-owned simulated source and recorder together:

```bash
ros2 launch rcvrl_motion motion_drift_sim_collection.launch.py \
  output_csv:=docs/rl_data/ros2_motion_drift/motion_drift_sim_log.csv \
  duration_s:=42.0
```

This publishes real ROS2 messages on the same topics and lets the recorder collect a reproducible CSV. The first collected report is documented in `docs/ros2_motion_drift_data_report.md`.

## Current WSL2 Observation

On the current machine, WSL2 exposes ROS2 Humble and Gazebo Sim 6.16.0, but no RoboCup arena Gazebo world/model bridge was already running. Before the simulated source was started, only `/rosout` and `/parameter_events` were visible. After starting `motion_drift_sim_source`, the recorder collected `463` rows into `docs/rl_data/ros2_motion_drift/motion_drift_sim_log.csv`.

The recorder is wired into the project so the same experiment can be repeated with real hardware, Gazebo, IsaacLab bridge output, or a rosbag2 replay.

## RL Calibration

The self-play environment uses the same principle while training: aggressive linear/angular acceleration lowers localization confidence and wheel/IMU consistency. Robot-to-robot contact does not trigger relocalization by itself; hard contact with walls, fixed armor, target boards, or jammed pushable boxes does.
