# ROS2 Motion Drift Data Report

Date: 2026-05-04

## Purpose

This experiment replaces the missing live hardware topics with a reproducible ROS2 simulated topic source. The source publishes the same motion/sensor contract inherited from the Mini ROS1 stack, and the `rcvrl_motion` recorder subscribes to those topics and writes a real CSV log.

## Source Interfaces

ROS1 source material reviewed:

- `shoot_robot(mini)`: `/cmd_vel`, `move_base`, AprilTag alignment, and serial laser `shoot` / `close` services.
- `robot_ws.zip`: wheel odometry, IMU, RPLidar `/scan`, and EKF-style fused odometry contracts.

ROS2 collection topics:

- `/cmd_vel`
- `/wheel/odom`
- `/odometry/filtered`
- `/imu/data_raw`
- `/scan`

## Environment Check

WSL2 has ROS2 Humble available and Gazebo Sim installed:

- `ros2`: `/opt/ros/humble/bin/ros2`
- Gazebo Sim: `ign gazebo`, version `6.16.0`
- ROS-Gazebo packages: `ros_gz`, `ros_gz_bridge`, `ros_gz_sim`

The current project does not yet contain a dedicated Gazebo world/model bridge for the RoboCup arena, so the first data collection uses the project-owned ROS2 simulated topic source:

- node: `rcvrl_motion.motion_drift_sim_source`
- recorder: `rcvrl_motion.motion_drift_recorder`
- launch: `ros2 launch rcvrl_motion motion_drift_sim_collection.launch.py`

## Collected Files

- CSV: `docs/rl_data/ros2_motion_drift/motion_drift_sim_log.csv`
- Summary JSON: `docs/rl_data/ros2_motion_drift/motion_drift_sim_summary.json`

## Data Summary

| Metric | Value |
| --- | ---: |
| Samples | 463 |
| Duration | 49.000 s |
| Max command linear speed | 0.45504 m/s |
| Max command angular speed | 0.70000 rad/s |
| Max linear acceleration | 13.24076 m/s^2 |
| Max angular acceleration | 18.49825 rad/s^2 |
| Mean odom XY residual | 0.141894 m |
| 95th percentile odom XY residual | 0.262370 m |
| Mean odom yaw residual | 0.076120 rad |
| 95th percentile odom yaw residual | 0.359690 rad |
| Mean front scan clearance | 0.622477 m |
| Mean drift risk | 0.219441 |
| Max drift risk | 1.000000 |
| Correlation: normalized acceleration vs drift risk | 0.61958 |

## Acceleration Bins

| Bin | Samples | Mean drift risk | Mean odom XY residual | Mean IMU-command yaw residual |
| --- | ---: | ---: | ---: | ---: |
| Low acceleration | 453 | 0.210588 | 0.140972 m | 0.000000 rad/s |
| Nominal threshold band | 0 | 0.000000 | 0.000000 m | 0.000000 rad/s |
| High acceleration | 9 | 0.578307 | 0.186304 m | 0.000000 rad/s |

## Interpretation

The recorded data supports the training-side assumption that aggressive command changes increase localization risk. In this run, high-acceleration samples show a mean drift risk of `0.578307`, compared with `0.210588` in low-acceleration samples.

For the RL environment, this justifies penalizing excessive linear/angular acceleration through the sensor-fusion confidence model while allowing robot-to-robot contact to remain a tactical event that does not automatically trigger relocalization.

## Next Data Upgrade

The next measurement layer should bridge the IsaacLab arena state into the same ROS2 topics. That will let the same recorder compare:

- IsaacLab ground-truth pose vs wheel odometry.
- EKF-like filtered odometry vs ground truth.
- Laser scan clearance near fixed boards, grounded base armor, and pushable boxes.
- Acceleration-induced map drift during base-rush trajectories.
