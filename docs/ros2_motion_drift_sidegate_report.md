# ROS2 Motion Drift Recheck For Side-Gated Base Attack

Date: 2026-05-05

This run verifies that WSL2 ROS2 realtime topics can be collected locally before retraining the IsaacLab self-play policy. The recorder launched the project ROS2 motion source and sampled live topics instead of reading a static fixture.

## Command

```bash
bash scripts/collect_ros2_motion_drift_recheck.sh \
  docs/rl_data/ros2_motion_drift_sidegate_20260505 \
  18.0
```

## Live Topics

The launch published and sampled:

- `/cmd_vel`
- `/imu/data_raw`
- `/odometry/filtered`
- `/scan`
- `/wheel/odom`

Topic metadata and one-shot samples are stored in `docs/rl_data/ros2_motion_drift_sidegate_20260505/`.

## Summary

| Metric | Value |
|---|---:|
| Samples | 358 |
| Duration | 38.600 s |
| Max command linear speed | 0.45507 m/s |
| Max command angular speed | 1.15000 rad/s |
| Max linear acceleration | 1.20034 m/s^2 |
| Max angular acceleration | 18.50178 rad/s^2 |
| Mean odom XY residual | 0.09064 m |
| 95th percentile odom XY residual | 0.19501 m |
| Mean odom yaw residual | 0.12165 rad |
| Mean front scan clearance | 0.61115 m |
| Minimum front scan clearance | 0.38001 m |
| Mean drift risk | 0.21272 |
| 95th percentile drift risk | 0.32000 |
| Max drift risk | 1.00000 |
| High-acceleration mean drift risk | 0.53916 |
| Low-acceleration mean drift risk | 0.20810 |

## Training Use

The IsaacLab/RL environment uses this data to keep acceleration-dependent localization drift in the observation and reward model. High acceleration is treated as a higher drift-risk regime, while robot-to-robot contact is handled as a tactical contact event instead of an automatic relocalization trigger.

