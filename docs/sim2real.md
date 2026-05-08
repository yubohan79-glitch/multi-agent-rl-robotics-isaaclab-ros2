# Sim2Real Plan

This project does not transfer a perfect Isaac Sim state directly to the real robot. The transferable layer is the ROS2 contract: `/cmd_vel`, `/target_detection`, Nav2 goals, shooter services, TF frames, and the opponent-target safety gate.

## Public Evidence Status

The repository publishes simulation, rule-environment evaluation and IsaacLab replay evidence, and the 1v1 two-robot line has subsequently been exercised on real robots. It does not yet publish a full quantified hardware benchmark package, real-arena win-rate table, migration success percentage or long-horizon real-world rosbag dataset.

The correct interpretation is:

- validated publicly: ROS2 deployment contract, calibration procedure, domain-randomization plan, rule tests, IsaacLab replay, simulation evaluation and 1v1 real-robot experiment coverage.
- not publicly validated here: full statistical 1v1 hardware success-rate table, large-scale real multi-robot deployment, 50v50 real-robot deployment and measured Sim2Real transfer percentage.

Any future quantified real-robot claim should add hardware setup, calibration logs, rosbag2 recordings, number of runs, success/failure definition, task success rate, collision/stuck statistics and arena condition notes. The exact capability matrix is maintained in `docs/capability_boundaries.md`.

## Rule Contract

- Yellow robot attacks blue-side targets only.
- Blue robot attacks yellow-side targets only.
- Own base detections are rejected before firing.
- Route files carry `target_owner`; the behavior node skips a target when `target_owner == team_color`.
- A normal target hit removes one opponent armor plate in order. A base target hit ends the match for the firing side when it is the opponent base; hitting own base loses the match.

## Calibration Order

1. Field frame: set `map` origin at the south-west arena corner and verify the 3.0m x 3.0m boundary, 0.5m start zones, 0.5m base zones, inner fences, and 0.3m cube obstacles.
2. Robot geometry: measure wheel radius, track width, chassis footprint, camera pose, lidar pose, shooter pose, and base link height. Update `rcvrl_description` and `rcvrl_bringup/config/sim2real.yaml`.
3. Sensor parity: mount and model IMU, wheel encoders, 2D lidar, RGB camera, depth/ToF range sensors, bumper contacts, and the fixed laser module with the same frames as the real robot.
4. Differential drive fit: drive straight 1m and rotate 360 degrees on the real robot. Fit wheel radius, track width, motor deadband, max speed, and acceleration limits until odom and measured motion agree.
5. EKF fit: fuse `/wheel/odom` and `/imu/data_raw` with `robot_localization`; verify yaw does not jump after collision and that odom drift stays bounded during a 360-degree spin.
6. Camera calibration: record `camera_info`, tag size 0.05m, camera-to-base TF, exposure, focus, and lighting. Validate target distance and center error with AprilTag boards at 0.3m, 0.5m, 0.8m, and 1.2m.
7. Shooter calibration: measure camera-to-beam offset, serial command latency, hit radius, and fire repeat interval. Save the offset in `sim2real.yaml` and the command bytes in `rcvrl_shooter/config/shooter.yaml`.
8. Map validation: compare Nav2 paths in sim and on the real arena. The robot must never cross fences, cube obstacles, or the other robot footprint.
9. Recovery validation: hit the bumper or push the chassis sideways, force localization confidence low, then verify the robot spins in place to rebuild the map before continuing.
10. Rule validation: run yellow and blue separately with their own route files. Confirm neither side fires on its own base tag or own target owner.

## Domain Randomization

Randomize only the values that are unstable in the real arena:

- lighting from 600 to 1200 lux
- tag yaw at 30, 45, and 60 degrees
- camera latency, image blur, exposure noise, and partial tag occlusion
- wheel slip, friction, motor deadband, and battery voltage scale
- obstacle and target placement error within the rule tolerance
- shooter fire latency and small beam-camera offset drift

## RL Interface

Use object-centric world-model SAC Flow self-play for elimination matches:

- observation: robot pose estimate, opponent-target bearing/distance, tag visibility, own/opponent armor count, time remaining, obstacle distances
- action: high-level tactical controls for target selection, base-rush timing, blocking, recovery, fire gating and risk preference
- reward: progress toward opponent targets, valid target detection, clean alignment, successful hit, legal blocking, collision penalty, own-target penalty, timeout penalty and safe push-box progress
- recovery: blocked motion and bumper contact reduce localization confidence; the policy is rewarded for spinning in place only when confidence is poor
- training: use many fast parallel rule environments before replaying the learned high-level policy in IsaacLab
- deployment: keep Nav2, AprilTag detection, EKF, and shooter services in the loop; do not deploy a raw full-state sim policy directly to the real robot

## Verification Ladder

1. Fast Python rule environment for world-model SAC Flow self-play and contract tests.
2. IsaacLab headless match replay.
3. IsaacLab GUI visual check: target falls, armor moves, two robots collide, lasers do not pass walls.
4. ROS2 dry run with `shooter_dry_run:=true`.
5. Real robot tethered test at low speed.
6. One opponent normal target drill.
7. Four opponent normal targets plus opponent base target.
8. Full 180s elimination match with rosbag2 logging.

## Required Logs

Record these for every real run:

- `/tf`, `/tf_static`, `/odom`, `/scan`, `/camera/image_raw`, `/camera/camera_info`
- `/target_detection`, `/cmd_vel`, Nav2 action feedback, shooter service calls
- target hit timestamps, armor removal order, collision/stuck events, final winner
