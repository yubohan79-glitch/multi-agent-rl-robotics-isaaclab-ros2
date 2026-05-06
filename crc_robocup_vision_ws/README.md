# CRC RoboCup Vision ROS2 Workspace

This workspace is the ROS2 portfolio version of the China Robot Competition / RoboCup China robot vision challenge robot. It is the runtime side of the IsaacLab-to-ROS2 project: navigation, target detection, shooter control, sensor fusion and behavior orchestration stay behind ROS2 interfaces so the learned strategy can transfer without depending on simulator-only state.

Target platform:

- Ubuntu 24.04
- ROS2 Jazzy
- Build tool: colcon

Build:

```bash
cd crc_robocup_vision_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Main launch:

```bash
ros2 launch rcvrl_bringup competition.launch.py
```

Hardware-free launch check:

```bash
ros2 launch rcvrl_bringup competition.launch.py start_navigation:=false shooter_dry_run:=true auto_start:=false
```

Record the real or simulated sensor contract for drift/fusion analysis:

```bash
ros2 bag record /tf /tf_static /scan /imu/data_raw /wheel/odom /odometry/filtered \
  /camera/image_raw /camera/camera_info /target_detection /cmd_vel \
  /range/front_left /range/front_right /bumper/front_left /bumper/front_right
```

WSL note: build from a native Linux path such as `~/crc_robocup_vision_ws`. ROSIDL may fail if the workspace is built directly from a Windows-mounted path containing non-ASCII characters.

Package layout:

- `rcvrl_bringup`: system launch files and top-level runtime wiring
- `rcvrl_navigation`: Nav2, slam_toolbox, maps and target route configuration
- `rcvrl_motion`: motion telemetry and map-drift data recorder
- `rcvrl_vision`: AprilTag Tag36h11 detection from camera images
- `rcvrl_shooter`: serial laser module controller
- `rcvrl_behavior`: competition state machine
- `rcvrl_description`: robot URDF and frame description
- `rcvrl_interfaces`: custom ROS2 message definitions
- `rcvrl_docs`: project documentation used for portfolio submission

Runtime strategy notes:

- Robots receive opponent pose/track information, but robot-robot contact does not trigger relocalization by itself.
- Shooter control should preserve the rule model used in IsaacLab: opponent targets only, 5-50 cm normal-target shooter-outlet range, 20-80 cm recessed-base range, line-of-sight blocking, and at least 0.80 s laser dwell.
- Base targets are protected by blue armor blockers in the simulation contract; the behavior layer should not assume the base target is visible until the armor route has opened a legal shot.
- The latest RL bridge trains with Sim2Real domain randomization and an action shield, so ROS2 integration should expose real odometry, IMU, lidar, camera, ToF, bumper and shooter-state signals rather than simulator-only privileged state.
