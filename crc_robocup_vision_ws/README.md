# CRC RoboCup Vision ROS2 Workspace

This workspace is the ROS2 portfolio version of the China Robot Competition / RoboCup China robot vision challenge robot.

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
ros2 launch wvb_bringup competition.launch.py
```

Hardware-free launch check:

```bash
ros2 launch wvb_bringup competition.launch.py start_navigation:=false shooter_dry_run:=true auto_start:=false
```

WSL note: build from a native Linux path such as `~/crc_robocup_vision_ws`. ROSIDL may fail if the workspace is built directly from a Windows-mounted path containing non-ASCII characters.

Package layout:

- `wvb_bringup`: system launch files and top-level runtime wiring
- `wvb_navigation`: Nav2, slam_toolbox, maps and target route configuration
- `wvb_vision`: AprilTag Tag36h11 detection from camera images
- `wvb_shooter`: serial laser module controller
- `wvb_behavior`: competition state machine
- `wvb_description`: robot URDF and frame description
- `wvb_interfaces`: custom ROS2 message definitions
- `wvb_docs`: project documentation used for portfolio submission
