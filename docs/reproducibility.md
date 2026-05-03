# Reproducibility

These commands reproduce the public smoke tests and dry-run paths without requiring private competition files.

## Python RL Smoke Tests

```bash
python -m pip install -r isaaclab_sim/rl/requirements.txt
python -m pytest tests -q
```

Run a deterministic rule-environment evaluation:

```bash
cd isaaclab_sim/rl
python evaluate_selfplay.py --episodes 16 --output ../output/eval/selfplay_summary.json
```

Run the vectorized self-play rollout check:

```bash
cd isaaclab_sim/rl
python robocup_visionrl_selfplay_vec.py
```

## ROS2 Build

Use Ubuntu 24.04 with ROS2 Jazzy:

```bash
cd crc_robocup_vision_ws
rosdep install --from-paths src --ignore-src -r -y --rosdistro jazzy
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --event-handlers console_direct+
```

## ROS2 Dry Run

```bash
source crc_robocup_vision_ws/install/setup.bash
ros2 launch rcvrl_bringup competition.launch.py start_navigation:=false shooter_dry_run:=true auto_start:=false
```

Yellow-side elimination route:

```bash
ros2 launch rcvrl_bringup competition.launch.py \
  team_color:=yellow \
  target_file:=$(ros2 pkg prefix rcvrl_navigation)/share/rcvrl_navigation/config/targets.elimination.yellow.yaml
```

Blue-side elimination route:

```bash
ros2 launch rcvrl_bringup competition.launch.py \
  team_color:=blue \
  target_file:=$(ros2 pkg prefix rcvrl_navigation)/share/rcvrl_navigation/config/targets.elimination.blue.yaml
```

## IsaacLab Preview

Run from a local IsaacLab checkout:

```powershell
.\isaaclab.bat -p <repo-root>\isaaclab_sim\robocup_visionrl_arena_sim.py
```

Headless smoke:

```powershell
.\isaaclab.bat -p <repo-root>\isaaclab_sim\robocup_visionrl_arena_sim.py --headless --duration 5
```

## Replacing Hardware Parameters

Edit these files for the real robot:

- `crc_robocup_vision_ws/src/rcvrl_bringup/config/sim2real.yaml`
- `crc_robocup_vision_ws/src/rcvrl_bringup/config/sensor_fusion.yaml`
- `crc_robocup_vision_ws/src/rcvrl_shooter/config/shooter.yaml`
- `crc_robocup_vision_ws/src/rcvrl_vision/config/vision.yaml`
- `crc_robocup_vision_ws/src/rcvrl_description/urdf/robocup_visionrl_robot.urdf.xacro`

Record every real run with rosbag2:

```bash
ros2 bag record /tf /tf_static /scan /imu/data_raw /wheel/odom /odometry/filtered \
  /camera/image_raw /camera/camera_info /target_detection /cmd_vel
```
