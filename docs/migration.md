# ROS1 to ROS2 Migration

## Baseline

The historical prototype used ROS1 Melodic with `catkin_make`, `move_base`, `actionlib`, Gmapping, AMCL, ROS1 AprilTag nodes and serial control code with hard-coded device names and command bytes.

## ROS2 Target

- Ubuntu 24.04
- ROS2 Jazzy
- `colcon build`
- `ament_cmake`
- Nav2 `NavigateToPose`
- `slam_toolbox`
- `rclcpp`
- Python launch files

## Main Changes

| ROS1 prototype | ROS2 portfolio project |
| --- | --- |
| `catkin_make` workspace | `colcon` workspace |
| `roscpp` nodes mixed with launch shell calls | `rclcpp` nodes with explicit services/actions |
| `move_base` + `actionlib` | Nav2 `NavigateToPose` action |
| `tf` | `tf2` and `robot_state_publisher` |
| XML launch files | Python launch files |
| Hard-coded navigation points in C++ | YAML target route files |
| Vision node directly controlled base motion | Vision publishes `TargetDetection`; behavior decides |
| `/dev/arm`, byte commands in code | YAML-configured serial port and command bytes |
| Third-party tutorial packages in workspace | Clean package boundaries and third-party notices |

## Migration Notes

The new architecture separates perception, decision, navigation and actuation. This makes the project easier to explain in a portfolio review: each node has one responsibility, all competition constants are visible in YAML, and the state machine can be tested without camera or shooter hardware by replaying messages and using dry-run shooter mode.

