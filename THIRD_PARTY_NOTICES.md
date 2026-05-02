# Third-Party Notices

This project is packaged as an original ROS2 integration and migration project. It depends on standard ROS2 ecosystem packages and documents historical references instead of embedding unrelated tutorial work.

Runtime dependencies expected from the ROS2 distribution:

- ROS2 Jazzy core packages
- Nav2
- slam_toolbox
- robot_state_publisher
- xacro
- OpenCV
- cv_bridge
- sensor_msgs
- geometry_msgs
- std_srvs
- robot_localization

Historical ROS1 baseline packages reviewed during migration:

- Slamtec `rplidar_ros`
- WaterPlus `waterplus_map_tools`
- ROS1 `move_base`, `gmapping`, `amcl`
- ROS1 `apriltag_ros`

Robot geometry reference:

- `wvb_description/meshes/zoo/base_link.stl`
- `wvb_description/meshes/zoo/laser_link.stl`

These meshes are adapted from the local Zoo robot description package included with the project material. They are used as CAD reference assets while collision geometry, sensor frames, ROS2 package boundaries and runtime behavior are maintained in this portfolio workspace.

The ROS2 workspace does not vendor those ROS1 repositories. If hardware drivers are required for a specific robot, install the vendor-provided ROS2 driver separately and document the exact version used.
