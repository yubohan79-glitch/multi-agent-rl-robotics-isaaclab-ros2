from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    output_csv = LaunchConfiguration("output_csv")
    use_sim_time = LaunchConfiguration("use_sim_time")
    params = PathJoinSubstitution([
        FindPackageShare("rcvrl_motion"),
        "config",
        "motion_drift.yaml",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("output_csv", default_value="motion_drift_log.csv"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        Node(
            package="rcvrl_motion",
            executable="motion_drift_recorder",
            name="motion_drift_recorder",
            output="screen",
            parameters=[
                params,
                {"output_csv": output_csv, "use_sim_time": ParameterValue(use_sim_time, value_type=bool)},
            ],
        ),
    ])
