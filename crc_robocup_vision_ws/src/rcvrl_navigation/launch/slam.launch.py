from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    slam_params_file = LaunchConfiguration("slam_params_file")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "slam_params_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("rcvrl_navigation"),
                "config",
                "slam_toolbox.yaml",
            ]),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("slam_toolbox"),
                    "launch",
                    "online_async_launch.py",
                ])
            ),
            launch_arguments={
                "use_sim_time": use_sim_time,
                "slam_params_file": slam_params_file,
            }.items(),
        ),
    ])

