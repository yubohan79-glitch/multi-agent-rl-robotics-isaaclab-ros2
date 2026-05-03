from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_file = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "map",
            default_value=PathJoinSubstitution([
                FindPackageShare("rcvrl_navigation"),
                "maps",
                "robocup_visionrl_arena.yaml",
            ]),
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("rcvrl_navigation"),
                "config",
                "nav2_params.yaml",
            ]),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("nav2_bringup"),
                    "launch",
                    "bringup_launch.py",
                ])
            ),
            launch_arguments={
                "map": map_file,
                "use_sim_time": use_sim_time,
                "params_file": params_file,
            }.items(),
        ),
    ])
