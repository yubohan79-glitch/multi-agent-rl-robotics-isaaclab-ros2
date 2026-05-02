from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    start_navigation = LaunchConfiguration("start_navigation")
    start_sensor_fusion = LaunchConfiguration("start_sensor_fusion")
    shooter_dry_run = LaunchConfiguration("shooter_dry_run")
    auto_start = LaunchConfiguration("auto_start")
    team_color = LaunchConfiguration("team_color")
    target_file = LaunchConfiguration("target_file")

    description_launch = PathJoinSubstitution([
        FindPackageShare("wvb_description"),
        "launch",
        "description.launch.py",
    ])
    navigation_launch = PathJoinSubstitution([
        FindPackageShare("wvb_navigation"),
        "launch",
        "navigation.launch.py",
    ])

    behavior_params = PathJoinSubstitution([
        FindPackageShare("wvb_behavior"),
        "config",
        "behavior.yaml",
    ])
    shooter_params = PathJoinSubstitution([
        FindPackageShare("wvb_shooter"),
        "config",
        "shooter.yaml",
    ])
    vision_params = PathJoinSubstitution([
        FindPackageShare("wvb_vision"),
        "config",
        "vision.yaml",
    ])
    sensor_fusion_params = PathJoinSubstitution([
        FindPackageShare("wvb_bringup"),
        "config",
        "sensor_fusion.yaml",
    ])
    default_targets = PathJoinSubstitution([
        FindPackageShare("wvb_navigation"),
        "config",
        "targets.elimination.yellow.yaml",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("start_navigation", default_value="true"),
        DeclareLaunchArgument("start_sensor_fusion", default_value="true"),
        DeclareLaunchArgument("shooter_dry_run", default_value="false"),
        DeclareLaunchArgument("auto_start", default_value="true"),
        DeclareLaunchArgument("team_color", default_value="yellow"),
        DeclareLaunchArgument("target_file", default_value=default_targets),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(description_launch),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(navigation_launch),
            condition=IfCondition(start_navigation),
            launch_arguments={
                "use_sim_time": use_sim_time,
            }.items(),
        ),

        Node(
            package="robot_localization",
            executable="ekf_node",
            name="ekf_filter_node",
            output="screen",
            condition=IfCondition(start_sensor_fusion),
            parameters=[
                sensor_fusion_params,
                {"use_sim_time": ParameterValue(use_sim_time, value_type=bool)},
            ],
        ),

        Node(
            package="wvb_shooter",
            executable="shooter_controller",
            name="shooter_controller",
            output="screen",
            parameters=[
                shooter_params,
                {"dry_run": ParameterValue(shooter_dry_run, value_type=bool)},
            ],
        ),

        Node(
            package="wvb_vision",
            executable="apriltag_detector",
            name="apriltag_detector",
            output="screen",
            parameters=[vision_params],
        ),

        Node(
            package="wvb_behavior",
            executable="competition_behavior",
            name="competition_behavior",
            output="screen",
            parameters=[
                behavior_params,
                target_file,
                {"auto_start": ParameterValue(auto_start, value_type=bool)},
                {"team_color": team_color},
            ],
        ),
    ])
