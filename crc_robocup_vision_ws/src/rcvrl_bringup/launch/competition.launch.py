from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def behavior_node_for_team(context):
    auto_start = LaunchConfiguration("auto_start")
    team_color = LaunchConfiguration("team_color")
    target_file = LaunchConfiguration("target_file").perform(context)

    behavior_params = PathJoinSubstitution([
        FindPackageShare("rcvrl_behavior"),
        "config",
        "behavior.yaml",
    ])
    if target_file == "auto":
        target_name = (
            "targets.elimination.blue.yaml"
            if team_color.perform(context).lower() == "blue"
            else "targets.elimination.yellow.yaml"
        )
        target_file = PathJoinSubstitution([
            FindPackageShare("rcvrl_navigation"),
            "config",
            target_name,
        ])

    return [
        Node(
            package="rcvrl_behavior",
            executable="competition_behavior",
            name="competition_behavior",
            output="screen",
            parameters=[
                behavior_params,
                target_file,
                {"auto_start": ParameterValue(auto_start, value_type=bool)},
                {"team_color": team_color},
            ],
        )
    ]


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    start_navigation = LaunchConfiguration("start_navigation")
    start_sensor_fusion = LaunchConfiguration("start_sensor_fusion")
    start_motion_drift_recorder = LaunchConfiguration("start_motion_drift_recorder")
    shooter_dry_run = LaunchConfiguration("shooter_dry_run")
    motion_drift_output = LaunchConfiguration("motion_drift_output")

    description_launch = PathJoinSubstitution([
        FindPackageShare("rcvrl_description"),
        "launch",
        "description.launch.py",
    ])
    navigation_launch = PathJoinSubstitution([
        FindPackageShare("rcvrl_navigation"),
        "launch",
        "navigation.launch.py",
    ])

    shooter_params = PathJoinSubstitution([
        FindPackageShare("rcvrl_shooter"),
        "config",
        "shooter.yaml",
    ])
    vision_params = PathJoinSubstitution([
        FindPackageShare("rcvrl_vision"),
        "config",
        "vision.yaml",
    ])
    sensor_fusion_params = PathJoinSubstitution([
        FindPackageShare("rcvrl_bringup"),
        "config",
        "sensor_fusion.yaml",
    ])
    motion_drift_params = PathJoinSubstitution([
        FindPackageShare("rcvrl_motion"),
        "config",
        "motion_drift.yaml",
    ])
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("start_navigation", default_value="true"),
        DeclareLaunchArgument("start_sensor_fusion", default_value="true"),
        DeclareLaunchArgument("start_motion_drift_recorder", default_value="false"),
        DeclareLaunchArgument("shooter_dry_run", default_value="false"),
        DeclareLaunchArgument("motion_drift_output", default_value="motion_drift_log.csv"),
        DeclareLaunchArgument("auto_start", default_value="true"),
        DeclareLaunchArgument("team_color", default_value="yellow"),
        DeclareLaunchArgument("target_file", default_value="auto"),

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
            package="rcvrl_motion",
            executable="motion_drift_recorder",
            name="motion_drift_recorder",
            output="screen",
            condition=IfCondition(start_motion_drift_recorder),
            parameters=[
                motion_drift_params,
                {
                    "output_csv": motion_drift_output,
                    "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
                },
            ],
        ),

        Node(
            package="rcvrl_shooter",
            executable="shooter_controller",
            name="shooter_controller",
            output="screen",
            parameters=[
                shooter_params,
                {"dry_run": ParameterValue(shooter_dry_run, value_type=bool)},
            ],
        ),

        Node(
            package="rcvrl_vision",
            executable="apriltag_detector",
            name="apriltag_detector",
            output="screen",
            parameters=[vision_params],
        ),

        OpaqueFunction(function=behavior_node_for_team),
    ])
