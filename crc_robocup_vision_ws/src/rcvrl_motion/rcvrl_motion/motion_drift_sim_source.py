from __future__ import annotations

import math
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


class MotionDriftSimSource(Node):
    """Publish repeatable ROS2 motion/sensor topics for drift experiments.

    This is a lightweight simulator for the Mini robot motion contracts found
    in the original ROS1 workspaces. It is not a replacement for Gazebo or
    IsaacLab physics; its job is to make the ROS2 recorder produce real CSV
    samples when no hardware/simulator topics are already live.
    """

    def __init__(self) -> None:
        super().__init__("motion_drift_sim_source")
        self.cmd_vel_topic = str(self.declare_parameter("cmd_vel_topic", "/cmd_vel").value)
        self.wheel_odom_topic = str(self.declare_parameter("wheel_odom_topic", "/wheel/odom").value)
        self.filtered_odom_topic = str(self.declare_parameter("filtered_odom_topic", "/odometry/filtered").value)
        self.imu_topic = str(self.declare_parameter("imu_topic", "/imu/data_raw").value)
        self.scan_topic = str(self.declare_parameter("scan_topic", "/scan").value)
        self.frame_id = str(self.declare_parameter("frame_id", "odom").value)
        self.base_frame_id = str(self.declare_parameter("base_frame_id", "base_link").value)
        self.rate_hz = float(self.declare_parameter("rate_hz", 30.0).value)
        self.duration_s = float(self.declare_parameter("duration_s", 42.0).value)
        self.linear_accel_warn = float(self.declare_parameter("linear_accel_warn", 1.05).value)
        self.angular_accel_warn = float(self.declare_parameter("angular_accel_warn", 4.20).value)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.wheel_pub = self.create_publisher(Odometry, self.wheel_odom_topic, 10)
        self.filtered_pub = self.create_publisher(Odometry, self.filtered_odom_topic, 10)
        self.imu_pub = self.create_publisher(Imu, self.imu_topic, 10)
        self.scan_pub = self.create_publisher(LaserScan, self.scan_topic, 10)

        self.true_pose = Pose2D()
        self.wheel_pose = Pose2D()
        self.filtered_pose = Pose2D()
        self.previous_linear = 0.0
        self.previous_angular = 0.0
        self.previous_time_s = self.get_clock().now().nanoseconds * 1e-9
        self.started_time_s = self.previous_time_s
        self.timer = self.create_timer(1.0 / max(self.rate_hz, 1.0), self._tick)
        self.get_logger().info(
            "Publishing simulated drift topics for %.1fs on %s, %s, %s, %s, %s"
            % (
                self.duration_s,
                self.cmd_vel_topic,
                self.wheel_odom_topic,
                self.filtered_odom_topic,
                self.imu_topic,
                self.scan_topic,
            )
        )

    def _tick(self) -> None:
        now = self.get_clock().now()
        now_s = now.nanoseconds * 1e-9
        elapsed = now_s - self.started_time_s
        dt = max(1e-4, min(0.10, now_s - self.previous_time_s))
        self.previous_time_s = now_s

        linear, angular = self._profile(elapsed)
        linear_accel = (linear - self.previous_linear) / dt
        angular_accel = (angular - self.previous_angular) / dt
        accel_risk = self._drift_risk(abs(linear_accel), abs(angular_accel))

        self._integrate(self.true_pose, linear, angular, dt)
        wheel_linear = linear * (1.0 + 0.018 + 0.055 * accel_risk)
        wheel_angular = angular * (1.0 - 0.012 + 0.035 * accel_risk)
        self._integrate(self.wheel_pose, wheel_linear, wheel_angular, dt)
        self.filtered_pose.x = 0.92 * self.filtered_pose.x + 0.08 * self.true_pose.x
        self.filtered_pose.y = 0.92 * self.filtered_pose.y + 0.08 * self.true_pose.y
        self.filtered_pose.yaw = wrap_angle(0.92 * self.filtered_pose.yaw + 0.08 * self.true_pose.yaw)

        self.cmd_pub.publish(self._twist(linear, angular))
        self.wheel_pub.publish(self._odom(now, self.wheel_pose, linear, angular, 0.035 + 0.08 * accel_risk))
        self.filtered_pub.publish(self._odom(now, self.filtered_pose, linear, angular, 0.010 + 0.035 * accel_risk))
        self.imu_pub.publish(self._imu(now, linear_accel, angular, accel_risk))
        self.scan_pub.publish(self._scan(now, elapsed, accel_risk))

        self.previous_linear = linear
        self.previous_angular = angular
        if elapsed >= self.duration_s:
            self.cmd_pub.publish(self._twist(0.0, 0.0))
            self.get_logger().info("Synthetic motion drift run complete.")
            raise SystemExit

    def _profile(self, elapsed_s: float) -> tuple[float, float]:
        phase = elapsed_s % 21.0
        if phase < 3.0:
            return 0.05 * phase, 0.0
        if phase < 7.0:
            return 0.16, 0.0
        if phase < 9.0:
            return 0.16 + 0.15 * (phase - 7.0), 0.0
        if phase < 12.0:
            return 0.44, 0.70
        if phase < 14.0:
            return 0.38, -1.15
        if phase < 16.0:
            return max(0.0, 0.38 - 0.20 * (phase - 14.0)), 0.0
        if phase < 18.0:
            return -0.12, -0.45
        return 0.0, 0.0

    def _drift_risk(self, linear_accel: float, angular_accel: float) -> float:
        risk = 0.0
        risk += max(0.0, linear_accel - self.linear_accel_warn) / max(self.linear_accel_warn, 1e-6) * 0.58
        risk += max(0.0, angular_accel - self.angular_accel_warn) / max(self.angular_accel_warn, 1e-6) * 0.42
        return max(0.0, min(1.0, risk))

    @staticmethod
    def _integrate(pose: Pose2D, linear: float, angular: float, dt: float) -> None:
        mid_yaw = wrap_angle(pose.yaw + angular * dt * 0.5)
        pose.x += linear * math.cos(mid_yaw) * dt
        pose.y += linear * math.sin(mid_yaw) * dt
        pose.yaw = wrap_angle(pose.yaw + angular * dt)

    @staticmethod
    def _twist(linear: float, angular: float) -> Twist:
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        return msg

    def _odom(self, stamp, pose: Pose2D, linear: float, angular: float, covariance: float) -> Odometry:
        msg = Odometry()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.base_frame_id
        msg.pose.pose.position.x = pose.x
        msg.pose.pose.position.y = pose.y
        qx, qy, qz, qw = quaternion_from_yaw(pose.yaw)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.pose.covariance[0] = covariance
        msg.pose.covariance[7] = covariance
        msg.pose.covariance[35] = covariance * 1.6
        msg.twist.twist.linear.x = linear
        msg.twist.twist.angular.z = angular
        return msg

    def _imu(self, stamp, linear_accel: float, angular: float, accel_risk: float) -> Imu:
        msg = Imu()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = "imu_link"
        msg.linear_acceleration.x = linear_accel
        msg.linear_acceleration.y = 0.05 * math.sin(stamp.nanoseconds * 1e-9 * 3.7)
        msg.linear_acceleration.z = 9.80665
        msg.angular_velocity.z = angular + 0.18 * accel_risk
        msg.orientation.w = 1.0
        return msg

    def _scan(self, stamp, elapsed_s: float, accel_risk: float) -> LaserScan:
        msg = LaserScan()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = "laser_link"
        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        msg.angle_increment = math.radians(2.0)
        msg.range_min = 0.05
        msg.range_max = 6.0
        count = int(round((msg.angle_max - msg.angle_min) / msg.angle_increment)) + 1
        front_clearance = max(0.18, 0.72 - 0.34 * max(0.0, math.sin(elapsed_s * 0.55)) - 0.08 * accel_risk)
        ranges = []
        angle = msg.angle_min
        for _ in range(count):
            if abs(angle) < 0.52:
                value = front_clearance + 0.025 * abs(angle)
            else:
                value = 1.75 + 0.18 * math.sin(angle * 3.0 + elapsed_s * 0.2)
            ranges.append(float(max(msg.range_min, min(msg.range_max, value))))
            angle += msg.angle_increment
        msg.ranges = ranges
        return msg


def main() -> None:
    rclpy.init()
    node = MotionDriftSimSource()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

