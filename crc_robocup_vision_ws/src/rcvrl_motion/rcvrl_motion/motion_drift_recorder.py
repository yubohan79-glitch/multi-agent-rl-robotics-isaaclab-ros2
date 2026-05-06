from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class MotionDriftRecorder(Node):
    """Record motion commands and fused sensor residuals for drift calibration.

    The source ROS1 Mini stack publishes `/cmd_vel`, wheel odometry, IMU and
    RPLidar scan data. This ROS2 node keeps the same contract and writes a CSV
    that can be used to fit the acceleration-to-localization-drift model used
    by the IsaacLab/RL environment.
    """

    def __init__(self) -> None:
        super().__init__("motion_drift_recorder")
        self.output_csv = Path(self.declare_parameter("output_csv", "motion_drift_log.csv").value)
        self.cmd_vel_topic = str(self.declare_parameter("cmd_vel_topic", "/cmd_vel").value)
        self.wheel_odom_topic = str(self.declare_parameter("wheel_odom_topic", "/wheel/odom").value)
        self.filtered_odom_topic = str(self.declare_parameter("filtered_odom_topic", "/odometry/filtered").value)
        self.imu_topic = str(self.declare_parameter("imu_topic", "/imu/data_raw").value)
        self.scan_topic = str(self.declare_parameter("scan_topic", "/scan").value)
        self.sample_period_s = float(self.declare_parameter("sample_period_s", 0.10).value)
        self.linear_accel_warn = float(self.declare_parameter("linear_accel_warn", 1.05).value)
        self.angular_accel_warn = float(self.declare_parameter("angular_accel_warn", 4.20).value)
        self.scan_front_window_rad = float(self.declare_parameter("scan_front_window_rad", 0.52).value)

        self.latest_cmd: Optional[Twist] = None
        self.latest_wheel: Optional[Odometry] = None
        self.latest_filtered: Optional[Odometry] = None
        self.latest_imu: Optional[Imu] = None
        self.latest_scan: Optional[LaserScan] = None
        self.prev_cmd: Optional[tuple[float, float, float]] = None

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.csv_handle = self.output_csv.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.csv_handle,
            fieldnames=[
                "time_s",
                "cmd_linear_x",
                "cmd_angular_z",
                "linear_accel",
                "angular_accel",
                "odom_xy_error_m",
                "odom_yaw_error_rad",
                "imu_yaw_rate",
                "imu_cmd_yaw_residual",
                "front_scan_min_m",
                "drift_risk",
            ],
        )
        self.writer.writeheader()

        self.create_subscription(Twist, self.cmd_vel_topic, self._on_cmd, 10)
        self.create_subscription(Odometry, self.wheel_odom_topic, self._on_wheel, 10)
        self.create_subscription(Odometry, self.filtered_odom_topic, self._on_filtered, 10)
        self.create_subscription(Imu, self.imu_topic, self._on_imu, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self.scan_topic, self._on_scan, qos_profile_sensor_data)
        self.timer = self.create_timer(self.sample_period_s, self._sample)
        self.get_logger().info(f"Recording motion drift CSV to {self.output_csv}")

    def destroy_node(self) -> bool:
        self.csv_handle.flush()
        self.csv_handle.close()
        return super().destroy_node()

    def _on_cmd(self, msg: Twist) -> None:
        self.latest_cmd = msg

    def _on_wheel(self, msg: Odometry) -> None:
        self.latest_wheel = msg

    def _on_filtered(self, msg: Odometry) -> None:
        self.latest_filtered = msg

    def _on_imu(self, msg: Imu) -> None:
        self.latest_imu = msg

    def _on_scan(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def _sample(self) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        cmd_linear = float(self.latest_cmd.linear.x) if self.latest_cmd is not None else 0.0
        cmd_angular = float(self.latest_cmd.angular.z) if self.latest_cmd is not None else 0.0
        linear_accel = 0.0
        angular_accel = 0.0
        if self.prev_cmd is not None:
            prev_t, prev_linear, prev_angular = self.prev_cmd
            dt = max(1e-6, now_s - prev_t)
            linear_accel = abs(cmd_linear - prev_linear) / dt
            angular_accel = abs(cmd_angular - prev_angular) / dt
        self.prev_cmd = (now_s, cmd_linear, cmd_angular)

        odom_xy_error, odom_yaw_error = self._odom_residual()
        imu_yaw_rate = float(self.latest_imu.angular_velocity.z) if self.latest_imu is not None else math.nan
        imu_cmd_residual = abs(imu_yaw_rate - cmd_angular) if self.latest_imu is not None else math.nan
        front_scan_min = self._front_scan_min()
        drift_risk = self._drift_risk(linear_accel, angular_accel, odom_xy_error, odom_yaw_error, imu_cmd_residual)

        self.writer.writerow(
            {
                "time_s": round(now_s, 4),
                "cmd_linear_x": round(cmd_linear, 5),
                "cmd_angular_z": round(cmd_angular, 5),
                "linear_accel": round(linear_accel, 5),
                "angular_accel": round(angular_accel, 5),
                "odom_xy_error_m": round(odom_xy_error, 5) if math.isfinite(odom_xy_error) else "",
                "odom_yaw_error_rad": round(odom_yaw_error, 5) if math.isfinite(odom_yaw_error) else "",
                "imu_yaw_rate": round(imu_yaw_rate, 5) if math.isfinite(imu_yaw_rate) else "",
                "imu_cmd_yaw_residual": round(imu_cmd_residual, 5) if math.isfinite(imu_cmd_residual) else "",
                "front_scan_min_m": round(front_scan_min, 5) if math.isfinite(front_scan_min) else "",
                "drift_risk": round(drift_risk, 5),
            }
        )
        self.csv_handle.flush()

    def _odom_residual(self) -> tuple[float, float]:
        if self.latest_wheel is None or self.latest_filtered is None:
            return math.nan, math.nan
        wp = self.latest_wheel.pose.pose.position
        fp = self.latest_filtered.pose.pose.position
        xy_error = math.hypot(float(wp.x - fp.x), float(wp.y - fp.y))
        wq = self.latest_wheel.pose.pose.orientation
        fq = self.latest_filtered.pose.pose.orientation
        wyaw = yaw_from_quaternion(wq.x, wq.y, wq.z, wq.w)
        fyaw = yaw_from_quaternion(fq.x, fq.y, fq.z, fq.w)
        return xy_error, abs(wrap_angle(wyaw - fyaw))

    def _front_scan_min(self) -> float:
        if self.latest_scan is None:
            return math.nan
        scan = self.latest_scan
        best = math.inf
        angle = float(scan.angle_min)
        for value in scan.ranges:
            if abs(angle) <= self.scan_front_window_rad and math.isfinite(value):
                best = min(best, float(value))
            angle += float(scan.angle_increment)
        return best

    def _drift_risk(
        self,
        linear_accel: float,
        angular_accel: float,
        odom_xy_error: float,
        odom_yaw_error: float,
        imu_cmd_residual: float,
    ) -> float:
        risk = 0.0
        risk += max(0.0, linear_accel - self.linear_accel_warn) / max(self.linear_accel_warn, 1e-6) * 0.35
        risk += max(0.0, angular_accel - self.angular_accel_warn) / max(self.angular_accel_warn, 1e-6) * 0.25
        if math.isfinite(odom_xy_error):
            risk += min(1.0, odom_xy_error / 0.10) * 0.20
        if math.isfinite(odom_yaw_error):
            risk += min(1.0, odom_yaw_error / 0.25) * 0.12
        if math.isfinite(imu_cmd_residual):
            risk += min(1.0, imu_cmd_residual / 1.20) * 0.08
        return max(0.0, min(1.0, risk))


def main() -> None:
    rclpy.init()
    node = MotionDriftRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
