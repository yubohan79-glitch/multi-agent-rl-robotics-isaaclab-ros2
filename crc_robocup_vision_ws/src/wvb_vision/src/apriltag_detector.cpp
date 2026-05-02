#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#if __has_include("cv_bridge/cv_bridge.hpp")
#include "cv_bridge/cv_bridge.hpp"
#else
#include "cv_bridge/cv_bridge.h"
#endif
#include "geometry_msgs/msg/quaternion.hpp"
#include "opencv2/aruco.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "wvb_interfaces/msg/target_detection.hpp"

class AprilTagDetector : public rclcpp::Node
{
public:
  AprilTagDetector()
  : Node("apriltag_detector")
  {
    image_topic_ = declare_parameter<std::string>("image_topic", "/camera/image_raw");
    camera_info_topic_ = declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");
    detection_topic_ = declare_parameter<std::string>("detection_topic", "/target_detection");
    tag_size_m_ = declare_parameter<double>("tag_size_m", 0.05);
    normal_tag_id_ = declare_parameter<int>("normal_tag_id", 1);
    yellow_base_tag_id_ = declare_parameter<int>("yellow_base_tag_id", 2);
    blue_base_tag_id_ = declare_parameter<int>("blue_base_tag_id", 3);
    center_tolerance_ = declare_parameter<double>("target_center_tolerance", 0.08);
    target_distance_m_ = declare_parameter<double>("target_distance_m", 0.52);
    distance_tolerance_m_ = declare_parameter<double>("target_distance_tolerance_m", 0.06);

    dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);

    detection_pub_ = create_publisher<wvb_interfaces::msg::TargetDetection>(detection_topic_, 10);
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        latest_camera_info_ = msg;
      });

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_, rclcpp::SensorDataQoS(),
      std::bind(&AprilTagDetector::image_callback, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(), "AprilTag36h11 detector subscribed to %s and publishing %s",
      image_topic_.c_str(), detection_topic_.c_str());
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv_bridge::CvImagePtr cv_image;
    try {
      cv_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception & error) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "cv_bridge conversion failed: %s", error.what());
      return;
    }

    cv::Mat gray;
    cv::cvtColor(cv_image->image, gray, cv::COLOR_BGR2GRAY);

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    cv::aruco::detectMarkers(gray, dictionary_, corners, ids);

    if (ids.empty()) {
      return;
    }

    const int selected = select_detection(ids, corners);
    if (selected < 0) {
      return;
    }

    auto detection = wvb_interfaces::msg::TargetDetection();
    detection.header = msg->header;
    detection.tag_id = ids[selected];
    detection.target_type = classify_tag(ids[selected]);
    detection.pose.header = msg->header;
    detection.center_x = normalized_center_x(corners[selected], static_cast<double>(gray.cols));
    detection.distance_z = std::numeric_limits<double>::quiet_NaN();
    detection.confidence = marker_area(corners[selected]) / static_cast<double>(gray.cols * gray.rows);

    fill_pose_if_possible(corners[selected], detection);
    detection.aligned =
      std::fabs(detection.center_x) <= center_tolerance_ &&
      (std::isnan(detection.distance_z) ||
       std::fabs(detection.distance_z - target_distance_m_) <= distance_tolerance_m_);

    detection_pub_->publish(detection);
  }

  int select_detection(
    const std::vector<int> & ids,
    const std::vector<std::vector<cv::Point2f>> & corners) const
  {
    int best_index = -1;
    double best_score = -1.0;

    for (size_t i = 0; i < ids.size(); ++i) {
      const bool known_target =
        ids[i] == normal_tag_id_ || ids[i] == yellow_base_tag_id_ || ids[i] == blue_base_tag_id_;
      const double score = marker_area(corners[i]) + (known_target ? 1000000.0 : 0.0);
      if (score > best_score) {
        best_score = score;
        best_index = static_cast<int>(i);
      }
    }

    return best_index;
  }

  std::string classify_tag(const int id) const
  {
    if (id == normal_tag_id_) {
      return "normal";
    }
    if (id == yellow_base_tag_id_) {
      return "yellow_base";
    }
    if (id == blue_base_tag_id_) {
      return "blue_base";
    }
    return "unknown";
  }

  static double marker_area(const std::vector<cv::Point2f> & marker_corners)
  {
    return std::fabs(cv::contourArea(marker_corners));
  }

  static double normalized_center_x(const std::vector<cv::Point2f> & marker_corners, const double image_width)
  {
    double center = 0.0;
    for (const auto & point : marker_corners) {
      center += point.x;
    }
    center /= static_cast<double>(marker_corners.size());
    return (center - image_width * 0.5) / (image_width * 0.5);
  }

  void fill_pose_if_possible(
    const std::vector<cv::Point2f> & marker_corners,
    wvb_interfaces::msg::TargetDetection & detection) const
  {
    if (!latest_camera_info_ || tag_size_m_ <= 0.0) {
      detection.pose.pose.orientation.w = 1.0;
      return;
    }

    cv::Mat camera_matrix(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        camera_matrix.at<double>(row, col) = latest_camera_info_->k[row * 3 + col];
      }
    }

    cv::Mat distortion(latest_camera_info_->d, true);
    std::vector<std::vector<cv::Point2f>> single_marker {marker_corners};
    std::vector<cv::Vec3d> rvecs;
    std::vector<cv::Vec3d> tvecs;
    cv::aruco::estimatePoseSingleMarkers(single_marker, tag_size_m_, camera_matrix, distortion, rvecs, tvecs);

    if (tvecs.empty() || rvecs.empty()) {
      detection.pose.pose.orientation.w = 1.0;
      return;
    }

    detection.pose.pose.position.x = tvecs.front()[0];
    detection.pose.pose.position.y = tvecs.front()[1];
    detection.pose.pose.position.z = tvecs.front()[2];
    detection.distance_z = tvecs.front()[2];

    cv::Mat rotation;
    cv::Rodrigues(rvecs.front(), rotation);
    tf2::Matrix3x3 basis(
      rotation.at<double>(0, 0), rotation.at<double>(0, 1), rotation.at<double>(0, 2),
      rotation.at<double>(1, 0), rotation.at<double>(1, 1), rotation.at<double>(1, 2),
      rotation.at<double>(2, 0), rotation.at<double>(2, 1), rotation.at<double>(2, 2));
    tf2::Quaternion quaternion;
    basis.getRotation(quaternion);
    quaternion.normalize();

    detection.pose.pose.orientation.x = quaternion.x();
    detection.pose.pose.orientation.y = quaternion.y();
    detection.pose.pose.orientation.z = quaternion.z();
    detection.pose.pose.orientation.w = quaternion.w();
  }

  std::string image_topic_;
  std::string camera_info_topic_;
  std::string detection_topic_;
  double tag_size_m_ {0.05};
  int normal_tag_id_ {1};
  int yellow_base_tag_id_ {2};
  int blue_base_tag_id_ {3};
  double center_tolerance_ {0.08};
  double target_distance_m_ {0.52};
  double distance_tolerance_m_ {0.06};

  cv::Ptr<cv::aruco::Dictionary> dictionary_;
  sensor_msgs::msg::CameraInfo::SharedPtr latest_camera_info_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Publisher<wvb_interfaces::msg::TargetDetection>::SharedPtr detection_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AprilTagDetector>());
  rclcpp::shutdown();
  return 0;
}
