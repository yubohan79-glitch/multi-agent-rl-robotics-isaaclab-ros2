#include <algorithm>
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "rcvrl_interfaces/msg/target_detection.hpp"

using namespace std::chrono_literals;

namespace
{
struct TargetPose
{
  double x {0.0};
  double y {0.0};
  double yaw {0.0};
  int tag_id {1};
  std::string owner {"unknown"};
  std::string name {"target"};
};

enum class State
{
  INIT,
  NAVIGATE,
  SEARCH,
  ALIGN,
  FIRE,
  NEXT_TARGET,
  RECOVER_LOCALIZATION,
  RETURN_HOME,
  END,
  FAILSAFE
};

std::string state_name(const State state)
{
  switch (state) {
    case State::INIT:
      return "INIT";
    case State::NAVIGATE:
      return "NAVIGATE";
    case State::SEARCH:
      return "SEARCH";
    case State::ALIGN:
      return "ALIGN";
    case State::FIRE:
      return "FIRE";
    case State::NEXT_TARGET:
      return "NEXT_TARGET";
    case State::RECOVER_LOCALIZATION:
      return "RECOVER_LOCALIZATION";
    case State::RETURN_HOME:
      return "RETURN_HOME";
    case State::END:
      return "END";
    case State::FAILSAFE:
      return "FAILSAFE";
  }
  return "UNKNOWN";
}
}  // namespace

class CompetitionBehavior : public rclcpp::Node
{
public:
  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using GoalHandleNavigate = rclcpp_action::ClientGoalHandle<NavigateToPose>;

  CompetitionBehavior()
  : Node("competition_behavior")
  {
    load_parameters();

    cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);
    filtered_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      filtered_odom_topic_, 10,
      [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
        const double xy_cov = std::max(0.0, msg->pose.covariance[0]) +
          std::max(0.0, msg->pose.covariance[7]);
        const double yaw_cov = std::max(0.0, msg->pose.covariance[35]);
        const double xy_score = 1.0 - std::clamp(xy_cov / filtered_odom_xy_cov_warn_, 0.0, 1.0);
        const double yaw_score = 1.0 - std::clamp(yaw_cov / filtered_odom_yaw_cov_warn_, 0.0, 1.0);
        filtered_odom_confidence_ = std::clamp(0.65 * xy_score + 0.35 * yaw_score, 0.0, 1.0);
        last_filtered_odom_time_ = now();
        if (filtered_odom_confidence_ < filtered_odom_min_confidence_) {
          request_localization_recovery("ekf covariance confidence low");
        }
      });
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_, rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
        const double lateral_accel =
          std::hypot(msg->linear_acceleration.x, msg->linear_acceleration.y);
        const double yaw_rate = std::fabs(msg->angular_velocity.z);
        if (lateral_accel >= collision_accel_threshold_ || yaw_rate >= collision_yaw_rate_threshold_) {
          handle_contact_impulse("imu collision impulse");
        }
      });
    bumper_left_sub_ = create_subscription<std_msgs::msg::Bool>(
      bumper_left_topic_, 10,
      [this](const std_msgs::msg::Bool::SharedPtr msg) {
        if (msg->data) {
          handle_contact_impulse("left bumper contact");
        }
      });
    bumper_right_sub_ = create_subscription<std_msgs::msg::Bool>(
      bumper_right_topic_, 10,
      [this](const std_msgs::msg::Bool::SharedPtr msg) {
        if (msg->data) {
          handle_contact_impulse("right bumper contact");
        }
      });
    detection_sub_ = create_subscription<rcvrl_interfaces::msg::TargetDetection>(
      target_detection_topic_, 10,
      [this](const rcvrl_interfaces::msg::TargetDetection::SharedPtr msg) {
        last_detection_ = *msg;
        last_detection_time_ = now();
        has_detection_ = true;
      });

    nav_client_ = rclcpp_action::create_client<NavigateToPose>(this, navigate_action_name_);
    enable_client_ = create_client<std_srvs::srv::Trigger>("/shooter/enable");
    fire_client_ = create_client<std_srvs::srv::Trigger>("/shooter/fire");
    disable_client_ = create_client<std_srvs::srv::Trigger>("/shooter/disable");

    match_start_time_ = now();
    state_enter_time_ = match_start_time_;
    timer_ = create_wall_timer(100ms, std::bind(&CompetitionBehavior::tick, this));

    RCLCPP_INFO(get_logger(), "Competition behavior ready with %zu configured targets", targets_.size());
  }

private:
  void load_parameters()
  {
    auto_start_ = declare_parameter<bool>("auto_start", true);
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    cmd_vel_topic_ = declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    target_detection_topic_ = declare_parameter<std::string>("target_detection_topic", "/target_detection");
    navigate_action_name_ = declare_parameter<std::string>("navigate_action_name", "navigate_to_pose");
    match_timeout_s_ = declare_parameter<double>("match_timeout_s", 180.0);
    stationary_timeout_s_ = declare_parameter<double>("stationary_timeout_s", 20.0);
    search_timeout_s_ = declare_parameter<double>("search_timeout_s", 6.0);
    alignment_timeout_s_ = declare_parameter<double>("alignment_timeout_s", 5.0);
    nav_retry_limit_ = declare_parameter<int>("nav_retry_limit", 2);
    search_angular_speed_ = declare_parameter<double>("search_angular_speed", 0.35);
    kp_angular_ = declare_parameter<double>("kp_angular", 0.9);
    kp_linear_ = declare_parameter<double>("kp_linear", 0.45);
    max_angular_speed_ = declare_parameter<double>("max_angular_speed", 0.5);
    max_linear_speed_ = declare_parameter<double>("max_linear_speed", 0.18);
    target_center_tolerance_ = declare_parameter<double>("target_center_tolerance", 0.08);
    target_distance_m_ = declare_parameter<double>("target_distance_m", 0.52);
    target_distance_tolerance_m_ = declare_parameter<double>("target_distance_tolerance_m", 0.06);
    laser_dwell_required_s_ = declare_parameter<double>("laser_dwell_required_s", 0.80);
    localization_recovery_spin_s_ = declare_parameter<double>("localization_recovery_spin_s", 5.8);
    localization_recovery_angular_speed_ = declare_parameter<double>("localization_recovery_angular_speed", 0.55);
    localization_recovery_limit_ = declare_parameter<int>("localization_recovery_limit", 2);
    collision_recovery_cooldown_s_ = declare_parameter<double>("collision_recovery_cooldown_s", 2.0);
    collision_accel_threshold_ = declare_parameter<double>("collision_accel_threshold", 4.0);
    collision_yaw_rate_threshold_ = declare_parameter<double>("collision_yaw_rate_threshold", 2.4);
    filtered_odom_topic_ = declare_parameter<std::string>("filtered_odom_topic", "/odometry/filtered");
    filtered_odom_min_confidence_ = declare_parameter<double>("filtered_odom_min_confidence", 0.35);
    filtered_odom_xy_cov_warn_ = declare_parameter<double>("filtered_odom_xy_cov_warn", 0.18);
    filtered_odom_yaw_cov_warn_ = declare_parameter<double>("filtered_odom_yaw_cov_warn", 0.16);
    imu_topic_ = declare_parameter<std::string>("imu_topic", "/imu/data_raw");
    bumper_left_topic_ = declare_parameter<std::string>("bumper_left_topic", "/bumper/front_left");
    bumper_right_topic_ = declare_parameter<std::string>("bumper_right_topic", "/bumper/front_right");
    team_color_ = declare_parameter<std::string>("team_color", "yellow");
    enforce_opponent_targets_ = declare_parameter<bool>("enforce_opponent_targets", true);
    home_.x = declare_parameter<double>("home_x", 0.0);
    home_.y = declare_parameter<double>("home_y", 0.0);
    home_.yaw = declare_parameter<double>("home_yaw", 0.0);

    const auto xs = declare_parameter<std::vector<double>>(
      "target_x", {1.92, 0.30, 2.70, 2.68, 0.23});
    const auto ys = declare_parameter<std::vector<double>>(
      "target_y", {2.52, 1.84, 1.84, 2.68, 2.31});
    const auto yaws = declare_parameter<std::vector<double>>(
      "target_yaw", {2.36, -2.36, -0.79, 0.79, 1.73});
    const auto tag_ids = declare_parameter<std::vector<int64_t>>(
      "target_tag_id", {1, 1, 1, 1, 3});
    const auto owners = declare_parameter<std::vector<std::string>>(
      "target_owner", std::vector<std::string>{"blue", "blue", "blue", "blue", "blue"});
    const auto names = declare_parameter<std::vector<std::string>>(
      "target_name",
      std::vector<std::string>{"T01_NorthMiddle", "T03_WestAboveGate", "T05_EastAboveGate", "T02_NorthEast", "BlueBaseTarget"});

    const size_t count = std::min({xs.size(), ys.size(), yaws.size(), tag_ids.size()});
    targets_.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      TargetPose target;
      target.x = xs[i];
      target.y = ys[i];
      target.yaw = yaws[i];
      target.tag_id = static_cast<int>(tag_ids[i]);
      target.owner = i < owners.size() ? owners[i] : infer_owner_from_tag(target.tag_id);
      target.name = i < names.size() ? names[i] : "target_" + std::to_string(i + 1);

      if (enforce_opponent_targets_ && target.owner == team_color_) {
        RCLCPP_WARN(
          get_logger(), "Skipping own target in route: %s owner=%s",
          target.name.c_str(), target.owner.c_str());
        continue;
      }
      targets_.push_back(target);
    }

    if (xs.size() != ys.size() || xs.size() != yaws.size() || xs.size() != tag_ids.size()) {
      RCLCPP_WARN(get_logger(), "Target parameter arrays have different sizes; using first %zu entries", count);
    }
    if (!owners.empty() && owners.size() != count) {
      RCLCPP_WARN(get_logger(), "target_owner size does not match route size; missing entries are inferred");
    }
  }

  void tick()
  {
    if (!auto_start_ || state_ == State::END) {
      return;
    }

    if (localization_recovery_requested_ && state_ != State::RECOVER_LOCALIZATION &&
      state_ != State::INIT && state_ != State::RETURN_HOME && state_ != State::END && state_ != State::FAILSAFE)
    {
      localization_recovery_requested_ = false;
      enter_state(State::RECOVER_LOCALIZATION);
      return;
    }

    if (elapsed_since(match_start_time_) > match_timeout_s_ && state_ != State::RETURN_HOME) {
      RCLCPP_WARN(get_logger(), "Match timeout reached; returning home");
      enter_state(State::RETURN_HOME);
    }

    if (elapsed_since(state_enter_time_) > stationary_timeout_s_ &&
      (state_ == State::SEARCH || state_ == State::ALIGN))
    {
      RCLCPP_WARN(get_logger(), "No progress timeout in %s; spinning to rebuild localization", state_name(state_).c_str());
      enter_state(State::RECOVER_LOCALIZATION);
    }

    switch (state_) {
      case State::INIT:
        if (!call_trigger(enable_client_, "enable shooter")) {
          return;
        }
        if (targets_.empty()) {
          enter_state(State::RETURN_HOME);
        } else {
          enter_state(State::NAVIGATE);
        }
        break;

      case State::NAVIGATE:
        handle_navigation(false);
        break;

      case State::SEARCH:
        handle_search();
        break;

      case State::ALIGN:
        handle_alignment();
        break;

      case State::FIRE:
        stop_robot();
        if (!fire_is_allowed()) {
          RCLCPP_WARN(get_logger(), "Fire command blocked by opponent-target safety gate");
          call_trigger(disable_client_, "disable shooter");
          shooter_dwell_active_ = false;
          enter_state(State::NEXT_TARGET);
          break;
        }
        if (!shooter_dwell_active_) {
          if (!call_trigger(enable_client_, "enable shooter dwell")) {
            return;
          }
          shooter_dwell_active_ = true;
          shooter_dwell_start_time_ = now();
          RCLCPP_INFO(get_logger(), "Laser dwell started for %.2f s before fire", laser_dwell_required_s_);
          return;
        }
        if (elapsed_since(shooter_dwell_start_time_) < laser_dwell_required_s_) {
          return;
        }
        if (!call_trigger(fire_client_, "fire shooter")) {
          return;
        }
        call_trigger(disable_client_, "disable shooter");
        shooter_dwell_active_ = false;
        enter_state(State::NEXT_TARGET);
        break;

      case State::NEXT_TARGET:
        stop_robot();
        ++current_target_index_;
        nav_retry_count_ = 0;
        localization_recovery_count_ = 0;
        if (current_target_index_ < targets_.size()) {
          enter_state(State::NAVIGATE);
        } else {
          enter_state(State::RETURN_HOME);
        }
        break;

      case State::RECOVER_LOCALIZATION:
        handle_localization_recovery();
        break;

      case State::RETURN_HOME:
        handle_navigation(true);
        break;

      case State::FAILSAFE:
        stop_robot();
        call_trigger(disable_client_, "disable shooter");
        enter_state(State::END);
        break;

      case State::END:
        break;
    }
  }

  void handle_navigation(const bool home_goal)
  {
    if (!nav_goal_sent_) {
      const TargetPose target = home_goal ? home_ : targets_.at(current_target_index_);
      if (!send_nav_goal(target)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Waiting for Nav2 action server");
        return;
      }
    }

    if (!nav_done_) {
      return;
    }

    if (nav_success_) {
      stop_robot();
      if (home_goal) {
        call_trigger(disable_client_, "disable shooter");
        enter_state(State::END);
      } else {
        enter_state(State::SEARCH);
      }
      return;
    }

    if (home_goal) {
      RCLCPP_WARN(get_logger(), "Home navigation failed; ending safely");
      call_trigger(disable_client_, "disable shooter");
      enter_state(State::END);
      return;
    }

    ++nav_retry_count_;
    if (nav_retry_count_ <= nav_retry_limit_) {
      RCLCPP_WARN(
        get_logger(), "Navigation failed for target %zu, retry %d/%d",
        current_target_index_ + 1, nav_retry_count_, nav_retry_limit_);
      nav_goal_sent_ = false;
      nav_done_ = false;
      nav_success_ = false;
    } else {
      if (localization_recovery_count_ < localization_recovery_limit_) {
        RCLCPP_WARN(
          get_logger(), "Navigation failed for target %zu; spinning to rebuild map/localization",
          current_target_index_ + 1);
        enter_state(State::RECOVER_LOCALIZATION);
      } else {
        RCLCPP_WARN(get_logger(), "Skipping unreachable target %zu", current_target_index_ + 1);
        enter_state(State::NEXT_TARGET);
      }
    }
  }

  void handle_localization_recovery()
  {
    if (elapsed_since(state_enter_time_) < localization_recovery_spin_s_) {
      geometry_msgs::msg::Twist twist;
      twist.angular.z = localization_recovery_angular_speed_;
      cmd_vel_pub_->publish(twist);
      return;
    }

    stop_robot();
    ++localization_recovery_count_;
    nav_retry_count_ = 0;
    nav_goal_sent_ = false;
    nav_done_ = false;
    nav_success_ = false;
    has_detection_ = false;
    if (current_target_index_ < targets_.size()) {
      enter_state(State::NAVIGATE);
    } else {
      enter_state(State::RETURN_HOME);
    }
  }

  void handle_search()
  {
    if (has_recent_matching_detection()) {
      enter_state(State::ALIGN);
      return;
    }

    if (elapsed_since(state_enter_time_) > search_timeout_s_) {
      RCLCPP_WARN(get_logger(), "Target %zu not found during search", current_target_index_ + 1);
      enter_state(State::NEXT_TARGET);
      return;
    }

    geometry_msgs::msg::Twist twist;
    twist.angular.z = search_angular_speed_;
    cmd_vel_pub_->publish(twist);
  }

  void handle_alignment()
  {
    if (!has_recent_matching_detection()) {
      enter_state(State::SEARCH);
      return;
    }

    if (last_detection_.aligned || is_aligned(last_detection_)) {
      enter_state(State::FIRE);
      return;
    }

    if (elapsed_since(state_enter_time_) > alignment_timeout_s_) {
      RCLCPP_WARN(get_logger(), "Alignment timeout for target %zu", current_target_index_ + 1);
      enter_state(State::NEXT_TARGET);
      return;
    }

    geometry_msgs::msg::Twist twist;
    twist.angular.z = std::clamp(-kp_angular_ * last_detection_.center_x, -max_angular_speed_, max_angular_speed_);
    if (!std::isnan(last_detection_.distance_z)) {
      const double distance_error = last_detection_.distance_z - target_distance_m_;
      twist.linear.x = std::clamp(kp_linear_ * distance_error, -max_linear_speed_, max_linear_speed_);
    }
    cmd_vel_pub_->publish(twist);
  }

  bool has_recent_matching_detection() const
  {
    if (!has_detection_ || current_target_index_ >= targets_.size()) {
      return false;
    }
    if ((now() - last_detection_time_).seconds() > 0.5) {
      return false;
    }
    if (enforce_opponent_targets_ && is_own_detection(last_detection_)) {
      return false;
    }
    return last_detection_.tag_id == targets_.at(current_target_index_).tag_id;
  }

  bool fire_is_allowed() const
  {
    if (!enforce_opponent_targets_) {
      return true;
    }
    if (!has_detection_ || current_target_index_ >= targets_.size()) {
      return false;
    }
    const auto & target = targets_.at(current_target_index_);
    return target.owner != team_color_ && !is_own_detection(last_detection_) &&
           last_detection_.tag_id == target.tag_id;
  }

  bool is_own_detection(const rcvrl_interfaces::msg::TargetDetection & detection) const
  {
    return detection.target_type == team_color_ + "_base";
  }

  std::string infer_owner_from_tag(const int tag_id) const
  {
    if (tag_id == 2) {
      return "yellow";
    }
    if (tag_id == 3) {
      return "blue";
    }
    return "unknown";
  }

  bool is_aligned(const rcvrl_interfaces::msg::TargetDetection & detection) const
  {
    const bool centered = std::fabs(detection.center_x) <= target_center_tolerance_;
    const bool distance_ok =
      std::isnan(detection.distance_z) ||
      std::fabs(detection.distance_z - target_distance_m_) <= target_distance_tolerance_m_;
    return centered && distance_ok;
  }

  bool send_nav_goal(const TargetPose & target)
  {
    if (!nav_client_->action_server_is_ready() && !nav_client_->wait_for_action_server(500ms)) {
      return false;
    }

    auto goal_msg = NavigateToPose::Goal();
    goal_msg.pose.header.frame_id = map_frame_;
    goal_msg.pose.header.stamp = now();
    goal_msg.pose.pose.position.x = target.x;
    goal_msg.pose.pose.position.y = target.y;
    goal_msg.pose.pose.orientation = yaw_to_quaternion(target.yaw);

    auto send_goal_options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    send_goal_options.goal_response_callback =
      [this](const GoalHandleNavigate::SharedPtr & goal_handle) {
        if (!goal_handle) {
          RCLCPP_WARN(get_logger(), "Nav2 rejected goal");
          nav_done_ = true;
          nav_success_ = false;
        }
      };
    send_goal_options.result_callback =
      [this](const GoalHandleNavigate::WrappedResult & result) {
        nav_done_ = true;
        nav_success_ = result.code == rclcpp_action::ResultCode::SUCCEEDED;
      };

    nav_client_->async_send_goal(goal_msg, send_goal_options);
    nav_goal_sent_ = true;
    nav_done_ = false;
    nav_success_ = false;
    RCLCPP_INFO(get_logger(), "Sent Nav2 goal: x=%.2f y=%.2f yaw=%.2f", target.x, target.y, target.yaw);
    return true;
  }

  static geometry_msgs::msg::Quaternion yaw_to_quaternion(const double yaw)
  {
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw);
    q.normalize();
    geometry_msgs::msg::Quaternion msg;
    msg.x = q.x();
    msg.y = q.y();
    msg.z = q.z();
    msg.w = q.w();
    return msg;
  }

  bool call_trigger(
    const rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr & client,
    const std::string & label)
  {
    if (!client->service_is_ready()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Service unavailable: %s", label.c_str());
      return false;
    }
    client->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
    return true;
  }

  void request_localization_recovery(const std::string & reason)
  {
    if ((now() - last_collision_recovery_time_).seconds() < collision_recovery_cooldown_s_) {
      return;
    }
    last_collision_recovery_time_ = now();
    localization_recovery_requested_ = true;
    RCLCPP_WARN(get_logger(), "Localization recovery requested: %s", reason.c_str());
  }

  void handle_contact_impulse(const std::string & reason)
  {
    if (filtered_odom_confidence_ < filtered_odom_min_confidence_) {
      request_localization_recovery(reason + " with low filtered odom confidence");
      return;
    }
    stop_robot();
    if (state_ == State::NAVIGATE || state_ == State::RETURN_HOME) {
      nav_goal_sent_ = false;
      nav_done_ = false;
      nav_success_ = false;
    }
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1500,
      "Contact handled without full relocalization: %s (filtered odom confidence %.2f)",
      reason.c_str(), filtered_odom_confidence_);
  }

  void enter_state(const State next_state)
  {
    if (state_ == next_state) {
      return;
    }

    RCLCPP_INFO(get_logger(), "%s -> %s", state_name(state_).c_str(), state_name(next_state).c_str());
    state_ = next_state;
    state_enter_time_ = now();

    if (state_ == State::NAVIGATE || state_ == State::RETURN_HOME) {
      nav_goal_sent_ = false;
      nav_done_ = false;
      nav_success_ = false;
    }
    if (state_ != State::FIRE) {
      shooter_dwell_active_ = false;
    }
  }

  void stop_robot()
  {
    cmd_vel_pub_->publish(geometry_msgs::msg::Twist());
  }

  double elapsed_since(const rclcpp::Time & timestamp) const
  {
    return (now() - timestamp).seconds();
  }

  bool auto_start_ {true};
  std::string map_frame_ {"map"};
  std::string cmd_vel_topic_ {"/cmd_vel"};
  std::string filtered_odom_topic_ {"/odometry/filtered"};
  std::string imu_topic_ {"/imu/data_raw"};
  std::string bumper_left_topic_ {"/bumper/front_left"};
  std::string bumper_right_topic_ {"/bumper/front_right"};
  std::string target_detection_topic_ {"/target_detection"};
  std::string navigate_action_name_ {"navigate_to_pose"};
  double match_timeout_s_ {180.0};
  double stationary_timeout_s_ {20.0};
  double search_timeout_s_ {6.0};
  double alignment_timeout_s_ {5.0};
  int nav_retry_limit_ {2};
  int nav_retry_count_ {0};
  double search_angular_speed_ {0.35};
  double kp_angular_ {0.9};
  double kp_linear_ {0.45};
  double max_angular_speed_ {0.5};
  double max_linear_speed_ {0.18};
  double target_center_tolerance_ {0.08};
  double target_distance_m_ {0.52};
  double target_distance_tolerance_m_ {0.06};
  double laser_dwell_required_s_ {0.80};
  double localization_recovery_spin_s_ {5.8};
  double localization_recovery_angular_speed_ {0.55};
  double collision_recovery_cooldown_s_ {2.0};
  double collision_accel_threshold_ {4.0};
  double collision_yaw_rate_threshold_ {2.4};
  double filtered_odom_min_confidence_ {0.35};
  double filtered_odom_xy_cov_warn_ {0.18};
  double filtered_odom_yaw_cov_warn_ {0.16};
  double filtered_odom_confidence_ {1.0};
  int localization_recovery_limit_ {2};
  int localization_recovery_count_ {0};
  bool localization_recovery_requested_ {false};
  std::string team_color_ {"yellow"};
  bool enforce_opponent_targets_ {true};

  State state_ {State::INIT};
  TargetPose home_;
  std::vector<TargetPose> targets_;
  size_t current_target_index_ {0};

  bool has_detection_ {false};
  rcvrl_interfaces::msg::TargetDetection last_detection_;
  rclcpp::Time last_detection_time_;
  rclcpp::Time match_start_time_;
  rclcpp::Time state_enter_time_;
  rclcpp::Time shooter_dwell_start_time_;
  rclcpp::Time last_collision_recovery_time_ {0, 0, RCL_ROS_TIME};
  rclcpp::Time last_filtered_odom_time_;
  bool shooter_dwell_active_ {false};

  bool nav_goal_sent_ {false};
  bool nav_done_ {false};
  bool nav_success_ {false};

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr filtered_odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr bumper_left_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr bumper_right_sub_;
  rclcpp::Subscription<rcvrl_interfaces::msg::TargetDetection>::SharedPtr detection_sub_;
  rclcpp_action::Client<NavigateToPose>::SharedPtr nav_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr enable_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr fire_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr disable_client_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CompetitionBehavior>());
  rclcpp::shutdown();
  return 0;
}
