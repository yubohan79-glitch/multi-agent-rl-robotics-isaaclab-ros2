#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <termios.h>
#include <unistd.h>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"

namespace
{
speed_t baud_to_termios(const int baudrate)
{
  switch (baudrate) {
    case 9600:
      return B9600;
    case 19200:
      return B19200;
    case 38400:
      return B38400;
    case 57600:
      return B57600;
    case 115200:
      return B115200;
    default:
      return B9600;
  }
}

std::vector<uint8_t> to_bytes(const std::vector<int64_t> & values)
{
  std::vector<uint8_t> bytes;
  bytes.reserve(values.size());
  for (const auto value : values) {
    bytes.push_back(static_cast<uint8_t>(value & 0xFF));
  }
  return bytes;
}
}  // namespace

class ShooterController : public rclcpp::Node
{
public:
  ShooterController()
  : Node("shooter_controller")
  {
    port_ = declare_parameter<std::string>("port", "/dev/arm");
    baudrate_ = declare_parameter<int>("baudrate", 9600);
    dry_run_ = declare_parameter<bool>("dry_run", false);
    close_after_disable_ = declare_parameter<bool>("close_after_disable", false);
    enable_command_ = to_bytes(declare_parameter<std::vector<int64_t>>("enable_command", {0xA3}));
    fire_command_ = to_bytes(declare_parameter<std::vector<int64_t>>("fire_command", {0xA3}));
    disable_command_ = to_bytes(declare_parameter<std::vector<int64_t>>("disable_command", {0xA0}));

    enable_service_ = create_service<std_srvs::srv::Trigger>(
      "/shooter/enable",
      [this](const std_srvs::srv::Trigger::Request::SharedPtr,
             std_srvs::srv::Trigger::Response::SharedPtr response) {
        response->success = send_command(enable_command_, "enable");
        response->message = response->success ? "shooter enabled" : last_error_;
      });

    fire_service_ = create_service<std_srvs::srv::Trigger>(
      "/shooter/fire",
      [this](const std_srvs::srv::Trigger::Request::SharedPtr,
             std_srvs::srv::Trigger::Response::SharedPtr response) {
        response->success = send_command(fire_command_, "fire");
        response->message = response->success ? "fire command sent" : last_error_;
      });

    disable_service_ = create_service<std_srvs::srv::Trigger>(
      "/shooter/disable",
      [this](const std_srvs::srv::Trigger::Request::SharedPtr,
             std_srvs::srv::Trigger::Response::SharedPtr response) {
        response->success = send_command(disable_command_, "disable");
        response->message = response->success ? "shooter disabled" : last_error_;
        if (response->success && close_after_disable_) {
          close_port();
        }
      });

    RCLCPP_INFO(get_logger(), "Shooter services ready on /shooter/{enable,fire,disable}");
  }

  ~ShooterController() override
  {
    close_port();
  }

private:
  bool open_port()
  {
    if (dry_run_) {
      return true;
    }
    if (fd_ >= 0) {
      return true;
    }

    fd_ = ::open(port_.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (fd_ < 0) {
      last_error_ = "failed to open " + port_ + ": " + std::strerror(errno);
      RCLCPP_ERROR(get_logger(), "%s", last_error_.c_str());
      return false;
    }

    termios tty {};
    if (tcgetattr(fd_, &tty) != 0) {
      last_error_ = "tcgetattr failed: " + std::string(std::strerror(errno));
      RCLCPP_ERROR(get_logger(), "%s", last_error_.c_str());
      close_port();
      return false;
    }

    const speed_t speed = baud_to_termios(baudrate_);
    cfsetospeed(&tty, speed);
    cfsetispeed(&tty, speed);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 5;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
      last_error_ = "tcsetattr failed: " + std::string(std::strerror(errno));
      RCLCPP_ERROR(get_logger(), "%s", last_error_.c_str());
      close_port();
      return false;
    }

    RCLCPP_INFO(get_logger(), "Opened shooter serial port %s at %d baud", port_.c_str(), baudrate_);
    return true;
  }

  bool send_command(const std::vector<uint8_t> & command, const std::string & label)
  {
    if (command.empty()) {
      last_error_ = label + " command is empty";
      RCLCPP_ERROR(get_logger(), "%s", last_error_.c_str());
      return false;
    }

    if (dry_run_) {
      RCLCPP_INFO(get_logger(), "Dry run: accepted shooter %s command", label.c_str());
      return true;
    }

    if (!open_port()) {
      return false;
    }

    const ssize_t written = ::write(fd_, command.data(), command.size());
    if (written < 0 || static_cast<size_t>(written) != command.size()) {
      last_error_ = "failed to write " + label + " command: " + std::strerror(errno);
      RCLCPP_ERROR(get_logger(), "%s", last_error_.c_str());
      return false;
    }

    tcdrain(fd_);
    RCLCPP_INFO(get_logger(), "Sent shooter %s command (%zu byte)", label.c_str(), command.size());
    return true;
  }

  void close_port()
  {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  std::string port_;
  int baudrate_ {9600};
  bool dry_run_ {false};
  bool close_after_disable_ {false};
  int fd_ {-1};
  std::string last_error_;
  std::vector<uint8_t> enable_command_;
  std::vector<uint8_t> fire_command_;
  std::vector<uint8_t> disable_command_;

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr enable_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr fire_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr disable_service_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ShooterController>());
  rclcpp::shutdown();
  return 0;
}

