cmake_minimum_required(VERSION 3.8)
project(wvb_vision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(wvb_interfaces REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc aruco calib3d)

add_executable(apriltag_detector src/apriltag_detector.cpp)
ament_target_dependencies(apriltag_detector
  rclcpp
  sensor_msgs
  geometry_msgs
  cv_bridge
  tf2
  wvb_interfaces
)
target_link_libraries(apriltag_detector ${OpenCV_LIBRARIES})

install(TARGETS apriltag_detector
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

