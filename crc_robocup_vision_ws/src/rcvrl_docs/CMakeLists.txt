cmake_minimum_required(VERSION 3.8)
project(rcvrl_docs)

find_package(ament_cmake REQUIRED)

install(DIRECTORY docs
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

