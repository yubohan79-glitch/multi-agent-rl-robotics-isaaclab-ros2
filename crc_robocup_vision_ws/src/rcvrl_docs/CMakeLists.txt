cmake_minimum_required(VERSION 3.8)
project(rcvrl_docs)

find_package(ament_cmake REQUIRED)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/docs")
  install(DIRECTORY docs
    DESTINATION share/${PROJECT_NAME}
  )
endif()

ament_package()
