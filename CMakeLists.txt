cmake_minimum_required(VERSION 3.8)
project(k9_stt_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ROS 2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(k9_interfaces_pkg REQUIRED)

#
# whisper.cpp installation/build locations
#
set(WHISPER_ROOT "/home/hopkira/whisper.cpp")
set(WHISPER_LIB_DIR "${WHISPER_ROOT}/build/bin")

#
# STT executable
#
add_executable(k9_stt
  src/k9_stt.cpp
)

ament_target_dependencies(k9_stt
  rclcpp
  std_msgs
  k9_interfaces_pkg
)

#
# whisper.cpp / ggml headers
#
target_include_directories(k9_stt PRIVATE
  "${WHISPER_ROOT}/include"
  "${WHISPER_ROOT}/ggml/include"
)

#
# Link directly to the libraries that we know exist.
#
target_link_libraries(k9_stt
  "${WHISPER_LIB_DIR}/libwhisper.so.1.9.2"
  "${WHISPER_LIB_DIR}/libggml.so.0.18.1"
  "${WHISPER_LIB_DIR}/libggml-base.so.0.18.1"
  "${WHISPER_LIB_DIR}/libggml-cpu.so.0.18.1"
  "${WHISPER_LIB_DIR}/libggml-cuda.so.0.18.1"
)

#
# whisper.cpp is currently in a non-system directory.
# Embed that directory in the executable's runtime search path.
#
set_target_properties(k9_stt PROPERTIES
  BUILD_RPATH "${WHISPER_LIB_DIR}"
  INSTALL_RPATH "${WHISPER_LIB_DIR}"
)

#
# Install ROS executable
#
install(TARGETS
  k9_stt
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
