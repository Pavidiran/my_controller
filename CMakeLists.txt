cmake_minimum_required(VERSION 3.5)
project(my_controller)

if(CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)")
  add_compile_options(-Wall -Wextra)
endif()

# find dependencies
set(REF_GEN_INCLUDE_DEPENDS
  kdl_parser
  rclcpp
  Eigen3
)
set(CONTROLLER_INCLUDE_DEPENDS
  pluginlib
  rcpputils
  controller_interface
  realtime_tools
  trajectory_msgs
  pinocchio
  my_controller_interface
)

# find dependencies
find_package(ament_cmake REQUIRED)
foreach(Dependency IN ITEMS ${REF_GEN_INCLUDE_DEPENDS})
  find_package(${Dependency} REQUIRED)
endforeach()
foreach(Dependency IN ITEMS ${CONTROLLER_INCLUDE_DEPENDS})
  find_package(${Dependency} REQUIRED)
endforeach()


## COMPILE

add_definitions(-DOPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)


add_library(
  my_controller
  SHARED
  controller/my_controller.cpp
)

target_compile_features(my_controller PUBLIC cxx_std_17)
target_include_directories(my_controller PUBLIC
$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/controller/include>
$<INSTALL_INTERFACE:include/my_controller>
)
ament_target_dependencies(
  my_controller PUBLIC
  ${CONTROLLER_INCLUDE_DEPENDS}
)

# Causes the visibility macros to use dllexport rather than dllimport,
# which is appropriate when building the dll but not consuming it.
target_compile_definitions(${PROJECT_NAME} PRIVATE "My_CONTROLLER")

# Export controller plugins
pluginlib_export_plugin_description_file(controller_interface my_controller_plugin.xml)

# INSTALL

install(TARGETS my_controller
  EXPORT my_controller
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY controller/urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
endif()

## EXPORTS
ament_export_targets(my_controller HAS_LIBRARY_TARGET)
ament_export_dependencies(${THIS_PACKAGE_INCLUDE_DEPENDS})
ament_package() 