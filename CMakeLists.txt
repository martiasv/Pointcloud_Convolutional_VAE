cmake_minimum_required(VERSION 2.8.3)
project(pointcloud_utils)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )




find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  geometry_msgs
  message_generation
)

add_message_files(
  FILES
  LatSpace.msg
)

generate_messages()

catkin_package(
 INCLUDE_DIRS include
 CATKIN_DEPENDS roscpp sensor_msgs message_runtime
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS scripts/gazebo_env_scrambler.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)



#add_executable(pc2_to_open3D src/pc2_to_open3D.cpp)
#target_link_libraries(pc2_to_open3D ${catkin_LIBRARIES}) # ${Open3D_LIBRARIES})
#add_dependencies(pc2_to_open3D ${catkin_EXPORTED_TARGETS} )#${open3d_conversions_EXPORTED_TARGETS}  )
#target_include_directories(pc2_to_open3D PUBLIC ${Open3D_INCLUDE_DIRS})

# catkin_install_python(PROGRAMS scripts/PC2_subscriber.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )