cmake_minimum_required(VERSION 2.8.3)
project(pointcloud_utils)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  Eigen3
  Open3D
  open3d_conversions
  geometry_msgs
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES open3d_conversions
 CATKIN_DEPENDS roscpp sensor_msgs
 DEPENDS EIGEN3 Open3D
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Open3D_INCLUDE_DIRS}
  ${open3d_conversions_INCLUDE_DIRS}
)



#add_executable(pc2_to_open3D src/pc2_to_open3D.cpp)
#target_link_libraries(pc2_to_open3D ${catkin_LIBRARIES}) # ${Open3D_LIBRARIES})
#add_dependencies(pc2_to_open3D ${catkin_EXPORTED_TARGETS} )#${open3d_conversions_EXPORTED_TARGETS}  )
#target_include_directories(pc2_to_open3D PUBLIC ${Open3D_INCLUDE_DIRS})

catkin_install_python(PROGRAMS scripts/PC2_subscriber.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)