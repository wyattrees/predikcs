cmake_minimum_required(VERSION 3.5)
project(predikcs)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(orocos_kdl REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(control_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(urdf REQUIRED)



set(THIS_PACKAGE_INCLUDE_DEPENDS
  control_msgs
  orocos_kdl
  kdl_parser
  rclcpp
  std_msgs
  sensor_msgs
  Eigen3
  tf2
  trajectory_msgs
  urdf
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

# catkin_package(
#   INCLUDE_DIRS include
# #  LIBRARIES predikcs
#  CATKIN_DEPENDS control_msgs kdl_parser roscpp rospy std_msgs sensor_msgs tf trajectory_msgs urdf
#  DEPENDS orocos_kdl EIGEN3
# )


# ROS2 Galactic

include_directories(
  include
  ${Boost_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/voo_bandit.cpp
  src/config_sample.cpp
  src/reward_calculator.cpp
  src/robot_model.cpp
  src/user_model.cpp
  src/goal_classifier_user_model.cpp
)

ament_target_dependencies(${PROJECT_NAME} ${THIS_PACKAGE_INCLUDE_DEPENDS})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

ament_export_dependencies(
  ${THIS_PACKAGE_INCLUDE_DEPENDS}
)

ament_export_include_directories(
  include
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations


## Declare a C++ library

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(predikcs_fetch src/controller.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
## Specify libraries to link a library or executable target against
target_link_libraries(predikcs_fetch
	${catkin_LIBRARIES} ${orocos_kdl_LIBRARIES} ${EIGEN3_LIBRARIES} ${PROJECT_NAME}
)

install(
  TARGETS predikcs_fetch
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# install the launch directory
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/reset.py
#   scripts/recorder.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_predictive_velocity_controller.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)

ament_package()