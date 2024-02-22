# cmake needs this line
cmake_minimum_required(VERSION 3.5)

# Define project name
project(3d_2d)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

include_directories("/usr/include/eigen3")
include_directories("/usr/include/suitesparse")
include_directories("/usr/local/include/g2o")

# Enable C++11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Declare the executable target built from your sources
add_executable(3d_2d main.cpp)

# Link your application with OpenCV libraries and g2o libraries
target_link_libraries(3d_2d PRIVATE ${OpenCV_LIBS} ${catkin_LIBRARIES} g2o_core g2o_types_sba g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension)
