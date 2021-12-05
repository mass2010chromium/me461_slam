cmake_minimum_required(VERSION 3.0)

project( undistort_cpp )
find_package( OpenCV REQUIRED )
find_package(nlohmann_json 3.2.0 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( undistort undistort.cpp )
target_link_libraries( undistort ${OpenCV_LIBS} nlohmann_json::nlohmann_json )
