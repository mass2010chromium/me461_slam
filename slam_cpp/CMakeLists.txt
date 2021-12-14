cmake_minimum_required(VERSION 3.0)

#set(CMAKE_BUILD_TYPE Debug)

project( slam )
find_package( OpenCV REQUIRED )
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package( Boost COMPONENTS system thread REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( slam slam.cpp dbscan.cpp fast_alloc.cpp )
target_precompile_headers( slam PUBLIC headers.hpp )
target_link_libraries( slam ${OpenCV_LIBS} ${Boost_LIBRARIES} Eigen3::Eigen )
