cmake_minimum_required(VERSION 3.0)

#set(CMAKE_BUILD_TYPE Debug)
#set (CMAKE_CXX_FLAGS "-O3 -funsafe-math-optimizations -mcpu=cortex-a72 -mfloat-abi=hard -mfpu=neon-fp-armv8 -mtune=cortex-a72")

project( slam )
find_package( OpenCV REQUIRED )
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package( Boost COMPONENTS system thread REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( slam slam.cpp dbscan.cpp fast_alloc.cpp )
target_precompile_headers( slam PUBLIC headers.hpp )
target_link_libraries( slam ${OpenCV_LIBS} ${Boost_LIBRARIES} Eigen3::Eigen )
