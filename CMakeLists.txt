cmake_minimum_required(VERSION 3.0)

#set(CMAKE_BUILD_TYPE Debug)

project( webserver )
find_package( Threads REQUIRED )
find_package( OpenCV REQUIRED )
find_library( atomic-lib NAMES libatomic.so libatomic.so.1 )
find_package( Boost COMPONENTS system thread filesystem REQUIRED )
add_executable( server webserver/webserver.cpp webserver/serial_dev.cpp )
target_link_libraries( server ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES} ${atomic-lib} ${OpenCV_LIBS} )

add_subdirectory( calibration )

project( slam )
find_package( OpenCV REQUIRED )
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package( Boost COMPONENTS system thread REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( slam slam.cpp )
target_link_libraries( slam ${OpenCV_LIBS} ${Boost_LIBRARIES} Eigen3::Eigen )

#project( serial_test )
#add_executable ( serial_test serial_main.c )
