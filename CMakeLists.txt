cmake_minimum_required(VERSION 3.0)

project( webserver )
find_package( Threads REQUIRED )
find_package( OpenCV REQUIRED )
find_package(nadjieb_mjpeg_streamer REQUIRED)
find_library( atomic-lib NAMES libatomic.so libatomic.so.1 )
find_package( Boost COMPONENTS system thread filesystem REQUIRED )
add_executable( server webserver/webserver.cpp webserver/serial_dev.cpp )
target_link_libraries( server ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES} ${atomic-lib} ${OpenCV_LIBS} )
add_executable( stream webserver/videostream.cpp )
target_link_libraries( stream nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer ${OpenCV_LIBS} )

add_subdirectory( calibration )
add_subdirectory( slam_cpp )

#project( serial_test )
#add_executable ( serial_test serial_main.c )
