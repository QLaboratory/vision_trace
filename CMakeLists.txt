cmake_minimum_required(VERSION 2.8)

PROJECT(vision_map)
FIND_PACKAGE(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++11")



find_package(OpenCV REQUIRED)	
include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory(. DIR_SRCS)

add_executable(vision_map ${DIR_SRCS})
target_link_libraries(vision_map ${OpenCV_LIBS})
