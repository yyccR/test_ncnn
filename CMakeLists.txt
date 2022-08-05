cmake_minimum_required(VERSION 3.17)
project(test_ncnn)

set(CMAKE_CXX_STANDARD 14)


find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED)

#include_directories(${CMAKE_CURRENT_LIST_DIR}/lib/opencv2.framework)
include_directories(${OpenCV_INCLUDE_DIRS})
#aux_source_directory(nanodet_plus DIR_SRCS)
#add_library(mylib '')

#include_directories(${CMAKE_CURRENT_LIST_DIR}/nanodet_plus)
add_subdirectory(nanodet_plus)
add_subdirectory(common)
#add_executable(nanodet_plus_lib main.cpp)

#add_executable(test_ncnn main.cpp nanodet_plus/nanodet.cpp nanodet_plus/nanodet.h)
add_executable(test_ncnn main.cpp)

target_link_libraries(test_ncnn ncnn)
target_link_libraries(test_ncnn ${OpenCV_LIBS})
target_link_libraries(test_ncnn nanodet_plus_lib)
target_link_libraries(test_ncnn common_lib)


