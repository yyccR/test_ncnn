cmake_minimum_required(VERSION 3.17)
project(test_ncnn2)

set(CMAKE_CXX_STANDARD 17)

find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED core imgproc highgui)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sherpa)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/realsr)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sherpa/sherpa-ncnn)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sherpa/sherpa-ncnn/ncnn)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sherpa/sherpa-ncnn/portaudio/include)

add_subdirectory(common)
add_subdirectory(yolov5-seg)
add_subdirectory(yolov8)
add_subdirectory(yolov8-pose)
add_subdirectory(yolov8-seg)
add_subdirectory(yolo-nas)
add_subdirectory(sherpa)
add_subdirectory(realsr)

#aux_source_directory(sherpa/sherpa-ncnn/c-api c-api-src-list)
#aux_source_directory(sherpa/sherpa-ncnn/csrc csrc-list)

add_library(kaldi-native-fbank-core SHARED IMPORTED)
set_target_properties(kaldi-native-fbank-core PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/sherpa/sherpa-ncnn/libkaldi-native-fbank-core.dylib)
add_library(libncnn SHARED IMPORTED)
set_target_properties(libncnn PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/sherpa/sherpa-ncnn/libncnn.dylib)
add_library(libsherpa-ncnn-c-api SHARED IMPORTED)
set_target_properties(libsherpa-ncnn-c-api PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/sherpa/sherpa-ncnn/libsherpa-ncnn-c-api.dylib)
add_library(libsherpa-ncnn-core SHARED IMPORTED)
set_target_properties(libsherpa-ncnn-core PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/sherpa/sherpa-ncnn/libsherpa-ncnn-core.dylib)
add_library(libsherpa-ncnn-portaudio SHARED IMPORTED)
set_target_properties(libsherpa-ncnn-portaudio PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/sherpa/sherpa-ncnn/libsherpa-ncnn-portaudio.dylib)

#add_executable(test_ncnn2 main.cpp ${c-api-src-list} ${csrc-list})
add_executable(test_ncnn2 main.cpp)

target_link_libraries(test_ncnn2 ncnn)
target_link_libraries(test_ncnn2 libncnn)
target_link_libraries(test_ncnn2 libsherpa-ncnn-portaudio)
target_link_libraries(test_ncnn2 kaldi-native-fbank-core)
target_link_libraries(test_ncnn2 libsherpa-ncnn-c-api)
target_link_libraries(test_ncnn2 libsherpa-ncnn-core)

target_link_libraries(test_ncnn2 ${OpenCV_LIBS})
target_link_libraries(test_ncnn2 common_lib)
target_link_libraries(test_ncnn2 ${yolov5_seg_lib})
target_link_libraries(test_ncnn2 ${yolov8_lib})
target_link_libraries(test_ncnn2 ${yolov8_pose_lib})
target_link_libraries(test_ncnn2 ${yolov8_seg_lib})
target_link_libraries(test_ncnn2 ${sherpa_lib})
target_link_libraries(test_ncnn2 ${realsr_lib})
