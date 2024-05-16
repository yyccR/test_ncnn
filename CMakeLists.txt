cmake_minimum_required(VERSION 3.17)
project(test_ncnn2)

set(CMAKE_CXX_STANDARD 17)

set(NCNN_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/lib/ncnn-20240410-macos-vulkan/ncnn.framework/Versions/A/Headers)
include_directories(${NCNN_INCLUDE_DIRS})

#find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED core imgproc highgui)

#option(WEBP_ENABLE_SIMD "" ON)
#option(WEBP_BUILD_ANIM_UTILS "" OFF)
#option(WEBP_BUILD_CWEBP "" OFF)
#option(WEBP_BUILD_DWEBP "" OFF)
#option(WEBP_BUILD_GIF2WEBP "" OFF)
#option(WEBP_BUILD_IMG2WEBP "" OFF)
#option(WEBP_BUILD_VWEBP "" OFF)
#option(WEBP_BUILD_WEBPINFO "" OFF)
#option(WEBP_BUILD_WEBPMUX "" OFF)
#option(WEBP_BUILD_EXTRAS "" OFF)
#option(WEBP_BUILD_WEBP_JS "" OFF)
#option(WEBP_NEAR_LOSSLESS "" OFF)
#option(WEBP_ENABLE_SWAP_16BIT_CSP "" OFF)


#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/libwebp/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sherpa)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/realsr)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/real_esrgan)
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
#add_subdirectory(lib/libwebp)
add_subdirectory(real_esrgan)

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

#target_link_libraries(test_ncnn2 ncnn)
target_link_libraries(test_ncnn2 "${CMAKE_CURRENT_SOURCE_DIR}/lib/ncnn-20240410-macos-vulkan/ncnn.framework")
target_link_libraries(test_ncnn2 "${CMAKE_CURRENT_SOURCE_DIR}/lib/ncnn-20240410-macos-vulkan/glslang.framework")
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
target_link_libraries(test_ncnn2 realsr_lib)
target_link_libraries(test_ncnn2 real_esrgan_lib)
#target_link_libraries(test_ncnn2 webp)
