cmake_minimum_required(VERSION 3.17)
project(test_ncnn2)

set(CMAKE_CXX_STANDARD 17)

find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED core imgproc highgui)


add_subdirectory(common)
add_subdirectory(yolov5-seg)
add_subdirectory(yolov8)
add_subdirectory(yolov8-pose)
add_subdirectory(yolov8-seg)

add_executable(test_ncnn2 main.cpp)

target_link_libraries(test_ncnn2 ncnn)
target_link_libraries(test_ncnn2 ${OpenCV_LIBS})
target_link_libraries(test_ncnn2 common_lib)
target_link_libraries(test_ncnn2 ${yolov5_seg_lib})
target_link_libraries(test_ncnn2 ${yolov8_lib})
target_link_libraries(test_ncnn2 ${yolov8_pose_lib})
target_link_libraries(test_ncnn2 ${yolov8_seg_lib})
target_link_libraries(test_ncnn2 ${yolov5_nas_lib})
