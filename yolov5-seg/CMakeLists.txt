aux_source_directory(. yolov5_seg)
#set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../libs/opencv-mobile-4.5.4-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc highgui)
find_package(ncnn REQUIRED)
add_library(yolov5_seg_lib ${yolov5_seg})
target_link_libraries(yolov5_seg_lib ${OpenCV_LIBS})
target_link_libraries(yolov5_seg_lib ncnn)
