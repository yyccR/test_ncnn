aux_source_directory(. realsr)
#find_package(ncnn REQUIRED)
set(NCNN_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../lib/ncnn-20240410-macos-vulkan/ncnn.framework/Versions/A/Headers/ncnn)
include_directories(${NCNN_INCLUDE_DIRS})

find_package(OpenCV REQUIRED core imgproc highgui)

add_library(realsr_lib ${realsr})
target_link_libraries(realsr_lib ${OpenCV_LIBS})
#target_link_libraries(realsr_lib ncnn)
#target_link_libraries(realsr_lib "/Users/yang/CLionProjects/test_ncnn2/lib/ncnn-20240410-macos-vulkan/ncnn.framework")
#target_link_libraries(realsr_lib "/Users/yang/CLionProjects/test_ncnn2/lib/ncnn-20240410-macos-vulkan/glslang.framework")



