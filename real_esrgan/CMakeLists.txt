aux_source_directory(. real_esrgan)
#find_package(ncnn REQUIRED)
set(NCNN_INCLUDE_DIRS /Users/yang/CLionProjects/test_ncnn2/lib/ncnn-20240410-macos-vulkan/ncnn.framework/Versions/A/Headers/ncnn)
include_directories(${NCNN_INCLUDE_DIRS})

find_package(OpenCV REQUIRED core imgproc highgui)

add_library(real_esrgan_lib ${real_esrgan})
target_link_libraries(real_esrgan_lib ${OpenCV_LIBS})
#target_link_libraries(realsr_lib ncnn)
#target_link_libraries(realsr_lib "/Users/yang/CLionProjects/test_ncnn2/lib/ncnn-20240410-macos-vulkan/ncnn.framework")
#target_link_libraries(realsr_lib "/Users/yang/CLionProjects/test_ncnn2/lib/ncnn-20240410-macos-vulkan/glslang.framework")



