aux_source_directory(. realsr)
find_package(ncnn REQUIRED)


add_library(realsr_lib ${realsr})
target_link_libraries(realsr_lib ncnn)




