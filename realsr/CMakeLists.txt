find_package(ncnn REQUIRED)

aux_source_directory(. realsr)

add_library(realsr_lib ${realsr})
target_link_libraries(realsr_lib ncnn)




