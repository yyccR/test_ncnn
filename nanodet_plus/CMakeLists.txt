aux_source_directory(. src_lists)
add_library(nanodet_plus_lib ${src_lists})
target_link_libraries(nanodet_plus_lib ncnn)
target_link_libraries(nanodet_plus_lib common_lib)
