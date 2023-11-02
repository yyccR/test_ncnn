aux_source_directory(. sherpa)
find_package(ncnn REQUIRED)

add_subdirectory(csrc)
add_subdirectory(c-api)

add_library(sherpa_lib ${sherpa})
target_link_libraries(sherpa_lib ncnn)
target_link_libraries(sherpa_lib ${csrc})
target_link_libraries(sherpa_lib ${api})




