aux_source_directory(. sherpa)
find_package(ncnn REQUIRED)

include_directories(c-api)
include_directories(csrc)

add_subdirectory(c-api)
add_subdirectory(csrc)

aux_source_directory(. sherpa)
aux_source_directory(csrc sherpa-ncnn-core-src)
aux_source_directory(c-api sherpa-ncnn-c-api)

add_library(sherpa_lib ${sherpa} ${sherpa-ncnn-core-src} ${sherpa-ncnn-c-api})
target_link_libraries(sherpa_lib ncnn)
#target_link_libraries(sherpa_lib ${sherpa-ncnn-core})
#target_link_libraries(sherpa_lib ${sherpa-ncnn-c-api})




