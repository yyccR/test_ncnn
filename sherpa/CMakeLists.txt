aux_source_directory(. sherpa)
find_package(ncnn REQUIRED)

include_directories(c-api)
include_directories(csrc)

add_subdirectory(csrc)
add_subdirectory(c-api)

#aux_source_directory(. sherpa)
#aux_source_directory(csrc sherpa-ncnn-core-src)
#aux_source_directory(c-api sherpa-ncnn-c-api)

#add_library(kaldi SHARED IMPORTED)
#set_target_properties(kaldi PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_LIST_DIR}/libkaldi-native-fbank-core.so)

add_library(sherpa_lib ${sherpa})
target_link_libraries(sherpa_lib ncnn)
target_link_libraries(sherpa_lib sherpa-ncnn-core)
target_link_libraries(sherpa_lib sherpa-ncnn-c-api)
#target_link_libraries(sherpa_lib ${sherpa-ncnn-core})
#target_link_libraries(sherpa_lib ${sherpa-ncnn-c-api})




