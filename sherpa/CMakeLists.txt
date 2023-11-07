find_package(ncnn REQUIRED)

#include_directories(${CMAKE_CURRENT_SOURCE_DIR})


#add_subdirectory(sherpa-ncnn/csrc)
#add_subdirectory(sherpa-ncnn/c-api)

aux_source_directory(. sherpa)
#aux_source_directory(sherpa-ncnn/c-api c-api-src-list)
#aux_source_directory(sherpa-ncnn/csrc csrc-list)
#aux_source_directory(csrc sherpa-ncnn-core-src)
#aux_source_directory(c-api sherpa-ncnn-c-api)

#add_library(kaldi-native-fbank-core SHARED IMPORTED)
#set_target_properties(kaldi-native-fbank-core PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_SOURCE_DIR}/sherpa-ncnn/libkaldi-native-fbank-core.dylib)

#add_library(kaldi SHARED IMPORTED)
#set_target_properties(kaldi PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_LIST_DIR}/libkaldi-native-fbank-core.so)

#add_library(sherpa_lib ${sherpa} ${c-api-src-list})
#add_library(sherpa_lib ${sherpa} ${c-api-src-list} ${csrc-list})
add_library(sherpa_lib ${sherpa})
target_link_libraries(sherpa_lib ncnn)
#target_link_libraries(sherpa_lib kaldi-native-fbank-core)
#target_link_libraries(sherpa_lib sherpa-ncnn-core)
#target_link_libraries(sherpa_lib sherpa-ncnn-c-api)
#target_link_libraries(sherpa_lib ${sherpa-ncnn-core})
#target_link_libraries(sherpa_lib ${sherpa-ncnn-c-api})




