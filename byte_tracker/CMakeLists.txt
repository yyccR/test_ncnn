cmake_minimum_required(VERSION 3.10)
project(byte_tracker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_library(byte_tracker STATIC
    src/byte_tracker.cpp
    src/kalman_filter.cpp
    src/matching.cpp
)

target_include_directories(byte_tracker PUBLIC include)

add_executable(example example.cpp)
target_link_libraries(example PRIVATE byte_tracker) 