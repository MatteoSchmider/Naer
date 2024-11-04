cmake_minimum_required(VERSION 3.18)

project(naer VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)
include_directories(random123/include)

file(
    GLOB 
    SOURCES
    src/*.cpp
)

add_executable(naer ${SOURCES})

find_package(OpenCL REQUIRED)
target_link_libraries(naer OpenCL::OpenCL)

set_target_properties(
    naer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}
)