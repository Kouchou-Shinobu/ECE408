cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc)
set(WBPATH libwb)

project(ece408 LANGUAGES CUDA CXX)

# include and make LibWB
include_directories(${WBPATH})
add_subdirectory(${WBPATH})

# Machine Problems
# Add executables and linking stuff iteratively.
file(GLOB_RECURSE mps ${CMAKE_CURRENT_SOURCE_DIR} MP*/template.cu)
message(STATUS ${mps})
set(_C 0)
foreach(mp_source IN LISTS mps)
    set(currentMP mp${_C})
    add_executable(${currentMP} ${mp_source})
    target_link_libraries(${currentMP} PUBLIC ${WBPATH})
    set_target_properties(${currentMP} PROPERTIES OUTPUT_PATH ${currentMP})
    set_target_properties(${currentMP} PROPERTIES OUTPUT_NAME template)
    math(EXPR _C "${_C} + 1" OUTPUT_FORMAT DECIMAL)
endforeach()

# Project Milestone 1
add_executable(CPU_FORWARD ${CMAKE_CURRENT_SOURCE_DIR}/Project/custom/cpu-new-forward.cc)
add_executable(GPU_FORWARD ${CMAKE_CURRENT_SOURCE_DIR}/Project/custom/new-forward.cu)
