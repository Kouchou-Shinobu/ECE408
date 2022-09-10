cmake_minimum_required(VERSION 3.16.3)

project(mp2 CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "-std=c++11")


include_directories(../libwb)
add_subdirectory(../libwb/lib)

file(GLOB files "MP*/template.cu")

#target_compile_options(
#
#)