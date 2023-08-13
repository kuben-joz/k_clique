cmake_minimum_required(VERSION 3.24)
project(k_clique CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(k_clique main.cu input.cu input.cuh direct.cu direct.cuh util.cuh)

set_target_properties(k_clique PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
