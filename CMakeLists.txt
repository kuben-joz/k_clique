cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
find_package(CUDAToolkit)

set(CMAKE_CUDA_ARCHITECTURES 60)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(k-clique CUDA)
find_package(CUDAToolkit)

set(CMAKE_CUDA_STANDARD 17)
# -g -G
#set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-G -g -dopt=on -O3 -ffast-math -arch=sm_60 -DNDEBUG -gencode arch=compute_60,code=sm_60") # todo add -DNDEBUG
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-O3 -ffast-math -Wall -Wextra -DNDEBUG -arch=sm_60 -gencode arch=compute_60,code=sm_60") # todo chanbe lineinfo to 

add_executable(k-clique)
target_sources(k-clique PRIVATE constants.cuh count_cliques.cuh count_cliques.cu
                direct.cu direct.cuh input.cu input.cuh main.cu util.cuh)

set_target_properties(k-clique PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)