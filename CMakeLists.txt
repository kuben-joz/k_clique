cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
find_package(CUDAToolkit)

set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86")
#set(CMAKE_CUDA_ARCHITECTURES "60")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(kcliques CUDA)
find_package(CUDAToolkit)

set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-G -g  -O3 -ffast-math -arch=sm_60 -DNDEBUG -gencode arch=compute_60,code=sm_60")
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-G -g -dopt=on -O3 -ffast-math -arch=sm_60 -DNDEBUG -gencode arch=compute_60,code=sm_60") # todo add -DNDEBUG
#set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-O3 -ffast-math -Wall -Wextra -DNDEBUG") # todo chanbe lineinfo to 

add_executable(kcliques)
target_sources(kcliques PRIVATE constants.cuh orient.cu orient.cuh pivot.cu pivot.cuh
                direct.cu direct.cuh input.cu input.cuh main.cu util.cuh)

#target_sources(kcliques PRIVATE constants.cuh
#                direct.cu direct.cuh input.cu input.cuh main.cu util.cuh count_cliques.cu count_cliques.cuh)

#set_target_properties(kcliques PROPERTIES
#    CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "60;61;70;75;80;86")
#set_target_properties(kcliques PROPERTIES
#    CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "60")
set_target_properties(kcliques PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)