cmake_minimum_required(VERSION 3.18)
project(limit_weight_runner LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
find_package(CUDAToolkit REQUIRED)
add_executable(limit_weight_runner
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_v4_runner_limit_weight.cu
)
target_include_directories(limit_weight_runner PRIVATE
  /mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/include
  /mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/include/standard_robot_cuda_ik
  /mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/src/cuda
)
target_link_libraries(limit_weight_runner PRIVATE CUDA::cudart)
set_target_properties(limit_weight_runner PROPERTIES CUDA_RUNTIME_LIBRARY Static)
