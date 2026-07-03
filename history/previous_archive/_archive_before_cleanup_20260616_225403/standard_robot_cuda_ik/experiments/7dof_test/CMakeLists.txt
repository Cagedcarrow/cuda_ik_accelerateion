cmake_minimum_required(VERSION 3.18)
project(panda_7dof_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

find_package(CUDAToolkit REQUIRED)

# Standalone 7DOF Panda IK test executable.
# panda_7dof_runner.cu #includes panda_7dof_kernel.cu, so only the runner
# is listed as source. This avoids double-definition of host functions.
add_executable(panda_7dof_test
  panda_7dof_runner.cu
)
target_include_directories(panda_7dof_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}  # for panda_model_constants.h
)
target_link_libraries(panda_7dof_test PRIVATE CUDA::cudart)
set_target_properties(panda_7dof_test PROPERTIES
  CUDA_RUNTIME_LIBRARY Static
)
