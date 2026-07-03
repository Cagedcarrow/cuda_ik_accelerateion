cmake_minimum_required(VERSION 3.18)
project(standard_robot_cuda_ik LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

find_package(CUDAToolkit REQUIRED)

set(PROJECT_INC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(PROJECT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---- OPT4C V4 Runner (Levenberg-Marquardt + Analytical Jacobian) ----
add_executable(standard_robot_cuda_v4_runner
  ${PROJECT_SRC}/cuda/cuda_v4_runner.cu
)
target_include_directories(standard_robot_cuda_v4_runner PRIVATE
  ${PROJECT_INC}
  ${PROJECT_INC}/standard_robot_cuda_ik
  ${PROJECT_SRC}/cuda
)
target_link_libraries(standard_robot_cuda_v4_runner PRIVATE
  CUDA::cudart
)
set_target_properties(standard_robot_cuda_v4_runner PROPERTIES
  CUDA_RUNTIME_LIBRARY Static
)

# ---- V4 Runner with register caps (for occupancy experiments) ----
function(add_v4_regcap_target REGCOUNT)
  set(TARGET_NAME standard_robot_cuda_v4_runner_r${REGCOUNT})
  add_executable(${TARGET_NAME}
    ${PROJECT_SRC}/cuda/cuda_v4_runner.cu
  )
  target_include_directories(${TARGET_NAME} PRIVATE
    ${PROJECT_INC}
    ${PROJECT_INC}/standard_robot_cuda_ik
    ${PROJECT_SRC}/cuda
  )
  target_compile_options(${TARGET_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v --maxrregcount=${REGCOUNT}>
  )
  target_link_libraries(${TARGET_NAME} PRIVATE
    CUDA::cudart
  )
  set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
  )
endfunction()

add_v4_regcap_target(160)
add_v4_regcap_target(128)

# ---- V4 Runner with verbose PTX assembly output ----
add_executable(standard_robot_cuda_v4_runner_ptxas
  ${PROJECT_SRC}/cuda/cuda_v4_runner.cu
)
target_include_directories(standard_robot_cuda_v4_runner_ptxas PRIVATE
  ${PROJECT_INC}
  ${PROJECT_INC}/standard_robot_cuda_ik
  ${PROJECT_SRC}/cuda
)
target_compile_options(standard_robot_cuda_v4_runner_ptxas PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)
target_link_libraries(standard_robot_cuda_v4_runner_ptxas PRIVATE
  CUDA::cudart
)
set_target_properties(standard_robot_cuda_v4_runner_ptxas PROPERTIES
  CUDA_RUNTIME_LIBRARY Static
)
