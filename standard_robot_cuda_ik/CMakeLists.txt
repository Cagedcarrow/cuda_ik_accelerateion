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
find_package(orocos_kdl QUIET)

set(PROJECT_INC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(PROJECT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(standard_robot_cuda_memory STATIC
  ${PROJECT_SRC}/cuda/cuda_memory.cu
)
target_include_directories(standard_robot_cuda_memory PUBLIC
  ${PROJECT_INC}
  ${PROJECT_INC}/standard_robot_cuda_ik
  ${PROJECT_SRC}/cuda
)
target_link_libraries(standard_robot_cuda_memory PUBLIC CUDA::cudart)
set_target_properties(standard_robot_cuda_memory PROPERTIES
  CUDA_RUNTIME_LIBRARY Static
  POSITION_INDEPENDENT_CODE ON
)

# ---- Ablation-level executable targets (A0-A6) ----
# Each target is compiled with a distinct ABLATION_LEVEL compile-time constant,
# enabling #if-based conditional compilation of optimizations.
#   A0: global memory params, unpadded, fixed lambda, no step clamp, no branch align
#   A1: + __constant__ memory
#   A2: + PaddedMat6x8 (PADDEDMAT_STRIDE=8)
#   A3: + register LDLT (already register-resident, no code change needed)
#   A4: + kernel fusion (already fused, no code change needed)
#   A5: + adaptive damping
#   A6: full optimized (all features enabled)
#   A7: + mixed precision (FP32 FK/Jacobian/Hessian + FP64 LDLT)
#   A8: + CUDA Graph replay (A7 + USE_CUDA_GRAPH)
function(add_ablation_target LEVEL)
  set(TARGET_NAME standard_robot_cuda_runner_A${LEVEL})
  add_executable(${TARGET_NAME}
    ${PROJECT_SRC}/cuda/cuda_benchmark_runner.cu
  )
  target_include_directories(${TARGET_NAME} PRIVATE
    ${PROJECT_INC}
    ${PROJECT_INC}/standard_robot_cuda_ik
    ${PROJECT_SRC}/cuda
  )
  target_compile_definitions(${TARGET_NAME} PRIVATE
    ABLATION_LEVEL=${LEVEL}
  )
  target_link_libraries(${TARGET_NAME} PRIVATE
    CUDA::cudart
    standard_robot_cuda_memory
  )
  set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
  )
endfunction()

foreach(LEVEL RANGE 0 7)
  add_ablation_target(${LEVEL})
endforeach()

# A8: A7 (mixed precision) + CUDA Graph replay
add_executable(standard_robot_cuda_runner_A8
  ${PROJECT_SRC}/cuda/cuda_benchmark_runner.cu
)
target_include_directories(standard_robot_cuda_runner_A8 PRIVATE
  ${PROJECT_INC}
  ${PROJECT_INC}/standard_robot_cuda_ik
  ${PROJECT_SRC}/cuda
)
target_compile_definitions(standard_robot_cuda_runner_A8 PRIVATE
  ABLATION_LEVEL=7
  USE_CUDA_GRAPH
)
target_link_libraries(standard_robot_cuda_runner_A8 PRIVATE
  CUDA::cudart
  standard_robot_cuda_memory
)
set_target_properties(standard_robot_cuda_runner_A8 PROPERTIES
  CUDA_RUNTIME_LIBRARY Static
)

# Default A6 runner (backward-compatible): same as standard_robot_cuda_runner_A6
add_executable(standard_robot_cuda_runner
  ${PROJECT_SRC}/cuda/cuda_benchmark_runner.cu
)
target_include_directories(standard_robot_cuda_runner PRIVATE
  ${PROJECT_INC}
  ${PROJECT_INC}/standard_robot_cuda_ik
  ${PROJECT_SRC}/cuda
)
target_link_libraries(standard_robot_cuda_runner PRIVATE
  CUDA::cudart
  standard_robot_cuda_memory
)
set_target_properties(standard_robot_cuda_runner PROPERTIES
  CUDA_RUNTIME_LIBRARY Static
)

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
  standard_robot_cuda_memory
)
set_target_properties(standard_robot_cuda_v4_runner PROPERTIES
  CUDA_RUNTIME_LIBRARY Static
)

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
    standard_robot_cuda_memory
  )
  set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
  )
endfunction()

add_v4_regcap_target(160)
add_v4_regcap_target(128)

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
  standard_robot_cuda_memory
)
set_target_properties(standard_robot_cuda_v4_runner_ptxas PROPERTIES
  CUDA_RUNTIME_LIBRARY Static
)

if(orocos_kdl_FOUND
   AND EXISTS "${PROJECT_SRC}/cpu_baseline/kdl_solver.cpp"
   AND EXISTS "${PROJECT_SRC}/cpu_baseline/numeric_dls_solver.cpp")
  add_library(standard_robot_cpu_baselines STATIC
    ${PROJECT_SRC}/cpu_baseline/kdl_solver.cpp
    ${PROJECT_SRC}/cpu_baseline/numeric_dls_solver.cpp
  )
  target_include_directories(standard_robot_cpu_baselines PUBLIC
    ${PROJECT_INC}
    ${PROJECT_INC}/standard_robot_cuda_ik
    ${PROJECT_SRC}/common
  )
  target_link_libraries(standard_robot_cpu_baselines PUBLIC orocos-kdl)
endif()
