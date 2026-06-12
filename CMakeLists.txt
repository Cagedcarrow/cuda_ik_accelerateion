cmake_minimum_required(VERSION 3.18)
project(cuda_low_level_optimization LANGUAGES CXX CUDA)

# ── Compiler Standards ──────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture: Ada Lovelace (RTX 4060 Laptop / sm_89)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# ── Include directories ─────────────────────────────────────────────
# The test file uses relative includes (../src/*.cuh) and compiles
# all CUDA source as a single translation unit. This avoids the NVCC
# extern __constant__ cross-TU linking limitation.

set(CUDA_BASE "${CMAKE_CURRENT_SOURCE_DIR}/cuda_low_level_optimization")

# ── Test executable (single-TU: includes all kernel source directly) ─
add_executable(test_cuda_kernel
    "${CUDA_BASE}/test/test_cuda_kernel.cu"
)

target_include_directories(test_cuda_kernel PRIVATE
    "${CUDA_BASE}/src"
    "${CUDA_BASE}/include/cuda_low_level_optimization"
)

target_link_libraries(test_cuda_kernel PRIVATE
    CUDA::cudart
)

# Set CUDA runtime to static for portability
set_target_properties(test_cuda_kernel PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
)

# ── Optional: also build a shared library for other projects to link ─
# Disabled by default — enable with: cmake .. -DBUILD_LIBRARY=ON
option(BUILD_LIBRARY "Build cuda_ik_core static library" OFF)
if(BUILD_LIBRARY)
    add_library(cuda_ik_core STATIC
        "${CUDA_BASE}/src/cuda_kernels.cu"
        "${CUDA_BASE}/src/cuda_collision.cu"
        "${CUDA_BASE}/src/cuda_memory.cu"
    )
    target_include_directories(cuda_ik_core PUBLIC
        "${CUDA_BASE}/src"
        "${CUDA_BASE}/include/cuda_low_level_optimization"
    )
    target_link_libraries(cuda_ik_core PUBLIC CUDA::cudart)
    set_target_properties(cuda_ik_core PROPERTIES
        CUDA_RUNTIME_LIBRARY Static
        POSITION_INDEPENDENT_CODE ON
    )
endif()
