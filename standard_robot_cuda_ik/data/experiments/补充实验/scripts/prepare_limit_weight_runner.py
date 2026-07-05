#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

from common_metrics import ROOT, SUPP

VARIANT = SUPP / "runner_variant"
SRC_DIR = VARIANT / "src"
BUILD_DIR = VARIANT / "build"
SRC = SRC_DIR / "cuda_v4_runner_limit_weight.cu"


def patch_source(text: str) -> str:
    text = text.replace(
        "constexpr double kLimitWeight = 0.03;",
        "__constant__ double c_limit_weight;",
    )
    text = text.replace(
        "cudaError_t upload_constants() {",
        "cudaError_t upload_constants(double limit_weight) {",
    )
    text = text.replace(
        "err = cudaMemcpyToSymbol(c_lambda_params, k_lambda_params, sizeof(k_lambda_params));\n  return err;",
        "err = cudaMemcpyToSymbol(c_lambda_params, k_lambda_params, sizeof(k_lambda_params));\n"
        "  if (err != cudaSuccess) return err;\n"
        "  err = cudaMemcpyToSymbol(c_limit_weight, &limit_weight, sizeof(double));\n"
        "  return err;",
    )
    text = text.replace(
        "int N = 0, K = 16, repeat = 1, warmup = 0, max_iter = 60;",
        "int N = 0, K = 16, repeat = 1, warmup = 0, max_iter = 60;\n"
        "  double limit_weight = 0.03;",
    )
    text = text.replace(
        'else if (std::strcmp(argv[i], "--max-iter") == 0 && i + 1 < argc) max_iter = std::atoi(argv[++i]);',
        'else if (std::strcmp(argv[i], "--max-iter") == 0 && i + 1 < argc) max_iter = std::atoi(argv[++i]);\n'
        '    else if (std::strcmp(argv[i], "--limit-weight") == 0 && i + 1 < argc) limit_weight = std::atof(argv[++i]);',
    )
    text = text.replace(
        'std::printf("Usage: %s --mode v4_static --variant baseline|opt4c_block_target --limit-gradient finite_diff|analytic --graph-mode off|capture_replay|persistent_replay --precision-mode fp64|mixed_safe|mixed_mid|mixed_aggressive|fp32_risky --fallback-mode none|strict_fail_to_fp64 --targets <raw> --seeds <raw> --N <N> --K 16 [--repeat 30 --warmup 10]\\n", argv[0]);',
        'std::printf("Usage: %s --mode v4_static --variant baseline|opt4c_block_target --limit-gradient finite_diff|analytic --graph-mode off|capture_replay|persistent_replay --precision-mode fp64|mixed_safe|mixed_mid|mixed_aggressive|fp32_risky --fallback-mode none|strict_fail_to_fp64 --targets <raw> --seeds <raw> --N <N> --K 16 [--repeat 30 --warmup 10] [--limit-weight 0.03]\\n", argv[0]);',
    )
    text = text.replace("if (upload_constants() != cudaSuccess) {", "if (upload_constants(limit_weight) != cudaSuccess) {")
    text = text.replace("kLimitWeight", "c_limit_weight")
    text = text.replace(
        'std::printf("mode=%s\\nvariant=%s\\nlimit_gradient=%s\\ngraph_mode=%s\\nprecision_mode=%s\\nfallback_mode=%s\\nN=%d\\nK=%d\\nrepeat=%d\\nwarmup=%d\\ngpu_stream_ms_mean=%.9g\\nthroughput_targets_per_s=%.9g\\nbest_csv=%s\\nsummary_csv=%s\\ntiming_csv=%s\\n",',
        'std::printf("mode=%s\\nvariant=%s\\nlimit_gradient=%s\\ngraph_mode=%s\\nprecision_mode=%s\\nfallback_mode=%s\\nlimit_weight=%.9g\\nN=%d\\nK=%d\\nrepeat=%d\\nwarmup=%d\\ngpu_stream_ms_mean=%.9g\\nthroughput_targets_per_s=%.9g\\nbest_csv=%s\\nsummary_csv=%s\\ntiming_csv=%s\\n",',
    )
    text = text.replace(
        '              N, K, repeat, warmup, mean, mean > 0.0 ? 1000.0 * N / mean : 0.0,',
        '              limit_weight, N, K, repeat, warmup, mean, mean > 0.0 ? 1000.0 * N / mean : 0.0,',
    )
    return text


def write_cmake() -> None:
    cmake = f"""cmake_minimum_required(VERSION 3.18)
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
  ${{CMAKE_CURRENT_SOURCE_DIR}}/src/cuda_v4_runner_limit_weight.cu
)
target_include_directories(limit_weight_runner PRIVATE
  {ROOT / "include"}
  {ROOT / "include" / "standard_robot_cuda_ik"}
  {ROOT / "src" / "cuda"}
)
target_link_libraries(limit_weight_runner PRIVATE CUDA::cudart)
set_target_properties(limit_weight_runner PROPERTIES CUDA_RUNTIME_LIBRARY Static)
"""
    (VARIANT / "CMakeLists.txt").write_text(cmake, encoding="utf-8")


def main() -> int:
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    original = (ROOT / "src" / "cuda" / "cuda_v4_runner.cu").read_text(encoding="utf-8")
    SRC.write_text(patch_source(original), encoding="utf-8")
    write_cmake()
    subprocess.run(["cmake", "-S", str(VARIANT), "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(BUILD_DIR), "-j"], check=True)
    print(BUILD_DIR / "limit_weight_runner")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

