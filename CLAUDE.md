# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CUDA-accelerated batch inverse kinematics (IK) solver for UR10 6-DOF manipulator, targeting SM 8.9 (Ada Lovelace) with CUDA 13.3.

**Current method: OPT4C (CUDA-V4-Final-K16)** = Analytical Jacobian + Levenberg-Marquardt + Sobol-K16 seeds + Limit Barrier + Smoothness Rerank + target-block seed-parallel mapping + fused in-block candidate selection.

> **主线**: `standard_robot_cuda_ik/` 是唯一活跃项目。`history/` 目录下为历史版本归档，仅供参考，不作为开发依据。

## Build

```bash
cd standard_robot_cuda_ik
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

**Target architecture**: `CMAKE_CUDA_ARCHITECTURES=89` (default; override with `-DCMAKE_CUDA_ARCHITECTURES=<arch>`).

Build outputs (under `build/`):

| Binary | Description |
|--------|-------------|
| `standard_robot_cuda_v4_runner` | Main OPT4C runner |
| `standard_robot_cuda_v4_runner_r128` | Register-capped at 128 (occupancy experiment) |
| `standard_robot_cuda_v4_runner_r160` | Register-capped at 160 |
| `standard_robot_cuda_v4_runner_ptxas` | Verbose PTX assembly output |

## Solver Architecture

**Algorithm**: Levenberg-Marquardt with analytical Jacobian. Single source file: `src/cuda/cuda_v4_runner.cu` (1150 lines).

### Key design features

1. **Analytical Jacobian**: FK outputs joint positions `p[18]` and z-axes `z[18]`, then Jacobian columns are computed via cross products: J_v = z_j × (p_ee − p_j), J_ω = z_j. This replaces 12 FK evaluations per iteration (6 columns × ±ε numerical differencing) with a single FK pass + 6 cross products.

2. **Limit Barrier**: Quadratic penalty on joint-limit proximity (margin 0.087 rad). Supports both finite-difference and analytic gradient modes (`--limit-gradient finite_diff|analytic`).

3. **Smoothness Rerank**: Candidate selection hierarchy: success rank (strict > medium > loose > fail) → limit proximity → pose cost → seed index.

4. **Block-per-target parallelism** (`--variant opt4c_block_target`): `<<<N, 32>>>` grid, each block's 16 lanes process K seeds via shared memory (`s_cand[16*kCandidateStride]`), with lane 0 performing in-block best-selection. This is the recommended variant.

5. **FP32 fallback** (`--fallback-mode strict_fail_to_fp64`): If mixed precision produces non-strict result, the block automatically retries with full FP64 precision.

6. **CUDA Graph support** (`--graph-mode capture_replay`): Captures H2D → kernel → D2H as a graph for reduced launch overhead.

### Launch variants

| Variant | Kernel | Grid | Notes |
|---------|--------|------|-------|
| `baseline` | `ik_lm_multiseed_v4_kernel` + `select_best_per_target_v4_kernel` | N×K + N×1 | Two-launch, uses global memory for candidates |
| `opt4c_block_target` | `ik_lm_multiseed_v4_block_target_kernel` | N×1 (32 threads) | Single launch, shared memory selection, supports fallback + Graph |
| `opt4b_warp_target` | `ik_lm_multiseed_v4_warp_target_kernel` | (N/4)×1 (128 threads) | 4 warps/block, 4 targets per block |

### Key source files

| File | Purpose |
|------|---------|
| `src/cuda/cuda_v4_runner.cu` | All kernels + main(): LM solver, candidate selection, FK/jacobian check, full CLI |
| `src/cuda/cuda_utilities.cuh` | Device helpers (FK, Rodrigues, pose_error, LDLT, PaddedMat6x8), `__constant__` variable declarations, CUDA_CHECK macros |
| `include/standard_robot_cuda_ik/generated/ur10_model_constants.h` | Auto-generated from URDF: segment origins, axes, joint limits, weight schedule, lambda params |

### Data flow

All benchmark I/O uses raw binary files of double-precision floats:
- Targets: `[N, 16]` row-major 4×4 transform matrices
- Seeds: `[N*K, 6]` joint angles
- Output: CSV files (best per target, candidates, summary statistics, per-repeat timing)

Each candidate solution is `kCandidateStride=16` doubles, each best result is `kBestStride=18` doubles. See `solve_candidate_v4()` and `write_best_from_candidate_v4()` for field layout.

## Running Benchmarks

### OPT4C runner

```bash
./build/standard_robot_cuda_v4_runner \
  --mode v4_static \
  --variant opt4c_block_target \
  --limit-gradient analytic \
  --graph-mode off \
  --precision-mode mixed_safe \
  --fallback-mode none \
  --targets data/cuda_inputs/targets_N1000_T4x4_f64.raw \
  --seeds data/cuda_inputs/seeds_N1000_K16_q_f64.raw \
  --N 1000 --K 16 \
  --max-iter 60 --repeat 30 --warmup 10 \
  --best-csv data/results/latest/cuda_opt4c_best_N1000.csv \
  --summary-csv data/results/latest/cuda_opt4c_summary_N1000.csv \
  --timing-csv data/results/latest/cuda_opt4c_timing_N1000.csv
```

Key flags:
- `--variant`: `baseline` | `opt4c_block_target` (recommended) | `opt4b_warp_target`
- `--limit-gradient`: `finite_diff` | `analytic` (faster)
- `--precision-mode`: `fp64` | `mixed_safe` (FP32 J+H only) | `mixed_mid` | `mixed_aggressive` (FP32 q snap) | `fp32_risky`
- `--fallback-mode`: `none` | `strict_fail_to_fp64`
- `--graph-mode`: `off` | `capture_replay`

### FK / Jacobian verification

```bash
./build/standard_robot_cuda_v4_runner \
  --mode fk_check --seeds <q.raw> --best-csv fk_check.csv
./build/standard_robot_cuda_v4_runner \
  --mode jacobian_check --seeds <q.raw> --best-csv jacobian_check.csv
```

## Data

- `data/cuda_inputs/`: Raw double binary files — targets `[N,16]`, seeds `[N*K,6]`, q_samples
- `data/results/latest/`: Latest OPT4C benchmark output CSVs
- `urdf/ur10_official.urdf`: Canonical UR10 model; `ur10_official_source.json` records provenance

## Python Scripts

### Benchmark orchestration
- `scripts/run_final_push.py` — Generates all OPT4C results CSVs, figures, and paper-ready output
- `scripts/run_opt4c_finalization.py` — OPT4C finalization: static/timing benchmarks, FK checks, cuRobo comparison
- `scripts/run_opt4_followup.py` — OPT4 followup: block-target, warp-target, FP32 fallback, CUDA Graph sweeps
- `scripts/run_v4_enhancement_plan.py` — Register-cap experiments via `_r128`/`_r160` variants

### Cross-solver comparison
- `scripts/run_v4_curobo_compare.py` — cuRobo comparison harness (uses cuRobo Python API)
- `scripts/audit_curobo_quality_round2.py` — cuRobo quality audit (consumes OPT4C runner output CSVs)
- `scripts/audit_ur10_model_consistency.py` — UR10 FK/Jacobian/limit barrier verification

### V4 CUDA port acceptance
- `scripts/run_v4_cuda_plan.py` — V4-Final-K16 FP64 correctness + benchmark reporting

### UR10 model tools
- `tools/robot_model.py` — UR10 FK model in Python (reference implementation, used by comparison scripts)
- `tools/fetch_official_ur10.py` — Fetch/clone official UR10 URDF from UniversalRobots repo
- `tools/verify_official_ur10.py` — FK verification against official model using yourdfpy

## Paper

论文源文件位于 `论文/`：
- `paper.tex` — LaTeX 源文件（xelatex 编译）
- `paper.pdf` — 编译后 PDF
- `paper.md` — Markdown 格式（含嵌入图片）
- `基于 CUDA 小矩阵加速的机械臂批量逆运动学求解.pdf` — 最终输出 PDF
- `绘图/` — 全部 7 张图片（PDF + SVG + draw.io 源文件）
- `计划/` — 审稿意见

编译：`cd 论文 && latexmk -xelatex paper.tex`（或使用 `.latexmkrc` 自动配置）。

## External Reference Solvers

Cloned under `external/` for source-code reading only (not built or run):
- `external/curobo/` — NVIDIA cuRobo (FP32 particle-search GPU IK)
- `external/hjcd_ik/` — HJCD-IK
- `external/pyroki/` — PyRoki (JAX-based GPU IK)

## Key Design Decisions

1. **Analytical Jacobian over numerical**: The single largest throughput gain. Replaces 12 FK evaluations/iteration with 1 FK + 6 cross products.
2. **Double precision required**: IK convergence cannot tolerate FP32 accumulation in FK chains. Mixed precision (`mixed_safe`) is the compromise — FP32 for FK/Jacobian/Hessian, FP64 for linear solve and damping.
3. **Block-per-target over grid N×K**: `opt4c_block_target` uses `<<<N, 32>>>` with 16 seeds in shared memory, avoiding N×K grid launch overhead. K is fixed at 16.
4. **No cuBLAS/cuSOLVER**: 6×6 linear systems are too small for library overhead. Hand-rolled Gaussian elimination with partial pivoting in registers.
5. **Sobol K=16 seeds**: Low-discrepancy sequences provide better IK coverage than random/grid sampling with fewer seeds.

## Historical Code (仅供参考)

所有历史版本（DLS A0-A8 消融、V2 解析雅可比实验、V3 Python LM 原型、V4 Python 原型、pre-OPT4C CUDA 移植）已归档至 `history/`。详见 `history/README.md`。

> ⚠️ **历史代码不作为开发依据。** 当前活跃开发仅针对 `standard_robot_cuda_ik/` 中的 OPT4C 求解器。历史代码中的实验数据（A0-A8 CSVs、旧 NCU 报告）使用不同的算法（数值 Jacobian DLS）和评价协议，不可直接与当前 OPT4C 结果混用。
