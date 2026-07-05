# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CUDA-accelerated batch inverse kinematics (IK) solver for UR10 6-DOF manipulator, targeting SM 8.9 (Ada Lovelace) with CUDA 13.3.

**Current method: OPT4C** = Structure-aware single-kernel fusion: Analytical Jacobian + Levenberg-Marquardt + Sobol-K16 low-discrepancy multi-start + Limit Barrier + target-block thread mapping + fused in-block candidate selection.

> **主线**: `standard_robot_cuda_ik/` 是唯一活跃项目。`history/` 为历史版本归档，仅供参考。

## Build

```bash
cd standard_robot_cuda_ik
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

**Target**: `CMAKE_CUDA_ARCHITECTURES=89` (Ada Lovelace). Build outputs under `build/`:

| Binary | Purpose |
|--------|---------|
| `standard_robot_cuda_v4_runner` | Main OPT4C runner (all modes, all precision/fallback/graph variants) |
| `standard_robot_cuda_v4_runner_r128` | Register-capped at 128 (occupancy experiment) |
| `standard_robot_cuda_v4_runner_r160` | Register-capped at 160 |
| `standard_robot_cuda_v4_runner_ptxas` | Verbose PTX assembly output |

## Solver Architecture

Single source file: `src/cuda/cuda_v4_runner.cu`. Key design features:

1. **Analytical Jacobian** via Fused FK: single forward pass outputs end-effector pose `T_ee`, joint world positions `p[6]`, and joint rotation axes `z[6]`. Jacobian columns = cross products: $J_{v,i}=z_i\times(p_{ee}-p_i)$, $J_{\omega,i}=z_i$. Replaces 12 FK evaluations/iteration → 1 FK + 6 cross products.

2. **Sobol K=16 multi-start**: 16 Sobol low-discrepancy seeds per target, each processed by one thread lane. Ablation experiment ($K=1$ vs $K=16$) proved this is the decisive factor for high success rate (SR drops from 0.954 → 0.522 when $K=1$).

3. **Target-block mapping** (`--variant opt4c_block_target`): `<<<N, 32>>>` grid, 16 active lanes process K=16 seeds via shared memory (`s_cand[16][kCandidateStride]`, stride=16 column-major → zero bank conflicts). Lane 0 performs in-block best-selection. Kernel launch count = 1 (constant, independent of N and K).

4. **Branch-free LM**: acceptance-rejection removed — trial solution always accepted, λ only scales. Eliminates warp divergence.

5. **Register-level 6×6 Gaussian elimination**: hand-rolled with partial pivoting, all intermediates in registers. No cuBLAS/cuSOLVER (library overhead > register latency for 6×6).

6. **Limit Barrier**: quadratic penalty, margin 0.087 rad (≈5°), analytic gradient.

7. **Precision modes**: `fp64` | `mixed_safe` (FP32 J/H) | `mixed_mid` | `mixed_aggressive` | `fp32_risky`. Mixed precision experiment showed only +2% throughput (bottleneck is FP64 linear solve, not Jacobian assembly).

8. **Fallback**: `strict_fail_to_fp64` — auto-retry in FP64 if mixed precision fails Strict threshold.

### CLI flags

```
--mode v4_static | fk_check | jacobian_check
--variant baseline | opt4c_block_target | opt4b_warp_target
--precision-mode fp64 | mixed_safe | mixed_mid | mixed_aggressive | fp32_risky
--fallback-mode none | strict_fail_to_fp64
--limit-gradient finite_diff | analytic
--graph-mode off | capture_replay
--N <targets> --K <seeds_per_target>
--max-iter 60 --repeat 30 --warmup 10
--targets <raw> --seeds <raw>
--best-csv <csv> --summary-csv <csv> --timing-csv <csv>
```

## Key Source Files

| File | Purpose |
|------|---------|
| `src/cuda/cuda_v4_runner.cu` | All kernels + main(): LM solver, candidate selection, FK/Jacobian check, full CLI |
| `src/cuda/cuda_utilities.cuh` | Device helpers (FK, Rodrigues, pose_error, LDLT), `__constant__` declarations, CUDA_CHECK |
| `include/standard_robot_cuda_ik/generated/ur10_model_constants.h` | Auto-generated from URDF: segment origins, axes, joint limits |

## Data

```
data/
├── cuda_inputs/             # Original target/seed raw files (N=100/500/1000/5000)
├── results/latest/          # Initial round benchmark output CSVs
└── experiments/             # Complete experiment data for paper
    ├── README.md            # Data-to-paper mapping (table X ← CSV file)
    ├── inputs/              # Dense N=100-1000 targets + seeds (K=16)
    ├── results/             # Main benchmark CSVs (10 N values)
    └── 补充实验/
        ├── inputs/          # K=1 seed files
        ├── results/         # K=1, cuRobo K=16, FP32 mixed precision CSVs
        └── *.py             # Benchmark scripts (see README.md)
```

## Paper

**Final version**: `论文/paper.tex` → `论文/paper.pdf` (10 pages, xelatex).

```
论文/
├── paper.tex           # Final LaTeX source
├── paper.pdf           # Compiled PDF (11 pages)
├── paper.txt           # Plain text for plagiarism check
├── .latexmkrc          # Force xelatex
├── 绘图/               # 5 figures (PDF + SVG + draw.io, bilingual captions)
├── 格式模板/           # Journal formatting reference
├── 计划/               # Reviewer comments + revision strategy
└── 名词解释/           # 47 technical terms glossary (CUDA/Robotics/Numerical)
```

Compile: `cd 论文 && latexmk -xelatex paper.tex`

### Paper key findings (all supported by experiment data in `data/experiments/`)

| Experiment | Key Result | Paper Location |
|-----------|-----------|---------------|
| OPT4C K=16 (main) | SR 0.940-0.960, throughput 15k-18k, p95 4.3-5.5mm | Table 5, §4.1 |
| K=1 ablation | SR collapses to 0.45-0.52, p95 surges to 642-685mm | Table 7, §4.3 |
| cuRobo K=16 fair comparison | SR 0.988 (quality ceiling), but throughput only 10k | Table 6, §4.2 |
| cuRobo K=1 (default) | Throughput 72k but SR only 0.84 | Table 6, §4.2 |
| FP32 mixed precision | Only +2% throughput, bottleneck is FP64 linear solve | Table 8, §4.4 |
| Nsight Compute | Long Scoreboard 83.2%, Issue Slot 2.32% | Table 9, §4.4 |
| PTX analysis | 194 registers/thread, ~44% occupancy, zero bank conflicts | Table 10, §4.4 |

### Paper narrative

Core contribution is **hardware-fused multi-start paradigm**, not optimizer superiority. Three-way Pareto frontier: cuRobo K=16 (quality ceiling, SR 0.988) → OPT4C K=16 (throughput拐点, SR 0.954, 1.75× faster) → cuRobo K=1 (extreme throughput, SR 0.840, quality unacceptable). All 15 references are cited in text.

## Python Scripts

### Benchmark orchestration (in `scripts/`)
- `run_final_push.py` — Figure generation + benchmark orchestration
- `run_v4_curobo_compare.py` — cuRobo comparison harness
- `audit_curobo_quality_round2.py` — cuRobo quality audit (multi-seed/optimizer sweeps)
- `audit_ur10_model_consistency.py` — UR10 FK/Jacobian/limit barrier verification

### Experiment scripts (in `data/experiments/补充实验/`)
- `run_dense_benchmarks.py` — OPT4C K=16 N=100-1000 benchmark
- `run_k1_benchmark.py` — K=1 ablation + comparison vs cuRobo
- `run_curobo_k16.py` — cuRobo K=16 fair comparison + three-way analysis
- `run_mixed_precision.py` — FP32 mixed precision ablation
- `generate_dense_inputs.py` — Slice N=200-900 targets/seeds from N=1000 base

### UR10 tools
- `tools/robot_model.py` — Python FK reference implementation
- `tools/verify_official_ur10.py` — Cross-check FK against yourdfpy

## Key Design Decisions

1. **Sobol K=16 multi-start is the decisive success factor**: Ablation proved LM optimizer alone (K=1) achieves only 45-52% SR. The 0.954 SR comes from 16 independent Sobol starting points — not from optimizer superiority.

2. **Kernel Fusion over multi-kernel pipelines**: Launch count = 1 regardless of N and K. cuRobo's multi-stage pipeline (particle init → evaluate → L-BFGS update) incurs fixed scheduling cost that dominates at N≤1000.

3. **FP64 linear solve is the bottleneck**: FP32 Jacobian/Hessian gives only +2% throughput. The 6×6 Gaussian elimination dominates iteration time — algorithm-level acceleration (warm-start, fewer iterations) is more promising than precision-level.

4. **No cuBLAS/cuSOLVER**: 6×6 systems are too small for library overhead. Hand-rolled Gaussian elimination in registers.

5. **Double precision required for convergence**: IK cannot tolerate FP32 accumulation in FK chains. Mixed precision (`mixed_safe`) preserves FP64 for linear solve and convergence check.

## Profiling

Nsight Compute was run on the main kernel (N=1000, K=16):
- Warp Stall Long Scoreboard: 83.2% → FP64 pipeline latency is the dominant bottleneck
- Issue Slot Utilization: 2.32% → extreme idle, consistent with compute-bound + heavily stalled
- Shared memory bank conflicts: ~112k (conflict rate <0.1%) → kCandidateStride=16 layout validated
- Compute (SM) Throughput: 84.2% → when active, compute-heavy
- Memory Throughput: 3.71% → not memory-bound

PTX static analysis: 194 registers/thread, ~44% theoretical occupancy, zero spill at default.

## Historical Code (仅供参考)

所有历史版本（DLS A0-A8 消融、V2 解析雅可比实验、V3 Python LM 原型、V4 Python 原型、pre-OPT4C CUDA 移植）已归档至 `history/`。详见 `history/README.md`。

> ⚠️ 历史代码不作为开发依据。当前活跃开发仅针对 `standard_robot_cuda_ik/` 中的 OPT4C 求解器。
