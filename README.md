# CUDA-Accelerated Industrial Robot Inverse Kinematics

**Hand-optimized CUDA batch IK solver achieving 36× throughput over cuRobo at small batch sizes, with near-perfect linear GPU scaling across 100–10,000 targets.**

This project demonstrates that a single-kernel, warp-parallel CUDA implementation — using mixed precision (FP32+FP64), register-resident linear algebra, shared memory bank-conflict elimination, and adaptive damping — can outperform framework-based GPU IK solvers by eliminating kernel-launch overhead, CPU-GPU synchronization, and framework dispatch costs.

---

## 1. Engineering Background — Why Batch Inverse Kinematics?

### The IK Problem

A 6-DOF serial industrial robot (like the UR10) must convert a desired end-effector pose (position + orientation in SE(3)) into six joint angles. This is the **inverse kinematics (IK)** problem. Unlike forward kinematics — which has a unique, closed-form solution — inverse kinematics is:

- **Non-convex**: up to 16 analytical solutions exist for a 6R manipulator
- **Singularity-prone**: near kinematic singularities, the Jacobian becomes ill-conditioned
- **Position-only ambiguity**: orientation constraints make analytical solutions unwieldy

Industrial practice therefore uses **numerical iterative methods** — typically the Damped Least Squares (DLS) algorithm:

```
while not converged:
    T_cur ← FK(q)                              # Forward kinematics
    e ← pose_error(T_cur, T_tgt)               # 6D error (3 pos + 3 rot)
    J ← numerical_jacobian(q, δ=1e-6)           # 6×6 Jacobian via central differencing
    H ← Jᵀ·W²·J + λ·I                          # Weighted Hessian
    g ← Jᵀ·W²·e                                # Gradient
    dq ← LDLT_solve(H, g)                       # 6×6 linear system
    q ← clamp(q + dq, joint_limits)             # Apply step with joint-limit enforcement
    λ ← adaptive_damping(λ, pos_err, stagnation) # Update damping
```

Each iteration requires 12 forward kinematics evaluations (6 columns × ±ε perturbations), one 6×6 LDLT decomposition, and convergence checking — all in double precision for numerical stability.

### Why Batch IK Matters

Modern robotics applications demand **thousands of IK solves per planning cycle**:

- **Trajectory optimization**: fitting a smooth joint-space path to Cartesian waypoints
- **Bin picking**: evaluating millions of grasp candidates
- **Collision-aware planning**: IK as the inner loop of sampling-based planners
- **Online replanning**: 50Hz control loops leave only 20ms per cycle

A CPU-based KDL solver requires ~6.2 seconds for 273 targets — 300× too slow for real-time use. GPU acceleration is not optional; it is the only path to interactive-rate batch IK.

### The UR10 Robot

<div align="center">

| Parameter | Value |
|:---|:---|
| DOF | 6 (all revolute) |
| Reach | 1,300 mm |
| Payload | 10 kg |
| Joint axes | Z / Y / Y / Y / −Z / Y |
| IK solution | Numerical only (mixed joint axes break analytical methods) |
| URDF source | [UniversalRobots/Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description) tag 4.3.1 |

</div>

The UR10's mixed joint axes (particularly the −Z wrist_2) make closed-form IK impractical; analytical solvers developed for standard 6R wrists (Pieper criterion) do not apply. This makes it an ideal test case for numerical GPU methods.

---

## 2. Project Structure

```
cuda_ik_accelerateion/
│
├── README.md                                   # ★ This file — project homepage
├── .gitignore                                  # Exclusion rules (build, IDE, profiling)
├── CMakeLists.txt                              # Root build (legacy test executable)
│
├── standard_robot_cuda_ik/                     # ★★★ MAIN PROJECT (Gen 2)
│   │                                           # Official standard robots, comprehensive benchmark
│   ├── src/
│   │   ├── cuda/                               # CUDA kernel source (3,194 LOC)
│   │   │   ├── cuda_ik_6dof.cu                 #   Core batch IK kernel (1,529 LOC)
│   │   │   │                                   #   · 9 ablation levels (A0-A8)
│   │   │   │                                   #   · Mixed precision kernel (A7)
│   │   │   │                                   #   · CUDA Graph kernel (A8)
│   │   │   ├── cuda_utilities.cuh              #   Device functions (447 LOC)
│   │   │   │                                   #   · FK, Rodrigues, PaddedMat6x8, LDLT
│   │   │   │                                   #   · Constant memory declarations
│   │   │   ├── cuda_benchmark_runner.cu        #   Host driver (483 LOC)
│   │   │   ├── cuda_collision.cu               #   GPU collision detection (716 LOC)
│   │   │   └── cuda_memory.cu                  #   DeviceBuffer RAII (19 LOC)
│   │   └── cpu_baseline/                       # CPU reference solvers
│   │       ├── kdl_solver.cpp                  #   Orocos KDL C++ wrapper
│   │       └── numeric_dls_solver.cpp          #   Python DLS baseline
│   │
│   ├── include/standard_robot_cuda_ik/         # C++ headers
│   │   ├── cuda_ik_6dof.h                      #   Kernel launch API
│   │   ├── cuda_collision.h
│   │   ├── cuda_memory.h
│   │   └── generated/ur10_model_constants.h    #   Auto-generated from URDF
│   │
│   ├── benchmark/                              # Multi-solver benchmark framework
│   │   ├── run_all.py                          #   Unified entry point
│   │   ├── common.py                           #   Shared framework (URDF, convergence)
│   │   ├── bench_cuda_6dof.py                  #   CUDA B5 solver wrapper
│   │   ├── bench_curobo.py                     #   cuRobo (NVIDIA) wrapper
│   │   ├── bench_pyroki.py                     #   PyRoki (JAX) wrapper
│   │   ├── bench_kdl.py                        #   KDL (C++ CPU) wrapper
│   │   ├── bench_numeric_dls.py                #   Numeric DLS (Python CPU) wrapper
│   │   └── compare_results.py                  #   Cross-solver comparison tool
│   │
│   ├── tools/                                  # Asset generation & verification
│   │   ├── generate_standard_assets.py         #   Batch target/seed generation
│   │   ├── robot_model.py                      #   URDF parser, FK, data export
│   │   ├── fetch_official_ur10.py              #   Official URDF downloader
│   │   └── verify_official_ur10.py             #   Model verification
│   │
│   ├── config/                                 # Project specification (tracked)
│   │   ├── benchmark.yaml                      #   Solver list, batch sizes, tolerances
│   │   ├── robots.yaml                         #   Robot model definitions
│   │   └── target_generation.yaml              #   Target/seed generation params
│   │
│   ├── urdf/                                   # Official robot models
│   │   ├── ur10_official.urdf                  #   UR10 from UniversalRobots
│   │   ├── ur5_official.urdf                   #   UR5
│   │   └── panda_7dof.urdf                     #   Franka Panda 7-DOF
│   │
│   ├── data/                                   # ★ All experiment data (184 files, 227MB)
│   │   ├── README.md                           #   Data-paper mapping index
│   │   ├── targets/                            #   36 IK target files (N=100→10000)
│   │   ├── seeds/                              #   96 seed files (4 strategies × 12 sizes)
│   │   ├── results/                            #   Benchmark results (11 CSVs)
│   │   │   ├── main_comparison/                #     B5 vs cuRobo @ Medium 10mm/5°
│   │   │   ├── ablation/                       #     B0/B3/B5 ablation
│   │   │   ├── threshold_scan/                 #     3-tier tolerance scan
│   │   │   ├── full_range/                     #     N=100→10000 linearity study
│   │   │   ├── cpu_baseline/                   #     CPU reference (KDL/numeric_dls)
│   │   │   ├── seed_strategy/                  #     zero_seed vs home_seed
│   │   │   └── panda_7dof/                     #     7-DOF Panda verification
│   │   ├── profiling/                          #   Nsight Compute GPU profiles
│   │   │   ├── ncu_summary.csv                 #     Key metrics summary
│   │   │   └── ncu_reports/                    #     5 raw .ncu-rep reports
│   │   └── figures/                            #   5 paper figures + generation script
│   │
│   ├── experiments/                            # Experiment workspaces
│   │   └── 7dof_test/                          #   Panda 7-DOF CUDA IK extension
│   │
│   └── CMakeLists.txt                          # Build: 10 ablation-level targets
│
├── cuda_low_level_optimization/                # Legacy Gen 1 (custom UR10+shovel)
│   ├── src/                                    #   CUDA kernels (2,751 LOC)
│   ├── test/                                   #   Test program + 273-target dataset
│   └── docs/                                   #   27 technical reference docs
│
├── docs/                                       # Central documentation
│   ├── PROJECT_OVERVIEW.md                     #   Complete project overview
│   ├── paper/                                  #   Paper draft (8 chapters, Markdown)
│   ├── logs/                                   #   Experiment logs (10 files)
│   ├── patent/                                 #   Patent disclosure documents
│   └── 修改意见/                                #   Revision feedback (10 files)
│
└── benchmark/          [gitignored]            # External solver clones (~456MB)
    ├── curobo/                                 #   NVIDIA cuRobo (PyTorch GPU IK)
    ├── hjcd_ik/                                #   HJCD-IK (PyTorch GPU IK)
    └── pyroki/                                 #   PyRoki (JAX GPU IK)
```

---

## 3. Quick Start

### Prerequisites

- **NVIDIA GPU** with Compute Capability ≥ sm_89 (Ada Lovelace: RTX 4060/4070/4080/4090)
- **CUDA Toolkit** ≥ 13.3
- **C++17** compiler (GCC ≥ 11 or Clang ≥ 14)
- **CMake** ≥ 3.22
- **Python** ≥ 3.10 (for benchmarks, matplotlib, numpy)

### Build

```bash
cd standard_robot_cuda_ik
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc)
```

This produces 10 executables:

| Executable | Ablation Level | Features |
|:---|:---|:---|
| `standard_robot_cuda_runner_A0` | B0 (Baseline) | Global memory, no padding, fixed lambda |
| `standard_robot_cuda_runner_A1` | B1 | + Constant memory |
| `standard_robot_cuda_runner_A2` | B2 | + PaddedMat6x8 (bank-conflict elimination) |
| `standard_robot_cuda_runner_A3` | — | + Register LDLT |
| `standard_robot_cuda_runner_A4` | — | + Kernel fusion |
| `standard_robot_cuda_runner_A5` | B3 | + Adaptive damping ★ |
| `standard_robot_cuda_runner_A6` | B4 | Full FP64: A5 + step clamp + branch alignment |
| `standard_robot_cuda_runner_A7` | B5 | **Mixed precision (FP32+FP64)** ★★ |
| `standard_robot_cuda_runner_A8` | B6 | A7 + CUDA Graph |
| `standard_robot_cuda_runner` | B6 alias | Default (links to A8) |

### Run Benchmark

```bash
# Single-solver benchmark
cd standard_robot_cuda_ik
./build/standard_robot_cuda_runner_A7 \
    --targets data/targets/ur10_seed42_N1000.csv \
    --seeds data/seeds/ur10_seed42_zero_seed_N1000.json \
    --repeat 30

# Full cross-solver comparison
python3 benchmark/run_all.py \
    --robot ur10 --seed 42 --N 1000 --repeat 30
```

---

## 4. Key Performance Results

All results use the **Medium convergence threshold** (position 10mm, orientation 5°) as the primary benchmark, 30 repeats, zero_seed strategy, on an NVIDIA GeForce RTX 4070 Laptop GPU.

### 4.1 B5 vs. cuRobo — Main Comparison

| Batch Size N | B5 Throughput | cuRobo Throughput | B5 Speedup | B5 GPU Time | cuRobo GPU Time |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | **112,414 t/s** | 3,118 t/s | **36.1×** | 0.89 ms | 32.1 ms |
| 500 | **158,251 t/s** | 15,844 t/s | **10.0×** | 3.16 ms | 31.6 ms |
| 1000 | **148,412 t/s** | 31,611 t/s | **4.7×** | 6.74 ms | 31.6 ms |
| 5000 | **168,683 t/s** | 155,059 t/s | **1.09×** | 29.64 ms | 32.2 ms |

**Key observations:**
- **Small-batch dominance**: At N=100, B5 delivers 36× more throughput than cuRobo. This is critical for interactive applications (e.g., real-time trajectory refinement with 100 waypoints).
- **Linear GPU scaling**: B5's GPU time vs. batch size achieves **R² > 0.999** — the per-target cost is nearly constant, with throughput stable at 148k–174k t/s across all batch sizes (±8%).
- **cuRobo batch oscillation**: At N=4000, 7000, 9000, and 10000, cuRobo intermittently enters a degraded ~230ms/batch mode (7× normal), likely due to PyTorch CUDA Caching Allocator fragmentation — a problem B5's fixed per-target memory model completely avoids.

### 4.2 Ablation — Progressive Optimization Impact

| Level | N=100 | N=500 | N=5000 | Key Optimization |
|:---:|:---:|:---:|:---:|:---|
| B0 (FP64 baseline) | 7,589 t/s (Conv 83%) | 9,896 t/s (Conv 52%) | 12,507 t/s (Conv 56%) | None — global memory, no padding, fixed lambda |
| B3 (+Adaptive Damping) | 51,361 t/s (**6.8×**) | 62,384 t/s (**6.3×**) | 66,050 t/s (**5.3×**) | Piecewise-linear damping with stagnation recovery |
| B5 (+Mixed Precision) | 113,097 t/s (**14.9×**) | 155,071 t/s (**15.7×**) | 164,207 t/s (**13.1×**) | FP32 FK/Jacobian/Hessian + FP64 LDLT |

**The critical insight**: Without adaptive damping, B0's convergence rate collapses to 52–83%. With adaptive damping (B3), convergence returns to 100% and throughput improves 4–6×. Mixed precision (B5) then adds 2.2–2.5× on top, exploiting Ada Lovelace's 64:1 FP32:FP64 throughput ratio.

---

## 5. CUDA Kernel Design

### 5.1 Block-Warp-Thread Mapping

```
Grid:  (N, 1, 1)     ← One block per target pose
Block: (128, 1, 1)   ← 4 warps × 32 lanes
```

Each block independently solves the IK problem for one target pose. The 4 warps are organized as a pipelined computation within each DLS iteration:

```
Warp 0 (lanes 0–31)     ──► Forward Kinematics
Warp 1 (lanes 32–63)    ──► Numerical Jacobian (6 columns, parallel)
Warp 2 (lanes 64–95)    ──► Hessian Construction (JᵀW²J + λI)
Warp 3 (lanes 96–127)   ──► LDLᵀ Solve + Convergence
```

| Warp | Task | Active Lanes | Key Operations |
|:---:|:---|:---:|:---|
| 0 | FK chain | 1 | 6-segment Rodrigues product, pose error |
| 1 | Jacobian | 6 | 12 FK calls via central differencing (δ=1e-6 rad) |
| 2 | Hessian + Gradient | 36 | Each thread computes one (r,c) element of JᵀW²J |
| 3 | LDLᵀ + Convergence | 6 | Custom 6×6 LDLᵀ (~93 FP64 ops), step clamp (0.45 rad), stagnation check |

**14 `__syncthreads()` barriers** per iteration ensure data consistency between warps. Nsight Compute reveals that **72.7% of warp stall cycles are barrier waits** — the fastest warp must wait for the slowest at each sync point. This is the primary performance bottleneck, not computation or memory bandwidth.

### 5.2 PaddedMat6x8 — Shared Memory Bank-Conflict Elimination

All intermediate matrices (Jacobian, Hessian, transforms) reside in shared memory using a **6×8 column-padded layout**:

```cpp
struct PaddedMat6x8 {
    double data[6 * 8];  // 48 doubles = 384 bytes (vs 36 for dense)
    __device__ double& operator()(int row, int col) {
        return data[row * 8 + col];  // stride=8, zero instruction overhead
    }
};
```

**Mechanism**: On Ada Lovelace, shared memory has 32 banks × 4 bytes. A `double` (8 bytes) spans 2 consecutive banks. With stride=8 (16 banks per row), each row maps to a non-overlapping bank set modulo 32 — rows 0 and 2 share banks 0–15, rows 1 and 3 share banks 16–31. This guarantees **zero bank conflicts** for all access patterns.

**Verified by Nsight Compute**: `l1tex__data_bank_conflicts_shared_pipe_lsu.sum = 0`. Disabling padding (Ablation Level 0, stride=6) increases memory throughput 151%, drops L2 hit rate by 8.22 percentage points, and reduces overall throughput by ~2%.

### 5.3 Mixed Precision (B5) — FP32 + FP64 Hybrid

Ada Lovelace's FP64:FP32 throughput ratio is **1:64**: the RTX 4070 has only 48 FP64 cores vs. 3,072 FP32 cores per SM. Mixed precision moves ~90% of the arithmetic to FP32 while preserving double precision where it matters:

| Component | Precision | Rationale |
|:---|:---:|:---|
| Forward Kinematics | **FP32** | ~90% of total FLOP; 64× throughput vs. FP64 |
| Numerical Jacobian (6 cols × 2 FK) | **FP32** | Dominates iteration time |
| Hessian JᵀW²J (36 elements) | **FP32** | 216 multiply-adds per iteration |
| Pose error | FP64 | Critical for convergence tolerance (10mm/5°) |
| Joint state q, dq | FP64 | Must be exact for joint-limit enforcement |
| LDLᵀ decomposition | **FP64** | Numerical stability of the linear solve |
| Damping parameter λ | FP64 | Prevents underflow in adaptive damping |

**The critical conversion point** is in the LDLᵀ assembly: `(double)H(r,c)` casts the FP32 Hessian to FP64 just before the solve. All downstream computation (joint update, convergence check) remains in FP64. This achieves 2.5× throughput over B4 (all-FP64) with no measurable convergence degradation (ConvRate: 1.000 → 0.998).

### 5.4 Adaptive Damping

Fixed damping (B0): `λ ≡ 2e-3` — too aggressive (oscillates) or too conservative (slow convergence). Adaptive damping (B3+) uses a piecewise-linear distance-proportional schedule:

```
If pos_err > 0.1 (far region):
    λ = min( max(λ_base, λ_far × pos_err / λ_scale), 3 × λ_far )
Else (near region):
    λ = λ_floor + λ_base × pos_err / λ_scale

Stagnation boost (≥5 consecutive non-improving iterations):
    λ *= 1.0 + 0.3 × (stagnation - 5)
    λ = min(λ, 0.5)  // hard cap
```

This provides strong damping when far from the solution (preventing divergence at singularities) and light damping when close (enabling fast terminal convergence). Stagnation recovery progressively increases damping to escape local minima, up to a hard cap of 0.5. If stagnation persists to 25 iterations, the solver restores the best-known solution and terminates.

**Impact**: Adaptive damping alone (B0 → B3) provides **4–6× throughput improvement** by reducing average iterations from ~31–80 to ~13–15, and restores convergence rate from 52–83% to 100%.

### 5.5 Constant Memory Broadcasting

1,384 bytes of kinematic parameters are stored in `__constant__` memory and broadcast to all threads:

| Symbol | Size | Content |
|:---|:---:|:---|
| `c_segment_origins[96]` | 768 B | 6 segment × 4×4 origin matrices |
| `c_segment_axes[18]` | 144 B | 6 rotation axis directions |
| `c_q_index[6]` | 24 B | Joint-to-segment mapping |
| `c_T_wrist3_to_tcp[16]` | 128 B | TCP tool transform |
| `c_joint_limits[12]` | 96 B | Joint position limits |
| `c_weight_schedule[24]` | 192 B | 4 weight levels × 6 DOF weights |
| `c_lambda_params[4]` | 32 B | Adaptive damping parameters |

All threads in a warp read the same kinematic parameter simultaneously — constant memory broadcasts with ~1 cycle latency (vs. ~400 cycles for global memory). Without constant memory, kinematic lookups would generate ~5.2 GB of additional global memory traffic per 273-target batch.

### 5.6 Zero Register Spill

The `ik_batch_solve` kernel uses **96 registers per thread** with **zero bytes of local memory spill**, verified via `ptxas` verbose output and Nsight Compute. All intermediate variables — FK results (8 registers), Jacobian columns (6), Hessian accumulators (4), error/gradient/step vectors (18), iteration state (10), and device-function locals — fit entirely in the register file.

At 128 threads/block × 96 registers = 12,288 registers/block, each SM can host 5 concurrent blocks (occupancy: 41.7% of warp slots). For this **compute-bound** kernel (arithmetic intensity ~193 FLOP/Byte vs. ridge point 0.93 FLOP/Byte, ~207× above the ridge), the compiler's decision to maximize register allocation (zero spill) is more beneficial than reducing register count to increase occupancy.

### 5.7 Custom 6×6 LDLᵀ — No cuBLAS

A 6×6 linear system solve via cuBLAS would require:
- A cuBLAS handle creation and context switch (~5–10 μs launch overhead)
- Data marshaling from shared memory → global memory → cuBLAS → shared memory
- cuBLAS internal dispatch for a problem too small to amortize the overhead

Instead, a hand-written register-resident LDLᵀ decomposition completes the 6×6 solve in **~93 FP64 operations (~0.1 μs)** with zero memory traffic:

1. **Decomposition** (57 ops): Compute L and D in-place from the upper triangle of H
2. **Forward substitution** (15 ops): Solve L·y = g
3. **Diagonal scaling** (6 ops): z = D⁻¹·y
4. **Backward substitution** (15 ops): Solve Lᵀ·dq = z

No `sqrt` required (unlike Cholesky LLᵀ), providing better numerical stability when H is near-singular.

### 5.8 Single-Kernel Encapsulation

All DLS iterations execute within **one CUDA kernel launch**. The CPU submits the batch, calls `cudaDeviceSynchronize()`, and receives the results — no CPU-GPU round trips during iteration. This eliminates:

- Kernel launch overhead: ~5–10 μs per launch on consumer GPUs, × 7 iterations = 35–70 μs per target
- CUDA API dispatch latency
- Device synchronize overhead between launches

For cuRobo and PyRoki, the equivalent would be hundreds or thousands of kernel launches per batch, each incurring Python→CUDA binding overhead and GPU stream synchronization.

---

## 6. CUDA Framework Advantages — Structural Comparison

### 6.1 Why Raw CUDA Outperforms Framework-Based Solvers

| Aspect | CUDA B5 (This Work) | cuRobo (NVIDIA) | PyRoki (JAX) |
|:---|:---|:---|:---|
| **Kernel launches per batch** | **1** | Hundreds (many small kernels) | Thousands (JIT + XLA) |
| **CPU-GPU sync points** | **0** (single sync at end) | Many (per-kernel launch) | Many (JAX dispatch) |
| **Framework layer** | **None** (raw CUDA C++) | cuda.core + Warp + PyTorch | JAX JIT + XLA + Python |
| **Precision model** | Mixed FP32+FP64 | FP32 | FP64 |
| **Matrix solve** | Custom register LDLᵀ (~0.1 μs) | cuBLAS (library call, ~5 μs) | JAX linear algebra |
| **Shared memory optimization** | PaddedMat6x8 (0 bank conflicts) | Not disclosed | N/A (XLA-managed) |
| **Iteration encapsulation** | Single kernel | Multi-kernel | Multi-kernel |
| **Memory allocation** | Fixed per-target (cudaMalloc) | PyTorch CUDA Caching Allocator | JAX memory pool |
| **Per-target cost scaling** | Linear (R² > 0.999) | Oscillating (R² ~0.7) | Not measured |

### 6.2 The Performance Gap Explained

**Kernel launch count dominates at small N.** At N=100, cuRobo launches hundreds of kernels per batch while B5 launches exactly 1. Each kernel launch costs ~5–10 μs on a consumer GPU driver stack. At N=100 with cuRobo's normal 32ms batch time, launch overhead alone can account for ~10% of execution time.

**Framework overhead is a fixed tax.** cuRobo's Python→C++→CUDA call chain and PyTorch tensor management add ~5–15ms per batch regardless of batch size. At N=5000, this is amortized. At N=100, it dominates — B5's 0.89ms includes zero framework tax.

**Custom LDLᵀ beats library calls for small matrices.** A 6×6 LDLᵀ is a trivially small problem. cuBLAS's internal dispatch, handle management, and memory marshaling overhead exceed the actual solve time by 50×. Register-resident LDLᵀ with zero memory traffic is the optimal strategy for matrices smaller than ~16×16.

**Fixed memory model avoids allocator pathologies.** B5 allocates exactly `N × sizeof(Target) + N × sizeof(Result)` once via `cudaMalloc`. cuRobo relies on PyTorch's CUDA Caching Allocator, which fragments over repeated allocations, intermittently triggering expensive defragmentation passes. This is the likely mechanism behind cuRobo's observed ~230ms degraded mode — the allocator enters a pathological state at specific batch sizes, causing 7× slowdown.

### 6.3 cuRobo Batch Oscillation

At N=4000, 7000, 9000, and 10000, cuRobo intermittently enters a **~230ms/batch degraded mode** (vs. normal ~32ms/batch):

| N | Normal Mode | Degraded Mode | Frequency |
|:---:|:---:|:---:|:---:|
| 4000 | ~32 ms | ~230 ms | Intermittent |
| 5000 | ~32 ms | — | Normal |
| 7000 | ~32 ms | ~230 ms | Intermittent |
| 8000 | ~32 ms | — | Normal |
| 10000 | ~32 ms | ~230 ms | Intermittent |

The non-monotonic pattern (N=5000 normal but 4000 degraded, 8000 normal but 7000 degraded) is characteristic of allocator fragmentation — not a compute or bandwidth limitation. B5's fixed per-target memory model is structurally immune to this failure mode.

---

## 7. Experiment Methodology

### 7.1 Convergence Thresholds

| Level | Position Tolerance | Orientation Tolerance | Usage |
|:---|:---|:---|:---|
| **Loose** | 30 mm | 10° (0.1745 rad) | cuRobo default; early experiments |
| **Medium** ★ | **10 mm** | **5° (0.0873 rad)** | **Primary benchmark** |
| **Strict** | 5 mm | 1° (0.0175 rad) | Precision-critical applications |

The Medium threshold is the main benchmark because it differentiates solver quality: B0's convergence deficit (52–83% at Medium, vs. 80–100% at Loose) is invisible at the old 30mm/10° threshold.

### 7.2 Ablation Levels (B-Series Naming)

| Paper Name | Executable | Optimization | Key Metric |
|:---|:---|:---|:---|
| B0 | A0 | FP64 baseline (no optimization) | Convergence: 52–83% |
| B1 | A1 | + Constant memory | +3% throughput |
| B2 | A2 | + PaddedMat6x8 | 0 bank conflicts |
| — | A3 | + Register LDLᵀ | — |
| — | A4 | + Kernel fusion | — |
| B3 ★ | A5 | + Adaptive damping | **+400–600% throughput** |
| B4 | A6 | Full FP64 (A5 + step clamp + branch align) | Reference baseline |
| B5 ★★ | A7 | + Mixed precision (FP32+FP64) | **+220–250% throughput** |
| B6 | A8 | + CUDA Graph | Marginal (~3.7%) |

### 7.3 Reproducibility

- **Target generation**: Deterministic PRNG (seed=42), joint angles uniformly sampled in [−π, π], FK-validated for reachability
- **Seed strategies**: `zero_seed` (all zeros, default), `home_seed` (UR10 home configuration), `random_seed` (uniform random), `near_ground_truth_seed` (ground-truth + 0.25 rad noise)
- **Repeat**: 30 independent runs per configuration; throughput reported as mean
- **Timing**: GPU end-to-end time (`cudaDeviceSynchronize` before and after kernel launch, excluding host-side target/seed preparation)
- **Hardware**: NVIDIA GeForce RTX 4070 Laptop GPU (4,608 CUDA cores, 8 GB GDDR6), CUDA 13.3

---

## 8. Data and Figures

All experiment data is centralized in [`standard_robot_cuda_ik/data/`](standard_robot_cuda_ik/data/). See the [data README](standard_robot_cuda_ik/data/README.md) for the complete data-paper mapping table, column descriptions, and experiment configuration.

| Dataset | Files | Contents |
|:---|:---:|:---|
| Targets | 36 | UR10 seed=42 end-effector poses, N=100→10000, .bin/.csv/.json |
| Seeds | 96 | 4 strategies × 12 batch sizes × 2 formats (.bin/.json) |
| Results | 11 CSVs | Main comparison, ablation, threshold scan, full range, CPU baseline, seed strategy, Panda 7-DOF |
| Profiling | 5 .ncu-rep | Nsight Compute: B4 FP64 N=100 (×3), B3 FP64 N=5000, B5 Mixed N=100 |
| Figures | 5 PNGs | Throughput comparison, speedup, ablation, convergence, iterations |

Regenerate figures:
```bash
cd standard_robot_cuda_ik/data/figures
python3 plot_all_figures.py
```

---

## 9. Documentation and Paper

- **Paper draft**: [`docs/paper/`](docs/paper/) — 8 chapters in Markdown
- **Patent disclosure**: [`docs/patent/`](docs/patent/) — Full technical disclosure document
- **Experiment logs**: [`docs/logs/`](docs/logs/) — 10 authoritative experiment reports
- **Project overview**: [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md)
- **CUDA technical docs**: [`cuda_low_level_optimization/docs/`](cuda_low_level_optimization/docs/) — 27 in-depth design documents covering memory hierarchy, kernel execution model, performance analysis, and GPU collision detection

---

## 10. Adding Your Own Solver

The benchmark framework is designed for fair cross-solver comparison. To add a new solver:

1. Create `bench_your_solver.py` following the pattern of [`bench_cuda_6dof.py`](standard_robot_cuda_ik/benchmark/bench_cuda_6dof.py)
2. Load targets and seeds from `data/targets/` and `data/seeds/`
3. Use `common.load_urdf()` and `common.check_convergence()` for consistent criteria
4. Output results in the standard JSON schema (see [`compare_results.py`](standard_robot_cuda_ik/benchmark/compare_results.py))
5. Run `run_all.py --solver your_solver` for unified comparison

All solvers are evaluated using identical targets, seeds, convergence thresholds, and repeat counts — ensuring apples-to-apples comparison.

---

## 11. Legacy Project — Gen 1 Proof-of-Concept

The [`cuda_low_level_optimization/`](cuda_low_level_optimization/) directory contains the original CUDA IK solver for a custom UR10 + shovel bucket assembly (non-standard robot). This was the proof-of-concept that validated the single-kernel approach, achieving **~960× speedup** over CPU KDL (273 targets: 6.2s CPU → 6.4ms GPU).

Its core CUDA techniques (warp parallelism, PaddedMat6x8, adaptive damping, register LDLᵀ) were migrated and generalized into `standard_robot_cuda_ik/`. The legacy code is preserved for reference and its 27 technical docs remain the most comprehensive documentation of the CUDA kernel internals.

---

## 12. License

This project is for research and benchmarking purposes. Source code originating from `assembly_rtfg_cuda` retains its original copyright.

---

*Last updated: 2026-06-12 | Target GPU: NVIDIA Ada Lovelace (sm_89) | CUDA 13.3*
