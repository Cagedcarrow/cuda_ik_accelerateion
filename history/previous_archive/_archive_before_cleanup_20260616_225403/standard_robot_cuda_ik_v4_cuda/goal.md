# standard_robot_cuda_ik_v4_cuda 最终 CUDA Port 执行计划

## 0. 总目标

当前工作目录：

```bash
/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik_v4_cuda
```

本项目目标：

```text
将已经冻结的 V4-Final-K16 Python 原型移植为 CUDA 实现，并生成足够支撑论文写作的最终实验指标。
```

V4-Final-K16 固定定义：

```text
Analytical Jacobian
+ Levenberg-Marquardt Solver
+ Sobol Seed Bank, K=16
+ Limit Barrier, w_limit=0.03, margin=0.087 rad
+ Smoothness Candidate Reranking
```

本阶段不是继续算法探索，而是：

```text
CUDA correctness
CUDA benchmark
cuRobo comparison
Nsight profiling
paper readiness decision
```

最终必须回答：

```text
1. CUDA 实现是否正确复现 Python V4-Final-K16？
2. CUDA 实现是否保持 V4 的 Strict SR、pos_p95、near_limit 指标？
3. CUDA 相比 Python 是否有明显加速？
4. CUDA-V4 和 cuRobo-Graph 相比，优势区间和劣势区间分别是什么？
5. 当前结果是否足够开始写论文？
```

---

# 1. 项目现状与文件结构

当前项目内已有：

```text
standard_robot_cuda_ik_v4_cuda/
├── PROJECT.md
├── CMakeLists.txt
├── src/cuda/
│   ├── cuda_utilities.cuh
│   ├── cuda_ik_6dof.cu
│   ├── cuda_benchmark_runner.cu
│   └── cuda_memory.cu
├── include/standard_robot_cuda_ik/
│   ├── cuda_ik_6dof.h
│   ├── cuda_memory.h
│   ├── cuda_collision.h
│   └── generated/
│       └── ur10_model_constants.h
├── data/
│   ├── targets/
│   │   ├── v4_targets_N200_seed42.npy
│   │   └── v4_targets_N1000_seed42.npy
│   ├── seed_banks/
│   │   ├── sobol_K16_N200_bank00.npy
│   │   ├── sobol_K32_N200_bank00.npy
│   │   ├── sobol_K16_N1000_bank00.npy
│   │   └── sobol_K32_N1000_bank00.npy
│   └── results/
│       ├── v4_limit_weight_sweep.csv
│       └── v4_m2_smooth_rerank_results.csv
├── experiments/
│   ├── run_v4_m0.py
│   ├── run_v4_m1_m2.py
│   └── run_v4_finalize.py
├── docs/
│   ├── v4_finalization_report.md
│   ├── v4_m0_results.md
│   └── v4_m1_limit_smooth_results.md
└── benchmark/
```

本计划基于现有结构执行，不要另起一套完全不同的项目。

---

# 2. 范围控制

## 2.1 必须做

本阶段必须完成：

```text
1. 读取 PROJECT.md，确认 V4-Final-K16 定义。
2. 编译现有 V1 CUDA 项目，确认基础工程可构建。
3. 实现或重写 FK with frames。
4. 实现解析 Jacobian。
5. 实现 LM single-seed CUDA kernel。
6. 实现 Sobol-K16 multi-seed CUDA kernel。
7. 实现 per-target best candidate selection kernel。
8. 集成 Limit Barrier, w_limit=0.03。
9. 实现 IK-only static batch benchmark。
10. 实现 trajectory smoothness rerank benchmark。
11. 实现 cuRobo 对标脚本。
12. 输出 correctness、performance、cuRobo comparison、Nsight、paper readiness 报告。
```

## 2.2 暂不做

本阶段禁止：

```text
1. 不加碰撞检测。
2. 不做完整 Motion Generation。
3. 不做 V5。
4. 不重新搜索 w_limit。
5. 不重新搜索 K。
6. 不把 K32 作为首要目标。
7. 不修改 V4-Final-K16 的算法定义。
8. 不为了追求速度擅自改变 LM 数学逻辑。
9. 不只输出日志，必须输出 CSV 和 Markdown 报告。
```

K32 可以作为可选扩展，但必须在 K16 全流程通过后再做。

---

# 3. 成功标准

## 3.1 Correctness 通过标准

CUDA 版本必须满足：

| 项目                            |       标准 |
| ----------------------------- | -------: |
| FK 4×4 matrix max abs diff    |   < 1e-5 |
| FK joint positions p_i diff   | < 1e-5 m |
| FK joint axes z_i diff        |   < 1e-5 |
| Jacobian rel Frobenius error  |   < 1e-4 |
| Limit loss diff               |   < 1e-8 |
| N=100 Strict SR 与 Python 差异   |   ≤ 2 pp |
| N=100 pos_p95_all 与 Python 差异 |   ≤ 2 mm |
| N=100 near_limit 与 Python 差异  |   ≤ 2 pp |
| no NaN / Inf                  |       必须 |

## 3.2 Static Batch Benchmark 通过标准

N=1000, K=16 时：

| 项目                |          标准 |
| ----------------- | ----------: |
| Strict SR         |       ≥ 93% |
| Medium SR         | ≥ Strict SR |
| Loose SR          | ≥ Medium SR |
| pos_p95_all       |      ≤ 8 mm |
| pos_p95_suc       |      ≤ 5 mm |
| near_limit        |        ≤ 4% |
| joint violation   |           0 |
| Python speedup    |       ≥ 20× |
| GPU stream timing |  可重复，std 合理 |

## 3.3 Smoothness 通过标准

Trajectory reranking 相比 independent candidate selection：

| 项目              |                      标准 |
| --------------- | ----------------------: |
| Strict SR 下降    |                  ≤ 2 pp |
| mean_delta_q 下降 |                   ≥ 20% |
| p95_delta_q 下降  |                   ≥ 15% |
| pos_p95_all     |                   不严重恶化 |
| monotonic       | Loose ≥ Medium ≥ Strict |

## 3.4 cuRobo 对比最低要求

必须完成：

```text
N = 100, 500, 1000, 5000
methods:
- CUDA-V4-Final-K16
- cuRobo-Graph
```

可选：

```text
- cuRobo-NoGraph
- CUDA-V3-Sobol-K16 without limit
- Python-V4-Final-K16
```

如果 CUDA-V4 没有全面超过 cuRobo，仍可开始论文，但必须明确写出边界。

---

# 4. 输出文件要求

请创建或更新以下文件：

```text
docs/
├── cuda_port_execution_log.md
├── cuda_correctness_report.md
├── cuda_static_benchmark_report.md
├── cuda_trajectory_rerank_report.md
├── cuda_curobo_comparison_report.md
├── nsight_summary.md
└── final_paper_readiness_report.md

data/results/
├── fk_correctness.csv
├── jacobian_correctness.csv
├── lm_single_seed_correctness.csv
├── cuda_vs_python_N100.csv
├── cuda_v4_static_benchmark.csv
├── cuda_v4_trajectory_benchmark.csv
├── cuda_v4_curobo_compare.csv
├── nsight_summary.csv
└── final_summary.csv

logs/
├── build.log
├── correctness.log
├── static_bench.log
├── trajectory_bench.log
├── curobo_compare.log
└── nsight.log
```

如果目录不存在，先创建。

---

# 5. Phase 0：工程盘点与构建验证

## 5.1 读取项目说明

先读取：

```bash
PROJECT.md
```

确认以下信息：

```text
1. V4-Final-K16 定义；
2. 现有 V1 CUDA 文件位置；
3. UR10 generated constants 位置；
4. data/targets 和 data/seed_banks 是否存在；
5. experiments/ Python 原型是否可运行；
6. benchmark/ cuRobo 脚本是否存在。
```

## 5.2 检查文件

执行：

```bash
cd /mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik_v4_cuda
find . -maxdepth 4 -type f | sort > logs/file_inventory.txt
```

重点检查：

```bash
ls -lh src/cuda/
ls -lh include/standard_robot_cuda_ik/generated/
ls -lh data/targets/
ls -lh data/seed_banks/
ls -lh experiments/
```

## 5.3 编译现有项目

执行：

```bash
mkdir -p build logs docs data/results
cmake -S . -B build 2>&1 | tee logs/cmake_configure.log
cmake --build build -j$(nproc) 2>&1 | tee logs/build.log
```

如果编译失败：

1. 不要直接重写全部工程；
2. 先修 CMake；
3. 保留 V1 可复用模块，如 `cuda_memory.cu`；
4. 将失败原因写入 `docs/cuda_port_execution_log.md`。

---

# 6. Phase 1：数据导出与 Python Reference 固定

## 6.1 为什么需要数据导出

当前数据是 `.npy`，CUDA/C++ 直接读取较麻烦。因此先新增 Python 脚本，把 `.npy` 转成 C++ 容易读取的 `.bin` 和 `.csv`。

新增脚本：

```text
scripts/export_v4_cuda_inputs.py
```

如果没有 `scripts/`，创建：

```bash
mkdir -p scripts
```

## 6.2 导出输入

读取：

```text
data/targets/v4_targets_N200_seed42.npy
data/targets/v4_targets_N1000_seed42.npy
data/seed_banks/sobol_K16_N200_bank00.npy
data/seed_banks/sobol_K16_N1000_bank00.npy
```

输出：

```text
data/cuda_inputs/targets_N200_T4x4_f64.bin
data/cuda_inputs/targets_N1000_T4x4_f64.bin
data/cuda_inputs/seeds_N200_K16_q_f64.bin
data/cuda_inputs/seeds_N1000_K16_q_f64.bin
```

二进制格式：

### targets

```text
int32 N
int32 cols = 16
double data[N][16]
```

### seeds

```text
int32 N
int32 K = 16
int32 dof = 6
double data[N][K][6]
```

如果 `.npy` 中 target 不是 4×4 矩阵而是 position/quaternion 或其他格式，先在报告中说明实际格式，并转换为 CUDA kernel 需要的统一格式。

## 6.3 生成 Python Reference

使用现有 Python 原型脚本，生成参考输出：

```text
data/results/python_ref_v4_N200_K16.csv
data/results/python_ref_v4_N1000_K16.csv
```

字段：

```text
target_id
best_seed_id
q0 q1 q2 q3 q4 q5 q6
pos_err_mm
rot_err_deg
loss
success_loose
success_medium
success_strict
near_limit
limit_score
iters
```

必须确认：

```text
N=1000 Strict SR ≈ 95%
pos_p95_all ≈ 5.03 mm
near_limit ≈ 2.4%
```

如果 Python reference 无法复现 PROJECT.md 中的数字，先停止 CUDA Port，修复 reference 复现问题。

---

# 7. Phase 2：FK with Frames CUDA 实现

## 7.1 修改文件

主要修改：

```text
src/cuda/cuda_utilities.cuh
```

或新增：

```text
src/cuda/cuda_fk_v4.cuh
```

如果当前工程已有 FK 函数，不要删除旧函数；新增 V4 函数：

```cpp
__device__ void fk_with_frames_v4(
    const double q[6],
    double T_ee[16],
    double p_joint[6][3],
    double z_joint[6][3]
);
```

## 7.2 实现要求

根据 `ur10_model_constants.h` 中的 UR10 常量实现：

```text
T = I4
for i = 0..5:
    T = T * origin[i]
    p_joint[i] = T.translation
    z_joint[i] = R_world * axis_local[i]
    T = T * Rodrigues(axis_local[i], q[i])
T_ee = T * T_tool
```

注意：

```text
1. origin[i] 的定义必须和 Python 原型一致。
2. axis[i] 的局部/世界转换必须和 Python 一致。
3. tool transform 必须一致。
4. 矩阵行主序/列主序必须统一。
```

## 7.3 FK correctness test

新增测试入口：

```text
src/cuda/test_fk_correctness.cu
```

或在 benchmark runner 中加入模式：

```bash
./build/cuda_benchmark_runner --mode fk_check --N 100
```

输出：

```text
data/results/fk_correctness.csv
docs/cuda_correctness_report.md
```

CSV 字段：

```text
sample_id
max_T_abs_diff
pos_diff_m
rot_diff_rad
max_p_joint_diff
max_z_joint_diff
pass
```

通过标准：

```text
max_T_abs_diff < 1e-5
max_p_joint_diff < 1e-5
max_z_joint_diff < 1e-5
```

如果失败，优先排查：

```text
1. 矩阵行列主序；
2. origin 乘法顺序；
3. axis 局部转世界；
4. tool transform；
5. 角度单位。
```

---

# 8. Phase 3：Analytical Jacobian CUDA 实现

## 8.1 修改文件

在：

```text
src/cuda/cuda_utilities.cuh
```

新增：

```cpp
__device__ void analytical_jacobian_v4(
    const double p_ee[3],
    const double p_joint[6][3],
    const double z_joint[6][3],
    double J[6][8]
);
```

使用 padded 6×8 存储，保持 V1 小矩阵优化习惯。

## 8.2 公式

```text
J[0:3, i] = z_i × (p_ee - p_i)
J[3:6, i] = z_i
```

注意：

```text
1. J 的上三行为线速度；
2. J 的下三行为角速度；
3. 单位必须和 pose error 一致；
4. 不要把旋转行和位置行反过来。
```

## 8.3 Jacobian correctness test

新增模式：

```bash
./build/cuda_benchmark_runner --mode jacobian_check --N 100
```

输出：

```text
data/results/jacobian_correctness.csv
```

字段：

```text
sample_id
max_abs_diff
fro_rel_error
pass
```

通过标准：

```text
fro_rel_error < 1e-4
max_abs_diff < 1e-4
```

如果失败，排查：

```text
1. FK frames 是否已通过；
2. z_i 是否是世界坐标；
3. p_ee 是否来自 tool 后 TCP；
4. Python Jacobian 是否使用同一 TCP；
5. J 行顺序是否一致。
```

---

# 9. Phase 4：6×6 求解器与 LM 单 seed

## 9.1 复用 V1 求解器

检查 V1 中是否已有：

```text
LDLT
Cholesky
solve_6x6
```

如果可用，优先复用；如果 V1 求解器只适合 DLS，也可以新增：

```cpp
__device__ bool solve_6x6_cholesky_v4(
    double H[6][8],
    double g[8],
    double dq[8]
);
```

## 9.2 LM 单 seed kernel

新增 kernel：

```cpp
__global__ void ik_lm_single_seed_kernel_v4(
    const TargetPose* targets,
    const double* seeds,
    CandidateResult* candidates,
    int N,
    SolverConfig cfg
);
```

先只跑：

```text
N targets
K = 1
```

不要直接上 K=16。

## 9.3 LM 逻辑必须与 Python 一致

LM 迭代：

```text
for iter in max_iter:
    FK + Jacobian
    pose error e
    H = J^T J + λ I
    g = J^T e
    加入 limit barrier 梯度或惩罚项
    solve H dq = -g
    clamp dq, max step = 0.35 rad
    q_new = q + dq
    compute loss_new
    if loss_new < loss_old:
        q = q_new
        λ *= 0.5
    else:
        λ *= 2.0
    λ = clamp(λ, 1e-6, 0.5)
```

注意 PROJECT.md 写的是“自适应 λ，无 rejection”。这里必须核对 Python 原型实际实现：

```text
如果 Python 中 loss_new >= loss_old 时不接受 q_new，只更新 λ，则 CUDA 也这样。
如果 Python 中仍接受 q_new，只是调 λ，则 CUDA 也这样。
```

以 Python 原型为准，不要按字面猜。

## 9.4 Pose error 一致性

必须确认 Python 中 pose error 定义：

```text
position error: p_current - p_target 或 p_target - p_current
rotation error: 几何 Jacobian 对应的 orientation residual
```

CUDA 必须保持符号一致。

如果 LM 不收敛，优先检查：

```text
1. e 的符号；
2. H dq = -g 还是 H dq = g；
3. rotation error 与 geometric Jacobian 是否一致；
4. position/rotation 权重；
5. rad / deg 单位。
```

## 9.5 单 seed correctness

测试：

```bash
./build/cuda_benchmark_runner --mode lm_single_check --N 10
```

输出：

```text
data/results/lm_single_seed_correctness.csv
```

字段：

```text
target_id
seed_id
python_pos_err_mm
cuda_pos_err_mm
python_rot_err_deg
cuda_rot_err_deg
python_loss
cuda_loss
pos_diff_mm
rot_diff_deg
pass
```

通过标准：

```text
pos_diff p95 < 1 mm
rot_diff p95 < 0.2 deg
无 NaN
```

---

# 10. Phase 5：Multi-Seed Kernel

## 10.1 核心 kernel

重写或新增：

```text
src/cuda/cuda_ik_6dof.cu
```

新增：

```cpp
__global__ void ik_lm_multiseed_kernel_v4(
    const TargetPose* targets,
    const double* seed_bank,
    CandidateResult* candidates,
    int N,
    int K,
    SolverConfig cfg
);
```

grid 设计：

```cpp
dim3 grid(N, K);
dim3 block(128);
```

映射：

```text
blockIdx.x = target_id
blockIdx.y = seed_id
```

每个 block 输出：

```text
candidates[target_id * K + seed_id]
```

## 10.2 CandidateResult 结构体

在 header 中定义：

```cpp
struct CandidateResult {
    double q[6];
    double pos_err_mm;
    double rot_err_deg;
    double pose_cost;
    double limit_score;
    double total_loss;
    int iters;
    int success_loose;
    int success_medium;
    int success_strict;
    int near_limit;
    int valid;
};
```

注意对齐：

```text
1. 尽量 8-byte aligned；
2. 如果结构太大导致全局内存写多，可先保正确，再优化；
3. 后期可分离 SoA 格式，但第一版不用。
```

## 10.3 Limit Barrier 集成

固定：

```text
w_limit = 0.03
margin = 0.087 rad
```

near-limit 判定：

```text
q_j < q_min_j + margin
or
q_j > q_max_j - margin
```

Limit score 记录：

```text
sum over joints of max(0, margin - distance_to_limit)^2
```

梯度加入方式必须与 Python 一致。如果 Python 只是加入 loss，而不是加入梯度，则 CUDA 也要复现 Python 的实际行为。

## 10.4 N=100 K16 correctness

运行：

```bash
./build/cuda_benchmark_runner --mode v4_check --N 100 --K 16
```

输出：

```text
data/results/cuda_vs_python_N100.csv
```

通过标准：

```text
Strict SR 差异 ≤ 2 pp
pos_p95_all 差异 ≤ 2 mm
near_limit 差异 ≤ 2 pp
monotonic = true
```

未通过则不要继续 benchmark。

---

# 11. Phase 6：Best Candidate Selection Kernel

## 11.1 IK-only selection

新增：

```cpp
__global__ void select_best_per_target_kernel_v4(
    const CandidateResult* candidates,
    BestResult* best,
    int N,
    int K,
    SelectionMode mode
);
```

静态 batch IK 选择顺序：

```text
success_rank → near_limit_flag → pose_cost
```

success_rank：

```text
0 = strict success
1 = medium success
2 = loose success
3 = failed
```

near_limit_flag：

```text
0 = not near limit
1 = near limit
```

pose_cost：

```text
pos_err_m^2 + rot_err_rad^2
```

排序 key：

```text
(success_rank, near_limit_flag, pose_cost)
```

不要把 pose_cost 放在 success_rank 前面，否则会选到不满足 Strict 的低误差但失败候选。

## 11.2 输出 BestResult

```cpp
struct BestResult {
    double q[6];
    int best_seed_id;
    double pos_err_mm;
    double rot_err_deg;
    double pose_cost;
    double limit_score;
    int iters;
    int success_loose;
    int success_medium;
    int success_strict;
    int near_limit;
};
```

## 11.3 Selection correctness

对 N=100：

```text
CUDA best result vs Python best result
```

best_seed_id 不强制完全一致，但：

```text
selected candidate 的 success_rank 必须一致；
pos/rot error 必须接近；
near_limit 统计必须接近。
```

---

# 12. Phase 7：N=200 / N=1000 稳定性复核

## 12.1 N=200 快速复核

运行：

```bash
./build/cuda_benchmark_runner --mode v4_static --N 200 --K 16 --repeat 5
```

输出：

```text
data/results/cuda_v4_static_N200.csv
```

预期：

```text
Strict SR 接近 Python；
pos_p95_all 不爆炸；
near_limit 接近 2.4% 量级；
monotonic = true。
```

## 12.2 N=1000 稳定性复核

运行：

```bash
./build/cuda_benchmark_runner --mode v4_static --N 1000 --K 16 --repeat 10
```

输出：

```text
data/results/cuda_v4_static_N1000.csv
```

通过标准：

```text
Strict SR ≥ 93%
pos_p95_all ≤ 8 mm
near_limit ≤ 4%
monotonic = true
no NaN/Inf
```

如果 N=1000 未通过，停止进入 cuRobo 对标，先修 correctness。

---

# 13. Phase 8：完整 Static Benchmark

## 13.1 数据准备

当前项目已有 N=200 和 N=1000。需要补齐：

```text
N=100
N=500
N=5000
```

可用原 Python target generation 逻辑生成，并保存到：

```text
data/targets/v4_targets_N100_seed42.npy
data/targets/v4_targets_N500_seed42.npy
data/targets/v4_targets_N5000_seed42.npy
data/seed_banks/sobol_K16_N100_bank00.npy
data/seed_banks/sobol_K16_N500_bank00.npy
data/seed_banks/sobol_K16_N5000_bank00.npy
```

如果时间允许，再生成：

```text
N=10000
```

但是 cuRobo 对标最低要求先做到 N=5000。

## 13.2 Benchmark 配置

运行：

```bash
./build/cuda_benchmark_runner --mode v4_static --N 100 --K 16 --warmup 10 --repeat 30
./build/cuda_benchmark_runner --mode v4_static --N 500 --K 16 --warmup 10 --repeat 30
./build/cuda_benchmark_runner --mode v4_static --N 1000 --K 16 --warmup 10 --repeat 30
./build/cuda_benchmark_runner --mode v4_static --N 5000 --K 16 --warmup 10 --repeat 30
```

可选：

```bash
./build/cuda_benchmark_runner --mode v4_static --N 10000 --K 16 --warmup 10 --repeat 30
```

## 13.3 输出 CSV

```text
data/results/cuda_v4_static_benchmark.csv
```

字段：

```text
method
N
K
warmup
repeat
gpu_stream_ms_mean
gpu_stream_ms_std
e2e_ms_mean
e2e_ms_std
raw_throughput_mean
raw_throughput_std
valid_throughput_strict
loose_sr
medium_sr
strict_sr
pos_p50_all_mm
pos_p95_all_mm
pos_p99_all_mm
pos_max_all_mm
pos_p95_suc_mm
rot_p50_all_deg
rot_p95_all_deg
rot_p95_suc_deg
near_limit_ratio
iter_mean
iter_p95
monotonic_pass
nan_count
inf_count
```

## 13.4 报告

生成：

```text
docs/cuda_static_benchmark_report.md
```

必须包含：

```text
1. CUDA-V4-Final-K16 方法定义；
2. N=100/500/1000/5000 表格；
3. 吞吐随 N 的变化；
4. Strict SR 随 N 的变化；
5. pos_p95_all 随 N 的变化；
6. near_limit 随 N 的变化；
7. 是否满足论文写作条件。
```

---

# 14. Phase 9：Trajectory Smoothness Rerank

## 14.1 目标

复现 Python 中：

```text
line mean_delta_q 下降约 47%
arc mean_delta_q 下降约 35%
local_random mean_delta_q 下降约 56%
```

CUDA 实现可以分两步：

### Step A：CUDA 生成候选，CPU 做 rerank

这是可接受的第一版。

### Step B：CUDA candidate_select 做 trajectory rerank

如果时间允许再做。

论文中必须清楚写：

```text
candidate generation 在 CUDA；
rerank 在 CPU 或 GPU；
```

如果 rerank 在 CPU，则论文不要把 smoothness rerank 作为纯 GPU kernel 贡献，只作为候选选择策略贡献。

## 14.2 轨迹数据

使用或生成：

```text
line_50
arc_50
local_random_50
```

保存：

```text
data/trajectories/
```

如果该目录不存在，创建。

## 14.3 对比方法

```text
independent:
success_rank → near_limit → pose_cost

rerank:
success_rank → near_limit → smoothness → pose_cost
```

## 14.4 输出 CSV

```text
data/results/cuda_v4_trajectory_benchmark.csv
```

字段：

```text
trajectory_type
method
waypoints
K
strict_sr
medium_sr
loose_sr
pos_p95_all_mm
pos_p95_suc_mm
rot_p95_all_deg
mean_delta_q_rad
p95_delta_q_rad
max_delta_q_rad
jump_count_linf_0p5
jump_count_l2_1p0
jerk_cost
gpu_candidate_ms
rerank_ms
e2e_ms
monotonic_pass
```

## 14.5 通过标准

```text
Strict SR 下降 ≤ 2 pp
mean_delta_q 下降 ≥ 20%
p95_delta_q 下降 ≥ 15%
pos_p95_all 不严重恶化
```

生成：

```text
docs/cuda_trajectory_rerank_report.md
```

---

# 15. Phase 10：cuRobo 对标

## 15.1 对比方法

至少：

```text
CUDA-V4-Final-K16
cuRobo-Graph
```

推荐增加：

```text
cuRobo-NoGraph
CUDA-V3-Sobol-K16, no limit
V1-CUDA-Mixed, 如果可复用
```

## 15.2 公平设置

必须统一：

```text
robot = UR10
targets = 相同目标集
collision = disabled
self_collision = disabled
thresholds = Loose / Medium / Strict 同时评价
warmup = 5 or 10
repeat = 30
```

记录：

```text
cuRobo seed 数量
cuRobo graph on/off
cuRobo 是否使用 collision
cuRobo 计时口径
```

## 15.3 运行 N

```text
N = 100, 500, 1000, 5000
```

可选：

```text
N = 10000
```

## 15.4 输出 CSV

```text
data/results/cuda_v4_curobo_compare.csv
```

字段：

```text
method
N
gpu_stream_ms_mean
gpu_stream_ms_std
e2e_ms_mean
e2e_ms_std
raw_throughput
strict_sr
medium_sr
loose_sr
valid_throughput_strict
pos_p50_all_mm
pos_p95_all_mm
pos_max_all_mm
rot_p95_all_deg
near_limit_ratio
speedup_vs_curobo_graph_stream
speedup_vs_curobo_graph_e2e
notes
```

## 15.5 结论分类

报告必须按结果自动归类：

### A：可以强调小批量优势

条件：

```text
N=100 CUDA-V4 gpu_stream_ms < cuRobo-Graph gpu_stream_ms
```

结论：

```text
CUDA-V4 在小批量低延迟场景具有优势。
```

### B：可以强调有效吞吐优势

条件：

```text
valid_throughput_strict 高于 cuRobo 某些 N
```

结论：

```text
CUDA-V4 在严格成功率约束下具有有效吞吐优势区间。
```

### C：只能强调算法质量和结构可预测性

条件：

```text
CUDA-V4 成功率好，但速度未超过 cuRobo
```

结论：

```text
本文贡献应定位为高成功率、约束感知和 CUDA-friendly 架构，而不是全面速度超越。
```

---

# 16. Phase 11：Nsight Profiling

## 16.1 选择 N

至少：

```text
N=100
N=1000
N=5000
```

## 16.2 Nsight Systems

记录：

```text
kernel launch count
GPU timeline
H2D / D2H time
stream synchronization
CPU overhead
```

## 16.3 Nsight Compute

记录核心 kernel：

```text
ik_lm_multiseed_kernel_v4
select_best_per_target_kernel_v4
```

指标：

```text
registers/thread
shared memory/block
local memory spill
achieved occupancy
SM utilization
global memory throughput
branch divergence
warp execution efficiency
```

## 16.4 输出

```text
data/results/nsight_summary.csv
docs/nsight_summary.md
logs/nsight.log
```

## 16.5 判断

必须写清楚：

```text
1. bottleneck 是 compute 还是 memory；
2. 是否存在 local memory spill；
3. K=16 是否导致 register pressure 过大；
4. N 增大后 SM utilization 是否上升；
5. kernel launch 是否固定低数量。
```

---

# 17. Phase 12：最终 Paper Readiness Report

生成：

```text
docs/final_paper_readiness_report.md
data/results/final_summary.csv
```

## 17.1 报告结构

```markdown
# CUDA V4 Final Paper Readiness Report

## 1. 项目版本与方法定义

## 2. Python 原型指标回顾

## 3. CUDA Port 完成情况

## 4. Correctness Check
### 4.1 FK
### 4.2 Jacobian
### 4.3 LM single seed
### 4.4 Multi-seed full pipeline

## 5. Static Batch IK Benchmark
### 5.1 N=100/500/1000/5000
### 5.2 Throughput
### 5.3 Strict SR
### 5.4 Error distribution
### 5.5 near_limit

## 6. Trajectory Smoothness Benchmark

## 7. cuRobo Comparison
### 7.1 GPU stream time
### 7.2 E2E time
### 7.3 valid throughput
### 7.4 success rate and error

## 8. Nsight Profiling
### 8.1 kernel launch
### 8.2 occupancy
### 8.3 registers and spill
### 8.4 bottleneck analysis

## 9. 是否可以开始论文
### 9.1 已满足条件
### 9.2 未满足条件
### 9.3 最终判断

## 10. 建议论文主线
```

## 17.2 最终判断规则

### 可以开始论文

如果满足：

```text
1. FK/Jacobian/LM correctness 通过；
2. N=1000 Strict SR ≥ 93%；
3. N=1000 pos_p95_all ≤ 8 mm；
4. N=1000 near_limit ≤ 4%；
5. CUDA 相比 Python N=1000 加速 ≥ 20×；
6. cuRobo-Graph 对比完成；
7. Nsight profiling 完成；
8. 能明确说明优势和劣势边界。
```

写：

```text
结论：可以开始论文写作。当前结果已经足以支撑“GPU-native constraint-aware batch IK solver”的论文主线。
```

### 可以开始论文，但结论保守

如果满足：

```text
1. correctness 通过；
2. CUDA 明显快于 Python；
3. V4 成功率保持；
4. cuRobo 对比完成；
5. 但 CUDA-V4 没有明显超过 cuRobo。
```

写：

```text
结论：可以开始论文写作，但论文结论应保守。本文不主张全面超过 cuRobo，而主张在 fixed-size batch IK 场景下提供高成功率、约束感知、CUDA-friendly 的专用求解器。
```

### 暂不能开始论文

如果出现：

```text
1. CUDA 与 Python 数值不一致；
2. Strict SR 明显低于 Python；
3. pos_p95_all 爆炸；
4. CUDA 加速不明显；
5. cuRobo 对比缺失；
6. Nsight 缺失。
```

写：

```text
结论：暂不能开始论文。必须先修复 CUDA correctness 或补齐 benchmark。
```

---

# 18. 论文主线预设

如果本阶段完成，论文主线建议为：

```text
本文提出一种面向严格批量逆运动学任务的 GPU 原生多种子 LM 求解框架。该方法利用解析 Jacobian 消除数值差分误差，采用 Sobol 低差异种子库提升收敛盆地覆盖能力，并通过关节限位 Barrier 与轨迹候选重排序实现约束感知求解。针对固定自由度机械臂的 6×6 小矩阵结构，本文设计 target-seed 并行映射和小矩阵 CUDA kernel，实现高成功率、低长尾误差和可预测的批量 IK 执行。
```

禁止主张：

```text
本文全面超过 cuRobo。
```

允许主张：

```text
1. 本文在 fixed-size batch IK 子问题上具有专用化优势；
2. 本文给出了与 cuRobo 的性能交叉区间；
3. 本文在小批量延迟、约束感知候选选择或结构可预测性上有优势；
4. cuRobo 在完整运动生成、大批量吞吐和碰撞约束方面仍更完整。
```

---

# 19. 最小可行执行顺序

如果时间有限，按这个最小闭环执行：

```text
1. Build current project.
2. Export N=200 / N=1000 targets and Sobol-K16 seeds.
3. FK CUDA implementation.
4. FK correctness.
5. Analytical Jacobian CUDA implementation.
6. Jacobian correctness.
7. LM single-seed CUDA.
8. N=10 LM correctness.
9. Multi-seed K16 CUDA.
10. Candidate selection.
11. N=100 correctness.
12. N=1000 correctness.
13. Static benchmark N=100/500/1000/5000.
14. cuRobo comparison N=100/500/1000/5000.
15. Nsight profiling N=100/1000/5000.
16. final_paper_readiness_report.md.
```

Trajectory smoothness rerank 可以作为增强项，但如果时间紧，优先级低于：

```text
correctness
static benchmark
cuRobo comparison
Nsight
```

---

# 20. 当前最重要原则

本阶段不要追求功能无限完整。

当前目标只有一个：

```text
让 V4-Final-K16 从 Python 原型变成可验证、可计时、可对标 cuRobo 的 CUDA 实现。
```

只要完成这个目标，就可以判断是否开始论文写作。

# END
