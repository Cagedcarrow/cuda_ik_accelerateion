# V4 CUDA Port 项目说明

## 0. 项目定位

本项目是《基于 CUDA 小矩阵加速的机械臂批量逆运动学求解》论文的 **V4 GPU 原生实现版本**。

论文主线进化：
```
V1: CUDA 小矩阵 DLS (数值 Jacobian, 12 FK/迭代)
V2: 解析 Jacobian + DLS (消除 ε 依赖, 5× 加速)
V3: 解析 Jacobian + LM + Multi-Seed / Sobol Seed Bank (Strict SR 从 52% → 95%)
V4: + Joint-Limit Barrier + Smoothness Candidate Reranking (约束感知批量 IK)
```

**当前 V4 算法已通过 Python 原型验证并冻结。本项目负责将其移植到 CUDA。**

---

## 1. 算法最终版本：V4-Final-K16

```
V4-Final-K16 =
    Analytical Jacobian (几何 Jacobian: J = [z_i×(p_ee-p_i); z_i])
  + LM Solver (Levenberg-Marquardt, 自适应 λ, 无 rejection)
  + Sobol Seed Bank (Latin Hypercube stratified sampling, K=16)
  + Limit Barrier (w_limit=0.03, margin=0.087rad≈5°)
  + Smoothness Candidate Reranking (字典序: success → near_limit → smoothness → pose)
```

### 1.1 核心公式

**FK 中间坐标系提取：**
```
T = I₄
for i = 0..5:
    T = T × origin[i]              // 到达关节 i 世界坐标系
    p[i] = T[0:3, 3]               // 保存关节位置
    z[i] = T[0:3, 0:2] × axis[i]   // 保存关节轴（世界坐标系）
    T = T × Rodrigues(axis[i], q[i])
T_ee = T × T_tool
```

**解析 Jacobian (几何 Jacobian)：**
```
J[0:3, i] = z[i] × (p_ee - p[i])
J[3:6, i] = z[i]
```

**LM 阻尼更新：**
```
(JᵀJ + λI) Δq = -Jᵀe
if loss_new < loss_old: λ *= 0.5
else: λ *= 2.0
λ ∈ [1e-6, 0.5]
```

**Limit Barrier (w=0.03)：**
```
L_limit = Σⱼ w_limit × max(0, margin - (qⱼ - qⱼ_min))² 
        + Σⱼ w_limit × max(0, margin - (qⱼ_max - qⱼ))²
margin = 0.087 rad (≈5°)
```

**Smoothness Reranking (不在 LM 残差中，在候选选择阶段)：**
```
sort_key = (success_rank, near_limit_flag, ||q - q_prev||², pose_cost)
优先级: Strict > Medium > Loose > Failed
```

---

## 2. 关键性能数据 (Python 原型，RTX 4060 Laptop)

### 2.1 IK-only (N=1000, Medium 10mm/5°)

| 方法 | K | Strict SR | pos_p95_all | pos_p95_suc | near_limit |
|------|---|-----------|-------------|-------------|------------|
| V3-Sobol-K16 | 16 | 95.1% | 4.69mm | 2.92mm | 9.3% |
| V3-Sobol-K32 | 32 | 97.2% | 3.32mm | 2.14mm | 8.1% |
| V4-Final-K16 | 16 | 95.0% | 5.03mm | 2.62mm | **2.4%** |

### 2.2 Trajectory Smoothness (50 waypoints)

| 轨迹类型 | independent mean_Δq | rerank mean_Δq | 改善 |
|---------|--------------------|----------------|------|
| line | 2.49 rad | 1.32 rad | **−47%** |
| arc | 2.39 rad | 1.55 rad | **−35%** |
| local_random | 2.59 rad | 1.15 rad | **−56%** |

### 2.3 Limit Barrier 权重扫描 (N=1000, K16)

| w_limit | Strict SR | pos_p95/mm | near_lim | 冻结? |
|---------|-----------|------------|----------|-------|
| 0 | 95.1% | 4.69 | 9.3% | baseline |
| **0.03** | **95.0%** | **5.03** | **2.4%** | ✅ **冻结** |
| 0.1 | 94.5% | 12.23 | 2.4% | ❌ p95 恶化 |
| 1.0 | 94.4% | 18.80 | 1.2% | ❌ |

### 2.4 参考：V1 数据 (论文已发表版本)

| N | CUDA-Mixed (V1 DLS) | cuRobo-Graph | V1 vs cuRobo |
|---|---------------------|-------------|-------------|
| 100 | 80,757 t/s | 38,398 t/s | 2.1× |
| 500 | 128,404 t/s | 192,400 t/s | 0.67× |
| 1000 | 138,719 t/s | 334,189 t/s | 0.42× |
| 5000 | 157,481 t/s | 1,056,171 t/s | 0.15× |
| 10000 | 157,454 t/s | 163,160 t/s | 0.97× |

---

## 3. 参考文件路径

### 3.1 本项目内文件

```
standard_robot_cuda_ik_v4_cuda/
├── PROJECT.md                              ← 本文件
├── CMakeLists.txt                          ← V1 构建系统 (需修改)
│
├── src/cuda/
│   ├── cuda_utilities.cuh                  ← V1 FK/Jacobian/LDLT 等 (需大幅重写)
│   ├── cuda_ik_6dof.cu                     ← V1 IK kernel (需重写为 LM+MultiSeed)
│   ├── cuda_benchmark_runner.cu            ← V1 benchmark (需升级)
│   └── cuda_memory.cu                      ← V1 内存管理 (可复用)
│
├── include/standard_robot_cuda_ik/
│   ├── cuda_ik_6dof.h                      ← kernel 声明 (需更新)
│   ├── cuda_memory.h                       ← DeviceBuffer (可复用)
│   ├── cuda_collision.h                    ← 碰撞检测声明 (V4 暂不使用)
│   └── generated/
│       └── ur10_model_constants.h          ← UR10 运动学常量 (不需要改)
│
├── data/
│   ├── targets/
│   │   ├── v4_targets_N200_seed42.npy      ← 固定 200 目标 (快速验证)
│   │   └── v4_targets_N1000_seed42.npy     ← 固定 1000 目标 (稳定性测试)
│   ├── seed_banks/
│   │   ├── sobol_K16_N200_bank00.npy       ← Sobol-K16 for N=200
│   │   ├── sobol_K32_N200_bank00.npy       ← Sobol-K32 for N=200
│   │   ├── sobol_K16_N1000_bank00.npy      ← Sobol-K16 for N=1000
│   │   └── sobol_K32_N1000_bank00.npy      ← Sobol-K32 for N=1000
│   └── results/
│       ├── v4_limit_weight_sweep.csv       ← Limit 权重扫描结果
│       └── v4_m2_smooth_rerank_results.csv ← Smoothness 重排序结果
│
├── experiments/                             ← Python 原型参考实现
│   ├── run_v4_m0.py                        ← M0: V3 freeze + 权重初步
│   ├── run_v4_m1_m2.py                     ← M1/M2: Limit N=1000 + Smoothness
│   └── run_v4_finalize.py                  ← 最终权重扫描 + 诊断
│
├── docs/                                    ← 所有历史报告
│   ├── v4_finalization_report.md           ← V4 最终冻结报告
│   ├── v4_m0_results.md                    ← M0 实验结果
│   └── v4_m1_limit_smooth_results.md       ← M1/M2 实验结果
│
└── benchmark/                               ← cuRobo 对标 (待实现)
```

### 3.2 外部参考文件

| 文件 | 位置 | 说明 |
|------|------|------|
| V1 论文 LaTeX | `../docs/latex论文/paper.tex` | 当前投稿论文 |
| V1 CUDA 源码 | `../standard_robot_cuda_ik/src/cuda/` | V1 参考实现 |
| V2 Jacobian 验证 | `../standard_robot_cuda_ik_v2/experiments/` | 解析 Jacobian 验证 |
| V3 改进方案 | `../standard_robot_cuda_ik_v3/docs/v3改进.md` | V3 改进计划 |
| V3 种子策略报告 | `../standard_robot_cuda_ik_v3/docs/v3改进报告.md` | Seed sweep 结果 |
| V4 设计方案 | `../standard_robot_cuda_ik_v4/docs/v4.md` | V4 总体设计 |
| 改进意见 v2 | `../docs/改进意见/意见2/` | 论文修改计划 |
| cuRobo benchmark | `../standard_robot_cuda_ik/benchmark/bench_curobo.py` | cuRobo 对标脚本 |
| URDF 模型 | `../standard_robot_cuda_ik/urdf/ur10_official.urdf` | UR10 机器人模型 |

---

## 4. CUDA Port 技术方案

### 4.1 需要实现的 CUDA Kernel

#### Kernel 1: `ik_lm_multiseed_kernel` (核心)

```
Grid: (N, K) — N 个目标 × K 个种子
Block: (128, 1, 1)

每个 (target_i, seed_k) 独立运行 LM 求解器:
  - FK (带中间坐标系提取: p_i, z_i)
  - 解析 Jacobian
  - LM 迭代 (自适应 λ, step clamp 0.35 rad)
  - Limit Barrier (梯度加入 LM 残差, w=0.03)
  - 输出: q_best[i,k,6], loss[i,k], success[i,k], iters[i,k]
```

共享内存预算 (per block):
```
s_q[8]           = 64 bytes    (FP64)
s_T[16]          = 128 bytes   (当前 FK 输出)
s_T_tgt[16]      = 128 bytes   (目标位姿)
s_J[6×8]         = 384 bytes   (PaddedMat6×8, FP64)
s_H[6×8]         = 384 bytes   (Hessian)
s_g[8]           = 64 bytes    (梯度)
s_p[6×3]         = 144 bytes   ← 新增: 关节位置
s_z[6×3]         = 144 bytes   ← 新增: 关节轴
s_dq[8]          = 64 bytes
─────────────────────────────────
Total:           ≈ 1504 bytes  (< 48KB SM 限制)
```

寄存器预算:
- V1: 94–98 regs/thread
- V4 增量: FK 中间坐标系提取 + limit gradient ≈ +6 regs
- 预估: ~104 regs/thread (仍低于 255 上限)

#### Kernel 2: `select_best_per_target_kernel` (候选选择)

```
Grid: (N, 1, 1)
Block: (K, 1, 1) 或 (128, 1, 1)

Per target: 从 K 个候选中选择最优解
  - IK-only: argmin_k loss[k]
  - Rerank: success_rank → near_limit → smoothness → pose_cost
```

#### Kernel 3 (可选): `sobol_seed_generator_kernel`

```
在 GPU 上直接生成 Sobol/Latin Hypercube 种子
或预加载 seed bank 到常量内存
```

### 4.2 需要修改的 V1 文件

| 文件 | 修改内容 |
|------|---------|
| `cuda_utilities.cuh` | 新增 `fk_with_frames()` — 提取 p_i, z_i |
| `cuda_utilities.cuh` | 新增 `analytical_jacobian()` — 交叉积 Jacobian |
| `cuda_utilities.cuh` | 新增 `limit_barrier_loss()` + gradient |
| `cuda_ik_6dof.cu` | 重写 kernel: DLS → LM + MultiSeed |
| `cuda_ik_6dof.cu` | 新增 `select_best_candidate()` kernel |
| `cuda_benchmark_runner.cu` | 更新 benchmark 逻辑: 加载 seed bank, 多阈值评价 |
| `CMakeLists.txt` | 新增 V4 ablation targets |

### 4.3 Python-CUDA 对齐验证

在 N=10 小样本上逐项验证:

| 验证项 | 误差要求 |
|--------|---------|
| FK 4×4 矩阵 | < 1e-5 per element |
| FK p_i (关节位置) | < 1e-5 m |
| FK z_i (关节轴) | < 1e-5 |
| Jacobian 6×6 | < 1e-4 rel Frobenius |
| Cholesky/LDLT 求解 | < 1e-12 rel |
| Limit loss | < 1e-10 |
| best seed selection | 一致或 loss 接近 (< 1e-6) |
| final pos_error | < 1e-3 m |

---

## 5. cuRobo 对标计划

CUDA Port 完成后, 与 cuRobo-Graph 进行公平对比:

### Benchmark 配置
```
N = 100, 500, 1000, 5000
K = 16 (V4-Final)
重复 = 30, 预热 = 5
阈值 = Medium (10mm/5°), Strict (5mm/1°)
统一目标、种子、评价协议
```

### 对标方法
```
- V4-Final-K16 (CUDA)
- V3-Sobol-K16 (CUDA, 无 limit barrier + 无 rerank)
- cuRobo-Graph (use_cuda_graph=True)
- cuRobo-NoGraph (use_cuda_graph=False)
```

### 需要的指标
```
GPU stream ms (mean±std)
E2E ms
raw throughput (targets/s)
valid throughput (targets/s × success_rate)
Strict SR, Medium SR
pos p50/p95/max (all + success-only)
rot p50/p95/max
kernel launch count
occupancy (%)
registers/thread
local memory spill
```

---

## 6. 已知性能边界与预期

### V4 可以主张的 (基于 Python 原型)
1. N=100 极小批量下 GPU stream 延迟低于 cuRobo-Graph (V1 已验证)
2. 全量程 GPU 时间严格线性 (R²>0.999)
3. Strict SR 95% at N=1000 K16
4. near_limit ratio 从 9.3% → 2.4% (Limit Barrier)
5. 轨迹 mean_Δq 降低 35–56% (Smoothness Rerank)

### V4 不可以主张的
1. 全面快于 cuRobo-Graph
2. 高精度最终 IK 替代 cuRobo
3. 碰撞检测能力 (V4 未实现)
4. 多 seed 全局搜索替代 cuRobo 的粒子优化
5. K32 性能数据 (Python 原型有, CUDA 待测)

---

## 7. 环境信息

```
GPU:       NVIDIA GeForce RTX 4060 Laptop (Ada Lovelace, sm_89)
VRAM:      8 GB
CUDA:      12.6 (UMD 13.3)
PyTorch:   支持 CUDA
cuRobo:    已安装, 可 import
Python:    3.10
编译器:    NVCC (via CMake)
OS:        Linux 6.8
```

## 8. 执行优先级

```
Phase 1: fk_with_frames + analytical_jacobian CUDA 实现
         → N=10 Python-CUDA 对齐验证
         
Phase 2: LM single-seed CUDA kernel
         → N=100 与 Python 原型对齐
         
Phase 3: Multi-seed kernel + best candidate selection
         → N=100 K=16 全流程验证
         
Phase 4: Limit Barrier 集成
         → N=1000 复核 (vs Python 原型)
         
Phase 5: Smoothness Rerank (仅在候选选择阶段, Kernel 2)
         
Phase 6: cuRobo 对标 benchmark
         → N=100/500/1000/5000, 30 repeats
         
Phase 7: Nsight profiling + 论文最终数据
```
