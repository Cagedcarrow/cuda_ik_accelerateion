# 04 — GPU 架构与 CUDA 实现细节

> **定位：** 本文档是 GPU 架构参数、CUDA 特性使用、kernel 设计决策的**唯一权威参考**。论文 methodology/implementation 章节必须以本文档为事实标准。

---

## 1. 目标 GPU 架构

### 1.1 硬件规格

| 参数 | 值 |
|------|-----|
| GPU 型号 | NVIDIA GeForce RTX 4060 Laptop GPU |
| 架构 | Ada Lovelace |
| 计算能力 | sm_89 |
| CUDA Cores | 3,072 |
| SM 数量 | 24 |
| GPU 频率 | 1.5–2.0 GHz (SM), 2.0–2.5 GHz (boost) |
| 显存 | 8 GB GDDR6 |
| 显存带宽 | 256 GB/s |
| L1/SM | 128 KB (configurable) |
| L2 Cache | 24 MB |
| CUDA Toolkit | 12.6 |
| 编译器 | NVCC (GCC 11.4.0 host compiler) |
| C++ 标准 | C++17 |
| CUDA 标准 | CUDA 17 |

### 1.2 SM 硬件限制

| 限制 | 值 | 本项目使用 |
|------|-----|:---:|
| 最大寄存器/线程 | 255 (32-bit) | **94–98**（远低于上限，零 spilling） |
| 最大寄存器/SM | 65,536 | — |
| 共享内存/SM | 100 KB (configurable up to 100 KB) | **1,616 bytes/block**（< 2%） |
| 最大线程/SM | 1,536 | — |
| 最大线程/block | 1,024 | **128** |
| 最大 block/SM | 24 | — |
| Warp 大小 | 32 threads | — |
| 共享内存 Bank 数 | 32 (4 bytes/bank) | — |
| FP64:FP32 吞吐比 | **1:64** (Ada Lovelace 消费级) | 决定混合精度策略 |

### 1.3 为什么选 Ada Lovelace (sm_89)

- 消费级 GPU 中**最具代表性的当前代架构**（2023+）
- FP64:FP32 = 1:64 的极端不对称性使混合精度策略的收益在该架构上最大化
- 255 寄存器/线程上限 + 100 KB 共享内存使寄存器级 6×6 LDLT 成为可行方案
- CMake 配置：`set(CMAKE_CUDA_ARCHITECTURES 89)` in `standard_robot_cuda_ik/CMakeLists.txt`

---

## 2. 线程映射模型

### 2.1 Grid → Block → 阶段式 flat threadIdx.x

```
Grid:  (N, 1, 1)     N = 目标位姿数
Block: (128, 1, 1)   4 warps × 32 lanes = 128 threads
```

**核心设计：1 block 对应 1 个 IK 目标。** Grid 中 N 个 block 完全并行发射，零 block 间通信，零全局同步。

### 2.2 Block 内阶段式线程分工

> **设计原则：** 代码使用 flat `threadIdx.x` 而非严格 warp 边界控制线程分工。以下为各阶段精确映射（与 `cuda_ik_6dof.cu` 源码严格一致）。

| 计算阶段 | 线程范围 | 条件 | 任务 | 并行粒度 |
|---------|:---:|------|------|:---:|
| FK（前向运动学） | `0` | `threadIdx.x == 0` | Rodrigues 公式 6 关节链式乘法 | 1 |
| 位姿误差 | `0` | `threadIdx.x == 0` | 位置误差 + 姿态 geodesic 距离 | 1 |
| 收敛判定 | `0` | `threadIdx.x == 0` | pos_err < tol AND rot_err < tol | 1 |
| 数值 Jacobian | `0–5` | `threadIdx.x < 6` | 每线程 1 列，中心差分 ε=1e-6 | 6 |
| 自适应阻尼更新 | `0` | `threadIdx.x == 0` | LM 阻尼调整 (ABLATION_LEVEL≥5) | 1 |
| **H 矩阵构造** | **`0–35`** | **`threadIdx.x < 36`** | **全部 36 元素** `row=tid/6, col=tid%6` | **36** |
| g 向量构造 | `0–5` | `threadIdx.x < 6` | `g[i] = Σ w²·J(k,i)·err[k]` | 6 |
| LDLᵀ 求解 | `0` | `threadIdx.x == 0` | 寄存器级 6×6 LDLᵀ | 1 |
| 步长钳位 | `0` | `threadIdx.x == 0` | `‖Δq‖∞ > 0.35 → scale` (ABLATION_LEVEL≥6) | 1 |
| 关节更新 + 限位 | `0–5` | `threadIdx.x < 6` | `q += clamp(dq, lo, hi)` | 6 |

### 2.3 ⚠️ H 构造跨越 warp 边界

6DOF 时 M²=36，`threadIdx.x=0–35` 参与 H 构造，这意味着：
- 线程 0–31 属于 Warp 0
- 线程 32–35 属于 Warp 1 的 lane 0–3

**因此 H 构造不应对应任何单一 warp。** 论文和专利中**禁止**使用 "W2 负责 H 构造" 等 warp 专属表述。

### 2.4 `__syncthreads()` 同步点

每个 DLS 迭代内最多 7 次 `__syncthreads()`：
1. 收敛判定后
2. Jacobian 组装后
3. 阻尼更新后
4. H 矩阵构造后
5. g 向量构造后
6. LDLᵀ 求解后
7. 步长钳位后（如启用）

---

## 3. 共享内存布局

### 3.1 总览

**每 block 共享内存总量：~1,616 bytes**（Nsight Compute 实测）

| 类别 | 变量 | 元素数 | 字节数 | Stride |
|------|------|:---:|:---:|:---:|
| 矩阵 (PaddedMat6x8) | `s_J` | 48 (6×8) | 384 | 8 |
| 矩阵 (PaddedMat6x8) | `s_H` | 48 (6×8) | 384 | 8 |
| 关节/位姿缓冲 | `s_q[8]` + `s_T[16]` + `s_T_tgt[16]` | 40 | 320 | 8 |
| 误差/梯度/步长 | `s_err[6]` + `s_g[6]` + `s_dq[6]` + `s_q_ref[6]` + `s_q_best[6]` | 30 | 240 | 8 |
| 工具变换 | `s_T_tcp[16]` + `s_T_tcp_tgt[16]` | 32 | 256 | — |
| 标量控制 | converged, iter_count, λ, best_pos_err, stagnation | — | 28 | — |

### 3.2 PaddedMat6x8 设计

**问题：** 自然 stride=6 布局下，Bank 访问模式 `gcd(12, 32)=4`，每 8 次访问周期重复，导致 2–3 路 Bank 冲突。

**方案：** 将行步长从 6 扩展至 8（64 bytes = 16 banks）。

| 布局 | 行步长(banks) | gcd(步长, 32) | 冲突模式 |
|------|:---:|:---:|------|
| 自然 stride=6 | 12 | 4 | 8 次周期，2–3 路冲突 |
| **Padded stride=8** | **16** | **16** | **2 次周期，偶数行/奇数行 Bank 集分离** |

**关键性质：**
- 偶数行 (r=0,2,4)：起始 Bank = `c×2`，使用 Bank 0–15
- 奇数行 (r=1,3,5)：起始 Bank = `16+c×2`，使用 Bank 16–31
- 两组 Bank 集合**完全不重叠**
- 核心计算（每次 ≤2 行同时访问）中，Bank 冲突显著降低

**代价：** 每矩阵额外 96 bytes（6×2×8），两个矩阵合计 192 bytes（< 0.2% SM 共享内存）。

**C++ 封装：** 仅 26 行代码，零模板依赖，`operator()` 重载，NVCC -O2 编译后 PTX 指令序列与裸指针一致。

**实测效果（NCU B4→B5）：** shared bank conflicts 从 3,522 降至 1,295（-63%）。仍存在残余冲突（来自非填充矩阵访问路径），因此效果表述为"降低"而非"消除"。

---

## 4. 寄存器预算分析

### 4.1 LDLᵀ 求解器寄存器占用

`ldlt_solve_6x6()` 中全部数据驻留寄存器：

| 数据结构 | 元素数 | 类型 | 32-bit 寄存器 |
|---------|:---:|------|:---:|
| L 矩阵 | 36 | double | 72 |
| D 对角 | 6 | double | 12 |
| y 中间向量 | 6 | double | 12 |
| x（解 Δq） | 6 | double | 12 |
| 循环索引/临时 | ~6 | — | 12 |
| **LDLᵀ 合计** | | | **~120** |

### 4.2 全线程寄存器占用

| 组件 | 寄存器数 |
|------|:---:|
| FK 中间变量 | ~64 |
| 误差/步长/控制 | ~12 |
| LDLᵀ（仅线程 0） | ~120 |
| **峰值（线程 0）** | **~196** |
| **其他线程** | **~76** |

NCU 实测：94 (B4) – 98 (B5) registers/thread。零 local memory spilling。
Ada Lovelace 上限 255 → 安全裕度充足。

---

## 5. 混合精度计算路径

### 5.1 精度边界

```
FP32 FK/Jacobian → FP64 H/g 累积 → FP64 LDLᵀ → FP32 关节更新
```

| 阶段 | 精度 | 原因 |
|------|:---:|------|
| FK 链式乘法 | FP32 | 三角函数密集，FP32 吞吐为 FP64 的 64× |
| 数值 Jacobian | FP32 | ε=1e-6 扰动下 FP32 相对舍入误差 ~1e-7，远小于收敛容差 1e-2 |
| **H 矩阵累加** | **FP64** | 36 个内积各 6 项累加，需抑制 FP32 截断误差传播 |
| **g 向量累加** | **FP64** | 同上 |
| **LDLᵀ 求解** | **FP64** | 除法对 D_j 偏小敏感，误差经回代放大 |
| 阻尼/收敛判定 | FP64 | 利用 LDLᵀ 结果的 FP64 精度 |
| 关节更新 | FP32 | Δq 已由 FP64 LDLᵀ 高精度确定，FP32 舍入 ~1e-7 rad |

### 5.2 性能收益

FP32 FK/Jacobian + FP64 LDLᵀ 将主体计算迁移至 FP32（64× 吞吐），同时保持 LDLᵀ 求解器数值稳定性。B3→B5 吞吐提升 120–149%，收敛率无退化。

---

## 6. 常量内存（Constant Memory）

### 6.1 使用情况

7 个 `__constant__` 数组，总计 ~1,160 bytes：

| 数组 | 大小 | 用途 |
|------|------|------|
| `c_segment_origins[96]` | 768 B | DH 参数原点 (12 segments × 8 doubles) |
| `c_segment_axes[18]` | 144 B | DH 参数轴 |
| `c_q_index[6]` | 24 B | 关节索引映射 |
| `c_T_wrist3_to_tcp[16]` | 128 B | 腕部→TCP 齐次变换 |
| `c_joint_limits[12]` | 96 B | 关节限位 [lo, hi] × 6 |
| `c_weight_schedule[24]` | 192 B | 误差权重 (4 levels × 6 dims) |
| `c_lambda_params[4]` | 32 B | 阻尼参数 [base, far, floor, scale] |

### 6.2 广播机制

Warp 内所有线程同时访问同一常量内存地址时，数据通过专用 cache 广播，零 bank 冲突，等效于寄存器访问延迟。

### 6.3 消融结果

B0→B1（常量内存）：吞吐增益 < 5%（UR10 FK 工作集 ~912 bytes，L1 缓存即可高效容纳）。大批量（N≥500）收益可忽略。因此不作为论文主要性能声明。

---

## 7. DLS 迭代 Pipeline（完整流程）

每个 block 内，以下阶段顺序执行（每轮迭代）：

```
1. FK 计算          (threadIdx.x == 0)
   └─ forward_kinematics(): Rodrigues × 6 joints → T_EE
2. 位姿误差         (threadIdx.x == 0)
   └─ pose_error(): pos = ‖Δp‖₂, rot = geodesic(R_diff)
3. 收敛判定         (threadIdx.x == 0)
   └─ if pos < 10mm AND rot < 5°: break
4. 数值 Jacobian    (threadIdx.x < 6)
   └─ 每线程: q±ε → FK± → J_col = (x⁺-x⁻)/(2ε)
5. 自适应阻尼       (threadIdx.x == 0) [ABLATION_LEVEL ≥ 5]
   └─ iter 0: distance-based init | iter 1+: Marquardt error-driven
6. H 矩阵构造       (threadIdx.x < 36)
   └─ H[row][col] = Σ w²·J(k,row)·J(k,col) + (row==col ? λ : 0)
7. g 向量构造       (threadIdx.x < 6)
   └─ g[i] = Σ w²·J(k,i)·err[k]
8. LDLᵀ 求解        (threadIdx.x == 0)
   └─ ldlt_solve_6x6(H, g, Δq): 86 scalar ops, all in registers
9. 步长钳位         (threadIdx.x == 0) [ABLATION_LEVEL ≥ 6]
   └─ if ‖Δq‖∞ > 0.35: Δq *= 0.35/‖Δq‖∞
10. 关节更新+限位   (threadIdx.x < 6)
    └─ q[i] = clamp(q[i] + Δq[i], lo[i], hi[i])
```

每阶段后 `__syncthreads()`。

---

## 8. 关键数值参数（代码级确认）

| 参数 | 值 | 代码位置 | 说明 |
|------|-----|---------|------|
| 步长钳位阈值 | **0.35 rad** | `cuda_ik_6dof.cu:279` | 注释："keeps stability while allowing wrist reorientation" |
| 阻尼约定 | **λ 直接加对角线** | `cuda_ik_6dof.cu:244` | `if (row == col) sum += s_lambda;` |
| 数值差分步长 | **ε = 1e-6** | `cuda_ik_6dof.cu:161` | 中心差分 |
| Jacobian 定义 | **∂x/∂q** (FK 中心差分) | `cuda_ik_6dof.cu:170-198` | 对 FK 输出 x(q) 差分 |
| 最大迭代 | **160** | Python: `bench_cuda_6dof.py` | B0–B5 统一 |
| 收敛阈值 Medium | **10mm / 5°** | `benchmark/common.py` | 主 benchmark |
| H 矩阵构造 | **全部 36 元素** | `cuda_ik_6dof.cu:233` | `row=tid/6, col=tid%6` |
| 线程映射 | **flat threadIdx.x** | `cuda_ik_6dof.cu` 全局 | 非 warp 边界 |

---

## 9. LDLᵀ 求解器设计

### 9.1 为什么 LDLᵀ 而非 Cholesky

| 方法 | 平方根 | FMA | 除法 | 优点 | 缺点 |
|------|:---:|:---:|:---:|------|------|
| Cholesky (LLᵀ) | **6 次** | ~50 | ~15 | 经典算法 | GPU sqrt 吞吐 ~1/4–1/8 FMA |
| **LDLᵀ** | **0 次** | **~65** | **~21** | 避免 sqrt，FMA 友好 | 额外 ~15 FMA |

在 n=6 的小规模下，LDLᵀ 以极少额外 FMA 避免 6 次低吞吐 sqrt 指令，总延迟更低。

### 9.2 运算分解

| 阶段 | 运算量 |
|------|:---:|
| 对角元更新 D_j | 15 FMA |
| 非对角元 L_ij 更新 | 20 FMA |
| L 缩放（除以 D_j） | 15 DIV |
| 前代（Ly = b） | 15 FMA |
| 对角缩放（D⁻¹z = y） | 6 DIV |
| 回代（Lᵀx = z） | 15 FMA |
| **合计** | **65 FMA + 21 DIV = 86 标量运算** |

在 1.5–2.0 GHz SM 频率下，~0.1 μs 即可完成。

### 9.3 编译时完全展开

矩阵维度 6 为编译时常量，NVCC `#pragma unroll` 将全部三重循环（j=0..5, k=0..j-1, i=j+1..5）展开为直线代码，消除分支预测失败和循环控制开销。

---

## 10. 自适应阻尼策略

```
迭代 0: 距离初始化
  if pos_err > 0.5m:  λ = λ_far (0.1)
  elif pos_err > 0.1m: λ = linear_interp
  else:               λ = λ_base (5e-4)

迭代 1+: Marquardt 误差驱动
  if pos_err improved:     λ *= 0.7   → 趋向 Gauss-Newton
  else if pos_err worse:   λ *= 2.0   → 趋向梯度下降

停滞超驰:
  if stagnation > 12:      λ *= 5.0   → 强制跳出局部极小

全局钳位: λ ∈ [1e-4, 0.5]
```

全部在 lane 0 串行执行（~15 次标量运算/迭代），对关键路径延迟 < 0.02 μs。

---

## 11. NCU Profiling 关键指标

**N=100, B4 (FP64) vs B5 (Mixed)：**

| 指标 | B4 (FP64) | B5 (Mixed) | 变化 | 解读 |
|------|:---:|:---:|:---:|------|
| Compute Throughput | 66.89% | 60.73% | -9% | 混合精度降低 compute 压力 |
| DRAM Throughput | 1.56% | 1.16% | — | Kernel 为 **compute-bound** |
| Registers/Thread | 94 | 98 | +4 | FP32/FP64 混合增加少量 reg |
| Achieved Occupancy | 32.51% | 33.3% | — | 受寄存器数量限制（非共享内存） |
| L1/TEX Hit Rate | 99.13% | 98.62% | — | 极优缓存行为 |
| Shared Bank Conflicts | 3,522 | 1,295 | -63% | PaddedMat6x8 + FP32 数据宽度减半 |
| Local Memory Spill | 0 | 0 | — | 零寄存器溢出 |

**瓶颈判定：** 计算吞吐（60–67%）>> DRAM 吞吐（1–2%），kernel 明确为 compute-bound。主要受限于寄存器压力和 warp 调度效率。

---

## 12. CUDA Graph（B6, ABLATION_LEVEL=8）

```
cudaStreamBeginCapture(stream)
ik_batch_solve_mixed<<<...>>>(...)   // kernel 被 capture
cudaStreamEndCapture(stream, &graph)
cudaGraphInstantiate(&exec, graph)
// 后续每次调用：
cudaGraphLaunch(exec, stream)         // replay 而非重新 launch
```

**收益：** N=100 时吞吐 +3.7%，cudaGraphLaunch kernel 时间与直接 launch 差异 < 0.2%（测量噪声范围）。Launch overhead 并非当前 kernel 配置的性能瓶颈，CUDA Graph 边际收益有限。

---

## 13. 寄存器级 6×6 与通用库（cuBLAS）的对比

| 特性 | 寄存器 LDLᵀ | cuBLAS (getrf/getrs) |
|------|:---:|:---:|
| 矩阵规模 | 6×6 | 针对 n≥32 优化 |
| 每次求解的 kernel launch | **0**（inline 在 kernel 内） | 至少 1 次（5–10 μs overhead） |
| 数据位置 | 寄存器 | 全局/共享内存 |
| 计算量 | 86 标量运算 (~0.1 μs) | 远大于 0.1 μs（含 launch overhead） |
| 平方根 | 0（LDLᵀ 选型） | Cholesky 路径需 sqrt |

对 6×6 极小矩阵，规避库调用的 kernel launch overhead 是寄存器级实现的**首要动机**。

---

## 14. `__constant__` vs 全局内存 消融 (B0 vs B1)

| 配置 | 参数存储 | 访问模式 |
|------|---------|---------|
| B0 (A0) | 全局内存 | `cudaMemcpy` → global memory, L1/L2 cache |
| B1+ (A1+) | `__constant__` | 专用 constant cache, warp 内广播 |

B0→B1 吞吐增益 < 5%，因为 UR10 工作集 ~912 bytes 本身即可被 L1 高效容纳（L1 128 KB/SM）。常量内存的收益在实际 problem size 上有限。

---

## 15. 代码→论文 映射参考

| 论文概念 | 代码实现 |
|---------|---------|
| Grid(N,1,1) 映射 | `launch_batch_ik()` → `ik_batch_solve<<<N, 128>>>` |
| 阶段式线程分工 | `if (threadIdx.x < 36)` / `< 6` / `== 0` |
| PaddedMat6x8 | `struct PaddedMat6x8` in `cuda_utilities.cuh` |
| 寄存器 LDLᵀ | `ldlt_solve_6x6()` in `cuda_utilities.cuh` |
| 混合精度 | `ik_batch_solve_mixed` kernel (ABLATION_LEVEL=7) |
| 自适应阻尼 | `#if ABLATION_LEVEL >= 5` block in `cuda_ik_6dof.cu` |
| 步长钳位 0.35 rad | `if (step_norm > 0.35)` line 279 |
| 消融控制 | `ABLATION_LEVEL` 宏 → CMake 10 个目标 |
| 三层计时 | `cuda_benchmark_runner.cu` 中 CUDA events |
