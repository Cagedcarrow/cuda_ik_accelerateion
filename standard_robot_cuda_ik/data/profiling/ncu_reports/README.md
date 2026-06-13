# Nsight 性能剖析文件完全解读

> 一份写给未来自己的技术备忘录——每个 `.ncu-rep` 和 `.nsys-rep` 文件是什么、怎么来的、在论文里证明了什么。

---

## 目录

- [0. 前置知识：两个 Nsight 工具的区别](#0-前置知识两个-nsight-工具的区别)
- [1. NCU 报告文件清单](#1-ncu-报告文件清单)
- [2. 逐文件解读](#2-逐文件解读)
- [3. NSYS 报告（cuRobo 诊断）](#3-nsys-报告curobo-诊断)
- [4. 关键指标速查表](#4-关键指标速查表)
- [5. 论文引用映射](#5-论文引用映射)
- [6. 如何复现这些 profile](#6-如何复现这些-profile)

---

## 0. 前置知识：两个 Nsight 工具的区别

很多刚接触 NVIDIA 工具链的人会混淆这两个名字相似的工具。一句话区分：

| | Nsight **Compute** (NCU) | Nsight **Systems** (NSYS) |
|:---|:---|:---|
| **分析粒度** | 单个 kernel 内部的 GPU 硬件指标 | 整个应用程序的 CUDA API 调用序列 |
| **回答的问题** | "这个 kernel 为什么慢？寄存器够吗？带宽瓶颈还是计算瓶颈？" | "程序启动了多少次 kernel？哪次 cudaMalloc 花了 200ms？" |
| **输出文件** | `.ncu-rep` | `.nsys-rep` |
| **典型指标** | 计算吞吐率、DRAM 带宽、寄存器/线程、Bank 冲突、占用率 | CUDA kernel launch 次数、cudaMalloc 耗时、stream 同步次数 |
| **类比** | 给一个函数做 CPU 性能剖析（perf record） | 给整个程序做系统调用追踪（strace） |

**本项目的使用方式：**
- **NCU** → 证明 CUDA kernel 是 compute-bound（非带宽瓶颈）、Bank 冲突从 3522 降至 1295、零寄存器溢出
- **NSYS** → 证明 cuRobo 退化 N 值（N=4000）的 kernel 启动数是正常 N 值（N=5000）的 2.37 倍

---

## 1. NCU 报告文件清单

所有 `.ncu-rep` 文件位于当前目录：

| 文件 | 大小 | 日期 | 对应实验 | 论文用途 |
|:---|:---|:---|:---|:---|
| `b4_fp64_n100_zero_seed.ncu-rep` | 14 MB | Jun 10 | B4 (FP64 全精度), N=100, zero_seed | **主 profiling 数据** — 论文表 6.7 NCU 对比 |
| `b5_mixed_n100_memory.ncu-rep` | 5.8 MB | Jun 10 | B5 (混合精度), N=100, zero_seed | **主 profiling 数据** — 论文表 6.7 NCU 对比 |
| `b3_fp64_n5000_full_zero_seed.ncu-rep` | 40 MB | Jun 10 | B3 (FP64 + 自适应阻尼), N=5000 | 大批量扩展性验证 |
| `b4_fp64_n100_full_home_seed.ncu-rep` | 15 MB | Jun 10 | B4 (FP64), N=100, home_seed | 种子策略对 kernel 行为的辅助验证 |
| `b4_fp64_n100_launch.ncu-rep` | 223 KB | Jun 10 | B4, N=100, launch-only profile | 轻量 profile（仅 kernel 启动阶段，不含完整内存追踪） |

**已导出摘要：** 同目录下的 `../ncu_summary.csv` 已提取了前三份报告的关键指标。

---

## 2. 逐文件解读

### 2.1 `b4_fp64_n100_zero_seed.ncu-rep` — B4 FP64 全精度基线（14 MB）

**实验条件：**
- 配置：B4 = A6 二进制 = FP64 全精度 + 自适应阻尼 + 步长钳位 + 分支对齐
- 批量：N=100 个目标
- 种子：zero_seed（全零初始关节角）
- 阈值：Medium（10mm / 5°）

**关键指标解读：**

```
Compute Throughput:  66.89%    ← GPU 计算单元有 66.89% 的时间在干活
DRAM Throughput:      1.56%    ← 显存带宽仅用了 1.56%
Registers/Thread:    94        ← 每个线程用 94 个 32-bit 寄存器
Occupancy:           32.51%    ← 每个 SM 的理论 warp 槽位只用了 32.51%
Bank Conflicts:      3,522     ← 共享内存 Bank 冲突次数
L1 Hit Rate:         99.13%    ← L1 缓存命中率
Kernel Duration:     2,920 μs  ← kernel 执行总时间
```

**这意味着什么？**

这个 kernel 是**典型的 compute-bound（计算密集型）**——66.89% 的计算单元利用率 vs 1.56% 的显存带宽利用率，差距超过 40 倍。它不是"数据喂不饱计算单元"的问题（那种叫 memory-bound），而是"计算单元本身就忙不过来"。

94 个寄存器/线程说明编译器给每个线程分配了大量寄存器来避免 spill（溢出到 local memory）。Ada Lovelace 上限是 255，所以 94 完全安全。但反过来，每个 SM 最多容纳 65536 个寄存器，128 threads × 94 regs = 12,032 regs/block，每个 SM 只能同时驻留约 5 个 block —— 这就是 Occupancy 只有 32.51% 的原因。

3,522 次 bank conflict 来自自然 stride=6 的共享内存布局。这是 PaddedMat6x8 要解决的问题。

### 2.2 `b5_mixed_n100_memory.ncu-rep` — B5 混合精度（5.8 MB）

**实验条件：**
- 配置：B5 = A7 二进制 = FP32 FK/Jacobian + FP64 LDLT（**论文主配置**）
- 批量：N=100
- 种子：zero_seed

**关键指标（与 B4 对比）：**

```
                    B4 (FP64)    B5 (Mixed)    变化
Compute Throughput:  66.89%  →   60.73%       -9%
DRAM Throughput:      1.56%  →    1.16%       —
Registers/Thread:    94      →   98           +4
Occupancy:           32.51%  →   33.30%       —
Bank Conflicts:      3,522    →   1,295        -63% ★
L1 Hit Rate:         99.13%  →   98.62%       —
Kernel Duration:     2,920 μs →   827 μs       -72% ★★★
```

**这意味着什么？**

**Kernel 时间从 2920 μs 降到 827 μs，快了 3.5 倍。** 这是整个项目最核心的性能数字之一。

Compute Throughput 从 66.89% 降到 60.73% 不是坏事——因为 FP32 指令吞吐是 FP64 的 64 倍，同样的计算量在 FP32 下执行时间大幅缩短，利用率自然会降低（分母变小了）。

**Bank Conflicts 从 3522 降到 1295（-63%）。** 这是两个因素叠加的结果：
1. PaddedMat6x8（stride=8）减少系统性 Bank 冲突
2. FP32 数据宽度是 FP64 的一半（4 bytes vs 8 bytes），单个 double 跨 2 个 bank，float 只跨 1 个

残余的 1,295 次冲突来自非填充矩阵访问路径。所以论文里写"降低"而非"消除"，正是因为 NCU 还记录到了这些残余。

寄存器从 94 增到 98（+4），因为混合精度引入了 FP32↔FP64 类型转换的中间变量。

### 2.3 `b3_fp64_n5000_full_zero_seed.ncu-rep` — B3 FP64 大批量（40 MB）

**实验条件：**
- 配置：B3 = A5 二进制 = FP64 + 自适应阻尼
- 批量：N=5000
- 种子：zero_seed

**关键指标：**

```
Compute Throughput:  85.00%    ← 大批量下计算单元利用率更高
DRAM Throughput:      5.00%
Registers/Thread:    94
Occupancy:           41.67%    ← 比 N=100 的 32.51% 高
Bank Conflicts:      0         ← N=5000 时恰好未触发
L1 Hit Rate:         99.00%
Kernel Duration:     70,060 μs ← 约 70ms
```

**这意味着什么？**

N=5000 时 Compute Throughput 升到 85%——大批量意味着更多 block 在 SM 上排队，GPU 调度器有更多 warp 可以切换来隐藏延迟。Occupancy 也从 32.51% 升到 41.67%。

Bank Conflicts = 0 是巧合，不是设计目标。不同 N 值下共享内存访问模式有微小差异，N=5000 恰好落在了无冲突的 pattern 上。

这个文件 40 MB 是最大的——因为 N=5000 的 kernel 执行时间长（70ms vs N=100 的 2.9ms），NCU 在此期间采样了更多的硬件计数器。

### 2.4 `b4_fp64_n100_full_home_seed.ncu-rep` — 种子策略对比（15 MB）

**实验条件：** 与 2.1 完全相同，但使用 `home_seed`（UR10 零位构型）代替 `zero_seed`。

**用途：** 验证不同种子策略对 kernel 内部行为的影响。论文种子策略分析（表 6.5）使用的是 B4 配置——这份 NCU 报告为"为什么 home_seed 吞吐更低"提供了硬件层面的解释（更多迭代 → 更多发散分支 → 更低计算效率）。

### 2.5 `b4_fp64_n100_launch.ncu-rep` — 轻量 profile（223 KB）

**实验条件：** B4, N=100, zero_seed，但只 profile kernel launch 阶段，不含完整内存访问追踪。

**用途：** 快速验证 kernel 是否正确启动、grid/block 维度是否正确。223 KB 极小（对比完整 profile 的 14 MB），因为只记录了 launch configuration 和顶层指标，没有逐指令采样。

---

## 3. NSYS 报告（cuRobo 诊断）

这两个 `.nsys-rep` 文件当前在备份目录（cuRobo 诊断实验整体归档），但它们的数据被论文 §6.2.4 和全量程分析直接引用。

| 文件 | 大小 | 场景 | 关键发现 |
|:---|:---|:---|:---|
| `nsys_N4000.nsys-rep` | 2.1 MB | cuRobo N=4000（**退化点**） | CUDA kernel 启动 14,108 次，cudaEventRecord 5,069 次，cudaStreamWaitEvent 4,985 次 |
| `nsys_N5000.nsys-rep` | 965 KB | cuRobo N=5000（**正常点**） | CUDA kernel 启动 5,945 次，cudaEventRecord 390 次，cudaStreamWaitEvent 350 次 |

**这两个文件是怎么来的：**

1. 全量程扫描发现 cuRobo 在 N=4000 时突然退化（~230ms vs 正常 ~32ms）
2. 用 `nsys profile --trace=cuda` 分别对 N=4000 和 N=5000 进行 CUDA API trace
3. 对比发现：退化点的 kernel 启动数是正常点的 **2.37 倍**（14,108 vs 5,945），事件同步调用数是正常点的 **13–14 倍**
4. 这直接证明了退化根因是 cuRobo 内部 sub-batch 划分策略在特定 N 值触发更多 kernel launch，而非 PyTorch 内存分配器（cudaMalloc/cudaFree 两个点都 <1% API 时间）

**如何在论文中找到这些数据的引用：**
- 论文 §6.2.4 "cuRobo 批量敏感性诊断"
- 论文 §6.5 "全量程批量扩展性分析"
- 摘要第 9 行

---

## 4. 关键指标速查表

当你在 NCU GUI 中打开 `.ncu-rep` 文件时，以下是需要关注的指标及其含义：

### GPU 吞吐类

| NCU 指标名 | 含义 | 本项目参考值 | 怎么判断好坏 |
|:---|:---|:---|:---|
| **Compute Throughput** | SM 计算单元利用率 | 60–85% | 越高越好；>60% 说明是 compute-bound |
| **DRAM Throughput** | 显存带宽利用率 | 1–5% | 越低越好（说明不是 memory-bound）；>60% 说明需要优化内存访问 |
| **L1/TEX Hit Rate** | L1 缓存命中率 | 98–99% | >95% 说明数据局部性优秀 |

### 资源占用类

| NCU 指标名 | 含义 | 本项目参考值 | 怎么判断好坏 |
|:---|:---|:---|:---|
| **Registers/Thread** | 每线程寄存器数 | 94–98 | <255（上限）即可；越低 occupancy 越高，但可能触发 spill |
| **Achieved Occupancy** | 实际 warp 占用率 | 32–42% | 越高越好，但 compute-bound kernel 不必强求 |
| **Local Memory (spill)** | 寄存器溢出到 local memory | **0** | 必须为 0；>0 说明编译器被迫把寄存器变量放到显存 |

### 共享内存类

| NCU 指标名 | 含义 | 本项目参考值 | 怎么判断好坏 |
|:---|:---|:---|:---|
| **Shared Bank Conflicts** | 共享内存 Bank 冲突次数 | B4: 3,522 / B5: 1,295 | 越低越好；揭示了 PaddedMat6x8 的效果 |
| **Shared Memory/Block** | 每 block 共享内存用量 | ~1,616 bytes | 远低于 100 KB 上限，不构成限制 |

### 瓶颈判定公式

```
若 Compute Throughput >> DRAM Throughput  →  compute-bound（本项目属于此类）
若 DRAM Throughput   >> Compute Throughput  →  memory-bound（需要优化显存访问模式）
若 Occupancy 低 + Compute Throughput 低     →  latency-bound（需要更多 warp 隐藏延迟）
```

---

## 5. 论文引用映射

| 论文位置 | 使用的 NCU/NSYS 数据 | 证明的论点 |
|:---|:---|:---|
| **摘要** | NCU B4 vs B5, NSYS N4000 vs N5000 | kernel compute-bound；cuRobo 退化 = 内部 sub-batch 策略 |
| **§6.3 消融实验** | NCU `b5_mixed_n100_memory` bank conflicts 1295 | PaddedMat6x8 + FP32 将 Bank 冲突降低 63% |
| **§6.4 NCU Profiling（表 6.7）** | NCU B4/B5 N=100 完整对比 | 混合精度 kernel 时间 -72%，compute-bound 本质不变 |
| **§6.5 全量程扫描** | NSYS N4000 vs N5000 kernel launch 2.37× | cuRobo 退化 N 值在 CUDA API 层面的直接证据 |
| **§6.2.4 cuRobo 诊断** | NSYS cudaEventRecord 13×, cudaStreamWaitEvent 14× | 排除 PyTorch Caching Allocator，定位到 sub-batch 策略 |
| **§8 讨论** | 全部 NCU + NSYS | 手写 CUDA 单 kernel 封装的结构免疫性 |

---

## 6. 如何复现这些 profile

### 生成 NCU 报告（kernel 内部指标）

```bash
# B4 FP64 N=100 zero_seed
ncu --set full \
    --export b4_fp64_n100_zero_seed.ncu-rep \
    ./build/standard_robot_cuda_runner_A6 \
    --targets data/targets/ur10_seed42_N100.bin \
    --seeds data/seeds/ur10_seed42_zero_seed_N100.bin \
    --repeat 1

# B5 混合精度 N=100
ncu --set full \
    --export b5_mixed_n100_memory.ncu-rep \
    ./build/standard_robot_cuda_runner_A7 \
    --targets data/targets/ur10_seed42_N100.bin \
    --seeds data/seeds/ur10_seed42_zero_seed_N100.bin \
    --repeat 1
```

### 生成 NSYS 报告（CUDA API 调用序列）

```bash
nsys profile --trace=cuda \
    --output nsys_N4000.nsys-rep \
    python3 benchmark/bench_curobo.py --N 4000 --repeat 1
```

### 命令行导出摘要（无需 GUI）

```bash
# NCU 导出 CSV 指标
ncu --import b5_mixed_n100_memory.ncu-rep --csv > ncu_b5_metrics.csv

# NSYS 导出 CUDA API 汇总
nsys stats nsys_N4000.nsys-rep --report cuda_api_sum
```

---

*最后更新：2026-06-14 | 文件数：5 × .ncu-rep（77 MB）+ 2 × .nsys-rep（3 MB，备份目录）*
