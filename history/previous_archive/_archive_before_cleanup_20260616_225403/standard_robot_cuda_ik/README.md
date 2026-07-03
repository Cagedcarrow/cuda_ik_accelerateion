# Standard Robot CUDA IK — 主线项目完整指南

> **本文档的定位：** 新会话启动后，AI 或开发者阅读本文档即可完全理解项目结构、关键事实和操作流程。不需要再翻阅其他文件来"摸清情况"。
>
> **事实优先级：** 源码 > 实验数据 CSV > 本文档 > `docs/项目总览/` > 论文草稿

---

## 目录

- [1. 项目定位](#1-项目定位)
- [2. 完整目录结构](#2-完整目录结构)
- [3. 构建系统](#3-构建系统)
- [4. 源码关键事实](#4-源码关键事实)
- [5. 实验数据地图](#5-实验数据地图)
- [6. Benchmark 框架](#6-benchmark-框架)
- [7. 关键性能数字](#7-关键性能数字)
- [8. 常用操作流程](#8-常用操作流程)
- [9. 论文与专利](#9-论文与专利)
- [10. 外部参考求解器](#10-外部参考求解器)
- [11. 铁律与禁止事项](#11-铁律与禁止事项)

---

## 1. 项目定位

本目录（`standard_robot_cuda_ik/`）是整个 `cuda_ik_accelerateion` 仓库的**唯一主线**。

- **做什么：** 用纯手写 CUDA 加速标准工业机器人（UR10）的批量逆运动学求解
- **对比对象：** NVIDIA cuRobo、PyRoki (JAX)、Orocos KDL (CPU)、NumPy DLS (CPU)
- **核心方法：** 1 block/target 映射，128 threads/block，单 kernel 封装全部 DLS 迭代，混合精度（FP32 FK/Jacobian + FP64 LDLT），寄存器驻留 6×6 LDLᵀ 求解

**与项目根目录的关系：**
```
cuda_ik_accelerateion/                  ← 仓库根
├── standard_robot_cuda_ik/             ← ★ 本目录 = 唯一主线
├── docs/                               ← 共享文档（论文、专利、参考标准）
├── external/                           ← 外部求解器源码 clone（仅阅读，gitignored）
└── README.md                           ← 项目主页（面向人类读者）
```

**历史：** 仓库原本有一个第一代项目 `cuda_low_level_optimization/`（自定义 UR10+铲斗装配体），已于 2026-06-14 归档至 `/mnt/linuxdata/cuda数据备份/`。其核心技术（PaddedMat6x8、寄存器 LDLᵀ、自适应阻尼）已迁移至本项目。

---

## 2. 完整目录结构

```
standard_robot_cuda_ik/
│
├── README.md                                   # ★ 本文件
│
├── CMakeLists.txt                              # 构建系统（CUDA 17, sm_89, 10个目标）
│
├── src/
│   ├── cuda/
│   │   ├── cuda_ik_6dof.cu                     # ★ 核心 kernel（~1530行）
│   │   │                                        #   内含 5 个 __global__ kernel：
│   │   │                                        #   · ik_batch_solve           — FP64 主 kernel (A1-A6)
│   │   │                                        #   · ik_batch_solve_ablation_A0 — A0 基线(global memory)
│   │   │                                        #   · ik_batch_solve_mixed      — ★ B5 混合精度 (A7/A8)
│   │   │                                        #   · ik_batch_solve_multi      — 多样本(辅助,未用)
│   │   │                                        #   · compute_continuity_cost   — 连续性代价(辅助,未用)
│   │   │                                        #   · filter_topk_per_target    — Top-K过滤(辅助,未用)
│   │   │                                        #   消融级别通过 ABLATION_LEVEL 宏在编译时控制
│   │   │
│   │   ├── cuda_utilities.cuh                  # ★ 设备端工具函数（~448行）
│   │   │                                        #   · forward_kinematics()      — Rodrigues FK
│   │   │                                        #   · PaddedMat6x8 / FloatPaddedMat6x8 — 共享内存矩阵
│   │   │                                        #   · ldlt_solve_6x6()          — 寄存器 LDLᵀ (86标量运算)
│   │   │                                        #   · pose_error()              — 位姿误差
│   │   │                                        #   · 7个 __constant__ 数组声明
│   │   │
│   │   ├── cuda_benchmark_runner.cu             # ★ 主机端 benchmark 驱动（~483行）
│   │   │                                        #   · main() 入口，解析命令行
│   │   │                                        #   · 三层计时：kernel_only / gpu_e2e / host_api
│   │   │                                        #   · CUDA event 精确计时 + CUDA Graph (B6)
│   │   │                                        #   · 统计输出：mean/std/P50/P95/P99/95%CI
│   │   │
│   │   ├── cuda_collision.cu                    # GPU 碰撞检测（716行，论文未使用）
│   │   └── cuda_memory.cu                       # DeviceBuffer RAII（19行）
│   │
│   └── cpu_baseline/
│       ├── kdl_solver.cpp                       # 空桩（仅返回状态字符串）
│       └── numeric_dls_solver.cpp               # 空桩（同上）
│       # 注意：实际 CPU baseline 在 Python benchmark wrapper 中
│
├── include/standard_robot_cuda_ik/
│   ├── cuda_ik_6dof.h                           # Kernel launch 函数声明
│   ├── cuda_collision.h                          # 碰撞检测声明（未编译）
│   ├── cuda_memory.h                             # DeviceBuffer<T> / ConstantMemory<T>
│   └── generated/
│       └── ur10_model_constants.h                # 自动生成：DH参数、关节限位、权重、阻尼参数
│
├── benchmark/                                    # ★ 多求解器对比框架
│   ├── run_all.py                               #   统一入口：--solver all/cuda/curobo/...
│   ├── common.py                                #   共享基础设施：
│   │                                            #   · BenchmarkResult dataclass
│   │                                            #   · 收敛判定 mark_convergence()
│   │                                            #   · 三档阈值定义 (Loose/Medium/Strict)
│   │                                            #   · 数据加载 load_robot_records/load_seed_values
│   ├── bench_cuda_6dof.py                       #   CUDA B5 wrapper（subprocess 调用二进制）
│   ├── bench_curobo.py                          #   cuRobo wrapper（pip 安装的 curobo 包）
│   ├── bench_pyroki.py                          #   PyRoki wrapper（JAX GPU）
│   ├── bench_kdl.py                             #   KDL wrapper（PyKDL, CPU C++）
│   ├── bench_numeric_dls.py                     #   NumPy DLS wrapper（CPU Python）
│   ├── compare_paper_data.py                    #   论文数据一致性检查工具
│   ├── compare_results.py                       #   通用结果对比工具
│   ├── run_b5_ablation.sh                       #   消融批量执行脚本
│   └── run_curobo_combo.py                      #   cuRobo 三档阈值扫描（独立脚本）
│
├── tools/                                        # 资产生成与验证
│   ├── generate_standard_assets.py               # ★ 入口：生成 target/seed/常量头文件
│   ├── robot_model.py                            # RobotModel 类：URDF解析/FK/关节限位
│   ├── fetch_official_ur10.py                    # 从 GitHub 下载 Universal Robots 官方 URDF
│   ├── verify_official_ur10.py                   # UR10 模型验证（FK 数值一致性）
│   └── verify_seed42_reproducibility.py           # 种子 42 可复现性 MD5 检查
│
├── config/                                       # YAML 配置（纳入版本管理）
│   ├── benchmark.yaml                            #   求解器列表、默认参数、容差
│   ├── robots.yaml                               #   机器人模型注册表
│   └── target_generation.yaml                    #   目标/种子生成参数
│
├── urdf/                                         # 官方机器人 URDF 模型
│   ├── ur10_official.urdf                        #   ★ UR10（UniversalRobts 官方 tag 4.3.1）
│   ├── ur5_official.urdf                         #   UR5
│   └── panda_7dof.urdf                           #   Franka Panda 7-DOF
│
├── data/                                         # ★ 全部实验数据（184文件, ~227MB）
│   ├── README.md                                 #   数据-论文映射索引
│   ├── targets/                                  #   36个目标文件（N=100→10000, .bin/.csv/.json）
│   ├── seeds/                                    #   96个种子文件（4策略×12规模, .bin/.json）
│   ├── results/
│   │   ├── main_comparison/                      #   ★ 主对比 B5 vs cuRobo (Medium 阈值)
│   │   │   ├── main_comparison.csv               #     权威主表数据
│   │   │   └── solver_comparison.csv              #     多求解器对比（旧阈值，仅参考）
│   │   ├── ablation/                             #   ★ 消融实验
│   │   │   ├── ablation_medium.csv               #     权威消融数据 B0/B3/B5 (Medium)
│   │   │   ├── ablation_ur10_old_threshold.csv    #     旧阈值消融（已废弃）
│   │   │   └── mixed_precision_ablation.csv       #     FP64 vs Mixed 对比
│   │   ├── threshold_scan/                       #   三档阈值扫描
│   │   │   ├── threshold_scan.csv                #     权威阈值扫描数据
│   │   │   ├── curobo_summary.csv                 #     cuRobo 汇总
│   │   │   └── *.log                             #     24个逐N详细日志
│   │   ├── full_range/                           #   ★ 全量程 N=100→10000
│   │   │   ├── full_range_comparison.csv          #     12个N值完整对比
│   │   │   └── oscillation_analysis.md            #     cuRobo 振荡分析
│   │   ├── cpu_baseline/                         #   CPU 参考基线（旧阈值，仅数量级参考）
│   │   ├── seed_strategy/                        #   zero_seed vs home_seed
│   │   ├── panda_7dof/                           #   7-DOF Panda 验证
│   │   └── errors/                               #   错误日志
│   ├── profiling/                                # Nsight Compute GPU 性能剖析
│   │   ├── ncu_summary.csv                       #   ★ 关键指标汇总（B4 vs B5）
│   │   └── ncu_reports/                          #   5个 .ncu-rep 原始报告 (77MB)
│   │       └── README.md                         #   ★ Nsight 文件完整解读
│   └── figures/                                  # 论文图表
│       ├── plot_all_figures.py                   #   图表生成脚本
│       ├── figure1_throughput_comparison.png
│       ├── figure2_speedup_bars.png
│       ├── figure3_ablation_throughput.png
│       ├── figure4_convergence_rate.png
│       └── figure5_avg_iterations.png
│
├── experiments/
│   └── 7dof_test/                                # Panda 7-DOF CUDA IK 扩展
│       ├── README.md                             #   明确标注："不产生论文数据"
│       ├── panda_7dof_kernel.cu                  #   7DOF CUDA kernel
│       ├── panda_7dof_runner.cu                  #   主机端测试
│       ├── panda_fk_reference.py                 #   Python 参考实现
│       └── CMakeLists.txt                        #   独立构建配置
│
├── docs/                                         # 子项目文档
│   ├── WORK_HANDOFF_AND_PAPER_TODO.md            #   工作交接与论文待办
│   └── logs/                                     #   结构化实验日志
│
└── build/                                        # 构建产物（gitignored）
    ├── standard_robot_cuda_runner_A0 ~ A8        #   10个消融二进制
    ├── standard_robot_cuda_runner                 #   默认（=A6）
    ├── libstandard_robot_cuda_memory.a
    └── libstandard_robot_cpu_baselines.a
```

---

## 3. 构建系统

### 3.1 编译命令

```bash
cd standard_robot_cuda_ik
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc)
```

**关键配置：**

| 参数 | 值 |
|:---|:---|
| C++ 标准 | C++17 |
| CUDA 标准 | CUDA 17 |
| GPU 架构 | sm_89 (Ada Lovelace) |
| CUDA Toolkit | 12.6 |
| Host 编译器 | GCC 11.4.0 |

### 3.2 消融级别与二进制映射

10 个编译目标，全部编译自同一个源文件（`cuda_benchmark_runner.cu`），通过 `ABLATION_LEVEL` 宏在编译时控制功能开关：

| 二进制 | ABLATION_LEVEL | 论文级别 | 功能描述 | 状态 |
|:---|:---:|:---:|:---|:---:|
| `standard_robot_cuda_runner_A0` | 0 | **B0** | 全局内存，无填充，固定阻尼 | 消融基线 |
| `standard_robot_cuda_runner_A1` | 1 | B1 | + 常量内存 | 消融 |
| `standard_robot_cuda_runner_A2` | 2 | B2 | + PaddedMat6x8 | 消融 |
| `standard_robot_cuda_runner_A3` | 3 | — | + 寄存器 LDLᵀ | 中间过渡 |
| `standard_robot_cuda_runner_A4` | 4 | — | + Kernel 融合 | 中间过渡 |
| `standard_robot_cuda_runner_A5` | 5 | **B3** | + 自适应阻尼 ★ | 消融关键级 |
| `standard_robot_cuda_runner_A6` | 6 | B4 | + 步长钳位 + 分支对齐 | 已知负收益 |
| `standard_robot_cuda_runner_A7` | 7 | **B5 ★★** | **混合精度 (FP32+FP64)** | **主配置** |
| `standard_robot_cuda_runner_A8` | 8 | B6 | A7 + CUDA Graph | 边际收益+3.7% |
| `standard_robot_cuda_runner` | (默认=6) | B6 别名 | 同 A6 | 向后兼容 |

**始终启用（所有级别，不可消融）：**
- 1 block/target, 128 threads/block
- 寄存器级 6×6 LDLᵀ（无外部库依赖）
- 单 kernel 封装全部 DLS 迭代

### 3.3 运行示例

```bash
# B5 主配置（推荐）
./build/standard_robot_cuda_runner_A7 \
    --targets data/targets/ur10_seed42_N1000.bin \
    --seeds data/seeds/ur10_seed42_zero_seed_N1000.bin \
    --repeat 30

# 命令行参数
# --targets       : 目标位姿文件 (.bin 或 .csv)
# --seeds         : 初始关节角种子文件 (.bin 或 .json)
# --repeat        : 重复次数（含 3 次 warmup）
# --pos-tol       : 位置容差（默认 0.01m = 10mm）
# --rot-tol       : 姿态容差（默认 0.08727rad = 5°）
# --weight-level  : 权重级别 0-3（默认 2）
```

---

## 4. 源码关键事实

> **这些是经过源码验证的铁律。论文和文档必须与此一致。**

### 4.1 线程映射（不是 Warp 边界！）

代码使用 **flat `threadIdx.x` 范围检查**，而非严格 warp 边界：

| 计算阶段 | 线程范围 | 条件 | 并行度 |
|:---|:---:|:---|:---:|
| FK | `0` | `threadIdx.x == 0` | 1 |
| 位姿误差/收敛判定 | `0` | `threadIdx.x == 0` | 1 |
| 数值 Jacobian | `0–5` | `threadIdx.x < 6` | 6 |
| 自适应阻尼 | `0` | `threadIdx.x == 0` | 1 |
| **H 矩阵构造** | **`0–35`** | **`threadIdx.x < 36`** | **36** |
| g 向量构造 | `0–5` | `threadIdx.x < 6` | 6 |
| LDLᵀ 求解 | `0` | `threadIdx.x == 0` | 1 |
| 步长钳位 | `0` | `threadIdx.x == 0` | 1 |
| 关节更新+限位 | `0–5` | `threadIdx.x < 6` | 6 |

**⚠️ H 矩阵构造跨越 Warp 边界：** 线程 0–35 包括 Warp 0 的全部 32 lanes 和 Warp 1 的前 4 lanes。因此**严禁**使用 "Warp 2 负责 H 矩阵" 等表述。

### 4.2 关键常量（源码级确认）

| 常量 | 值 | 代码位置 | 说明 |
|:---|:---|:---|:---|
| 步长钳位阈值 | **0.35 rad** | `cuda_ik_6dof.cu:279,1458` | ≈20.1° |
| 阻尼约定 | **λ 直接加对角线** | `cuda_ik_6dof.cu:244` | `sum += s_lambda`，非 λ² |
| 数值差分步长 | **ε = 1e-6** | `cuda_ik_6dof.cu:161` | 中心差分 |
| Jacobian 定义 | **J = ∂x/∂q** | `cuda_ik_6dof.cu:170-198` | FK 中心差分，非 ∂e/∂q |
| H 矩阵构造 | **全部 36 元素** | `cuda_ik_6dof.cu:233` | `row=tid/6, col=tid%6`，非上三角21个 |
| 最大迭代 Kmax | **160** | `bench_cuda_6dof.py` | B0-B5 统一 |
| 收敛阈值 Medium | **10mm / 5°** | `benchmark/common.py` | 主 benchmark |
| GPU 架构 | **sm_89** | `CMakeLists.txt` | Ada Lovelace |
| H 矩阵累加精度 | **FP64** | `cuda_ik_6dof.cu` B5 kernel | 抑制截断误差 |
| LDLᵀ 精度 | **FP64** | `ldlt_solve_6x6()` | 关键路径 |

### 4.3 Bank 冲突：降低而非消除

PaddedMat6x8（stride=8）将 Bank 冲突从 B4 的 3,522 降至 B5 的 1,295（**-63%**）。NCU 仍记录到残余冲突。论文中**必须**使用"降低"而非"消除"。

### 4.4 计时口径差异

| 求解器 | 计时工具 | 测量范围 |
|:---|:---|:---|
| CUDA B5 | `cudaEventElapsedTime` | GPU kernel 设备端执行时间 |
| cuRobo | `time.perf_counter()` | Python→C++→CUDA 全调用栈 |

两者**不能**直接比较"GPU 时间"。论文使用"吞吐比值"或"测量时间"表述。

---

## 5. 实验数据地图

### 5.1 数据文件速查

| 你要找... | 文件位置 |
|:---|:---|
| 主对比数据（B5 vs cuRobo） | `data/results/main_comparison/main_comparison.csv` |
| 消融数据（B0/B3/B5） | `data/results/ablation/ablation_medium.csv` |
| 全量程 12 N 值 | `data/results/full_range/full_range_comparison.csv` |
| 三档阈值扫描 | `data/results/threshold_scan/threshold_scan.csv` |
| NCU profiling 汇总 | `data/profiling/ncu_summary.csv` |
| NCU 原始报告 | `data/profiling/ncu_reports/*.ncu-rep` |
| NCU 报告解读 | `data/profiling/ncu_reports/README.md` |
| 目标位姿（输入） | `data/targets/ur10_seed42_N*.{bin,csv,json}` |
| 初始种子（输入） | `data/seeds/ur10_seed42_*_seed_N*.{bin,json}` |
| 论文图表 | `data/figures/figure*.png` |

### 5.2 权威 vs 废弃数据

| 权威（使用） | 废弃（勿用） | 原因 |
|:---|:---|:---|
| `main_comparison.csv` | `solver_comparison.csv` | 旧 A-series 命名，混合阈值 |
| `ablation_medium.csv` | `ablation_ur10_old_threshold.csv` | 旧 30mm/30° 阈值 |
| `full_range_comparison.csv` | — | — |
| `threshold_scan.csv` | — | — |
| `ncu_summary.csv` | — | — |

### 5.3 标准实验配置

```yaml
robot:             UR10 (official URDF, tool0)
seed:              42
seed_strategy:     zero_seed          # 主实验默认
convergence:       Medium             # pos=0.010m, rot=0.0873rad (5°)
max_iterations:    160
repeat:            30 (3 warmup + 30 timed)
weight_level:      2                  # CUDA B5
cuda_graph:        false
curobo:
  num_seeds:              1
  self_collision_check:   false
  use_cuda_graph:         false
```

---

## 6. Benchmark 框架

### 6.1 设计原则

所有求解器使用**相同的目标、种子、收敛阈值和重复次数**进行评估。

### 6.2 求解器入口

| 脚本 | 求解器 | 类型 | 计时方式 |
|:---|:---|:---|:---|
| `bench_cuda_6dof.py` | **CUDA B5** | GPU (手写 CUDA) | subprocess → CUDA event |
| `bench_curobo.py` | cuRobo | GPU (PyTorch) | `time.perf_counter()` + sync |
| `bench_pyroki.py` | PyRoki | GPU (JAX) | `time.perf_counter()` |
| `bench_kdl.py` | KDL | CPU (C++) | `time.perf_counter()` |
| `bench_numeric_dls.py` | NumPy DLS | CPU (Python) | `time.perf_counter()` |

### 6.3 常用命令

```bash
# 单求解器
python3 benchmark/run_all.py --solver cuda --N 1000 --repeat 30

# 全部求解器对比
python3 benchmark/run_all.py --solver all --robot ur10 --seed 42 --N 1000 --repeat 30

# 自定义阈值
python3 benchmark/run_all.py --solver cuda --N 1000 --pos-tol 0.005 --rot-tol 0.0175

# 消融批量执行
bash benchmark/run_b5_ablation.sh

# 论文数据一致性检查
python3 benchmark/compare_paper_data.py
```

### 6.4 收敛阈值三档

| 级别 | 位置容差 | 姿态容差 | 用途 |
|:---|:---|:---|:---|
| Loose | 30 mm | 10° (0.1745 rad) | 旧实验参考 |
| **Medium** ★ | **10 mm** | **5° (0.0873 rad)** | **主 benchmark** |
| Strict | 5 mm | 1° (0.0175 rad) | 精密应用压力测试 |

---

## 7. 关键性能数字

> 全部来自 `data/results/main_comparison/main_comparison.csv`（Medium 阈值，zero_seed，repeat=30）

### 7.1 主对比

| N | CUDA B5 吞吐 | CUDA B5 测量时间 | cuRobo 吞吐 | cuRobo 测量时间 | 吞吐比值 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | 112,414 t/s | 0.89 ms | 3,118 t/s | 32.1 ms | **36.1×** |
| 500 | 158,251 t/s | 3.16 ms | 15,844 t/s | 31.6 ms | **10.0×** |
| 1000 | 148,412 t/s | 6.74 ms | 31,611 t/s | 31.6 ms | **4.7×** |
| 5000 | 168,683 t/s | 29.64 ms | 155,059 t/s | 32.3 ms | **1.09×** |

### 7.2 全量程（N=100→10000）

| 指标 | 值 |
|:---|:---|
| CUDA B5 吞吐范围 | 148k–174k t/s（N≥500，±8%） |
| GPU 时间线性度 | R² > 0.999 |
| cuRobo 退化 N 值 | 4000, 7000, 9000, 10000（4/12） |
| 退化模式耗时 | ~230 ms（正常 ~32 ms，7×） |

### 7.3 消融关键数字

| 转换 | 吞吐提升 | 收敛率变化 |
|:---|:---:|:---:|
| B0 → B3（自适应阻尼） | +428% ~ +577% | 0.52–0.83 → 1.000 |
| B3 → B5（混合精度） | +120% ~ +149% | 保持 0.998+ |

### 7.4 NCU Profiling 摘要

| 指标 | B4 (FP64) | B5 (Mixed) |
|:---|:---:|:---:|
| Compute Throughput | 66.89% | 60.73% |
| DRAM Throughput | 1.56% | 1.16% |
| Registers/Thread | 94 | 98 |
| Shared Bank Conflicts | 3,522 | 1,295 |
| Local Memory Spill | 0 | 0 |
| Kernel Duration (N=100) | 2,920 μs | 827 μs |

---

## 8. 常用操作流程

### 8.1 首次设置

```bash
cd standard_robot_cuda_ik

# 1. 生成 UR10 标准资产
python3 tools/generate_standard_assets.py --robot ur10 --seed 42

# 2. 编译全部 10 个消融二进制
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc)
```

### 8.2 运行实验

```bash
# 快速验证（N=100, 5次）
./build/standard_robot_cuda_runner_A7 \
    --targets data/targets/ur10_seed42_N100.bin \
    --seeds data/seeds/ur10_seed42_zero_seed_N100.bin \
    --repeat 5

# 正式 benchmark（N=1000, 30次）
./build/standard_robot_cuda_runner_A7 \
    --targets data/targets/ur10_seed42_N1000.bin \
    --seeds data/seeds/ur10_seed42_zero_seed_N1000.bin \
    --repeat 30

# Python 多求解器对比
python3 benchmark/run_all.py --solver cuda,curobo --N 1000 --repeat 30
```

### 8.3 添加新求解器

1. 参照 `benchmark/bench_cuda_6dof.py` 创建 `bench_your_solver.py`
2. 从 `data/targets/` 和 `data/seeds/` 加载数据
3. 使用 `common.BenchmarkResult` 和 `common.mark_convergence()` 确保口径一致
4. 在 `run_all.py` 中注册新求解器

### 8.4 NCU Profiling

```bash
# B5 混合精度完整 profile
ncu --set full --export b5_n100.ncu-rep \
    ./build/standard_robot_cuda_runner_A7 \
    --targets data/targets/ur10_seed42_N100.bin \
    --seeds data/seeds/ur10_seed42_zero_seed_N100.bin \
    --repeat 1

# 命令行查看摘要（无需 GUI）
ncu --import b5_n100.ncu-rep --csv > ncu_b5.csv
```

---

## 9. 论文与专利

### 9.1 最新版本

| 文档 | 位置 | 状态 |
|:---|:---|:---:|
| **论文** | `../docs/paper/paper_complete.md` | 最新完整版（Jun 12, B-series） |
| **专利** | `../docs/专利/技术交底书.md` | 最新版（Jun 12） |
| **权威参考** | `../docs/项目总览/` | 5 份标准文档，写论文前必读 |

### 9.2 论文数据验证

```bash
# 检查论文中的数据与最新 CSV 是否一致
python3 benchmark/compare_paper_data.py
```

---

## 10. 外部参考求解器

以下求解器的源代码 clone 在 `../external/`（gitignored），仅供阅读理解其内部实现，benchmark 运行时使用 pip 安装的包：

| 目录 | 大小 | 用途 |
|:---|:---|:---|
| `../external/curobo/` | 373 MB | NVIDIA cuRobo — 了解 sub-batch 策略、particle 搜索 |
| `../external/pyroki/` | 21 MB | PyRoki (JAX) — JIT 编译流程 |
| `../external/hjcd_ik/` | 62 MB | HJCD-IK — 另一 GPU IK 实现参考 |

---

## 11. 铁律与禁止事项

> **以下表述在论文和文档中绝对禁止出现，因为它们与源码矛盾：**

| 禁止表述 | 原因 | 正确表述 |
|:---|:---|:---|
| "Warp 2 负责 H 的 36 个元素" | 代码用 flat threadIdx.x，H 跨越 Warp 0+1 | "线程 0–35 负责 H 全部 36 元素" |
| "上三角 21 个独立元素" | 代码计算全部 36 元素 | "全部 M²=36 个元素直接并行构造" |
| "J = ∂e/∂q" | 代码用 FK 中心差分 | "J = ∂x/∂q" |
| "λ²·I" | 代码 `sum += s_lambda` | "λ·I（λ 直接加对角线）" |
| "完全消除 Bank 冲突"/"零 Bank 冲突" | NCU 残余 1,295 | "显著降低系统性 Bank 冲突（-63%）" |
| "cuRobo GPU 时间" | cuRobo 用 host perf_counter | "cuRobo host 端测量时间" |
| "cuRobo 退化根因是 PyTorch Caching Allocator" | NSYS 已排除 | "cuRobo 内部 sub-batch 划分策略" |
| "严格 GPU 加速比" | 两种计时口径不同 | "吞吐比值" |
| Kmax = 50 | 代码/实验为 160 | "Kmax = 160" |
| 步长钳位 0.25 rad / 0.45 rad | 代码为 0.35 rad | "0.35 rad" |
| GPU 型号 4070 | 实际为 4060 Laptop | "RTX 4060 Laptop GPU" |

---

*最后更新：2026-06-14 | 硬件：RTX 4060 Laptop (Ada Lovelace, sm_89, 3,072 CUDA Cores) | CUDA Toolkit 12.6*
