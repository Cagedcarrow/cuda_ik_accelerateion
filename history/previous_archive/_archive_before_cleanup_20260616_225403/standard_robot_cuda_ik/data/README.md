# data/ — 论文数据中心化索引

> **唯一数据依据（Single Source of Truth）**  
> 本目录包含《基于CUDA的工业机器人运动学逆解加速方法研究》中所有实验测量数据。  
> 论文中所有图表、表格、分析均以本目录数据为准。  
> 更新日期：2026-06-12

---

## 目录结构

```
data/
├── README.md                            # 本文件：数据-论文映射及使用说明
├── targets/                             # IK 目标位姿（输入）
│   └── ur10_seed42_N{100..10000}.{bin,csv,json}
├── seeds/                               # 初始关节种子（输入）
│   └── ur10_seed42_{strategy}_N{100..10000}.{bin,json}
│       strategy ∈ {zero_seed, home_seed, random_seed, near_ground_truth_seed}
├── results/                             # 基准测试结果（输出）
│   ├── main_comparison/                 # 主对比：CUDA B5 vs cuRobo @ Medium阈值
│   ├── ablation/                        # 消融实验：B0/B3/B5 @ Medium + 旧阈值参考
│   ├── threshold_scan/                  # 三档阈值扫描 (Loose/Medium/Strict)
│   ├── full_range/                      # 全量程 N=100→10000 线性度分析
│   ├── cpu_baseline/                    # CPU 参考基线 (KDL/numeric_dls/PyRoki)
│   ├── seed_strategy/                   # 种子策略敏感性分析
│   ├── panda_7dof/                      # 7-DOF Panda 机械臂验证
│   └── errors/                          # 错误日志
├── profiling/                           # Nsight Compute 性能剖析
│   ├── ncu_reports/                     # .ncu-rep 原始报告
│   └── ncu_summary.csv                  # 关键指标汇总
└── figures/                             # 论文图表
    ├── plot_all_figures.py              # 图表生成脚本
    ├── figure1_throughput_comparison.png # 图1: 批量吞吐量对比
    ├── figure2_speedup_bars.png          # 图2: 加速比柱状图
    ├── figure3_ablation_throughput.png   # 图3: 消融吞吐量
    ├── figure4_convergence_rate.png      # 图4: 收敛率变化
    └── figure5_avg_iterations.png        # 图5: 平均迭代次数
```

---

## 数据↔论文映射表

| 论文表格/图表 | 数据源文件 | 说明 |
|:---|:---|:---|
| 表6.1 主对比 (B5 vs cuRobo) | `results/main_comparison/main_comparison.csv` | Medium阈值, repeat=30, zero_seed |
| 表6.2 CPU基线 | `results/cpu_baseline/cpu_baseline.csv` | 旧30°阈值，仅数量级参照 |
| 表6.3 三档阈值扫描 | `results/threshold_scan/threshold_scan.csv` | N=100/500/1000/5000/10000 |
| 表6.4 1-N批量消融 | `results/full_range/full_range_comparison.csv` | N=100→10000 线性度分析 |
| 表6.5 种子策略敏感性 | `results/seed_strategy/seed_strategy.csv` | zero_seed vs home_seed |
| 表6.6 消融实验（论文6.3节） | `results/ablation/ablation_medium.csv` | **Medium阈值权威数据**（B0/B3/B5） |
| 表6.7 7-DOF Panda验证 | `results/panda_7dof/7dof_verification.csv` | Panda 7-DOF |
| 图1 吞吐量对比 | `figures/figure1_throughput_comparison.png` | 从plot_all_figures.py生成 |
| 图2 加速比 | `figures/figure2_speedup_bars.png` | 同上 |
| 图3 消融吞吐量 | `figures/figure3_ablation_throughput.png` | 同上 |
| 图4 收敛率 | `figures/figure4_convergence_rate.png` | 同上 |
| 图5 平均迭代次数 | `figures/figure5_avg_iterations.png` | 同上 |
| 表6.x Nsight Compute剖析 | `profiling/ncu_summary.csv` | B4/B5 N=100/5000 |

---

## 实验配置

### 统一配置参数

| 参数 | 值 |
|:---|:---|
| 机器人模型 | UR10 (6-DOF) |
| 目标生成随机种子 | seed=42 |
| 权威实验 repeat 数 | 30 |
| 权威实验种子策略 | zero_seed（零向量初始化） |
| 收敛容差（主基准） | **Medium: 位置 10mm, 姿态 0.0873rad (≈5°)** |
| GPU | NVIDIA GeForce RTX 4070 (Laptop) |
| 时间口径 | `gpu_end_to_end_time` (GPU端到端，不含CPU预处理) |
| 吞吐量定义 | `N / gpu_end_to_end_time` (targets per second) |

### 三级收敛阈值

| 级别 | 位置容差 (m) | 姿态容差 (rad) | 用途 |
|:---|:---|:---|:---|
| Loose | 0.030 (30mm) | 0.1745 (≈10°) | cuRobo的默认阈值 |
| **Medium** | **0.010 (10mm)** | **0.0873 (≈5°)** | **论文主基准** |
| Strict | 0.005 (5mm) | 0.0175 (≈1°) | 最严格精度要求 |

### 消融级别映射

| 论文名称 (B系列) | 旧名称 (A系列) | 说明 |
|:---|:---|:---|
| B0 | A0 | FP64 基础版本（无优化） |
| B1 | A1 | + QR分解替代SVD |
| B2 | A2 | + 等式约束处理 |
| **B3** | **A5** | **+ 自适应阻尼**（FP64） |
| B4 | A6 | + Cholesky (LDLT) 分解 |
| **B5** | **A7** | **+ 混合精度**（FP32迭代 + FP64 LDLT） |
| B6 | A8 | + 收敛提前终止 |

---

## 各子目录详细说明

### `targets/` — IK目标位姿

UR10 机器人末端执行器的目标位姿（SE(3)），由种子42随机生成。

- 文件格式：`.bin`（二进制blob）、`.csv`（7列文本）、`.json`（含metadata）
- 规模：N ∈ {100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000}
- 命名规范：`ur10_seed42_N{count}.{ext}`
- 数据列 (CSV): `px, py, pz, qw, qx, qy, qz` (位姿+四元数)

### `seeds/` — 初始关节种子

IK 求解器的初始关节角度，对应四种种子策略。

| 种子策略 | 说明 |
|:---|:---|
| `zero_seed` | 全零向量 (论文默认) |
| `home_seed` | UR10 home位姿关节角 |
| `random_seed` | 均匀随机采样 |
| `near_ground_truth_seed` | ground-truth加小噪声 |

- 命名规范：`ur10_seed42_{strategy}_N{count}.{bin,json}`

### `results/main_comparison/` — 主对比数据

**论文 表6.1 对应数据**。

- `main_comparison.csv` — 权威主对比（Medium阈值）：N, Solver, GPU_ms, Throughput, ConvRate, AvgIters, Speedup_B5_vs_cuRobo
- `solver_comparison.csv` — 旧阈值参考（含更多solver）

**关键结果（Medium阈值）**：

| N | B5 Throughput | cuRobo Throughput | 加速比 |
|:---|:---|:---|:---|
| 100 | 112,414 t/s | 3,118 t/s | 36.1× |
| 500 | 158,251 t/s | 15,844 t/s | 10.0× |
| 1000 | 148,412 t/s | 31,611 t/s | 4.7× |
| 5000 | 168,683 t/s | 155,059 t/s | 1.09× |

### `results/ablation/` — 消融实验

**论文 6.3节（表6.6）对应数据**。

- `ablation_medium.csv` — **权威数据**：B0/B3/B5 在 Medium阈值下 N=100/500/5000
- `ablation_ur10_old_threshold.csv` — 旧30°阈值参考（B0-B6全部级别）
- `mixed_precision_ablation.csv` — 混合精度独立消融

**关键发现（Medium阈值）**：
- B0（无优化FP64）：收敛率仅 0.522–0.830（N=500/5000），旧30°阈值下误报为 0.804–1.000
- B3（+自适应阻尼）：收敛率恢复至 1.000，吞吐量提升 4–6×（vs B0）
- B5（+混合精度）：吞吐量在B3基础上再提升 2.2–2.5×

### `results/threshold_scan/` — 三档阈值扫描

**论文 表6.3 对应数据**。

- `threshold_scan.csv` — 汇总数据：N, Threshold, B5_TP, B5_ConvRate, cuRobo_TP, cuRobo_ConvRate, Speedup
- `cuda_b5_{threshold}_N{count}.log` — CUDA B5 原始日志（24个）
- `curobo_{threshold}_N{count}.log` — cuRobo 原始日志（24个）
- `run_scan.sh` — 扫描执行脚本
- `comparison_combo.md` — 结果解读

### `results/full_range/` — 全量程线性度分析

**论文 表6.4 对应数据**。

- `full_range_comparison.csv` — B5 在 N=100→10000 全量程数据
- `oscillation_analysis.md` — cuRobo批量震荡现象分析（N=4000/7000/9000/10000 出现 ~230ms 退化模式）

**关键发现**：
- CUDA B5：吞吐量 148k–174k t/s（±8%），GPU时间 vs N 的 R² > 0.999（近乎完美线性）
- cuRobo：正常模式 ~32ms/batch，退化模式 ~230ms/batch（偶发）

### `results/cpu_baseline/` — CPU参考基线

**论文 表6.2 对应数据**。

- `cpu_baseline.csv`：KDL (C++), numeric_dls (Python), PyRoki (JAX GPU)

**注意**：CPU基线使用旧30°阈值，标注为"仅作数量级参照，不参与Medium主结论"。

### `results/seed_strategy/` — 种子策略分析

**论文 表6.5 对应数据**。

- `seed_strategy.csv`：zero_seed vs home_seed，CUDA B4 / cuRobo，N=100/500/1000/5000

**关键发现**：zero_seed 优于 home_seed（吞吐量高 16–45%，因搜索空间更窄）。

### `results/panda_7dof/` — 7-DOF验证

**论文 表6.7 对应数据**。

- `7dof_verification.csv`：Panda 7-DOF 机械臂上的 B4/B5 验证

### `results/errors/` — 错误日志

- `ur10_curobo_N5000_seed42_repeat30_error.json`：cuRobo N=5000 运行的部分错误详情

### `profiling/` — Nsight Compute GPU性能剖析

**论文 表6.x（Nsight分析）对应**。

- `ncu_summary.csv` 列说明：

| 列名 | 说明 |
|:---|:---|
| Config | 配置标签 (B4 FP64 full / B5 Mixed FP32+FP64) |
| N | 批量大小 |
| ComputeThroughput_pct | 计算吞吐量占比 (%) |
| DRAMThroughput_pct | 显存吞吐量占比 (%) |
| RegistersPerThread | 每线程寄存器数 |
| Occupancy_pct | 占用率 (%) |
| BankConflicts | 共享内存Bank冲突数 |
| L1HitRate_pct | L1缓存命中率 (%) |
| KernelDuration_us | 内核执行时间 (微秒) |
| NCUReportFile | 对应的 .ncu-rep 报告文件名 |

- `ncu_reports/` 内 `.ncu-rep` 原始报告：

| 文件 | 大小 | 内容 |
|:---|:---|:---|
| `b4_fp64_n100_zero_seed.ncu-rep` | 13.9 MB | B4 FP64 N=100 zero_seed 全剖析 |
| `b4_fp64_n100_full_home_seed.ncu-rep` | 15.0 MB | B4 FP64 N=100 home_seed 全剖析 |
| `b4_fp64_n100_launch.ncu-rep` | 228 KB | B4 FP64 N=100 仅kernel launch |
| `b3_fp64_n5000_full_zero_seed.ncu-rep` | 40.0 MB | B3 FP64 N=5000 zero_seed 全剖析 |
| `b5_mixed_n100_memory.ncu-rep` | 6.1 MB | B5 混合精度 N=100 内存访问剖析 |

### `figures/` — 论文图表

5张论文图表 + 生成脚本。重新生成命令：

```bash
cd data/figures && python3 plot_all_figures.py
```

---

## 数据版本历史

| 日期 | 变更 |
|:---|:---|
| 2026-06-10 | 旧30°阈值数据收集（B0-B6全级别，N=100/500/5000） |
| 2026-06-11 | Medium阈值 (10mm/5°) 消融重测（B0/B3/B5, repeat=30） |
| 2026-06-11 | Nsight Compute剖析（B4/B5 N=100, B3 N=5000） |
| 2026-06-11 | 三档阈值扫描（Loose/Medium/Strict, N=100→10000） |
| 2026-06-11 | 种子策略分析（zero_seed vs home_seed） |
| 2026-06-12 | 数据目录重组、正式CSV生成、图表更新、本README编写 |

---

## 注意事项

1. **Medium阈值是当前主基准**。旧30°阈值（Loose）数据已移至 `*_old_threshold.csv` 或标注为"参考值"，不要用于主结论。
2. **B1/B2/B4/B6 的 Medium 阈值数据尚未实测**。图表中这些级别的数据点来自旧30°阈值，仅供参考。消融主结论（B0→B3→B5递进）基于实测Medium数据。
3. 时间口径统一使用 `gpu_end_to_end_time`（GPU端到端时间）。该值排除了主机端目标拷贝、种子准备等预处理开销，反映纯GPU求解性能。
4. 图表脚本 `plot_all_figures.py` 中的 `PAPER_B5_THROUGHPUT` 等字典是 CSV 不可用时的 fallback，**以 CSV 文件为最高优先级**。任何数据更新应同时修改CSV和fallback字典。
5. Nsight Compute 报告文件较大（总计 ~75 MB），未纳入版本管理。需要时从 `profiling/ncu_reports/` 用 NVIDIA Nsight Compute GUI 打开。
6. 全量程 `N=10000` 数据因GPU显存限制仅在特定配置下可用，部分对比不含该数据点。
7. CPU KDL / numeric_dls 数据使用旧30°阈值（且numeric_dls收敛率极低 ~1-1.5%），仅作"GPU vs CPU数量级对比"使用，**不参与精确加速比计算**。
