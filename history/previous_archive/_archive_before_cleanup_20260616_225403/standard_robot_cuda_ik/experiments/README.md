# Experiments 目录说明

本目录包含《基于 CUDA 小矩阵加速的机械臂批量逆运动学求解》论文修改所需的补充实验。

## 目录结构

```
experiments/
├── README.md                          # 本文件
├── plot_all_new_figures.py            # 生成 9 张论文图片
├── figures/                           # 图片输出目录（9 张 PNG）
│
├── 01_official_ur10_main/             # 主实验：统一 UR10 benchmark
│   ├── run_main_experiments.py        #   全方法 × 全 N 值 × 30 次重复
│   ├── run_statistical_significance.py #  统计显著性分析（待实现）
│   ├── export_benchmark_targets.py    #   目标位姿参考 CSV 导出
│   ├── performance_summary.csv        #   [输出] 主性能汇总
│   ├── benchmark_targets.csv          #   [输出] 目标位姿参考
│   └── errors/                        #   [输出] 逐目标误差 CSV
│
├── 02_batch_size_scaling/             # 批量扩展性实验
│   ├── run_full_range.py              #   N=100→10000 全量程扫描
│   └── full_range_scaling.csv         #   [输出]
│
├── 03_solver_comparison/              # 求解器对比实验
│   ├── run_cuda_graph_comparison.py   #   CUDA Graph A7 vs A8 对比
│   ├── run_curobo_graph_comparison.py #   cuRobo Graph 开/关对比
│   ├── run_threshold_scan.py          #   三档阈值扫描（待实现）
│   ├── cuda_graph_comparison.csv      #   [输出]
│   └── curobo_graph_comparison.csv    #   [输出]
│
├── 04_ablation/                       # 消融实验
│   ├── run_full_ablation.py           #   A0→A7 全部 8 个级别
│   └── ablation_full.csv              #   [输出]
│
├── 05_roofline/                       # 精度验证与性能剖析
│   ├── run_jacobian_precision.py      #   Jacobian FP32 vs FP64 精度验证
│   ├── run_nsight_systems.py          #   Nsight Systems 机制分析（待实现）
│   ├── run_nsight_compute.py          #   Nsight Compute kernel 剖析（待实现）
│   ├── jacobian_precision_summary.csv #   [输出]
│   ├── nsight_systems_summary.csv     #   [输出]
│   └── nsight_compute_summary.csv     #   [输出]
│
├── 06_7dof_extension/                 # 7 自由度扩展（预留，待后续工作）
│
└── 7dof_test/                         # 已有 7-DOF Panda 测试（不变）
    └── ...
```

## 实验数据汇总

### 主实验（01_official_ur10_main/performance_summary.csv）

30 次重复实验，Medium 阈值（10mm/5°），3 次预热。

关键发现：
- **CUDA-Mixed 吞吐**：N≥500 时稳定在 128k–157k targets/s，N=100 时约 81k t/s
- **cuRobo-Graph 大幅优于 cuRobo-NoGraph**：N=100 时 Graph 模式 13 倍吞吐提升
- **低批量优势**：N≤1000 时 CUDA-Mixed GPU stream 时间始终低于 cuRobo-Graph
- **大批量可比**：N≥4000 时两者吞吐可比（~157k vs 145–163k t/s）
- **cuRobo 批量振荡**：N=4000/10000 在 NoGraph 模式下触发 ~230ms 退化

### Jacobian 精度验证（05_roofline/jacobian_precision_summary.csv）

500 个随机 UR10 关节构型，FP32 vs FP64 中心差分 Jacobian 比较。

| ε | 中位相对 Frobenius 误差 |
|---|------------------------|
| 1e-4 | 8.1×10⁻⁴ |
| 1e-5 | 1.9×10⁻³ |
| 1e-6 | 4.2×10⁻² |
| 1e-7 | 7.0×10⁻¹（减法抵消灾难） |

结论：ε=1e-6 下 FP32 Jacobian 精度约为 2 位有效数字，但由于 LDLᵀ 求解保留 FP64，该误差未导致收敛退化。

### 消融实验（04_ablation/ablation_full.csv）

A0→A7 逐级叠加，30 次重复。

关键发现：
- A1–A4（常量内存 + PaddedMat + 寄存器 LDLT + Kernel 融合）的独立贡献合计 <5%
- A5（自适应阻尼）是收敛率的决定性因素：收敛率从 A1–A4 的 41–56% 恢复至 100%
- A7（混合精度）额外贡献 120–235% 吞吐提升，收敛率无退化
- A0（独立 kernel 路径）性能异常高，与 A1–A6 非同代码路径，不具可比性

## 复现方法

### 依赖
```bash
pip install numpy torch curobo matplotlib seaborn
```

### 运行实验
```bash
# 主实验（需要 GPU，约 2 小时）
cd standard_robot_cuda_ik
python experiments/01_official_ur10_main/run_main_experiments.py

# Jacobian 精度验证（纯 Python，数分钟）
python experiments/05_roofline/run_jacobian_precision.py

# 消融实验（需要 GPU，约 1 小时）
python experiments/04_ablation/run_full_ablation.py

# 全量程扫描
python experiments/02_batch_size_scaling/run_full_range.py

# 生成图片
python experiments/plot_all_new_figures.py
```

### 编译 CUDA 二进制
```bash
cd standard_robot_cuda_ik
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc)
```

## 数据文件格式

### performance_summary.csv
| 字段 | 说明 |
|------|------|
| method | 方法名（CUDA-Mixed / CUDA-Mixed-Graph / cuRobo） |
| graph | CUDA Graph 状态（N/A / Off / On） |
| N | 批量规模 |
| time_ms_mean/std/min/max/p50/p95 | GPU stream 时间统计（ms） |
| h2d_time_ms | H2D 传输时间（ms） |
| d2h_time_ms_mean | D2H 传输时间均值（ms） |
| e2e_time_ms | 端到端时间（H2D+Kernel+D2H，ms） |
| raw_targets_per_s | 原始吞吐 |
| valid_targets_per_s | 有效吞吐（raw × success_rate） |
| success_rate[_loose/_strict] | 三档阈值成功率 |
| pos_error_p50_m/p95_m | 位置误差分位数（m） |
| rot_error_p50_rad/p95_rad | 姿态误差分位数（rad） |

### error_summary.csv（per-target）
| 字段 | 说明 |
|------|------|
| target_id | 目标编号（0–N-1） |
| pos_error_m | 位置误差（m） |
| rot_error_rad | 姿态误差（rad） |
| num_iterations | 迭代次数 |
| converged | 是否收敛（0/1，Medium 阈值） |

## 图片清单

| 编号 | 文件名 | 内容 |
|------|--------|------|
| 图 1 | fig1_block_target_mapping.png | CUDA block/target 映射架构 |
| 图 2 | fig2_throughput_n_curve.png | 吞吐-N 曲线（raw+valid） |
| 图 3 | fig3_e2e_latency.png | 端到端延迟-N 曲线 |
| 图 4 | fig4_gpu_stream_time.png | GPU Stream 时间-N 曲线 |
| 图 5 | fig5_curobo_degradation.png | cuRobo 退化点分析 |
| 图 6 | fig6_ablation.png | 消融实验柱状图 |
| 图 7 | fig7_convergence_vs_valid_throughput.png | 收敛率 vs 有效吞吐 |
| 图 8 | fig8_error_distribution.png | 误差分布 |
| 图 9 | fig9_ncu_profiling.png | Nsight Compute 剖析 |
