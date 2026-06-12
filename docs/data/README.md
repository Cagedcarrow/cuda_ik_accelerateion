# 实验数据 CSV 说明文档

本目录包含论文《GPU 底层架构适配在批量逆运动学求解中的量化优势》所需的全部实验数据。
每个 CSV 文件用于生成论文中的一张或多张图表。

---

## 文件清单

| # | 文件 | 用途 | 对应论文章节 |
|---|------|------|-------------|
| 1 | `ablation_ur10.csv` | 消融实验 A0-A8 全数据 | §5.3 消融研究 |
| 2 | `solver_comparison.csv` | CUDA vs 开源求解器对比 | §5.2 主公平对比 |
| 3 | `ncu_profiling.csv` | Nsight Compute 关键指标 | §5.4 架构级分析 |
| 4 | `mixed_precision_ablation.csv` | FP64 vs 混合精度对比 | §5.3 混合精度消融 |
| 5 | `7dof_verification.csv` | 7DOF Panda 逻辑验证 | §6 扩展性验证 |

---

## 1. `ablation_ur10.csv` — 消融实验

### 列定义

| 列名 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `level` | str | — | 消融级别 (A0–A8) |
| `n` | int | — | 批量目标数 (100, 500, 1000, 5000) |
| `throughput_targets_per_s` | float | targets/s | **主性能指标**：每秒求解目标数，越高越好 |
| `gpu_time_ms` | float | ms | GPU 端到端时间 (event-based, 含 memcpy) |
| `kernel_time_ms` | float | ms | **纯 kernel 执行时间** (仅 A7/A8 有) |
| `conv_rate` | float | — | 收敛率 = converged / N，越高越好 |
| `avg_iters` | float | — | 平均迭代次数 |

### 消融级别说明

| 级别 | 常量内存 | PaddedMat | 寄存器 LDLT | Kernel Fusion | 自适应阻尼 | 步长钳位 | 分支对齐 | 精度 |
|:----:|:--------:|:---------:|:-----------:|:-------------:|:----------:|:--------:|:--------:|:----:|
| A0 | — | — | ✓ | ✓ | — | — | — | FP64 |
| A1 | ✓ | — | ✓ | ✓ | — | — | — | FP64 |
| A2 | ✓ | ✓ | ✓ | ✓ | — | — | — | FP64 |
| A3 | ✓ | ✓ | ✓ (注1) | ✓ | — | — | — | FP64 |
| A4 | ✓ | ✓ | ✓ | ✓ (注2) | — | — | — | FP64 |
| A5 | ✓ | ✓ | ✓ | ✓ | **✓** | — | — | FP64 |
| A6 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | FP64 |
| A7 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **FP32+FP64** |
| A8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | FP32+FP64+**Graph** |

注1: A3 的寄存器 LDLT 与 A2 代码相同（基线已使用寄存器）。
注2: A4 的 kernel fusion 与 A3 代码相同（基线已使用 fused kernel）。

### 数据来源

- A0–A6: 从 A0–A6 各 target 独立编译，运行 `repeat=30, zero_seed`，数据来自实验日志
- A7: 从 standard_robot_cuda_runner_A7 直接运行，kernel_time 来自 event-based 纯 GPU 计时
- A8: A7 + CUDA Graph replay，kernel_time 仅测量 `cudaGraphLaunch` 区间

### 建议图表

1. **柱状图**：各消融级别在 N=5000 的 throughput 对比（突出 A5 的自适应阻尼拐点和 A7 的混合精度跳升）
2. **折线图**：A5 vs A7 vs A6 的 throughput 随 N 变化曲线（x 轴: N, y 轴: targets/s）
3. **双轴图**：throughput + conv_rate 对比（验证混合精度不降低收敛率）

---

## 2. `solver_comparison.csv` — 求解器对比

### 列定义

| 列名 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `solver` | str | — | 求解器名称 (cuda_a7 / curobo / kdl / numeric_dls / pyroki) |
| `n` | int | — | 批量目标数 |
| `throughput_targets_per_s` | float | targets/s | 每秒求解目标数 |
| `host_ms` | float | ms | Host 端端到端时间 |
| `conv_rate` | float | — | 收敛率 |
| `avg_iters` | float | — | 平均迭代次数 |

### 求解器说明

| 求解器 | GPU/CPU | 实现方式 |
|--------|:-------:|----------|
| `cuda_a7` | GPU | 本文 CUDA 框架 (A7 mixed precision) |
| `curobo` | GPU | cuRobo (NVIDIA 发布，FP32 particle 搜索) |
| `kdl` | CPU | Orocos KDL (几何 IK) |
| `numeric_dls` | CPU | 数值 DLS 参考实现 |
| `pyroki` | GPU | PyRoki (基于 JAX) |

### 公平性约束

所有求解器使用**完全相同**的：
- URDF 模型文件 (`ur10_official.urdf`)
- TCP (`tool0`)
- Target 位姿 (`seed=42`)
- 初始种子 (`zero_seed`)
- 收敛阈值 (pos_tol=0.03m, orient_tol=π/6)
- 最大迭代次数 (max_iter=160)

### 数据来源

- cuda_a7: 从 A7 runner 直接运行 (repeat=30, zero_seed)
- curobo: 从 `data/results/ur10_curobo_*_zero_seed_summary.json` 提取
- kdl: 从 `data/results/ur10_kdl_*_zero_seed_summary.json` 提取
- numeric_dls: 从 `data/results/ur10_numeric_dls_*_zero_seed_summary.json` 提取
- pyroki: 从 `data/results/ur10_pyroki_*_zero_seed_summary.json` 提取

### 建议图表

1. **分组柱状图**：各求解器在 N=100/500/1000/5000 的 throughput（log scale y 轴）
2. **折线图**：CUDA vs cuRobo 的 throughput 随 N 变化（标注交叉点）
3. **散点图**：throughput vs conv_rate（验证加速不牺牲精度）

---

## 3. `ncu_profiling.csv` — Nsight Compute 分析

### 列定义

| 列名 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `kernel` | str | — | 内核名称 (ik_batch_solve / ik_batch_solve_mixed) |
| `n` | int | — | 批量目标数 |
| `compute_throughput_pct` | float | % | SM 计算吞吐利用率 |
| `dram_throughput_pct` | float | % | DRAM 带宽利用率 |
| `registers_per_thread` | int | — | 每线程寄存器数 |
| `occupancy_pct` | float | % | 达到的 occupancy |
| `bank_conflicts` | int | — | 共享内存 bank 冲突数 |
| `duration_us` | float | μs | Kernel 执行时长 (NCU 动态 profiling) |
| `l1_hit_rate_pct` | float | % | L1/TEX cache 命中率 |

### 关键结论

- **非 DRAM-bound**: DRAM throughput 仅 1-5%，远低于计算吞吐
- **Mixed precision 减少 bank conflict 49%**: A6 3,522 → A7 1,295
- **Occupancy 受寄存器限制**: 94-98 regs/thread → 理论 occupancy ~33%
- **FP64 pipeline 主导**: 最高利用率的计算单元 (60.7%)

### 数据来源

- A6 N=100: `ncu --set full` 对 `standard_robot_cuda_runner` (A6) 的分析
- A7 N=100: `ncu --metrics` 对 `standard_robot_cuda_runner_A7` 的分析
- A5 N=5000: 基于 A5 N=5000 NCU profiling 估算

### 建议图表

1. **Roofline 风格图**：compute vs memory throughput（两个 kernel 的对比点）
2. **雷达图**：多个架构维度的对比 (compute, dram, occupancy, bank conflicts)
3. **柱状图**：A6 vs A7 的 bank conflicts 对比

---

## 4. `mixed_precision_ablation.csv` — 混合精度对比

### 列定义

| 列名 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `precision` | str | — | 精度配置 (FP64 (A5) / Mixed (A7)) |
| `n` | int | — | 批量目标数 |
| `throughput_targets_per_s` | float | targets/s | 每秒求解目标数 |
| `gpu_time_ms` | float | ms | GPU 端到端时间 |
| `kernel_time_ms` | float | ms | 纯 kernel 时间 (仅 A7) |
| `conv_rate` | float | — | 收敛率 |
| `avg_iters` | float | — | 平均迭代次数 |

### 设计

A5 = 纯 FP64 + 自适应阻尼（最佳 FP64 配置）
A7 = FP32 FK/Jacobian/Hessian + FP64 LDLT/阻尼/误差（混合精度）

### 关键发现

| N | FP64 (A5) | Mixed (A7) | 加速比 | 收敛率变化 |
|:--:|:---------:|:----------:|:------:|:---------:|
| 100 | 52,064 | 107,250 | **2.06×** | 1.000 → 1.000 |
| 500 | 59,821 | 149,787 | **2.50×** | 1.000 → 0.998 |
| 5000 | 71,380 | 180,962 | **2.54×** | 1.000 → 0.9998 |

### 建议图表

1. **双组柱状图**：FP64 vs Mixed 在各 N 的 throughput（显示加速比标注）
2. **堆叠图**：kernel_time 分解（如果可获取 FP64 vs FP32 各部分占比）

---

## 5. `7dof_verification.csv` — 7DOF 扩展验证

### 列定义

| 列名 | 类型 | 说明 |
|------|------|------|
| `test_item` | str | 验证项名称 |
| `expected` | str | 预期结果 |
| `actual` | str | 实际结果 |
| `status` | str | PASS / FAIL |

### 验证内容

1. **Python FK 自洽性** — FK 计算在 Python 内一致
2. **CPU DLS IK 收敛率** — Python CPU DLS IK 在 N=10 上收敛率 ≥50%
3. **CUDA FK 与 CPU FK 一致性** — 最大误差 2.78e-16（机器精度）
4. **CUDA DLS IK 收敛率** — 50%，与 CPU 参考一致
5. **yourdfpy 交叉验证** — FK 结果与 yourdfpy 一致
6. **q=0 恒等验证** — FK(0) 输出恒等变换
7. **编译验证** — CUDA kernel 编译通过
8. **运行稳定性** — kernel 执行不崩溃

### 实验设置

- 机器人: Franka Panda (7DOF)
- 求解器: DLS (数值 Jacobian + 自适应阻尼 + LDLT 7×7)
- 测试规模: N=10 随机目标
- 数据路径: `standard_robot_cuda_ik/experiments/7dof_test/`

### 建议图表

表格形式直接用于论文 §6 "7DOF Extension" 章节。
无需额外图表，用 ✓ 标记列表即可。

---

## 通用画图建议

### 工具推荐

1. **Matplotlib + Seaborn** (Python) — 适用于学术论文级的图表
2. **Plotly** (Python) — 适用于交互式探索
3. **pgfplots** (LaTeX) — 直接在论文中使用

### 颜色方案

建议使用色盲友好的调色板：

| 求解器/级别 | 颜色 | HEX |
|------------|------|-----|
| CUDA (ours) | 蓝色 | #0072B2 |
| cuRobo | 橙色 | #E69F00 |
| KDL | 绿色 | #009E73 |
| numeric_dls | 红色 | #D55E00 |
| PyRoki | 紫色 | #CC79A7 |

### 统一设置

- 字体: 8-10pt (论文标准)
- 线条宽度: 1.5-2pt
- DPI: ≥300 (印刷品质)
- 格式: PDF 或 EPS (矢量图形)
- 图例位置: 最佳自动放置，避免遮挡数据

### 图表中英标签对照

| 中文 | English |
|------|---------|
| 批量大小 (N) | Batch Size (N) |
| 吞吐量 (targets/s) | Throughput (targets/s) |
| 收敛率 | Convergence Rate |
| GPU 时间 (ms) | GPU Time (ms) |
| 消融级别 | Ablation Level |
| 混合精度 | Mixed Precision |

---

## 数据完整性校验

所有 CSV 数据来自以下原始文件，三方可交叉验证：

```
standard_robot_cuda_ik/data/results/ur10_*_summary.json   (基准测试)
standard_robot_cuda_ik/data/profiling/ur10_*.ncu-rep       (NCU 分析)
standard_robot_cuda_ik/experiments/7dof_test/*              (7DOF 验证)
docs/logs/ablation_official_ur10.md                         (消融日志)
```

如需重新生成数据，运行：

```bash
cd standard_robot_cuda_ik
cmake --build build -j$(nproc)

# A7 benchmarks (repeat=30, zero_seed)
for N in 100 500 1000 5000; do
  ./build/standard_robot_cuda_runner_A7 \
    --targets data/targets/ur10_seed42_N${N}.bin \
    --seeds data/seeds/ur10_seed42_zero_seed_N${N}.bin \
    --max-iter 160 --weight-level 2 --repeat 30
done
```

---

*最后更新: 2026-06-11*
*数据生成脚本: `docs/data/generate_csv.py` (会话中临时执行)*
