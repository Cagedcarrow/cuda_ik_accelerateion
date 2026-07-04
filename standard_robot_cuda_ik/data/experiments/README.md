# 实验数据说明

> 所有数据对应论文《结构感知单核融合的机械臂批量逆运动学CUDA加速方法》（`论文/paper.tex`）

---

## 目录结构

```
data/experiments/
├── README.md              ← 本文件
├── inputs/                 ← N=100-1000 的目标位姿和种子 raw 文件
├── results/                ← 全部基准测试 CSV 结果
└── 补充实验/               ← K=1 消融、cuRobo K=16 公平对比、FP32 混合精度
```

---

## 一、论文数据溯源表

| 论文表/图 | 数据来源 CSV | 说明 |
|----------|-------------|------|
| 表 5（静态批量 IK 综合性能） | `results/dense_static_summary.csv` | OPT4C K=16，N=100-1000，10 个规模 |
| 表 6（三种配置系统级对比） | `补充实验/results/curobo_k16_summary.csv` + `results/dense_curobo_summary.csv` | cuRobo K=16（N=100/500/1000）+ cuRobo K=1（N=100-1000） |
| 表 7（种子数消融实验） | `补充实验/results/k1_static_summary.csv` | OPT4C K=1 vs K=16（N=100/500/1000） |
| 表 8（混合精度消融实验） | `补充实验/results/mixed_precision_summary.csv` | FP64 vs mixed_safe（N=100/500/1000） |
| 表 9（Nsight Compute 动态指标） | Nsight 实测数据（已写入论文） | ncu 采集，N=1000，K=16 |
| 表 10（PTX 寄存器统计） | PTX 汇编分析（已写入论文） | `--ptxas-options=-v` 编译输出 |
| 图 1（批量吞吐量对比） | `results/dense_static_summary.csv` + `results/dense_curobo_summary.csv` | 折线图，N=100-1000 |
| 图 2（Strict 成功率对比） | 同上 | 折线图 |
| 图 4（批量扩展性分析） | `results/dense_static_summary.csv` | GPU 时间+吞吐量 |
| 图 6（线程映射架构） | 无数据依赖 | draw.io 流程图 |
| 图 7（算法流程） | 无数据依赖 | draw.io 流程图 |

> 注：图 3（位置误差 p95 对比）已删除，改为正文文字描述。

---

## 二、核心数据文件说明

### `results/` 目录 —— 主基准测试

| 文件 | 内容 | 对应论文 |
|------|------|---------|
| `dense_static_summary.csv` | OPT4C K=16 10 规模汇总（SR/吞吐/p95/GPU时间/迭代次数） | 表 5, 图 1/2/4 |
| `dense_curobo_summary.csv` | cuRobo K=1 10 规模汇总（SR/吞吐/p95/成功p95） | 表 6, 图 1/2 |
| `dense_vs_curobo_summary.csv` | OPT4C K=16 vs cuRobo K=1 合并对比 | 4.2 节文字 |
| `cuda_opt4c_summary_N*.csv` | 每个 N 值的详细 30 次重复统计 | 附录/验证 |
| `cuda_opt4c_best_N*.csv` | 每个 N 值逐目标最佳解详情（关节角/误差） | 质量审计 |
| `cuda_opt4c_timing_N*.csv` | 每个 N 值逐次 H2D/Kernel/D2H 时间分解 | GPU 时间分解 |

### `补充实验/results/` 目录 —— 消融与公平对比

| 文件 | 实验配置 | 对应论文 |
|------|---------|---------|
| `k1_static_summary.csv` | OPT4C K=1，N=100-1000 | 表 7（种子数消融） |
| `k1_vs_curobo_comparison.csv` | OPT4C K=1 vs cuRobo K=1 对比 | 4.3 节文字 |
| `curobo_k16_summary.csv` | cuRobo num_seeds=16，N=100/500/1000 | 表 6（公平对比） |
| `fair_comparison_k16_vs_k16.csv` | OPT4C K=16 vs cuRobo K=16 对比 | 4.2 节文字 |
| `mixed_precision_summary.csv` | FP64/mixed_safe/mixed_aggressive/mixed_safe+fallback | 表 8（混合精度消融） |

### `inputs/` 目录 —— 原始输入数据

| 文件模式 | 数量 | 说明 |
|---------|------|------|
| `targets_N*_T4x4_f64.raw` | 10 | 目标位姿（4×4 齐次矩阵，fp64），N=100,200,...,1000 |
| `seeds_N*_K16_q_f64.raw` | 10 | Sobol 种子（K=16，fp64），对应 N 值 |

> 所有 raw 文件由 `seed=42` 固定随机种子生成，确保可复现。

### `补充实验/inputs/` 目录

| 文件模式 | 说明 |
|---------|------|
| `seeds_N*_K1_q_f64.raw` | Sobol 种子 K=1 版本（仅保留每组第一个种子） |

---

## 三、脚本说明

| 脚本 | 功能 | 运行方式 |
|------|------|---------|
| `补充实验/generate_dense_inputs.py` | 从 N=1000 raw 文件切片生成 N=200-900 输入 | `python3 generate_dense_inputs.py` |
| `补充实验/run_k1_benchmark.py` | 运行 OPT4C K=1 基准测试 + 与 cuRobo 对比 | `python3 run_k1_benchmark.py` |
| `补充实验/run_curobo_k16.py` | 运行 cuRobo K=16 公平对比 + 三方对比 | `python3 run_curobo_k16.py` |
| `补充实验/run_mixed_precision.py` | 运行 FP32 混合精度消融实验 | `python3 run_mixed_precision.py` |
| `补充实验/run_dense_benchmarks.py` | 运行 OPT4C K=16 密集 N 值基准测试 | `python3 run_dense_benchmarks.py` |
| `补充实验/collect_dense_results.py` | 汇总 CUDA + cuRobo 全部结果 | `python3 collect_dense_results.py` |

---

## 四、关键实验发现速查

| 实验 | 核心结论 |
|------|---------|
| **OPT4C K=16（默认）** | SR 0.940-0.960，吞吐 15k-18k，p95 4.3-5.5mm |
| **OPT4C K=1（消融）** | SR 暴跌至 0.45-0.52，p95 飙升至 642-685mm——多起点是成功率的决定性来源 |
| **cuRobo K=16（公平对比）** | SR 0.988，p95 0.9mm——质量天花板，但吞吐仅 10k |
| **cuRobo K=1（默认）** | 吞吐 72k 但 SR 仅 0.84——极端吞吐/质量不可用 |
| **FP32 混合精度** | 吞吐仅 +2%，瓶颈在 FP64 高斯消元而非精度选择 |
| **Nsight Compute** | Long Scoreboard 83.2%，Issue Slot 2.32%——FP64 管线发射延迟是主瓶颈 |

---

## 五、复现指南

```bash
# 1. 编译求解器
cd standard_robot_cuda_ik
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 2. 运行主基准（K=16, N=100-1000）
python3 data/experiments/补充实验/run_dense_benchmarks.py

# 3. 运行消融实验（K=1）
python3 data/experiments/补充实验/run_k1_benchmark.py

# 4. 运行 cuRobo 公平对比（需 cuRobo Python API）
python3 data/experiments/补充实验/run_curobo_k16.py

# 5. 运行 FP32 混合精度
python3 data/experiments/补充实验/run_mixed_precision.py
```
