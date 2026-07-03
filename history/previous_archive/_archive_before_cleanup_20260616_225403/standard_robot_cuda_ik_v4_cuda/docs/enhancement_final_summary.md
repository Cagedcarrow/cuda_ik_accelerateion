# V4-Final-K16 CUDA Enhancement Final Summary

## 1. Purpose

本文件是增强计划的最终入口文档。对应计划：

```text
docs/V4-Final-K16 CUDA 增强实验最终执行计划.md
```

本轮工作不是重做已经完成的 `V4-Final-K16 CUDA Port` baseline，而是在不破坏原论文闭环的前提下补齐三组增强实验：

1. Kernel optimization。
2. Adaptive-K。
3. Full ablation。

## 2. Baseline Protection

本轮没有覆盖或改写以下 baseline 结论文件：

- `docs/final_paper_readiness_report.md`
- `data/results/final_summary.csv`
- `data/results/cuda_v4_static_benchmark.csv`
- `data/results/cuda_v4_curobo_compare.csv`
- `data/results/nsight_summary.csv`

新增结果均隔离在：

- `data/results/opt/`
- `data/results/adaptive/`
- `data/results/ablation/`
- `docs/opt/`
- `docs/adaptive/`
- `docs/ablation/`
- `logs/opt/`
- `logs/adaptive/`
- `logs/ablation/`

## 3. Implementation Summary

代码侧新增或调整：

- `src/cuda/cuda_v4_runner.cu`
  - 新增 analytical / piecewise Limit Barrier gradient。
  - 新增 `--limit-gradient finite_diff|analytic`。
  - 新增 `--variant ...` 输出标记，用于区分增强实验版本。
- `CMakeLists.txt`
  - 新增 `standard_robot_cuda_v4_runner_r160`。
  - 新增 `standard_robot_cuda_v4_runner_r128`。
  - 两个 target 使用 `--maxrregcount` 和 `-Xptxas -v` 记录 register/spill 信息。
- `scripts/run_v4_enhancement_plan.py`
  - 统一执行 OPT、Adaptive-K、Ablation 和增强报告生成。

构建验证：

```text
cmake -S . -B build
cmake --build build --target standard_robot_cuda_v4_runner standard_robot_cuda_v4_runner_r160 standard_robot_cuda_v4_runner_r128
```

构建成功。ptxas 证据写入 `logs/opt/ptxas_registers.log`。

## 4. Required Output Audit

所有增强计划要求的核心文件已生成：

| required output | status |
|---|---|
| `docs/opt/kernel_optimization_report.md` | generated |
| `docs/adaptive/adaptive_k_report.md` | generated |
| `docs/ablation/full_ablation_report.md` | generated |
| `docs/enhancement_final_summary.md` | generated |
| `data/results/enhancement_final_summary.csv` | generated |
| `data/results/opt/opt0_baseline_rebuild.csv` | generated |
| `data/results/opt/opt1_limit_gradient_correctness.csv` | generated |
| `data/results/opt/opt1_limit_gradient_benchmark.csv` | generated |
| `data/results/opt/opt2_register_reduction.csv` | generated |
| `data/results/opt/opt3_candidate_layout.csv` | generated |
| `data/results/opt/opt4_warp_per_seed_correctness.csv` | generated |
| `data/results/opt/opt4_warp_per_seed_benchmark.csv` | generated |
| `data/results/opt/opt5_best_combined_static_benchmark.csv` | generated |
| `data/results/opt/opt5_best_combined_vs_baseline.csv` | generated |
| `data/results/adaptive/adaptive_k_benchmark.csv` | generated |
| `data/results/ablation/static_ablation_N1000.csv` | generated |
| `data/results/ablation/module_contribution_table.csv` | generated |
| `data/results/ablation/trajectory_ablation.csv` | generated |
| `data/results/ablation/curobo_boundary_table.csv` | generated |
| `logs/opt/ptxas_registers.log` | generated |

图表数据 CSV 也已生成：

- `figures/fig_v1_to_v4_pipeline.csv`
- `figures/fig_limit_barrier_effect.csv`
- `figures/fig_smoothness_rerank_effect.csv`
- `figures/fig_curobo_boundary.csv`
- `figures/fig_static_ablation_sr_pos.csv`
- `figures/fig_nsight_bottleneck.csv`

## 5. Kernel Optimization Decision

详细报告：

```text
docs/opt/kernel_optimization_report.md
```

### OPT0

Baseline rebuild 通过：

- N=1000 Strict SR = 0.954。
- pos_p95_all = 4.508959 mm。
- near_limit = 0.007。
- no NaN/Inf。

说明增强实验框架没有破坏 baseline。

### OPT1

Analytical Limit Gradient 通过：

- strict_sr_diff_pp = 0.0。
- pos_p95_diff_mm = 9.403565e-07。
- correctness pass。
- N=1000 gpu_stream_ms = 597.124017。
- N=5000 gpu_stream_ms = 2983.860970。

结论：OPT1 可进入工程实现和论文 discussion，属于稳定小幅优化。

### OPT2

Register cap 结果：

- baseline: 184 registers/thread, no spill。
- r160: 160 registers/thread, 116 spill stores, 168 spill loads。
- r128: 128 registers/thread, 244 spill stores, 304 spill loads。

结论：硬压寄存器会引入 spill，不能简单认为 register count 降低就能显著提升整体性能。OPT2 适合写入 Nsight / performance discussion。

### OPT3

SoA candidate layout 没有进入主结果。

原因：selection 不是当前主要瓶颈，主耗时仍来自 FP64 scalar LM 和 one-thread-per-block 映射。

### OPT4

Warp-per-seed 未推广为可替换 kernel。

原因：独立 cooperative warp solve 没有在本轮完成到 correctness + benchmark 可替换状态。按照计划，该方向允许失败，但必须写成 future work。

### OPT5

Best combined 结果：

| N | baseline ms | OPT5 ms | speedup | Strict SR | pos_p95_all mm | near_limit |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 73.666339 | 64.475074 | 1.142555 | 0.960 | 4.384513 | 0.010 |
| 500 | 319.969279 | 302.718519 | 1.056986 | 0.954 | 4.337968 | 0.004 |
| 1000 | 639.158346 | 601.146330 | 1.063233 | 0.954 | 4.563133 | 0.007 |
| 5000 | 3186.436120 | 2986.249340 | 1.067036 | 0.954 | 4.563133 | 0.007 |

主表替换门槛要求 N=1000 和 N=5000 speedup ≥ 1.10x。OPT5 未达到该门槛。

Kernel optimization 最终判断：

```text
completed, but conditional.
OPT1 is useful and correct.
OPT5 does not replace baseline main table.
OPT3/OPT4 remain appendix/future work.
```

## 6. Adaptive-K Decision

详细报告：

```text
docs/adaptive/adaptive_k_report.md
```

N=1000 关键结果：

| method | avg seeds | Strict SR | pos_p95_all mm | speedup vs K16 | pass_quality | pass_perf |
|---|---:|---:|---:|---:|---:|---:|
| K16 baseline | 16.000 | 0.954 | 4.508959 | 1.000000 | 1 | 1 |
| K8-only | 8.000 | 0.924 | 36.093694 | 2.118939 | 0 | 1 |
| AK-8+8 | 8.000 | 0.924 | 36.093694 | 2.125971 | 0 | 1 |
| AK-4+4+8 | 5.192 | 0.954 | 4.828379 | 2.941184 | 1 | 1 |

结论：

- K8-only 速度快，但质量不达标。
- AK-8+8 速度快，但质量仍不达标。
- AK-4+4+8 同时通过质量和性能门槛。

Adaptive-K 最终判断：

```text
completed.
AK-4+4+8 can be presented as an optional main-text adaptive extension.
K8-only and AK-8+8 should remain ablation/appendix negative results.
```

## 7. Full Ablation Decision

详细报告：

```text
docs/ablation/full_ablation_report.md
```

核心输出：

- Static ablation table generated。
- Module contribution table generated。
- Trajectory rerank table generated。
- cuRobo boundary table generated。
- Figure data CSVs generated。

必须保留的 source 边界：

- 当前 CUDA 实测与 historical/frozen V4 数据已经在 CSV 中用 source 字段区分。
- 未在本轮重跑的数据不能写成本轮新实验。

Full ablation 最终判断：

```text
completed.
Full ablation should enter the paper.
Historical/source-limited rows must be labeled honestly.
```

## 8. Updated Paper Claims

可以写入论文主文：

1. V4-Final-K16 baseline 已完成 correctness、static benchmark、cuRobo comparison、Nsight profiling 和 final readiness。
2. Limit Barrier 降低 near-limit 风险，是 V4 constraint-aware behavior 的核心模块。
3. Analytical Limit Gradient 保持 correctness，并带来小幅稳定性能收益。
4. AK-4+4+8 在该目标集上以平均约 5.19 seeds 达到 K16 的 Strict SR，并显著加速。
5. Full ablation 支持 V4 各模块贡献分析。
6. cuRobo comparison 是统一目标集和统一评价协议下的系统级比较。

必须保守写：

1. CUDA-V4-Final-K16 质量强，但 cuRobo-Graph throughput 更强。
2. OPT5 没有达到替换 baseline 主表的 1.10x speedup 门槛。
3. Adaptive-K 是可选扩展，不替代原 V4-Final-K16 baseline。
4. Smoothness rerank 是 candidate-level post-selection，不是 CUDA kernel 主要性能贡献。

不能写：

1. 不能写 CUDA-V4 全面超过 cuRobo。
2. 不能写所有 Adaptive-K 版本都保持 K16 质量。
3. 不能写 warp-per-seed 已完成并有效。
4. 不能把 historical rows 写成本轮新实验。

## 9. Final Enhancement Gate

| area | status | main text | appendix | future work |
|---|---|---|---|---|
| kernel_optimization | completed | conditional | yes | warp-per-seed cooperative kernel |
| adaptive_k | completed | conditional_on_gates | yes | threshold rescue |
| full_ablation | completed | yes | source-limited rows | rerun unavailable historical methods |
| paper_claim_boundary | completed | conservative | yes | do not claim full cuRobo speed superiority |

最终判断：

```text
V4-Final-K16 CUDA enhancement package is complete.
The original baseline remains the main locked result.
Full ablation can enter the paper.
AK-4+4+8 can enter as an optional adaptive extension.
OPT1 can enter as an engineering optimization.
OPT5, OPT3, and OPT4 must be reported conservatively.
The paper still must not claim full superiority over cuRobo.
```
