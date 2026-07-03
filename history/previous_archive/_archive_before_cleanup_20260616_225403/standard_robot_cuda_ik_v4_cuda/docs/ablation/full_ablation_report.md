# V4-Final-K16 Full Ablation Report

## 1. Scope

本报告对应增强计划 Phase C：Full Ablation。

目的不是重新发明 V4，而是补齐论文所需的模块贡献证据：

- Static IK ablation。
- Limit Barrier contribution。
- Smoothness rerank contribution。
- CUDA vs Python acceleration evidence。
- cuRobo comparison boundary。
- Figure data for paper plots。

所有新增 ablation 输出均写入：

- `data/results/ablation/`
- `docs/ablation/`
- `figures/*.csv`

未覆盖 baseline 主结果文件。

## 2. Generated Files

核心 CSV：

- `data/results/ablation/static_ablation_N1000.csv`
- `data/results/ablation/module_contribution_table.csv`
- `data/results/ablation/trajectory_ablation.csv`
- `data/results/ablation/curobo_boundary_table.csv`

图表数据 CSV：

- `figures/fig_v1_to_v4_pipeline.csv`
- `figures/fig_limit_barrier_effect.csv`
- `figures/fig_smoothness_rerank_effect.csv`
- `figures/fig_curobo_boundary.csv`
- `figures/fig_static_ablation_sr_pos.csv`
- `figures/fig_nsight_bottleneck.csv`

## 3. Static Ablation

结果文件：`data/results/ablation/static_ablation_N1000.csv`

表格中混合了两类数据源：

- `current_cuda`: 当前 CUDA-V4-Final-K16 或本轮增强实际结果。
- `historical_v4_limit_sweep`: 已冻结的 V4 limit weight sweep 历史结果。
- `not_rerun_current_plan`: 当前计划没有重新运行的历史/占位模块，报告中不得当作本轮新实验。

该设计符合增强计划“不推翻已有 baseline、不重做主线”的要求，但写论文时必须区分当前实测与历史冻结数据。

## 4. Module Contribution

结果文件：`data/results/ablation/module_contribution_table.csv`

模块贡献的核心解释：

| module | contribution | paper usage |
|---|---|---|
| Analytical Jacobian + LM | V4 correctness 和收敛质量的基础 | 主文方法部分 |
| Sobol-K16 | 提供多种子覆盖和高 Strict SR | 主文方法 + ablation |
| Limit Barrier | 显著降低 near-limit ratio，约束关节极限风险 | 主文 ablation |
| Smoothness rerank | 降低 trajectory candidate 的关节跳变 | 主文或 appendix，取决于叙事篇幅 |
| CUDA port | 相对 Python 获得大幅批处理加速 | 主文性能部分 |
| cuRobo comparison | 给出系统边界，不主张全面超过 cuRobo | 主文 comparison |

## 5. Limit Barrier Effect

Limit Barrier 固定参数：

```text
w_limit = 0.03
margin = 0.087
```

关键论文叙事：

- 不应写成“Limit Barrier 只提升成功率”。
- 更准确的贡献是“在保持可接受 pose quality 的同时降低 near-limit 风险”。
- 在 CUDA port 中，Limit Barrier 是 V4-Final-K16 constraint-aware behavior 的核心组成。

本轮还通过 OPT1 证明了 Limit Barrier gradient 可以从 finite-difference 替换为 analytical/piecewise gradient，并保持 metrics 对齐。

## 6. Smoothness Rerank

结果文件：`data/results/ablation/trajectory_ablation.csv`

该部分来自冻结的 V4 M2 smooth rerank 轨迹结果，用于补充 trajectory-level candidate selection 证据。

解释边界：

- Smoothness rerank 是 candidate-level post-selection。
- 如果 CUDA 只负责生成 candidate，而 CPU 做 trajectory rerank，论文中必须明确这一点。
- trajectory rerank 不作为 CUDA kernel 主要性能贡献。

论文可写：

```text
Smoothness reranking reduces inter-step joint discontinuity by selecting among already generated IK candidates, while preserving the pose success constraints.
```

不能写：

```text
trajectory rerank 是主要 CUDA kernel 加速来源。
```

## 7. cuRobo Boundary

结果文件：`data/results/ablation/curobo_boundary_table.csv`

必须保留的比较边界：

- 统一 targets。
- 统一 robot。
- 统一评价阈值。
- 统一 timing protocol。
- 不强行要求 cuRobo 使用与 Sobol-K16 完全相同的 seed。

不可完全对齐因素：

- cuRobo 内部 seed 策略。
- cuRobo 优化器。
- CUDA Graph。
- cuRobo 内部并行策略。
- cuRobo pipeline 与本文固定 K16 多种子 LM pipeline 不等价。

论文结论必须保守：

```text
This is a system-level comparison under shared targets and evaluation protocol, not an equivalent algorithm-to-algorithm comparison.
```

已有结果支持的边界：

- CUDA-V4-Final-K16 在该目标集上质量指标强。
- cuRobo-Graph 的 throughput 更强，尤其 batch size 增大后优势明显。
- 不允许主张本文 CUDA-V4 全面超过 cuRobo。

## 8. Figure Data

本轮生成的是绘图数据 CSV，而不是强制生成 PNG/PDF。增强计划允许“如果不画图，也必须生成画图数据 CSV”。

建议论文图：

| figure csv | intended figure |
|---|---|
| `fig_v1_to_v4_pipeline.csv` | V1 到 V4-Final-K16 pipeline 演进 |
| `fig_limit_barrier_effect.csv` | Limit Barrier 对 near-limit / SR / pose error 的影响 |
| `fig_smoothness_rerank_effect.csv` | Smoothness rerank 对 mean/p95 delta q 的影响 |
| `fig_curobo_boundary.csv` | CUDA-V4 与 cuRobo-Graph 的 SR / throughput 边界 |
| `fig_static_ablation_sr_pos.csv` | Static ablation 的 SR 与 pos_p95 对比 |
| `fig_nsight_bottleneck.csv` | registers / occupancy / DRAM throughput bottleneck |

## 9. Paper Claim Decision

必须进入主文：

- V4-Final-K16 的模块构成和 static benchmark。
- Limit Barrier 对 near-limit 的贡献。
- CUDA vs Python acceleration。
- cuRobo comparison boundary。

建议进入主文或 appendix：

- Smoothness rerank 轨迹结果。
- Adaptive-K AK-4+4+8 作为可选扩展。
- OPT1 analytical limit gradient 作为工程优化。

必须作为 appendix / future work：

- OPT3 SoA layout。
- OPT4 warp-per-seed prototype。
- 未在当前计划中重跑的 historical rows。

不能写的结论：

- 不能写 V4-CUDA 全面超过 cuRobo。
- 不能写所有 Adaptive-K 版本都保持 K16 质量。
- 不能写 OPT5 已达到主表替换门槛。
- 不能把历史数据伪装成本轮新实验。

最终判断：

```text
Full ablation package complete.
The ablation evidence is sufficient for paper main text, with source labels retained for historical rows.
The strongest paper claims remain: V4 quality, constraint-aware near-limit reduction, Python-to-CUDA acceleration, and a clear cuRobo system-comparison boundary.
```
