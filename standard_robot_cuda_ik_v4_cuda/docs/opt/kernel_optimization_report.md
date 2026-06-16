# V4-Final-K16 CUDA Kernel Optimization Report

## 1. Scope

本报告对应增强计划 Phase A：Kernel Optimization。实验目标是在不破坏已锁定 `CUDA-V4-Final-K16` baseline 的前提下，评估若干 CUDA 优化候选是否可以进入论文主文。

本轮没有覆盖或重写以下 baseline 文件：

- `docs/final_paper_readiness_report.md`
- `data/results/final_summary.csv`
- `data/results/cuda_v4_static_benchmark.csv`
- `data/results/cuda_v4_curobo_compare.csv`
- `data/results/nsight_summary.csv`

所有新增结果均写入：

- `data/results/opt/`
- `docs/opt/`
- `logs/opt/`

## 2. Experimental Protocol

基础配置：

- Robot / target / seed 数据沿用 V4-Final-K16 baseline 的 shared raw assets。
- K 固定为 `16`，不重新搜索 K。
- Limit Barrier 固定为 `w_limit=0.03`、`margin=0.087`。
- 不修改 Loose / Medium / Strict 阈值。
- 不修改 success 判定逻辑。
- 不做碰撞检测、V5 或 Motion Generation。

计时配置：

- OPT0、OPT1、OPT2、OPT5 的关键 benchmark 使用 `warmup=10`、`repeat=30`。
- OPT2 编译参数实验使用 `-Xptxas -v` 记录 register 和 spill 信息。
- OPT3 / OPT4 作为探索性记录，不进入主性能表。

核心输出：

- `data/results/opt/opt0_baseline_rebuild.csv`
- `data/results/opt/opt1_limit_gradient_correctness.csv`
- `data/results/opt/opt1_limit_gradient_benchmark.csv`
- `data/results/opt/opt2_register_reduction.csv`
- `data/results/opt/opt3_candidate_layout.csv`
- `data/results/opt/opt4_warp_per_seed_correctness.csv`
- `data/results/opt/opt4_warp_per_seed_benchmark.csv`
- `data/results/opt/opt5_best_combined_static_benchmark.csv`
- `data/results/opt/opt5_best_combined_vs_baseline.csv`
- `logs/opt/ptxas_registers.log`

## 3. OPT0: Baseline Rebuild

目的：确认增强实验框架没有破坏原 baseline runner。

结果文件：`data/results/opt/opt0_baseline_rebuild.csv`

关键结果：

| N | K | warmup | repeat | Strict SR | pos_p95_all mm | near_limit | gpu_stream_ms | NaN | Inf |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 16 | 10 | 30 | 0.954 | 4.508959 | 0.007 | 646.025281 | 0 | 0 |

通过标准：

- Strict SR ≥ 93%：通过。
- pos_p95_all ≤ 8 mm：通过。
- near_limit ≤ 4%：通过。
- no NaN/Inf：通过。

结论：OPT0 通过，增强实验框架没有破坏 baseline correctness 或质量指标。

## 4. OPT1: Analytical Limit Gradient

实现内容：

- 新增解析 / 分段 Limit Barrier gradient。
- runner 支持 `--limit-gradient finite_diff` 和 `--limit-gradient analytic`。
- `analytic` 模式保持 loss 定义不变，只替换 gradient 计算路径。

解析梯度：

```text
lower distance: d = q_j - q_min_j
if d < margin:
    grad_j += -2 * w_limit * (margin - d)

upper distance: d = q_max_j - q_j
if d < margin:
    grad_j +=  2 * w_limit * (margin - d)
```

Correctness 结果：

| strict_sr_diff_pp | pos_p95_diff_mm | pass |
|---:|---:|---:|
| 0.0 | 9.403565e-07 | 1 |

性能结果：

| N | warmup | repeat | Strict SR | pos_p95_all mm | near_limit | gpu_stream_ms |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 10 | 30 | 0.960 | 4.384513 | 0.010 | 64.540694 |
| 1000 | 10 | 30 | 0.954 | 4.563133 | 0.007 | 597.124017 |
| 5000 | 10 | 30 | 0.954 | 4.563133 | 0.007 | 2983.860970 |

相对 OPT0 / baseline 的解释：

- 质量保持：Strict SR 没有下降，pos_p95_all 保持在 8 mm 门限内。
- N=1000 相比 OPT0 baseline rebuild 约 `646.025 / 597.124 = 1.08x`。
- N=5000 相比 baseline static benchmark 约 `3186.436 / 2983.861 = 1.07x`。
- 该优化可进入工程实现和论文 discussion；但单独作为“显著性能主贡献”仍偏弱。

结论：OPT1 通过 correctness 和质量门槛，带来稳定但中等幅度的性能收益。

## 5. OPT2: Register Reduction

实验内容：

- 保留 baseline 编译。
- 新增 `--maxrregcount=160` target。
- 新增 `--maxrregcount=128` target。
- 使用 `-Xptxas -v` 记录寄存器、spill、local memory。

结果文件：

- `data/results/opt/opt2_register_reduction.csv`
- `logs/opt/ptxas_registers.log`

关键结果：

| variant | maxrregcount | registers/thread | spill stores | spill loads | local memory bytes | N=1000 ms | N=5000 ms | Strict SR | pos_p95_all mm | pass_quality | pass_perf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline |  | 184 | 0 | 0 | 0 | 601.443764 | 2980.479070 | 0.954 | 4.563133 | 1 | 0 |
| r160 | 160 | 160 | 116 | 168 | 880 | 590.225413 | 2943.214300 | 0.954 | 4.563133 | 1 | 1 |
| r128 | 128 | 128 | 244 | 304 | 976 | 599.247424 | 2987.444680 | 0.954 | 4.563133 | 1 | 1 |

解释：

- `r160` 将寄存器从 184 降至 160，但引入 spill。
- `r128` 进一步降至 128，但 spill 明显增加，N=5000 性能反而不稳定。
- `r160` 的 N=1000 / N=5000 速度略好于当前 analytic baseline，但没有达到 1.10x 主文替换门槛。
- `r128` 证明硬压寄存器并不可靠，spill 会抵消 occupancy 潜在收益。

结论：OPT2 提供了 register pressure 证据和可解释的编译参数边界，但不应单独作为主优化贡献。若进入 appendix，应强调“寄存器降低不等价于整体性能显著提升”。

## 6. OPT3: Candidate Result SoA Layout

计划目标：

- 将 candidate result 从 AoS 拆为 SoA buffers。
- 分离 `q_candidates`、`pos_err`、`rot_err`、`pose_cost`、`near_limit`、`success_rank` 等字段。
- 评估 selection 和 D2H 阶段是否为瓶颈。

本轮结果：

| variant | implemented | selection_kernel_speedup | total_gpu_stream_ms_speedup | quality_unchanged | notes |
|---|---:|---:|---:|---:|---|
| aos_baseline | 1 |  | 1.0 | 1 | SoA kernel not promoted; baseline selection is not dominant relative to LM kernel |

解释：

- 当前 Nsight 和 benchmark 证据显示主耗时来自 FP64 scalar LM、register pressure 和低 occupancy。
- candidate selection 不是主瓶颈。
- 因此 SoA layout 没有进入 OPT5。

结论：OPT3 作为瓶颈排除证据保留，不进入论文主结果。若论文需要，可以在 appendix 中说明 selection layout 不是当前性能限制的主要来源。

## 7. OPT4: Warp-per-Seed Prototype

计划目标：

- 新增独立 cooperative warp kernel。
- 以 1 warp = 1 target-seed pair 的方式提高每个 block 内并行度。
- 验证是否改善 occupancy / register pressure。

本轮结果：

| variant | implemented | pass_quality | speedup | notes |
|---|---:|---:|---:|---|
| warp_per_seed_prototype | 0 | 0 |  | Not promoted; requires a separate cooperative warp solve kernel beyond current correctness baseline |

解释：

- 增强计划明确 OPT4 是高风险实验，不要求必然成功。
- 本轮没有将 cooperative warp solve 推到可替换 baseline 的程度。
- 由于没有通过 correctness 和 benchmark，不允许进入 OPT5 或论文主结果。
- 该方向仍是合理 future work：需要重写 FK/Jacobian/H/g/solve 的 lane 分工和 warp broadcast，并重新做 N=10 / N=100 correctness。

结论：OPT4 失败但有价值。最终论文中不能声称 warp-per-seed 已完成或有效，只能作为 future work。

## 8. OPT5: Best Combined Variant

组合规则：

- 只组合已通过 correctness 的优化。
- 本轮 OPT5 实际采用 `analytic limit gradient` 作为主有效优化。
- 没有合入 OPT4。
- register cap 没有成为默认主配置，因为 speedup 没有达到稳定 1.10x 且存在 spill。

结果文件：

- `data/results/opt/opt5_best_combined_static_benchmark.csv`
- `data/results/opt/opt5_best_combined_vs_baseline.csv`

关键结果：

| N | baseline ms | OPT5 ms | speedup | Strict SR | pos_p95_all mm | near_limit |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 73.666339 | 64.475074 | 1.142555 | 0.960 | 4.384513 | 0.010 |
| 500 | 319.969279 | 302.718519 | 1.056986 | 0.954 | 4.337968 | 0.004 |
| 1000 | 639.158346 | 601.146330 | 1.063233 | 0.954 | 4.563133 | 0.007 |
| 5000 | 3186.436120 | 2986.249340 | 1.067036 | 0.954 | 4.563133 | 0.007 |

主文门槛判断：

- correctness pass：通过。
- N=1000 Strict SR ≥ 93%：通过。
- N=1000 pos_p95_all ≤ 8 mm：通过。
- near_limit ≤ 4%：通过。
- N=1000 speedup ≥ 1.10x：未通过，实际约 1.063x。
- N=5000 speedup ≥ 1.10x：未通过，实际约 1.067x。
- Nsight / register 指标有解释，但 OPT5 没有形成足够强的 profiling 改善闭环。

结论：OPT5 不替换 baseline 主表。论文可以写为“解析 limit gradient 提供小幅稳定收益，并确认主要瓶颈仍在 FP64 LM / register pressure / one-thread mapping”，但不能把 OPT5 作为新的主配置。

## 9. Paper Claim Decision

可以进入主文或 discussion：

- OPT1 解析 Limit Barrier gradient 的 correctness 和稳定小幅加速。
- OPT2 register cap 实验作为性能瓶颈分析证据。
- OPT5 作为“不足以替换 baseline 的优化探索”。

只能放 appendix / future work：

- OPT3 SoA candidate layout：selection 不是主瓶颈，未作为主优化。
- OPT4 warp-per-seed：未完成可替换 correctness，不得声称有效。

不能写的结论：

- 不能写“OPT5 显著提升 CUDA-V4 主性能”。
- 不能写“warp-per-seed 已完成并有效”。
- 不能写“register cap 解决了低 occupancy 问题”。

最终判断：

```text
Kernel optimization package complete.
Main V4-Final-K16 baseline remains unchanged.
Analytical limit gradient can be merged into engineering implementation.
OPT5 does not satisfy 1.10x main-table replacement gate.
```
