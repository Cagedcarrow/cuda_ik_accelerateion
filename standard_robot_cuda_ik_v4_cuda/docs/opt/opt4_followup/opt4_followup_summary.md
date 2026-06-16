# OPT4 后续 CUDA 线程映射优化总结

## 1. 执行范围

本轮按照 `docs/OPT4 后续 CUDA 线程映射优化执行计划.md` 执行，目标是把原先一句话记录为 future work 的 OPT4 扩展成完整的线程映射边界实验。

本轮没有修改 V4-Final-K16 数学定义：

- 没有修改 Loose / Medium / Strict 阈值。
- 没有修改 success 判定逻辑。
- 没有重新搜索 `w_limit`。
- 没有重新搜索 `K`。
- 没有覆盖已有 `final_paper_readiness_report.md`。
- 没有覆盖已有 baseline benchmark CSV。
- 没有写“全面超过 cuRobo”。

新增结果全部写入：

- `data/results/opt/opt4_followup/`
- `docs/opt/opt4_followup/`
- `logs/opt/opt4_followup/`

## 2. 新增实现

代码入口：

- `src/cuda/cuda_v4_runner.cu`
- `scripts/run_opt4_followup.py`
- `CMakeLists.txt`

新增 CUDA variant：

| variant | 映射方式 | 说明 |
|---|---|---|
| `opt4_warp_per_seed` | 1 warp = 1 seed candidate | 保持为失败分支，显式返回非零，避免把 baseline 误标为 warp-per-seed |
| `opt4c_block_target` | 1 block = 1 target，thread 0-15 = K16 seeds | 每个线程完整求解一个 seed，block 内 shared memory 选择 best |
| `opt4b_warp_target` | 1 warp = 1 target，lane 0-15 = K16 seeds | 每个 lane 完整求解一个 seed，warp 对应一个 target |
| `opt4d_fused_selection_from_opt4c` | fused candidate generation + selection | 由 OPT4C 实现 D1 路线，去掉 candidate global write/read 和独立 selection kernel |

关键实现原则：

- 每个 seed 的 LM 求解仍由单个线程完整执行，不拆分 6x6 小矩阵求解。
- `q_trial` 仍总是接受，`lambda` 只按 `loss_new < loss_old` 调整。
- `max_iter=60`、`dq clamp=0.35`、`w_limit=0.03`、`margin=0.087` 保持不变。
- selection 仍为 `success_rank -> near_limit -> pose_cost`。

## 3. Baseline Snapshot

baseline snapshot 文件：

```text
data/results/opt/opt4_followup/opt4_baseline_snapshot.csv
```

本轮为了公平比较，使用同一 runner、同一 raw targets/seeds、同一 `warmup=10/repeat=30`，并使用 `--limit-gradient analytic` 作为固定比较口径。

| N | K | gpu_stream_ms | Strict SR | pos_p95_all mm | near_limit | NaN | Inf |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 16 | 64.212135 | 0.960 | 4.384513 | 0.010 | 0 | 0 |
| 1000 | 16 | 595.614864 | 0.954 | 4.563133 | 0.007 | 0 | 0 |
| 5000 | 16 | 2967.950390 | 0.954 | 4.563133 | 0.007 | 0 | 0 |

## 4. 原始 Warp-per-Seed Postmortem

输出：

- `data/results/opt/opt4_followup/opt4_warp_per_seed_postmortem.csv`
- `docs/opt/opt4_followup/opt4_warp_per_seed_postmortem.md`
- `logs/opt/opt4_followup/opt4_warp_per_seed_ncu.csv`
- `logs/opt/opt4_followup/opt4_warp_per_seed_ncu_raw.log`

结论：

原始 `1 warp = 1 target-seed candidate` 没有进入主结果。原因不是“没有尝试”，而是该映射方向与问题粒度不匹配：

- 单个 seed 内部是强串行 LM 迭代。
- 6x6 小矩阵求解很难高效拆给 32 lanes。
- lambda update、convergence 判断和 joint clamp 都带有控制流。
- 大量 lane 会空闲或等待同步。
- shuffle/sync 协作成本可能抵消潜在收益。

本轮 runner 对 `--variant opt4_warp_per_seed` 显式返回非零，避免把 baseline 误当成 warp-per-seed 成功数据。论文中该方向只能作为 Discussion / Future Work。

## 5. OPT4C：Block-per-Target / Thread-per-Seed

输出：

- `data/results/opt/opt4_followup/opt4c_block_target_correctness.csv`
- `data/results/opt/opt4_followup/opt4c_block_target_benchmark.csv`
- `data/results/opt/opt4_followup/opt4c_block_target_nsight.csv`
- `docs/opt/opt4_followup/opt4c_block_target_report.md`

### 5.1 Correctness

N=100 correctness 与 baseline 完全对齐：

| metric | value |
|---|---:|
| correctness_pass | 1 |
| Strict SR | 0.960 |
| pos_p95_all mm | 4.384513 |
| near_limit | 0.010 |
| Strict SR diff pp | 0.000 |
| pos_p95 diff mm | 0.000 |
| near_limit diff pp | 0.000 |
| best_seed_diff_count | 0 |
| max_q_abs_diff | 0.000 |
| NaN / Inf | 0 / 0 |

### 5.2 Benchmark

| N | gpu_stream_ms | speedup vs baseline | Strict SR | pos_p95_all mm | near_limit |
|---:|---:|---:|---:|---:|---:|
| 100 | 6.580417 | 9.752920x | 0.960 | 4.384513 | 0.010 |
| 1000 | 57.358053 | 10.384154x | 0.954 | 4.563133 | 0.007 |
| 5000 | 274.726237 | 10.803302x | 0.954 | 4.563133 | 0.007 |

### 5.3 Nsight / ptxas

| metric | value |
|---|---:|
| registers/thread | 194 |
| spill | 0B stores / 0B loads |
| achieved occupancy | 16.04% |
| Compute SM throughput | 84.24% |
| DRAM throughput | 0.58% |
| branch efficiency | 98.24% |
| Avg. active threads per warp | 10.57 |
| Avg. divergent branches | 2375.97 |

解释：

- OPT4C 没有降低寄存器，反而略高于 baseline。
- 真正收益来自线程映射粒度变化：baseline 是大量 one-thread blocks，OPT4C 是一个 target 一个 block、K16 seeds 在 block 内并行。
- fused in-block selection 去掉了 candidate global write/read 和独立 selection kernel。
- 仍然是 FP64 compute-heavy，DRAM throughput 很低，说明不是内存带宽瓶颈。

判定：

```text
OPT4C 满足 main_result 条件。
```

## 6. OPT4B：Warp-per-Target / Lane-per-Seed

输出：

- `data/results/opt/opt4_followup/opt4b_warp_target_correctness.csv`
- `data/results/opt/opt4_followup/opt4b_warp_target_benchmark.csv`
- `data/results/opt/opt4_followup/opt4b_warp_target_nsight.csv`
- `docs/opt/opt4_followup/opt4b_warp_target_report.md`

### 6.1 Correctness

N=100 correctness 与 baseline 完全对齐：

| metric | value |
|---|---:|
| correctness_pass | 1 |
| Strict SR | 0.960 |
| pos_p95_all mm | 4.384513 |
| near_limit | 0.010 |
| Strict SR diff pp | 0.000 |
| pos_p95 diff mm | 0.000 |
| near_limit diff pp | 0.000 |
| best_seed_diff_count | 0 |
| max_q_abs_diff | 0.000 |
| NaN / Inf | 0 / 0 |

### 6.2 Benchmark

| N | gpu_stream_ms | speedup vs baseline | Strict SR | pos_p95_all mm | near_limit |
|---:|---:|---:|---:|---:|---:|
| 100 | 10.895910 | 5.893233x | 0.960 | 4.384513 | 0.010 |
| 1000 | 57.467597 | 10.364360x | 0.954 | 4.563133 | 0.007 |
| 5000 | 276.859213 | 10.720071x | 0.954 | 4.563133 | 0.007 |

### 6.3 Nsight / ptxas

| metric | value |
|---|---:|
| registers/thread | 193 |
| spill | 0B stores / 0B loads |
| achieved occupancy | 16.24% |
| Compute SM throughput | 84.19% |
| DRAM throughput | 0.55% |
| branch efficiency | 98.24% |
| Avg. active threads per warp | 10.57 |
| Avg. divergent branches | 2375.97 |

解释：

- OPT4B 也保持质量完全一致。
- N=1000 和 N=5000 速度与 OPT4C 接近，但 N=100 下比 OPT4C 慢。
- 它把一个 target 映射到一个 warp，理论上更贴合 K16，但 lane 内完整 LM 仍有 divergence 和高私有状态。
- 当前实现使用 shared memory staging 做复杂 key selection，稳定性优先。

判定：

```text
OPT4B 也满足 main_result 条件，但推荐 OPT4C 作为主实现，因为它更简单，N=100 更快，整体性能略优。
```

## 7. OPT4D：Fused Candidate Generation + Selection

输出：

- `data/results/opt/opt4_followup/opt4d_fused_selection.csv`
- `docs/opt/opt4_followup/opt4d_fused_selection_report.md`

OPT4D 采用计划中的 D1 路线：由 OPT4C 自带 fused selection 实现。

| N | baseline two-kernel ms | fused generation+selection ms | total speedup |
|---:|---:|---:|---:|
| 100 | 64.212135 | 6.580417 | 9.752920x |
| 1000 | 595.614864 | 57.358053 | 10.384154x |
| 5000 | 2967.950390 | 274.726237 | 10.803302x |

结论：

fused selection 有意义，但收益不是单独来自 selection kernel，而是来自更合理的 target-level 映射、减少 candidate global write/read、减少独立 selection launch，以及更充分利用 K16 多 seed 的天然结构。

## 8. Summary CSV

总表：

```text
data/results/opt/opt4_followup/opt4_followup_summary.csv
```

| variant | correctness | N1000 speedup | N5000 speedup | decision |
|---|---:|---:|---:|---|
| opt4_warp_per_seed | 0 |  |  | future_work |
| opt4c_block_target | 1 | 10.384154x | 10.803302x | main_result |
| opt4b_warp_target | 1 | 10.364360x | 10.720071x | main_result |

## 9. 计划要求问题逐项回答

1. 原始 warp-per-seed 为什么失败？

因为它试图把一个强串行、小规模、控制流复杂的 seed-level LM 求解拆到一个 warp 内，lane 利用率和同步成本都不理想。该分支没有 correctness/benchmark 可替换实现，只能作为 future work。

2. block-per-target/thread-per-seed 是否更好？

是。OPT4C 保留每个 seed 的单线程完整 LM，把 K16 seeds 并行放入同一个 target block，并在 block 内完成 best selection。N=1000 speedup 为 10.384154x，N=5000 speedup 为 10.803302x。

3. warp-per-target/lane-per-seed 是否更好？

是，但略弱于 OPT4C。OPT4B 在 N=1000 / N=5000 上分别达到 10.364360x / 10.720071x，但 N=100 下慢于 OPT4C，且实现复杂度更高。

4. fused selection 是否有意义？

有意义。OPT4D D1 由 OPT4C 实现，去掉了 candidate global write/read 和单独 selection kernel。它和 target-level mapping 共同构成主要性能收益。

5. 当前 CUDA-V4 的线程映射瓶颈是什么？

旧 baseline 的主要问题是粒度太碎：每个 target-seed 是一个 one-thread block，GPU 需要调度大量小 block，再额外启动 selection kernel。OPT4C/OPT4B 把并行粒度提升到 target 级，保留 seed 内串行 LM，同时并行 K16 seeds，明显更适合该问题。

6. 继续优化应该走 kernel mapping，还是 Adaptive-K？

两条都有效，但作用不同。Adaptive-K 通过减少平均 seed 数获得算法-系统协同加速；OPT4C 在固定 K16 下显著改善 kernel mapping。论文中可以把 OPT4C 写作固定 K16 CUDA mapping 主优化，把 Adaptive-K 写作进一步减少计算量的可选策略。

7. 哪些结果进入主文？

推荐进入主文：

- OPT4C block-per-target/thread-per-seed。
- OPT4D fused selection 作为 OPT4C 的组成机制。

可进入主文或 appendix：

- OPT4B warp-per-target/lane-per-seed，作为对比映射。

8. 哪些结果进入 appendix？

- OPT4B 的详细 Nsight 和 N=100 小 batch 差异。
- ptxas register/spill 对比。

9. 哪些结果进入 future work？

- 原始 one-warp-per-seed cooperative solve。
- 更细粒度的 warp shuffle reduction。
- shared memory stall / active threads per warp 的进一步优化。

## 10. 最终决策

```text
OPT4 follow-up 完成。
OPT4C 是本轮最强 fixed-K16 CUDA 线程映射结果。
OPT4B 是有效但略弱的 target-level mapping 对照。
OPT4D 证明 fused candidate generation + selection 有意义。
原始 warp-per-seed 保持 future work。
```

论文更新建议：

```text
在保持 V4-Final-K16 数学逻辑和解质量不变的前提下，target-level K16 seed parallel mapping 将固定 K16 CUDA 求解从 one-thread-block-per-seed 的细粒度调度，改为 block/warp-per-target 的自然多 seed 并行结构，从而显著提升 throughput。
```
