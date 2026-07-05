# 论文补充实验工作区

本目录用于存放 `论文/paper.tex` 修订前后的补充实验、图表、验收报告和只在补充实验中使用的 runner 变体。除最终拷贝到 `论文/绘图/` 的 PDF 图外，新增脚本、CSV、日志和中间产物均保留在本目录下。

## 目录

```text
scripts/         新实验脚本、绘图脚本、验收脚本
inputs/          near singular / near limit / trajectory / seed scan 输入
results/         新增或归一化 CSV
figures/         由 CSV 生成的新增图
reports/         验收报告、论文合并备份、后续审稿报告
patches/         预留补丁目录
logs/            长任务日志
runner_variant/  仅用于补充实验的 runner 变体与构建产物
```

## 已完成的 P0/P1 补充实验

- `fair_curobo_k16_summary.csv`：统一 OPT4C K=16/K=1 与 cuRobo K=16/K=1 的公平对比口径。
- `kernel_time_breakdown.csv`：统计 H2D、kernel、D2H 与 launch/sync 的时间组成。
- `threshold_scan.csv`：扫描 Loose / Medium / Strict / Ultra 阈值。
- `seed_count_scan.csv`：扫描 $K=1,2,4,8,16$。
- `near_singular_summary.csv`：构造 wrist / elbow / shoulder 近奇异目标。
- `near_limit_barrier_summary.csv`：比较 near-limit 目标下 Barrier ON/OFF。
- `barrier_weight_scan.csv`：扫描 $w_{\mathrm{limit}}$ 权重，结论为默认值是保守折中而非严格最优。
- `trajectory_continuity_summary.csv`：比较原始 best 选择与离线 smoothness rerank。
- `lm_iter_scan.csv`：扫描最大迭代次数 20/40/60/80/100。
- `cpu_baseline_summary.csv`：Python/NumPy CPU-LM-K1/K16 单进程量级对照。
- `nsys_opt4c_n1000_summary.txt`：Nsight Systems 对 `N=1000,K=16` 的 CUDA timeline 摘要。

总体验收见 `reports/acceptance_report.md`。当前补充实验、CPU baseline、Nsight Systems timeline 和论文审稿报告均已生成，论文正文已引用关键结果和图。

## 复现实验命令

在项目根目录 `standard_robot_cuda_ik` 下：

```bash
python3 data/experiments/补充实验/scripts/run_fair_curobo_k16.py
python3 data/experiments/补充实验/scripts/run_kernel_time_breakdown.py
python3 data/experiments/补充实验/scripts/run_threshold_scan.py
python3 data/experiments/补充实验/scripts/generate_special_targets.py
python3 data/experiments/补充实验/scripts/run_seed_count_scan.py
python3 data/experiments/补充实验/scripts/run_near_singular.py
python3 data/experiments/补充实验/scripts/prepare_limit_weight_runner.py
python3 data/experiments/补充实验/scripts/run_near_limit_barrier.py
python3 data/experiments/补充实验/scripts/run_barrier_weight_scan.py
python3 data/experiments/补充实验/scripts/run_trajectory_continuity.py
python3 data/experiments/补充实验/scripts/run_lm_iter_scan.py
python3 data/experiments/补充实验/scripts/run_cpu_baseline.py
bash data/experiments/补充实验/scripts/run_nsight_systems.sh
python3 data/experiments/补充实验/scripts/plot_all.py
python3 data/experiments/补充实验/scripts/validate_outputs.py
```

`prepare_limit_weight_runner.py` 会在 `runner_variant/` 下复制并构建只用于补充实验的 runner，支持 `--limit-weight`。该变体不修改 `src/cuda` 主线源码。

## 已生成图件

`figures/` 中的 PDF 已同步到 `论文/绘图/`，供 `论文/paper.tex` 直接引用：

- `fig_pareto_throughput_success.pdf`
- `fig_kernel_time_breakdown.pdf`
- `fig_threshold_scan.pdf`
- `fig_seed_count_scan.pdf`
- `fig_near_singular_sr.pdf`
- `fig_near_limit_barrier.pdf`
- `fig_barrier_weight_scan.pdf`
- `fig_trajectory_delta_q.pdf`
- `fig_lm_iter_scan.pdf`
- `fig_nsys_timeline_opt4c.pdf`
- `fig_thread_mapping_redraw.pdf`
- `fig_algorithm_pipeline_redraw.pdf`

## 尚未纳入当前论文主线的扩展项

- 轨迹连续性实验使用候选级离线 rerank，只能说明分支跳变可被缓解，不能等同于在线 warm-start 或连续约束优化。
- CPU baseline 是 Python/NumPy 单进程实现，只用于量级对照；不代表优化后的 C++ CPU 求解器或 KDL/TRAC-IK 性能。
- Nsight Systems 采样时本机不允许 CPU context switch tracing；CUDA kernel、memcpy 和 memset timeline 已成功采集。
