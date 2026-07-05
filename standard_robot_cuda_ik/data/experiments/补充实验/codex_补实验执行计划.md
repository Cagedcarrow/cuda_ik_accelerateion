# CUDA IK 论文补实验与正文修正执行计划

生成时间：2026-07-05

本计划依据：

- `standard_robot_cuda_ik/PROJECT.md`
- `standard_robot_cuda_ik/论文/计划/cuda_ik_paper_revision_plan_for_codex.md`
- `standard_robot_cuda_ik/论文/计划/cuda_ik_paper_acceptance_criteria_plan_mode.md`
- `standard_robot_cuda_ik/论文/格式模板/系统工程与电子技术投稿格式规范.md`

执行约束：

- 先在本目录 `standard_robot_cuda_ik/data/experiments/补充实验/` 内完成脚本、结果、图和验收报告。
- 未通过补充实验验收前，不把实验性代码合并到 `src/cuda/` 或论文正文。
- 论文最终修改目标是 `standard_robot_cuda_ik/论文/paper.tex`，但需等待补充实验 CSV 与图表通过自检后再合并。
- 可用 skill 中没有独立 `nature` skill；后续使用 `paper-review` skill 的审稿框架，并按其包含的 nature-reader / nature-reviewer / nature-polishing 方法论进行审稿式检查和语言收敛。

## 1. 当前仓库结构识别

活跃项目：

```text
standard_robot_cuda_ik/
├── PROJECT.md
├── src/cuda/cuda_v4_runner.cu
├── include/standard_robot_cuda_ik/generated/ur10_model_constants.h
├── data/experiments/
│   ├── inputs/
│   ├── results/
│   └── 补充实验/
└── 论文/
    ├── paper.tex
    ├── 绘图/
    ├── 计划/
    └── 格式模板/
```

现有可复用数据：

- 主基准：`data/experiments/results/dense_static_summary.csv`
- cuRobo K=1：`data/experiments/results/dense_curobo_summary.csv`
- K=1 消融：`data/experiments/补充实验/results/k1_static_summary.csv`
- cuRobo K=16：`data/experiments/补充实验/results/curobo_k16_summary.csv`
- K=16 公平对比：`data/experiments/补充实验/results/fair_comparison_k16_vs_k16.csv`
- 混合精度：`data/experiments/补充实验/results/mixed_precision_summary.csv`

当前缺口：

- near singular 目标生成与实验。
- near limit 与 barrier ON/OFF 实验。
- trajectory continuity 与 smoothness rerank 对比。
- CPU baseline。
- kernel time breakdown 汇总图。
- seed count scan。
- barrier weight scan。
- LM max_iter scan。
- threshold scan。
- Nsight Systems timeline 摘要与示意图。
- 新图统一由脚本从 CSV 生成。
- `paper.tex` 中危险表述仍需修正：`branchless` / `消除 warp divergence`、`32 与 16 互质`、`Bank 冲突为零`、`N×K×W+N`、`5 mm 工业抓取精度要求`、`硬件固化` 等。

## 2. 需要修改或新增的代码模块

### 2.1 只在补充实验目录新增的脚本

所有新增实验脚本先放在：

```text
standard_robot_cuda_ik/data/experiments/补充实验/scripts/
```

计划新增：

| 文件 | 作用 |
|---|---|
| `common_metrics.py` | 统一 CSV 字段、误差阈值、成功率、p95、near-limit、轨迹跳变统计 |
| `generate_special_targets.py` | 生成 near singular、near limit、trajectory raw 输入 |
| `run_fair_curobo_k16.py` | 归一化已有 OPT4C / cuRobo K=1 / cuRobo K=16 数据，必要时重跑 |
| `run_near_singular.py` | 运行近奇异实验 |
| `run_near_limit_barrier.py` | 运行近限位与 Barrier ON/OFF 实验 |
| `run_trajectory_continuity.py` | 运行连续轨迹实验与 smoothness rerank 对比 |
| `run_cpu_baseline.py` | CPU-LM-K1 / CPU-LM-K16 量级 baseline |
| `run_seed_count_scan.py` | K=1,2,4,8,16 扫描 |
| `run_lm_iter_scan.py` | max_iter=20,40,60,80,100 扫描 |
| `run_threshold_scan.py` | Loose/Medium/Strict/Ultra 阈值扫描 |
| `run_kernel_time_breakdown.py` | 汇总 timing CSV 为 H2D/kernel/D2H/launch 占比 |
| `run_nsight_systems.sh` | nsys 采样命令与结果导出 |
| `plot_all.py` | 从 CSV 自动生成所有新增图 |
| `validate_outputs.py` | 检查 CSV 字段、图文件、论文引用、可追溯性 |

### 2.2 可能需要临时实验补丁的主 runner 能力

`src/cuda/cuda_v4_runner.cu` 当前支持：

- `--K`
- `--max-iter`
- `--precision-mode`
- `--fallback-mode`
- `--limit-gradient`
- `--summary-csv`
- `--best-csv`
- `--timing-csv`

当前不足：

- 无 `--wlimit` 或 `--barrier-enabled`，不能直接做 barrier ON/OFF 与权重扫描。
- 无输出全部候选用于轨迹 smoothness rerank 的稳定接口。
- 无显式 trajectory rerank 模式。

执行策略：

1. 第一阶段不改主源码，只在补充实验目录生成脚本和计划。
2. 若确需 runner 扩展，在 `补充实验/patches/` 中生成补丁或实验性复制文件。
3. 补充实验验收通过后，再把最小必要改动合并到 `src/cuda/cuda_v4_runner.cu`。

## 3. 新增实验设计

所有新增 CSV 统一放在：

```text
standard_robot_cuda_ik/data/experiments/补充实验/results/
```

推荐通用字段：

```text
experiment,method,robot,N,K,seed_type,max_iter,wlimit,barrier_enabled,
target_type,repeat,gpu_time_ms_mean,gpu_time_ms_std,
throughput_targets_per_s_mean,throughput_targets_per_s_std,
strict_sr,medium_sr,loose_sr,
pos_p95_all_mm,pos_p95_success_mm,
rot_p95_all_deg,rot_p95_success_deg,
mean_iter,p95_iter,near_limit_ratio,joint_violation_count,
nan_count,inf_count,fail_count,notes
```

### 3.1 cuRobo K=16 公平对比

输入：

- `data/experiments/results/dense_static_summary.csv`
- `data/experiments/results/dense_curobo_summary.csv`
- `data/experiments/补充实验/results/curobo_k16_summary.csv`
- `data/experiments/补充实验/results/k1_static_summary.csv`

输出：

- `results/fair_curobo_k16_summary.csv`
- `figures/fig_pareto_throughput_success.pdf`

方法：

- OPT4C-K16
- OPT4C-K1
- cuRobo-Graph-K16
- cuRobo-Graph-K1

规模：

- 优先 N=100,200,...,1000。
- 若 cuRobo K=16 现有数据只有 N=100,500,1000，则先标注为部分公平对比；需要时重跑补齐。

论文映射：

- 实验协议与公平性设置。
- 与 cuRobo 的系统级与同等 K 对比。
- 讨论章节 Pareto 边界。

### 3.2 Near Singular 近奇异实验

新增输入：

- `inputs/near_singular/targets_{type}_N{N}_T4x4_f64.raw`
- `inputs/near_singular/seeds_{type}_N{N}_K{K}_q_f64.raw`

目标类型：

- `wrist_singular`：`q5 in [-0.03, 0.03]`
- `elbow_singular`：肘部接近伸直或折叠区域
- `shoulder_singular`：肩部组合接近退化区域

规模：

- N=100,500,1000

方法：

- OPT4C-K16
- OPT4C-K1
- cuRobo-K16 若环境稳定则加入，否则在 README 和论文局限性说明。

输出：

- `results/near_singular_summary.csv`
- `figures/fig_near_singular_sr.pdf`
- 可选 `figures/fig_near_singular_error.pdf`

论文映射：

- 新增近奇异鲁棒性小节。
- 必须说明目标生成方式、成功率是否下降、适用边界。

### 3.3 Near Limit 与 Barrier ON/OFF 实验

新增输入：

- `inputs/near_limit/targets_N{N}_T4x4_f64.raw`
- `inputs/near_limit/seeds_N{N}_K{K}_q_f64.raw`

规模：

- N=100,500,1000

方法：

- OPT4C-K16-BarrierON
- OPT4C-K16-BarrierOFF
- OPT4C-K1-BarrierON
- OPT4C-K1-BarrierOFF

输出：

- `results/near_limit_barrier_summary.csv`
- `figures/fig_near_limit_barrier.pdf`

依赖：

- 需要 runner 支持 `wlimit=0` 或 `barrier_enabled=false`。
- 若主 runner 暂不合并，则在 `patches/` 中先保存最小补丁并在补充实验验收后合并。

论文映射：

- 新增 Barrier ON/OFF 表。
- 说明 Barrier 主要降低 near-limit 解比例，不写成主要成功率来源。

### 3.4 Trajectory 连续性实验

新增输入：

- `inputs/trajectory/line_50_*.raw`
- `inputs/trajectory/arc_50_*.raw`
- `inputs/trajectory/random_local_50_*.raw`

规模：

- 每类至少 20 条轨迹，每条 50 点。

方法：

- OPT4C-K16-no-rerank
- OPT4C-K16-smoothness-rerank
- 可选 cuRobo-K16

输出：

- `results/trajectory_continuity_summary.csv`
- `figures/fig_trajectory_delta_q.pdf`
- `figures/fig_trajectory_success.pdf`

指标：

- `point_success_rate`
- `trajectory_success_rate`
- `mean_delta_q`
- `p95_delta_q`
- `max_delta_q`
- `joint_jump_count`

依赖：

- 若 runner 不能直接输出候选集，则先用 best CSV 做 no-rerank；smoothness-rerank 需要 `--candidates-csv` 或补丁。

论文映射：

- 新增轨迹连续性小节。
- 必须报告跳变指标，不能只报告成功率。

### 3.5 CPU Baseline 实验

方法：

- 最低实现 CPU-LM-K1、CPU-LM-K16。
- 若本机 KDL/TRAC-IK 环境可用，再补 KDL/TRAC-IK；否则在 README 和论文局限性说明环境原因。

规模：

- 至少 N=100。
- 优先 N=100,500,1000；若 CPU 太慢，保留 N=100,200。

输出：

- `results/cpu_baseline_summary.csv`

论文映射：

- 新增 CPU 与 GPU IK 基准对比表。
- 明确 CPU baseline 只作量级对照，不声称 GPU 对单目标总是更优。

### 3.6 Kernel Time Breakdown

输入：

- `data/experiments/results/cuda_opt4c_timing_N100.csv`
- `data/experiments/results/cuda_opt4c_timing_N500.csv`
- `data/experiments/results/cuda_opt4c_timing_N1000.csv`

输出：

- `results/kernel_time_breakdown.csv`
- `figures/fig_kernel_time_breakdown.pdf`

字段：

```text
N,h2d_ms,kernel_ms,d2h_ms,launch_ms,total_ms,
h2d_percent,kernel_percent,d2h_percent,launch_percent
```

论文映射：

- 系统级时间分解。
- 明确 fusion 后主要瓶颈转向 kernel 内部算术，而不是 launch。

### 3.7 Barrier 权重扫描

参数：

- `wlimit = 0, 0.005, 0.01, 0.03, 0.05, 0.1`
- N=1000
- random reachable 和 near-limit 两类目标。

输出：

- `results/barrier_weight_scan.csv`
- `figures/fig_barrier_weight_scan.pdf`

依赖：

- 需要 runner 支持 `--wlimit`。

论文映射：

- 说明默认 `wlimit=0.03` 的数据依据。

### 3.8 Seed Count Scan

参数：

- K=1,2,4,8,16
- N=100,500,1000

输入：

- 从 K=16 seed raw 切片生成 K=1/2/4/8。

输出：

- `results/seed_count_scan.csv`
- `figures/fig_seed_count_scan.pdf`

论文映射：

- 说明 K 增大带来的 SR/throughput trade-off。
- 解释为什么默认 K=16。

### 3.9 LM Max Iter Scan

参数：

- `max_iter = 20,40,60,80,100`
- N=1000
- K=16

输出：

- `results/lm_iter_scan.csv`
- `figures/fig_lm_iter_scan.pdf`

论文映射：

- 说明为什么选择 `max_iter=60`。

### 3.10 Threshold Scan

阈值：

- Loose: 30 mm / 10 deg
- Medium: 10 mm / 5 deg
- Strict: 5 mm / 1 deg
- Ultra: 2 mm / 0.5 deg

方法：

- OPT4C-K16
- OPT4C-K1
- cuRobo-K16
- cuRobo-K1

输出：

- `results/threshold_scan.csv`
- `figures/fig_threshold_scan.pdf`

论文映射：

- 说明 OPT4C 主要适用于工程可用阈值，不泛化为所有工业任务。

### 3.11 Nsight Systems Timeline

命令文件：

- `scripts/run_nsight_systems.sh`

输出：

- `reports/nsys_opt4c_n1000_summary.txt`
- `figures/fig_nsys_timeline_opt4c.pdf`

要求：

- 图中标注 H2D、single IK kernel、D2H、N=1000、K=16。
- 若当前环境无法运行 nsys，则生成命令、解析脚本和失败记录，不伪造真实 profile。

论文映射：

- 系统级执行路径验证。

## 4. 新增和重绘图清单

所有图先放在：

```text
standard_robot_cuda_ik/data/experiments/补充实验/figures/
```

通过验收后再复制或引用到：

```text
standard_robot_cuda_ik/论文/绘图/
```

| 图文件 | 数据源 |
|---|---|
| `fig_pareto_throughput_success.pdf` | `results/fair_curobo_k16_summary.csv` |
| `fig_thread_mapping_redraw.pdf` | 无实验数据；由 `plot_all.py` 绘制结构示意 |
| `fig_algorithm_pipeline_redraw.pdf` | 无实验数据；由 `plot_all.py` 绘制结构示意 |
| `fig_kernel_time_breakdown.pdf` | `results/kernel_time_breakdown.csv` |
| `fig_seed_count_scan.pdf` | `results/seed_count_scan.csv` |
| `fig_barrier_weight_scan.pdf` | `results/barrier_weight_scan.csv` |
| `fig_trajectory_delta_q.pdf` | `results/trajectory_continuity_summary.csv` |
| `fig_trajectory_success.pdf` | `results/trajectory_continuity_summary.csv` |
| `fig_near_singular_sr.pdf` | `results/near_singular_summary.csv` |
| `fig_near_limit_barrier.pdf` | `results/near_limit_barrier_summary.csv` |
| `fig_nsys_timeline_opt4c.pdf` | `reports/nsys_opt4c_n1000_summary.txt` 或 nsys 导出数据 |

图表格式约束：

- 图内文字中文。
- 坐标轴含单位。
- 黑白打印可区分。
- 图随文后，论文正文先引用后插图。
- 不使用截图替代可生成图，Nsight timeline 除外。

## 5. `paper.tex` 修改范围

待补充实验验收后修改：

| 论文位置 | 修改内容 |
|---|---|
| 摘要 | 改为固定 6-DOF、中小批量、结构感知单核融合；不写全面优于 cuRobo；不写 Bank 冲突为零 |
| 引言 | 区分通用多阶段 pipeline 调度开销与本文 fusion 后算术瓶颈 |
| 方法 | `硬件固化` 降级为编译期特化；`无分支 LM` 改为低控制流复杂度 LM |
| 方法 | `经线` 统一为 lane；删除 `消除 warp divergence` |
| 方法 | 删除 `32 与 16 互质` 推导；改写共享内存段落 |
| 方法 | 删除正常 baseline 的 `N×K×W+N` 启动次数论证 |
| 实验 | 增加实验协议与公平性设置 |
| 实验 | 加入 cuRobo K=16、公平 Pareto 图 |
| 实验 | 加入 near singular、near limit、trajectory、CPU baseline |
| 实验 | 加入 seed/barrier/max_iter/threshold 扫描 |
| 实验 | 加入 kernel time breakdown 和 Nsight Systems |
| 讨论 | 适用边界：固定 UR10/6DOF、无碰撞、FK 可达目标为主、FP64 小矩阵瓶颈 |
| 结论 | 不写全新范式或全面超过 cuRobo；收束为特定场景下的专用 CUDA 路线 |

投稿模板约束：

- 中文题名不超过 20 个汉字。
- 摘要 200--250 字，第三人称，目的/方法/结果/结论完整。
- 引言编号为 0。
- 图表全文连续编号，三线表。
- 图内文字中文，表内文字中文。
- 数值与单位留空格，例如 `10 mm`。
- 参考文献 15 条以上，英文不少于 2/3。
- 文后作者简介保留。

## 6. 执行顺序

1. 建立补充实验工程目录：
   - `scripts/`
   - `results/`
   - `figures/`
   - `reports/`
   - `patches/`
   - `logs/`
2. 编写 `common_metrics.py` 和 `validate_outputs.py`。
3. 先完成不需要改 runner 的实验：
   - fair cuRobo K=16 汇总。
   - kernel time breakdown。
   - threshold scan。
   - seed count scan 中 K=1/2/4/8/16 的 seed 切片与运行。
   - LM max_iter scan。
4. 生成 near singular 与 near limit 输入。
5. 若 barrier 和 smoothness 需要 runner 支持，生成 `patches/cuda_v4_runner_experiment.patch`，先不合并。
6. 运行 near singular、near limit、trajectory、CPU baseline。
7. 运行或记录 Nsight Systems。
8. 执行 `python scripts/plot_all.py` 生成所有图。
9. 执行 `python scripts/validate_outputs.py`，产出 `reports/acceptance_report.md`。
10. 通过补充实验验收后，把必要脚本/图/CSV 合并到论文引用路径。
11. 修改 `论文/paper.tex`。
12. 使用 `paper-review` 审稿框架做七维自审：
    - 新颖性与意义
    - 方法与技术正确性
    - 结果与验证
    - 可复现性
    - 相关工作与引用
    - 清晰度与组织
    - 局限性
13. 编译论文：
    - `cd standard_robot_cuda_ik/论文 && latexmk -xelatex paper.tex`
14. 最终验收：
    - 无缺图。
    - 无未定义引用。
    - 无未定义文献。
    - 表格不溢出。
    - 摘要/正文/表格/CSV/图一致。

## 7. 风险点与不可完成项

| 风险 | 影响 | 处理 |
|---|---|---|
| `cuda_v4_runner.cu` 无 `--wlimit` | barrier ON/OFF 与权重扫描不能直接运行 | 先在 `patches/` 写最小补丁，验收后合并 |
| smoothness rerank 需要候选输出 | trajectory 对比可能缺少 rerank 数据 | 增加候选 CSV 输出或在补丁中实现候选级重排序 |
| cuRobo K=16 现有 N 不全 | 公平对比可能只覆盖 N=100/500/1000 | 尝试重跑 N=200--900；失败则在报告和论文中说明 |
| KDL/TRAC-IK 环境可能不可用 | CPU baseline 方法受限 | 使用 CPU-LM-K1/K16，并说明是量级 baseline |
| nsys 可能不可用或权限不足 | timeline 图无法来自真实采样 | 记录失败命令和环境原因；不伪造 nsys 结果 |
| 论文两栏版面紧张 | 新图表过多导致页数和排版失控 | 优先保留 P0 图表，部分参数扫描放补充说明或压缩表格 |
| 实验耗时较长 | 一次性全量重跑成本高 | 先跑 N=100 smoke test，再跑验收规模 |

## 8. 预计最终新增文件清单

```text
standard_robot_cuda_ik/data/experiments/补充实验/
├── codex_补实验执行计划.md
├── README.md
├── scripts/
│   ├── common_metrics.py
│   ├── generate_special_targets.py
│   ├── run_fair_curobo_k16.py
│   ├── run_near_singular.py
│   ├── run_near_limit_barrier.py
│   ├── run_trajectory_continuity.py
│   ├── run_cpu_baseline.py
│   ├── run_seed_count_scan.py
│   ├── run_lm_iter_scan.py
│   ├── run_threshold_scan.py
│   ├── run_kernel_time_breakdown.py
│   ├── run_nsight_systems.sh
│   ├── plot_all.py
│   └── validate_outputs.py
├── inputs/
│   ├── near_singular/
│   ├── near_limit/
│   └── trajectory/
├── results/
│   ├── fair_curobo_k16_summary.csv
│   ├── near_singular_summary.csv
│   ├── near_limit_barrier_summary.csv
│   ├── trajectory_continuity_summary.csv
│   ├── cpu_baseline_summary.csv
│   ├── nsight_systems_summary.csv
│   ├── kernel_time_breakdown.csv
│   ├── barrier_weight_scan.csv
│   ├── seed_count_scan.csv
│   ├── lm_iter_scan.csv
│   └── threshold_scan.csv
├── figures/
│   ├── fig_pareto_throughput_success.pdf
│   ├── fig_thread_mapping_redraw.pdf
│   ├── fig_algorithm_pipeline_redraw.pdf
│   ├── fig_kernel_time_breakdown.pdf
│   ├── fig_seed_count_scan.pdf
│   ├── fig_barrier_weight_scan.pdf
│   ├── fig_trajectory_delta_q.pdf
│   ├── fig_trajectory_success.pdf
│   ├── fig_near_singular_sr.pdf
│   ├── fig_near_limit_barrier.pdf
│   └── fig_nsys_timeline_opt4c.pdf
├── reports/
│   ├── nsys_opt4c_n1000_summary.txt
│   ├── paper_review_report.md
│   └── acceptance_report.md
├── patches/
│   └── cuda_v4_runner_experiment.patch
└── logs/
```

## 9. 阶段验收门槛

进入 `paper.tex` 合并前，本补充实验目录至少必须满足：

1. P0 CSV 存在并通过字段检查：
   - `fair_curobo_k16_summary.csv`
   - `near_singular_summary.csv`
   - `near_limit_barrier_summary.csv`
   - `trajectory_continuity_summary.csv`
   - `seed_count_scan.csv`
   - `kernel_time_breakdown.csv`
2. P0 图存在并可由 `scripts/plot_all.py` 重新生成。
3. `reports/acceptance_report.md` 明确列出完成、缺失与无法完成原因。
4. 不存在手工编造数据；所有表和图均能追溯到 CSV 或明确的 profiler 输出。
5. 若缺失 CPU baseline、Nsight Systems 或 cuRobo 全 N 数据，必须写清环境原因，并在论文局限性中保守表述。

