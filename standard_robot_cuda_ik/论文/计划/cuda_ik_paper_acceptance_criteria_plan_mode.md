# CUDA IK 论文补实验与正文修正：Plan 模式验收标准

> 使用方式：把本文档和 `cuda_ik_paper_revision_plan_for_codex.md` 一起交给 Codex。  
> Codex 在 **plan 模式** 下必须先给出实施计划，不得直接改代码。计划通过后再执行。  
> 本文档用于判断 Codex 的计划是否合格，以及后续修改是否可以验收。

---

## 0. Plan 模式总验收原则

### 0.1 Codex 首轮回复必须是计划，不得直接修改

合格计划必须包含：

- [ ] 当前仓库结构识别结果；
- [ ] 需要修改的代码模块；
- [ ] 需要新增的实验脚本；
- [ ] 需要新增的绘图脚本；
- [ ] 需要修改的论文 `.tex` 章节；
- [ ] 每个实验的输入、输出、CSV 字段；
- [ ] 每张图对应的数据源；
- [ ] 每个任务的执行顺序；
- [ ] 风险点和不可完成项说明；
- [ ] 预计生成的文件清单。

不合格情况：

- [ ] 直接开始改代码；
- [ ] 只给泛泛建议，没有文件级任务；
- [ ] 没有说明如何复现实验；
- [ ] 没有说明如何验证结果；
- [ ] 没有把任务映射到论文正文、数据、图表三类产物。

---

## 1. 总体验收结果分级

### 1.1 通过

满足以下条件：

- [ ] 所有 P0 实验均已完成；
- [ ] 所有 P0 图表均可从 CSV 自动生成；
- [ ] 所有危险正文表述均已修正；
- [ ] `paper.tex` 可一次编译通过；
- [ ] 新增数据、图、正文之间一致；
- [ ] README 中有完整复现命令；
- [ ] 没有伪造、硬编码或无法追溯的数据。

### 1.2 有条件通过

满足：

- [ ] P0 实验完成 80% 以上；
- [ ] 缺失项已在 README 和论文局限性中说明；
- [ ] 不影响论文主结论；
- [ ] 缺失原因合理，例如环境缺少 TRAC-IK 或无第二块 GPU。

### 1.3 不通过

出现任一情况即不通过：

- [ ] cuRobo K=16 公平对比缺失；
- [ ] Near Singular 或 Near Limit 实验缺失；
- [ ] 论文仍写“完全消除 warp divergence”；
- [ ] 论文仍保留错误的 “32 与 16 互质”；
- [ ] 论文仍用 `NKW+N` 作为正常 GPU baseline；
- [ ] 实验数据只有图片，没有 CSV；
- [ ] 图表数据无法追溯；
- [ ] `.tex` 无法编译；
- [ ] 结论仍声称全面优于 cuRobo；
- [ ] 使用手工编造数据。

---

## 2. 仓库与文件结构验收标准

最终目录建议至少包含：

```text
project_root/
├── paper/
│   ├── paper.tex
│   ├── paper.pdf
│   └── references.bib 或 bibliography.bib
├── figures/
│   ├── fig_pareto_throughput_success.pdf
│   ├── fig_thread_mapping_redraw.pdf
│   ├── fig_algorithm_pipeline_redraw.pdf
│   ├── fig_kernel_time_breakdown.pdf
│   ├── fig_seed_count_scan.pdf
│   ├── fig_barrier_weight_scan.pdf
│   ├── fig_trajectory_delta_q.pdf
│   ├── fig_near_singular_sr.pdf
│   ├── fig_near_limit_barrier.pdf
│   └── fig_nsys_timeline_opt4c.pdf 或 .png
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
├── scripts/
│   ├── run_fair_curobo_k16.py 或 .sh
│   ├── run_near_singular.py 或 .sh
│   ├── run_near_limit_barrier.py 或 .sh
│   ├── run_trajectory_continuity.py 或 .sh
│   ├── run_cpu_baseline.py 或 .sh
│   ├── run_nsight_systems.sh
│   ├── run_barrier_weight_scan.py 或 .sh
│   ├── run_seed_count_scan.py 或 .sh
│   ├── run_lm_iter_scan.py 或 .sh
│   ├── run_threshold_scan.py 或 .sh
│   └── plot_all.py
├── reports/
│   ├── nsys_opt4c_n1000_summary.txt
│   └── 可选：nsys/cupti/nsight 原始报告
└── README.md
```

验收要求：

- [ ] 所有论文引用的图片都存在；
- [ ] 所有论文引用的表格都能追溯到 CSV；
- [ ] `plot_all.py` 或等效脚本可以重新生成所有新增图；
- [ ] README 中写明一键或分步复现实验命令；
- [ ] 不允许论文中引用不存在的图号、表号或数据文件。

---

## 3. 数据文件通用验收标准

所有新增 CSV 必须满足：

- [ ] UTF-8 编码；
- [ ] 第一行为字段名；
- [ ] 字段名稳定，不随脚本运行变化；
- [ ] 每行对应一个方法、一个 N、一个配置；
- [ ] 数值单位明确；
- [ ] 缺失值为空或 `NA`，不要写中文说明；
- [ ] 至少包含均值；
- [ ] 如果做 30 次重复，应包含标准差或置信区间；
- [ ] 不允许只有最终汇总而无方法配置字段。

推荐基础字段：

```text
experiment
method
robot
N
K
seed_type
max_iter
wlimit
barrier_enabled
target_type
repeat
gpu_time_ms_mean
gpu_time_ms_std
throughput_targets_per_s_mean
throughput_targets_per_s_std
strict_sr
medium_sr
loose_sr
pos_p95_all_mm
pos_p95_success_mm
rot_p95_all_deg
rot_p95_success_deg
mean_iter
p95_iter
near_limit_ratio
joint_violation_count
nan_count
inf_count
fail_count
notes
```

---

## 4. P0 实验逐项验收标准

## 4.1 cuRobo K=16 公平对比

### 必须产物

- [ ] `results/fair_curobo_k16_summary.csv`
- [ ] `figures/fig_pareto_throughput_success.pdf`
- [ ] 论文新增“同等 K=16 公平对比”表格；
- [ ] 论文新增对 cuRobo-K16 的客观讨论。

### 实验配置验收

必须至少包含：

```text
N = 100, 200, 300, ..., 1000
method = OPT4C-K16, cuRobo-Graph-K16
```

强烈建议同时包含：

```text
OPT4C-K1
cuRobo-Graph-K1
```

### 指标验收

CSV 中必须包含：

- [ ] `throughput_targets_per_s_mean`
- [ ] `strict_sr`
- [ ] `pos_p95_all_mm`
- [ ] `pos_p95_success_mm`
- [ ] `gpu_time_ms_mean`
- [ ] `K`
- [ ] `method`

### 正文验收

正文必须明确：

- [ ] 默认 cuRobo-K1 对比是系统默认配置对比；
- [ ] cuRobo-K16 是同等种子数量公平对比；
- [ ] 不得声称 OPT4C 全面优于 cuRobo；
- [ ] 必须说明二者是 Pareto trade-off。

### 不通过条件

- [ ] 只比较 cuRobo-K1；
- [ ] 没有说明 cuRobo 是否开启 CUDA Graph；
- [ ] 没有外部 FK 复评；
- [ ] 没有写 cuRobo 的 `num_seeds`；
- [ ] 图中只画 OPT4C，不画 cuRobo-K16。

---

## 4.2 Near Singular 近奇异实验

### 必须产物

- [ ] `results/near_singular_summary.csv`
- [ ] `figures/fig_near_singular_sr.pdf`
- [ ] 可选：`figures/fig_near_singular_error.pdf`
- [ ] 论文新增近奇异小节。

### 目标类型验收

至少包含一种近奇异类型：

```text
wrist_singular
```

建议包含：

```text
wrist_singular
elbow_singular
shoulder_singular
```

### 规模验收

至少：

```text
N = 100, 500, 1000
```

### 方法验收

至少：

```text
OPT4C-K16
OPT4C-K1
```

建议：

```text
cuRobo-K16
```

### 指标验收

必须包含：

- [ ] `strict_sr`
- [ ] `pos_p95_all_mm`
- [ ] `pos_p95_success_mm`
- [ ] `rot_p95_success_deg`
- [ ] `mean_iter`
- [ ] `p95_iter`
- [ ] `fail_count`
- [ ] `nan_count`

### 正文验收

必须说明：

- [ ] 近奇异目标如何生成；
- [ ] 与随机可达目标相比，成功率是否下降；
- [ ] 若下降，必须承认适用边界；
- [ ] 不允许只挑结果好的奇异类型汇报。

---

## 4.3 Near Limit 与 Barrier ON/OFF 实验

### 必须产物

- [ ] `results/near_limit_barrier_summary.csv`
- [ ] `figures/fig_near_limit_barrier.pdf`
- [ ] 论文新增 Barrier ON/OFF 对比表。

### 实验配置验收

必须至少包含：

```text
target_type = near_limit
method = OPT4C-K16-BarrierON
method = OPT4C-K16-BarrierOFF
N = 100, 500, 1000
```

建议同时包含：

```text
OPT4C-K1-BarrierON
OPT4C-K1-BarrierOFF
```

### 指标验收

必须包含：

- [ ] `strict_sr`
- [ ] `near_limit_ratio`
- [ ] `joint_violation_count`
- [ ] `pos_p95_all_mm`
- [ ] `pos_p95_success_mm`
- [ ] `mean_iter`

### 正文验收

必须说明：

- [ ] Barrier 的主要作用是降低 near-limit 解比例；
- [ ] 不要把 Barrier 写成主要成功率来源；
- [ ] 如果 Barrier 牺牲成功率，要如实写；
- [ ] 必须说明 `margin = 0.087 rad` 和 `wlimit` 的取值依据。

---

## 4.4 Trajectory 连续性实验

### 必须产物

- [ ] `results/trajectory_continuity_summary.csv`
- [ ] `figures/fig_trajectory_delta_q.pdf`
- [ ] `figures/fig_trajectory_success.pdf` 或等效图
- [ ] 论文新增轨迹连续性小节。

### 轨迹类型验收

至少包含两类：

```text
line_50
arc_50
```

建议包含：

```text
random_local_50
```

### 方法验收

至少包含：

```text
OPT4C-K16-no-rerank
OPT4C-K16-smoothness-rerank
```

建议：

```text
cuRobo-K16
```

### 指标验收

必须包含：

- [ ] `point_success_rate`
- [ ] `trajectory_success_rate`
- [ ] `mean_delta_q`
- [ ] `p95_delta_q`
- [ ] `max_delta_q`
- [ ] `joint_jump_count`

### 正文验收

必须说明：

- [ ] 多起点选择可能导致分支跳变；
- [ ] smoothness rerank 是否降低 `delta_q`；
- [ ] smoothness rerank 是否牺牲成功率；
- [ ] 不得只报告成功率而不报告跳变指标。

---

## 4.5 CPU Baseline 实验

### 必须产物

- [ ] `results/cpu_baseline_summary.csv`
- [ ] 论文新增 CPU baseline 表。

### 方法验收

至少包含一种：

```text
KDL
TRAC-IK
CPU-LM-K1
CPU-LM-K16
```

如果 KDL/TRAC-IK 环境不可用，必须在 README 中说明，并用 CPU-LM baseline 替代。

### 规模验收

至少：

```text
N = 100
```

建议：

```text
N = 100, 500, 1000
```

### 指标验收

必须包含：

- [ ] `time_ms`
- [ ] `throughput_targets_per_s`
- [ ] `strict_sr`
- [ ] `pos_p95_all_mm`
- [ ] `device = CPU/GPU`

### 正文验收

必须说明：

- [ ] CPU baseline 用于量级对照；
- [ ] 不要暗示 GPU 对单目标 IK 必然更优；
- [ ] 重点是批量 N 达到百级后的吞吐优势。

---

## 4.6 Nsight Systems Timeline 实验

### 必须产物

- [ ] `reports/nsys_opt4c_n1000_summary.txt`
- [ ] `figures/fig_nsys_timeline_opt4c.pdf` 或 `.png`
- [ ] README 中包含 Nsight Systems 采样命令。

### 图像验收

图中必须标注：

- [ ] H2D；
- [ ] single IK kernel；
- [ ] D2H；
- [ ] N=1000；
- [ ] K=16。

### 正文验收

必须说明：

- [ ] 核心求解阶段是单个长 kernel；
- [ ] 没有多阶段 host-device 同步链；
- [ ] 该图用于验证执行路径，不用于证明数值精度。

---

## 4.7 Kernel Time Breakdown

### 必须产物

- [ ] `results/kernel_time_breakdown.csv`
- [ ] `figures/fig_kernel_time_breakdown.pdf`

### 规模验收

至少：

```text
N = 100, 500, 1000
```

### 字段验收

必须包含：

- [ ] `h2d_ms`
- [ ] `kernel_ms`
- [ ] `d2h_ms`
- [ ] `launch_ms`
- [ ] `total_ms`
- [ ] `h2d_percent`
- [ ] `kernel_percent`
- [ ] `d2h_percent`
- [ ] `launch_percent`

### 正文验收

必须说明：

- [ ] fusion 后瓶颈转向 kernel 内部算术；
- [ ] 不再把本文方法的主要瓶颈写成 launch；
- [ ] 后续优化方向应是 FP64/三角函数/小矩阵求解。

---

## 4.8 Barrier 权重扫描

### 必须产物

- [ ] `results/barrier_weight_scan.csv`
- [ ] `figures/fig_barrier_weight_scan.pdf`

### 参数验收

至少包含：

```text
wlimit = 0
wlimit = 0.01
wlimit = 0.03
wlimit = 0.1
```

建议：

```text
0, 0.005, 0.01, 0.03, 0.05, 0.1
```

### 指标验收

必须包含：

- [ ] `strict_sr`
- [ ] `near_limit_ratio`
- [ ] `pos_p95_all_mm`
- [ ] `mean_iter`

### 正文验收

必须说明：

- [ ] 为什么默认取 `wlimit=0.03`；
- [ ] 是否存在成功率与 near-limit ratio 的折中；
- [ ] 不得把 `wlimit=0.03` 写成未经验证的经验值。

---

## 4.9 Seed 数量扫描

### 必须产物

- [ ] `results/seed_count_scan.csv`
- [ ] `figures/fig_seed_count_scan.pdf`

### 参数验收

至少包含：

```text
K = 1, 2, 4, 8, 16
```

建议包含：

```text
K = 32
```

### 规模验收

至少：

```text
N = 1000
```

建议：

```text
N = 100, 500, 1000
```

### 指标验收

必须包含：

- [ ] `strict_sr`
- [ ] `throughput_targets_per_s`
- [ ] `gpu_time_ms`
- [ ] `pos_p95_all_mm`
- [ ] `mean_iter`

### 正文验收

必须说明：

- [ ] 多起点收益随 K 增加的趋势；
- [ ] K=16 是否接近饱和；
- [ ] 为什么不是 K=1、K=8 或 K=32；
- [ ] 不得只用 K=1 vs K=16 支撑所有结论。

---

## 4.10 LM 最大迭代次数扫描

### 必须产物

- [ ] `results/lm_iter_scan.csv`
- [ ] `figures/fig_lm_iter_scan.pdf`

### 参数验收

至少包含：

```text
max_iter = 20, 40, 60, 80
```

建议：

```text
20, 40, 60, 80, 100
```

### 指标验收

必须包含：

- [ ] `strict_sr`
- [ ] `throughput_targets_per_s`
- [ ] `mean_iter`
- [ ] `p95_iter`
- [ ] `pos_p95_all_mm`

### 正文验收

必须说明：

- [ ] 为什么默认取 `max_iter=60`；
- [ ] 增大迭代次数是否继续提升成功率；
- [ ] 迭代次数与时间是否近似线性；
- [ ] 不得无依据固定 60。

---

## 4.11 不同误差阈值扫描

### 必须产物

- [ ] `results/threshold_scan.csv`
- [ ] `figures/fig_threshold_scan.pdf`

### 阈值验收

至少包含：

```text
Loose: 30 mm / 10 deg
Medium: 10 mm / 5 deg
Strict: 5 mm / 1 deg
```

建议包含：

```text
Ultra: 2 mm / 0.5 deg
```

### 方法验收

至少包含：

```text
OPT4C-K16
OPT4C-K1
```

建议：

```text
cuRobo-K16
cuRobo-K1
```

### 正文验收

必须说明：

- [ ] OPT4C 在哪些阈值下优势明显；
- [ ] 如果 Ultra 下 cuRobo 更强，必须如实承认；
- [ ] 不得把 5 mm / 1° 泛化为所有工业任务要求。

---

# 5. 图表验收标准

## 5.1 通用图表要求

所有图必须满足：

- [ ] 可由脚本自动生成；
- [ ] 坐标轴有单位；
- [ ] 图例清晰；
- [ ] 字号在论文 PDF 中可读；
- [ ] 黑白打印仍能区分；
- [ ] 图题和正文引用一致；
- [ ] 不使用截图替代可生成图，Nsight timeline 除外；
- [ ] 不允许图中数据与 CSV 不一致。

## 5.2 必须新增或重绘图

| 图文件 | 验收内容 |
|---|---|
| `fig_pareto_throughput_success.pdf` | OPT4C-K1、OPT4C-K16、cuRobo-K1、cuRobo-K16 至少四类点 |
| `fig_thread_mapping_redraw.pdf` | 清楚画出 block-target、lane-seed、shared memory、lane 0 selection |
| `fig_algorithm_pipeline_redraw.pdf` | 输入、FK、Jacobian、LM、barrier、selection、输出完整 |
| `fig_kernel_time_breakdown.pdf` | H2D/kernel/D2H/launch 堆叠图 |
| `fig_seed_count_scan.pdf` | K 对 SR 和 throughput 的影响 |
| `fig_barrier_weight_scan.pdf` | wlimit 对 SR 和 near-limit ratio 的影响 |
| `fig_trajectory_delta_q.pdf` | 至少比较 no-rerank 与 smoothness-rerank |
| `fig_near_singular_sr.pdf` | 近奇异目标成功率 |
| `fig_near_limit_barrier.pdf` | Barrier ON/OFF 对比 |
| `fig_nsys_timeline_opt4c.pdf` | 单 kernel timeline |

---

# 6. 正文修正验收标准

## 6.1 摘要

必须满足：

- [ ] 不再声称全面超过 cuRobo；
- [ ] 不再写 “Bank 冲突为零”，除非有明确计数器；
- [ ] 不再把方法称为 FPGA/ASIC 意义上的“硬件固化”；
- [ ] 明确本文边界：固定 6-DOF、中小批量、成功率优先。

## 6.2 引言

必须满足：

- [ ] 区分“通用多阶段 pipeline 的调度开销”和“本文 fusion 后的算术瓶颈”；
- [ ] 不再说本文方法中 GPU launch 是主要瓶颈；
- [ ] 不再用不合理的 `NKW+N` 描述正常 batched GPU baseline。

## 6.3 方法部分

必须满足：

- [ ] “无分支 LM”改为“低控制流复杂度 LM”；
- [ ] 删除“完全消除 warp divergence”；
- [ ] 删除“32 与 16 互质”；
- [ ] `lane` 术语统一；
- [ ] 解析雅可比优势改为“减少 FK 调用、避免差分步长调参、改善梯度稳定性”；
- [ ] 不再写“差分误差接近 5 mm 阈值”。

## 6.4 实验部分

必须满足：

- [ ] 增加实验协议与公平性设置；
- [ ] 明确所有方法的 K、迭代次数、计时方式；
- [ ] 增加 cuRobo K=16；
- [ ] 增加 Near Singular；
- [ ] 增加 Near Limit；
- [ ] 增加 Trajectory；
- [ ] 增加 CPU baseline；
- [ ] 增加 Nsight Systems；
- [ ] 增加 seed / barrier / max_iter / threshold 参数扫描。

## 6.5 讨论与结论

必须满足：

- [ ] 结论收敛，不写“确立全新范式”之类过强表述；
- [ ] 必须承认 cuRobo 在大批量、高精度、通用规划方面仍有优势；
- [ ] 必须写清楚适用边界；
- [ ] 必须写局限性：
  - 固定 UR10/6DOF；
  - 未考虑碰撞；
  - 目标主要由 FK 生成；
  - FP64 吞吐低；
  - K=16 增加计算量；
  - 轨迹连续性需要 rerank 或 warm-start。

---

# 7. 编译与复现验收标准

## 7.1 LaTeX 编译

必须通过：

```bash
cd paper
latexmk -xelatex paper.tex
```

或：

```bash
xelatex paper.tex
bibtex paper
xelatex paper.tex
xelatex paper.tex
```

验收：

- [ ] 无缺图错误；
- [ ] 无未定义引用 `??`；
- [ ] 无未定义文献 `[?]`；
- [ ] 无表格溢出页面；
- [ ] PDF 中图中文字可读；
- [ ] 图表编号连续。

## 7.2 图表生成

必须通过：

```bash
python scripts/plot_all.py
```

验收：

- [ ] 所有新增图自动生成；
- [ ] 生成时间不依赖手工操作；
- [ ] 图中数据来自 `results/*.csv`；
- [ ] 不直接从论文 PDF 截图造图。

## 7.3 实验复现

README 至少包含：

```bash
# 1. 静态随机目标
...

# 2. cuRobo K=16
...

# 3. Near singular
...

# 4. Near limit
...

# 5. Trajectory
...

# 6. CPU baseline
...

# 7. 参数扫描
...

# 8. 绘图
python scripts/plot_all.py

# 9. 编译论文
cd paper && latexmk -xelatex paper.tex
```

验收：

- [ ] 每条命令都能在项目根目录执行或明确说明工作目录；
- [ ] 依赖版本写清楚；
- [ ] GPU 型号、CUDA 版本、驱动版本写清楚；
- [ ] cuRobo 版本写清楚。

---

# 8. 数据一致性验收

必须逐项检查：

- [ ] 摘要中的 SR、throughput、p95 与正文表格一致；
- [ ] 正文表格与 CSV 一致；
- [ ] 图中数据与 CSV 一致；
- [ ] 结论中引用的百分比与表格计算一致；
- [ ] “提升 xx%”和“提升 xx 个百分点”没有混用；
- [ ] K=1 vs K=16 的成功率差值计算正确；
- [ ] 如果写“约 35% 速度代价”，必须有数据支持；
- [ ] 如果写“p95 < 5 mm”，必须注明范围，例如 N≥500 或全部 N；
- [ ] 如果某些 N 下 p95 超过 5 mm，不得在摘要中笼统写全部低于 5 mm。

---

# 9. 术语与表述验收

全文替换或统一：

| 原表述 | 建议表述 |
|---|---|
| 硬件固化 | 编译期特化 / 结构感知特化 |
| 经线 | lane / 线程通道 |
| 无分支 LM | 低控制流复杂度 LM |
| 消除 warp divergence | 降低控制流复杂度 |
| Bank 冲突为零 | 共享内存不是主要瓶颈 / 实测冲突率低 |
| NKW+N | 朴素逐目标实现 / 多阶段 batched pipeline |
| 工业抓取 5 mm 要求 | 粗定位抓取或规划前端可用阈值 |
| 全面超过 cuRobo | 与 cuRobo 形成 Pareto 取舍 |

---

# 10. 最终交付验收清单

Codex 完成后，最终交付必须包含：

- [ ] 修改后的源码；
- [ ] 修改后的 `paper.tex`；
- [ ] 新生成的 `paper.pdf`；
- [ ] 所有新增 `results/*.csv`；
- [ ] 所有新增 `figures/*.pdf/.png`；
- [ ] 所有新增实验脚本；
- [ ] `plot_all.py`；
- [ ] README 复现说明；
- [ ] 变更摘要 `CHANGELOG.md` 或在回复中列出；
- [ ] 无法完成项说明。

---

# 11. 最低可接受版本

如果时间不够，最低可接受版本必须完成：

1. cuRobo K=16 公平对比；
2. Near Singular；
3. Near Limit + Barrier ON/OFF；
4. Trajectory continuity；
5. Seed count scan；
6. Kernel time breakdown；
7. 正文危险表述修正：
   - warp divergence；
   - bank conflict；
   - NKW+N；
   - 硬件固化；
   - 5 mm 工业泛化；
8. `paper.tex` 可编译；
9. 图表可从 CSV 生成。

如果这 9 项没有完成，不建议进入投稿版本。

---

# 12. 最终验收结论模板

Codex 完成后，用下面模板验收：

```text
验收结论：通过 / 有条件通过 / 不通过

一、实验完成情况
- cuRobo K=16：
- Near Singular：
- Near Limit：
- Trajectory：
- CPU baseline：
- Nsight Systems：
- 参数扫描：

二、正文修正情况
- 摘要：
- 引言：
- 方法：
- 实验：
- 讨论与结论：

三、数据与图表一致性
- CSV：
- 图：
- 表：
- 摘要数值：

四、编译与复现
- paper.tex：
- plot_all.py：
- README：

五、仍需人工检查的问题
1.
2.
3.
```
