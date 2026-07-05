# CUDA IK 论文图表全面升级任务书（给 Codex）

> 目标：对当前论文中的约 8 张核心图片进行替换和全面升级，使用高级绘图库生成统一、期刊化、可复现的图表，并自动插入 `paper.tex`。  
> 适用论文：当前最新版 CUDA IK 论文。  
> 核心原则：图表不是装饰，而是服务于论文主线：**OPT4C 在固定 6-DOF、中小批量、无碰撞 IK 前端中，通过 Sobol 多起点 + 单核融合，在满足中等精度阈值时形成与 cuRobo 的吞吐量—成功率—精度 Pareto 取舍。**

---

## 0. 总体绘图要求

### 0.1 使用高级绘图库

优先使用 Python 绘图栈：

```text
matplotlib
seaborn
pandas
numpy
scipy
```

可选：

```text
plotly 仅用于探索，不建议用于最终论文 PDF
scienceplots 可选，但不要强依赖
adjustText 可选，用于避免标注重叠
```

最终输出格式：

```text
figures/*.pdf
```

同时可导出预览：

```text
figures_preview/*.png
```

论文中统一插入 PDF。

---

## 0.2 风格要求

统一风格：

```python
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "SimSun", "Noto Serif CJK SC", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
```

如果中文字体不可用，自动 fallback，不要让脚本崩溃。

建议配色：

```text
OPT4C-K16：深蓝
OPT4C-K1：浅蓝或蓝灰
cuRobo-K16：深橙 / 红橙
cuRobo-K1：浅橙 / 棕橙
Barrier ON：蓝
Barrier OFF：灰
No rerank：灰
Smoothness rerank：蓝
```

但必须确保：

- 黑白打印可区分；
- 使用不同 marker / line style；
- 不只依赖颜色。

推荐 marker：

```text
OPT4C-K16：circle
OPT4C-K1：square
cuRobo-K16：diamond
cuRobo-K1：triangle
```

线型：

```text
OPT4C-K16：solid
OPT4C-K1：dashed
cuRobo-K16：dashdot
cuRobo-K1：dotted
```

---

## 0.3 统一图尺寸

建议：

### 单栏图

```python
figsize=(3.35, 2.45)
```

### 双栏图

```python
figsize=(6.9, 2.8)
```

### 热力图 / 矩阵图

```python
figsize=(6.9, 3.2)
```

### 架构图

```python
figsize=(6.9, 3.0)
```

输出：

```python
plt.savefig(path, bbox_inches="tight", dpi=300)
```

---

## 0.4 数据来源要求

所有图必须从 CSV 或可复现实验结果文件读取，不允许在绘图脚本中硬编码最终数据。

建议数据目录：

```text
results/
├── static_summary.csv
├── fair_curobo_k16_summary.csv
├── threshold_scan.csv
├── seed_count_scan.csv
├── near_singular_summary.csv
├── near_limit_barrier_summary.csv
├── barrier_weight_scan.csv
├── trajectory_continuity_summary.csv
├── kernel_time_breakdown.csv
├── nsight_compute_summary.csv
├── candidate_success_matrix.csv
├── candidate_cost_matrix.csv
└── trajectory_deltaq_matrix.csv
```

如果某些矩阵型图当前没有 CSV，请 Codex 从已有 raw / runner 输出中生成对应 CSV。  
若确实无法生成某张图的数据，则不要伪造；在 `CHANGELOG.md` 中说明缺失原因。

---

## 0.5 脚本要求

新增或重构绘图入口：

```text
scripts/plot_all_figures.py
```

运行：

```bash
python scripts/plot_all_figures.py
```

应生成：

```text
figures/fig1_thread_mapping.pdf
figures/fig2_time_breakdown.pdf
figures/fig3_static_performance.pdf
figures/fig4_pareto_front.pdf
figures/fig5_threshold_scan.pdf
figures/fig6_seed_success_heatmap.pdf
figures/fig7_trajectory_deltaq_heatmap.pdf
figures/fig8_robustness_boundary.pdf
```

同时生成 PNG 预览：

```text
figures_preview/*.png
```

---

# 1. 建议替换的 8 张核心图

---

## Figure 1：Target-Block 线程映射架构图

### 替换对象

当前论文中的：

```text
图 1 Target-Block 线程映射架构
```

### 图类型

高级流程架构图 / CUDA 数据流图。

### 绘图建议

使用 `matplotlib.patches` 手动画，不要使用简单 flowchart 截图。

结构：

```text
Grid: N blocks
        ↓
Block i → Target i
        ↓
Lane 0–15 → Sobol seeds
Lane 16–31 → inactive
        ↓
FK + Analytical Jacobian + LM per active lane
        ↓
Shared candidate buffer s_cand[16][16]
        ↓
Lane 0 hierarchical selection
        ↓
Global best output: q*, error, flags
```

视觉要求：

- 使用横向布局；
- block / lane / shared memory / output 使用不同浅色背景；
- 箭头清晰；
- lane 0–15 与 lane 16–31 分开；
- 图中文字足够大；
- 避免过度花哨；
- 最终黑白打印也能识别。

### 输出文件

```text
figures/fig1_thread_mapping.pdf
```

### LaTeX 替换建议

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=0.95\textwidth]{figures/fig1_thread_mapping.pdf}
  \caption{Target-Block 线程映射架构。一个 block 处理一个目标位姿，lane 0--15 分别处理 16 个 Sobol 起点，lane 16--31 保留为空闲线程；候选解暂存于共享内存，并由 lane 0 完成块内最佳候选选择后写入全局输出。}
  \label{fig:thread_mapping}
\end{figure*}
```

### 验收标准

- [ ] 图中文字清楚；
- [ ] 结构逻辑完整；
- [ ] 可以替代当前图 1；
- [ ] 与算法 3 术语一致；
- [ ] 图注和正文引用更新。

---

## Figure 2：GPU 时间组成与非 kernel 放大图

### 替换对象

当前论文中的：

```text
图 3 OPT4C 时间组成
```

### 图类型

双子图：

```text
(a) stacked percentage bar
(b) non-kernel absolute time zoom
```

### 数据来源

```text
results/kernel_time_breakdown.csv
```

字段建议：

```text
N
h2d_ms
launch_ms
kernel_ms
d2h_ms
total_ms
h2d_percent
launch_percent
kernel_percent
d2h_percent
```

### 绘图设计

左图：

- x-axis：N = 100, 500, 1000
- y-axis：时间占比 / %
- stacked bar：H2D、Launch、Core kernel、D2H

右图：

- x-axis：N = 100, 500, 1000
- y-axis：非 kernel 时间 / ms
- stacked bar：H2D、Launch、D2H
- 不画 kernel，突出小项

### 输出文件

```text
figures/fig2_time_breakdown.pdf
```

### 图注建议

```latex
\caption{OPT4C 的 GPU 时间组成与非 kernel 时间放大图。Kernel fusion 后，总时间主要由核心 IK kernel 贡献，H2D、D2H 和 launch 开销均为次要项。}
```

### 验收标准

- [ ] 能看清非 kernel 时间；
- [ ] 不再只有一大块 kernel；
- [ ] 单位明确；
- [ ] 数据与正文描述一致。

---

## Figure 3：静态批量性能总览图

### 替换对象

当前论文中的：

```text
图 5 默认配置批量吞吐量对比
图 6 默认配置 Strict 成功率对比
```

可以合并为一张双子图。

### 图类型

双子图：

```text
(a) Throughput vs N
(b) Strict SR vs N
```

### 数据来源

```text
results/static_summary.csv
results/fair_curobo_k16_summary.csv
```

### 绘图对象

至少画：

```text
OPT4C-K16
cuRobo-Graph-K1
```

可选加入浅色虚线：

```text
cuRobo-Graph-K16
```

但如果图太乱，则只保留默认配置，图注说明 K16 公平对比见 Pareto 图。

### 绘图要求

- x-axis：batch size N
- y-axis 左图：Throughput / targets s^-1，使用科学计数或 10^4 单位；
- y-axis 右图：Strict SR / %
- 使用 marker + line；
- 图例明确写 K1/K16；
- 不出现含混的 “cuRobo-Graph”。

### 输出文件

```text
figures/fig3_static_performance.pdf
```

### 图注建议

```latex
\caption{静态批量 IK 默认配置性能对比。图中 cuRobo-Graph 为默认 K=1 配置；同等 K=16 的公平对比见图~\ref{fig:pareto} 和表~\ref{tab:curobo_compare}。}
```

### 验收标准

- [ ] 替代当前图 5 和图 6；
- [ ] 图例明确 K；
- [ ] y 轴单位清楚；
- [ ] 正文引用更新。

---

## Figure 4：增强版 Pareto 前沿图

### 替换对象

当前论文中的：

```text
图 7 吞吐量–成功率帕累托对比
```

### 图类型

高级 scatter / bubble Pareto 图。

### 数据来源

```text
results/fair_curobo_k16_summary.csv
```

使用：

```text
N = 1000
methods:
OPT4C-K16
OPT4C-K1
cuRobo-Graph-K16
cuRobo-Graph-K1
```

### 绘图设计

横轴：

```text
Throughput / targets s^-1
```

纵轴：

```text
Strict success rate
```

点大小：

```text
success p95 或 all-sample p95 的反向尺度
```

但注意 all-sample p95 差距极大，建议：

```text
marker size = clipped/log-scaled p95
```

或者更稳妥：

```text
颜色表示 all-sample p95，采用 log norm
点形状表示方法家族
```

建议设计：

- x：throughput；
- y：Strict SR；
- marker：方法；
- color：log10(all-sample p95 + 1)；
- 添加颜色条：All-sample p95 / mm；
- 标注每个点；
- 加一条淡灰色 Pareto envelope 或箭头说明 trade-off。

### 输出文件

```text
figures/fig4_pareto_front.pdf
```

### 图注建议

```latex
\caption{N=1000 时吞吐量--成功率--误差的 Pareto 对比。颜色表示全样本位置误差 p95，点的位置表示吞吐量与 Strict 成功率。cuRobo-Graph-K16 位于高精度端，OPT4C-K16 位于满足 Strict 阈值下的吞吐优先区域。}
```

### 验收标准

- [ ] 四个方法点均存在；
- [ ] 点标注不重叠；
- [ ] 颜色条清楚；
- [ ] 数据与表 6 一致；
- [ ] 图形明显比原图更有论文感。

---

## Figure 5：阈值扫描图

### 替换对象

当前论文中的：

```text
图 8 不同误差阈值下的成功率扫描
```

### 图类型

多方法 line plot / slope chart。

### 数据来源

```text
results/threshold_scan.csv
```

字段：

```text
method
threshold_level
position_threshold_mm
rotation_threshold_deg
success_rate
```

### 绘图对象

```text
OPT4C-K16
OPT4C-K1
cuRobo-Graph-K16
cuRobo-Graph-K1
```

### 绘图要求

- x-axis：Loose / Medium / Strict / Ultra
- y-axis：Success rate
- 标注 Strict 位置，例如竖向浅灰背景或 marker highlight；
- 统一 marker；
- y 轴范围建议 0.2–1.02，避免低值被压扁；
- 图中注释：
  - `Strict = 5 mm / 1°`
  - `Ultra = 2 mm / 0.5°`

### 输出文件

```text
figures/fig5_threshold_scan.pdf
```

### 图注建议

```latex
\caption{不同误差阈值下的成功率扫描。OPT4C-K16 主要满足 Strict 附近的中等精度阈值，而 cuRobo-Graph-K16 在 Ultra 阈值下仍保持较高成功率。}
```

### 验收标准

- [ ] 图和表 7 数据一致；
- [ ] Strict 阈值突出；
- [ ] 图例清楚；
- [ ] 不混淆 Strict 和 Ultra 结论。

---

## Figure 6：Target × Seed 成功等级热力图

### 新增图，建议替换或插入到种子消融章节

当前种子数消融章节只有 line plot。建议新增一张更有表现力的热力图。

### 图类型

矩阵热力图：

```text
rows: target index
columns: seed index 0–15
color: success rank
```

Success rank 编码建议：

```text
0 = Fail
1 = Loose
2 = Medium
3 = Strict
```

或者：

```text
0 = Fail
1 = non-Strict success
2 = Strict
```

### 数据来源

需要生成：

```text
results/candidate_success_matrix.csv
```

格式建议：

```text
target_id, seed_id, success_rank, pos_error_mm, rot_error_deg, pose_cost
```

选择数据：

```text
N = 100 或 N = 200
```

不要画 N=1000，否则热力图太密。建议抽取代表性目标：

- 目标数 100；
- 或按 K=16 中存在 seed 差异的目标筛选前 100 个；
- 排序方式：按最佳 seed 的 success rank 或按 target difficulty 排序。

### 绘图设计

- 使用 `seaborn.heatmap`
- x-axis：Seed index
- y-axis：Target index / sorted target
- colormap：离散色图
- colorbar：Fail / Loose / Medium / Strict
- 在图注中说明目标排序方式。

### 输出文件

```text
figures/fig6_seed_success_heatmap.pdf
```

### 图注建议

```latex
\caption{Target--Seed 成功等级热力图。每列对应一个 Sobol 起点，每行对应一个目标位姿，颜色表示该 seed 的收敛等级。大量目标仅由部分 seed 达到 Strict，说明 K=16 多起点覆盖是一次性成功率的主要来源。}
```

### 插入位置

放在：

```text
4.4 种子数消融实验
```

在图 9 之前或之后。

### 验收标准

- [ ] 不是随机伪造；
- [ ] 来自真实 candidate 结果；
- [ ] 能直观看到不同 seed 收敛差异；
- [ ] colorbar 离散标签清楚；
- [ ] 图中文字可读。

---

## Figure 7：轨迹 Δq 热力图

### 替换或增强当前图 13

当前图 13 是 bar chart，只显示 p95。建议加入热力图，更直观展示 jump 分布。

### 图类型

双热力图：

```text
(a) no rerank
(b) smoothness rerank
```

矩阵：

```text
rows: trajectory id
columns: time step
color: ||Δq||
```

### 数据来源

需要生成：

```text
results/trajectory_deltaq_matrix.csv
```

字段：

```text
trajectory_type
trajectory_id
step
method
delta_q_norm
is_jump
```

建议只画一个代表性轨迹类型，或画三类合并：

方案 A：

```text
只画 random_local_50，最能体现局部随机轨迹
```

方案 B：

```text
画三类轨迹，每类一个小 panel
```

如果页面空间有限，选方案 A。

### 绘图设计

- 使用 `seaborn.heatmap`
- x-axis：step 1–49
- y-axis：trajectory id 1–20
- colorbar：||Δq|| / rad
- 在 colorbar 或图中标出 jump threshold = 0.5 rad；
- 可用相同色标范围，便于 no rerank vs rerank 对比；
- 不要因为极端值导致大部分颜色看不清，可用 `vmax=pi` 或 `vmax=4`，并说明 clipped。

### 输出文件

```text
figures/fig7_trajectory_deltaq_heatmap.pdf
```

### 图注建议

```latex
\caption{轨迹相邻 IK 解跳变量热力图。颜色表示相邻点关节差范数 $\|\Delta q\|$，其中 $\Delta q$ 使用旋转关节环绕差分计算。Smoothness rerank 能降低部分跳变幅值，但大量相邻段仍超过 0.5 rad 阈值，说明该方法不能直接保证连续控制轨迹。}
```

### 验收标准

- [ ] 使用 wrap 差分数据；
- [ ] 同一色标对比 no rerank 和 rerank；
- [ ] jump threshold 在图注说明；
- [ ] 与正文 jump ratio 一致。

---

## Figure 8：鲁棒性边界综合图

### 替换对象

当前：

```text
图 10 近奇异目标成功率
图 11 近限位 ON/OFF
图 12 Barrier 权重扫描
```

可以合并成一张 2×2 综合图，减少图数量并提升整体感。

### 图类型

2×2 panel：

```text
(a) Near singular SR by type
(b) Barrier ON/OFF: Strict SR and near-limit ratio
(c) Barrier weight scan: Strict SR vs wlimit
(d) Barrier weight scan: near-limit ratio vs wlimit
```

### 数据来源

```text
results/near_singular_summary.csv
results/near_limit_barrier_summary.csv
results/barrier_weight_scan.csv
```

### 绘图设计

Panel (a):

- grouped bar；
- x-axis：wrist / elbow / shoulder；
- bars：OPT4C-K1、OPT4C-K16；
- y-axis：Strict SR。

Panel (b):

- two grouped bars 或 twin axis；
- 不建议复杂 twin axis，可拆成两个并排 bar：
  - Strict SR；
  - near-limit ratio。

Panel (c):

- line plot；
- x-axis：wlimit；
- y-axis：Strict SR。

Panel (d):

- line plot；
- x-axis：wlimit；
- y-axis：near-limit ratio。

### 输出文件

```text
figures/fig8_robustness_boundary.pdf
```

### 图注建议

```latex
\caption{近奇异与近限位场景下的鲁棒性边界。近奇异实验显示 elbow singular 下成功率明显下降；近限位和权重扫描表明关节限位正则主要轻微降低 near-limit 解比例，而非显著提升成功率。}
```

### 验收标准

- [ ] 四个 panel 数据准确；
- [ ] 图注明确 barrier 不是主要成功率来源；
- [ ] y 轴单位和范围合理；
- [ ] 能替代当前图 10–12 或作为综合主图。

---

# 2. 可以考虑但不一定加入的高级图

以下图可以作为备选，如果版面允许再加入。

---

## Optional Figure A：候选解代价热力图

### 图类型

```text
target × seed pose cost heatmap
```

颜色：

```text
log10(pose_cost + eps)
```

价值：

- 展示不同 seed 的局部收敛差异；
- 支撑多起点策略；
- 比 success-rank heatmap 更连续。

输出：

```text
figures/optional_candidate_cost_heatmap.pdf
```

如果和 success-rank heatmap 二选一，优先选择 success-rank heatmap。

---

## Optional Figure B：Sobol vs Random seed 覆盖图

### 图类型

关节空间种子覆盖图。

方法：

```text
生成 K=16 Sobol seeds 与 Random seeds
对 6D joint space 做 PCA 投影到 2D
scatter plot
```

价值：

- 解释 Sobol 为什么合理；
- 增强方法直观性。

风险：

- 如果没有 Random-K16 实验支撑，容易显得装饰；
- 不建议作为主结果图。

输出：

```text
figures/optional_seed_coverage_pca.pdf
```

---

## Optional Figure C：LM 多 seed 收敛曲线族

### 图类型

```text
x-axis: iteration
y-axis: pose cost or position error
lines: seed 0–15
```

价值：

- 展示不同 seed 收敛路径；
- 方法解释性强。

风险：

- 需要保存 per-iteration log；
- 如果没有日志，不要伪造。

输出：

```text
figures/optional_lm_convergence_seeds.pdf
```

---

## Optional Figure D：寄存器限制—溢出—吞吐图

### 图类型

双 y-axis 或三 panel：

```text
register limit vs registers/thread
register limit vs spill bytes
register limit vs throughput
```

价值：

- 支撑微架构分析；
- 替代表 10 或增强表 10。

风险：

- 若数据点只有 3 个，图的价值有限；
- 可以保留表格，不强制画图。

输出：

```text
figures/optional_register_spill_tradeoff.pdf
```

---

# 3. 建议最终论文图表配置

如果论文控制在 8 张图左右，推荐最终图表如下：

| 图号 | 文件 | 内容 | 替代原图 |
|---|---|---|---|
| 图 1 | `fig1_thread_mapping.pdf` | Target-block 线程映射架构 | 原图 1 |
| 图 2 | `fig2_time_breakdown.pdf` | 时间组成 + 非 kernel 放大 | 原图 3 |
| 图 3 | `fig3_static_performance.pdf` | 吞吐 + Strict SR 默认配置对比 | 原图 5、6 |
| 图 4 | `fig4_pareto_front.pdf` | 吞吐—成功率—p95 Pareto 图 | 原图 7 |
| 图 5 | `fig5_threshold_scan.pdf` | 阈值扫描 | 原图 8 |
| 图 6 | `fig6_seed_success_heatmap.pdf` | Target × Seed 成功等级热力图 | 新增 / 增强图 9 |
| 图 7 | `fig7_trajectory_deltaq_heatmap.pdf` | 轨迹 Δq 热力图 | 原图 13 |
| 图 8 | `fig8_robustness_boundary.pdf` | 近奇异 + near-limit + barrier 综合图 | 原图 10、11、12 |

保留表格：

```text
表 5 静态批量 IK 综合性能
表 6 cuRobo-Graph 与 OPT4C 系统级对比
表 7 阈值扫描
表 8 CPU-LM 量级对照
表 9 K=16 vs K=1 消融
表 10 寄存器与溢出
表 11 混合精度
表 12 Nsight Compute
```

如果图太多，可删除原图 14 最大迭代次数扫描，改为表述或附录；也可以将最大迭代扫描保留为小图，但不是核心主图。

---

# 4. LaTeX 插入与编号调整要求

Codex 需要修改 `paper.tex`：

1. 替换旧图路径；
2. 删除被合并替代的旧图；
3. 更新 `\label{}`；
4. 更新正文中的图号引用；
5. 保证所有图片路径正确；
6. 确保图表顺序符合正文出现顺序。

建议 label：

```latex
\label{fig:thread_mapping}
\label{fig:time_breakdown}
\label{fig:static_performance}
\label{fig:pareto}
\label{fig:threshold_scan}
\label{fig:seed_success_heatmap}
\label{fig:trajectory_deltaq_heatmap}
\label{fig:robustness_boundary}
```

---

# 5. README 与 CHANGELOG 要求

更新 README：

```text
如何重新生成全部图：
python scripts/plot_all_figures.py
```

说明数据来源：

```text
results/*.csv
```

说明无法复现项：

```text
Nsight timeline 如需重新生成，需要本机安装 Nsight Systems。
```

新增或更新 `CHANGELOG.md`：

```text
- 替换了哪些图；
- 合并了哪些旧图；
- 新增了哪些热力图；
- 哪些图来自 CSV；
- 哪些图来自 Nsight 输出；
- paper.tex 中修改了哪些 figure 环境；
- latexmk 是否编译通过。
```

---

# 6. 最低验收标准

完成后必须满足：

- [ ] `python scripts/plot_all_figures.py` 可运行；
- [ ] 至少生成 8 张新版 PDF 图；
- [ ] `paper.tex` 引用新版图；
- [ ] LaTeX 编译无缺图；
- [ ] 图 5/6 不再含混使用 `cuRobo-Graph`；
- [ ] 新增 seed heatmap 或 candidate matrix 图；
- [ ] 新增 trajectory heatmap；
- [ ] 图 7 Pareto 图视觉质量明显提升；
- [ ] 所有图风格统一；
- [ ] 图中数据与表格一致；
- [ ] 不伪造不存在的数据。

---

# 7. Codex 最终交付物

Codex 完成后应交付：

```text
scripts/plot_all_figures.py
figures/*.pdf
figures_preview/*.png
results/*.csv（如有新增）
paper.tex
paper.pdf
README.md
CHANGELOG.md
```

并在最终回复中列出：

```text
1. 新绘制了哪些图；
2. 替换了哪些旧图；
3. 哪些图使用了真实 CSV；
4. 哪些数据文件是新增生成的；
5. 是否成功编译论文；
6. 仍需人工检查的地方。
```

---

## 8. 推荐给 Codex 的执行顺序

1. 扫描当前 `paper.tex` 中所有 `figure` 环境；
2. 建立旧图号到新图号的替换映射；
3. 检查 `results/` 是否已有所需 CSV；
4. 若缺少 matrix 数据，先从 runner 输出生成：
   - `candidate_success_matrix.csv`
   - `trajectory_deltaq_matrix.csv`
5. 编写 `scripts/plot_all_figures.py`；
6. 生成 8 张新版图；
7. 修改 `paper.tex`；
8. 编译 PDF；
9. 检查图号、引用和版面；
10. 写 `CHANGELOG.md`。
