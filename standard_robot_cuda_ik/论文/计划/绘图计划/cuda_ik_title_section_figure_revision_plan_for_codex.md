# CUDA IK 论文标题、章节标题与绘图重构修改方案（给 Codex）

> 适用工程：最新版 LaTeX 工程与绘图脚本。  
> 目标：在不改变论文核心实验结论的前提下，完成标题学术化、章节标题统一化、实验图信息压缩、图件 PDF 化与 LaTeX 引用更新。  
> 原则：论文正文已经基本稳定，本轮重点是**图表表达质量、图件工程稳定性和章节标题学术规范性**。

---

## 0. 当前判断

当前论文主体已经基本完成，不建议继续大规模补实验。主要问题集中在：

1. 标题中“单核融合”存在歧义，不够学术；
2. 部分章节标题中英文术语混用，表述偏工程说明；
3. 图表数量偏多，部分图信息密度高但结论不突出；
4. SVG 插图存在字体/编码/编译兼容风险；
5. 部分实验图和表格重复，应该删图或转为文字说明；
6. 目前应从“展示所有数据”转向“每张图只证明一个结论”。

本轮修改目标：

```text
标题学术化
章节标题统一化
实验图信息压缩
SVG 改 PDF
删除低价值图
保留强叙事图
LaTeX 编译无乱码、无缺图、无未定义引用
```

---

# 1. 标题修改方案

## 1.1 当前标题

```text
结构感知单核融合的机械臂批量逆运动学 CUDA 加速方法
```

## 1.2 当前标题问题

1. “单核融合”容易被误解为 CPU 单核或单个物理核心，而不是 CUDA kernel fusion；
2. “CUDA 加速方法”偏工程报告表达；
3. 标题没有明确六自由度对象边界；
4. 与正文中“固定 6-DOF、中小批量、核函数融合”的定位不完全一致。

## 1.3 推荐标题

### 中文标题

```text
面向六自由度机械臂批量逆运动学的结构感知 CUDA 核函数融合方法
```

### 英文标题

```text
Structure-Aware CUDA Kernel Fusion for Batch Inverse Kinematics of Six-DOF Manipulators
```

## 1.4 修改要求

在 `paper.tex` 中修改：

```latex
\title{...}
\etitle{...}
```

或模板中对应的中英文标题字段。

## 1.5 验收标准

- [ ] 中文标题改为推荐标题；
- [ ] 英文标题改为推荐标题；
- [ ] 全文不再使用“单核融合”作为标题关键词；
- [ ] 正文第一次出现 CUDA kernel fusion 时可写成“CUDA 核函数融合（CUDA kernel fusion）”；
- [ ] 后文统一使用“核函数融合”或“单核函数融合”，避免“单核融合”。

---

# 2. 章节标题修改方案

## 2.1 总体章节标题

| 当前标题 | 修改为 | 修改理由 |
|---|---|---|
| 引 言 | 引言 | 去掉中间空格，更规范 |
| 问题定义与评价协议 | 问题形式化与评价指标 | 更学术，和第 1 章内容一致 |
| 运动学结构驱动的 CUDA 小矩阵加速方法 | 结构感知 CUDA 核函数融合方法 | 与论文标题和核心贡献一致 |
| 实验设置 | 实验设置与复评协议 | 突出外部 FK 复评、公平计时协议 |
| 实验结果与分析 | 实验结果 | 更简洁，分析在正文中自然展开 |
| 讨论 | 讨论 | 保持 |
| 结论 | 结论 | 保持 |

---

## 2.2 方法章节小标题

| 当前标题 | 修改为 |
|---|---|
| 运动学模型至 GPU 常量内存的编译期编码 | 运动学参数的编译期常量编码 |
| 融合 FK 函数与解析雅可比组装 | 融合正运动学与解析雅可比构造 |
| 低控制流复杂度 LM 迭代算法 | 低控制流复杂度 LM 迭代 |
| Target-Block 融合映射：单 Kernel 端到端执行 | Target-Block 映射与单核函数端到端融合 |
| Kernel Fusion 的调度开销分析 | 核函数融合的调度开销分析 |

---

## 2.3 实验章节小标题

| 当前标题 | 修改为 | 说明 |
|---|---|---|
| 静态批量 IK 综合性能 | 静态可达目标的批量求解性能 | 更明确 |
| 与 cuRobo 的系统级对比与公平性分析 | 与 cuRobo 的同种子数公平对比 | 更聚焦 K=16 公平对比 |
| Python/NumPy CPU-LM 量级对照 | CPU-LM 量级基线对照 | 简洁 |
| 种子数消融实验：多起点策略的影响 | 多起点策略消融 | 更像论文标题 |
| 近奇异与近限位鲁棒性实验 | 近奇异与近限位场景分析 | “鲁棒性”略大，建议降级 |
| 轨迹连续性实验 | 连续轨迹上的分支跳变分析 | 与实验结论一致 |
| 最大迭代次数扫描 | 最大迭代次数敏感性分析 | 如果保留 |
| 微架构瓶颈的 PTX 级定量分析 | PTX 与微架构瓶颈分析 | 更简洁 |
| 批量扩展性分析 | 批量规模扩展性分析 | 保持或并入 4.1 |

---

## 2.4 特别注意

“轨迹连续性实验”建议必须改为：

```text
连续轨迹上的分支跳变分析
```

原因：当前实验结论不是证明方法能保证连续轨迹，而是证明 smoothness rerank 只能降低跳变幅值，仍不能保证关节连续性。标题必须与实际结论一致。

## 2.5 验收标准

- [ ] 所有章节标题按表格修改；
- [ ] 术语“Kernel Fusion / 单 Kernel / 单核融合”统一；
- [ ] 标题表达不夸大结论；
- [ ] 修改后目录层级和编号正常；
- [ ] 正文引用章节编号无错误。

---

# 3. 绘图工程修改方案

## 3.1 当前问题

当前工程使用：

```latex
\includesvg{...}
```

并且绘图脚本主要输出：

```text
图片/SVG源图/*.svg
图片/PNG预览/*.png
```

存在以下风险：

1. 依赖 Inkscape；
2. SVG 文本在 LaTeX 编译中可能出现字体缺失；
3. 中文、下标、特殊字符容易出现乱码；
4. 投稿系统可能不支持 SVG；
5. 当前 PDF 中已出现类似 `N=���`、`OPT�C-K��` 的风险迹象。

## 3.2 修改目标

改为：

```text
figures/*.pdf
figures_preview/*.png
```

LaTeX 使用：

```latex
\includegraphics[width=...]{figures/fig_xxx.pdf}
```

不再使用：

```latex
\includesvg{...}
```

## 3.3 绘图脚本修改要求

修改 `plot_all_figures.py`：

1. 同时输出 PDF 和 PNG；
2. PDF 用于论文；
3. PNG 用于预览；
4. 自动创建输出目录；
5. 图内文字尽量使用英文 ASCII；
6. 中文说明放入 LaTeX caption；
7. 设置 PDF 字体嵌入；
8. 保证脚本可在压缩包根目录直接运行。

建议输出目录：

```text
LaTeX工程/figures/
图片/PNG预览/
```

或：

```text
figures/
figures_preview/
```

二选一，但 `paper.tex` 和脚本必须一致。

推荐保存函数：

```python
def save_figure(fig, name):
    pdf_dir = ROOT / "LaTeX工程" / "figures"
    png_dir = ROOT / "图片" / "PNG预览"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(png_dir / f"{name}.png", dpi=300, bbox_inches="tight")
```

推荐 rcParams：

```python
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
```

## 3.4 LaTeX 修改要求

将所有：

```latex
\includesvg[width=...]{图片/SVG源图/xxx}
```

改为：

```latex
\includegraphics[width=...]{figures/xxx.pdf}
```

并确认导言区包含：

```latex
\usepackage{graphicx}
```

如果不再使用 SVG，可以删除或保留但不再依赖：

```latex
\usepackage{svg}
```

## 3.5 验收标准

- [ ] `python 脚本/plot_all_figures.py` 可运行；
- [ ] 生成 PDF 图；
- [ ] 生成 PNG 预览图；
- [ ] `paper.tex` 不再使用 `\includesvg`；
- [ ] `paper.tex` 使用 `\includegraphics` 引用 PDF；
- [ ] 编译后无图中文字乱码；
- [ ] 编译不依赖 Inkscape；
- [ ] 无缺图、无 `??`、无未定义引用。

---

# 4. 最终推荐图结构

当前论文主图建议压缩为 7 张：

| 新图号 | 内容 | 处理 |
|---|---|---|
| 图 1 | Target-Block 架构图 | 保留，重画 |
| 图 2 | Kernel vs Non-kernel 时间组成 | 简化 |
| 图 3 | OPT4C-K16 vs cuRobo-K16 批量趋势 | 保留，但只画公平对比 |
| 图 4 | Pareto 四点图 | 保留，删除 colorbar |
| 图 5 | Target-Seed 成功等级热图 | 保留，降饱和 |
| 图 6 | Near singular / near limit 边界图 | 简化为 2-panel |
| 图 7 | Trajectory Δq 热图 | 保留，降噪 |

建议删除正文图：

```text
阈值扫描图
最大迭代次数图
wlimit sweep 子图
```

相关结果用表格或正文说明。

---

# 5. 逐图修改方案

---

## 图 1：Target-Block 架构图

### 当前问题

图 1 内容正确，但视觉仍接近 PPT 流程图。颜色、箭头和模块层级较杂。作为论文核心方法图，应更简洁、更期刊化。

### 修改要求

重画为灰阶或低饱和 IEEE 风格图。图内文字尽量使用英文 ASCII。

图内建议元素：

```text
Target T_i
Block i
Lane 0--15: Sobol seeds
Lane 16--31: inactive
Fused FK + Jacobian + LM
Shared candidate buffer
Lane-0 selection
Best output q*
```

### 推荐 caption

```latex
\caption{Target-Block 线程映射架构。一个线程块处理一个目标位姿，lane 0--15 分别处理 16 个 Sobol 起点，lane 16--31 为空闲线程；候选解暂存于共享内存，并由 lane 0 完成块内最佳候选选择后写入全局输出。}
```

### Codex 指令

```text
重画 fig1_thread_mapping。采用 IEEE 风格灰阶矢量图，减少彩色块和装饰性箭头。图内文字全部使用英文 ASCII，中文解释放入 caption。最终输出 PDF。
```

### 验收标准

- [ ] 不再像 PPT 流程图；
- [ ] 图内结构层级清楚；
- [ ] lane、block、target、shared buffer、output 的关系明确；
- [ ] 图中文字无乱码；
- [ ] 可单栏或双栏清晰显示。

---

## 图 2：时间组成图

### 当前问题

当前图 2 分为总时间占比和非 kernel 放大，但左图几乎全是 Kernel，信息价值有限。

### 修改要求

改成极简横向堆叠条形图，只区分：

```text
Core kernel
Non-kernel overhead
```

不要再拆：

```text
H2D
D2H
Launch
```

这些细分移入正文说明即可。

### 推荐图形

```text
N=100    [Core kernel | Non-kernel]
N=500    [Core kernel | Non-kernel]
N=1000   [Core kernel | Non-kernel]
```

右侧标注：

```text
Non-kernel < 0.3%
```

### Codex 指令

```text
将 fig2_time_breakdown 改成极简 horizontal stacked bar，只区分 Core kernel 与 Non-kernel overhead。右侧直接标注 Non-kernel < 0.3%。H2D/D2H/Launch 的细分移入正文，不在图中单独显示。
```

### 验收标准

- [ ] 图只回答“瓶颈是否已转移到 kernel 内部”；
- [ ] 不再展示过小的 H2D/D2H/Launch 细分；
- [ ] 图面简洁；
- [ ] 单位和注释明确。

---

## 图 3：批量吞吐率与 Strict 成功率对比

### 当前问题

当前图 3 同时包含：

```text
OPT4C-K16
cuRobo-Graph-K1
cuRobo-Graph-K16
```

这混合了默认配置对比和同种子数公平对比。

### 修改要求

图 3 只保留公平对比：

```text
OPT4C-K16
cuRobo-Graph-K16
```

cuRobo-K1 只保留在表 6 和 Pareto 图中。

### 图形结构

双子图：

```text
(a) Throughput vs N
(b) Strict SR vs N
```

### Codex 指令

```text
fig3_static_performance 只保留 OPT4C-K16 与 cuRobo-Graph-K16。左图为 throughput vs N，右图为 Strict SR vs N。cuRobo-Graph-K1 从该图移除，保留在表 6 和 Pareto 图中。
```

### 验收标准

- [ ] 图例只有 OPT4C-K16 与 cuRobo-Graph-K16；
- [ ] 图注说明这是同种子数 K=16 公平对比；
- [ ] 表 6 仍保留 K1/K16 完整数值；
- [ ] 正文叙事不再混淆默认配置与公平配置。

---

## 图 4：Pareto 四点图

### 当前问题

当前 Pareto 图使用颜色条表示 p95。由于只有 4 个点，colorbar 有些过度，反而增加视觉复杂度。

### 修改要求

改为四点 Pareto 图：

- x-axis：Throughput / 10^4 targets s^-1
- y-axis：Strict SR / %
- marker：方法
- 点旁文本标注：

```text
p95 = 4.56 mm
p95 = 0.88 mm
```

删除 colorbar。

添加淡灰色方向箭头：

```text
higher throughput →
higher success ↑
```

### Codex 指令

```text
重画 fig4_pareto_front。删除 colorbar，改为四个方法点 + 旁注 p95_all。保留 marker 区分方法，不使用过度鲜艳配色。添加淡灰色方向箭头说明 throughput-success trade-off。
```

### 验收标准

- [ ] 四个方法点完整；
- [ ] 不使用 colorbar；
- [ ] 每个点标注 p95_all；
- [ ] 图能直接表达 Pareto 取舍；
- [ ] 数据与表 6 一致。

---

## 图 5：Target-Seed 成功等级热图

### 当前评价

这是当前最有价值的实验图之一，应保留。

### 当前问题

配色略强，绿色面积大，视觉刺激；排序规则需要更明确。

### 修改要求

使用低饱和离散色图：

```text
Fail   : light gray
Loose  : pale yellow
Medium : light blue
Strict : deep blue or deep green
```

排序规则：

```text
按每个 target 的 Strict seed 数量升序排序；
若相同，再按 best_pose_cost 排序。
```

colorbar 标签必须是：

```text
Fail / Loose / Medium / Strict
```

### Codex 指令

```text
保留 fig6_seed_success_heatmap。改用低饱和离散色图。排序规则改为 strict_seed_count 升序，若相同按 best_pose_cost 排序。colorbar 标签必须为 Fail / Loose / Medium / Strict。
```

### 验收标准

- [ ] 热图配色低饱和；
- [ ] colorbar 是离散等级；
- [ ] 排序规则在 caption 或正文中说明；
- [ ] 能清楚表达多起点候选覆盖；
- [ ] 不使用彩虹色或过度鲜艳配色。

---

## 图 6：Near singular / near limit 边界图

### 当前问题

当前鲁棒性图有 4 个 panel：

```text
(a) 近奇异目标
(b) 近限位目标
(c) 限位权重扫描
(d) 随机目标权重影响
```

信息太多，而且后两个 panel 不是主贡献。

### 修改要求

简化为 2-panel：

```text
(a) Near singular success rate
(b) Near-limit Barrier ON/OFF
```

Panel (a)：

```text
wrist / elbow / shoulder
K1 vs K16
Strict SR
```

Panel (b)：

```text
Barrier ON/OFF
near-limit ratio
```

Strict SR 的 ON/OFF 数值放正文，不必画图。

删除：

```text
wlimit sweep
random target weight effect
```

相关结论用正文一句话：

```text
权重扫描表明提高 wlimit 不会单调提升 Strict SR，默认 wlimit=0.03 是保守折中。
```

### Codex 指令

```text
将 fig8_robustness_boundary 简化为 2-panel。Panel (a) 展示 wrist/elbow/shoulder 下 K1 与 K16 的 Strict SR。Panel (b) 展示 Barrier ON/OFF 下 near-limit ratio，Strict SR 只在正文报告。删除 wlimit sweep 子图。
```

### 验收标准

- [ ] 只保留 2 个 panel；
- [ ] 图面不拥挤；
- [ ] barrier 不被夸大为成功率提升手段；
- [ ] wlimit sweep 结论转为正文；
- [ ] 图注与正文一致。

---

## 图 7：Trajectory Δq 热图

### 当前评价

应保留。它是轨迹章节最有价值的证据图。

### 当前问题

当前色图偏火焰图，视觉刺激较强。需要降噪。

### 修改要求

使用低饱和单色图：

```text
white → light blue → dark blue
```

统一色标：

```text
vmin = 0
vmax = 4 rad
```

两个 panel：

```text
No rerank
Smoothness rerank
```

每个 panel 内标注：

```text
p95 = xx rad
jump ratio = xx%
```

### Codex 指令

```text
重绘 fig7_trajectory_deltaq_heatmap。使用 Blues 或 white-blue 自定义 cmap，统一 vmin=0, vmax=4 rad。保留 no rerank 与 smoothness rerank 两个 panel。每个 panel 标注 p95 和 jump ratio。避免 inferno/magma/plasma 等高饱和配色。
```

### 验收标准

- [ ] 使用单色低饱和 cmap；
- [ ] 两个 panel 共享同一色标；
- [ ] 标注 p95 和 jump ratio；
- [ ] 图注说明 delta q 使用 wrap 差分；
- [ ] 清楚表达 rerank 只能缓解跳变，不能根治连续性问题。

---

# 6. 删除或转为表格/正文的图

## 6.1 删除阈值扫描图

### 原因

表 7 已经完整展示 Loose / Medium / Strict / Ultra 成功率。图 5 与表 7 重复，边际信息价值较低。

### Codex 指令

```text
删除 fig5_threshold_scan 的正文 figure 环境。保留表 7，并在正文中用表 7 解释 Strict 与 Ultra 下的差异。若需要保留图件，只作为附录图，不进入正文主图。
```

---

## 6.2 删除最大迭代次数图

### 原因

最大迭代次数扫描只是参数选择说明，不是主贡献。

### Codex 指令

```text
删除 fig9_iteration_tradeoff 的正文 figure 环境。将最大迭代次数扫描结果改为正文描述或小表格：max_iter=20 时 SR 较低，增至 60 后收益明显，继续增至 80/100 收益有限，因此采用 60。
```

---

## 6.3 删除 wlimit sweep 子图

### 原因

wlimit sweep 的结论是“不单调、收益有限”。这不是主贡献，不应占主图空间。

### Codex 指令

```text
从 robustness 图中删除 wlimit sweep 和 random target weight effect 两个 panel。相关结论保留为正文一句话。
```

---

# 7. LaTeX figure 环境更新要求

## 7.1 推荐 label

```latex
\label{fig:thread_mapping}
\label{fig:time_breakdown}
\label{fig:static_performance}
\label{fig:pareto_front}
\label{fig:seed_success_heatmap}
\label{fig:robustness_boundary}
\label{fig:trajectory_deltaq_heatmap}
```

## 7.2 删除或不再正文引用的 label

```latex
\label{fig:threshold_scan}
\label{fig:iteration_tradeoff}
```

如果正文中仍引用，应删除或改为“表 7”。

## 7.3 必须检查

- [ ] 图号连续；
- [ ] 正文引用全部有效；
- [ ] 不再引用已删除图；
- [ ] 所有 caption 与新图内容一致；
- [ ] 所有图路径正确；
- [ ] 编译无 `??`。

---

# 8. 给 Codex 的完整执行清单

请 Codex 按顺序执行：

```text
1. 打开 paper.tex，记录所有 figure 环境和 \ref 引用。
2. 修改论文中英文标题。
3. 修改章节标题。
4. 修改 plot_all_figures.py：
   - 同时输出 PDF 和 PNG；
   - PDF 放入 LaTeX工程/figures/；
   - PNG 放入 图片/PNG预览/；
   - 图内文字使用英文 ASCII；
   - 修正所有图的配色和内容。
5. 删除或停用 threshold scan 图。
6. 删除或停用 iteration tradeoff 图。
7. 重画以下图：
   - fig1_thread_mapping
   - fig2_time_breakdown
   - fig3_static_performance
   - fig4_pareto_front
   - fig6_seed_success_heatmap
   - fig8_robustness_boundary
   - fig7_trajectory_deltaq_heatmap
8. 修改 paper.tex：
   - \includesvg 改 \includegraphics；
   - 引用 PDF 图；
   - 更新 caption；
   - 删除低价值 figure 环境；
   - 更新正文图号引用。
9. 编译 paper.tex。
10. 检查 paper.pdf：
    - 无乱码；
    - 无缺图；
    - 无未定义引用；
    - 图号连续；
    - 图表与正文一致。
11. 写 CHANGELOG.md。
```

---

# 9. CHANGELOG 要求

新增或更新 `CHANGELOG.md`：

```text
# 图表与标题重构记录

## 标题修改
- 中文标题：
- 英文标题：

## 章节标题修改
- 列出改动前后标题。

## 图表修改
- 删除了哪些图；
- 重画了哪些图；
- 哪些图改为表格/正文；
- 每张图的数据来源 CSV；
- 每张图输出路径。

## LaTeX 修改
- \includesvg -> \includegraphics
- 图路径修改
- label 修改
- caption 修改

## 编译结果
- 编译命令：
- 是否成功：
- 是否存在 warning：

## 仍需人工检查
- ...
```

---

# 10. 最低验收标准

完成后必须满足：

- [ ] 标题已替换为推荐标题；
- [ ] 章节标题已统一；
- [ ] 正文主图压缩到约 7 张；
- [ ] 删除阈值扫描图；
- [ ] 删除最大迭代次数图；
- [ ] 删除 wlimit sweep 子图；
- [ ] 图 3 只保留 K=16 公平对比；
- [ ] Pareto 图删除 colorbar；
- [ ] Seed heatmap 降饱和并说明排序规则；
- [ ] Trajectory heatmap 使用低饱和单色图并标注 p95/jump ratio；
- [ ] `plot_all_figures.py` 一键生成 PDF；
- [ ] LaTeX 使用 `\includegraphics` 引用 PDF；
- [ ] 编译后无乱码、无缺图、无 `??`；
- [ ] `CHANGELOG.md` 已更新。

---

# 11. 最终预期效果

修改完成后，论文应达到：

```text
正文逻辑：基本稳定
章节标题：更学术、更一致
图表数量：减少但证据更集中
实验图：从“展示数据”转为“支撑结论”
图件工程：PDF化、可复现、无乱码风险
投稿观感：更接近期刊论文
```

最终主图结构应服务以下叙事：

```text
图1：方法如何映射到 CUDA
图2：瓶颈已转移到 kernel 内部
图3：同种子数下吞吐/成功率取舍
图4：OPT4C 与 cuRobo 的 Pareto 位置
图5：K=16 多起点为何有效
图6：近奇异/近限位边界在哪里
图7：轨迹跳变仍是局限性
```
