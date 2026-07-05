# 图表与标题重构记录

## 标题修改

- 中文标题：由“结构感知单核融合的机械臂批量逆运动学 CUDA 加速方法”改为“面向六自由度机械臂批量逆运动学的结构感知 CUDA 核函数融合方法”。
- 英文标题：由“Structure-Aware Single-Kernel Fusion CUDA Acceleration for Batch Inverse Kinematics of Robotic Manipulators”改为“Structure-Aware CUDA Kernel Fusion for Batch Inverse Kinematics of Six-DOF Manipulators”。
- 摘要中首次使用“结构感知 CUDA 核函数融合（CUDA kernel fusion）”，后文统一使用“核函数融合”或“单核函数”表述，避免将“单核融合”作为标题关键词。

## 章节标题修改

- `引 言` -> `引言`
- `问题定义与评价协议` -> `问题形式化与评价指标`
- `运动学结构驱动的 CUDA 小矩阵加速方法` -> `结构感知 CUDA 核函数融合方法`
- `实验设置` -> `实验设置与复评协议`
- `实验结果与分析` -> `实验结果`
- `讨 论` -> `讨论`
- `结 论` -> `结论`
- `运动学模型至 GPU 常量内存的编译期编码` -> `运动学参数的编译期常量编码`
- `融合 FK 函数与解析雅可比组装` -> `融合正运动学与解析雅可比构造`
- `低控制流复杂度 LM 迭代算法` -> `低控制流复杂度 LM 迭代`
- `Target-Block 融合映射：单 Kernel 端到端执行` -> `Target-Block 映射与单核函数端到端融合`
- `Kernel Fusion 的调度开销分析` -> `核函数融合的调度开销分析`
- `静态批量 IK 综合性能` -> `静态可达目标的批量求解性能`
- `与 cuRobo 的系统级对比与公平性分析` -> `与 cuRobo 的同种子数公平对比`
- `Python/NumPy CPU-LM 量级对照` -> `CPU-LM 量级基线对照`
- `种子数消融实验：多起点策略的影响` -> `多起点策略消融`
- `近奇异与近限位鲁棒性实验` -> `近奇异与近限位场景分析`
- `轨迹连续性实验` -> `连续轨迹上的分支跳变分析`
- `最大迭代次数扫描` -> `最大迭代次数敏感性分析`
- `微架构瓶颈的 PTX 级定量分析` -> `PTX 与微架构瓶颈分析`
- `批量扩展性分析` -> `批量规模扩展性分析`

## 图表修改

- 正文主图由 9 张压缩为 7 张。
- 删除正文阈值扫描图 `fig5_threshold_scan`；四级阈值结果保留在表格和正文分析中。
- 删除正文最大迭代次数图 `fig9_iteration_tradeoff`；最大迭代次数扫描改为正文说明。
- 从鲁棒性图中删除 `wlimit sweep` 和随机目标权重影响两个子图；权重扫描结论改为正文说明。
- 所有正文图由 SVG 改为 PDF，PNG 仅作为预览。

| 正文图件 | 修改内容 | 数据来源 |
|---|---|---|
| `fig1_thread_mapping.pdf` | 重画为低饱和灰阶 Target-Block 架构图，图内文字改为英文 ASCII | 无 CSV，脚本绘制结构示意 |
| `fig2_time_breakdown.pdf` | 改为横向堆叠条形图，仅区分 Core kernel 与 Non-kernel overhead | `kernel_time_breakdown.csv` |
| `fig3_static_performance.pdf` | 只保留 OPT4C-K16 与 cuRobo-Graph-K16 的同种子数公平对比 | `fair_curobo_k16_summary.csv` |
| `fig4_pareto_front.pdf` | 删除 colorbar，改为四点 Pareto 图并标注 `p95_all` | `fair_curobo_k16_summary.csv` |
| `fig5_seed_success_heatmap.pdf` | 使用低饱和离散色图，按 Strict seed 数量和 best pose cost 排序 | `trajectory_dump_candidates_random_local_50_N1000_K16.csv`，派生 `candidate_success_matrix.csv` |
| `fig6_robustness_boundary.pdf` | 简化为 Near singular success rate 与 Near-limit ratio 两个 panel | `near_singular_summary.csv`、`near_limit_barrier_summary.csv` |
| `fig7_trajectory_deltaq_heatmap.pdf` | 改为低饱和蓝色热图，共享 0--4 rad 色标，并标注 `p95` 和 `jump ratio` | `trajectory_dump_candidates_random_local_50_N1000_K16.csv`、`trajectory_dump_best_random_local_50_N1000_K16.csv`，派生 `trajectory_deltaq_matrix.csv` |

## LaTeX 修改

- 删除 `svg` 宏包、`\svgpath` 和 `\svgsetup`。
- 将所有 `\includesvg` 替换为 `\includegraphics`。
- `paper.tex` 当前只引用 `figures/*.pdf`。
- `.latexmkrc` 删除 `-shell-escape`，当前编译不依赖 Inkscape。
- figure label 更新为：
  - `fig:thread_mapping`
  - `fig:time_breakdown`
  - `fig:static_performance`
  - `fig:pareto_front`
  - `fig:seed_success_heatmap`
  - `fig:robustness_boundary`
  - `fig:trajectory_deltaq_heatmap`

## 脚本修改

- 更新 `scripts/plot_all_figures.py`。
- 源工程运行时读取 `data/experiments/补充实验/results/`，输出 `论文/figures/*.pdf` 和 `论文/figures_preview/*.png`。
- 复制到审稿压缩包后，脚本自动读取 `数据/CSV/`，输出 `LaTeX工程/figures/*.pdf` 和 `图片/PNG预览/*.png`。
- 主入口仅生成 7 张正文图，不再生成阈值扫描图和最大迭代次数图。

## 编译结果

- 绘图命令：`python3 standard_robot_cuda_ik/scripts/plot_all_figures.py`
- 编译命令：在 `standard_robot_cuda_ik/论文` 下运行 `latexmk -xelatex -interaction=nonstopmode paper.tex`
- 编译结果：成功，`paper.pdf` 为 12 页。
- 日志核查：未发现 fatal error、LaTeX Error、undefined reference/citation、缺图、SVG 转换错误或 Overfull。
- PDF 文本核查：新标题、章节标题和 7 张 PDF 图引用均已进入最终 PDF。

## 仍需人工检查

- 期刊模板如强制要求图内中文，可在保持 PDF 输出的前提下按编辑部要求替换图内英文标签。
- 当前图 4 的 p95 标注采用手动偏移；若后续数据更新，需要重新做视觉检查。
