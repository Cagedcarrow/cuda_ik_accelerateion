# 论文图件与编译说明

## 当前图件口径

正文当前使用 1 张流程图和 8 张数据图。流程图不计入数据图数量：

- 流程图：`figures/fig1_thread_mapping.svg`
- 数据图 1：`figures/fig2_time_breakdown.svg`
- 数据图 2：`figures/fig3_static_performance.svg`
- 数据图 3：`figures/fig4_pareto_front.svg`
- 数据图 4：`figures/fig5_threshold_scan.svg`
- 数据图 5：`figures/fig6_seed_success_heatmap.svg`
- 数据图 6：`figures/fig7_trajectory_deltaq_heatmap.svg`
- 数据图 7：`figures/fig8_robustness_boundary.svg`
- 数据图 8：`figures/fig9_iteration_tradeoff.svg`

旧的 `fig_kernel_time_breakdown.pdf` 和 `fig_nsys_timeline_opt4c.pdf` 不再被正文引用。当前 `paper.tex` 通过 `\includesvg` 直接引用 SVG 源图。

## 重新生成

在仓库根目录运行：

```bash
python3 standard_robot_cuda_ik/scripts/plot_all_figures.py
```

脚本输出：

- 论文 SVG 图：`standard_robot_cuda_ik/论文/figures/*.svg`
- PNG 预览：`standard_robot_cuda_ik/论文/figures_preview/*.png`
- 派生矩阵数据：`standard_robot_cuda_ik/data/experiments/补充实验/results/candidate_success_matrix.csv`
- 派生矩阵数据：`standard_robot_cuda_ik/data/experiments/补充实验/results/trajectory_deltaq_matrix.csv`

## 编译

编译依赖 Inkscape，`.latexmkrc` 已为 XeLaTeX 加入 `-shell-escape`。在 `standard_robot_cuda_ik/论文` 下运行：

```bash
latexmk -xelatex -interaction=nonstopmode paper.tex
```

当前验收结果：`paper.pdf` 已生成，13 页；日志未发现 fatal error、undefined reference、undefined citation 或 SVG conversion error。
