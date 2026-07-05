# 审稿材料说明

本文件夹汇总论文 PDF、最小 LaTeX 工程、论文图件、绘图脚本和绘图所用 CSV 数据，便于审稿人核查图表来源与复现路径。

## 论文意义

论文《面向六自由度机械臂批量逆运动学的结构感知 CUDA 核函数融合方法》面向固定 6 自由度机械臂的中小批量逆运动学（IK）求解场景。研究重点不是提出通用 IK 优化器，而是将 UR10 机械臂的固定运动学结构、解析雅可比、Sobol 多起点和 Levenberg-Marquardt 迭代融合到单个 CUDA kernel 中，以减少多阶段 GPU pipeline 的固定调度开销，并在规划前端所需的中等精度阈值下获得更稳定的一次性求解吞吐。

论文主要回答三个问题：

- 固定结构的 6x6 小矩阵 IK 是否适合用 target-block 核函数融合实现；
- 与 cuRobo-Graph K=1/K=16 等配置相比，吞吐、Strict 成功率和误差尾部分布如何权衡；
- 在近奇异、近限位和连续轨迹目标上，该方法的适用边界在哪里。

## 目录结构

```text
论文压缩包/
├── README.md
├── CHANGELOG.md
├── 论文/
│   └── paper.pdf
├── LaTeX工程/
│   ├── paper.tex
│   ├── .latexmkrc
│   ├── README_LaTeX.md
│   ├── CHANGELOG.md
│   └── figures/
│       └── fig*.pdf
├── 图片/
│   ├── PDF图件/
│   │   └── fig*.pdf
│   └── PNG预览/
│       └── fig*.png
├── 脚本/
│   └── plot_all_figures.py
└── 数据/
    └── CSV/
        └── *.csv
```

- `论文/paper.pdf`：最终编译后的论文 PDF。
- `LaTeX工程/`：最小可编译 LaTeX 工程，正文通过 `\includegraphics` 引用 `figures/*.pdf`。
- `图片/PDF图件/`：正文使用的 7 张主图副本。
- `图片/PNG预览/`：与 PDF 同名的快速预览图。
- `脚本/plot_all_figures.py`：生成全部主图的统一 Python 脚本，已改为相对路径。
- `数据/CSV/`：绘图脚本读取或生成的 CSV 数据文件。
- `CHANGELOG.md`：本轮标题、章节标题、图件 PDF 化和正文图压缩记录。

## 图件与 CSV 对应关系

| 论文图件 | 图件含义 | 直接数据来源 |
|---|---|---|
| `fig1_thread_mapping.pdf` | Target-Block 核函数映射架构图 | 无 CSV，脚本直接绘制结构示意 |
| `fig2_time_breakdown.pdf` | Core kernel 与 non-kernel overhead 时间组成 | `kernel_time_breakdown.csv` |
| `fig3_static_performance.pdf` | OPT4C-K16 与 cuRobo-Graph-K16 同种子数公平对比 | `fair_curobo_k16_summary.csv` |
| `fig4_pareto_front.pdf` | N=1000 吞吐率、成功率和 p95 误差 Pareto 关系 | `fair_curobo_k16_summary.csv` |
| `fig5_seed_success_heatmap.pdf` | Target-Seed 候选成功等级热图 | `trajectory_dump_candidates_random_local_50_N1000_K16.csv`，派生 `candidate_success_matrix.csv` |
| `fig6_robustness_boundary.pdf` | 近奇异成功率与 near-limit ratio 边界 | `near_singular_summary.csv`、`near_limit_barrier_summary.csv` |
| `fig7_trajectory_deltaq_heatmap.pdf` | 轨迹相邻点 wrapped joint delta 热图 | `trajectory_dump_candidates_random_local_50_N1000_K16.csv`、`trajectory_dump_best_random_local_50_N1000_K16.csv`，派生 `trajectory_deltaq_matrix.csv` |

## CSV 文件说明

- `fair_curobo_k16_summary.csv`：OPT4C K=1/K=16 与 cuRobo-Graph K=1/K=16 的系统级对比。
- `kernel_time_breakdown.csv`：core kernel 与非 kernel 开销的时间组成来源。
- `threshold_scan.csv`：四级阈值成功率表格来源。
- `near_singular_summary.csv`：wrist、elbow、shoulder 近奇异目标结果。
- `near_limit_barrier_summary.csv`：近限位目标下 Barrier ON/OFF 对比。
- `barrier_weight_scan.csv`：关节限位权重扫描，用于正文说明。
- `lm_iter_scan.csv`：最大迭代次数扫描，用于正文说明。
- `trajectory_dump_candidates_random_local_50_N1000_K16.csv`：轨迹目标的逐 seed 候选解与误差。
- `trajectory_dump_best_random_local_50_N1000_K16.csv`：轨迹目标的原始 best 候选解。
- `candidate_success_matrix.csv`：由绘图脚本从逐 seed 候选解中派生，用于图 5。
- `trajectory_deltaq_matrix.csv`：由绘图脚本重算 wrapped joint delta 后派生，用于图 7。

## 复现绘图

绘图脚本以本文件夹为根目录读取 `数据/CSV/`，并输出到 `LaTeX工程/figures/` 和 `图片/PNG预览/`。解压后可在任意位置运行：

```bash
python3 论文压缩包/脚本/plot_all_figures.py
```

脚本依赖 Python 3、NumPy、Pandas 和 Matplotlib。

## 编译 LaTeX 工程

当前论文正文使用 PDF 图件，不依赖 Inkscape 或 shell-escape。进入 `LaTeX工程/` 后运行：

```bash
latexmk -xelatex -interaction=nonstopmode paper.tex
```

## 注意事项

- 正文主图为 7 张 PDF 图。
- PNG 文件仅作预览；论文正文使用 PDF 图件。
- 阈值扫描和最大迭代次数扫描结果不再作为正文主图展示，相关结论保留在表格或正文说明中。
