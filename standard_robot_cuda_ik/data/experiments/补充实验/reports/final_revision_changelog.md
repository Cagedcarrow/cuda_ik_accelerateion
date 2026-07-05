# CUDA IK 终稿修订记录

日期：2026-07-05

## 总体说明

本轮修订面向 `standard_robot_cuda_ik/论文/计划/cuda_ik_goal_mode_final_revision_acceptance.md` 的 P0 清单，并额外按 `history/journal_references/参考论文/基于CUDA纹理插值的空间多目标交会序列快速规划方法_刘浩 (1).pdf` 对中文工程期刊写作风格进行统一。

## P0 对照

| 项 | 处理结果 | 证据 |
|---|---|---|
| P0-1 标题 | 已改为结构感知单核融合标题，并同步英文标题 | `论文/paper.tex` 标题区 |
| P0-2 摘要 | 已加入吞吐范围、Strict SR、N=1000 cuRobo-K16 对比和 Pareto 定位 | `论文/paper.tex` 摘要 |
| P0-3 cuRobo 协议 | 已说明 cuRobo-Graph-K1/K16、collision off、CUDA Graph on、计时边界和统一 FK 复核 | `论文/paper.tex` 第 3 节 |
| P0-4 cuRobo 表 | 已扩展 N=100/500/1000，含 K、Strict SR、吞吐、全样本/成功样本 p95、有效吞吐 | `论文/paper.tex` 表 `tab:curobo` |
| P0-5 轨迹连续性 | 已将 delta-q 改为 `atan2(sin dq, cos dq)`，重算 CSV 和图 | `results/trajectory_continuity_summary.csv`, `figures/fig_trajectory_delta_q.pdf` |
| P0-6 Barrier | 已降级为安全裕度正则项，说明不是成功率来源 | `论文/paper.tex` 近限位段 |
| P0-7 Nsight | 已降调为长延迟依赖，避免单一 FP64 管线直接证明 | `论文/paper.tex` Nsight Compute 段 |
| P0-8 精度宣称 | 已删除 `O(10^-6)->O(10^-16)` 结论 | `论文/paper.tex` 结论 |
| P0-9 阈值扫描 | 已补 N=1000 四方法四阈值表/图，cuRobo 数据来自逐目标 rows | `results/threshold_scan.csv`, `figures/fig_threshold_scan.pdf` |
| P0-10 关键图 | 已重画线程映射、时间组成、Pareto、阈值图并同步到论文绘图目录 | `figures/*.pdf`, `论文/绘图/*.pdf` |
| P0-11 CPU baseline | 已统一为 Python/NumPy CPU-LM 量级对照 | `论文/paper.tex` CPU 小节 |
| P0-12 特殊目标规则 | 已写明 seed=42、FK 可达、近奇异/近限位生成规则 | `论文/paper.tex` 鲁棒性小节 |
| P0-13 结论边界 | 已重写贡献与边界：固定 UR10/6DOF、无碰撞、FK 可达、轨迹连续性不足、cuRobo-K16 精度更高 | `论文/paper.tex` 讨论与结论 |

## 风格对齐

- 摘要改为“针对……问题，提出……方法。该方法首先……然后……最终……。实验结果表明……”结构。
- 引言保留“应用背景、计算难点、现有方法局限、CUDA/GPU 价值、本文方法定位”的顺序。
- 实验段落统一为“设置、指标、表格、对比、解释”的工程期刊叙述。
- 结论语气改为克制的结果归纳和边界声明，不再写最高精度或全面优越性表述。

## 重新生成的关键产物

- `standard_robot_cuda_ik/data/experiments/补充实验/results/curobo_threshold_N1000_K1.rows.csv`
- `standard_robot_cuda_ik/data/experiments/补充实验/results/curobo_threshold_N1000_K16.rows.csv`
- `standard_robot_cuda_ik/data/experiments/补充实验/results/threshold_scan.csv`
- `standard_robot_cuda_ik/data/experiments/补充实验/results/trajectory_continuity_summary.csv`
- `standard_robot_cuda_ik/data/experiments/补充实验/figures/fig_thread_mapping_redraw.pdf`
- `standard_robot_cuda_ik/data/experiments/补充实验/figures/fig_kernel_time_breakdown.pdf`
- `standard_robot_cuda_ik/data/experiments/补充实验/figures/fig_pareto_throughput_success.pdf`
- `standard_robot_cuda_ik/data/experiments/补充实验/figures/fig_threshold_scan.pdf`
