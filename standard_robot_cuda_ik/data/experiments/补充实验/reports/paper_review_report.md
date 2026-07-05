# paper.tex 审稿式复核报告

复核对象：`standard_robot_cuda_ik/论文/paper.tex`

复核时间：2026-07-05

## 结论

当前稿件已经完成补充实验合并，并通过 XeLaTeX 编译生成 `paper.pdf`。正文的主要结论已从“绝对优于 cuRobo / 完全消除某类硬件问题”调整为“在 `N<=1000`、固定 6-DOF、无碰撞约束、FK 可达目标条件下，形成吞吐量、成功率和成功样本精度之间的 Pareto 取舍”。该表述与现有 CSV、Nsight Systems timeline 和图件证据一致。

## 新颖性与贡献边界

- 贡献点保留为结构感知单 kernel fusion、Sobol 多起点、解析雅可比与寄存器级小矩阵求解的组合实现。
- 已删除或弱化“硬件固化”“完全消除 warp divergence”“Bank 冲突消除”“工业抓取精度充足”等容易被审稿人质疑的强断言。
- 与 cuRobo 的关系已改成 Pareto 取舍：cuRobo K=16 精度更高，OPT4C K=16 吞吐更高，cuRobo K=1 质量不可用但吞吐极高。

## 方法学与实验设计

- P0 补充实验已完成：公平 cuRobo K=16 对比、kernel time breakdown、threshold scan、seed count scan、near singular、near limit Barrier ON/OFF、trajectory continuity。
- P1 扩展实验已加入：Barrier weight scan、LM max-iter scan、CPU-LM baseline、Nsight Systems timeline。
- 轨迹连续性实验已明确为候选级离线 rerank，不能等同于在线 warm-start 或连续性约束优化。
- near-limit 实验已明确 barrier 主要作为安全裕度约束，不是成功率来源。

## 结果解释

- 种子数扫描显示成功率主要来自多起点策略：`K=1` 到 `K=16` 时 Strict SR 从 0.522 提升至 0.954。
- 近奇异实验显示 elbow singular 是主要失败边界，`N=1000,K=16` Strict SR 降至 0.883。
- 近限位实验显示 Barrier ON/OFF 差异不支持“显著提高成功率”的结论。
- 最大迭代次数扫描支持 `max_iter=60` 作为成功率与运行时间的折中，但该扫描为独立批次，正文已注明其吞吐不反向覆盖主基准。

## 可复现性

- `data/experiments/补充实验/README.md` 已更新复现实验命令和产物清单。
- `reports/acceptance_report.md` 显示 P0 阶段验收通过。
- `paper.tex` 已引用同步到 `论文/绘图/` 的补充图件。
- `cpu_baseline_summary.csv` 为 Python/NumPy 单进程 CPU-LM 量级对照，不能解释为优化 C++ CPU solver 或 KDL/TRAC-IK 性能。
- `nsys_opt4c_n1000_summary.txt` 和 `fig_nsys_timeline_opt4c.pdf` 来自真实 Nsight Systems 采样；本机不允许 CPU context switch tracing，正文已限定只分析 CUDA kernel、memcpy 和 memset。

## 格式与编译

- 命令：`latexmk -xelatex -interaction=nonstopmode paper.tex`
- 输出：`standard_robot_cuda_ik/论文/paper.pdf`
- PDF：A4，12 页。
- 编译日志：无 LaTeX error、无 undefined reference、无 overfull hbox。
- 图题已清理为中文题注，不再保留英文 `Fig.` 副题。

## 仍需避免的写法

- 不能写成“全面验证”或“完全消除分支/冲突”。
- 不能把 5 mm 阈值泛化为精密装配或亚毫米任务可用。
- 不能把 Barrier 解释为成功率提升来源。
- 不能把 trajectory rerank 解释成在线轨迹连续 IK。
- 不能把 Python/NumPy CPU baseline 写成 KDL/TRAC-IK 或优化 C++ CPU solver。
- 不能把 Nsight Systems 采样解释为包含 CPU context switch tracing。

## 后续增强项

- 实现优化 C++ CPU solver 或接入 KDL/TRAC-IK，给出更接近工程 CPU 库的公平对比。
- 如需要更完整系统级分析，可在支持 CPU context switch tracing 的环境下重跑 Nsight Systems。
- 实现在线 warm-start 或连续性约束，替代当前离线 smoothness rerank。
