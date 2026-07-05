# CUDA IK 论文终稿修订记录

## 审稿意见处理

- P0-1 吞吐归因：将“完全来源于单核函数融合”改为“主要与单核函数融合减少中间同步和固定结构 LM 较轻量计算路径有关”，并明确 cuRobo-Graph-K16 的优化流程更复杂。
- P0-2 cuRobo-K1 表述：删除“质量不可用”，改为“失败尾部较重”，并保留成功样本误差很低的事实。
- P0-3 有效吞吐量：承认 cuRobo-Graph-K1 在 `throughput × Strict SR` 指标上更高，OPT4C-K16 的优势限定为一次性高成功率和少回退的规划前端可控性。
- P0-4 图 5/图 6 图例：默认对比图改为 OPT4C-K16 与 cuRobo-Graph-K1，并在正文和图题中说明同等 K=16 对比见表 6 和 Pareto 图。
- P0-5 表 6：吞吐和有效吞吐压缩为 `10^4 targets/s`，方法名用 `cu-Graph` 缩短并加表注说明。
- P0-6 图 1：线程映射图改为彩色并在 LaTeX 中使用通栏 `figure*` 放大；图注说明 block、lane、shared memory 和 global output 关系。
- P0-7 轨迹连续性：加入 `20×49=980` 相邻间隔、jump ratio 约 `73.3%--79.4%`，并强调 smoothness rerank 不能保证连续控制轨迹。
- P0-8 阈值定义：补充 Loose、Medium、Strict、Ultra 四级位置/姿态误差阈值，Strict 仍作为主指标。
- P0-9 近奇异关节编号：改为物理关节 `q1--q6`，并补充脚本中的 0-based index 映射。
- P1-1 同等算力：全文使用“同等种子数”或“同等 K 值”，不使用“同等算力”。
- P1-2 CPU baseline：保留 Python/NumPy CPU-LM 仅作量级对照，不代表工业级优化 CPU IK。
- P1-3 图 7：Pareto 图改为彩色，手动调整点标注偏移，避免点标注与数据点重叠。
- P1-4 结论局限性：补充不处理碰撞、不保证连续控制轨迹、不面向亚毫米级精密装配任务。

## 摘要与语言

- 中文摘要按参考论文和投稿格式改为“针对问题、提出方法、首先、然后、最终、实验结果表明”的结构。
- 摘要中不再出现阿拉伯数字。
- 中文摘要主体估算约 216 字，符合 200--250 字范围。
- 通过 Nature polish 风格润色，删除或降级“天花板、决定性、灾难性、零延迟、零额外”等不够克制的表达。

## 图表和脚本

- 修改 `data/experiments/补充实验/scripts/plot_all.py`，将补充实验图统一改为彩色风格。
- 修改 `论文/绘图/generate_all_figures_bw.py`，将主性能图改为彩色，并明确 cuRobo 默认曲线为 cuRobo-Graph-K1。
- 重新生成并同步以下论文图：`fig_thread_mapping_redraw.pdf`、`fig_algorithm_pipeline_redraw.pdf`、`fig_kernel_time_breakdown.pdf`、`fig_nsys_timeline_opt4c.pdf`、`fig1_throughput.pdf`、`fig2_strict_sr.pdf`、`fig_pareto_throughput_success.pdf`、`fig_threshold_scan.pdf`、`fig_trajectory_delta_q.pdf`、`fig4_scalability.pdf` 等。
- 对 `fig_thread_mapping_redraw.pdf` 和 `fig_pareto_throughput_success.pdf` 做了 PNG 视觉检查，并修正了说明文字/点标注重叠问题。

## 自检

- 风险表达扫描已通过：未发现“完全来源于”“质量不可用”“全面优于 cuRobo”“同等算力”“轨迹连续性问题已解决”等高风险表达。
- 引用检查已通过：`bibitems=25`，`cited_unique=25`，无 unused/missing citation。
- 审稿自检见 `reports/final_revision_review_self_check.md`。

## 编译验收

- 已执行 `latexmk -xelatex -interaction=nonstopmode paper.tex`。
- `paper.pdf` 已重新生成，当前为 13 页。
- 日志检查未发现 fatal error、LaTeX Error、undefined citation/reference、`Overfull`。
- `pdftotext` 检查未发现 `??` 或 `[?]`。
- PDF 文本已检出新摘要、cuRobo-Graph-K1/K16、四级阈值定义、jump ratio、近奇异关节编号映射和结论局限性。
- 抽查 PDF 第 6--8 页，表 6、图 7、图 8 未见明显版心溢出或数据标注遮挡。
