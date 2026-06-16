# Figure List

| figure_id | filename_png | filename_svg | data_source | caption_cn | used_in_main_text | notes |
| --- | --- | --- | --- | --- | --- | --- |
| fig1 | fig1_method_pipeline.png | fig1_method_pipeline.svg | fig1_method_pipeline.csv | 本文方法从批量目标位姿输入到 Sobol-K16 多种子、OPT4C 并行求解、限制屏障和轨迹平滑重排序的完整流程。 | 1 |  |
| fig2 | fig2_opt4c_mapping.png | fig2_opt4c_mapping.svg | fig2_opt4c_mapping.csv | OPT4C 将一个目标映射到一个 CUDA block，并用 thread 0-15 并行求解 K16 种子后在块内完成候选选择。 | 1 |  |
| fig3a | fig3a_throughput.png | fig3a_throughput.svg | fig3_static_throughput.csv | OPT4C 在 N=100/500/1000/5000 下的吞吐与有效吞吐。 | 1 |  |
| fig3b | fig3b_quality.png | fig3b_quality.svg | fig3_quality.csv | OPT4C 在不同 batch 规模下保持稳定 Strict SR 与 near-limit 比例。 | 1 |  |
| fig4 | fig4_curobo_boundary.png | fig4_curobo_boundary.svg | fig4_curobo_boundary.csv | CUDA-V4-OPT4C 与 cuRobo-Graph 的系统边界对比。 | 1 |  |
| fig5 | fig5_mixed_precision.png | fig5_mixed_precision.svg | fig5_mixed_precision.csv | Mixed precision 扫描在 N=500 下的吞吐和有效吞吐。 | 0 | 进入附录或 negative ablation 取决于质量和速度门控。 |
| fig6 | fig6_cuda_graph_e2e.png | fig6_cuda_graph_e2e.svg | fig6_cuda_graph.csv | CUDA Graph replay 对 N=100/500/1000 E2E 延迟的影响。 | 0 | 系统优化边界分析。 |
| fig7 | fig7_ablation_summary.png | fig7_ablation_summary.svg | fig7_ablation_summary.csv | 算法和 CUDA 映射消融结果摘要。 | 1 | 受历史结果文件可用性限制。 |
| fig8 | fig8_nsight_bottleneck.png | fig8_nsight_bottleneck.svg | fig8_nsight_bottleneck.csv | Nsight 指标显示 OPT4C 主要受 FP64 计算、寄存器压力和 occupancy 限制。 | 1 |  |