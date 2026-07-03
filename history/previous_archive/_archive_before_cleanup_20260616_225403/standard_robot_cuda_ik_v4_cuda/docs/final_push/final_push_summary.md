# Final Push Summary

1. N=500 是否超过 cuRobo：`False`。
2. Raw throughput 是否超过：`False`；valid throughput 是否超过：`False`。
3. N=500 best raw gap: `-61.70%`；best valid gap: `-56.40%`。
4. CUDA Graph 是否有效：`False`。
5. Mixed Precision 是否有效：`False`。
6. 主文结果：OPT4C 静态 benchmark、cuRobo boundary、Nsight bottleneck、保守结论。
7. 附录/negative：CUDA Graph、Mixed Precision、Adaptive-K。
8. 最新论文 MD：`paper/final/cuda_ik_paper_latest.md`。

| item | status | main_metric | pass_or_fail | paper_decision | path | notes |
| --- | --- | --- | --- | --- | --- | --- |
| N500 raw throughput vs cuRobo | done | -61.699952191110796 | 0 | main_text_if_pass_else_gap_analysis | data/results/final_push/final_push_summary.csv | {'source': 'cuda_graph', 'mode': 'off/fp64/none', 'throughput': 17198.596853562653, 'valid_throughput': 16407.46139829877, 'pass_quality': 1} |
| N500 valid throughput vs cuRobo | done | -56.39827492878244 | 0 | main_text_if_pass_else_gap_analysis | data/results/final_push/final_push_summary.csv | {'source': 'cuda_graph', 'mode': 'off/fp64/none', 'throughput': 17198.596853562653, 'valid_throughput': 16407.46139829877, 'pass_quality': 1} |
| CUDA Graph | done | 1.0 | 0 | appendix_or_system_optimization | data/results/final_push/cuda_graph_benchmark.csv | E2E speedup criterion |
| Mixed Precision | done | 1.1128798148719323 | 0 | main_text_candidate_or_negative_ablation | data/results/final_push/mixed_precision_benchmark.csv | quality and speed gates |
| Figures | done | 8 figures | 1 | main_text_and_appendix | figures/final_push | PNG and SVG generated |
| Latest paper markdown | done | generated | 1 | handoff | paper/final/cuda_ik_paper_latest.md | Word-ready version also generated |