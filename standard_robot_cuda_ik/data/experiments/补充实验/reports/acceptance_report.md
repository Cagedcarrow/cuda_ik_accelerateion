# 补充实验总体验收报告

## CSV 产物
- [x] `fair_curobo_k16_summary.csv`: rows=33
- [x] `near_singular_summary.csv`: rows=18
- [x] `near_limit_barrier_summary.csv`: rows=12
- [x] `trajectory_continuity_summary.csv`: rows=6
- [x] `seed_count_scan.csv`: rows=15
- [x] `kernel_time_breakdown.csv`: rows=3
- [x] `cpu_baseline_summary.csv`: rows=6
- [x] `threshold_scan.csv`: rows=16

## 图产物
- [x] `fig_pareto_throughput_success.pdf`: 18964 bytes
- [x] `fig_near_singular_sr.pdf`: 15192 bytes
- [x] `fig_near_limit_barrier.pdf`: 12601 bytes
- [x] `fig_trajectory_delta_q.pdf`: 24040 bytes
- [x] `fig_seed_count_scan.pdf`: 15978 bytes
- [x] `fig_kernel_time_breakdown.pdf`: 26255 bytes
- [x] `fig_threshold_scan.pdf`: 14601 bytes
- [x] `fig_thread_mapping_redraw.pdf`: 14388 bytes
- [x] `fig_nsys_timeline_opt4c.pdf`: 17793 bytes

## 报告产物
- [x] `nsys_opt4c_n1000_summary.txt`: 675 bytes
- [x] `paper_review_report.md`: 3760 bytes
- [x] `final_revision_changelog.md`: 3508 bytes
- [x] `final_revision_acceptance_report.md`: 2362 bytes

## 通用字段检查
- [x] `fair_curobo_k16_summary.csv` common fields complete
- [x] `near_singular_summary.csv` common fields complete
- [x] `near_limit_barrier_summary.csv` common fields complete
- `trajectory_continuity_summary.csv`: specialized schema, header columns=21
- [x] `seed_count_scan.csv` common fields complete
- `kernel_time_breakdown.csv`: specialized schema, header columns=14
- [x] `cpu_baseline_summary.csv` common fields complete
- `threshold_scan.csv`: specialized schema, header columns=11

## 结论
总体验收：通过。补充实验、CPU baseline、Nsight Systems timeline、论文审稿报告和正文合并产物均已生成。
