# opt4c_block_target Report

## Correctness

| metric | value |
|---|---:|
| correctness_pass | 1 |
| strict_sr | 0.96 |
| pos_p95_all_mm | 4.38451277 |
| near_limit | 0.01 |
| strict_sr_diff_pp | 0.0 |
| pos_p95_diff_mm | 0.0 |
| near_limit_diff_pp | 0.0 |
| best_seed_diff_count | 0 |
| max_q_abs_diff | 0.0 |

## Benchmark

| method | N | K | warmup | repeat | gpu_stream_ms_mean | gpu_stream_ms_std | e2e_ms_mean | e2e_ms_std | raw_throughput_mean | raw_throughput_std | valid_throughput_strict | loose_sr | medium_sr | strict_sr | pos_p50_all_mm | pos_p95_all_mm | pos_p99_all_mm | pos_max_all_mm | pos_p95_suc_mm | rot_p50_all_deg | rot_p95_all_deg | rot_p95_suc_deg | near_limit_ratio | iter_mean | iter_p95 | monotonic_pass | nan_count | inf_count | speedup_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CUDA-V4-opt4c_block_target-analytic | 100 | 16 | 10 | 30 | 6.58284372 | 0.280973336 | 6.58284372 | 0.280973336 | 15191.0032 | 0 | 14583.3631 | 0.98 | 0.96 | 0.96 | 0.151638842 | 4.38451277 | 57.9649387 | 78.5528607 | 3.1743141 | 0.0431771737 | 0.573046801 | 0.519748983 | 0.01 | 21 | 60 | 1 | 0 | 0 | 9.754467511496689 |
| CUDA-V4-opt4c_block_target-analytic | 1000 | 16 | 10 | 30 | 57.3580533 | 0.320213377 | 57.3580533 | 0.320213377 | 17434.3434 | 0 | 16632.3636 | 0.965 | 0.954 | 0.954 | 0.181917525 | 4.56313281 | 96.5181052 | 169.785555 | 2.54634627 | 0.0298171471 | 0.529917399 | 0.448161392 | 0.007 | 19 | 60 | 1 | 0 | 0 | 10.384154094016298 |
| CUDA-V4-opt4c_block_target-analytic | 5000 | 16 | 10 | 30 | 274.726237 | 1.24399593 | 274.726237 | 1.24399593 | 18199.9363 | 0 | 17362.7392 | 0.965 | 0.954 | 0.954 | 0.181917525 | 4.56313281 | 96.5181052 | 169.785555 | 2.55241929 | 0.0298171471 | 0.529917399 | 0.448621612 | 0.007 | 19 | 60 | 1 | 0 | 0 | 10.803301579091624 |

## Nsight / Profiling

| metric | value |
|---|---|
| ncu_status | pass |
| achieved_occupancy | 16.04 |
| sm_utilization | 84.24 |
| dram_throughput | 0.58 |
| branch_divergence | 2375.97 |
| branch_efficiency | 98.24 |
| warp_execution_efficiency | 10.57 |

## Decision

`main_result`
