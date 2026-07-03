# Final Paper Readiness Report v2: OPT4C

## 1. Final Method Definition

- Quality mode: `CUDA-V4-Final-K16-OPT4C`
- Fast mode: `CUDA-V4-OPT4C-AK-4+4+8` (does not pass all gates)

## 2. Correctness and Quality

N=1000 Quality mode:

- Strict SR = 0.954
- pos_p95_all = 4.56313281 mm
- near_limit = 0.007
- NaN/Inf = 0/0

The fixed K16 OPT4C quality gate passes.

## 3. OPT4C Static Benchmark

| N | gpu_stream_ms_mean | raw_throughput_mean | valid_throughput_strict | strict_sr | pos_p95_all_mm | near_limit_ratio | speedup_vs_old_baseline | quality_gate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 7.09497388 | 14094.4846 | 13530.7052 | 0.96 | 4.38451277 | 0.01 | 10.38289074011362 | 1 |
| 500 | 30.5672481 | 16357.377 | 15604.9376 | 0.954 | 4.33796842 | 0.004 | 10.46771622859959 | 1 |
| 1000 | 56.1285248 | 17816.253 | 16996.7054 | 0.954 | 4.56313281 | 0.007 | 11.387406818146056 | 1 |
| 5000 | 267.9697 | 18658.826 | 17800.52 | 0.954 | 4.56313281 | 0.007 | 11.891031411387182 | 1 |

## 4. OPT4C vs cuRobo Boundary

| N | cuda_method | curobo_method | cuda_gpu_ms | curobo_gpu_ms | cuda_raw_throughput | curobo_raw_throughput | cuda_valid_throughput_strict | curobo_valid_throughput_strict | cuda_strict_sr | curobo_strict_sr | cuda_pos_p95_all_mm | curobo_pos_p95_all_mm | throughput_winner | strict_sr_winner | pos_p95_winner | valid_throughput_winner | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | CUDA-V4-Final-K16-OPT4C | cuRobo-Graph | 7.09497388 | 9.209430472056072 | 14094.4846 | 10858.434764606489 | 13530.7052 | 9338.25389756158 | 0.96 | 0.86 | 4.38451277 | 74.05754538199889 | CUDA-V4-OPT4C | CUDA-V4-OPT4C | CUDA-V4-OPT4C | CUDA-V4-OPT4C | Targets/UR10/tool0/collision-disabled/thresholds shared; cuRobo internal seed, optimizer, CUDA Graph, and parallel strategy are not algorithmically equivalent to Sobol-K16 LM. |
| 500 | CUDA-V4-Final-K16-OPT4C | cuRobo-Graph | 30.5672481 | 11.134643173217773 | 16357.377 | 44904.896566659016 | 15604.9376 | 37630.30332286025 | 0.954 | 0.838 | 4.33796842 | 108.96247329653461 | cuRobo-Graph | CUDA-V4-OPT4C | CUDA-V4-OPT4C | cuRobo-Graph | Targets/UR10/tool0/collision-disabled/thresholds shared; cuRobo internal seed, optimizer, CUDA Graph, and parallel strategy are not algorithmically equivalent to Sobol-K16 LM. |
| 1000 | CUDA-V4-Final-K16-OPT4C | cuRobo-Graph | 56.1285248 | 13.487415409088134 | 17816.253 | 74143.18975644339 | 16996.7054 | 62354.42258516889 | 0.954 | 0.841 | 4.56313281 | 100.08661538959697 | cuRobo-Graph | CUDA-V4-OPT4C | CUDA-V4-OPT4C | cuRobo-Graph | Targets/UR10/tool0/collision-disabled/thresholds shared; cuRobo internal seed, optimizer, CUDA Graph, and parallel strategy are not algorithmically equivalent to Sobol-K16 LM. |
| 5000 | CUDA-V4-Final-K16-OPT4C | cuRobo-Graph | 267.9697 | 34.60245259602865 | 18658.826 | 144498.42785345955 | 17800.52 | 121869.97405160779 | 0.954 | 0.8434 | 4.56313281 | 74.08098230147469 | cuRobo-Graph | CUDA-V4-OPT4C | CUDA-V4-OPT4C | cuRobo-Graph | Targets/UR10/tool0/collision-disabled/thresholds shared; cuRobo internal seed, optimizer, CUDA Graph, and parallel strategy are not algorithmically equivalent to Sobol-K16 LM. |

This remains a system-level boundary comparison. CUDA-V4-OPT4C wins N=100 throughput and all-N quality metrics in this target set; cuRobo-Graph wins N>=500 throughput and valid strict throughput. Do not claim full cuRobo replacement.

## 5. Adaptive-K Decision

| method | N | K_max | avg_seeds | strict_sr | medium_sr | loose_sr | pos_p95_all_mm | pos_p95_suc_mm | rot_p95_all_deg | near_limit | gpu_stream_ms_mean | raw_throughput | valid_throughput_strict | speedup_vs_OPT4C_K16 | quality_drop_vs_OPT4C_K16 | pass_quality | pass_fast_mode | stage1_active_ratio | stage1_failed_ratio | stage2_active_ratio | stage2_failed_ratio | stage3_active_ratio | stage3_failed_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CUDA-V4-Final-K16-OPT4C | 100 | 16 | 16.0 | 0.96 | 0.96 | 0.98 | 4.38451277 | 3.1743141 | 0.573046801 | 0.01 | 7.09497388 | 14094.4846 | 13530.7052 | 1.0 | 0.0 | 1 | 1 | nan | nan | nan | nan | nan | nan |
| CUDA-V4-OPT4C-AK-4+4+8 | 100 | 16 | 5.199999999999999 | 0.96 | 0.96 | 0.98 | 4.794822656228 | 4.566035196575 | 0.8606200444447498 | 0.04 | 10.403897670000001 | 9611.78235041214 | 9227.311056395654 | 0.6819534471641915 | 0.0 | 1 | 0 | 1.0 | 0.16 | 0.16 | 0.07 | 0.07 | 0.04 |
| CUDA-V4-Final-K16-OPT4C | 500 | 16 | 16.0 | 0.954 | 0.954 | 0.966 | 4.33796842 | 2.09849052 | 0.568232018 | 0.004 | 30.5672481 | 16357.377 | 15604.9376 | 1.0 | 0.0 | 1 | 1 | nan | nan | nan | nan | nan | nan |
| CUDA-V4-OPT4C-AK-4+4+8 | 500 | 16 | 5.239999999999999 | 0.954 | 0.954 | 0.966 | 4.823028831318998 | 3.923661989513999 | 0.8601312995198486 | 0.02 | 36.34324796999999 | 13757.713686259729 | 13124.85885669178 | 0.8410708950733333 | 0.0 | 1 | 0 | 1.0 | 0.154 | 0.154 | 0.078 | 0.078 | 0.046 |
| CUDA-V4-Final-K16-OPT4C | 1000 | 16 | 16.0 | 0.954 | 0.954 | 0.965 | 4.56313281 | 2.54634627 | 0.529917399 | 0.007 | 56.1285248 | 17816.253 | 16996.7054 | 1.0 | 0.0 | 1 | 1 | nan | nan | nan | nan | nan | nan |
| CUDA-V4-OPT4C-AK-4+4+8 | 1000 | 16 | 5.191999999999999 | 0.954 | 0.954 | 0.965 | 4.8283794001909985 | 4.016664339863498 | 0.8641780462561997 | 0.027 | 67.80155827 | 14748.923557446728 | 14070.473073804178 | 0.8278353216674529 | 0.0 | 1 | 0 | 1.0 | 0.146 | 0.146 | 0.076 | 0.076 | 0.046 |
| CUDA-V4-Final-K16-OPT4C | 5000 | 16 | 16.0 | 0.954 | 0.954 | 0.965 | 4.56313281 | 2.55241929 | 0.529917399 | 0.007 | 267.9697 | 18658.826 | 17800.52 | 1.0 | 0.0 | 1 | 1 | nan | nan | nan | nan | nan | nan |
| CUDA-V4-OPT4C-AK-4+4+8 | 5000 | 16 | 5.191999999999999 | 0.954 | 0.954 | 0.965 | 4.828379400191006 | 4.03069685746 | 0.8641780462562014 | 0.027 | 314.37461970000004 | 15904.591804425487 | 15172.980581421914 | 0.8523897388908712 | 0.0 | 1 | 0 | 1.0 | 0.146 | 0.146 | 0.076 | 0.076 | 0.046 |

Fast mode decision: appendix/future work.

## 6. Nsight Update

OPT4C Nsight:

- achieved occupancy: 16.04
- SM throughput: 84.24
- DRAM throughput: 0.58
- branch efficiency: 98.24
- avg active threads per warp: 10.57

Interpretation: OPT4C primarily fixes thread mapping and kernel granularity. It does not make the problem memory-bound; FP64 compute remains a key limit.

## 7. Final Paper Claims

Can write:

- fixed-size batch IK on UR10
- constraint-aware multi-seed IK
- target-block seed-parallel CUDA mapping
- fused candidate generation and selection
- Adaptive-K as an appendix/negative result after OPT4C, because it preserves quality but is slower than OPT4C-K16
- system-level performance boundary against cuRobo-Graph

Cannot write:

- fully surpasses cuRobo in all settings
- complete motion planning
- collision-aware planning
- V5 or motion generation
- direct drop-in replacement for cuRobo

## 8. Final Decision

`CUDA-V4-Final-K16-OPT4C` is the final main CUDA quality result. `CUDA-V4-OPT4C-AK-4+4+8` is not a main-text fast mode.
