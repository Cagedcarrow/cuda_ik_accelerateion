# CUDA Static Benchmark Report

Method: CUDA-V4-Final-K16 fp64_debug. Static selection uses `success_rank -> near_limit -> pose_cost`.

| N | Strict SR | pos_p95_all_mm | near_limit | GPU ms | throughput |
|---|---:|---:|---:|---:|---:|
| 100 | 0.9600 | 4.385 | 0.0100 | 73.6663 | 1357.5 |
| 500 | 0.9540 | 4.340 | 0.0040 | 319.9693 | 1562.7 |
| 1000 | 0.9540 | 4.509 | 0.0070 | 639.1583 | 1564.6 |
| 5000 | 0.9540 | 4.509 | 0.0070 | 3186.4361 | 1569.2 |

N=5000 is a tiled scaling set derived from the frozen seed42 N=1000 assets.
