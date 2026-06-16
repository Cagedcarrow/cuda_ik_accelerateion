# CUDA cuRobo Comparison Report

This is a system comparison under unified target sets and evaluation protocol, not an equivalent-algorithm comparison. V4 uses fixed Sobol-K16 seeds; cuRobo uses its internal optimizer, CUDA Graph path, and internal parallel strategy.

| Method | N | GPU ms | throughput | Strict SR | pos_p95 mm | notes |
|---|---:|---:|---:|---:|---:|---|
| CUDA-V4-Final-K16-fp64_debug | 100 | 73.6663386 | 1357.47211 | 0.96 | 4.38451371 | V4 uses fixed Sobol-K16 seeds and finite-difference limit gradient |
| CUDA-V4-Final-K16-fp64_debug | 500 | 319.969279 | 1562.65002 | 0.954 | 4.33960598 | V4 uses fixed Sobol-K16 seeds and finite-difference limit gradient |
| CUDA-V4-Final-K16-fp64_debug | 1000 | 639.158346 | 1564.55753 | 0.954 | 4.50895888 | V4 uses fixed Sobol-K16 seeds and finite-difference limit gradient |
| CUDA-V4-Final-K16-fp64_debug | 5000 | 3186.43612 | 1569.15118 | 0.954 | 4.50895888 | V4 uses fixed Sobol-K16 seeds and finite-difference limit gradient |
| cuRobo-Graph | 100 | 9.747084808349609 | 10259.477778867522 | 0.86 | 74.32701949661212 | collision disabled; zero external seed; cuRobo internal optimizer/graph/seeding not equivalent to V4 Sobol-K16 |
| cuRobo-Graph | 500 | 12.535373115539551 | 39887.12544823831 | 0.836 | 109.01179323780562 | collision disabled; zero external seed; cuRobo internal optimizer/graph/seeding not equivalent to V4 Sobol-K16 |
| cuRobo-Graph | 1000 | 15.65944938659668 | 63859.20573017882 | 0.84 | 98.53029432352942 | collision disabled; zero external seed; cuRobo internal optimizer/graph/seeding not equivalent to V4 Sobol-K16 |
| cuRobo-Graph | 5000 | 38.749427795410156 | 129034.16345653101 | 0.8418 | 74.11459640010173 | collision disabled; zero external seed; cuRobo internal optimizer/graph/seeding not equivalent to V4 Sobol-K16 |
