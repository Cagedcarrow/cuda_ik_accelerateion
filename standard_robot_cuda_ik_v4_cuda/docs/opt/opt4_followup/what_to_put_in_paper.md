# What To Put In The Paper After OPT4C

## Main Text Results

- Quality mode: `CUDA-V4-Final-K16-OPT4C`
- OPT4C static benchmark table
- OPT4C vs cuRobo-Graph boundary table: CUDA wins N=100 throughput and all-N quality; cuRobo wins N>=500 throughput
- OPT4C thread mapping figure: target-block / seed-thread / fused selection


## Appendix Results

- OPT4B warp-target mapping
- Raw Nsight metrics
- Detailed Adaptive-K stage table
- Original baseline vs OPT4C speedup table
- OPT4C + Adaptive-K, because it did not meet all fast-mode gates

## Discussion / Future Work

- Original warp-per-seed cooperative solve
- Further reduction of FP64 pressure
- Collision-aware IK and full motion planning are outside this paper
- cuRobo is not replaced; comparison is a system boundary under shared targets/evaluation

## Abstract Contribution Points

- Fixed-size batch IK acceleration on GPU
- Constraint-aware multi-seed LM formulation
- Target-block seed-parallel CUDA mapping
- Fused candidate generation and selection
- Adaptive-K negative/appendix result after OPT4C, showing that reduced seeds are not automatically faster once fixed-K16 mapping is optimized
- Honest cuRobo boundary rather than overclaiming
