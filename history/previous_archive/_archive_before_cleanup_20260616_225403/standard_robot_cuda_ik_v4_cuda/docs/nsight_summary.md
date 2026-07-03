# Nsight Summary

Nsight Compute basic set was collected for `ik_lm_multiseed_v4_kernel` at N=100/1000/5000. The kernel is not DRAM-bound: DRAM throughput stays below 1%, while registers/thread is high and occupancy is low. This points to FP64 scalar LM work, register pressure, occupancy, and one-thread-per-block mapping as the dominant performance limits. The finite-difference limit gradient is retained for Python correctness alignment and should be replaced by an analytical piecewise gradient in a later optimization pass.

| N | duration_ms | registers/thread | achieved occupancy | memory throughput | DRAM throughput | waves/SM |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 69.66 | 184 | 13.19 | 2.26 | 0.75 | 8.33 |
| 1000 | 653.34 | 184 | 13.06 | 2.32 | 0.67 | 83.33 |
| 5000 | 3260.0 | 184 | 13.07 | 2.33 | 0.80 | 416.67 |
