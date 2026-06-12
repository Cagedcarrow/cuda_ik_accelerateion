# Roofline And NCU Report For Official UR10

## Profiled Artifact

- Launch-only report:
  `standard_robot_cuda_ik/data/profiling/ur10_cuda_n100_launch.ncu-rep`
- Full dynamic report:
  `standard_robot_cuda_ik/data/profiling/ur10_cuda_n100_full_zero_seed.ncu-rep`
- Profiled executable:
  `standard_robot_cuda_ik/build/standard_robot_cuda_runner`
- Data:
  `standard_robot_cuda_ik/data/targets/ur10_seed42_N100.bin`
  `standard_robot_cuda_ik/data/seeds/ur10_seed42_zero_seed_N100.bin`

## Full NCU Metrics (`N=100`)

下表来自 `ncu --import ... --print-summary per-kernel` 与 raw CSV 抽取结果。

| Metric | Value | Interpretation |
|---|---:|---|
| Grid | `(100, 1, 1)` | 1 block / target |
| Block | `(128, 1, 1)` | 4 warps / block |
| Kernel duration | `2.92 ms` | 动态 profiling 下的单 kernel 时间 |
| Registers / thread | `94` | 寄存器使用偏高，但仍无 spill |
| Static shared memory / block | `1.616 KB` | 共享内存占用很小 |
| Compute (SM) Throughput | `66.89%` | 已明显高于内存吞吐 |
| DRAM Throughput | `1.56%` | 远低于 compute throughput，不是 DRAM-bound |
| L1/TEX Throughput | `1.03%` | 不是 L1 带宽瓶颈 |
| L2 Throughput | `0.31%` | 不是 L2 带宽瓶颈 |
| Memory Throughput | `3.99 GB/s` | 远未接近设备峰值 |
| L1/TEX hit rate | `99.13%` | 数据复用良好 |
| L2 hit rate | `17.68%` | 主路径更依赖寄存器/常量/L1 |
| Theoretical occupancy | `41.67%` | 受寄存器上限约束 |
| Achieved occupancy | `32.51%` | `N=100` 时还受 waves/SM 不足影响 |
| Waves / SM | `0.83` | 小批次 under-fill 明显 |
| Local spilling requests | `0` | 无寄存器 spill |
| Shared spilling requests | `0` | 无共享 spill |
| Shared bank conflicts | `3522` | 有共享访问冲突痕迹，但不是主瓶颈 |
| Shared excessive wavefronts | `3522` | 与上项一致，说明可继续优化共享访问模式 |

## Bottleneck Classification

当前 `N=100` 的标准化 UR10 kernel 更接近：

- `COMPUTE / OCCUPANCY MIXED`

判断依据：

1. `Compute (SM) Throughput = 66.89%`，而 `DRAM Throughput = 1.56%`，排除 DRAM-bound。
2. `94 registers/thread` 导致 `Block Limit Registers = 5`，理论 occupancy 被压到 `41.67%`。
3. `Waves Per SM = 0.83` 说明小批次下还有明显 under-fill。
4. `No Eligible = 97.83%` 与较低的 `Eligible Warps Per Scheduler` 说明瓶颈更偏向小矩阵迭代中的依赖链，而非远程内存等待。

## Why The Full NCU Runtime Is Not A Benchmark Number

同一命令在 `ncu --set full` 下会输出：

- `kernel_time_only_ms_mean ≈ 3572 ms`
- `gpu_end_to_end_ms_mean ≈ 3572 ms`
- `host_api_total_ms_mean ≈ 3572 ms`

这不是正常运行速度，而是 45 次 profiling pass 叠加后的观测开销。真实基准应以非 profiling 运行结果为准，例如：

- `N=100, repeat=30, cuda`:
  `host_api_total = 2.635002 ms`
- `N=1000, repeat=30, cuda`:
  `host_api_total = 21.970249 ms`

因此，本文件只用于瓶颈分类，不用于吞吐对比。

## Roofline Interpretation

虽然当前没有额外导出图形化 roofline 截图，但依据 Nsight Compute 的 `GPU Speed Of Light Throughput` 数据，可以得到：

- 当前 kernel 显著偏离 memory roof；
- 主要优化方向不在 DRAM/L2 带宽，而在：
  - 迭代内指令依赖链；
  - occupancy 提升；
  - 小批次时 block/SM 填充；
  - 共享访问模式进一步压缩冲突。

## Remaining Phase 7 Gaps

仍未完成的 profiling 工作：

- 对 `A1-A6` 逐版本做 full NCU；
- 对 `N=5000` 做动态 profile 以区分“大批次更高吞吐”是 grid 填充改善还是调度行为变化；
- 将 stall 指标系统化整理到消融表中。

因此，本文件已经提供了**真实 full NCU 动态证据**，但还不是最终投稿版的完整 roofline 附录。
