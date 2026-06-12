# Roofline 模型验证

> 测试日期: 2026-06-05 | RTX 4060 Laptop | sm_89

## Roofline 模型简介

Roofline 模型分析 GPU 计算性能的标准工具，将算术强度（FLOP/Byte）映射到可实现性能（GFLOPS）：

```
可实现性能 = min(峰值 GFLOPS, 峰值带宽 × 算术强度)
```

- **计算绑定** (Compute-Bound): 算术强度 > Ridge Point，性能受限于计算吞吐
- **内存绑定** (Memory-Bound): 算术强度 < Ridge Point，性能受限于内存带宽

## RTX 4060 Laptop (Ada Lovelace) Roofline 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| FP64 峰值 | **228 GFLOPS** | 48 FP64 cores × 24 SM × 2.25 GHz × 2 (FMA) |
| FP32 峰值 | 14 TFLOPS | 3,072 FP32 cores × 2.25 GHz × 2 (FMA) |
| FP64:FP32 | 1:64 | 消费级 GPU 硬件限制 |
| DRAM 带宽 | 256 GB/s | GDDR6, 128-bit |
| **Ridge Point (FP64)** | **0.89 FLOP/Byte** | 228 GFLOPS / 256 GB/s |

## Kernel 在 Roofline 图上的位置

### 两种 FLOP 统计口径

| 口径 | FLOP 总量 | 算术强度 | vs Ridge Point |
|------|----------|---------|---------------|
| **全线程理论 FP64** | 273×6.7×5,800 ≈ 10.6 MFLOP | ~150 FLOP/Byte | **~168×** |
| **ncu 纯 FP64 指令** | 2.58M inst × 1.7 FLOP ≈ 4.4 MFLOP | ~62 FLOP/Byte | **~70×** |

> 两种口径的差异来源：(a) ncu 仅统计实际执行的 FP64 指令（编译器已优化/融合冗余计算）；(b) ncu 不含 FP32 超越函数（sin/cos/atan2）；(c) 地址计算和分支不产生浮点指令。无论哪种口径，kernel 均在 Ridge Point 极右侧 → **极端 Compute-Bound**。

### 关键计算

| 参数 | 值 |
|------|-----|
| 算术强度（全线程口径） | ~150 FLOP/Byte |
| Ridge Point (FP64) | 0.89 FLOP/Byte |
| 距离 Ridge | **~168×** |
| 实测 FP64 吞吐 | **1.65 GFLOPS** |
| FP64 利用率（vs 228 GFLOPS 峰值） | **0.72%** |

## 每次迭代 FLOP 明细

| 阶段 | FP64 FLOP | 贡献 |
|------|----------|------|
| FK (Rodrigues × 6 关节) | 1,012 | 17.5% |
| Pose Error | 60 | 1.0% |
| Jacobian (6 列 × 2 FK) | 3,030 | 52.2% |
| Hessian J^T·W²·J | 540 | 9.3% |
| Adaptive Damping | 6 | 0.1% |
| LDL^T Solve (63 scalar ops) | 99 | 1.7% |
| 关节更新 + 夹紧 + 限位 | 1,053 | 18.2% |
| **每迭代总计（全线程）** | **5,800** | 100% |

## 为什么实测性能远低于 FP64 峰值？

这是 Roofline 分析的核心发现——**不是访存瓶颈，是 FP64 硬件限制**：

| 原因 | 量化 | 说明 |
|------|------|------|
| **FP64 核心仅 48 个** | 1:64 比例 | 每个 SM 仅 2 个 FP64 核心，其余 128 个 FP32 核心闲置 |
| **`__syncthreads()` 屏障** | 72.7% stall | 每个 DLS 迭代 14 个同步点，Warp 大量等待 |
| **SM Busy** | 41.76% | SM 在其余 58.24% 时间里等待屏障完成 |
| LDL^T 串行瓶颈 | Lane 0 独占 | 6×6 LDL^T 由单个线程串行执行，无法并行 |
| 小问题规模 | 273 targets | 仅 2.27 waves/SM，尾部效应明显 |

**核心结论**：kernel 处于极端 Compute-Bound 状态（~168× Ridge Point），DRAM Throughput 仅 0.16%，性能瓶颈完全在于消费级 GPU 的 FP64 硬件限制和同步屏障等待——而非访存设计。在当前硬件上继续优化访存将无显著收益。

## 优化方向

1. **FP32/FP64 混合精度**: DLS 主体用 FP32（14 TFLOPS），仅 LDL^T 用 FP64 → 预期 10-20× 加速
2. **减少同步点**: 探索 Warp Shuffle 替代部分 `__syncthreads()` → 降低 72.7% 屏障等待
3. **换用更高 FP64 比例的 GPU**: RTX 6000 Ada (FP64:FP32=1:32) 可翻倍 FP64 吞吐
4. **Tensor Core 批处理**: 将多目标的 Hessian 合并为 batch GEMM
