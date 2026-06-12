# 4. CUDA 底层设计

## 4.1 设计目标

本文的 CUDA 设计目标不是追求”理论上最复杂的 GPU 结构”，而是针对 6DOF 批量 IK 的小矩阵、强控制流、重复迭代特点，做一条在消费级 GPU 上可运行、可分析、可对比的主线。

## 4.2 数据布局

当前主线保留了以下核心布局：

- `PaddedMat6x8`：把共享内存中的 6 列小矩阵按 8 列 stride 做 padding；
- 常量内存：存放 segment origins、axes、joint limits、tool transform、weight schedule、lambda 参数；
- 寄存器：存放 `LDLT` 求解中的小矩阵中间量和步长向量；
- 单 block 内共享状态：
  - 当前关节 `q`
  - 当前 FK `T`
  - Jacobian `J`
  - Hessian `H`
  - 误差向量 `err`

## 4.3 迭代流程

单个 block 的迭代可以概括为：

1. 载入一个 target 与对应 seed；
2. 计算当前 FK；
3. 计算位置与姿态误差；
4. 构造数值 Jacobian；
5. 构造 `J^T W^2 J + λI` 与 `J^T W^2 e`；
6. 用寄存器内 `6x6 LDLT` 解出 `dq`；
7. 做步长裁剪、关节限位与 branch alignment；
8. 直到收敛或达到最大迭代数。

## 4.4 Phase 6 的关键修复

与性能同样重要的是数值定义修复。当前工作树相对旧迁移版，新增了以下关键修复：

- 主动关节前 fixed transform 正确折叠进每段 origin；
- 最终写回关节解前，重新计算一次 FK，确保误差统计与输出一致；
- 姿态残差改为与数值 Jacobian 一致的定义，同时用 geodesic angle 做最终收敛判据；
- 在主公平 `zero_seed` 主线上重新校准权重、阻尼与步长裁剪。

这些修改解释了为什么同样是 DLS 框架，修复后 CUDA 可以从“位置很准但姿态停在错误分支”恢复到高收敛区间。

## 4.5 Nsight Compute 观察到的硬件特征

对 `N=100` 的 full NCU 动态 profiling 表明：

- Registers / thread: `94`
- Static shared memory / block: `1.616 KB`
- Compute throughput: `65.66%`
- DRAM throughput: `1.57%`
- Achieved occupancy: `32.92%`
- Local / shared spill: `0 / 0`

这说明当前 kernel 的主要限制不在外部带宽，而在：

- 寄存器占用导致的 occupancy 上限；
- 小批次时 `waves per SM = 0.83` 的 under-fill；
- 小矩阵迭代里的依赖链与 scoreboard stall。

## 4.6 为什么当前设计仍然值得继续优化

虽然当前 `cuda` 已经在主 GPU 对比中取得吞吐优势，但它仍不是“最终最优设计”：

- `94 registers/thread` 仍有压低 occupancy 的副作用；
- `PaddedMat6x8` 虽然避免了最严重的问题，但共享访问仍有冲突痕迹；
- `curobo` 在解精度上更接近机器精度；
- 当前消融还不足以量化每个优化项的独立贡献。

因此，本文把当前设计描述为“**已验证有效的标准化主线**”，而不是最终封顶版本。
