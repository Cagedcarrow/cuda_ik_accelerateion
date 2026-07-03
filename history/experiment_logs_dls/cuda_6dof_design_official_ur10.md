# CUDA 6DOF Design For Official UR10

## 当前实现状态

- 新子项目路径：`/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/`
- CUDA runner：`standard_robot_cuda_ik/build/standard_robot_cuda_runner`
- 标准模型：`standard_robot_cuda_ik/urdf/ur10_official.urdf`
- 常量头：`standard_robot_cuda_ik/include/standard_robot_cuda_ik/generated/ur10_model_constants.h`

## 已迁移的底层设计

- `1 block / target`
- `Grid = (N, 1, 1)`
- `Block = (128, 1, 1)`
- `4 warp` 分工
- `PaddedMat6x8`
- register-resident `6x6 LDLT`
- `constant memory` 机器人参数
- 单 kernel 内 DLS 迭代

## 与旧铲斗 UR10 版本的关键差异

- 目标 TCP 已切换为标准 `tool0`，不再使用铲斗 shovel tip。
- 机器人参数来自标准 UR10 URDF 自动导出，而不是旧工程手写常量。
- 新 runner 使用外部 `target/seed` 二进制文件，不再绑定旧 `273` 固定数据集。
- benchmark 的公平性基线改为同一 URDF / TCP / seed / tolerance，而不是旧工程内部约定。

## 当前问题

- 直接迁移旧 kernel 后，在 `zero_seed` + 标准 UR10 目标集上，当前 CUDA 收敛率仍偏低。
- 现阶段吞吐和 host 侧接口已经可以量化，但还没有完成真正的竞争性优化循环。
- 旧实现里部分数值策略仍带有“从旧装配体调出来”的痕迹，尚未重新针对标准 UR10 校准。

## 下一轮优化优先级

1. 固定同一 target/seed/tolerance，先提高标准 UR10 下的收敛率。
2. 在不放松公平条件的前提下，检查 warm-start 与 early-iteration damping。
3. 结合 Nsight Compute 继续看 register pressure、shared memory 访问和 stall。
4. 当收敛率进入可比区间后，再继续压缩 kernel / end-to-end 时间。

