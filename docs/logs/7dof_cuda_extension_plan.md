# 7DOF CUDA Extension Plan

## 目标

将当前标准 UR10 6DOF 主线扩展到 `Franka Panda 7DOF`，用于展示框架可扩展性，不阻塞 UR10 主实验。

## 已完成

- 7DOF 模型文件已固化：
  `standard_robot_cuda_ik/urdf/panda_7dof.urdf`
- 配置入口已建立：
  `standard_robot_cuda_ik/config/robots.yaml`

## 尚未完成

- 7x7 Hessian / LDLT 实现
- `PaddedMat7x8` 或 `PaddedMat8x8` 版本
- 7DOF benchmark runner
- Panda target/seed 数据链路实测

## 必须修改的模块

- `src/cuda/cuda_ik_7dof.cu`
- `src/cuda/cuda_utilities.cuh`
- 共享内存矩阵封装
- benchmark wrapper

## 主要风险

- 冗余自由度导致解不唯一
- 初始 seed 和 null-space 策略会直接影响公平 benchmark
- 7x7 小矩阵会提高寄存器压力并压低 occupancy

## 约束

- 7DOF 未完成时，不得把它写成“已有结果”。
- UR10 主实验、消融和 profiling 优先级高于 7DOF。

