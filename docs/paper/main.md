# 面向标准机械臂批量逆运动学求解的 CUDA 底层优化框架与标准化评测

## 备选标题

1. 面向标准机械臂批量逆运动学求解的 CUDA 底层优化框架与标准化评测
2. 基于官方 UR10 模型的批量逆运动学 CUDA 求解与公平 Benchmark
3. 从模型统一到性能追赶：标准机械臂批量 IK 的 CUDA 实现与分析

## 摘要

本文面向标准机械臂的批量逆运动学求解，构建了一个以官方 UR10 模型为基准的标准化 CUDA 研究工程。与旧工程中“自定义装配体 + 固定 273 数据集 + 非统一比较口径”的实验环境不同，本文将机器人模型、TCP 定义、target/seeds 资产、误差阈值和 benchmark 入口全部重构为统一且可复现的资产链路。核心 CUDA 求解器采用 `1 block / target` 的并行映射、`128 threads / block` 的四 warp 分工、`PaddedMat6x8` 共享内存布局、寄存器内 `6x6 LDLT` 小矩阵求解、常量内存参数广播，以及单 kernel 内的 DLS 迭代。

在标准化过程中，我们发现旧 CUDA kernel 直接迁移到官方 UR10 后会出现严重收敛退化。进一步排查表明，问题并非单纯来自参数调节，而是涉及三个关键一致性缺陷：主动关节前固定变换未被正确折叠、输出关节与最终 FK/误差统计可能错位、姿态残差与数值 Jacobian 的定义不一致。修正这些问题后，自研 CUDA 恢复到稳定满收敛区间。

在此基础上，本文建立了 A0–A8 共九组消融配置并完成了完整实测。消融研究表明：自适应阻尼是对收敛率影响最显著的数值策略，在大批量（N=5000）上实现 147% 的吞吐提升并将收敛率从 83.4% 恢复至 100%；步长钳位与分支对齐（A6）在所有批量上持续降低吞吐 15–20%；混合精度（A7，FP32 主体 + FP64 LDLT 关键路径）在所有批量上带来 148–154% 的吞吐提升。本文采用 A7 配置作为主对比版本，在 N=100/500/1000/5000 上分别达到 `107250/149787/140246/180962 targets/s`。与 cuRobo 相比，CUDA 在小中批量上显著占优（N=100 领先 37.6 倍，N=500 领先 10.3 倍，N=1000 领先 4.8 倍），在 N=5000 上以 1.25 倍反超 cuRobo。

本文强调，所报告的实验是**随机可达目标位姿批量 IK 测试**，不是路径规划，也不是完整运动规划流水线。除主结果外，文中还给出完整 A0–A6 消融实测表、Nsight Compute 动态 profiling 证据，以及对 7DOF Panda 扩展的后续计划。当前稿件已形成完整 Markdown 论文初稿，所有实验数字均可在标准化的数据资产与 benchmark 入口上复现验证。

## 关键词

- CUDA
- inverse kinematics
- UR10
- benchmark fairness
- Nsight Compute
- small-matrix solver

## 论文结构

- [01_introduction.md](/mnt/linuxdata/cuda_ik_accelerateion/docs/paper/01_introduction.md)
- [02_related_work.md](/mnt/linuxdata/cuda_ik_accelerateion/docs/paper/02_related_work.md)
- [03_kinematics_and_cuda_mapping.md](/mnt/linuxdata/cuda_ik_accelerateion/docs/paper/03_kinematics_and_cuda_mapping.md)
- [04_cuda_low_level_design.md](/mnt/linuxdata/cuda_ik_accelerateion/docs/paper/04_cuda_low_level_design.md)
- [05_experiments.md](/mnt/linuxdata/cuda_ik_accelerateion/docs/paper/05_experiments.md)
- [06_7dof_extension.md](/mnt/linuxdata/cuda_ik_accelerateion/docs/paper/06_7dof_extension.md)
- [07_discussion.md](/mnt/linuxdata/cuda_ik_accelerateion/docs/paper/07_discussion.md)
- [08_conclusion.md](/mnt/linuxdata/cuda_ik_accelerateion/docs/paper/08_conclusion.md)
- [references.md](/mnt/linuxdata/cuda_ik_accelerateion/docs/paper/references.md)
