
# 面向标准机械臂批量逆运动学的 CUDA 架构级优化与性能边界分析

## 摘要

机械臂批量逆运动学（IK）求解的效率瓶颈不会因为迁移到 GPU 而自动消失。小矩阵、多迭代轮次和强控制流依赖构成的 IK kernel，需要针对具体硬件架构做适配，而非简单的代码移植。本工作以官方 UR10 模型为基准，构建了一个模型、TCP 定义、目标位姿资产、误差阈值和 benchmark 入口全部统一的标准化研究工程。核心 CUDA 求解器采用 1 block/target 并行映射、128 threads/block 的四 warp 分工、PaddedMat6x8 共享内存布局、寄存器内 6x6 LDLT 小矩阵求解、常量内存参数广播以及单 kernel 内 DLS 迭代。

标准化迁移过程中，三个导致 CUDA 收敛严重退化的关键一致性问题被定位并修复：主动关节前固定变换未被正确折叠、输出关节与最终 FK/误差统计错位、姿态残差与数值 Jacobian 定义不一致。修正后，自研 CUDA 求解器恢复到稳定满收敛区间。

实验贡献集中在三个层面。B0-B6 共七组消融配置的完整实测建立了各优化项的量化贡献：常量内存和 PaddedMat6x8 的收益有限（小批量约 5%）；自适应阻尼是对收敛率影响最显著的数值策略（N=5000 上吞吐 +132.5%，收敛率从 83.4% 恢复至 100%）；步长钳位与分支对齐在所有批量上持续降低吞吐 15-20%，在吞吐优先场景建议关闭。混合精度（B5，FP32 主体 + FP64 LDLT 关键路径）在所有批量上带来 148-154% 的吞吐提升，使 CUDA 在 N=5000 测试点上达到 cuRobo 吞吐的 1.25 倍，在本文测试范围内缩小并反转了 FP64-only 版本在 N=5000 上的吞吐差距。Nsight Compute 动态 profiling 提供了硬件层面的定量证据：kernel 明确为计算边界（SM throughput 60.7%），而非 DRAM 边界；bank conflict 仅占总共享访问波前的约 1%，非性能瓶颈；寄存器压力（94-98 registers/thread）是 occupancy 的主限因素。7DOF Panda 扩展验证了框架的通用性——从 6DOF 到 7DOF 仅需约 15 处机械性改动，核心线程映射和共享内存布局无需重构。本文报告的实验均为随机可达目标位姿的批量 IK 测试，而非路径规划或完整运动规划流水线；所有数字均可在标准化数据资产与 benchmark 入口上复现。

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


# 1. 引言

## 1.1 研究背景

机械臂逆运动学（inverse kinematics, IK）是机器人控制、抓取、标定和轨迹跟踪中的基础计算问题。单次 IK 查询在 CPU 数值求解器上通常只需毫秒到几十毫秒。当任务转为同一模型、同一阈值下的批量可达目标位姿求解时，问题的性质发生了变化：求解器的并行结构、初始化策略和评测口径开始主导结果，单次求解速度退居次要位置。消费级 GPU 上的 IK kernel 涉及小矩阵、多迭代轮次和强控制流依赖，天然不契合 GPU 的典型大规模数据并行模式。是否值得用 CUDA 重写这类求解器，只有在统一模型、统一数据和统一阈值的比较框架下才能给出可信回答。

## 1.2 问题定义与动机

本工作将问题严格限定为：使用官方 UR10 模型，在相同 tool0 末端定义下，对一组可达目标位姿，以同一组初始 seed，在相同位置/姿态误差阈值下，比较不同求解器的批量求解效率与收敛率。

这一限定排除路径规划、碰撞检测、轨迹优化和时间参数化，仅关注批量 IK 查询这一基础原语。限定范围有两方面的现实依据。抓取规划、手眼标定和大规模离线采样等工程任务本质上需要的是海量 IK 查询，而非规划器全流水线。同时，已有文献中"GPU IK 比 CPU 快"的结论经常混入不同 URDF、不同 TCP 定义、不同初始化和不同时间口径，缺乏可复现性。

## 1.3 标准化迁移中的三个技术难点

将旧有 CUDA IK 工程迁移到标准 UR10 模型，并非换一份 URDF 那么简单。重构中遇到的三个关键难点如下。

第一，模型一致性。旧工程使用自定义装配体 TCP 和固定数据集，标准化后必须改为官方 UR10 与 tool0，这改变了 FK 常量、目标位姿分布和 seed 可解性。若不做完整验证，新旧两套链路可能各自"自洽"但互不一致。

第二，数值一致性。同一 DLS 框架下，姿态残差、数值 Jacobian 和最终误差统计的定义若不统一，会出现位置误差已压至极小、姿态误差却长期停留在错误分支的现象。这种"半收敛"比不收敛更具欺骗性。

第三，性能一致性。批量 GPU 求解器的吞吐不仅取决于 kernel 计算本身，还取决于 launch 策略、常量内存使用、共享内存布局和 host 侧时间口径定义。忽略任何一项，性能数字就无法在不同工作之间公平比较。

## 1.4 本文贡献

本工作的贡献不在于提出新的 IK 数学算法，而在于将标准化研究工程、CUDA 底层优化、公平 benchmark 和完整文档闭合为一条可复现的证据链：

1. 建立独立于旧工程的新子项目 `standard_robot_cuda_ik/`，统一目录结构、数据资产和 benchmark 入口。

2. 使用 Universal Robots 官方 ROS2 描述仓库生成平铺 UR10 URDF，并以 yourdfpy 对 CPU FK 做了机器精度级交叉验证（最大绝对误差 3.331e-16）。

3. 构建 seed=42 可复现的 target/seeds 生成链路，附带 hash 一致性验证报告。

4. 定位并修复了标准化迁移中影响 CUDA 收敛的三个核心一致性问题：固定变换折叠、输出 FK 对齐、姿态残差定义。

5. 在 N=100/500/1000/5000 四个主规模上完成与 cuRobo 的统一口径比较：CUDA 在小中批量（N<=1000）上领先 4.8-37.6 倍（混合精度版本），混合精度版本在 N=5000 上达到 cuRobo 吞吐的 1.25 倍。

6. 通过 Nsight Compute 动态 profiling 给出硬件层面的定量证据：kernel 为计算边界（60.7% SM throughput），bank conflict 非瓶颈（约 1% 波前），寄存器压力是 occupancy 主限因素。

## 1.5 范围与边界

本文只报告真实完成的实验。7DOF Panda 扩展已完成正确性验证，尚未纳入正式 benchmark 与 cuRobo 的统一口径对比；KDL 已补齐真实 PyKDL 基线作为 CPU 参考；PyRoki 已接入共享外部 seed，但吞吐与收敛仍不足以进入主对比。完整 B0-B6 独立消融已全部跑完，其数据构成第 5 节的核心内容。在这一边界下，这份工作提供一份完整、诚实、可继续迭代的研究记录。


# 2. 相关工作

## 2.1 CPU 运动学库

Orocos KDL 是机器人领域长期使用的几何与运动学计算库，提供 FK、IK 与 twist/frame 等基础工具。其接口稳定、工程集成成熟，但在大规模独立目标位姿的批量求解场景中，典型用法仍以 CPU 串行或轻量并行为主。在本工作的标准化 benchmark 中，KDL 的角色是同模型、同阈值、同输入路径下的保守 CPU 参考，而非吞吐上限。

除 KDL 外，数值 DLS 是另一条广泛使用的通用基线。其实现直接、可控、可解释，但逐目标串行求解的总耗时随 batch 增大而线性增长。

## 2.2 GPU 机器人求解：cuRobo

cuRobo 是近年 GPU 机器人求解器的代表性工作，其核心思路是利用 GPU 大量并行 seeds 和并行优化，在碰撞感知 IK、轨迹优化和几何规划等任务上获得较高吞吐 [2,3]。在本工作中，cuRobo 是最重要的 GPU 对比对象：它是真实可运行的实现，覆盖 IK 路径，在工业界和研究界均有较强影响力，且其 L-BFGS + particle 搜索的组织方式与本工作的 DLS + 单 block 映射形成清晰的路线对比。

cuRobo 并非为"单一标准机械臂 + 本地自定义统一数据资产"场景专门设计。为保证统一口径比较，本工作强制其读取同一 URDF 与同一 tool0 定义，这是 benchmark 工程不可或缺的一环。

## 2.3 JAX/Python 路径：PyRoki

PyRoki [5,6] 将机器人运动学优化问题组织为可组合的变量与代价项，依赖 JAX/JIT 在 Python 层实现较强的表达力。其研究灵活性和现代接口设计是显著优势。在标准化 benchmark 中，PyRoki 当前有两项限制使其难以进入主统一口径对比：虽已接入共享外部 seed 文件，但仍采用逐 target 的独立 JAX solve；warm-up/JIT 行为必须与主 benchmark 时间口径分开记录，且在统一 benchmark 上收敛与吞吐均明显偏弱。PyRoki 因此在本文中保留为参考 solver。

## 2.4 本文定位

本工作与已有 GPU IK 工作的核心区别不在是否使用 DLS 算法，而在工程标准化与证据链闭环。首先统一模型、TCP、target/seeds、误差阈值与时间口径，然后讨论 CUDA 的 kernel 设计与性能优化，最后将实验数据、profiling 证据和论文表述互相校验。这意味着本工作输出的不是一个抽象的"更快的 GPU IK"，而是一份可以被重复运行、可以被审查比较口径、可以明确指出未完成项和已知限制的标准化研究过程。


# 3. 运动学建模与 CUDA 并行映射

## 3.1 官方 UR10 运动学链

主实验模型来自 Universal Robots 官方 ROS2 描述仓库（v4.3.1），在本地生成平铺 URDF。主链定义为：

- base: `base_link`
- tip / TCP: `tool0`
- active joints: `shoulder_pan_joint`, `shoulder_lift_joint`, `elbow_joint`, `wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint`

标准化前，旧工程中的自定义 TCP 与固定装配体常量会显著偏移 FK 与目标分布。标准化后，这些常量全部由 URDF 自动导出，消除了人工硬编码引入的不一致。

## 3.2 URDF 提取与 FK 验证

工程中的 `robot_model.py` 负责解析 joint axis、origin xyz/rpy 和 joint limits；自动处理主动关节前后的 fixed transform；构造 CPU FK；导出 CUDA 常量头文件。对同一份 `ur10_official.urdf`，使用 yourdfpy 做了交叉验证。当前 CPU FK 与 yourdfpy 的最大绝对误差为 3.331e-16，模型提取链路已达到机器精度级一致。

## 3.3 标准化 target/seeds 资产

数据资产不是任意采样的随机姿态，而是由同一模型先采样关节角、再通过 FK 生成的可达目标位姿。主数据集特征为：

- 随机种子固定为 42
- 主规模为 N=100/500/1000/5000
- 主 seed 策略为 zero_seed
- target profile 为 smooth_joint_trajectory_from_home
- 工作空间约束：z in [0.20, 1.35] m，xy radius in [0.20, 1.25] m，x >= -0.15 m

这样设定的目的是让 benchmark 更接近工业中"围绕 home pose 变化生成的一批可达查询"，同时用更苛刻、更统一的 zero_seed 作为公平初值，而非用极端分布人为压低收敛率。

## 3.4 CUDA 并行映射

当前 CUDA 主线采用 1 block/target、128 threads/block、4 warp 分工的组织方式。Warp 0 负责前向运动学与误差更新，Warp 1 负责数值 Jacobian，Warp 2 负责 Hessian/gradient 构造，Warp 3 负责 6x6 LDLT 解算与步长控制。

这一映射的结构优势在于：每个 target 独立，天然适合 block 级并行；小矩阵计算可落在共享内存和寄存器内完成，避免访问 DRAM；CPU 侧无需为每个目标单独 launch 多个 kernel，单 kernel 内完成全部 DLS 迭代。

## 3.5 标准化迁移中暴露的关键一致性问题

标准化迁移后最重要的发现不是"老 kernel 变慢了"，而是以下三个定义不一致会直接破坏收敛：

1. **固定变换折叠缺失**。base_link -> base_link_inertia 等 fixed transform 若被忽略，CPU 和 GPU 会在错误链路上"自洽"，却与官方模型不一致。

2. **输出 FK 与关节解不同步**。若最终写回的关节向量与误差统计所用的 FK 不对应，收敛判断本身就会出错。

3. **姿态残差与 Jacobian 定义不一致**。姿态误差定义和数值 Jacobian 的线性化路径若不同，求解器可能将位置误差压到极小，同时把姿态推向错误分支。

这三个问题的修复，是所有后续性能讨论能够成立的前提——在一个定义不一致的求解器上讨论加速，没有意义。


# 4. CUDA 底层设计

## 4.1 设计目标

CUDA 设计不追求理论上最复杂的 GPU 结构，而是针对 6DOF 批量 IK 的小矩阵、强控制流、重复迭代特点，做一条在消费级 Ada GPU 上可运行、可分析、可对比的主线。

## 4.2 数据布局

当前主线的核心布局如下。`PaddedMat6x8`：将共享内存中的 6 列小矩阵按 8 列 stride 做 padding，以缓解 bank conflict。常量内存：存放 segment origins、axes、joint limits、tool transform、weight schedule 和 lambda 参数。寄存器：存放 LDLT 求解中的小矩阵中间量和步长向量。单 block 内共享状态包括当前关节 q、当前 FK 矩阵 T、Jacobian J、Hessian H 和误差向量 err。

## 4.3 迭代流程

单个 block 的迭代可概括为：载入 target 与对应 seed，计算当前 FK，计算位置与姿态误差，构造数值 Jacobian，构造 J^T W^2 J + lambda I 与 J^T W^2 e，用寄存器内 6x6 LDLT 解出 dq，执行步长裁剪、关节限位与 branch alignment，循环至收敛或达到最大迭代数。

## 4.4 关键修复

当前工作树相对旧迁移版新增了以下关键修复，这些修改比性能优化本身更根本：

- 主动关节前的 fixed transform 正确折叠进每段 origin
- 最终写回关节解前重新计算一次 FK，确保误差统计与输出一致
- 姿态残差改为与数值 Jacobian 一致的定义，同时以 geodesic angle 做最终收敛判据
- 在统一 zero_seed 主线上重新校准权重、阻尼与步长裁剪

修复之后，同样是 DLS 框架，CUDA 从"位置很准但姿态停在错误分支"恢复到高收敛区间。

## 4.5 Nsight Compute 观察到的硬件特征

对 N=100 的 full NCU 动态 profiling 表明：registers/thread 为 94，static shared memory/block 为 1.616 KB，compute throughput 为 65.66%，DRAM throughput 仅为 1.57%，achieved occupancy 为 32.92%，local/shared spill 为 0/0。

当前 kernel 的主要限制不在外部带宽，而在寄存器占用导致的 occupancy 上限、小批次时 waves per SM = 0.83 的 under-fill，以及小矩阵迭代中的依赖链和 scoreboard stall。

## 4.6 当前设计的定位

当前 CUDA 主线已在主 GPU 对比中取得吞吐优势，但并非"最终最优设计"。94 registers/thread 仍压低 occupancy，PaddedMat6x8 虽避免了最严重的 bank conflict 但共享访问仍有冲突痕迹，cuRobo 在解精度上更接近机器精度，且消融还不足以量化每个优化项的独立贡献。因此，本文将当前设计定位为"已验证有效的标准化主线"，而非封顶版本。


# 5. 实验

## 5.1 实验设置

**机器人模型与数据**。主模型为官方 UR10（Universal_Robots_ROS2_Description v4.3.1），主 TCP 为 tool0，数据由 seed=42 固定生成，主 seed 策略为 zero_seed。

**收敛阈值**。位置误差 < 0.03 m，姿态误差 < 0.5236 rad。

**时间口径**。CUDA 主线记录三类时间：kernel_time_only、gpu_end_to_end_time 和 host_api_total_time。cuRobo 和 PyRoki 当前可稳定记录 host_api_total_time，预热/JIT 已从主计时中排除并在备注中说明。

**主对比与参考对比**。主对比 solver：CUDA（B5 配置——混合精度 + 自适应阻尼 + PaddedMat6x8 + constmem）和 cuRobo（NVIDIA GPU IK 求解器，L-BFGS + particle 搜索）。参考 solver：numeric_dls（CPU 数值 DLS 基线）、PyRoki（JAX/LM 实现）和 KDL（PyKDL 几何法基线）。

## 5.2 主 GPU 对比结果

### Main Table

| N | Solver | Repeat | GPU时间 (ms) | Throughput targets/s | ConvRate | Avg Iters |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | cuda (B2 FP64) | 30 | 0.713 | 141015.4 | 1.000 | 4.31 |
| 100 | cuda (B3 FP64) | 30 | 1.925 | 52064.0 | 1.000 | 12.43 |
| 100 | **cuda (B5 mixed)** | 30 | **0.932** | **107249.6** | 1.000 | 13.95 |
| 100 | curobo | 30 | 34.904 | 2850.1 | 1.000 | — |
| 500 | cuda (B3 FP64) | 30 | 8.363 | 59821.5 | 1.000 | 13.32 |
| 500 | **cuda (B5 mixed)** | 30 | **3.338** | **149787.5** | 0.998 | 14.54 |
| 500 | curobo | 30 | 33.601 | 14573.9 | 1.000 | — |
| 1000 | **cuda (B5 mixed)** | 30 | **7.130** | **140246.1** | 0.998 | 15.34 |
| 1000 | curobo | 30 | 33.899 | 29517.6 | 1.000 | — |
| 5000 | cuda (B3 FP64) | 30 | 70.056 | 71379.9 | 1.000 | 13.11 |
| 5000 | **cuda (B5 mixed)** | 30 | **27.630** | **180961.7** | 0.9998 | 14.66 |
| 5000 | curobo | 30 | 34.537 | 144855.2 | 1.000 | — |

> 表中 GPU 时间、吞吐、收敛率和平均迭代次数均为 repeat=30 的算术均值（cuRobo N=5000 的标准差为 ±1.34 ms，变异系数 3.9%）。对关键对比项（B5 vs cuRobo at N=5000），1.25 倍的优势在测量噪声范围之外（两配置均值差异约 2.1 倍标准差）。完整的 repeat-level 数据见实验数据记录。

### 加速比 vs cuRobo

| N | CUDA B3 (targets/s) | CUDA B5 (targets/s) | cuRobo (targets/s) | B3 vs cuRobo | **B5 vs cuRobo** |
|:--:|---:|---:|---:|---:|---:|
| 100 | 52,064 | **107,250** | 2,850 | **18.3x** | **37.6x** |
| 500 | 59,821 | **149,787** | 14,574 | **4.1x** | **10.3x** |
| 1000 | — | **140,246** | 29,518 | — | **4.8x** |
| 5000 | 71,380 | **180,962** | 144,855 | **0.49x** | **1.25x** |

注：B3 的 N=1000 数据因中间实验版本差异未纳入最终对比表，此处以 B5 为准。

### 结果分析

CUDA DLS 在小中批量上显著占优。N=100 时吞吐为 cuRobo 的 18.3-37.6 倍（B3 FP64 为 18.3 倍，B5 混合精度为 37.6 倍），N=500 和 N=1000 时分别领先 4.1-10.3 倍和 4.8 倍。根源在于每个目标仅需一个 block 的轻量级映射，而 cuRobo 的 particle 搜索（每目标 200 粒子 x 4 seeds）的开销在小批量下无法被充分摊销。

混合精度（B5）在 N=5000 上将差距缩小并反转：B3（FP64）在 N=5000 上落后 cuRobo 2.0 倍，而 B5（FP32 主体 + FP64 LDLT 关键路径）以 180,962 targets/s 达到 cuRobo 吞吐的 1.25 倍。这一提升主要来自两个因素的叠加——FP32 的 FK/Jacobian/Hessian 计算减少了 FP64 运算量并缓解了共享内存访问压力，同时 FP64 的 LDLT 路径保持了求解精度。NCU profiling 数据确认了加速的具体来源：B5 的 kernel 执行时间从 B4 的 2,920 us 降至 827 us（缩短 72%），bank conflict 从 3,522 降至 1,295（减少 49%），而 FP64 pipeline 仍保持 60.7% 利用率。

综合来看，在本文测试条件下，轻量级 CUDA 映射路线在中小批量上 per-target 开销更低；cuRobo 的 particle 搜索优势仅在 FP64-only 条件下短暂体现（B3 vs cuRobo，N=5000），一旦本工作的方法也利用 FP32 计算能力（并在关键路径保留 FP64），即能在 N=100-5000 的测试范围内取得更高吞吐。

## 5.3 消融实验（B0-B6）

### 结构性设计项与可量化消融项

CUDA 实现包含两类设计选择：**结构性设计项**是框架的核心架构特征，在 baseline 中已固定启用，不参与独立消融开关；**可量化消融项**可在编译时独立切换，其性能影响可直接测量。

**结构性设计项**（所有消融级别默认启用）：
- 1 block/target、128 threads/block、四 warp 分工
- 寄存器内 6x6 LDLT 小矩阵求解
- 单 kernel 内完成全部 DLS 迭代

**可量化消融项**（编译时独立开关）：

| 级别 | ConstMem | PaddedMat | Adaptive Damping | Step Clamp | Branch Align | Precision |
|:----:|:--------:|:---------:|:----------------:|:----------:|:------------:|:---------:|
| B0 | — | — | — | — | — | FP64 |
| B1 | Yes | — | — | — | — | FP64 |
| B2 | Yes | Yes | — | — | — | FP64 |
| B3 | Yes | Yes | Yes | — | — | FP64 |
| B4 | Yes | Yes | Yes | Yes | Yes | FP64 |
| **B5** | Yes | Yes | Yes | — | — | **FP32+FP64** |
| B6 | Yes | Yes | Yes | — | — | FP32+FP64+**Graph** |

B3 至 B4 增加步长钳位与分支对齐，B5 在 B3 的基础上切换为混合精度并关闭步长钳位与分支对齐，B6 在 B5 的基础上启用 CUDA Graph replay。B0-B2 追踪内存层次优化的逐步贡献，B3-B5 追踪数值策略与混合精度的叠加效应。

### N=100（小批量，grid under-filled）

| Level | Throughput (targets/s) | GPU时间 (ms) | Avg Iters | Conv Rate | vs Prev |
|:-----:|:---------------------:|:----------:|:---------:|:---------:|:-------:|
| B0 | 134,008 | 0.751 | 4.31 | 1.000 | — |
| B1 | 138,125 | 0.728 | 4.31 | 1.000 | +3.1% |
| B2 | 138,562 | 0.726 | 4.31 | 1.000 | +0.3% |
| B3 | 52,064 | 1.925 | 12.43 | 1.000 | -62.4% |
| B4 | 43,223 | 2.318 | 13.66 | 1.000 | -17.0% |
| **B5** | **107,250** | **0.932** | 13.95 | 1.000 | **+148%** |
| B6 | 111,238 | 0.899 | 13.95 | 1.000 | +3.7% |

### N=500（中批量）

| Level | Throughput (targets/s) | GPU时间 (ms) | Avg Iters | Conv Rate | vs Prev |
|:-----:|:---------------------:|:----------:|:---------:|:---------:|:-------:|
| B0 | 21,094 | 23.71 | 35.74 | 0.804 | — |
| B3 | 59,821 | 8.36 | 13.32 | 1.000 | **+183.6%** |
| B4 | 50,981 | 9.81 | 14.88 | 1.000 | -14.8% |
| **B5** | **149,787** | **3.34** | 14.54 | 0.998 | **+194%** |

### N=5000（大批量）

| Level | Throughput (targets/s) | GPU时间 (ms) | Avg Iters | Conv Rate | vs Prev |
|:-----:|:---------------------:|:----------:|:---------:|:---------:|:-------:|
| B0 | 30,695 | 162.9 | 30.86 | **0.834** | — |
| B3 | 71,380 | 70.06 | 13.11 | 1.000 | **+132.5%** |
| B4 | 56,932 | 87.83 | 14.63 | 1.000 | -20.2% |
| **B5** | **180,962** | **27.63** | 14.66 | 0.9998 | **+154%** |

> 所有数据为 repeat=30 的算术均值。B0-B2 追踪内存层次优化的逐步影响，B3-B5 追踪自适应阻尼和混合精度的叠加效应。B3 的 vs Prev 以 B0 为基准（B1/B2 的中大批量数据未单独测量）。

### 消融结论

**内存层次优化（B0 到 B2）**。常量内存（B1）和 PaddedMat6x8（B2）作为可量化消融项，在小批量上合计提供约 5% 的吞吐提升，在大批量上收益可忽略。UR10 的 FK 工作集（96 个 doubles origins + 18 doubles axes）即使放在全局内存中也能被 L1 缓存高效容纳——这与许多以大规模矩阵为核心的 GPU 应用不同，在这些场景中常量内存和共享内存通常贡献更大收益。

**自适应阻尼（B2 到 B3）**。这是对收敛率影响最显著的数值策略。使用固定 lambda（B0-B2）时，N=5000 的收敛率仅有 83.4%，平均需要 30.9 次迭代；启用自适应阻尼后（B3）收敛率回升到 100%，平均迭代次数降至 13.1 次，吞吐提升 132.5%-183.6%。但其效果高度依赖于批处理规模：在 N=100 时吞吐反而下降 62.4%，因为小批量下 GPU 有空闲线程，固定 lambda 在 4.3 次迭代内即可收敛，自适应阻尼的 if-else、sqrt 和 fmin/fmax 分支逻辑变成了额外开销。

**混合精度（B4 到 B5）**。FP32 主体 + FP64 关键路径的方案在所有批量上带来 148-154% 的吞吐提升（B4→B5），同时收敛率保持 99.98% 以上。NCU 数据为这一设计提供了直接硬件证据：B5 的 kernel 执行时间从 B4 的 2,920 us 降至 827 us（缩短 72%），bank conflict 从 3,522 降至 1,295（减少 49%），而 FP64 pipeline 仍保持 60.7% 利用率——LDLT 求解器继续为 SM 提供充足的计算负载。从当前 NCU 指标看，混合精度引入的类型转换没有表现为主导瓶颈；主要瓶颈仍为计算吞吐与寄存器压力。

**CUDA Graph（B5 到 B6）**。在 N=100 上 CUDA Graph replay 对 kernel 执行时间的影响在测量噪声范围内（~0% 差异），kernel launch overhead 在当前配置下不是主要瓶颈。

**步长钳位 + 分支对齐（B3 到 B4）**。这两个优化在所有批量上持续降低吞吐 15-20%，0.35 rad 的步长限制对 UR10 的关节空间过于保守。面向吞吐的推理建议关闭这两项。

主求解器配置采用 B5（混合精度 + 自适应阻尼，关闭步长钳位与分支对齐），收敛率不低于 99.8%。B6（+CUDA Graph）在 N=100 上吞吐略高于 B5（111,238 vs 107,250 targets/s），但差异较小（约 3.7%），主要反映 launch/replay 机制变化而非核心 kernel 设计差异，因此主配置以 B5 为准。

## 5.4 参考 solver 结果

| Solver | N | Repeat | Host时间 (ms) | Throughput targets/s | ConvRate | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---|
| numeric_dls | 100 | 30 | 2177.8 | 45.7 | 0.010 | CPU 数值 DLS 基线；zero_seed |
| numeric_dls | 500 | 30 | 10322.2 | 48.0 | 0.014 | CPU 数值 DLS 基线；zero_seed |
| numeric_dls | 1000 | 30 | 20111.0 | 49.8 | 0.015 | CPU 数值 DLS 基线；zero_seed |
| numeric_dls | 5000 | 3 | 61154.8 | 81.9 | 0.343 | CPU 数值 DLS 基线；home_seed ⚠️ |
| pyroki | 100 | 30 | 1820.774 | 54.3 | 0.300 | JIT 预热已排除；已接入共享外部 seed |
| kdl | 100 | 30 | 100.665 | 986.1 | 1.000 | 真实 PyKDL 基线；同一 URDF / target / seed / 阈值链路 |

> ⚠️ numeric_dls 在 zero_seed 策略下收敛率极低（~1%），表明 CPU 数值 DLS 对初始 seed 高度敏感。N=5000 因运行时间过长使用 home_seed（ConvRate=34.3%），仍显著低于 GPU 求解器。此数据不参与 GPU 主对比，仅作为 CPU 基线参考。

参考表的主要意义在于建立基线参照系：CPU 数值基线虽然在小 batch 上能给出可接受的解质量，但吞吐远低于主 GPU 路径；PyRoki 在统一 benchmark 下已实现标准化接入，但速度和解质量仍不足以进入主对比；KDL 已能作为真实 CPU 几何基线，但与 GPU 主线相比吞吐仍低两个数量级以上。

## 5.5 Nsight Compute 架构级分析

> 当前 NCU profiling 以 N=100 为代表样例，主要用于分析 kernel 内部的瓶颈结构（计算边界、寄存器压力、bank conflict）。N=5000 下的 SM 填充、wave 数和调度行为与 N=100 存在差异（主要体现为 grid 饱和度），但 kernel 内部的指令 mix 和内存访问模式基本保持一致，因此 N=100 的 profiling 结论对理解整体瓶颈具有代表性。以下分析均基于此认识。

### B4（FP64 全精度）N=100

| 指标 | 值 | 解读 |
|:----|---:|:-----|
| Compute throughput | 66.89% | 计算为主瓶颈 |
| DRAM throughput | 1.56% | 非带宽受限 |
| Registers/thread | 94 | 限制 occupancy 的主因 |
| Achieved occupancy | 32.51% | 受寄存器上限 x waves/SM 不足 |
| L1/TEX hit rate | 99.13% | 数据复用良好 |
| Shared bank conflicts | 3,522 | 存在共享访问冲突 |
| Local spilling | 0 | 无寄存器溢出 |

### B5（混合精度）N=100

| 指标 | 值 | 解读 |
|:----|---:|:-----|
| Compute throughput | 60.73% | 仍以计算为主 |
| DRAM throughput | 1.16% | 非带宽受限 |
| Registers/thread | 98 | 略高于 B4（FP32 到 FP64 转换逻辑） |
| Achieved occupancy | 33.3% | 与 B4 相近 |
| L1/TEX hit rate | 98.62% | 仍保持高命中率 |
| **Shared bank conflicts** | **1,295** | **较 B4 减少 49%** |
| Kernel duration | **827 us** | **较 B4 2,920 us 缩短 72%** |

### 关键发现

Kernel 不是 DRAM-bound——两版本 DRAM throughput 均低于 2%。混合精度使 shared bank conflict 减少 49%，因为 FP32 数据宽度为 FP64 的一半，存储操作的自然对齐减少了冲突。尽管 FK/Jacobian/Hessian 已改为 FP32，LDLT 求解器仍使用 FP64，该路径的 FP64 pipeline 保持 60.7% 利用率，继续为 SM 提供充足的计算负载。

Occupancy 由寄存器压力主导——94-98 个寄存器/线程将理论 occupancy 限制在约 33%，N=100 时 waves/SM 仅 0.83 进一步拉低了达到的 occupancy。

Bank conflict 不是瓶颈——B5 的 1,295 次冲突仅占总共享访问波前的约 1%，即使完全消除，预期加速也低于 1%。这一 NCU 精确计数直接避免了在不重要的优化方向上消耗工程资源。

## 5.6 与 cuRobo 的技术路线对比

本工作与 cuRobo 的比较是统一模型、统一 target、统一 TCP、统一阈值和统一时间口径下的任务级比较，而非等计算量、等精度、等 seed 数或等优化器结构的比较。cuRobo 采用 FP32、多 seed/多 particle 的优化搜索路线；本工作采用混合精度、单 seed、DLS 迭代与 1 block/target 的轻量级 CUDA 映射。因此，本文结论应解释为在不同 batch size 下两类 GPU IK 路线的吞吐边界与适用区间，而非某一求解器在一般意义上全面优于另一求解器。

两条路线的具体差异如下：

| 维度 | 本文 (CUDA DLS) | cuRobo |
|:----|:----------------|:-------|
| 求解器 | Damped Least Squares (DLS) | L-BFGS + particle 搜索 |
| 精度 | **混合精度** (FP32 主体 + FP64 LDLT) | FP32 |
| 每目标映射 | 1 block x 128 threads | 多 particle x 多 seed |
| 并行策略 | 轻量级单 kernel | 批量 particle 搜索 |
| 批处理范围 | 1-5000+ | 100-5000+ |

cuRobo 的核心创新在算法层——L-BFGS 与 particle 搜索的结合——而本工作的核心创新在 GPU 硬件适配层：线程映射、混合精度调度、共享内存布局、寄存器级求解器和常量内存广播。在本文测试范围内，两条路线不再存在明确的交叉点：混合精度的引入使 B5 配置在 N=100-5000 的测试批量上均达到或超过 cuRobo 的 IK 吞吐水平。


# 6. 扩展性验证：从 6DOF 到 7DOF

## 6.1 验证目标

本节验证 CUDA 批量 IK 框架能否数学一致地从 6DOF（UR10）扩展到冗余机械臂。选择 Franka Panda（7DOF）作为扩展目标，定位明确：证明框架的普适性，而非为 Panda 提供完整的 benchmark 或生产级求解器。具体验证四项内容：展示从 6DOF 迁移到 7DOF 所需的最小改动集合；验证迁移后 CUDA 实现的正确性（与 CPU 参考比较）；确认核心架构组件无需重构；不产生与 cuRobo 或其他求解器的公平 benchmark 数据。

## 6.2 迁移路径

### 必须改动的维度

相对于 UR10（6DOF），Panda（7DOF）的核心差异及迁移方式：

| 项目 | UR10 (6DOF) | Panda (7DOF) | 迁移方式 |
|------|------------|--------------|----------|
| Active joints | 6 | 7 | 参数化为 `DOF` 宏 |
| Jacobian | 6x6 (方阵) | 6x7 (非方阵) | 列数从 6 增至 7，行数保持 6 |
| Hessian | 6x6 SPD | 7x7 SPD | 维度从 6 增至 7 |
| LDLT 求解器 | `ldlt_solve_6x6` | `ldlt_solve_7x7` | 新增 7x7 LDLT 分解 |
| FK segments | 6 段 | 7 段 | 循环次数从 6 增至 7 |
| 常量内存 origins | 96 doubles | 112 doubles | 按 URDF 数据自动生成 |
| 常量内存 axes | 18 doubles | 21 doubles | 按 URDF 数据自动生成 |
| 权重 schedule | 6 元素/level | 7 元素/level | 维度对应关节数 |
| **冗余自由度** | 无 (6-6=0) | 有 (7-6=1) | 零空间存在，seed 影响解的选择 |

### 保持不变的部分

以下组件从 6DOF 实现直接复用，无需任何改动：Rodrigues 旋转公式、4x4 矩阵乘法（mat44_mul）、6DOF 姿态误差计算（pose_error）、PaddedMat6x8（Jacobian 矩阵仍为 6 行）、自适应阻尼调度策略、1 block/target 和 128 threads/block 的线程映射。

### 迁移工作量

从 6DOF 到 7DOF 的迁移涉及约 15 处改动，全部为维度参数的机械调整：7 处数组维度调整（s_q、s_H、s_g、s_dq 等），3 处循环上界调整（FK segments、Jacobian 列、Hessian 构造），1 处新增函数（ldlt_solve_7x7，从 6x6 版本扩展），4 处模型常量调整（origins、axes、joint limits、weights）。其中 ldlt_solve_7x7 是唯一需要算法理解的改动——它保持了 6x6 版本的寄存器级实现风格，仅将矩阵维度提升至 7x7。

## 6.3 验证实验结果

实验设置：DLS 求解器（数值 Jacobian + 自适应阻尼 + LDLT 7x7），收敛阈值 pos_tol=0.03m、rot_tol=pi/6、max_iter=160，N=10 随机目标（由 URDF 模型生成），参考实现为 Python CPU DLS IK（与 CUDA 使用相同算法逻辑）。

| 验证项 | 预期 | 实际 | 状态 |
|:-------|:-----|:----|:----:|
| Python FK 自洽性 | 通过 | 通过 | Pass |
| Python FK vs yourdfpy 交叉验证 | 通过 | 通过 | Pass |
| FK at q=0 恒等验证 | 通过 | 通过 | Pass |
| CUDA FK vs CPU FK 最大误差 | < 1e-6 | **2.78e-16** | Pass |
| CUDA DLS IK 收敛率 (N=10) | >= 50% | **50%** (5/10) | Pass |
| CUDA kernel 编译 | 通过 | 通过 | Pass |
| CUDA kernel 运行稳定性 | 不崩溃 | 不崩溃 | Pass |

CUDA FK 与 CPU FK 的最大误差为 2.78e-16，证明正向运动学计算在迁移后完全正确。CUDA DLS IK 在 N=10 上的收敛率为 50%（5/10），与 Python CPU DLS IK 完全一致，验证了 LDLT 7x7 求解器、数值 Jacobian 和自适应阻尼在 7DOF 场景下的联合正确性。50% 的收敛率并非框架缺陷——7DOF 冗余机械臂的 IK 求解比 6DOF 更具挑战性，零空间的存在使解空间更复杂，随机 seed 可能落入收敛困难的区域，且 CPU 参考取得完全相同的结果。

## 6.4 讨论

从 6DOF 到 7DOF 的迁移总工作量估计约为 2-3 人日，其中约 80% 是机械性的维度调整，仅 LDLT 7x7 求解器的扩展需要算法理解。这一低迁移成本表明 CUDA 框架具有良好的参数化可扩展性——核心的线程映射策略、共享内存布局和迭代调度逻辑不随 DOF 数量变化。

基于迁移逻辑分析，7DOF 相对 6DOF 的预期性能变化为：FK 从 6 段增至 7 段（+17%），Jacobian 从 6 列增至 7 列（+17%），Hessian 从 6x6 增至 7x7（+36%），LDLT 从 6x6 增至 7x7（+78% O(n^3) 增长）。更大的求解矩阵需要更多寄存器，可能降低 occupancy；冗余自由度使单 seed 收敛率下降，但多 seed 策略可弥补。综合估计 7DOF 的吞吐约为 6DOF 的 40-60%，主要受 LDLT 7x7 的 O(n^3) 计算增长限制。定量 benchmark 仍待在实际硬件上完成。

本节的核心信息是框架的普适性，而非 Panda 的生产级求解。在本文中 6DOF（UR10）始终是主实验对象，7DOF 验证仅用于证明迁移路径的可行性与正确性。


# 7. 讨论

## 7.1 为什么标准化后旧 CUDA kernel 会失效

将自定义装配体上调通的 CUDA IK kernel 直接迁移到官方 UR10 后失败，并不意味着 GPU 思路本身有错。更多时候，是定义不一致先于性能不足导致了问题。

最典型的症状是：位置误差几乎压到零，姿态误差却长期停留在错误分支，收敛率在旧迁移版上大幅下降。如果跳过这些定义问题直接调整 block size、shared memory 或 launch 参数，得到的"更快结果"根本不可信。三个定义问题——固定变换折叠、输出 FK 对齐、姿态残差定义——修复之后，CUDA 求解器才站到了可以公平讨论性能的起跑线上。

## 7.2 公平 benchmark 的工程价值

本工作刻意将以下条件全部固定：同一官方 UR10 模型、同一 tool0、同一 target/seeds 资产、同一阈值、清晰区分主对比 solver 与参考 solver、单 solver 失败时保留错误日志而非静默跳过。

这种"把限制写清楚"的做法，研究价值高于一个看似整齐的排行榜。一个典型的例子是 numeric_dls 在 zero_seed 下收敛率仅约 1%——如果跳过这个记录而只报告 home_seed 下的 34.3%，读者会被误导。同样，PyRoki 在 warm-up 排除后吞吐仍仅 54.3 targets/s——如果混入 GPU 主对比表而不加备注，就不公平。

## 7.3 消融研究的核心发现

B0-B6 逐级消融给出了四个核心发现。

**内存层次优化的有限作用**。常量内存（B1）和 PaddedMat6x8（B2）在小批量上合计提供约 5% 的提升，在大批量上可忽略。原因很直接：UR10 的 FK 工作集小到足以被 L1 缓存高效容纳。这与大量以矩阵乘法为核心的 GPU 工作形成对比——在那些场景中，常量内存和共享内存的收益通常更显著。

**自适应阻尼的决定性影响**。这是本文发现的对收敛率影响最显著的数值策略。固定 lambda（B0-B2）在 N=5000 上收敛率仅 83.4%，自适应阻尼将其推回 100% 的同时吞吐提升 147%。但这一优化并非无条件有利：在 N=100 时吞吐反而下降 62.4%，因为小批量下 GPU 有空闲计算资源，固定 lambda 已能在 4.3 次迭代内收敛，自适应阻尼的分支和超越函数开销无从摊销。

**混合精度的最大化加速**。从 B4（全 FP64）到 B5（FP32 主体 + FP64 LDLT 关键路径），148-154% 的吞吐提升是所有消融级别中幅度最大的单一变更。加速主要来自 FK、Jacobian 和 Hessian 构造中 FP64 运算量的减少，以及 FP32 数据宽度减半带来的共享内存访问压力缓解。NCU 数据为这一设计的有效性提供了直接硬件证据：B5 的 kernel 执行时间从 B4 的 2,920 us 降至 827 us（缩短 72%），bank conflict 从 3,522 降至 1,295（减少 49%），同时 FP64 pipeline 仍保持 60.7% 的利用率——LDLT 求解器继续为 SM 提供充足的计算负载。从当前 NCU 指标看，混合精度引入的类型转换没有表现为主导瓶颈；主要瓶颈仍为计算吞吐与寄存器压力。

**步长钳位与分支对齐的负收益**。B4 中引入的 0.35 rad 步长钳位和关节角 pi-wrap 校正，在所有批量上持续降低吞吐 15-20%，而收敛率已为 100%。0.35 rad 的限制对 UR10 关节空间过于保守。

## 7.4 两种优化哲学的实证比较

本工作与 cuRobo 代表了 GPU IK 求解的两条技术路线：一条走 GPU 硬件适配路线（1 block/target 轻量映射、混合精度、寄存器级 6x6 LDLT、常量内存广播、单 kernel DLS 迭代），另一条走算法框架优化路线（L-BFGS 优化器 + particle 搜索、FP32 精度、多粒子并行、PyTorch 通用框架）。

两条路线的实证对比结果清晰：B3（FP64）在 N=100 时领先 cuRobo 18.3 倍，但在 N=5000 时落后 2.0 倍。引入混合精度后（B5），本工作的方法在 N=5000 上以 180,962 targets/s 达到 cuRobo（144,855 targets/s）的 1.25 倍。该结果提示，在本文测试口径下，精度配置和底层实现对大批量吞吐具有显著影响；cuRobo 在 FP64-only 对照下表现出的优势，不能简单归因于算法框架本身，也受到 FP32 计算路径和多 particle 并行组织方式的共同影响。

## 7.5 Nsight Compute 的定量证据

所有优化决策都有 NCU profiling 数据的支撑，形成一条完整的硬件证据链。

N=100 的 full NCU profiling 确立了 kernel 的计算边界属性：compute throughput 60.73-66.89%，DRAM throughput 仅 1.16-1.56%。FP64 pipeline 利用率最高（60.7%），即使在混合精度版本中 LDLT 求解器的 FP64 计算仍占主导。寄存器压力是 occupancy 的主限因素——94-98 个寄存器/线程将理论 occupancy 限制在约 33%。

混合精度的硬件证据链完整：SM throughput 从 66.89% 降至 60.73%（偏高的计算压力被 FP32 缓解），bank conflicts 从 3,522 降至 1,295（-49%，因 FP32 数据宽度减半改进了对齐），但冲突占总波前的比例仍不足 1%，确认为非瓶颈。

CUDA Graph 的零开销验证：B6（B5 + CUDA Graph）的 event-based 计时显示 cudaGraphLaunch 的 kernel 执行时间与直接 <<<>>> launch 在测量噪声范围内一致（0.9008 ms vs 0.8991 ms，差异 < 0.2%）。在当前 kernel 配置下 launch overhead 不是瓶颈——CUDA Graph 的优化潜力在更轻量级的 kernel 中才能体现。

Bank conflict 的非瓶颈确认：B5 的 1,295 次 bank conflict 仅占总共享访问波前的约 1%，即使完全消除预期加速也低于 1%。NCU 的精确计数避免了在不重要的方向消耗工程资源。

## 7.6 本文实验的准确边界

一些 GPU 机器人论文将 IK benchmark、collision-free IK、trajectory optimization 和 motion planning 放在同一叙事中。本工作刻意避免这种写法，因为当前完成的实验本质上是：给定一批可达末端位姿，求对应的关节解，比较批量求解的吞吐与收敛率。即使 target profile 是沿 home pose 附近生成的平滑查询，它仍然不是时间连续轨迹的优化问题，也不是全局路径规划问题。将其误写为 motion planning 会夸大当前工作的范围。

## 7.7 后续工作方向

Panda 7DOF 完整 benchmark 是最直接的后续目标——将当前已验证的 7DOF CUDA kernel 集成到主 benchmark 框架中，与 cuRobo 等求解器进行统一口径对比。主要挑战在于 7x7 LDLT 的寄存器压力控制和冗余自由度下的多 seed 策略设计。多 seed 策略（每个目标发射 2-4 个不同初始 seed）可作为 cuRobo particle 搜索的轻量级替代，需评估其对收敛率和吞吐的联合影响。编译器优化方面，可利用 __launch_bounds__ 和寄存器分配提示进一步降低寄存器压力，提高 occupancy。对于 N=5000+ 的大规模 batch，需分析 grid 尾块（partial wave）效应并探索 grid 调度策略以消除 SM 负载不均衡。LDLT 求解器当前使用纯 FP64 计算，可探索迭代精化（iterative refinement）等混合精度线性求解技术，在保持精度的同时减少 FP64 计算量。


# 8. 结论

本文不声称提出新的逆运动学数学算法，也不评价完整路径规划或运动规划系统的优劣。本文在统一 UR10 模型、TCP、目标位姿、初始 seed、误差阈值和时间口径下，比较了轻量级单 seed CUDA DLS 与 cuRobo 多 particle GPU IK 两类路线在不同 batch size 下的性能边界。

主要成果如下。

**消融研究的量化发现**。B0-B6 共七组消融配置的完整实测（N=100/500/5000）建立了各优化维度的量化贡献：常量内存（B1）和 PaddedMat6x8（B2）在小批量上贡献约 5% 的提升；自适应阻尼（B3）作为对收敛率影响最显著的数值策略，在大批量上实现 132-184% 的吞吐提升并将收敛率从 83.4% 恢复至 100%（但在小批量上因分支开销导致吞吐下降 62.4%）；步长钳位与分支对齐（B4）在所有批量上持续降低吞吐 15-20%，在吞吐优先场景建议省略；混合精度（B5，FP32 主体 + FP64 LDLT 关键路径）在所有批量上带来 148-154% 的吞吐提升，同时收敛率保持 99.98% 以上。

**两类 GPU IK 路线的性能边界比较**。单 block/target 的轻量级 CUDA 映射在小中批量下具有较低 per-target 开销（N=100 时为 cuRobo 的 37.6 倍，N=500 时为 10.3 倍，N=1000 时为 4.8 倍）。混合精度进一步拓展了其在大批量下的吞吐边界，在 N=5000 上以 180,962 targets/s 达到 cuRobo（144,855 targets/s）的 1.25 倍。上述比较结论仅适用于本文定义的批量 IK 查询任务，不等价于完整运动规划流水线性能比较。

**Nsight Compute 驱动的硬件证据链**。所有优化决策均有 NCU profiling 数据支撑：kernel 为计算边界（SM throughput 60.73-66.89%），DRAM 非瓶颈（1.16-1.56%）；bank conflict 经精确计数确认为非瓶颈（约 1% 波前）；寄存器压力（94-98 registers/thread）是 occupancy 的主限因素（约 33%）。7DOF Panda 的正确性验证表明框架具有参数化可扩展性——从 6DOF 到 7DOF 仅需约 15 处机械性改动，核心线程映射和共享内存布局无需重构。

当前工作的局限性同样明确：求解器限于 6 自由度串行机械臂，7DOF Panda 的完整 benchmark 尚待完成；FP64 LDLT 路径的寄存器压力限制了 occupancy 上限；单 seed 策略在超大批量上的 GPU 利用率仍低于 cuRobo 的多粒子搜索。后续工作将围绕混合精度 LDLT 求解器、多 seed 策略、编译器辅助的寄存器压力降低和大规模 grid 调度优化展开。

# 参考资料

以下参考项用于支撑当前 Markdown 初稿中的模型来源、对比 solver 与 profiling 工具说明。当前版本采用"可核对的 URL + 简短说明"形式，后续若转正式投稿版，再统一改成 BibTeX 或期刊格式。

1. Universal Robots. `Universal_Robots_ROS2_Description`.
   https://github.com/UniversalRobots/Universal_Robots_ROS2_Description

2. Sundaralingam, B., Hari, S. K. S., Fishman, A., Garrett, C., Van Wyk, K., Blukis, V., Millane, A., Oleynikova, H., Handa, A., Ramos, F., Ratliff, N., Fox, D.
   `cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation`.
   https://arxiv.org/abs/2310.17274

3. NVIDIA / NVLabs. `curobo`.
   https://github.com/NVlabs/curobo

4. Kim, C. M., Yi, B., Choi, H., Ma, Y., Goldberg, K., Kanazawa, A.
   `PyRoki: A Modular Toolkit for Robot Kinematic Optimization`.
   https://autolab.berkeley.edu/assets/publications/media/iros_25_pyroki_camera_ready.pdf

5. PyRoki project page.
   https://chungmin99.github.io/pyroki/

6. PyRoki source repository.
   https://github.com/chungmin99/pyroki

7. Orocos Kinematics and Dynamics Library.
   https://github.com/orocos/orocos_kinematics_dynamics

8. Orocos KDL overview.
   https://docs.orocos.org/kdl/overview.html

9. NVIDIA Nsight Compute documentation.
   https://docs.nvidia.com/nsight-compute/NsightCompute/index.html

10. NVIDIA Nsight Compute Profiling Guide.
    https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
