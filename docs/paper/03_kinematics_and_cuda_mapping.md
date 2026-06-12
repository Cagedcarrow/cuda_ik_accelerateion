# 3. 运动学建模与 CUDA 并行映射

## 3.1 官方 UR10 运动学链

本文的主实验模型来自 Universal Robots 官方 ROS2 描述仓库，并在本地生成平铺 URDF。主链定义为：

- base: `base_link`
- tip / TCP: `tool0`
- active joints:
  - `shoulder_pan_joint`
  - `shoulder_lift_joint`
  - `elbow_joint`
  - `wrist_1_joint`
  - `wrist_2_joint`
  - `wrist_3_joint`

在标准化前，旧工程中的自定义 TCP 与固定装配体常量会显著影响 FK 与目标分布。标准化后，这些常量都改为由 URDF 自动导出。

## 3.2 URDF 提取与 FK 验证

工程中的 `robot_model.py` 负责：

- 解析 joint `axis`、`origin xyz/rpy`、joint limits；
- 自动处理主动关节前后的 fixed transform；
- 构造 CPU FK；
- 导出 CUDA 常量头文件。

对同一份 `ur10_official.urdf`，我们使用 `yourdfpy` 做了交叉验证。当前 CPU FK 与 `yourdfpy` 的最大绝对误差为 `3.331e-16`，说明模型提取链路已经达到机器精度级一致。

## 3.3 标准化 target / seeds 资产

本文的数据资产不是任意随机姿态，而是由同一模型先采样关节，再通过 FK 生成的可达目标位姿。当前主数据集特征为：

- 随机种子固定为 `42`；
- 主规模为 `N=100/500/1000/5000`；
- 主 seed 策略为 `zero_seed`；
- target profile 为 `smooth_joint_trajectory_from_home`；
- 工作空间约束：
  - `z in [0.20, 1.35] m`
  - `xy radius in [0.20, 1.25] m`
  - `x >= -0.15 m`

这样做的目的是让 benchmark 更接近真实工业中“围绕 home pose 变化生成的一批可达查询”，同时又用更苛刻、更统一的 `zero_seed` 作为主公平初值，而不是故意用极端分布压低收敛率。

## 3.4 CUDA 并行映射

当前 CUDA 主线采用：

- `1 block / target`
- `128 threads / block`
- `4 warp` 分工

其基本分工如下：

- Warp 0：前向运动学与误差更新
- Warp 1：数值 Jacobian
- Warp 2：Hessian / gradient 构造
- Warp 3：`6x6 LDLT` 解算与步长控制

这种映射的优势在于：

1. 每个 target 独立，天然适合 block 级并行；
2. 小矩阵计算可以落在共享内存和寄存器内完成；
3. CPU 侧不需要为每个目标单独 launch 多个 kernel。

## 3.5 Phase 6 中暴露的关键一致性问题

标准化迁移后，最重要的发现不是“老 kernel 变慢了”，而是以下三个问题会直接破坏收敛性：

1. **固定变换折叠缺失**：`base_link -> base_link_inertia` 这样的 fixed transform 如果被忽略，CPU/GPU 会在同错链路上“自洽”，却与官方模型不一致。
2. **输出 FK 与关节解不同步**：若最终输出关节向量与误差统计使用的 FK 不对应，会污染收敛判断。
3. **姿态残差与 Jacobian 不一致**：如果姿态误差定义和数值 Jacobian 的线性化路径不同，求解器可能把位置压到很小，同时把姿态推向错误分支。

这三个问题的修复，是后续 CUDA 性能追赶能够成立的前提。
