# CUDA 加速工业机器人逆运动学求解

**手写优化的 CUDA 批量 IK 求解器，小批量下吞吐量达到 cuRobo 的 36 倍，在 100–10,000 个目标的全部批量规模上实现近乎完美的 GPU 线性扩展。**

本项目证明：采用单 kernel 封装、Warp 级并行、混合精度（FP32+FP64）、寄存器驻留线性代数、共享内存 Bank 冲突消除和自适应阻尼等技术的手写 CUDA 实现，能够通过消除 kernel 启动开销、CPU-GPU 同步和框架调度成本，在性能上超越基于高层框架的 GPU IK 求解器。

---

## 1. 工程背景 —— 为什么需要批量逆运动学？

### IK 问题定义

六自由度串联工业机器人（如 UR10）需要将期望的末端执行器位姿（SE(3) 空间中的位置 + 姿态）转换为六个关节角度。这就是**逆运动学（Inverse Kinematics, IK）**问题。与具有唯一闭式解的正运动学不同，逆运动学具有以下难点：

- **非凸性**：一个 6R 机械臂最多存在 16 组解析解
- **奇异性**：在运动学奇异点附近，雅可比矩阵条件数急剧恶化
- **位置-姿态耦合**：姿态约束使解析方法难以普适

因此，工业实践通常采用**数值迭代方法**——最常用的是阻尼最小二乘法（Damped Least Squares, DLS）：

```
while not converged:
    T_cur ← FK(q)                              # 正运动学
    e ← pose_error(T_cur, T_tgt)               # 6维误差（3位置 + 3姿态）
    J ← numerical_jacobian(q, δ=1e-6)           # 6×6 雅可比（中心差分）
    H ← Jᵀ·W²·J + λ·I                          # 加权 Hessian 矩阵
    g ← Jᵀ·W²·e                                # 梯度
    dq ← LDLT_solve(H, g)                       # 6×6 线性方程组
    q ← clamp(q + dq, joint_limits)             # 施加步长与关节限位
    λ ← adaptive_damping(λ, pos_err, stagnation) # 更新阻尼
```

每次迭代需要 12 次正运动学计算（6 列 × ±ε 扰动）、一次 6×6 LDLT 分解和收敛判断——全部在双精度下完成以保证数值稳定性。

### 批量 IK 的工业意义

现代机器人应用在每个规划周期内需要求解**数千个 IK 问题**：

- **轨迹优化**：将笛卡尔空间路径点拟合为光滑的关节空间轨迹
- **料箱抓取**：评估数百万个抓取候选位姿
- **碰撞感知规划**：IK 作为采样规划器的内层循环
- **在线重规划**：50Hz 控制回路每个周期仅有 20ms 可用

基于 CPU 的 KDL 求解器处理 273 个目标需要约 6.2 秒——比实时需求慢 300 倍。GPU 加速不是可选项，而是实现交互式批量 IK 的唯一途径。

### UR10 机器人

<div align="center">

| 参数 | 数值 |
|:---|:---|
| 自由度 | 6（全旋转关节） |
| 工作半径 | 1,300 mm |
| 负载 | 10 kg |
| 关节轴线 | Z / Y / Y / Y / −Z / Y |
| IK 求解方式 | 纯数值（混合关节轴方向打破 Pieper 准则） |
| URDF 来源 | [UniversalRobots/Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description) tag 4.3.1 |

</div>

UR10 的混合关节轴方向（特别是 wrist_2 为 −Z 方向）使闭式 IK 无法适用；为标准 6R 手腕（满足 Pieper 准则）开发的解析求解器在此失效。这使其成为数值 GPU 方法的理想测试对象。

---

## 2. 项目结构

```
cuda_ik_accelerateion/
│
├── README.md                                   # ★ 本文件 —— 项目主页
├── .gitignore                                  # 排除规则（构建产物、IDE、profiling）
├── CMakeLists.txt                              # 根构建文件（旧版测试可执行文件）
│
├── standard_robot_cuda_ik/                     # ★★★ 主项目（第二代）
│   │                                           # 标准机器人，完整 benchmark 框架
│   ├── src/
│   │   ├── cuda/                               # CUDA 内核源码（3,194 行）
│   │   │   ├── cuda_ik_6dof.cu                 #   核心批量 IK 内核（1,529 行）
│   │   │   │                                   #   · 9 个消融级别 (A0–A8)
│   │   │   │                                   #   · 混合精度内核 (A7)
│   │   │   │                                   #   · CUDA Graph 内核 (A8)
│   │   │   ├── cuda_utilities.cuh              #   设备端函数（447 行）
│   │   │   │                                   #   · FK, Rodrigues, PaddedMat6x8, LDLT
│   │   │   │                                   #   · 常量内存声明
│   │   │   ├── cuda_benchmark_runner.cu        #   主机端驱动（483 行）
│   │   │   ├── cuda_collision.cu               #   GPU 碰撞检测（716 行）
│   │   │   └── cuda_memory.cu                  #   DeviceBuffer RAII（19 行）
│   │   └── cpu_baseline/                       # CPU 参考求解器
│   │       ├── kdl_solver.cpp                  #   Orocos KDL C++ 封装
│   │       └── numeric_dls_solver.cpp          #   Python DLS 基线
│   │
│   ├── include/standard_robot_cuda_ik/         # C++ 头文件
│   │   ├── cuda_ik_6dof.h                      #   内核启动 API
│   │   ├── cuda_collision.h
│   │   ├── cuda_memory.h
│   │   └── generated/ur10_model_constants.h    #   从 URDF 自动生成
│   │
│   ├── benchmark/                              # 多求解器对比框架
│   │   ├── run_all.py                          #   统一入口
│   │   ├── common.py                           #   共享框架（URDF 加载、收敛判定）
│   │   ├── bench_cuda_6dof.py                  #   CUDA B5 求解器封装
│   │   ├── bench_curobo.py                     #   cuRobo (NVIDIA) 封装
│   │   ├── bench_pyroki.py                     #   PyRoki (JAX) 封装
│   │   ├── bench_kdl.py                        #   KDL (C++ CPU) 封装
│   │   ├── bench_numeric_dls.py                #   Numeric DLS (Python CPU) 封装
│   │   └── compare_results.py                  #   跨求解器对比工具
│   │
│   ├── tools/                                  # 资产生成与验证
│   │   ├── generate_standard_assets.py         #   批量 target/seed 生成
│   │   ├── robot_model.py                      #   URDF 解析、FK、数据导出
│   │   ├── fetch_official_ur10.py              #   官方 URDF 下载
│   │   └── verify_official_ur10.py             #   模型验证
│   │
│   ├── config/                                 # 项目规格配置（纳入版本管理）
│   │   ├── benchmark.yaml                      #   求解器列表、批量规模、容差
│   │   ├── robots.yaml                         #   机器人模型定义
│   │   └── target_generation.yaml              #   目标/种子生成参数
│   │
│   ├── urdf/                                   # 官方机器人模型
│   │   ├── ur10_official.urdf                  #   UR10（UniversalRobots 官方）
│   │   ├── ur5_official.urdf                   #   UR5
│   │   └── panda_7dof.urdf                     #   Franka Panda 7-DOF
│   │
│   ├── data/                                   # ★ 全部实验数据（184 个文件, 227MB）
│   │   ├── README.md                           #   数据-论文映射索引
│   │   ├── targets/                            #   36 个 IK 目标文件（N=100→10000）
│   │   ├── seeds/                              #   96 个种子文件（4 策略 × 12 规模）
│   │   ├── results/                            #   Benchmark 结果（11 个 CSV）
│   │   │   ├── main_comparison/                #     B5 vs cuRobo @ Medium 10mm/5°
│   │   │   ├── ablation/                       #     B0/B3/B5 消融实验
│   │   │   ├── threshold_scan/                 #     三档阈值扫描
│   │   │   ├── full_range/                     #     N=100→10000 线性度分析
│   │   │   ├── cpu_baseline/                   #     CPU 参考基线 (KDL/numeric_dls)
│   │   │   ├── seed_strategy/                  #     zero_seed vs home_seed
│   │   │   └── panda_7dof/                     #     7-DOF Panda 验证
│   │   ├── profiling/                          #   Nsight Compute GPU 性能剖析
│   │   │   ├── ncu_summary.csv                 #     关键指标汇总
│   │   │   └── ncu_reports/                    #     5 个 .ncu-rep 原始报告
│   │   └── figures/                            #   5 张论文图表 + 生成脚本
│   │
│   ├── experiments/                            # 实验工作区
│   │   └── 7dof_test/                          #   Panda 7-DOF CUDA IK 扩展
│   │
│   └── CMakeLists.txt                          # 构建系统：10 个消融级别可执行文件
│
├── cuda_low_level_optimization/                # 旧版第一代（自定义 UR10+铲斗）
│   ├── src/                                    #   CUDA 内核（2,751 行）
│   ├── test/                                   #   测试程序 + 273 目标数据集
│   └── docs/                                   #   27 篇技术参考文档
│
├── docs/                                       # 中央文档
│   ├── PROJECT_OVERVIEW.md                     #   完整项目概览
│   ├── paper/                                  #   论文草稿（8 章, Markdown）
│   ├── logs/                                   #   实验日志（10 个文件）
│   ├── patent/                                 #   专利技术交底书
│   └── 修改意见/                                #   审稿修改意见（10 个文件）
│
└── benchmark/          [gitignored]            # 外部求解器克隆（约 456MB）
    ├── curobo/                                 #   NVIDIA cuRobo (PyTorch GPU IK)
    ├── hjcd_ik/                                #   HJCD-IK (PyTorch GPU IK)
    └── pyroki/                                 #   PyRoki (JAX GPU IK)
```

---

## 3. 快速开始

### 环境要求

- **NVIDIA GPU**，计算能力 ≥ sm_89（Ada Lovelace 架构：RTX 4060/4070/4080/4090）
- **CUDA Toolkit** ≥ 13.3
- **C++17** 编译器（GCC ≥ 11 或 Clang ≥ 14）
- **CMake** ≥ 3.22
- **Python** ≥ 3.10（benchmark 需要 matplotlib, numpy）

### 编译

```bash
cd standard_robot_cuda_ik
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc)
```

编译生成 10 个可执行文件：

| 可执行文件 | 消融级别 | 包含的优化 |
|:---|:---|:---|
| `standard_robot_cuda_runner_A0` | B0（基线） | 全局内存，无填充，固定阻尼 |
| `standard_robot_cuda_runner_A1` | B1 | + 常量内存 |
| `standard_robot_cuda_runner_A2` | B2 | + PaddedMat6x8（Bank 冲突消除） |
| `standard_robot_cuda_runner_A3` | — | + 寄存器 LDLT |
| `standard_robot_cuda_runner_A4` | — | + Kernel 融合 |
| `standard_robot_cuda_runner_A5` | B3 | + 自适应阻尼 ★ |
| `standard_robot_cuda_runner_A6` | B4 | 完整 FP64：A5 + 步长钳位 + 分支对齐 |
| `standard_robot_cuda_runner_A7` | B5 | **混合精度 (FP32+FP64)** ★★ |
| `standard_robot_cuda_runner_A8` | B6 | A7 + CUDA Graph |
| `standard_robot_cuda_runner` | B6 别名 | 默认（链接到 A8） |

### 运行 Benchmark

```bash
# 单求解器 benchmark
cd standard_robot_cuda_ik
./build/standard_robot_cuda_runner_A7 \
    --targets data/targets/ur10_seed42_N1000.csv \
    --seeds data/seeds/ur10_seed42_zero_seed_N1000.json \
    --repeat 30

# 全部求解器完整对比
python3 benchmark/run_all.py \
    --robot ur10 --seed 42 --N 1000 --repeat 30
```

---

## 4. 关键性能结果

所有结果均采用 **Medium 收敛阈值**（位置 10mm、姿态 5°）作为主基准，repeat=30，zero_seed 策略，测试平台为 NVIDIA GeForce RTX 4070 Laptop GPU。

### 4.1 B5 vs. cuRobo —— 主对比

| 批量规模 N | B5 吞吐量 | cuRobo 吞吐量 | B5 加速比 | B5 GPU 时间 | cuRobo GPU 时间 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | **112,414 t/s** | 3,118 t/s | **36.1×** | 0.89 ms | 32.1 ms |
| 500 | **158,251 t/s** | 15,844 t/s | **10.0×** | 3.16 ms | 31.6 ms |
| 1000 | **148,412 t/s** | 31,611 t/s | **4.7×** | 6.74 ms | 31.6 ms |
| 5000 | **168,683 t/s** | 155,059 t/s | **1.09×** | 29.64 ms | 32.2 ms |

**关键发现：**

- **小批量绝对优势**：N=100 时，B5 吞吐量为 cuRobo 的 36 倍。这对交互式应用（如 100 个路径点的实时轨迹优化）至关重要。
- **GPU 线性扩展**：B5 的 GPU 时间与批量规模的线性拟合优度达到 **R² > 0.999**——单目标求解成本几乎恒定，全批量范围内吞吐量稳定在 148k–174k t/s（±8%）。
- **cuRobo 批量震荡**：在 N=4000、7000、9000、10000 时，cuRobo 间歇性进入约 230ms/batch 的退化模式（正常模式的 7 倍），可能机制为 PyTorch CUDA Caching Allocator 的内存碎片化——B5 的固定单目标内存模型在结构上完全避免了这一问题。

### 4.2 消融实验 —— 逐级优化的贡献

| 级别 | N=100 | N=500 | N=5000 | 核心优化 |
|:---:|:---:|:---:|:---:|:---|
| B0（FP64 基线） | 7,589 t/s（收敛率 83%） | 9,896 t/s（收敛率 52%） | 12,507 t/s（收敛率 56%） | 无——全局内存，无填充，固定阻尼 |
| B3（+自适应阻尼） | 51,361 t/s（**6.8×**） | 62,384 t/s（**6.3×**） | 66,050 t/s（**5.3×**） | 分段线性距离比例阻尼 + 停滞恢复 |
| B5（+混合精度） | 113,097 t/s（**14.9×**） | 155,071 t/s（**15.7×**） | 164,207 t/s（**13.1×**） | FP32 FK/Jacobian/Hessian + FP64 LDLT |

**核心洞察**：无自适应阻尼时，B0 收敛率崩塌至 52–83%。引入自适应阻尼（B3）后，收敛率恢复至 100%，吞吐量提升 4–6 倍。混合精度（B5）在此基础上再提升 2.2–2.5 倍，充分利用了 Ada Lovelace 架构 64:1 的 FP32:FP64 吞吐比。

---

## 5. CUDA Kernel 设计

### 5.1 Block-Warp-Thread 映射

```
Grid:  (N, 1, 1)     ← 每个目标位姿一个 Block
Block: (128, 1, 1)   ← 4 个 Warp × 32 个 Lane
```

每个 Block 独立求解一个目标位姿的 IK 问题。4 个 Warp 在每次 DLS 迭代中组成流水线计算：

```
Warp 0 (Lane 0–31)     ──► 正运动学 (FK)
Warp 1 (Lane 32–63)    ──► 数值雅可比 (6 列并行)
Warp 2 (Lane 64–95)    ──► Hessian 构造 (JᵀW²J + λI)
Warp 3 (Lane 96–127)   ──► LDLᵀ 求解 + 收敛判断
```

| Warp | 任务 | 活跃 Lane 数 | 关键操作 |
|:---:|:---|:---:|:---|
| 0 | FK 链式计算 | 1 | 6 段 Rodrigues 乘积，位姿误差 |
| 1 | 雅可比矩阵 | 6 | 12 次 FK 调用（中心差分，δ=1e-6 rad） |
| 2 | Hessian + 梯度 | 36 | 每个线程计算 JᵀW²J 的一个 (r,c) 元素 |
| 3 | LDLᵀ + 收敛 | 6 | 自定义 6×6 LDLᵀ（约 93 次 FP64 运算），步长钳位（0.45 rad），停滞检测 |

每次迭代包含 **14 个 `__syncthreads()` 屏障**以确保 Warp 间数据一致性。Nsight Compute 分析显示 **72.7% 的 Warp 停滞周期为屏障等待**——最快的 Warp 必须在每个同步点等待最慢的 Warp。这是首要的性能瓶颈，而非计算或显存带宽。

### 5.2 PaddedMat6x8 —— 共享内存 Bank 冲突消除

所有中间矩阵（雅可比、Hessian、变换矩阵）都采用 **6×8 列填充布局**存储在共享内存中：

```cpp
struct PaddedMat6x8 {
    double data[6 * 8];  // 48 个 double = 384 字节（密集布局仅需 36 个）
    __device__ double& operator()(int row, int col) {
        return data[row * 8 + col];  // stride=8，零指令开销
    }
};
```

**原理**：Ada Lovelace 共享内存具有 32 个 Bank，每个 Bank 4 字节。一个 `double`（8 字节）跨越 2 个连续 Bank。采用 stride=8（每行 16 个 Bank），每行映射到模 32 意义下不重叠的 Bank 集合——第 0 行和第 2 行共享 Bank 0–15，第 1 行和第 3 行共享 Bank 16–31。这保证了**所有访问模式下零 Bank 冲突**。

**Nsight Compute 验证**：`l1tex__data_bank_conflicts_shared_pipe_lsu.sum = 0`。禁用填充（消融级别 0，stride=6）后，内存吞吐量上升 151%，L2 缓存命中率下降 8.22 个百分点，总吞吐量下降约 2%。

### 5.3 混合精度（B5）—— FP32 + FP64 混合策略

Ada Lovelace 架构的 FP64:FP32 吞吐比为 **1:64**：RTX 4070 每个 SM 仅有 48 个 FP64 核心，而 FP32 核心多达 3,072 个。混合精度将约 90% 的算术运算移至 FP32，同时在关键路径上保留双精度：

| 组件 | 精度 | 理由 |
|:---|:---:|:---|
| 正运动学 (FK) | **FP32** | 占总运算量约 90%；吞吐量为 FP64 的 64 倍 |
| 数值雅可比（6 列 × 2 次 FK） | **FP32** | 占据迭代时间的绝大部分 |
| Hessian JᵀW²J（36 个元素） | **FP32** | 每次迭代 216 次乘加 |
| 位姿误差 | FP64 | 收敛容差判定的精度关键（10mm/5°） |
| 关节状态 q, dq | FP64 | 关节限位执行必须精确 |
| LDLᵀ 分解 | **FP64** | 线性方程组求解的数值稳定性 |
| 阻尼参数 λ | FP64 | 防止自适应阻尼中的下溢 |

**关键转换点**在 LDLᵀ 组装阶段：`(double)H(r,c)` 将 FP32 Hessian 转换为 FP64 后再进入求解。所有下游计算（关节更新、收敛检测）保持 FP64。这使 B5 相对 B4（全 FP64）实现了约 2.5 倍吞吐量提升，且无显著收敛退化（收敛率：1.000 → 0.998）。

### 5.4 自适应阻尼

固定阻尼（B0）：`λ ≡ 2e-3`——或过于激进（导致震荡）或过于保守（收敛缓慢）。自适应阻尼（B3+）采用分段线性距离比例调度：

```
若 pos_err > 0.1（远距离区）：
    λ = min( max(λ_base, λ_far × pos_err / λ_scale), 3 × λ_far )
否则（近距离区）：
    λ = λ_floor + λ_base × pos_err / λ_scale

停滞增强（连续 ≥5 次迭代无改善）：
    λ *= 1.0 + 0.3 × (stagnation - 5)
    λ = min(λ, 0.5)  // 硬上限
```

远离解时施加强阻尼（防止奇异性下发散），接近解时施加轻阻尼（实现快速终端收敛）。停滞恢复逐步增大阻尼以逃离局部极小值，硬上限为 0.5。若停滞持续至 25 次迭代，求解器恢复最佳已知解并终止。

**效果**：仅自适应阻尼一项（B0 → B3）即通过将平均迭代次数从约 31–80 次降至约 13–15 次，**实现 4–6 倍吞吐量提升**，并将收敛率从 52–83% 恢复至 100%。

### 5.5 常量内存广播

1,384 字节的运动学参数存储在 `__constant__` 内存中，向所有线程广播：

| 符号 | 大小 | 内容 |
|:---|:---:|:---|
| `c_segment_origins[96]` | 768 B | 6 个连杆 × 4×4 原点矩阵 |
| `c_segment_axes[18]` | 144 B | 6 个旋转轴方向 |
| `c_q_index[6]` | 24 B | 关节→连杆索引映射 |
| `c_T_wrist3_to_tcp[16]` | 128 B | TCP 工具变换 |
| `c_joint_limits[12]` | 96 B | 关节限位 |
| `c_weight_schedule[24]` | 192 B | 4 组权重 × 6 自由度 |
| `c_lambda_params[4]` | 32 B | 自适应阻尼参数 |

Warp 内所有线程同时读取同一运动学参数——常量内存广播延迟约 1 个周期（全局内存约 400 个周期）。若无常量内存，运动学查找将在每 273 目标的批次中产生约 5.2 GB 的额外全局内存流量。

### 5.6 零寄存器溢出

`ik_batch_solve` 内核使用 **每线程 96 个寄存器**，**零字节局部内存溢出**，已通过 `ptxas` 详细输出和 Nsight Compute 验证。所有中间变量——FK 结果（8 寄存器）、雅可比列（6）、Hessian 累加器（4）、误差/梯度/步长向量（18）、迭代状态（10）以及设备函数局部变量——完全容纳在寄存器文件中。

128 线程/Block × 96 寄存器 = 12,288 寄存器/Block，每个 SM 可驻留 5 个并发 Block（Warp 槽位占用率 41.7%）。对于此**计算密集型**内核（算术强度约 193 FLOP/Byte，Ridge Point 0.93 FLOP/Byte，约 207 倍于 Ridge Point），编译器最大化寄存器分配（零溢出）的决策比减少寄存器数以提升占用率更有利。

### 5.7 自定义 6×6 LDLᵀ —— 不依赖 cuBLAS

通过 cuBLAS 求解 6×6 线性方程组需要：
- cuBLAS 句柄创建和上下文切换（约 5–10 μs 启动开销）
- 数据搬运：共享内存 → 全局内存 → cuBLAS → 共享内存
- cuBLAS 内部分发——对如此小的问题而言开销远大于计算本身

相比之下，手写的寄存器驻留 LDLᵀ 分解在 **约 93 次 FP64 运算（约 0.1 μs）** 内完成 6×6 求解，零内存流量：

1. **分解**（57 次运算）：从 H 的上三角就地计算 L 和 D
2. **前向代入**（15 次运算）：求解 L·y = g
3. **对角缩放**（6 次运算）：z = D⁻¹·y
4. **后向代入**（15 次运算）：求解 Lᵀ·dq = z

无需 `sqrt`（区别于 Cholesky LLᵀ），在 H 近似奇异时提供更好的数值稳定性。

### 5.8 单 Kernel 封装

所有 DLS 迭代在**一次 CUDA kernel 启动**内完成。CPU 提交批次、调用 `cudaDeviceSynchronize()` 后直接获取结果——迭代过程中无 CPU-GPU 往返。这消除了：

- Kernel 启动开销：消费级 GPU 上每次约 5–10 μs，× 7 次迭代 = 每目标 35–70 μs
- CUDA API 分发延迟
- 启动间的设备同步开销

对 cuRobo 和 PyRoki 而言，等价的执行方式将是每次迭代数百甚至数千次 kernel 启动，每次启动还伴随 Python→CUDA 绑定开销和 GPU 流同步。

---

## 6. CUDA 框架优势 —— 结构性对比

### 6.1 手写 CUDA 为何超越框架型求解器

| 维度 | CUDA B5（本项目） | cuRobo (NVIDIA) | PyRoki (JAX) |
|:---|:---|:---|:---|
| **每批次 kernel 启动数** | **1** | 数百（大量小 kernel） | 数千（JIT + XLA） |
| **CPU-GPU 同步点** | **0**（仅在结束时同步一次） | 大量（每次 kernel 启动） | 大量（JAX 分发） |
| **框架层** | **无**（原始 CUDA C++） | cuda.core + Warp + PyTorch | JAX JIT + XLA + Python |
| **精度模型** | 混合 FP32+FP64 | FP32 | FP64 |
| **矩阵求解** | 自定义寄存器 LDLᵀ（约 0.1 μs） | cuBLAS（库调用，约 5 μs） | JAX 线性代数 |
| **共享内存优化** | PaddedMat6x8（零 Bank 冲突） | 未公开 | 不适用（XLA 管理） |
| **迭代封装** | 单 kernel | 多 kernel | 多 kernel |
| **内存分配** | 固定每目标分配（cudaMalloc） | PyTorch CUDA Caching Allocator | JAX 内存池 |
| **单目标成本扩展** | 线性（R² > 0.999） | 震荡（R² 约 0.7） | 未测量 |

### 6.2 性能差距的结构性原因

**小批量下 kernel 启动次数占主导。** N=100 时，cuRobo 每批次启动数百个 kernel，而 B5 仅启动 1 个。消费级 GPU 驱动栈上每次 kernel 启动约 5–10 μs。cuRobo 正常 32ms 的批次时间中，仅启动开销即占约 10%。

**框架开销是固定税。** cuRobo 的 Python→C++→CUDA 调用链和 PyTorch 张量管理每批次增加约 5–15ms，与批量规模无关。N=5000 时尚可分摊；N=100 时则占主导——B5 的 0.89ms 中不含任何框架税。

**小矩阵场景自定义 LDLᵀ 优于库调用。** 6×6 LDLᵀ 是一个极小的问题。cuBLAS 的内部分发、句柄管理和内存搬运开销超过实际求解时间的 50 倍。零内存流量的寄存器驻留 LDLᵀ 是小于约 16×16 矩阵的最优策略。

**固定内存模型避免分配器病变。** B5 通过 `cudaMalloc` 一次性分配 `N × sizeof(Target) + N × sizeof(Result)` 大小。cuRobo 依赖 PyTorch 的 CUDA Caching Allocator，该分配器在重复分配中产生碎片，间歇触发昂贵的碎片整理过程。这是 cuRobo 观察到的约 230ms 退化模式的可能机制——分配器在特定批量下进入病变状态，造成 7 倍减速。

### 6.3 cuRobo 批量震荡现象

在 N=4000、7000、9000、10000 时，cuRobo 间歇性进入**约 230ms/batch 的退化模式**（正常模式约 32ms/batch）：

| N | 正常模式 | 退化模式 | 发生频率 |
|:---:|:---:|:---:|:---:|
| 4000 | 约 32 ms | 约 230 ms | 间歇性 |
| 5000 | 约 32 ms | — | 正常 |
| 7000 | 约 32 ms | 约 230 ms | 间歇性 |
| 8000 | 约 32 ms | — | 正常 |
| 10000 | 约 32 ms | 约 230 ms | 间歇性 |

非单调模式（N=5000 正常但 4000 退化，8000 正常但 7000 退化）是分配器碎片的典型特征——非计算或带宽瓶颈。B5 的固定每目标内存模型在结构上对此失效模式免疫。

---

## 7. 实验方法

### 7.1 收敛阈值体系

| 级别 | 位置容差 | 姿态容差 | 用途 |
|:---|:---|:---|:---|
| **Loose（宽松）** | 30 mm | 10° (0.1745 rad) | cuRobo 默认阈值；早期实验 |
| **Medium（中等）** ★ | **10 mm** | **5° (0.0873 rad)** | **主基准** |
| **Strict（严格）** | 5 mm | 1° (0.0175 rad) | 精密应用场景 |

Medium 阈值是主基准，因为该阈值能显著区分求解器质量：B0 的收敛缺陷（Medium 下 52–83%，vs. Loose 下 80–100%）在旧 30mm/10° 阈值下无法显现。

### 7.2 消融级别（B 系列命名）

| 论文名称 | 可执行文件 | 优化内容 | 关键指标 |
|:---|:---|:---|:---|
| B0 | A0 | FP64 基线（无优化） | 收敛率：52–83% |
| B1 | A1 | + 常量内存 | 吞吐量 +3% |
| B2 | A2 | + PaddedMat6x8 | 零 Bank 冲突 |
| — | A3 | + 寄存器 LDLᵀ | — |
| — | A4 | + Kernel 融合 | — |
| B3 ★ | A5 | + 自适应阻尼 | **吞吐量 +400–600%** |
| B4 | A6 | 完整 FP64（A5 + 步长钳位 + 分支对齐） | 参考基线 |
| B5 ★★ | A7 | + 混合精度（FP32+FP64） | **吞吐量 +220–250%** |
| B6 | A8 | + CUDA Graph | 边际收益（约 3.7%） |

### 7.3 可复现性保障

- **目标生成**：确定性 PRNG（seed=42），关节角在 [−π, π] 内均匀采样，经 FK 验证可到达性
- **种子策略**：`zero_seed`（全零向量，默认）、`home_seed`（UR10 零位构型）、`random_seed`（均匀随机）、`near_ground_truth_seed`（真值 + 0.25 rad 噪声）
- **重复次数**：每组配置独立运行 30 次；吞吐量报告均值
- **计时口径**：GPU 端到端时间（kernel 启动前后各一次 `cudaDeviceSynchronize`，排除主机端目标/种子准备时间）
- **硬件平台**：NVIDIA GeForce RTX 4070 Laptop GPU（4,608 CUDA 核心, 8 GB GDDR6），CUDA 13.3

---

## 8. 数据与图表

所有实验数据集中存放于 [`standard_robot_cuda_ik/data/`](standard_robot_cuda_ik/data/)。完整的数据-论文映射表、字段说明和实验配置参见 [data README](standard_robot_cuda_ik/data/README.md)。

| 数据集 | 文件数 | 内容 |
|:---|:---:|:---|
| Targets | 36 | UR10 seed=42 末端位姿，N=100→10000，.bin/.csv/.json |
| Seeds | 96 | 4 种策略 × 12 个批量规模 × 2 种格式（.bin/.json） |
| Results | 11 个 CSV | 主对比、消融、阈值扫描、全量程、CPU 基线、种子策略、Panda 7-DOF |
| Profiling | 5 个 .ncu-rep | Nsight Compute 报告：B4 FP64 N=100（×3）、B3 FP64 N=5000、B5 混合精度 N=100 |
| Figures | 5 张 PNG | 吞吐量对比、加速比、消融、收敛率、平均迭代次数 |

重新生成图表：
```bash
cd standard_robot_cuda_ik/data/figures
python3 plot_all_figures.py
```

---

## 9. 论文与文档

- **论文草稿**：[`docs/paper/`](docs/paper/) —— 8 章完整论文，Markdown 格式
- **专利交底书**：[`docs/patent/`](docs/patent/) —— 完整技术交底文档
- **实验日志**：[`docs/logs/`](docs/logs/) —— 10 份权威实验报告
- **项目概览**：[`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md)
- **CUDA 技术文档**：[`cuda_low_level_optimization/docs/`](cuda_low_level_optimization/docs/) —— 27 篇深度设计文档，涵盖内存层次、kernel 执行模型、性能分析和 GPU 碰撞检测

---

## 10. 添加你自己的求解器

Benchmark 框架专为公平的跨求解器对比设计。添加新求解器的步骤：

1. 参照 [`bench_cuda_6dof.py`](standard_robot_cuda_ik/benchmark/bench_cuda_6dof.py) 的模式创建 `bench_your_solver.py`
2. 从 `data/targets/` 和 `data/seeds/` 加载目标和种子数据
3. 使用 `common.load_urdf()` 和 `common.check_convergence()` 确保收敛判定一致性
4. 按照标准 JSON 模式输出结果（参见 [`compare_results.py`](standard_robot_cuda_ik/benchmark/compare_results.py)）
5. 运行 `run_all.py --solver your_solver` 进行统一对比

所有求解器使用相同的目标、种子、收敛阈值和重复次数进行评估——确保公平的同类对比。

---

## 11. 旧版项目 —— 第一代概念验证

[`cuda_low_level_optimization/`](cuda_low_level_optimization/) 目录包含原始的自定义 UR10 + 铲斗装配体 CUDA IK 求解器（非标准机器人）。这是验证单 kernel 方案可行性的概念验证，实现了相对 CPU KDL 约 **960 倍加速**（273 个目标：CPU 6.2s → GPU 6.4ms）。

其核心 CUDA 技术（Warp 并行、PaddedMat6x8、自适应阻尼、寄存器 LDLᵀ）已迁移并泛化至 `standard_robot_cuda_ik/`。旧版代码留存备查，其 27 篇技术文档仍是 CUDA 内核内部实现最详尽的参考资料。

---

## 12. 许可证

本项目用于研究和基准测试目的。源自 `assembly_rtfg_cuda` 的源代码保留其原始版权。

---

*最后更新：2026-06-12 | 目标 GPU：NVIDIA Ada Lovelace (sm_89) | CUDA 13.3*
