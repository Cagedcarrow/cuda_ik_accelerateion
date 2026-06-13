# CUDA 加速工业机器人逆运动学求解

**手写优化的 CUDA 批量 IK 求解器，小批量下吞吐量达到 cuRobo 的 36 倍，在 100–10,000 个目标的全部批量规模上实现近乎完美的 GPU 线性扩展。**

本项目证明：采用单 kernel 封装、阶段式 flat threadIdx.x 并行分工、混合精度（FP32+FP64）、寄存器驻留 LDLᵀ 求解、PaddedMat6x8 共享内存 Bank 冲突降低和自适应阻尼等技术的手写 CUDA 实现，能够通过消除 kernel 启动开销、CPU-GPU 同步和框架调度成本，在吞吐量上显著超越基于高层框架的 GPU IK 求解器。

> **权威参考：** 本文档的所有性能数字、架构参数和代码事实均以 [`docs/项目总览/`](docs/项目总览/) 中的 5 份参考文档为准。源码（`standard_robot_cuda_ik/src/cuda/cuda_ik_6dof.cu`）是第一事实。

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
    J ← ∂x/∂q  (numerical, ε=1e-6)             # 6×6 雅可比（FK 中心差分）
    H ← Jᵀ·W²·J + λ·I                          # 加权 Hessian 矩阵（λ 直接加对角线）
    g ← Jᵀ·W²·e                                # 梯度
    dq ← LDLT_solve(H, g)                       # 6×6 线性方程组
    q ← clamp(q + dq, joint_limits)             # 施加步长与关节限位
    λ ← adaptive_damping(λ, pos_err, stagnation) # 更新阻尼
```

每次迭代需要 12 次正运动学计算（6 列 × ±ε 扰动）、一次 6×6 LDLᵀ 分解和收敛判断。

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
│
├── standard_robot_cuda_ik/                     # ★★★ 主项目（第二代）
│   │                                           # 标准机器人，完整 benchmark 框架
│   ├── src/
│   │   ├── cuda/                               # CUDA 内核源码
│   │   │   ├── cuda_ik_6dof.cu                 #   核心批量 IK 内核（~1,530 行）
│   │   │   │                                   #   · 5 个 GPU kernel
│   │   │   │                                   #   · 9 个消融级别 (A0–A8)
│   │   │   │                                   #   · 混合精度内核 (A7, B5)
│   │   │   │                                   #   · CUDA Graph 内核 (A8, B6)
│   │   │   ├── cuda_utilities.cuh              #   设备端函数（~448 行）
│   │   │   │                                   #   · FK, Rodrigues, PaddedMat6x8, LDLT
│   │   │   │                                   #   · 常量内存声明
│   │   │   ├── cuda_benchmark_runner.cu        #   主机端驱动（~483 行）
│   │   │   ├── cuda_collision.cu               #   GPU 碰撞检测（716 行，论文未使用）
│   │   │   └── cuda_memory.cu                  #   DeviceBuffer RAII
│   │   └── cpu_baseline/                       # CPU 参考求解器
│   │       ├── kdl_solver.cpp                  #   Orocos KDL C++ 封装
│   │       └── numeric_dls_solver.cpp          #   Python/NumPy DLS 基线
│   │
│   ├── include/standard_robot_cuda_ik/         # C++ 头文件
│   │   ├── cuda_ik_6dof.h                      #   内核启动 API
│   │   ├── cuda_collision.h
│   │   ├── cuda_memory.h
│   │   └── generated/ur10_model_constants.h    #   从 URDF 自动生成
│   │
│   ├── benchmark/                              # 多求解器对比框架
│   │   ├── run_all.py                          #   统一入口
│   │   ├── common.py                           #   共享框架（收敛判定、阈值定义）
│   │   ├── bench_cuda_6dof.py                  #   CUDA B5 求解器封装
│   │   ├── bench_curobo.py                     #   cuRobo (NVIDIA) 封装
│   │   ├── bench_pyroki.py                     #   PyRoki (JAX) 封装
│   │   ├── bench_kdl.py                        #   KDL (C++ CPU) 封装
│   │   ├── bench_numeric_dls.py                #   Numeric DLS (Python CPU) 封装
│   │   ├── compare_paper_data.py               #   论文数据一致性检查
│   │   └── run_b5_ablation.sh                  #   消融批量执行脚本
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
│   ├── data/                                   # ★ 全部实验数据（184 个文件, ~227MB）
│   │   ├── targets/                            #   36 个 IK 目标文件（N=100→10000）
│   │   ├── seeds/                              #   96 个种子文件（4 策略 × 12 规模）
│   │   ├── results/                            #   Benchmark 结果
│   │   │   ├── main_comparison/                #     B5 vs cuRobo @ Medium 10mm/5°
│   │   │   ├── ablation/                       #     B0/B3/B5 消融实验
│   │   │   ├── threshold_scan/                 #     三档阈值扫描
│   │   │   ├── full_range/                     #     N=100→10000 线性度分析
│   │   │   ├── cpu_baseline/                   #     CPU 参考基线
│   │   │   ├── seed_strategy/                  #     zero_seed vs home_seed
│   │   │   └── panda_7dof/                     #     7-DOF Panda 验证
│   │   ├── profiling/                          #   Nsight Compute GPU 性能剖析
│   │   │   ├── ncu_summary.csv                 #     关键指标汇总
│   │   │   └── ncu_reports/                    #     5 个 .ncu-rep 原始报告
│   │   └── figures/                            #   论文图表 + 生成脚本
│   │
│   ├── experiments/                            # 实验工作区
│   │   └── 7dof_test/                          #   Panda 7-DOF CUDA IK 扩展
│   │
│   └── CMakeLists.txt                          # 构建系统：10 个消融级别可执行文件
│
└── docs/                                       # 中央文档
    ├── 项目总览/                                #   ★★★ 权威参考标准（5 份文档）
    │   ├── README.md                           #     总索引 + 快速查找表
    │   ├── 01-实验数据目录.md                    #     实验数据完整地图
    │   ├── 02-源代码目录.md                      #     源代码完整地图
    │   ├── 03-论文贡献与核心声明.md               #     贡献、创新点、性能声明、黑名单
    │   └── 04-GPU架构与CUDA实现细节.md           #     GPU 架构参数、CUDA 特性、kernel 设计
    ├── paper/                                  #   论文（paper_complete.md 为最新版）
    ├── 专利/                                   #   专利技术交底书 + 6 张附图描述
    ├── data/                                   #   论文图表 CSV 数据副本
    ├── logs/                                   #   当前项目实验日志
    ├── 优势/                                   #   CUDA vs cuRobo 结构化对比分析
    └── goal.txt                               #   项目需求规格文档
```

---

## 3. 快速开始

### 环境要求

| 组件 | 规格 |
|:---|:---|
| **GPU** | NVIDIA GeForce RTX 4060 Laptop GPU (Ada Lovelace, sm_89) |
| **CUDA Toolkit** | 12.6 |
| **Host 编译器** | GCC 11.4.0 |
| **C++ 标准** | C++17 |
| **CMake** | ≥ 3.22 |
| **Python** | ≥ 3.10（benchmark 需要 matplotlib, numpy） |

> **注意：** 本项目针对 sm_89 (Ada Lovelace) 编译和测试。其他计算能力版本可能兼容，但未经验证。

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
| `standard_robot_cuda_runner_A2` | B2 | + PaddedMat6x8（Bank 冲突降低） |
| `standard_robot_cuda_runner_A3` | — | + 寄存器 LDLᵀ |
| `standard_robot_cuda_runner_A4` | — | + Kernel 融合 |
| `standard_robot_cuda_runner_A5` | B3 ★ | + 自适应阻尼（收敛率关键驱动） |
| `standard_robot_cuda_runner_A6` | B4 | A5 + 步长钳位 + 分支对齐（已知负收益，-15%~-20%） |
| `standard_robot_cuda_runner_A7` | B5 ★★ | **混合精度（FP32+FP64）**，主配置 |
| `standard_robot_cuda_runner_A8` | B6 | A7 + CUDA Graph（边际收益 +3.7%） |
| `standard_robot_cuda_runner` | B6 别名 | 默认（链接到 A8） |

### 运行 Benchmark

```bash
# 单求解器 benchmark（B5 主配置）
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

所有结果均采用 **Medium 收敛阈值**（位置 10mm、姿态 5°）作为主基准，repeat=30，zero_seed 策略。测试平台为 NVIDIA GeForce RTX 4060 Laptop GPU（Ada Lovelace, sm_89, 3,072 CUDA Cores, 8 GB GDDR6）。

> **权威数据来源：** [`docs/项目总览/03-论文贡献与核心声明.md`](docs/项目总览/03-论文贡献与核心声明.md) §3。原始 CSV 位于 `standard_robot_cuda_ik/data/results/main_comparison/main_comparison.csv`。

### 4.1 B5 vs. cuRobo —— 主对比

| 批量规模 N | B5 吞吐量 | cuRobo 吞吐量 | 吞吐比值 | B5 测量时间 | cuRobo 测量时间 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | **112,414 t/s** | 3,118 t/s | **36.1×** | 0.89 ms | 32.1 ms |
| 500 | **158,251 t/s** | 15,844 t/s | **10.0×** | 3.16 ms | 31.6 ms |
| 1000 | **148,412 t/s** | 31,611 t/s | **4.7×** | 6.74 ms | 31.6 ms |
| 5000 | **168,683 t/s** | 155,059 t/s | **1.09×** | 29.64 ms | 32.3 ms |

> **计时口径说明：** CUDA B5 的"测量时间"使用 CUDA event `cudaEventElapsedTime`，仅测量 GPU kernel 设备端执行时间；cuRobo 的"测量时间"使用 host 端 `time.perf_counter()`，测量 Python→C++→CUDA 全调用栈。两者计时口径不同——表中比值反映统一 benchmark 下的工程测量吞吐比，不代表两个求解器内部 kernel 的严格同口径 GPU 时间对比。详见 [`docs/项目总览/03-论文贡献与核心声明.md`](docs/项目总览/03-论文贡献与核心声明.md) §6。

**关键发现：**

- **小批量绝对优势**：N=100 时，B5 吞吐量达 cuRobo 的 36 倍。这对交互式应用（如 100 个路径点的实时轨迹优化）至关重要。
- **GPU 线性扩展**：B5 的 GPU 时间与批量规模的线性拟合优度达到 **R² > 0.999**——单目标求解成本近乎恒定，全批量范围内（N=100→10000）吞吐量稳定在 148k–174k t/s（±8%）。
- **cuRobo 批量振荡**：在 N=4000、7000、9000、10000 时，cuRobo 间歇性进入约 230ms/batch 的退化模式（正常模式的 7 倍）。Nsight Systems CUDA API trace 证据（N=4000: 14,108 kernel vs N=5000: 5,945 kernel, 2.37×）强指向 cuRobo 内部 sub-batch 划分策略的批量依赖性，PyTorch CUDA Caching Allocator 已明确排除（cudaMalloc/cudaFree < 1% API 时间）。B5 的单 kernel 全迭代封装在结构上对此类问题免疫。

### 4.2 消融实验 —— 逐级优化的贡献

| 级别 | N=100 | N=500 | N=5000 | 核心优化 |
|:---:|:---:|:---:|:---:|:---|
| B0（FP64 基线） | 7,589 t/s（收敛率 83%） | 9,896 t/s（收敛率 52%） | 12,507 t/s（收敛率 56%） | 无——全局内存，无填充，固定阻尼 |
| B3（+自适应阻尼） | 51,361 t/s（**6.8×**） | 62,384 t/s（**6.3×**） | 66,050 t/s（**5.3×**） | 距离比例阻尼 + Marquardt 更新 + 停滞超驰 |
| B5（+混合精度） | 112,414 t/s（**14.8×**） | 158,251 t/s（**16.0×**） | 168,683 t/s（**13.5×**） | FP32 FK/Jacobian + FP64 H/g 累积 + FP64 LDLᵀ |

> **权威数据来源：** [`docs/项目总览/03-论文贡献与核心声明.md`](docs/项目总览/03-论文贡献与核心声明.md) §3.3。原始 CSV：`data/results/ablation/ablation_medium.csv`。

**核心洞察**：无自适应阻尼时，B0 收敛率崩塌至 52–83%。引入自适应阻尼（B3）后，收敛率恢复至 100%，吞吐量提升 4–6 倍。混合精度（B5）在此基础上再提升 2.2–2.5 倍，充分利用了 Ada Lovelace 架构 64:1 的 FP32:FP64 吞吐比。

---

## 5. CUDA Kernel 设计

> **权威参考：** [`docs/项目总览/04-GPU架构与CUDA实现细节.md`](docs/项目总览/04-GPU架构与CUDA实现细节.md)。以下为关键设计摘要。

### 5.1 Block-Thread 映射

```
Grid:  (N, 1, 1)     ← 每个目标位姿一个 Block
Block: (128, 1, 1)   ← 4 个 Warp × 32 Lane = 128 线程
```

每个 Block 独立求解一个目标位姿的 IK 问题。**代码使用 flat `threadIdx.x` 范围检查（非严格 warp 边界）控制线程分工：**

| 计算阶段 | 线程范围 | 条件 | 任务 | 并行度 |
|---------|:---:|------|------|:---:|
| FK（正运动学） | 0 | `threadIdx.x == 0` | Rodrigues 公式 6 关节链式乘法 | 1 |
| 位姿误差 | 0 | `threadIdx.x == 0` | 位置误差 + 姿态 geodesic 距离 | 1 |
| 收敛判定 | 0 | `threadIdx.x == 0` | pos < 10mm AND rot < 5° | 1 |
| 数值 Jacobian | 0–5 | `threadIdx.x < 6` | 每线程 1 列，中心差分 ε=1e-6，J = ∂x/∂q | 6 |
| 自适应阻尼更新 | 0 | `threadIdx.x == 0` | LM 阻尼调整 | 1 |
| **H 矩阵构造** | **0–35** | **`threadIdx.x < 36`** | **全部 36 元素** `row=tid/6, col=tid%6` | **36** |
| g 向量构造 | 0–5 | `threadIdx.x < 6` | g = JᵀW²e | 6 |
| LDLᵀ 求解 | 0 | `threadIdx.x == 0` | 寄存器级 6×6 LDLᵀ，86 次标量运算 | 1 |
| 步长钳位 | 0 | `threadIdx.x == 0` | ‖Δq‖∞ > 0.35 rad → scale | 1 |
| 关节更新 + 限位 | 0–5 | `threadIdx.x < 6` | q = clamp(q + dq, lo, hi) | 6 |

> ⚠️ **重要澄清：** H 矩阵构造使用线程 0–35，这跨越了 Warp 0（lane 0–31）和 Warp 1（lane 0–3）。因此 H 矩阵不应对应任何单一 warp。论文和专利中**禁止**使用 "Warp 2 负责 H 构造" 等 warp 专属表述。

### 5.2 PaddedMat6x8 —— 共享内存 Bank 冲突降低

所有中间矩阵（雅可比、Hessian）采用 **6×8 列填充布局**存储在共享内存中：

```cpp
struct PaddedMat6x8 {
    double data[6 * 8];  // 48 个 double = 384 字节（自然布局仅需 36 个）
    __device__ double& operator()(int row, int col) {
        return data[row * 8 + col];  // stride=8，零指令开销
    }
};
```

**原理：** 自然 stride=6 布局下，Bank 访问模式 `gcd(12, 32)=4`，每 8 次访问周期重复，导致 2–3 路 Bank 冲突。采用 stride=8（每行 16 个 Bank），偶数行使用 Bank 0–15，奇数行使用 Bank 16–31——两组 Bank 集合完全不重叠，核心计算中 Bank 冲突显著降低。

**实测效果（NCU B4→B5）：** shared bank conflicts 从 3,522 降至 1,295（**-63%**）。仍存在残余冲突（来自非填充矩阵访问路径），因此效果表述为"**降低**"而非"消除"。

> ⚠️ 论文中**禁止**使用"完全消除 Bank 冲突"、"零 Bank 冲突"等表述。

### 5.3 混合精度（B5）—— FP32 + FP64 混合策略

Ada Lovelace 消费级 GPU 的 FP64:FP32 吞吐比为 **1:64**。混合精度将大部分算术运算移至 FP32，同时在关键路径上保留 FP64：

| 组件 | 精度 | 理由 |
|:---|:---:|:---|
| 正运动学 (FK) | **FP32** | 三角函数密集，FP32 吞吐为 FP64 的 64× |
| 数值 Jacobian（6 列 × 2 次 FK） | **FP32** | ε=1e-6 扰动下 FP32 相对舍入误差 ~1e-7，远小于收敛容差 |
| H 矩阵累加（JᵀW²J + λI） | **FP64** | 36 个内积各 6 项累加，需抑制 FP32 截断误差传播 |
| g 向量累加（JᵀW²e） | **FP64** | 同上 |
| 位姿误差 | FP64 | 收敛容差判定的精度关键（10mm/5°） |
| LDLᵀ 分解 | **FP64** | 除法对 D_j 偏小敏感，误差经回代放大 |
| 阻尼参数 λ | FP64 | 防止自适应阻尼中的下溢 |
| 关节更新 | FP32 | Δq 已由 FP64 LDLᵀ 高精度确定 |

**效果：** B3→B5 吞吐提升 120–149%（2.2–2.5×），收敛率无退化（保持 0.998+）。

### 5.4 自适应阻尼

固定阻尼（B0）使用 `λ ≡ 2e-3`——或过于激进（导致震荡）或过于保守（收敛缓慢）。自适应阻尼（B3+）采用三阶段策略：

```
迭代 0（距离初始化）：
  若 pos_err > 0.5m：    λ = λ_far (0.1)
  若 pos_err > 0.1m：    λ = 线性插值
  否则：                  λ = λ_base (5e-4)

迭代 1+（Marquardt 误差驱动）：
  若 pos_err 改善：       λ *= 0.7   → 趋向 Gauss-Newton
  若 pos_err 恶化：       λ *= 2.0   → 趋向梯度下降

停滞超驰（连续 > 12 次迭代无改善）：
  λ *= 5.0   → 强制跳出局部极小

全局钳位：λ ∈ [1e-4, 0.5]
```

**效果**：仅自适应阻尼一项（B0→B3），通过将平均迭代次数从 ~31–80 降至 ~13–15，实现 4–6 倍吞吐提升，并将收敛率从 52–83% 恢复至 100%。

### 5.5 常量内存

1,384 字节的运动学参数存储在 `__constant__` 内存中，warp 内广播延迟约 1 个周期（全局内存约 400 个周期）：

| 符号 | 大小 | 内容 |
|:---|:---:|:---|
| `c_segment_origins[96]` | 768 B | DH 参数原点 |
| `c_segment_axes[18]` | 144 B | 旋转轴方向 |
| `c_q_index[6]` | 24 B | 关节→连杆索引映射 |
| `c_T_wrist3_to_tcp[16]` | 128 B | TCP 工具变换 |
| `c_joint_limits[12]` | 96 B | 关节限位 |
| `c_weight_schedule[24]` | 192 B | 误差权重（4 level × 6 维） |
| `c_lambda_params[4]` | 32 B | 阻尼参数 |

B0→B1 吞吐增益 < 5%，因为 UR10 工作集 ~912 bytes 本身即可被 L1 高效容纳（128 KB/SM）。常量内存收益在 problem size 有限时并非主要性能驱动。

### 5.6 寄存器驻留 LDLᵀ —— 不依赖外部库

通过 cuBLAS 求解 6×6 线性方程组需要 kernel launch 开销（5–10 μs）、数据搬运（共享→全局→cuBLAS→共享）和库内部分发——对小矩阵而言开销远超计算本身。

手写的寄存器驻留 LDLᵀ 分解在 **86 次标量运算（约 0.1 μs）** 内完成 6×6 求解，零内存流量，零 kernel launch：

| 阶段 | 运算量 |
|------|:---:|
| 对角元更新 D_j | 15 FMA |
| 非对角元 L_ij 更新 | 20 FMA |
| L 缩放（除以 D_j） | 15 DIV |
| 前代（Ly = b） | 15 FMA |
| 对角缩放（D⁻¹z = y） | 6 DIV |
| 回代（Lᵀx = z） | 15 FMA |
| **合计** | **65 FMA + 21 DIV = 86 标量运算** |

**为什么 LDLᵀ 而非 Cholesky：** Cholesky (LLᵀ) 需要 6 次 `sqrt` 指令，GPU sqrt 吞吐仅 FMA 的 1/4–1/8。LDLᵀ 以 15 次额外 FMA 避免全部 sqrt，总延迟更低。

NCU 实测：94 (B4) – 98 (B5) registers/thread，**零 local memory spilling**。Ada Lovelace 上限 255 registers/thread，安全裕度充足。

### 5.7 单 Kernel 封装

所有 DLS 迭代在**一次 CUDA kernel 启动**内完成。CPU 提交批次、调用 `cudaDeviceSynchronize()` 后直接获取结果——迭代过程中无 CPU-GPU 往返。这消除了：

- Kernel 启动开销（消费级 GPU 上每次约 5–10 μs）
- CUDA API 分发延迟
- 启动间的设备同步开销

cuRobo 和 PyRoki 的等价执行路径是每次迭代数百甚至数千次 kernel 启动，每次启动还伴随 Python→CUDA 绑定开销和 GPU 流同步。

---

## 6. 手写 CUDA 为何超越框架型求解器

> **权威参考：** [`docs/项目总览/03-论文贡献与核心声明.md`](docs/项目总览/03-论文贡献与核心声明.md) §5。

| 维度 | CUDA B5（本项目） | cuRobo (NVIDIA) | PyRoki (JAX) |
|:---|:---|:---|:---|
| **每批次 kernel 启动数** | **1** | 数百–数千（N 依赖） | 数千（JIT + XLA） |
| **CPU-GPU 同步点** | **0**（仅在结束时同步一次） | 大量（每次 kernel 启动） | 大量（JAX 分发） |
| **框架层** | **无**（原始 CUDA C++） | cuda.core + Warp + PyTorch | JAX JIT + XLA + Python |
| **精度模型** | 混合 FP32+FP64 | FP32 | FP64 |
| **矩阵求解** | 自定义寄存器 LDLᵀ（约 0.1 μs） | cuBLAS（库调用开销约 5 μs） | JAX 线性代数 |
| **共享内存优化** | PaddedMat6x8（Bank 冲突 -63%） | 未公开 | 不适用（XLA 管理） |
| **迭代封装** | 单 kernel | 多 kernel | 多 kernel |
| **单目标成本扩展** | 严格线性（R² > 0.999） | 振荡（4/12 N 值退化） | 未测量 |

**性能差距的结构性原因：**

- **小批量下 kernel 启动次数占主导**：N=100 时，cuRobo 每批次启动数百个 kernel，B5 仅启动 1 个。消费级 GPU 驱动栈上每次 kernel 启动约 5–10 μs。
- **框架开销是固定税**：cuRobo 的 Python→C++→CUDA 调用链每批次增加约 5–15ms，与批量规模无关。N=5000 时尚可分摊；N=100 时 B5 的 0.89ms 中不含任何框架税。
- **小矩阵自定义 LDLᵀ 优于库调用**：6×6 LDLᵀ 是极小问题。cuBLAS 的内部分发和内存搬运开销超过实际求解时间的 50 倍。寄存器驻留 LDLᵀ 是 n ≤ 16 矩阵的最优策略。
- **固定内存模型避免批量敏感性**：B5 通过 `cudaMalloc` 一次性分配 `N × sizeof(Target) + N × sizeof(Result)`。cuRobo 依赖内部 sub-batch 划分策略，在不同 N 值触发不同的 kernel launch 粒度（N=4000: 14,108 kernel vs N=5000: 5,945 kernel, 2.37×），导致某些批量规模出现约 230ms 退化模式。

---

## 7. 实验方法

> **权威参考：** [`docs/项目总览/01-实验数据目录.md`](docs/项目总览/01-实验数据目录.md)。

### 7.1 收敛阈值体系

| 级别 | 位置容差 | 姿态容差 | 用途 |
|:---|:---|:---|:---|
| **Loose（宽松）** | 30 mm | 10° (0.1745 rad) | 旧实验参考标准 |
| **Medium（中等）** ★ | **10 mm** | **5° (0.0873 rad)** | **主 benchmark** |
| **Strict（严格）** | 5 mm | 1° (0.0175 rad) | 精密应用压力测试 |

Medium 阈值是主基准，因为该阈值能显著区分求解器质量：B0 的收敛缺陷（Medium 下 52–83%，vs Loose 下 80–100%）在旧阈值下无法显现。

### 7.2 消融级别（B 系列命名）

| 论文名称 | 可执行文件 | 优化内容 | 关键指标 |
|:---|:---|:---|:---|
| B0 | A0 | FP64 基线（无优化） | 收敛率：52–83% |
| B1 | A1 | + 常量内存 | 吞吐量 +<5% |
| B2 | A2 | + PaddedMat6x8 | Bank 冲突 -63% |
| — | A3 | + 寄存器 LDLᵀ | — |
| — | A4 | + Kernel 融合 | — |
| B3 ★ | A5 | + 自适应阻尼 | **吞吐量 +400–600%**，收敛率恢复至 100% |
| B4 | A6 | A5 + 步长钳位 + 分支对齐 | 已知负收益（吞吐 -15%~-20%） |
| B5 ★★ | A7 | **混合精度（FP32+FP64）** | **吞吐 +120–149%**，收敛率无退化 |
| B6 | A8 | A7 + CUDA Graph | 边际收益 +3.7%（N=100） |

**始终启用（所有级别，不可消融）：** 1 block/target、128 threads/block、寄存器级 LDLᵀ、单 kernel 全迭代封装。

### 7.3 计时口径

| 口径 | 测量范围 | 工具 | 论文使用 |
|------|---------|------|:---:|
| `kernel_time_only` | GPU kernel 设备端执行时间 | `cudaEventElapsedTime` | **CUDA B5 主数据** |
| `gpu_end_to_end_time` | kernel + H2D/D2H + device sync | `cudaEventElapsedTime` | 辅助参考 |
| `host_api_total_time` | 含主机端预处理和 API 调用 | `std::chrono::steady_clock` | 工程参考 |
| cuRobo host time | Python→C++→CUDA 全调用栈 | `time.perf_counter()` + `torch.cuda.synchronize()` | **cuRobo 主数据** |

> ⚠️ CUDA B5 和 cuRobo 的计时口径不同，不能直接比较"GPU 时间"。论文中应使用"吞吐比值"或"测量时间"等中性表述。

### 7.4 可复现性保障

- **目标生成**：确定性 PRNG（seed=42），关节角在 [−π, π] 内均匀采样，经 FK 验证可到达性
- **种子策略**：`zero_seed`（全零向量，主实验默认）、`home_seed`（UR10 零位构型）、`random_seed`、`near_ground_truth_seed`
- **重复次数**：每组配置 30 次独立运行（3 warmup + 30 计时）
- **最大迭代**：Kmax = 160
- **硬件平台**：NVIDIA GeForce RTX 4060 Laptop GPU（Ada Lovelace, sm_89, 3,072 CUDA Cores, 8 GB GDDR6），CUDA Toolkit 12.6

---

## 8. 数据与图表

所有实验数据集中存放于 [`standard_robot_cuda_ik/data/`](standard_robot_cuda_ik/data/)。完整的数据-论文映射表参见 [`docs/项目总览/01-实验数据目录.md`](docs/项目总览/01-实验数据目录.md) §4。

| 数据集 | 文件数 | 内容 |
|:---|:---:|:---|
| Targets | 36 | UR10 seed=42 末端位姿，N=100→10000，.bin/.csv/.json |
| Seeds | 96 | 4 种策略 × 12 个批量规模 × 2 种格式 |
| Results | 11 CSV | 主对比、消融、阈值扫描、全量程、CPU 基线、种子策略、Panda 7-DOF |
| Profiling | 5 .ncu-rep | Nsight Compute 报告（B4/B5, N=100/5000） |
| Figures | 5 PNG | 吞吐量对比、加速比、消融、收敛率、平均迭代次数 |

重新生成图表：
```bash
cd standard_robot_cuda_ik/data/figures
python3 plot_all_figures.py
```

---

## 9. 文档导航

| 文档 | 用途 |
|------|------|
| [`docs/项目总览/README.md`](docs/项目总览/README.md) | 权威参考总索引 + 快速查找表 |
| [`docs/项目总览/01-实验数据目录.md`](docs/项目总览/01-实验数据目录.md) | 184 文件完整数据地图 + 数据→论文映射 |
| [`docs/项目总览/02-源代码目录.md`](docs/项目总览/02-源代码目录.md) | CUDA/Python/CMake 完整代码地图 |
| [`docs/项目总览/03-论文贡献与核心声明.md`](docs/项目总览/03-论文贡献与核心声明.md) | 贡献、创新点、性能声明、禁止表述黑名单 |
| [`docs/项目总览/04-GPU架构与CUDA实现细节.md`](docs/项目总览/04-GPU架构与CUDA实现细节.md) | GPU 架构参数、线程映射、寄存器预算、混合精度 |
| [`docs/paper/paper_complete.md`](docs/paper/paper_complete.md) | 论文最新完整版 |
| [`docs/专利/技术交底书.md`](docs/专利/技术交底书.md) | 专利技术交底书 |
| [`docs/优势/cuda_vs_curobo_结构性优势.md`](docs/优势/cuda_vs_curobo_结构性优势.md) | CUDA vs cuRobo 结构化对比分析 |
| [`docs/data/README.md`](docs/data/README.md) | 论文图表 CSV 数据目录 |
| [`docs/goal.txt`](docs/goal.txt) | 项目需求规格文档 |

---

## 10. 添加你自己的求解器

Benchmark 框架专为公平的跨求解器对比设计。添加新求解器的步骤：

1. 参照 [`bench_cuda_6dof.py`](standard_robot_cuda_ik/benchmark/bench_cuda_6dof.py) 的模式创建 `bench_your_solver.py`
2. 从 `data/targets/` 和 `data/seeds/` 加载目标和种子数据
3. 使用 `common.load_urdf()` 和 `common.check_convergence()` 确保收敛判定一致性
4. 按照标准 JSON 模式输出结果
5. 运行 `run_all.py --solver your_solver` 进行统一对比

所有求解器使用相同的目标、种子、收敛阈值和重复次数进行评估——确保公平的同类对比。

---

## 11. 历史版本

本项目的第一代概念验证代码（自定义 UR10 + 铲斗装配体 CUDA IK 求解器，相对 CPU KDL 约 960 倍加速）已归档至 `/mnt/linuxdata/cuda数据备份/cuda_low_level_optimization/`。其核心 CUDA 技术（阶段式并行分工、PaddedMat6x8、自适应阻尼、寄存器 LDLᵀ）已迁移并泛化至当前主线 `standard_robot_cuda_ik/`。

---

*最后更新：2026-06-14 | 测试平台：NVIDIA GeForce RTX 4060 Laptop GPU (Ada Lovelace, sm_89) | CUDA Toolkit 12.6 | GCC 11.4.0*
