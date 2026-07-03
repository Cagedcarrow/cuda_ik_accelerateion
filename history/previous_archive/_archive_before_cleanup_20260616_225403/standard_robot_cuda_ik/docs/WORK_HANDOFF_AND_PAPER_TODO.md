# `standard_robot_cuda_ik` 后续待办清单与工程导读

## 1. 文档用途

这份文档是 `standard_robot_cuda_ik` 的**唯一主交接文档**，服务对象是下一位继续工作的 AI 或工程师。

它的职责不是替代论文、README 或验收报告，而是把以下信息集中到一个地方：

- 这个工程现在在做什么；
- 为什么这样设计；
- 当前代码结构和各目录职责；
- 当前论文和日志分别在哪里；
- 当前已经真实完成了哪些工作；
- 当前还没完成什么；
- 接下来必须按什么顺序继续做；
- 当前使用到的 CUDA 优化工作流和论文写作工作流是什么；
- 哪些本地文件和外部来源是后续工作必须阅读的。

这份文档的定位是：**读完之后可以直接进入后续优化与论文完善，而不需要重新梳理对话历史。**

## 2. 项目核心定位

`standard_robot_cuda_ik` 研究的是：

- 标准机械臂上的批量逆运动学（batch IK）求解；
- 只通过 CUDA 底层架构适配来加速求解；
- 在相同模型、相同 target、相同 seed、相同阈值下做公平 benchmark；
- 用 Nsight Compute / Roofline / 消融实验解释性能来源；
- 为论文提供可复现的工程、实验和证据链。

这个工程**不**研究：

- 路径规划；
- 运动规划；
- 避障系统；
- 完整控制流水线；
- 轨迹优化；
- 新的 IK 数学算法包装。

这个工程的核心意义必须始终保持为：

**针对机器人 IK 这种小规模、高频、大批量任务，只通过 GPU 线程映射、寄存器级小矩阵求解、共享内存布局、常量内存广播、kernel fusion、批量任务组织等硬件适配手段，提高单位硬件效率。**

如果后续写文档、论文或实验说明时把当前工作写成“motion planning”“path planning”“完整运动学系统性能”，都属于错误表述。

## 3. 当前工程状态总览

### 3.1 已完成

- 官方 `UR10` 模型已经统一到 `UniversalRobots/Universal_Robots_ROS2_Description` 的 `4.3.1` tag，本地主模型文件是 `standard_robot_cuda_ik/urdf/ur10_official.urdf`。
- `seed=42` 的标准化 target/seeds 资产已经生成，主公平配置固定为 `zero_seed`。
- 主 benchmark 入口已经统一到 `python3 standard_robot_cuda_ik/benchmark/run_all.py`。
- `cuda`、`curobo`、`pyroki`、`kdl`、`numeric_dls` 已经接到统一 benchmark 框架；其中主公平 GPU 表当前只收 `cuda` 和 `curobo`。
- Phase 6 的一轮竞争性优化已经做完，当前 CUDA 主线固定 `weight_level=2`。
- Phase 7 已经有真实 full NCU 动态 profiling 结果和结构化消融状态说明。
- Phase 8 已经完成完整 Markdown 论文初稿。

### 3.2 当前主结论

- 主公平配置：官方 `UR10` + `tool0` + `seed=42` + `zero_seed` + `position_error < 0.03 m` + `orientation_error < 0.5236 rad`。
- **A7 混合精度在 N=5000 上以 180,962 targets/s 反超 cuRobo 1.25 倍**，彻底关闭了 N=5000 差距。
- CUDA A7（混合精度）在全部测试批量 N=100/500/1000/5000 上均超过 cuRobo。
- 当前 CUDA 主线不是 memory-bound（DRAM throughput < 2%），主要受限于 FP64 pipeline（60.7% SM throughput）和寄存器压力（94-98 regs/thread）。

### 3.3 当前主结果

以下结果来自主公平 `zero_seed`、`repeat=30`、**最优消融配置 A7**（混合精度 + 自适应阻尼，步长钳位与分支对齐关闭）：

| N | CUDA A7 (targets/s) | cuRobo (targets/s) | 加速比 |
|:--:|---:|---:|---:|
| 100 | 107,250 | 2,850 | **37.6×** |
| 500 | **149,787** | 14,574 | **10.3×** |
| 1000 | **140,246** | 29,518 | **4.8×** |
| 5000 | **180,962** | 144,855 | **1.25×** |

与 A5（纯 FP64）的对比：

| N | A5 FP64 | A7 Mixed | 加速比 (A7/A5) |
|:--:|---:|---:|---:|
| 100 | 52,064 | 107,250 | **2.06×** |
| 500 | 59,821 | 149,787 | **2.50×** |
| 5000 | 71,380 | 180,962 | **2.54×** |

这些结果说明：

- A7 混合精度在所有批量上远超 A5 FP64，证实 Ada Lovelace FP32:FP64=2:1 的吞吐比可被有效利用；
- A7 在 N=5000 上反超 cuRobo 1.25 倍，**cuRobo 的最大规模优势已被混合精度消除**；
- 核心叙事升级：GPU 底层硬件适配+混合精度在 1–5000 全部范围内优于算法框架优化；
- NCU 证据链完整：kernel 非 DRAM-bound（1.16%），bank conflict 非瓶颈（1% wavefronts）。

### 3.4 当前未完成

当前未完成项已经大幅减少。经过本轮工作，以下项已经完成：

- ✅ **A0-A6 独立消融**：7 个独立可编译 target（见 `docs/logs/ablation_official_ur10.md`）
- ✅ **numeric_dls 补齐**：全规模参考（见 `data/results/`）
- ✅ **论文整合与润色**：`docs/paper/paper_full.md` + nature-polishing 去 AI 化
- ✅ **A7 混合精度**（ABLATION_LEVEL=7）：FP32 FK/Jacobian/Hessian + FP64 LDLT，N=100/500/1000/5000 完整 benchmark
- ✅ **A8 CUDA Graph**（A7 + USE_CUDA_GRAPH）：kernel_time ~0% overhead vs direct launch
- ✅ **Phase 5 Bank Conflict 分析**：A7 仅 1,295 store conflicts（1% wavefronts），非瓶颈
- ✅ **N=5000 NCU profiling**：A5 N=5000 `.ncu-rep` 已生成
- ✅ **7DOF Panda 独立验证**：CUDA FK 误差 2.78e-16，IK 收敛 50% 与 CPU 一致（见 `experiments/7dof_test/`）
- ✅ **实验数据 CSV 归档**：`docs/data/` 包含 5 个 CSV 文件 + README
- ✅ **论文三章节更新**：05_experiments（加入 A7 数据）、06_7dof_extension（加入验证结果）、07_discussion（NCU 分析深化）
- ✅ **Nature 学术润色**：`docs/paper/paper_full.md` 已去 AI 化

当前仍实际未完成（扩展方向）：

- 7DOF 集成到主 benchmark 框架（当前仅独立验证）
- N=5000+ 的 grid 调度优化（消除 partial wave 效应）
- 多 seed 策略对 7DOF 收敛率的提升评估
- N=200, N=2000, N=10000 等更多规模的 benchmark

## 4. 代码结构导读

以下导读按目录职责组织，不按“文件清单堆砌”组织。

### 4.1 `benchmark/`

这个目录负责统一实验入口和各 solver 包装层。

- `benchmark/run_all.py`
  统一 benchmark 入口。负责选择 solver、执行 benchmark、写 `CSV + JSON + Markdown`，并在 solver 失败时写结构化错误日志。
- `benchmark/bench_cuda_6dof.py`
  CUDA 主线包装层。先检查 target/seed 资产，再显式解析官方 `UR10` URDF，再调用 `build/standard_robot_cuda_runner`。
- `benchmark/bench_curobo.py`
  cuRobo 公平包装层。通过自定义 robot dict 固定同一 URDF/TCP，并通过 `current_state + seed_config` 传入共享外部 seed。
- `benchmark/bench_pyroki.py`
  PyRoki 包装层。当前已接入共享外部 seed，并排除 JIT 预热。
- `benchmark/bench_kdl.py`
  真实 PyKDL CPU 几何 baseline。
- `benchmark/bench_numeric_dls.py`
  CPU 数值 DLS 参考实现。
- `benchmark/common.py`
  benchmark 公共数据结构与持久化逻辑，例如 `BenchmarkResult`、summary 输出、错误日志输出、target/seed 读取、误差计算。

### 4.2 `src/cuda/`

这个目录是当前 CUDA 主线核心。

- `src/cuda/cuda_ik_6dof.cu`
  当前最重要的 CUDA IK 核心实现。研究重点在这里：线程映射、小矩阵求解、共享内存布局、阻尼、步长约束、融合式单 kernel 迭代。
- `src/cuda/cuda_benchmark_runner.cu`
  CUDA benchmark 可执行入口。负责加载二进制 target/seed 文件、上传常量、运行 kernel，并输出三类时间口径。
- `src/cuda/cuda_utilities.cuh`
  设备端辅助函数与矩阵封装。`PaddedMat6x8`、FK、误差、Rodrigues 旋转、小工具函数都在这里。
- `src/cuda/cuda_memory.cu`
  CUDA memory 管理薄层实现。
- `src/cuda/cuda_collision.cu`
  当前论文主线不研究碰撞系统；这个文件不是当前主实验重点。

### 4.3 `src/cpu_baseline/`

这个目录提供 CPU 对照求解器。

- `src/cpu_baseline/kdl_solver.cpp`
  KDL 基线实现。
- `src/cpu_baseline/numeric_dls_solver.cpp`
  数值 DLS 基线实现。

### 4.4 `tools/`

这个目录负责模型、资产和验证脚本，是“标准化工程成立”的关键。

- `tools/fetch_official_ur10.py`
  从官方仓库抓取并生成本地 `ur10_official.urdf`。
- `tools/generate_standard_assets.py`
  生成 target/seeds 资产、CUDA 常量头、目标生成日志。后续任何要重建数据的人，必须先看它。
- `tools/verify_official_ur10.py`
  生成官方 UR10 模型验证报告，检查 joint、TCP、FK 一致性等。
- `tools/verify_seed42_reproducibility.py`
  验证 `seed=42` 资产重复生成是否 hash 一致。
- `tools/robot_model.py`
  Python 侧 URDF 链解析、FK、轴与原点提取、数据导出工具。它是标准化数据链路的基础。

### 4.5 `config/`

这个目录放实验默认配置。

- `config/benchmark.yaml`
  主 benchmark 默认 batch、容差、当前 CUDA 主权重等级等。
- `config/robots.yaml`
  机器人模型入口，包含 UR10、UR5、Panda。
- `config/target_generation.yaml`
  target/seeds 生成配置。

### 4.6 `data/targets`

这里是可复现的目标位姿资产。

- `ur10_seed42_N100/500/1000/5000.{json,csv,bin}`
  同一批标准目标数据的不同持久化格式。
- 目标来源是“先采样关节角，再做 FK”，因此目标必然可达。

### 4.7 `data/seeds`

这里是标准化初始种子资产。

- `zero_seed`
- `home_seed`
- `random_seed`
- `near_ground_truth_seed`

当前主公平 benchmark 固定使用 `zero_seed`。后续不能私自改成更有利的 seed 策略再声称结果更公平。

### 4.8 `data/results`

这里是 benchmark 真值结果目录。

- `*_summary.json`
  最权威的结构化实验结果。
- `*.csv`
  便于快速做表或额外处理。
- `*.md`
  单次 benchmark 的说明性结果。
- `data/results/errors/`
  solver 失败日志，验证失败隔离逻辑的重要证据。

### 4.9 `data/profiling`

这里是 Nsight Compute 产物。

- `ur10_cuda_n100_launch.ncu-rep`
- `ur10_cuda_n100_full_zero_seed.ncu-rep`

这是当前 Phase 7 的原始 profiling 证据来源。

### 4.10 `experiments/`

这个目录当前主要是实验组织占位目录，方便后续把实验脚本、版本化中间结果和子实验说明按主题归档。

- `01_official_ur10_main/`
- `02_batch_size_scaling/`
- `03_solver_comparison/`
- `04_ablation/`
- `05_roofline/`
- `06_7dof_extension/`

后续如果补完整消融或 7DOF 扩展，建议把各自实验说明放回这些目录，而不是继续把所有中间材料散落到根级 `docs/logs/`。

### 4.11 `docs/logs` 与 `standard_robot_cuda_ik/docs/logs`

这里必须区分清楚：

- 根目录 `docs/logs/`
  这是**人类可读、论文可引用、验收可核查**的权威日志目录。
- 子项目目录 `standard_robot_cuda_ik/docs/logs/`
  当前只保存 machine-readable JSON 辅助日志，例如：
  - `official_ur10_model_verification.json`
  - `ur10_seed42_reproducibility.json`

后续写论文和做工程交接时，应优先引用根目录 `docs/logs/` 的 Markdown。

## 5. 当前论文与日志入口

### 5.1 论文位置

- 根级论文目录：`/mnt/linuxdata/cuda_ik_accelerateion/docs/paper/`
- 主入口：`docs/paper/main.md`
- 实验章节：`docs/paper/05_experiments.md`
- 7DOF 章节：`docs/paper/06_7dof_extension.md`
- 讨论章节：`docs/paper/07_discussion.md`
- 结论章节：`docs/paper/08_conclusion.md`

这些文件的职责是：

- `docs/paper/main.md`
  论文总入口、摘要、标题候选、章节导航。
- `docs/paper/05_experiments.md`
  主实验事实、主表、参考表、Phase 6 优化结论、当前未完成实验项。
- `docs/paper/06_7dof_extension.md`
  7DOF 扩展计划，只能写计划，不能写假结果。

### 5.2 根级日志位置

- 根级日志目录：`/mnt/linuxdata/cuda_ik_accelerateion/docs/logs/`

后续工作最重要的日志入口是：

- `docs/logs/final_acceptance_report.md`
  当前最完整的验收汇总，先看这个文件可快速判断什么已经完成、什么只是扩展项。
- `docs/logs/official_ur10_solver_benchmark.md`
  主 benchmark 真值与公平性约束说明。
- `docs/logs/ablation_official_ur10.md`
  当前 Phase 7 消融状态说明，明确哪些已测、哪些未测。
- `docs/logs/roofline_ncu_official_ur10.md`
  当前 NCU / Roofline 结论来源。
- `docs/logs/official_ur10_model_verification.md`
  官方 UR10 模型一致性验证。
- `docs/logs/target_generation_seed42.md`
  target 生成策略与空间包络说明。
- `docs/logs/ur10_seed42_reproducibility.md`
  资产可复现性和 hash 一致性报告。

### 5.3 论文正文、实验真值、验证证据分别看哪里

- 论文正文主叙述：
  `docs/paper/main.md`、`docs/paper/05_experiments.md`
- 实验真值结果：
  `standard_robot_cuda_ik/data/results/*_summary.json`
- 公平性与模型一致性证据：
  `docs/logs/official_ur10_model_verification.md`
  `docs/logs/official_ur10_solver_benchmark.md`
- 可复现与数据资产证据：
  `docs/logs/target_generation_seed42.md`
  `docs/logs/ur10_seed42_reproducibility.md`
- Profiling / 消融证据：
  `docs/logs/roofline_ncu_official_ur10.md`
  `docs/logs/ablation_official_ur10.md`

## 6. 当前已完成工作清单

- 已固定官方 UR10 模型来源到 `UniversalRobots/Universal_Robots_ROS2_Description` 的 `4.3.1` tag。
  证据：`docs/logs/official_ur10_model_verification.md`
- 已验证标准 UR10 joint 数、axis、origin、TCP、joint limits，并给出 CPU FK 一致性检查。
  证据：`docs/logs/official_ur10_model_verification.md`
- 已验证 CPU FK 对 `yourdfpy` 的最大误差是 `3.331e-16`，并记录 CUDA FK 对 CPU FK 的 spot check。
  证据：`docs/logs/official_ur10_model_verification.md`
- 已生成 `seed=42` 的标准化 target/seeds 资产，并验证重复生成 hash 一致。
  证据：`docs/logs/target_generation_seed42.md`
  证据：`docs/logs/ur10_seed42_reproducibility.md`
- 已把 `cuda`、`curobo`、`pyroki`、`kdl`、`numeric_dls` 接到统一 benchmark 入口。
  证据：`benchmark/run_all.py`
- 已实现 `run_all.py` 的结构化错误日志与单 solver 失败隔离。
  证据：`docs/logs/solver_failure_isolation.md`
- 已让 `cuda` benchmark 在运行前显式完成 target / seed / URDF 检查。
  证据：`benchmark/bench_cuda_6dof.py`
- 已让 `curobo` 使用同一 URDF/TCP，并通过 `current_state + seed_config` 接入共享外部 seed。
  证据：`benchmark/bench_curobo.py`
- 已让 `pyroki` 使用共享外部 seed，并排除 JIT 预热。
  证据：`benchmark/bench_pyroki.py`
- 已完成真实 PyKDL baseline。
  证据：`benchmark/bench_kdl.py`
- 主公平 `zero_seed` 下，`cuda` 与 `curobo` 已完成 `N=100/500/1000/5000` 的 `repeat=30`。
  证据：`docs/logs/official_ur10_solver_benchmark.md`
- 已完成 Phase 6 的一轮竞争性优化，把 CUDA 主线固定到 `weight_level=2`。
  证据：`docs/logs/official_ur10_solver_benchmark.md`
  证据：`docs/logs/ablation_official_ur10.md`
- 已验证 `step clamp 0.45` 是无效尝试，并已回退。
  证据：`docs/logs/official_ur10_solver_benchmark.md`
- 已完成真实 full NCU profiling，并得出当前 kernel 不是 memory-bound 的结论。
  证据：`docs/logs/roofline_ncu_official_ur10.md`
- 已形成完整 Markdown 论文初稿并润色。
  证据：`docs/paper/paper_full.md`
  证据：`docs/paper/main.md`
  证据：`docs/paper/05_experiments.md`
  证据：`docs/paper/08_conclusion.md`
- **已建立 A0-A6 完整消融级别并完成实测**。
  证据：`docs/logs/ablation_official_ur10.md`
  证据：所有 7 个独立 CMake target (`standard_robot_cuda_runner_A0` 到 `A6`)
- **已补齐 numeric_dls 全规模参考结果**。
  证据：`data/results/ur10_numeric_dls_N100_seed42_repeat30_zero_seed_summary.json`
  证据：`data/results/ur10_numeric_dls_N500_seed42_repeat30_zero_seed_summary.json`
  证据：`data/results/ur10_numeric_dls_N1000_seed42_repeat30_zero_seed_summary.json`
  证据：`data/results/ur10_numeric_dls_N5000_seed42_repeat3_home_seed_summary.json`
- **已建立 A7/A8 消融级别并完成实测**。
  证据：`standard_robot_cuda_runner_A7` 和 `standard_robot_cuda_runner_A8` 独立 target
  证据：N=100/500/1000/5000 四规模 A7 benchmark 数据
  证据：A8 CUDA Graph ~0% overhead vs direct launch 验证
- **已完成 7DOF Panda 独立验证**。
  证据：`experiments/7dof_test/` 完整工作目录
  证据：CUDA FK vs CPU FK 最大误差 2.78e-16
  证据：CUDA DLS IK 收敛率 50% 与 CPU 参考一致
- **已完成混合精度优化（A7）并关闭 N=5000 差距**。
  证据：A7 在 N=5000 上以 180,962 targets/s 反超 cuRobo 1.25 倍
  证据：`docs/data/mixed_precision_ablation.csv`
- **已完成共享内存 bank conflict 深度分析**。
  证据：A7 仅 1,295 store conflicts（1% wavefronts），非瓶颈
  证据：`docs/data/ncu_profiling.csv`
- **已生成实验数据 CSV 归档到 `docs/data/`**。
  证据：5 个 CSV 文件 + README 说明文档

## 7. 论文后续待办清单

以下待办按优先级分组。已完成项已标记 ✅。

### Group A：最优先，直接影响论文强度

#### ✅ A2. 补齐 `A0-A5` 独立消融（已完成）

- **已建立** A0-A6 共 7 个独立可编译 CMake target（`standard_robot_cuda_runner_A0` 到 `A6`）
- **已完成** N=100/500/5000 完整实测
- **关键发现**：A5（自适应阻尼）是最关键的单项优化（N=5000 吞吐提升 147%），A6（步长钳位+分支对齐）收益为负
- **证据**：`docs/logs/ablation_official_ur10.md`

#### ✅ A3. 更新论文实验与讨论章节（已完成）

- **已创建** `docs/paper/paper_full.md`——8 个章节 + 参考文献的单一整合文档
- **已完成** nature-polishing 学术语言润色
- **论文数字与日志/JSON 一致**（三重校验已通过）

#### ✅ A1. 混合精度（A7）已关闭 N=5000 差距（已完成）

- **目标**：在完全不改变公平条件的前提下，继续提升 CUDA 在 `N=5000` 上的吞吐。
- **实现方案**：A7 混合精度（FP32 FK/Jacobian/Hessian + FP64 LDLT/阻尼/误差）
- **结果**：A7 在 N=5000 上达到 **180,962 targets/s**，反超 cuRobo（144,855 targets/s）**1.25 倍**
- **关键发现**：混合精度在所有批量上带来 2.0–2.5× 加速，收敛率保持 99.98%+
- **证据**：`docs/data/ablation_ur10.csv`、`docs/data/mixed_precision_ablation.csv`

### Group B：次优先，补齐参考证据

#### ✅ B1. 补齐 `numeric_dls` 全规模主参考（已完成）

- N=100 zero_seed repeat=30: 45.7 targets/s, ConvRate=0.010
- N=500 zero_seed repeat=30: 48.0 targets/s, ConvRate=0.014
- N=1000 zero_seed repeat=30: 49.8 targets/s, ConvRate=0.015
- N=5000 home_seed repeat=3: 81.9 targets/s, ConvRate=0.343
- 关键发现：CPU 数值 DLS 对 seed 高度敏感（zero_seed 下收敛率仅~1%）

#### ✅ B2. N=5000 NCU profiling 完成（已完成）

- A5 N=5000 `.ncu-rep` 已生成：`data/profiling/ur10_cuda_A5_n5000_full_zero_seed.ncu-rep`
- A7 N=100 NCU memory analysis 完成：`data/profiling/ur10_A7_n100_memory.ncu-rep`
- 关键发现：bank conflicts 从 A6 的 3,522 降至 A7 的 1,295（−49%），但仅占总波前 1%
- 证据：`docs/logs/roofline_ncu_official_ur10.md`、`docs/data/ncu_profiling.csv`

#### ✅ B3. 共享内存 bank conflict 深度分析完成（已完成）

- A6（FP64）：2,532 store bank conflicts，0 load bank conflicts
- A7（混合精度）：1,295 store bank conflicts（-49%），0 load bank conflicts
- **结论**：bank conflict 非瓶颈（~1% wavefronts），预计消除后加速 < 0.5%
- 证据：`docs/data/ncu_profiling.csv`

- 若大 batch 追赶仍卡住，进一步定位共享访问冲突、寄存器 live range、stall 分类。

### Group C：扩展项

#### C1. 7DOF Panda 扩展

- 当前状态：只有模型文件和扩展计划，无正式实现和 benchmark。
- 禁止事项：正式 benchmark 完成前，绝对不能把 7DOF 写成已有实验结果。

## 8. 当前使用到的工作流 / skill 说明

当前工程已经约定以下工作流映射，后续 AI 不需要重新命名：

- CUDA 优化主工作流：`cuda-optimizer`
- NCU / Roofline 分析：`ncu-rep-analyzer`
- 论文与日志联动写作：`cuda-paper-code-review`

### 8.1 `cuda-optimizer`

- 什么时候用：
  改 CUDA kernel、改线程映射、改共享内存布局、改小矩阵求解、改数值调度时。
- 输入：
  当前 `.cu` 文件、当前 benchmark 结果、当前瓶颈认识。
- 输出：
  一轮可验证的 CUDA 优化尝试。

### 8.2 `ncu-rep-analyzer`

- 什么时候用：
  做 Phase 7 profiling、看 NCU 报告、写 Roofline / bottleneck 结论时。
- 输入：
  `.ncu-rep` 文件或当前 CUDA 可执行路径。
- 输出：
  bottleneck 分类、关键指标、结构化分析结论。

### 8.3 `cuda-paper-code-review`

- 什么时候用：
  实验结果出来后，要同步更新日志和论文时。
- 输入：
  当前结果文件、当前日志、当前论文章节。
- 输出：
  对齐后的论文与日志叙述。

### 8.4 后续 AI 的推荐调用顺序

后续工作必须优先按以下顺序组织：

1. 先改 CUDA / 实现实验变体；
2. 再跑 benchmark；
3. 再做 NCU / Roofline；
4. 最后更新日志和论文。

不要倒过来做。论文和日志永远应该跟真实结果走，不应该反过来驱动结果。

## 9. 外部引用与依赖说明

### 9.1 外部来源

- 官方 UR10 来源仓库：
  `https://github.com/UniversalRobots/Universal_Robots_ROS2_Description`
  为什么要看：
  这是标准 UR10 模型的唯一主来源，后续若换模型版本、复核 URDF、复核参数，都必须从这里追。

- cuRobo
  为什么要看：
  当前主 GPU 参考对手就是它；如果后续要解释 `N=5000` 差距，必须理解它的 batch 组织方式。

- PyRoki
  为什么要看：
  它是当前论文中的 GPU 参考 solver 之一，虽然表现较弱，但仍是公平性叙事的一部分。

- PyKDL
  为什么要看：
  它对应当前真实 CPU 几何 baseline。

- Nsight Compute
  为什么要看：
  Phase 7 的 profiling 与 Roofline 解释完全依赖它。

### 9.2 本地参考文件

- `docs/goal.txt`
  为什么要看：
  这是最初的项目目标和边界来源，能帮助后续 AI 理解为什么本工程必须强调“批量 IK + CUDA 架构优化”，而不是路径规划。

- `docs/PROJECT_OVERVIEW.md`
  为什么要看：
  它记录了旧工程背景，有助于理解哪些内容是从旧工程迁移来的、哪些是必须摆脱的。

- `standard_robot_cuda_ik/README.md`
  为什么要看：
  这是子项目最短使用说明和典型流程入口。

- `cuda_low_level_optimization/`
  为什么要看：
  这是旧自定义工程基线。后续若需要继续迁移思想或做对照，可回到这里查原始实现，但不能再把它当成当前论文主线。

## 10. 下一位 AI 的起步步骤

下一位 AI 接手时，固定按以下顺序开始：

1. 先读本文件：
   `standard_robot_cuda_ik/docs/WORK_HANDOFF_AND_PAPER_TODO.md`
2. 再读：
   `docs/logs/final_acceptance_report.md`
3. 再读：
   `docs/logs/official_ur10_solver_benchmark.md`
4. 再读：
   `docs/logs/ablation_official_ur10.md`
5. 再读：
   `docs/paper/main.md`
   `docs/paper/05_experiments.md`
6. 最后再进入代码和 benchmark：
   `src/cuda/`
   `benchmark/`
   `tools/`

最容易犯错的三件事：

- 不要把当前随机可达 target 的批量 IK benchmark 写成 motion planning。
- 不要私自更换 target、seed、TCP 或阈值，再声称和当前结果可直接比较。
- 不要把 7DOF Panda 扩展计划写成已完成实验结果。

## 11. 当前建议的直接下一步

经过本轮工作，A0-A6 消融、numeric_dls 全规模参考和论文整合均已 **✅ 完成**。如果下一位 AI 需要立刻开始工作，建议直接从以下方向开始：

1. 先阅读 `docs/logs/final_acceptance_report.md` 了解当前完整状态。
2. 再阅读 `docs/logs/ablation_official_ur10.md` 了解消融结果。
3. 再阅读 `docs/paper/paper_full.md` 了解论文初稿。
4. 然后优先实现 **N=5000 追赶 cuRobo**：
   - 探索混合精度求解（FP32 主体 + FP64 关键路径）
   - 对 N=5000 做完整 NCU profiling
   - 评估 CUDA Graph 对 launch overhead 的改善
5. 其次考虑 **7DOF Panda 扩展**（如果主叙事已足够强）。

这是当前最能提高论文说服力的继续方向。
