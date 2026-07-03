# History — 历史版本归档

本目录包含 CUDA 批量逆运动学求解项目的所有历史版本代码、实验数据、论文草稿与日志。
当前活跃版本为 **OPT4C (CUDA-V4-Final-K16-OPT4C)**，位于 `../standard_robot_cuda_ik/`。

## 版本演化时间线

| 版本 | 时间 | 方法 | 备注 |
|------|------|------|------|
| **V1 (A0-A8 DLS)** | 2026-06-10 ~ 06-12 | 数值 Jacobian + DLS + 消融控制 | 首次实现 CUDA IK，A0-A8 消融实验 |
| **V2** | 2026-06-15 | DLS + 解析 Jacobian | 尝试用解析 Jacobian 替换数值差分（CUDA 层面未完成） |
| **V3** | 2026-06-15 | LM + 多种子策略 (Python) | 纯 Python 原型，探索 LM 求解器和 Sobol 种子 |
| **V4 Python** | 2026-06-15 | LM + Limit Barrier + Smoothness Rerank (Python) | 纯 Python 原型，引入极限障碍和平滑重排 |
| **V4 CUDA (pre-OPT4C)** | 2026-06-16 | V4 的 CUDA 移植（OPT4 前版本） | 当前版本的前身，含完整文档和论文草稿 |
| **OPT4C (当前)** | 2026-06-16 ~ | 解析 Jacobian + LM + Sobol-K16 + Limit Barrier + Smoothness Rerank + Block-Target 映射 | 当前主线 |

## 目录清单

### `ablation_dls/` — V1 A0-A8 消融实验代码

| 路径 | 说明 |
|------|------|
| `src/cuda_ik_6dof.cu` | DLS 内核：batch solve, multi-seed, continuity cost, top-K filter, ablation A0, mixed precision |
| `src/cuda_benchmark_runner.cu` | A0-A8 消融实验 runner main() |
| `src/cuda_memory.cu` | RAII DeviceBuffer 模板实例化 |
| `headers/cuda_ik_6dof.h` | DLS 内核启动函数声明 |
| `headers/cuda_memory.h` | DeviceBuffer<T> 和 ConstantMemory<T> RAII 封装 |
| `headers/cuda_collision.h` | GPU 碰撞检测接口（AABB + OBB/GJK，未使用） |

**数据来源**: 从 `standard_robot_cuda_ik/src/cuda/` 和 `include/` 移动而来。

### `paper_drafts/` — 论文草稿

| 路径 | 说明 |
|------|------|
| `a0_a8_dls/` | A0-A8 DLS 代论文草稿 + MATLAB 图表脚本 + 实验结果 CSV |
| `a0_a8_dls/data/` | 消融实验 CSV (`ablation_ur10.csv`, `solver_comparison.csv`, `ncu_profiling.csv`, `mixed_precision_ablation.csv`, `7dof_verification.csv`) |
| `latex_backup_v2/` | LaTeX 论文备份（v2，含 OPT4C 重写版本），包含 `.tex` 源码、编译后 PDF、审稿回复、专利交底书 |

**数据来源**: `docs/paper/`, `docs/data/`, `docs/latex论文_backup_v2_*`。

### `experiment_logs_dls/` — V1 实验日志 (8 个文件)

| 文件 | 内容 |
|------|------|
| `7dof_cuda_extension_plan.md` | Franka Panda 7DOF 扩展计划 |
| `cuda_6dof_design_official_ur10.md` | CUDA 设计：1 block/target, 128 threads, 4 warps |
| `official_ur10_model_verification.md` | UR10 模型来源验证（MD5、FK 校验） |
| `official_ur10_solver_benchmark.md` | 标准化基准测试（targets + seeds） |
| `roofline_ncu_official_ur10.md` | Nsight Compute Roofline 分析 |
| `solver_failure_isolation.md` | 基准测试框架故障隔离验证 |
| `target_generation_seed42.md` | 目标位姿生成报告（seed=42, N=100-5000） |
| `ur10_seed42_reproducibility.md` | 可复现性核查（资产哈希） |

**数据来源**: `docs/logs/`。

### `v4_python_prototype_experiments/` — V4 Python 原型实验

| 文件 | 说明 |
|------|------|
| `run_v4_m0.py` | M0: V3 freeze + joint-limit barrier + q_prev smoothness |
| `run_v4_m1_m2.py` | M1+M2: limit barrier weight sweep + smoothness candidate reranking |
| `run_v4_finalize.py` | V4 定型：limit weight sweep + 失败案例诊断 |

纯 Python 实现（无 CUDA），使用 `tools/robot_model.py`。

**数据来源**: `standard_robot_cuda_ik/experiments/`。

### `previous_archive/` — 完整历史归档 (2026-06-16 清理时生成)

包含所有历史版本的完整项目目录：

| 子目录 | 说明 |
|--------|------|
| `standard_robot_cuda_ik/` | V1 完整项目（含 CPU 基线求解器和碰撞检测 .cu） |
| `standard_robot_cuda_ik_v2/` | V2 完整项目（DLS + 解析 Jacobian 实验） |
| `standard_robot_cuda_ik_v3/` | V3 Python LM 原型（多种子策略探索） |
| `standard_robot_cuda_ik_v4/` | V4 Python 原型（Limit Barrier + Smoothness Rerank） |
| `standard_robot_cuda_ik_v4_cuda/` | V4 CUDA 移植完整项目（当前 OPT4C 的前身，含全部文档、论文、日志） |
| `ur10_robot_cuda_ik_cuda/` | 空目录（未启动的独立项目） |
| `cleanup_removed_active_nsight_binary_with_old_paths/` | 含旧路径的 NCU 报告 |
| `README.old.md` | 旧版项目 README (31KB，2026-06-14) |

**注意**: `standard_robot_cuda_ik_v4_cuda/` 中的 `cuda_v4_runner.cu` 与当前版本 **完全相同**。该目录还包含 V4 时代的完整文档（移植日志、正确性报告、基准报告、实验报告）以及最新的 OPT4C 论文草稿（`paper/final/`）。

### `project_planning/` — 项目规划文档

| 文件 | 说明 |
|------|------|
| `goal.txt` (31KB) | 原始项目任务书（2026-06-10），规定了标准化 CUDA IK 项目的全部需求 |
| `项目整理计划.md` | 空文件（未编写） |

### `scripts_old/` — 过时脚本

| 文件 | 说明 |
|------|------|
| `finalize_v4_cuda_reports.py` | 合并旧 CSV 命名规范的文件（`cuda_v4_static_benchmark.csv` 等），当前使用 `cuda_opt4c_*` 命名 |

### `data_old/` — 历史数据 (Python .npy 格式)

| 路径 | 说明 |
|------|------|
| `v4_targets_N*.npy` (2 个) | V4 Python 原型使用的目标位姿 |
| `sobol_K*_N*_bank00.npy` (4 个) | Sobol 种子库 |

当前 CUDA runner 使用 `data/cuda_inputs/` 中的 `.raw` 二进制文件。

### `root_artifacts/` — 根级零散文件

| 文件 | 说明 |
|------|------|
| `texput.log` | XeTeX 编译失败日志（2026-06-15） |
| `types.json` | 空 JSON `{}`（用途不明） |

### `journal_references/` — 期刊参考文献

`系统工程与电子技术` 期刊投稿的参考文献库，包含 24 篇 GPU-加速 IK 和 CUDA 优化的相关论文，及期刊模板。

---

## 在历史代码中查找

- **DLS 求解器（数值 Jacobian）** → `ablation_dls/src/cuda_ik_6dof.cu`
- **CPU 基线求解器（KDL, numeric DLS）** → `previous_archive/standard_robot_cuda_ik/src/cpu_baseline/`
- **V4 Python 原型算法** → `v4_python_prototype_experiments/` 和 `previous_archive/standard_robot_cuda_ik_v4/`
- **V4 CUDA 移植文档** → `previous_archive/standard_robot_cuda_ik_v4_cuda/docs/`
- **OPT4C 最新论文草稿** → `previous_archive/standard_robot_cuda_ik_v4_cuda/paper/final/`
- **A0-A8 消融数据** → `paper_drafts/a0_a8_dls/data/`
- **原始项目需求** → `project_planning/goal.txt`

## 当前活跃项目

参见 `../standard_robot_cuda_ik/` 和 `../CLAUDE.md`。

---

*归档完成日期: 2026-07-03*
