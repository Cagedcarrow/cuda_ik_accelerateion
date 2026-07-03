# Official UR10 Solver Benchmark

## Scope

- Robot: `ur10`
- Official model source: `UniversalRobots/Universal_Robots_ROS2_Description`
- Source ref: `4.3.1`
- Source commit: `ae333289875f9ba5a9ea6649a54036efb5ccabee`
- Target profile: `smooth_joint_trajectory_from_home`
- Main seed strategy: `zero_seed`
- Tolerances: `position_error < 0.03 m`, `orientation_error < 0.5236 rad`
- Python entry: `python3 standard_robot_cuda_ik/benchmark/run_all.py`

本日志只记录已经真实完成并保留结果文件的标准化实验。这里的 target/seeds 测试是可达位姿批量 IK 基准，不是路径规划，也不是完整运动规划流水线。

## Main Comparison Solvers

主对比 solver 定义为当前已经满足以下条件的路径：

- 同一 `urdf/ur10_official.urdf`
- 同一 `tool0`
- 同一 `target/seeds`
- 同一误差阈值
- 重复 30 次

当前进入主表的是：

- `cuda`
- `curobo`

### Main Table

| N | Solver | Repeat | p50 host_api_total ms | p95 ms | p99 ms | Throughput targets/s | ConvRate | Avg Pos Err m | Avg Rot Err rad | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 100 | cuda | 30 | 2.635 | 2.635 | 2.635 | 38015.0 | 1.000 | 0.010222 | 0.046140 | kernel/gpu/host 三类时间由 CUDA runner 输出；`max_iter=160`；Phase 6 固化 `weight_level=2` |
| 100 | curobo | 30 | 34.904 | 38.366 | 46.251 | 2850.1 | 1.000 | 0.000000 | 0.000000 | 单 external seed + LM seed stage(1 seed)；预热 solve 已排除；当前记录的是 host API 总时间 |
| 500 | cuda | 30 | 11.112 | 11.112 | 11.112 | 45014.7 | 1.000 | 0.013034 | 0.058045 | 同上 |
| 500 | curobo | 30 | 33.601 | 42.077 | 46.262 | 14573.9 | 1.000 | 0.000000 | 0.000000 | 同上 |
| 1000 | cuda | 30 | 21.970 | 21.970 | 21.970 | 45530.4 | 1.000 | 0.010292 | 0.037275 | 同上 |
| 1000 | curobo | 30 | 33.899 | 35.927 | 36.389 | 29517.6 | 1.000 | 0.000000 | 0.000000 | 同上 |
| 5000 | cuda | 30 | 94.944 | 94.944 | 94.944 | 52666.4 | 1.000 | 0.011699 | 0.050415 | 同上 |
| 5000 | curobo | 30 | 34.537 | 36.758 | 38.743 | 144855.2 | 1.000 | 0.000000 | 0.000000 | 同上；不再 OOM |

## Reference Solvers

参考 solver 的结果保留，但当前不进入主公平对比表。

### Reference Table

| Solver | N | Repeat | p50 host_api_total ms | Throughput targets/s | ConvRate | Status | Fairness / Limitation |
|---|---:|---:|---:|---:|---:|---|---|
| numeric_dls | 100 | 30 | 2177.786 | 45.7 | 0.010 | reference | 同一 target/seeds 路径；CPU 数值基线，`zero_seed` 下收敛显著偏弱 |
| numeric_dls | 1000 | 3 | 12233.212 | 82.3 | 0.332 | exploratory | 已用同一 target/seeds，但未完成 `repeat=30` |
| pyroki | 100 | 30 | 1820.774 | 54.3 | 0.300 | reference | JIT 预热已排除；当前已接入共享外部 seed，但吞吐和收敛仍较弱 |
| kdl | 100 | 30 | 100.665 | 986.1 | 1.000 | reference | 真实 PyKDL 基线；同一 URDF / target / seed / 阈值链路 |

## Competitive Optimization Outcome

Phase 6 的竞争性优化以“相同 target / 相同 seed / 相同 URDF-TCP / 相同阈值”为硬约束，没有通过换数据或放宽判据来追结果。当前阶段结论如下：

- `cuda` 在 `N=100/500/1000/5000` 上都完成了 `repeat=30`。
- `cuda` 在主公平 `zero_seed` 下，对四个主规模全部达到 `1.000` 收敛率。
- 在不改 target / seed / 阈值的前提下，把 CUDA 主线从硬编码 `weight_level=0` 调整为 `weight_level=2` 后，四个主规模吞吐都提升，同时仍保持 `1.000` 收敛率。
- 更激进的 `step clamp 0.45` 试验已经真实做过，但 `N=5000` 吞吐回落到约 `4.94e4 targets/s`，因此已回退，不把无效尝试留在主线。
- `cuda` 在 `N=100/500/1000` 上领先公平单 seed `curobo` 的 host API 吞吐。
- `curobo` 在同一公平 `zero_seed` 设定下已经不再 OOM，并且在 `N=5000` 上明显快于当前 CUDA。
- 这意味着 Phase 6 的“公平性修复”已经完成，但“性能追赶”并没有在所有规模上结束：
  - `cuda` 在小到中等 batch 上仍有优势；
  - `curobo` 在超大 batch 上展现出更强的批处理吞吐。

因此，本阶段可以宣称的是：**在当前标准化工程下，自研 CUDA 已经进入第一梯队，并在主公平 `zero_seed` GPU 对比中取得了稳定的满收敛表现；最新一轮 Phase 6 优化把 `N=5000` 吞吐从 `42397.4` 提升到 `52666.4 targets/s`，但仍落后于公平单 seed 的 cuRobo，需要继续执行竞争性优化。**

## Failure Isolation Evidence

`run_all.py` 已支持单 solver 失败隔离：当某 solver 抛出异常时，会写结构化错误日志并继续执行其他 solver。

- 演示命令：
  `python3 standard_robot_cuda_ik/benchmark/run_all.py --robot badrobot --seed 42 --N 100 --repeat 1 --solver all --seed-strategy zero_seed`
- 结果 Markdown：
  `standard_robot_cuda_ik/data/results/badrobot_all_N100_seed42_repeat1_zero_seed.md`
- 错误日志目录：
  `standard_robot_cuda_ik/data/results/errors/`

## Result Files

### Main GPU comparison

- `standard_robot_cuda_ik/data/results/ur10_cuda_N100_seed42_repeat30_zero_seed_summary.json`
- `standard_robot_cuda_ik/data/results/ur10_cuda_N500_seed42_repeat30_zero_seed_summary.json`
- `standard_robot_cuda_ik/data/results/ur10_cuda_N1000_seed42_repeat30_zero_seed_summary.json`
- `standard_robot_cuda_ik/data/results/ur10_cuda_N5000_seed42_repeat30_zero_seed_summary.json`
- `standard_robot_cuda_ik/data/results/ur10_curobo_N100_seed42_repeat30_zero_seed_summary.json`
- `standard_robot_cuda_ik/data/results/ur10_curobo_N500_seed42_repeat30_zero_seed_summary.json`
- `standard_robot_cuda_ik/data/results/ur10_curobo_N1000_seed42_repeat30_zero_seed_summary.json`
- `standard_robot_cuda_ik/data/results/ur10_curobo_N5000_seed42_repeat30_zero_seed_summary.json`

### Reference runs

- `standard_robot_cuda_ik/data/results/ur10_numeric_dls_N100_seed42_repeat30_zero_seed_summary.json`
- `standard_robot_cuda_ik/data/results/ur10_numeric_dls_N1000_seed42_repeat3_summary.json`
- `standard_robot_cuda_ik/data/results/ur10_pyroki_N100_seed42_repeat30_zero_seed_summary.json`
- `standard_robot_cuda_ik/data/results/ur10_kdl_N100_seed42_repeat30_zero_seed_summary.json`
