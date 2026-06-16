# Standard Robot CUDA IK Project Handoff

更新时间：2026-06-16 23:11 CST (Asia/Shanghai)

本文件是当前项目的唯一权威交接入口。切换到其他窗口或其他 AI 后，先读本文件，再读 `src/cuda/cuda_v4_runner.cu` 和 `data/results/latest/*.csv`。

## 1. 当前状态

当前主目录：

```text
/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik
```

本轮已按用户要求执行瘦身：旧 CSV、旧图片、旧文档、旧 paper、旧 logs、旧 build、旧 cleanup 产物已从本项目内直接删除；未创建本轮归档。当前只保留核心 CUDA 源码、构建配置、UR10 模型、输入资产、相关 Python 脚本和最新 CSV。

最终方法：

```text
CUDA-V4-Final-K16-OPT4C
= Analytical Jacobian
+ LM
+ Sobol-K16
+ Limit Barrier
+ Smoothness Rerank
+ target-block seed-parallel mapping
+ fused in-block selection
```

本轮图片没有重跑，当前项目内不保留旧 figures/paper/docs。最新结果只看 `data/results/latest/`。

## 2. 环境

| 项目 | 当前值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Driver | 610.43.02 |
| CUDA Toolkit | 13.3 / nvcc V13.3.33 |
| CMake build | pass |
| 主 runner | `build/standard_robot_cuda_v4_runner` |
| cuRobo default graph | pass |

## 3. 当前文件结构

```text
CMakeLists.txt
PROJECT.md
README.md
data/cuda_inputs/q_samples_N20_f64.raw
data/cuda_inputs/seeds_N1000_K16_q_f64.raw
data/cuda_inputs/seeds_N100_K16_q_f64.raw
data/cuda_inputs/seeds_N5000_K16_q_f64.raw
data/cuda_inputs/seeds_N500_K16_q_f64.raw
data/cuda_inputs/targets_N1000_T4x4_f64.raw
data/cuda_inputs/targets_N100_T4x4_f64.raw
data/cuda_inputs/targets_N5000_T4x4_f64.raw
data/cuda_inputs/targets_N500_T4x4_f64.raw
data/results/latest/cuda_opt4c_best_N100.csv
data/results/latest/cuda_opt4c_best_N1000.csv
data/results/latest/cuda_opt4c_best_N500.csv
data/results/latest/cuda_opt4c_best_N5000.csv
data/results/latest/cuda_opt4c_static.csv
data/results/latest/cuda_opt4c_static_N100.csv
data/results/latest/cuda_opt4c_static_N1000.csv
data/results/latest/cuda_opt4c_static_N500.csv
data/results/latest/cuda_opt4c_static_N5000.csv
data/results/latest/cuda_opt4c_timing_N100.csv
data/results/latest/cuda_opt4c_timing_N1000.csv
data/results/latest/cuda_opt4c_timing_N500.csv
data/results/latest/cuda_opt4c_timing_N5000.csv
data/results/latest/cuda_vs_curobo_summary.csv
data/results/latest/curobo_graph_compare.csv
data/results/latest/fk_check.csv
data/results/latest/latest_run_manifest.csv
data/results/latest/latest_run_status.csv
data/seed_banks/sobol_K16_N1000_bank00.npy
data/seed_banks/sobol_K16_N200_bank00.npy
data/seed_banks/sobol_K32_N1000_bank00.npy
data/seed_banks/sobol_K32_N200_bank00.npy
data/targets/v4_targets_N1000_seed42.npy
data/targets/v4_targets_N200_seed42.npy
experiments/run_v4_finalize.py
experiments/run_v4_m0.py
experiments/run_v4_m1_m2.py
include/standard_robot_cuda_ik/cuda_collision.h
include/standard_robot_cuda_ik/cuda_ik_6dof.h
include/standard_robot_cuda_ik/cuda_memory.h
include/standard_robot_cuda_ik/generated/ur10_model_constants.h
scripts/audit_curobo_quality_round2.py
scripts/audit_ur10_model_consistency.py
scripts/finalize_v4_cuda_reports.py
scripts/run_final_push.py
scripts/run_opt4_followup.py
scripts/run_opt4c_finalization.py
scripts/run_v4_cuda_plan.py
scripts/run_v4_curobo_compare.py
scripts/run_v4_enhancement_plan.py
src/cuda/cuda_benchmark_runner.cu
src/cuda/cuda_ik_6dof.cu
src/cuda/cuda_memory.cu
src/cuda/cuda_utilities.cuh
src/cuda/cuda_v4_runner.cu
tools/fetch_official_ur10.py
tools/robot_model.py
tools/verify_official_ur10.py
urdf/ur10_official.urdf
urdf/ur10_official_source.json
```

## 4. CUDA 实现思路

主入口是 `src/cuda/cuda_v4_runner.cu`。

关键实现点：

- `fk_with_frames_v4`：按 UR10 官方 URDF 常量执行 FK，同时输出末端位姿、各关节世界坐标 `p_joint` 和关节轴 `z_joint`。
- `analytical_jacobian_v4`：解析 Jacobian，线速度项为 `Jv_i = z_i x (p_ee - p_i)`，角速度项为 `Jw_i = z_i`。
- LM 更新：每轮构造 `H = J^T J + lambda I` 和 `g = J^T e + w_limit * grad_limit`，步长 clamp 后直接接受 `q_trial`，`lambda` 仅根据 `loss_new < loss_old` 做自适应缩放。
- Limit Barrier：`w_limit=0.03`，margin 固定为 0.087 rad。本轮 benchmark 使用 `--limit-gradient analytic`。
- OPT4C 线程映射：`ik_lm_multiseed_v4_block_target_kernel` 使用 `blockIdx.x` 对应 target，一个 block 内处理该 target 的 K=16 seeds，并在 block 内完成候选选择，避免额外全局候选选择 kernel。
- 输出选择规则：`success_rank -> near_limit -> pose_cost`，写入 best CSV。

UR10 模型入口：

- `urdf/ur10_official.urdf`
- `include/standard_robot_cuda_ik/generated/ur10_model_constants.h`
- joint order 固定为 `shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint`
- ee/tool frame 固定为 `tool0`

## 5. 最新 CUDA 数据

来源：`data/results/latest/cuda_opt4c_static.csv`

| N | K | warmup | repeat | gpu_ms_mean | throughput/s | strict_sr | pos_p95_all_mm | near_limit | monotonic | nan | inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 16 | 10 | 30 | 6.435 | 15539.2 | 0.960 | 4.385 | 0.010 | 1 | 0 | 0 |
| 500 | 16 | 10 | 30 | 30.592 | 16344.0 | 0.954 | 4.338 | 0.004 | 1 | 0 | 0 |
| 1000 | 16 | 10 | 30 | 56.123 | 17817.9 | 0.954 | 4.563 | 0.007 | 1 | 0 | 0 |
| 5000 | 16 | 10 | 30 | 270.411 | 18490.4 | 0.954 | 4.563 | 0.007 | 1 | 0 | 0 |

验收：

- N=100 Strict SR = 0.960，满足 `>= 0.90`。
- N=100 pos_p95_all = 4.385 mm，满足 `<= 10 mm`。
- N=100/500/1000/5000 全部 `nan_count=0`、`inf_count=0`。

## 6. 最新 cuRobo 对比

来源：

- CUDA：`data/results/latest/cuda_opt4c_static.csv`
- cuRobo：`data/results/latest/curobo_graph_compare.csv`
- 汇总：`data/results/latest/cuda_vs_curobo_summary.csv`

| N | CUDA throughput/s | CUDA Strict SR | CUDA pos_p95 mm | cuRobo throughput/s | cuRobo Strict SR | cuRobo pos_p95 mm | CUDA/cuRobo thr ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 15539.2 | 0.960 | 4.385 | 10508.7 | 0.870 | 74.324 | 1.479 |
| 500 | 16344.0 | 0.954 | 4.338 | 41815.6 | 0.836 | 115.637 | 0.391 |
| 1000 | 17817.9 | 0.954 | 4.563 | 64928.2 | 0.840 | 98.920 | 0.274 |
| 5000 | 18490.4 | 0.954 | 4.563 | 137148.3 | 0.844 | 75.047 | 0.135 |

解释：

- N=100：CUDA OPT4C 吞吐和质量均优于默认 cuRobo-Graph。
- N>=500：默认 cuRobo-Graph 吞吐更强，但 Strict SR 和 pos_p95 质量明显低于 CUDA OPT4C。
- cuRobo 对比不是等价算法对比：cuRobo 使用其内部 optimizer、CUDA Graph 和 seed/config 策略；本项目 CUDA 使用固定 Sobol-K16 和 V4 selection 规则。

## 7. 最新 CSV 清单

| path | role | size_bytes |
| --- | --- | --- |
| data/results/latest/cuda_opt4c_best_N100.csv | latest_generated_csv | 21636 |
| data/results/latest/cuda_opt4c_best_N1000.csv | latest_generated_csv | 215615 |
| data/results/latest/cuda_opt4c_best_N500.csv | latest_generated_csv | 107834 |
| data/results/latest/cuda_opt4c_best_N5000.csv | latest_generated_csv | 1081831 |
| data/results/latest/cuda_opt4c_static.csv | latest_generated_csv | 1716 |
| data/results/latest/cuda_opt4c_static_N100.csv | latest_generated_csv | 652 |
| data/results/latest/cuda_opt4c_static_N1000.csv | latest_generated_csv | 659 |
| data/results/latest/cuda_opt4c_static_N500.csv | latest_generated_csv | 658 |
| data/results/latest/cuda_opt4c_static_N5000.csv | latest_generated_csv | 654 |
| data/results/latest/cuda_opt4c_timing_N100.csv | latest_generated_csv | 2125 |
| data/results/latest/cuda_opt4c_timing_N1000.csv | latest_generated_csv | 2111 |
| data/results/latest/cuda_opt4c_timing_N500.csv | latest_generated_csv | 2152 |
| data/results/latest/cuda_opt4c_timing_N5000.csv | latest_generated_csv | 2126 |
| data/results/latest/cuda_vs_curobo_summary.csv | latest_generated_csv | 2093 |
| data/results/latest/curobo_graph_compare.csv | latest_generated_csv | 1658 |
| data/results/latest/fk_check.csv | latest_generated_csv | 24139 |
| data/results/latest/latest_run_status.csv | latest_generated_csv | 572 |

字段说明：

- `cuda_opt4c_static.csv`：N=100/500/1000/5000 的 CUDA 汇总主表。
- `cuda_opt4c_static_N*.csv`：每个 N 的 runner 原始 summary。
- `cuda_opt4c_best_N*.csv`：每个 target 的最佳候选解与误差。
- `cuda_opt4c_timing_N*.csv`：每次 repeat 的 H2D/kernel/D2H/e2e timing。
- `curobo_graph_compare.csv`：默认 cuRobo-Graph 对比结果。
- `cuda_vs_curobo_summary.csv`：论文/汇报优先读取的对比汇总表。
- `latest_run_status.csv`：本轮 build、FK、CUDA、cuRobo 状态。
- `latest_run_manifest.csv`：最新 CSV 的文件大小和 sha256。

## 8. 本轮状态

| item | status | detail |
| --- | --- | --- |
| cleanup_policy | done | old generated csv/md/images/paper/logs/build removed directly without new archive |
| build | pass | cmake configure and build completed; standard_robot_cuda_v4_runner generated |
| fk_check | pass | data/results/latest/fk_check.csv generated |
| cuda_static_all_N | pass | N=100/500/1000/5000 OPT4C analytic fp64 completed |
| cuda_N100_quality_gate | pass | strict_sr=0.96; pos_p95_all_mm=4.38451277 |
| cuda_nan_inf_gate | pass | all latest CUDA rows have nan_count=0 and inf_count=0 |
| curobo_graph | pass | rows=4; output=data/results/latest/curobo_graph_compare.csv |

## 9. 复现命令

构建：

```bash
cd /mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

FK check：

```bash
./build/standard_robot_cuda_v4_runner --mode fk_check   --seeds data/cuda_inputs/q_samples_N20_f64.raw   --best-csv data/results/latest/fk_check.csv
```

CUDA OPT4C static benchmark 示例：

```bash
./build/standard_robot_cuda_v4_runner --mode v4_static   --variant opt4c_block_target   --limit-gradient analytic   --graph-mode off   --precision-mode fp64   --fallback-mode none   --targets data/cuda_inputs/targets_N1000_T4x4_f64.raw   --seeds data/cuda_inputs/seeds_N1000_K16_q_f64.raw   --N 1000 --K 16 --warmup 10 --repeat 30   --summary-csv data/results/latest/cuda_opt4c_static_N1000.csv   --best-csv data/results/latest/cuda_opt4c_best_N1000.csv   --timing-csv data/results/latest/cuda_opt4c_timing_N1000.csv
```

cuRobo default graph：

```bash
python3 scripts/run_v4_curobo_compare.py
```

该脚本默认输出 `data/results/cuda_v4_curobo_compare.csv`；本轮已移动为 `data/results/latest/curobo_graph_compare.csv`。

## 10. 后续 AI 接手顺序

1. 读本文件 `PROJECT.md`。
2. 查看 `data/results/latest/cuda_vs_curobo_summary.csv` 和 `data/results/latest/latest_run_status.csv`。
3. 若要看 CUDA 实现，读 `src/cuda/cuda_v4_runner.cu`。
4. 若要看 UR10 模型，读 `urdf/ur10_official.urdf` 和 `include/standard_robot_cuda_ik/generated/ur10_model_constants.h`。
5. 若要重跑数据，先删除 `data/results/latest/*.csv`，再按第 9 节命令执行。
6. 若要写论文或重新生成图片，基于 `data/results/latest/*.csv` 重新生成，不要引用旧归档或旧文档。

## 11. 注意事项

- 当前目录故意不保留旧 docs/paper/figures，避免后续 AI 误读过期结论。
- 真机实验前必须使用 `ur_calibration` 提取真实 UR10 factory calibration，并用同一 calibrated model 重新生成 CUDA constants、Python evaluator 和 cuRobo robot config。
- 如果后续要恢复历史材料，可到根目录已有 `_archive_before_cleanup_*` 中查找，但论文/汇报应以本轮 `data/results/latest/*.csv` 为准。
