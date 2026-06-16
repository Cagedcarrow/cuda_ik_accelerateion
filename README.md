# CUDA IK Acceleration Workspace

当前唯一主项目目录：

```text
standard_robot_cuda_ik
```

最终主线：

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

旧 v 系列实验目录和旧主线已经移动到时间戳归档目录。根目录不再保留 v 系列项目作为主线入口。

## 当前实验结论

默认对比口径来自 `standard_robot_cuda_ik/data/results/final_push/baseline_snapshot.csv`：

| N | CUDA OPT4C throughput/s | CUDA Strict SR | CUDA pos_p95 mm | cuRobo-Graph throughput/s | cuRobo Strict SR | cuRobo pos_p95 mm |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 14094 | 0.960 | 4.38 | 10858 | 0.860 | 74.06 |
| 500 | 16357 | 0.954 | 4.34 | 44905 | 0.838 | 108.96 |
| 1000 | 17816 | 0.954 | 4.56 | 74143 | 0.841 | 100.09 |
| 5000 | 18659 | 0.954 | 4.56 | 144498 | 0.843 | 74.08 |

结论：

- N=100 时，CUDA-V4-OPT4C 的吞吐和质量均优于默认 cuRobo-Graph。
- N>=500 时，默认 cuRobo-Graph 的吞吐更强。
- CUDA-V4-OPT4C 的 Strict SR 和 pos_p95 更稳定。
- cuRobo quality-tuned 配置可以显著提高质量，但吞吐会下降；不能把 default cuRobo-Graph 与 quality-tuned cuRobo 混为同一实验口径。
- 真机实验前必须使用 `ur_calibration` 提取真实 UR10 factory calibration，并基于同一 calibrated URDF/YAML 重新生成 CUDA、Python evaluator 和 cuRobo 配置。

## 主要入口

```bash
cd /mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

关键文档：

- `standard_robot_cuda_ik/docs/final_paper_readiness_report_v2_opt4c.md`
- `standard_robot_cuda_ik/docs/final_push/final_push_summary.md`
- `standard_robot_cuda_ik/docs/final_push/curobo_quality_audit_round2.md`
- `standard_robot_cuda_ik/docs/final_push/curobo_paper_wording_after_audit.md`
- `standard_robot_cuda_ik/docs/final_push/ur10_model_consistency_audit.md`
- `standard_robot_cuda_ik/paper/final/cuda_ik_paper_latest.md`
- `standard_robot_cuda_ik/paper/final/cuda_ik_paper_latest_for_word.md`
