# CUDA IK Acceleration Workspace

CUDA 加速批量逆运动学求解 — UR10 6-DOF 机械臂，SM 8.9 (Ada Lovelace)，CUDA 13.3。

**当前方法: OPT4C (CUDA-V4-Final-K16)** = 解析 Jacobian + Levenberg-Marquardt + Sobol-K16 种子 + 关节极限障碍 + 平滑重排 + target-block 并行映射 + 融合块内选择。

## 项目结构

```text
standard_robot_cuda_ik/   ← 唯一活跃项目
history/                  ← 历史版本归档（V1-V4, 论文草稿, 实验日志）
external/                 ← 参考求解器（cuRobo, PyRoki, HJCD-IK，只读）
```

## 快速开始

```bash
cd standard_robot_cuda_ik
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## 当前实验结论

默认对比口径来自 `standard_robot_cuda_ik/data/results/latest/baseline_snapshot.csv`：

| N | CUDA OPT4C throughput/s | CUDA Strict SR | CUDA pos_p95 mm | cuRobo-Graph throughput/s | cuRobo Strict SR | cuRobo pos_p95 mm |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 14094 | 0.960 | 4.38 | 10858 | 0.860 | 74.06 |
| 500 | 16357 | 0.954 | 4.34 | 44905 | 0.838 | 108.96 |
| 1000 | 17816 | 0.954 | 4.56 | 74143 | 0.841 | 100.09 |
| 5000 | 18659 | 0.954 | 4.56 | 144498 | 0.843 | 74.08 |

结论：
- N=100 时，CUDA OPT4C 的吞吐和质量均优于默认 cuRobo-Graph。
- N>=500 时，默认 cuRobo-Graph 的吞吐更强。
- CUDA OPT4C 的 Strict SR 和 pos_p95 更稳定。
- cuRobo quality-tuned 配置可以显著提高质量，但吞吐会下降；不能把 default cuRobo-Graph 与 quality-tuned cuRobo 混为同一实验口径。

## 关键文档

- `CLAUDE.md` — AI 辅助开发配置与项目架构说明
- `history/README.md` — 历史版本归档清单
- `standard_robot_cuda_ik/PROJECT.md` — 项目交接文档
