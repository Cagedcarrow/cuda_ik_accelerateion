# 7DOF Panda CUDA Batch IK — 独立验证实验

## 目的

验证 CUDA 批量 IK 框架可以从 6DOF (UR10) **数学一致地**扩展到 7DOF (Franka Panda)。

**这个实验只验证基本逻辑正确性，不产生论文用的 benchmark 数据。**

## 关键差异（6DOF → 7DOF）

| 项目 | UR10 (6DOF) | Panda (7DOF) |
|------|-------------|--------------|
| Active joints | 6 | 7 |
| Jacobian | 6×6 (方阵) | 6×7 (非方阵) |
| Hessian | 6×6 SPD | 7×7 SPD |
| LDLT 求解器 | ldlt_solve_6x6 | ldlt_solve_7x7 |
| FK segments | 6 段 | 7 段 |
| 冗余自由度 | 无 | 有 (7-6=1) |
| 常量内存 origins | 6×16 = 96 doubles | 7×16 = 112 doubles |
| 常量内存 axes | 6×3 = 18 doubles | 7×3 = 21 doubles |

## 文件清单

- `panda_fk_reference.py` — Python Panda FK + CPU DLS IK 参考
- `panda_7dof_kernel.cu` — CUDA 7DOF DLS kernel（最简版）
- `panda_7dof_runner.cu` — CUDA host runner
- `test_fk.py` — FK 正确性验证
- `CMakeLists.txt` — 独立编译配置
- `run_7dof_test.sh` — 一键运行脚本

## 复用（不改动）

- `tools/robot_model.py` — load_robot_model()
- `urdf/panda_7dof.urdf`
- CUDA 基础设施：Rodrigues, mat44_mul, pose_error

## 验证标准

- [x] Python FK 与 CPU FK 自洽（FK(0) 验证通过）
- [x] Python CPU DLS IK 求解 N=10 随机目标：**5/10 (50%) 收敛**
- [x] CUDA FK 与 CPU FK 一致：**最大误差 2.78e-16**（机器精度）
- [x] CUDA DLS IK kernel 运行不崩溃
- [x] CUDA DLS IK 收敛率：**5/10 (50%)**，与 Python CPU 参考一致

## 禁止

- 不写入论文作为实验结果
- 不与主 CUDA 框架交叉依赖
