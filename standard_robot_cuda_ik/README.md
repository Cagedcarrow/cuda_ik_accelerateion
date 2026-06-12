# Standard Robot CUDA IK

独立于旧的自定义 UR10+铲斗工程，本子项目是 **标准机械臂批量逆运动学 CUDA 底层优化** 的主研究载体，包含完整的 CUDA 内核、多求解器公平对比框架、可复现实验数据与论文素材。

> **项目主页**: 参见根目录 [`README.md`](../README.md) —— 包含完整的工程背景、GPU核心设计、性能结果和CUDA框架优势分析。

## 目标

- 统一官方标准机器人模型（UR10/UR5/Panda 7-DOF）、TCP、关节限位和目标位姿生成流程。
- 用同一批 target/seed 公平比较 `cuda`（本项目）、`curobo`、`pyroki`、`kdl`、`numeric_dls` 五种求解器。
- 将 `1 block/target`、`PaddedMat6x8`、寄存器 `LDLT`、constant memory、自适应阻尼、混合精度等优化技术全面应用于标准 UR10。
- 输出可复现实验数据、结构化日志和 Markdown 论文草稿。

## 当前实现（已完成）

- **CUDA 内核**: 9 个消融级别（A0-A8），覆盖从全局内存基线到混合精度 + CUDA Graph 的完整优化层级
- **多求解器对比框架**: 5 种求解器的统一 benchmark 入口 [`benchmark/run_all.py`](benchmark/run_all.py)
- **数据资产**: 36 个 target 文件（N=100→10000）、96 个 seed 文件（4 策略 × 12 规模）、11 个结果 CSV、5 个 Nsight Compute 剖析报告
- **论文图表**: 5 张论文图（`data/figures/`）及生成脚本
- **7-DOF 扩展**: Panda 7-DOF 验证（`experiments/7dof_test/`）

## 典型流程

```bash
cd standard_robot_cuda_ik

# 1) 生成标准 UR10 数据与常量
python3 tools/generate_standard_assets.py --robot ur10 --seed 42

# 2) 编译 CUDA/C++ 入口（生成 A0-A8 共 10 个可执行文件）
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j

# 3) 运行 CUDA B5（混合精度）单求解器测试
./build/standard_robot_cuda_runner_A7 \
    --targets data/targets/ur10_seed42_N1000.csv \
    --seeds data/seeds/ur10_seed42_zero_seed_N1000.json \
    --repeat 30

# 4) 运行跨求解器完整对比
python3 benchmark/run_all.py --robot ur10 --seed 42 --N 1000 --repeat 30
```

## 目录

| 目录 | 说明 |
|:---|:---|
| `src/cuda/` | CUDA 6-DOF IK 内核、工具函数、benchmark runner、碰撞检测 |
| `src/cpu_baseline/` | CPU 参考求解器（KDL, Numeric DLS） |
| `include/` | C++ 头文件及自动生成的 UR10 模型常量 |
| `benchmark/` | 跨求解器统一对比框架（5 种 solver wrapper） |
| `tools/` | 资产生成与验证脚本（target/seed 生成、模型验证） |
| `config/` | YAML 配置（benchmark 参数、机器人定义、目标生成） |
| `urdf/` | 官方 UR10/UR5/Panda 7-DOF 模型文件 |
| `data/` | 所有实验数据 —— 详见 [`data/README.md`](data/README.md) |
| `experiments/` | 实验工作区（7-DOF 扩展等） |
| `docs/` | 子项目文档与日志 |

## 性能速览

| N | CUDA B5 (混合精度) | cuRobo | 加速比 |
|:---:|:---:|:---:|:---:|
| 100 | 112,414 t/s | 3,118 t/s | **36.1×** |
| 500 | 158,251 t/s | 15,844 t/s | **10.0×** |
| 5000 | 168,683 t/s | 155,059 t/s | **1.09×** |

Medium 阈值 (10mm/5°), repeat=30, zero_seed, RTX 4070 Laptop GPU.  
完整性能数据参见根目录 [README.md](../README.md) 第 4 节。
