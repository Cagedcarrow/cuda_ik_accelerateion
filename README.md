# CUDA IK Acceleration — OPT4C

CUDA 加速批量逆运动学求解器，面向 UR10 6-DOF 机械臂，SM 8.9 (Ada Lovelace)，CUDA 13.3。

**当前方法: OPT4C** = 结构感知单核融合：解析雅可比 + Levenberg-Marquardt + Sobol-K16 低差异多起点 + 关节限位障碍 + Target-Block 线程映射 + 块内候选选择。

---

## 项目结构

```
cuda_ik_accelerateion/
├── CLAUDE.md                       # AI 辅助开发配置
├── README.md                       # 本文件
├── standard_robot_cuda_ik/         # 唯一活跃项目
│   ├── src/cuda/                   # CUDA 求解器源码
│   │   ├── cuda_v4_runner.cu       # 主核函数 + CLI
│   │   └── cuda_utilities.cuh      # 设备端辅助函数
│   ├── include/                    # 头文件（UR10 模型常量）
│   ├── build/                      # 编译输出
│   ├── data/
│   │   ├── experiments/            # ★ 论文全部实验数据
│   │   │   ├── README.md           # 数据-论文溯源表
│   │   │   ├── inputs/             # N=100-1000 目标/种子 raw 文件
│   │   │   ├── results/            # 主基准测试 CSV
│   │   │   └── 补充实验/           # K=1 消融、cuRobo K=16、FP32 混合精度
│   │   ├── cuda_inputs/            # 原始目标/种子文件
│   │   └── results/latest/         # 首轮基准输出
│   ├── 论文/                       # ★ 论文源文件
│   │   ├── paper.tex               # LaTeX 源文件（xelatex 编译）
│   │   ├── paper.pdf               # 已编译 PDF（11 页）
│   │   ├── paper.txt               # 查重用纯文本
│   │   ├── .latexmkrc              # 编译配置
│   │   ├── 绘图/                   # 6 张图（PDF+SVG+draw.io）
│   │   ├── 格式模板/               # 期刊投稿格式规范
│   │   ├── 计划/                   # 审稿意见与修改策略
│   │   └── 名词解释/               # 47 个 CUDA/机器人/数值计算术语
│   ├── scripts/                    # Python 脚本（基准编排、cuRobo 对比）
│   ├── tools/                      # UR10 FK 参考实现
│   └── urdf/                       # UR10 官方 URDF 模型
├── history/                        # 历史版本归档（仅供参考）
└── external/                       # 参考求解器（cuRobo, PyRoki，只读）
```

---

## 快速开始

```bash
cd standard_robot_cuda_ik
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 运行基准测试 (N=1000, K=16)
./build/standard_robot_cuda_v4_runner \
  --mode v4_static --variant opt4c_block_target \
  --limit-gradient analytic --precision-mode fp64 \
  --N 1000 --K 16 --max-iter 60 --repeat 30 --warmup 10 \
  --targets data/experiments/inputs/targets_N1000_T4x4_f64.raw \
  --seeds data/experiments/inputs/seeds_N1000_K16_q_f64.raw
```

---

## 论文核心结论

### 帕累托前沿定位（三方对比，$N=1000$，同等 16 种子）

| 方法 | Strict SR | 吞吐量 (t/s) | $p95$ (mm) | 定位 |
|------|-----------|-------------|-----------|------|
| cuRobo K=16 | **0.988** | 10,399 | **0.9** | 质量天花板 |
| **OPT4C K=16** | 0.954 | **18,226** | 4.6 | 吞吐优先（满足 5mm 阈值） |
| cuRobo K=1 | 0.840 | 72,624 | 88.7 | 极端吞吐（质量不可用） |

### 关键发现

| 实验 | 结论 |
|------|------|
| **K=1 消融** | SR 从 0.954 暴跌至 0.522 — 多起点策略是成功率的决定性来源 |
| **cuRobo K=16 公平对比** | cuRobo 质量更高 (SR 0.988)，但 OPT4C 吞吐为其 1.75 倍 |
| **FP32 混合精度** | 仅 +2% 吞吐提升 — 瓶颈在 FP64 高斯消元，非精度选择 |
| **Nsight Compute** | Long Scoreboard 83.2%，Issue Slot 2.32% — FP64 管线延迟是主瓶颈 |
| **PTX 分析** | 194 寄存器/线程，~44% 占用率，共享内存 Bank 冲突为零 |

### 论文意义

1. **范式贡献**：确立"硬件固化多起点"（成功率优先、延迟确定）与"单粒子智能搜索"（高吞吐、适合大批量）两种批量 IK 范式的帕累托边界
2. **方法论贡献**：证明针对 $6\times6$ 固定规模小矩阵，编译期结构编码 + 单核融合的硬件-算法协同设计，优于通用 GPU 优化器流水线
3. **工程价值**：在消费级 GPU 上实现了满足工业抓取精度（<5mm）前提下的 1.75 倍吞吐优势，为采样式运动规划前端提供了无需后处理的求解保障

---

## 关键文档

| 文档 | 路径 | 说明 |
|------|------|------|
| 论文 LaTeX | `standard_robot_cuda_ik/论文/paper.tex` | xelatex 编译 |
| 论文 PDF | `standard_robot_cuda_ik/论文/paper.pdf` | 11 页终稿 |
| 数据溯源 | `standard_robot_cuda_ik/data/experiments/README.md` | 每张表/图 ↔ CSV 文件对应关系 |
| 名词解释 | `standard_robot_cuda_ik/论文/名词解释/名词解释.md` | 47 个 CUDA/机器人术语 |
| 格式规范 | `standard_robot_cuda_ik/论文/格式模板/系统工程与电子技术投稿格式规范.md` | 期刊投稿要求 |
| 历史归档 | `history/README.md` | V1-V4 历史版本清单 |
| AI 开发配置 | `CLAUDE.md` | 项目架构与开发指南 |
