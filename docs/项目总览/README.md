# 项目总览 — 标准机械臂批量 IK CUDA 加速

> **定位：** 本目录是项目的**唯一权威参考标准**。今后所有论文写作、实验复现、代码修改均以本文档为准。
>
> **原则：源码是第一事实。** 文档如有与源码不一致之处，以源码为准；本目录文档如有与实验数据不一致之处，以 `standard_robot_cuda_ik/data/` 中的 CSV/JSON 实测数据为准。

---

## 目录结构

| 文档 | 用途 | 适用场景 |
|------|------|---------|
| [README.md](README.md)（本文件） | 总索引与快速查找 | 首次了解项目、快速定位信息 |
| [01-实验数据目录.md](01-实验数据目录.md) | 实验数据完整地图 | 查找某张论文图/表对应的原始数据文件 |
| [02-源代码目录.md](02-源代码目录.md) | 源代码完整地图 | 查找某个功能的实现位置、理解代码架构 |
| [03-论文贡献与核心声明.md](03-论文贡献与核心声明.md) | 论文贡献、创新点、关键性能声明 | 写论文 introduction/conclusion、核对性能数字 |
| [04-GPU架构与CUDA实现细节.md](04-GPU架构与CUDA实现细节.md) | GPU 架构参数、CUDA 特性、kernel 设计 | 写论文 methodology/implementation 章节、技术答辩 |

---

## 快速查找表

| 你想知道... | 去看... |
|------------|--------|
| 论文表 6.1 的数据在哪？ | [01-实验数据目录.md §2.1](01-实验数据目录.md) |
| B5 vs cuRobo 加速比数字 | [03-论文贡献与核心声明.md §3](03-论文贡献与核心声明.md) |
| CUDA kernel 源码在哪？ | [02-源代码目录.md §2](02-源代码目录.md) |
| 消融级别 B0-B6 各是什么意思？ | [03-论文贡献与核心声明.md §4](03-论文贡献与核心声明.md) |
| GPU 用了多少寄存器？ | [04-GPU架构与CUDA实现细节.md §5](04-GPU架构与CUDA实现细节.md) |
| 为什么步长钳位是 0.35 rad？ | [04-GPU架构与CUDA实现细节.md §8](04-GPU架构与CUDA实现细节.md) |
| LDLT 为什么不是 Cholesky？ | [04-GPU架构与CUDA实现细节.md §9](04-GPU架构与CUDA实现细节.md) |
| cuRobo 振荡的根因是什么？ | [03-论文贡献与核心声明.md §7](03-论文贡献与核心声明.md) |
| 收敛阈值 Loose/Medium/Strict 定义？ | [03-论文贡献与核心声明.md §6](03-论文贡献与核心声明.md) + [04 §12](04-GPU架构与CUDA实现细节.md) |
| 论文中绝对不能写什么？ | [03-论文贡献与核心声明.md §9](03-论文贡献与核心声明.md) |

---

## 项目速览

```
项目根目录: /mnt/linuxdata/cuda_ik_accelerateion/

核心子项目: standard_robot_cuda_ik/        ← 主代码、全部实验数据、benchmark 框架
遗留代码:   cuda_low_level_optimization/    ← 第一代概念验证 (custom UR10)
外部依赖:   benchmark/                      ← gitignored, cuRobo/PyRoki 等 clone
文档中心:   docs/
  ├── paper/paper_complete.md              ← 论文最新完整版
  ├── 专利/技术交底书.md                    ← 专利技术交底书
  ├── 修改意见/                            ← 历次修改记录
  └── 项目总览/                            ← 【本目录】权威参考标准
实验诊断:   experiments/curobo_latency_spike_diagnosis/  ← cuRobo 振荡排查
```

### 关键数字

| 指标 | 值 |
|------|-----|
| 目标 GPU | NVIDIA GeForce RTX 4060 Laptop (Ada Lovelace, sm_89) |
| 实验机器人 | UR10 (official URDF), tool0 TCP |
| 主收敛阈值 | Medium: 10 mm 位置 / 5° 姿态 |
| 最大迭代 | K_max = 160 |
| 步长钳位 | 0.35 rad |
| 阻尼约定 | λ 直接加对角线（`H_ii += s_lambda`），非 λ² |
| CUDA B5 吞吐 | 148k–174k targets/s (N=100→10000, ±8%) |
| cuRobo 退化 | 4/12 N 值出现 ~230ms 退化模式（N=4000/7000/9000/10000） |

---

## 使用约定

1. **路径规范：** 本文档中所有相对路径均以项目根目录 `/mnt/linuxdata/cuda_ik_accelerateion/` 为基准。
2. **版本锁定：** 本文档基于 2026-06-12 的代码和实验数据状态编写。如有重大更新，需同步更新本文档。
3. **冲突解决：** 源码 > 实验数据 CSV > 本文档 > 论文草稿 > 修改意见。
4. **引用格式：** 论文中引用实验数据时，建议标注数据来源文件路径（如 `data/results/main_comparison/main_comparison.csv`），以便审稿人/合作者复现。
