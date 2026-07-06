# standard_robot_cuda_ik

CUDA-accelerated batch inverse kinematics (IK) solver for UR10 6-DOF manipulator.

**当前唯一权威交接入口是 `PROJECT.md`。**

## 论文状态

✅ **终稿定稿** — 投稿《系统工程与电子技术》

| 格式 | 路径 |
|:----|:-----|
| LaTeX 源码 | [`论文/paper.tex`](论文/paper.tex) |
| 编译 PDF | [`论文/paper.pdf`](论文/paper.pdf) |
| Markdown 版 | [`论文/paper.md`](论文/paper.md) |
| Word 版 | [`论文/paper.docx`](论文/paper.docx) |
| **定稿合集** | [`../论文定稿/`](../论文定稿/)（独立目录，可直接发给审稿人） |

## 关键数据

最新实验数据位于 `data/experiments/`，详见 README。

## 构建

```bash
cd standard_robot_cuda_ik
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Target: SM 8.9 (Ada Lovelace), CUDA 13.3.
