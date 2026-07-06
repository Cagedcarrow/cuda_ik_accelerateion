# Standard Robot CUDA IK — Project Handoff

更新时间：2026-07-06 22:21 CST (Asia/Shanghai)

本文件是当前项目的唯一权威交接入口。切换到其他窗口或其他 AI 后，先读本文件，再读 `论文/paper.tex` 和 `data/experiments/`。

---

## 1. 当前状态

**当前方法：OPT4C** = 结构感知单核融合

```
OPT4C = Analytical Jacobian (via Fused FK)
      + Branchless Levenberg-Marquardt
      + Sobol-K16 multi-start
      + Limit Barrier (analytic gradient)
      + Target-Block thread mapping (<<<N, 32>>>)
      + Shared-memory in-block candidate selection
      + Single-kernel end-to-end fusion
```

**论文状态**：终稿定稿，11 页，xelatex 编译，投稿《系统工程与电子技术》。已按期刊模板规范完成全文重写。

**论文定稿位置**：`../论文定稿/`（含 .tex / .pdf / .docx / .md 四格式 + 全部插图）

---

## 2. 项目结构

```
standard_robot_cuda_ik/
├── CMakeLists.txt                    # CUDA 构建配置 (SM 8.9)
├── PROJECT.md                       # 本文件
├── src/cuda/
│   ├── cuda_v4_runner.cu            # 主核函数 + CLI（全部求解器逻辑）
│   └── cuda_utilities.cuh           # 设备辅助函数（FK, Rodrigues, LDLT...）
├── include/standard_robot_cuda_ik/generated/
│   └── ur10_model_constants.h       # UR10 模型常量（编译期 __constant__）
├── build/                           # 编译输出（gitignored）
│   ├── standard_robot_cuda_v4_runner
│   ├── standard_robot_cuda_v4_runner_r128
│   ├── standard_robot_cuda_v4_runner_r160
│   └── standard_robot_cuda_v4_runner_ptxas
├── 论文/                             # ★ 论文源文件（工作区）
│   ├── paper.tex                    # LaTeX 源（11 页，xelatex 编译）
│   ├── paper.pdf                    # 已编译 PDF
│   ├── paper.md                     # Markdown 版本（含渲染图片）
│   ├── paper.docx                   # Word 版本
│   ├── paper.txt                    # 查重纯文本
│   ├── .latexmkrc                   # 编译配置
│   ├── paper_md_figures/            # Markdown 用 PNG 插图
│   ├── figures/                     # 插图 PDF（fig3-fig7 数据图）
│   ├── 绘图/                        # 5 张图（PDF+SVG+draw.io，双语标题）
│   ├── 格式模板/                    # 系统工程与电子技术投稿格式规范
│   ├── 计划/                        # 审稿意见、语言润色、实验补充意见
│   └── 名词解释/                    # 47 个 CUDA/机器人/数值计算术语
├── data/
│   ├── experiments/                 # ★ 论文全部实验数据
│   │   ├── README.md                # 数据-论文溯源表
│   │   ├── inputs/                  # N=100-1000 目标/种子 raw 文件
│   │   ├── results/                 # 主基准测试 CSV（10 个 N 值）
│   │   └── 补充实验/                # K=1 消融、cuRobo K=16、FP32 混合精度
│   ├── cuda_inputs/                 # 原始目标/种子文件（N=100/500/1000/5000）
│   └── results/latest/              # 首轮基准输出（旧版，仅供参考）
├── scripts/                         # Python 编排脚本
│   ├── run_final_push.py            # 主基准编排
│   ├── run_v4_curobo_compare.py     # cuRobo 对比
│   ├── audit_curobo_quality_round2.py # cuRobo 质量审计
│   └── audit_ur10_model_consistency.py # 模型一致性验证
├── tools/                           # UR10 FK 参考实现
│   ├── robot_model.py               # Python URDF FK
│   └── verify_official_ur10.py      # FK 交叉验证
├── urdf/
│   ├── ur10_official.urdf           # 官方 UR10 模型
│   └── ur10_official_source.json    # 来源记录
├── 论文提交包.tar.gz                 # 发给导师的压缩包（925 KB）
└── 论文定稿 -> ../论文定稿/           # 最终定稿合集

论文定稿/ (位于项目根目录 ../论文定稿/)
├── paper.tex                        # LaTeX 源码（图片路径已适配）
├── paper.pdf                        # 编译 PDF（11页）
├── paper.md                         # Markdown 版（518行）
├── paper.docx                       # Word 版（11MB）
├── figures/                         # 全部插图（7 PDF + 4 PNG）
└── paper_md_figures/                # Markdown 用 PNG（11张）
```

---

## 3. 论文核心数据

### 帕累托前沿（$N=1000$，三方对比）

| 方法 | Strict SR | 吞吐量 (t/s) | $p95$ (mm) | 定位 |
|------|-----------|-------------|-----------|------|
| cuRobo K=16 | **0.988** | 10,399 | **0.9** | 质量天花板 |
| **OPT4C K=16** | 0.954 | **18,226** | 4.6 | 吞吐优先（满足阈值） |
| cuRobo K=1 | 0.840 | 72,624 | 88.7 | 极端吞吐（质量不可用） |

### K=1 消融（证明多起点的决定性作用）

| N | K=16 SR | K=1 SR | K=16 p95 | K=1 p95 |
|---|---------|--------|----------|---------|
| 100 | 0.960 | 0.450 | 4.38 mm | 642 mm |
| 1000 | 0.954 | 0.522 | 4.56 mm | 685 mm |

### FP32 混合精度（证明瓶颈在 FP64 线性求解）

| 配置 | 吞吐量 | vs FP64 |
|------|--------|---------|
| FP64 基线 | 18,930 | 1.00× |
| mixed_safe (FP32 J/H) | 19,275 | **1.02×** |

### Nsight Compute 微架构

- Warp Stall Long Scoreboard: **83.2%** → FP64 管线延迟是主瓶颈
- Issue Slot Utilization: **2.32%** → 指令发射大量空闲
- 共享内存 Bank 冲突: ~112k (**<0.1%**) → 列步长布局验证通过

---

## 4. 环境

| 项目 | 当前值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (SM 8.9) |
| Driver | 610.43.02 |
| CUDA Toolkit | 13.3 / nvcc V13.3.33 |
| CMake | 3.18+ |
| 主 runner | `build/standard_robot_cuda_v4_runner` |
| 论文编译 | `xelatex` (CTeX 系) |
| cuRobo API | 可用（Python） |
| Zotero MCP | 已配置（zoteus，本地 API + 云 API） |

---

## 5. CLI 速查

```bash
# 构建
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 基准测试 (N=1000, K=16, FP64)
./build/standard_robot_cuda_v4_runner \
  --mode v4_static --variant opt4c_block_target \
  --limit-gradient analytic --graph-mode off \
  --precision-mode fp64 --fallback-mode none \
  --N 1000 --K 16 --max-iter 60 --repeat 30 --warmup 10 \
  --targets data/experiments/inputs/targets_N1000_T4x4_f64.raw \
  --seeds data/experiments/inputs/seeds_N1000_K16_q_f64.raw \
  --summary-csv tmp.csv

# FK 正确性验证
./build/standard_robot_cuda_v4_runner \
  --mode fk_check \
  --seeds data/cuda_inputs/q_samples_N20_f64.raw \
  --best-csv tmp_fk.csv

# Jacobian 正确性验证
./build/standard_robot_cuda_v4_runner \
  --mode jacobian_check \
  --seeds data/cuda_inputs/q_samples_N20_f64.raw \
  --best-csv tmp_jac.csv

# 混合精度版本
./build/standard_robot_cuda_v4_runner \
  ... --precision-mode mixed_safe --fallback-mode strict_fail_to_fp64

# PTX 寄存器分析
./build/standard_robot_cuda_v4_runner_ptxas \
  ... 2>&1 | grep -E 'registers|spill|bank'

# Nsight Compute 动态分析
ncu --set full -o ncu_report --target-processes all \
  ./build/standard_robot_cuda_v4_runner \
  --mode v4_static ... --repeat 1 --warmup 0
```

### 精度模式

| 参数 | 含义 |
|------|------|
| `fp64` | 全部 FP64（论文默认） |
| `mixed_safe` | FK/Jacobian/Hessian 用 FP32，线性求解用 FP64 |
| `mixed_mid` | 更多操作用 FP32 |
| `mixed_aggressive` | q 累加也用 FP32 |
| `fp32_risky` | 全部 FP32 |
| `strict_fail_to_fp64` | mixed 失败时自动回退 FP64 |

---

## 6. 数据文件速查

### 核心结果（论文使用）

| 数据 | 位置 |
|------|------|
| OPT4C K=16 汇总 | `data/experiments/results/dense_static_summary.csv` |
| cuRobo K=1 汇总 | `data/experiments/results/dense_curobo_summary.csv` |
| 三方对比 | `data/experiments/补充实验/results/curobo_k16_summary.csv` + `fair_comparison_k16_vs_k16.csv` |
| K=1 消融 | `data/experiments/补充实验/results/k1_static_summary.csv` |
| 混合精度 | `data/experiments/补充实验/results/mixed_precision_summary.csv` |
| 数据 README | `data/experiments/README.md`（表→CSV 对应关系） |

### 输入文件

| 文件 | 说明 |
|------|------|
| `data/experiments/inputs/targets_N*_T4x4_f64.raw` | 目标位姿（N=100-1000 步长 100） |
| `data/experiments/inputs/seeds_N*_K16_q_f64.raw` | Sobol 种子（K=16） |
| `data/experiments/补充实验/inputs/seeds_N*_K1_q_f64.raw` | Sobol 种子（K=1，仅第一行） |
| `data/cuda_inputs/targets_N*_T4x4_f64.raw` | 原始 N=100/500/1000/5000 目标 |
| `data/cuda_inputs/seeds_N*_K16_q_f64.raw` | 原始 N=100/500/1000/5000 种子 |

---

## 7. 论文编译

```bash
cd 论文
latexmk -xelatex paper.tex    # 或
xelatex paper.tex && xelatex paper.tex && xdvipdfmx paper.xdv

# 定稿版（独立目录，不受工作区图片路径影响）
cd ../论文定稿
latexmk -xelatex paper.tex
```

编译产物：`论文/paper.pdf` 或 `论文定稿/paper.pdf`（11 页）

---

## 8. 关键实验结果速查

| 实验（章节） | 核心结论 |
|-------------|---------|
| 静态综合性能（§4.1） | SR 0.940-0.960，吞吐 15k-18k，p95 4.3-5.5mm |
| 公平对比（§4.2） | cuRobo K=16 SR 0.988，OPT4C 吞吐 1.75× |
| K=1 消融（§4.3） | 多起点策略是 SR 的决定性来源 |
| 混合精度（§4.4） | 仅 +2%，瓶颈在 FP64 高斯消元 |
| Nsight Compute（§4.8） | Long Scoreboard 83.2%，Issue Slot 2.32% |
| PTX 分析（§4.8） | 194 寄存器/线程，~44% 占用率，零 Bank 冲突 |

---

## 9. 论文修改记录

| 日期 | 修改内容 |
|------|---------|
| 2026-07-06 | 标题改为"面向小矩阵批量逆运动学的 CUDA 单核函数融合求解" |
| 2026-07-06 | 参考文献替换：删除 8 篇旧文献（1969-2008），新增 6 篇（2024-2025） |
| 2026-07-06 | 按系统工程与电子技术模板规范重写摘要、正文、章节标题 |
| 2026-07-06 | 建立 论文定稿/ 目录，含 .tex/.pdf/.docx/.md 四格式 |
| 2026-07-06 | Zotero MCP 集成（zoteus），可直接读取/写入 Zotero 文献库 |

---

## 10. 后续 AI 接手顺序

1. 读本文件 `PROJECT.md`。
2. 读 `data/experiments/README.md`（数据-论文对应关系）。
3. 读 `论文/paper.tex` 了解全文结构和论证逻辑。
4. 读 `src/cuda/cuda_v4_runner.cu` 了解求解器实现。
5. 若要修改论文，编辑 `论文/paper.tex`，运行 `cd 论文 && latexmk -xelatex paper.tex`。
6. 若要重跑实验，参考 `data/experiments/补充实验/` 下的脚本。
7. 定稿版本在 `../论文定稿/` 中。

---

## 11. 注意事项

- **论文终稿已定稿**。修改前务必确认是否影响核心论证和已定数据。
- **所有实验数据在 `data/experiments/` 中**。首轮数据在 `data/results/latest/` 仅作参考，论文表格基于 `experiments/` 数据生成。
- **CUDA 求解器不再做功能修改**，仅当修复编译器兼容性或添加新精度模式时更新。
- 真机部署前必须使用 `ur_calibration` 提取真实工厂标定参数，并用同一标定模型重新生成 CUDA constants、Python FK 和 cuRobo 配置。
- 历史版本（DLS A0-A8、V2、V3、V4 原型）存在于 `../history/` 目录中，仅作参考。
