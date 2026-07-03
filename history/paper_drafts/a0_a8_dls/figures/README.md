# 论文图表绘制说明

> **用途：** 记录每张图的绘制思路、数据来源、关键设计决策。方便后期使用其他工具（Python matplotlib、draw.io、Visio、TikZ 等）重绘。

---

## 目录

- [文件清单](#文件清单)
- [图表设计通用规范](#图表设计通用规范)
- [逐图说明](#逐图说明)
- [如何用其他工具重绘](#如何用其他工具重绘)

---

## 文件清单

```
figures/
├── README.md                          # 本文件
│
├── plot_fig1_throughput_comparison.m   # 图1 MATLAB脚本
├── plot_fig2_speedup_trend.m           # 图2 MATLAB脚本
├── plot_fig3_ablation.m                # 图3 MATLAB脚本
├── plot_fig4_ncu_parallel.m            # 图4 MATLAB脚本
├── plot_fig5_full_range.m              # 图5 MATLAB脚本
├── plot_fig6_threshold_heatmap.m       # 图6 MATLAB脚本
├── plot_fig7_system_architecture.m     # 图7 MATLAB脚本
├── plot_fig8_thread_mapping.m          # 图8 MATLAB脚本
├── plot_fig9_mixed_precision.m         # 图9 MATLAB脚本
├── plot_fig10_bank_conflict.m          # 图10 MATLAB脚本
├── plot_fig11_ldlt_flow.m              # 图11 MATLAB脚本
│
├── fig1_throughput_comparison.png      # 图1 PNG输出
├── fig2_speedup_trend.png              # 图2 PNG输出
├── fig3_ablation.png                   # 图3 PNG输出
├── fig4_ncu_parallel.png               # 图4 PNG输出
├── fig5_full_range.png                 # 图5 PNG输出
├── fig6_threshold_heatmap.png          # 图6 PNG输出
├── fig7_system_architecture.png        # 图7 PNG输出
├── fig8_thread_mapping.png             # 图8 PNG输出
├── fig9_mixed_precision.png            # 图9 PNG输出
├── fig10_bank_conflict.png             # 图10 PNG输出
└── fig11_ldlt_flow.png                 # 图11 PNG输出
```

---

## 图表设计通用规范

### 配色方案

| 元素 | 颜色 | MATLAB RGB | 用途 |
|:---|:---|:---|:---|
| CUDA B5 主色 | 蓝色 | `[0.0 0.45 0.74]` | 所有表示"本文方法"的元素 |
| cuRobo 对比色 | 橙色 | `[0.85 0.33 0.10]` | 对比对象 |
| 基线/中性色 | 灰色 | `[0.7 0.7 0.7]` | B0基线、背景填充 |
| B3 中间色 | 绿色 | `[0.0 0.6 0.3]` | 消融中间级别 |
| FP32 区域 | 浅蓝 | `[0.8 0.9 1.0]` | 架构图中的FP32区域 |
| FP64 区域 | 浅红 | `[1.0 0.85 0.75]` | 架构图中的FP64关键路径 |
| 警告/退化 | 红色 | `[1.0 0.2 0.0]` | cuRobo退化标注 |

### 排版规范

- **字号：** 标题 14-16pt bold，坐标轴标签 11-12pt，数据标注 7-10pt
- **分辨率：** 300 DPI，PNG 格式
- **导出函数：** `exportgraphics(gcf, 'figN_*.png', 'Resolution', 300)`
- **全部中文：** 标题、坐标轴标签、图例、数据标注

---

## 逐图说明

### 图1: CUDA B5与cuRobo吞吐量对比

| 属性 | 内容 |
|:---|:---|
| **数据源** | `standard_robot_cuda_ik/data/results/main_comparison/main_comparison.csv` |
| **图表类型** | 分组柱状图 (grouped bar chart) |
| **核心信息** | N=100/500/1000/5000 四个批量下 B5 与 cuRobo 的绝对吞吐量对比 |
| **Y轴** | 对数坐标 (targets/s)，因为 3k~170k 跨两个数量级 |
| **标注** | 每组柱顶标注加速比 (36.1×, 10.0×, 4.7×, 1.09×) |
| **设计理由** | 对数坐标使小值和大值都能清晰显示；分组柱状图直观对比两种求解器在同一N下的差异 |

**用其他工具重绘要点：**
- Python matplotlib: `plt.bar(x, height, log=True)` + 自定义 yticklabels
- Visio/draw.io: 手动绘制，注意对数比例

---

### 图2: 加速比与吞吐量趋势

| 属性 | 内容 |
|:---|:---|
| **数据源** | `main_comparison.csv` |
| **图表类型** | 双Y轴折线+散点图 (dual-Y line plot) |
| **核心信息** | 左Y轴=加速比趋势(36.1→1.09)，右Y轴=B5和cuRobo各自吞吐趋势 |
| **设计理由** | 双Y轴将"比值"和"绝对值"放在同一张图中，揭示加速比下降不是B5变慢而是cuRobo批量弹性显现 |

**用其他工具重绘要点：**
- Python: `fig, ax1 = plt.subplots(); ax2 = ax1.twinx()`
- 注意两条Y轴的颜色要与对应数据线一致

---

### 图3: 消融实验 B0→B3→B5

| 属性 | 内容 |
|:---|:---|
| **数据源** | `standard_robot_cuda_ik/data/results/ablation/ablation_medium.csv` |
| **图表类型** | 双Y轴组合图: 分组柱状图(左Y) + 折线图(右Y) |
| **核心信息** | 柱状=B0/B3/B5吞吐量，折线=收敛率；B0收敛率崩塌(B3恢复)，B3→B5吞吐翻倍 |
| **标注** | B0→B3: +577%/+530%/+428%；B3→B5: +120%/+149%/+149% |
| **设计理由** | 将"吞吐量"和"收敛率"两个不同量纲的指标放在同一图中，直观展示"自适应阻尼决定收敛率，混合精度决定吞吐量"的核心结论 |

**用其他工具重绘要点：**
- 柱状和折线必须在不同Y轴(收敛率0-100% vs 吞吐0-170k)
- 颜色: B0灰→B3绿→B5蓝，体现递进关系

---

### 图4: Nsight Compute剖析 B4 vs B5

| 属性 | 内容 |
|:---|:---|
| **数据源** | `standard_robot_cuda_ik/data/profiling/ncu_summary.csv` |
| **图表类型** | 平行坐标图 (parallel coordinates plot) |
| **核心信息** | 8个NCU指标归一化后B4(蓝线) vs B5(橙线)的全面对比 |
| **8个维度** | 计算吞吐率 / DRAM吞吐率 / 寄存器/线程 / 占用率 / Bank冲突 / L1命中率 / Kernel时间 / 局部内存溢出 |
| **关键差异** | Bank冲突线大幅收缩(-63%)，Kernel时间线大幅收缩(-72%)，其余指标基本平行 |
| **设计理由** | 平行坐标图是展示两个配置在多个指标上"全面对比"的最佳选择；一眼看出哪几个指标有显著变化 |

**用其他工具重绘要点：**
- Python: `pandas.plotting.parallel_coordinates()`
- 必须先将各指标归一化到[0,1]区间
- 标注原始数值(非归一化值)在各节点旁边

---

### 图5: 全量程批量扩展性 N=100→10000

| 属性 | 内容 |
|:---|:---|
| **数据源** | `standard_robot_cuda_ik/data/results/full_range/full_range_comparison.csv` |
| **图表类型** | 双线对比图 + 置信带 (dual-line with confidence band) |
| **核心信息** | B5吞吐稳定在148k-174k(浅蓝带)，cuRobo在4个N值上退化(红色倒三角标注) |
| **12个N值** | 100/500/1000/2000/3000/4000/5000/6000/7000/8000/9000/10000 |
| **设计理由** | 浅蓝置信带直观展示B5的±8%窄幅波动；红色倒三角+退化标注突出cuRobo的不可预测性 |

**用其他工具重绘要点：**
- Python: `plt.fill_between()` 绘制置信带
- cuRobo正常点和退化点用不同 marker 和颜色
- X轴12个N值全部标注

---

### 图6: 三级收敛阈值吞吐矩阵

| 属性 | 内容 |
|:---|:---|
| **数据源** | `standard_robot_cuda_ik/data/results/threshold_scan/threshold_scan.csv` |
| **图表类型** | 热力图 + 气泡图并排 (heatmap + bubble chart) |
| **核心信息** | 左图=三阈值×五N的B5吞吐热力图，右图=加速比气泡图 |
| **三级阈值** | 宽松(30mm/10°) / 中等(10mm/5°) / 严格(5mm/1°) |
| **设计理由** | 热力图展示矩阵数据的全局模式；气泡图同时编码加速比(大小)和数值(颜色) |

**用其他工具重绘要点：**
- Python: `sns.heatmap()` + `plt.scatter(s=size, c=color)`
- 气泡大小需调参使视觉效果合理(当前用 `sp_mat*80`)

---

### 图7: 系统总体架构

| 属性 | 内容 |
|:---|:---|
| **数据源** | 无—纯架构图 |
| **图表类型** | 系统流程图 (block diagram) |
| **核心信息** | URDF→参数解析→常量内存导出→Kernel Launch → DLS迭代循环(10个阶段)→收敛输出 |
| **设计要点** | 左右流向；Block内10个阶段分上下两排；收敛判断分支(否→继续)箭头；共享内存/寄存器标注 |
| **颜色编码** | 蓝色=控制流，橙色=常量内存，绿色=Kernel Launch，灰色=普通阶段 |

**用其他工具重绘要点：**
- **推荐用 draw.io / Visio** 而非 MATLAB——这种框图在矢量绘图工具中远更高效
- 关键元素不能遗漏: Grid维度、Block内10阶段完整列表、收敛判断分支(否→继续迭代)、共享内存/寄存器/常量内存三级标注
- 虚线大框表示Block边界

---

### 图8: Block内线程分工与数据流

| 属性 | 内容 |
|:---|:---|
| **数据源** | 无—纯架构图 |
| **图表类型** | 层级映射图 (hierarchical mapping diagram) |
| **核心信息** | Grid→N个Block→单个Block内9个阶段的thread范围+内存层次 |
| **设计要点** | 顶层Grid示意5个Block+省略号；展开Block虚线框；9个阶段分三列(thread0独占/thread0-5/thread0-35)；右侧三级内存层次 |
| **红色警示** | H矩阵(thread 0-35)跨越Warp0+Warp1 |

**用其他工具重绘要点：**
- **推荐用 draw.io** 层级嵌套结构
- 9个阶段框必须准确标注 threadIdx.x 条件
- H矩阵框必须红色高亮

---

### 图9: FP32/FP64混合精度计算路径

| 属性 | 内容 |
|:---|:---|
| **数据源** | 无—纯架构图 |
| **图表类型** | 精度转换流程图 |
| **核心信息** | FP32区域(FK+Jacobian) → 精度转换点 → FP64区域(H+g累积+LDL^T) → FP32(关节更新) |
| **设计要点** | 左蓝(FP32)右红(FP64)区域划分；箭头标注FP32→FP64和FP64→FP32转换点；底部说明Ada Lovelace 1:64吞吐比 |
| **关键标注** | 收敛率0.998+ / Bank冲突-63% / Kernel时间-72% |

**用其他工具重绘要点：**
- **推荐用 draw.io** 画流程图
- 两个大矩形背景区分FP32和FP64区域
- 转换箭头必须标出具体转换位置

---

### 图10: PaddedMat6x8 Bank冲突降低原理

| 属性 | 内容 |
|:---|:---|
| **数据源** | 无—原理示意图 |
| **图表类型** | 左右对比示意图 |
| **核心信息** | 左=stride=6自然布局(Bank索引+冲突)，右=stride=8填充布局(PAD列+两组不重叠Bank集) |
| **设计要点** | 每个格子标注Bank索引号(0-31)；左侧红色冲突标注；右侧绿色(偶数行Bank0-15)+蓝色(奇数行Bank16-31)+灰色PAD列 |
| **数学原理** | gcd(12,32)=4→每8次周期2-3路冲突 vs gcd(16,32)=16→两组Bank完全分离 |

**用其他工具重绘要点：**
- 可用 MATLAB(当前实现) 或 Python matplotlib 的 `pcolor`/`imagesc`
- 关键是 Bank 索引号的计算: `bank = mod(row*stride_doubles + col*2, 32)`
- 填充列用不同颜色和虚线边框区分

---

### 图11: 寄存器级 LDL^T 求解流程

| 属性 | 内容 |
|:---|:---|
| **数据源** | 无—算法流程图 |
| **图表类型** | 四阶段算法流程图 |
| **核心信息** | 分解(35FMA)→前代(15FMA)→对角缩放(6DIV)→回代(15FMA)，合计86标量运算 |
| **设计要点** | 每阶段框内写伪代码；关键特性标注: 0次sqrt, 98 regs, ~0.1us |
| **与Cholesky对比** | LDL^T避免6次sqrt(GPU上sqrt吞吐仅FMA的1/4-1/8) |

**用其他工具重绘要点：**
- **推荐用 TikZ** (LaTeX) 画算法流程图，伪代码的排版效果远好于MATLAB
- 四个阶段框 + 箭头流向 + 伪代码内部细节
- 必须标注"0次 sqrt"和"86次标量运算"

---

## 如何用其他工具重绘

### Python matplotlib 重绘数据图 (图1-6)

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV
df = pd.read_csv('path/to/main_comparison.csv')
# 参照 .m 脚本中的逻辑重绘
```

### draw.io / Visio 重绘架构图 (图7-9)

推荐！MATLAB 的 annotation 功能不适合绘制复杂框图。在 draw.io 中：
1. 导入当前 PNG 作为背景参考
2. 用矩形+文本框+箭头重建各元素
3. 导出为 SVG(矢量) 和 PNG(位图)

### TikZ / LaTeX 重绘算法流程图 (图11)

LDL^T流程最适合用 TikZ 重绘，伪代码排版效果远好于 MATLAB。参考模板：
```latex
\documentclass[tikz]{standalone}
\usetikzlibrary{shapes,arrows,positioning}
\begin{document}
\begin{tikzpicture}[node distance=2cm, auto]
    % 四个阶段节点 + 箭头
\end{tikzpicture}
\end{document}
```

---

## 数据源路径速查

| 数据图 | 数据源绝对路径 |
|:---|:---|
| 图1, 图2 | `/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/data/results/main_comparison/main_comparison.csv` |
| 图3 | `/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/data/results/ablation/ablation_medium.csv` |
| 图4 | `/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/data/profiling/ncu_summary.csv` |
| 图5 | `/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/data/results/full_range/full_range_comparison.csv` |
| 图6 | `/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/data/results/threshold_scan/threshold_scan.csv` |
| 图7-11 | 无数据源 — 纯架构/原理图 |

---

*整理日期：2026-06-14*
