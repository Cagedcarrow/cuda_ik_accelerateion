# CUDA IK 论文最终修改问题与验收标准（Codex Goal 模式）

> 适用对象：最新版论文 `paper(2).pdf` 对应的 LaTeX 工程。  
> 使用方式：将本文档作为 Codex **goal 模式**任务说明，让 Codex 以“完成目标”为导向自动修改论文、图表、脚本与 README。  
> 目标不是继续大规模扩展论文，而是完成**投稿前最后一轮一致性修正与质量压实**。

---

## 0. 总目标

将当前论文修改为一个可以投稿的稳健版本。论文最终定位必须是：

> 面向固定 6-DOF 机械臂、中小批量目标 `N ≤ 1000`、无碰撞检测 IK 前端、`5 mm / 1°` 工程阈值的结构感知单核融合 CUDA 批量逆运动学实现。  
> 本文方法不是全面优于 cuRobo 的通用 IK 求解器，而是在满足阈值的前提下，以更高吞吐和更确定执行路径形成与 cuRobo 的 Pareto 取舍。

---

## 1. Goal 模式执行要求

Codex 需要直接完成修改，而不只是给建议。

### 1.1 必须先做仓库检查

Codex 开始前必须识别：

- [ ] 最新论文源文件路径，例如 `paper.tex`；
- [ ] 当前图目录，例如 `figures/`、`图/`、`paper/figures/`；
- [ ] 当前实验结果目录，例如 `results/`；
- [ ] 当前绘图脚本目录，例如 `scripts/`；
- [ ] 是否已有 CSV 数据；
- [ ] 是否已有 `plot_all.py` 或类似绘图入口；
- [ ] 是否已有 README 复现说明；
- [ ] 当前 LaTeX 是否能编译；
- [ ] 当前 PDF 是否存在未定义图表、引用或文献。

### 1.2 禁止行为

- [ ] 不得伪造实验数据；
- [ ] 不得手工改图中数据而不改 CSV；
- [ ] 不得删除不利结果；
- [ ] 不得把轨迹连续性、Barrier、Nsight 结果包装成明显强于数据支持的结论；
- [ ] 不得只改 PDF，不改 `.tex`；
- [ ] 不得只改正文，不同步更新图表、表格、README；
- [ ] 不得保留 `??`、`[?]`、缺图、缺表；
- [ ] 不得把 `cuRobo-K1` 对比写成公平质量上限对比。

---

# 2. 必须修改的问题与验收标准

---

## P0-1. 修改论文标题

### 当前问题

当前标题：

```text
单核融合批量逆运动学方法
```

过短、过泛，无法体现：

- 机械臂；
- CUDA；
- 批量 IK；
- 结构感知；
- 加速方法。

### 修改要求

改为下面二选一，优先使用第一个：

```text
结构感知单核融合的机械臂批量逆运动学 CUDA 加速方法
```

或：

```text
面向固定六自由度机械臂的结构感知单核融合 CUDA 批量逆运动学方法
```

英文标题对应修改为：

```text
Structure-Aware Single-Kernel Fusion CUDA Acceleration for Batch Inverse Kinematics of Robotic Manipulators
```

### 验收标准

- [ ] 中文标题体现“机械臂、批量逆运动学、CUDA、单核融合”；
- [ ] 英文标题与中文标题语义一致；
- [ ] 页眉、摘要、PDF metadata 如有标题字段也同步更新；
- [ ] 不再使用过泛的“单核融合批量逆运动学方法”。

---

## P0-2. 修改摘要：加入关键量化数据，但不夸大

### 当前问题

当前摘要已经安全，但偏弱：

- 缺少吞吐量范围；
- 缺少 `N=1000` 代表性 cuRobo-K16 对比数值；
- 没有明确说明“本文不是最高精度方法，而是吞吐优先且满足阈值”。

### 修改要求

摘要中加入如下信息：

```text
在 RTX 4060 Laptop GPU 上，N=100–1000 时本文方法吞吐量为约 1.50×10^4–1.82×10^4 targets/s，Strict 成功率为 0.94–0.96。
以 N=1000 为例，cuRobo-Graph K=16 的 Strict SR 为 0.988、p95 为 0.9 mm，本文方法 Strict SR 为 0.954、p95 为 4.6 mm，但吞吐量约为其 1.75 倍。
```

同时明确：

```text
结果表明，本文方法并非追求最高残差精度，而是在满足 5 mm / 1° 阈值的前提下提供更高吞吐和确定执行路径。
```

### 禁止写法

不得写：

```text
全面优于 cuRobo
精度优于 cuRobo
确立全新通用范式
适用于所有工业抓取
```

### 验收标准

- [ ] 摘要出现吞吐量范围；
- [ ] 摘要出现 Strict SR 范围；
- [ ] 摘要出现 `N=1000` cuRobo-K16 代表性对比；
- [ ] 摘要明确 Pareto trade-off；
- [ ] 摘要没有过度声称；
- [ ] 摘要中的所有数值与正文表格一致。

---

## P0-3. 统一第 3 节实验设置中的 cuRobo 协议

### 当前问题

第 3 节仍然写：

```text
cuRobo 对比使用默认 cuRobo-Graph 模式，单种子 num_seeds=1
```

但后文已经加入：

```text
cuRobo-K16
```

这会造成实验协议不一致。

### 修改要求

将第 3 节 cuRobo 实验协议改成：

```text
cuRobo 对比包含两种配置：
1）cuRobo-Graph-K1：默认单种子配置，用于系统默认配置对比；
2）cuRobo-Graph-K16：与本文 K=16 保持同等种子数，用于公平质量-吞吐对比。
两者均关闭碰撞检测，开启 CUDA Graph，并使用同一外部 URDF FK 评估管线复评。CUDA Graph capture、warmup、repeat、计时方式和内存分配是否计入时间需明确说明。
```

必须写清楚：

- [ ] `num_seeds=1`；
- [ ] `num_seeds=16`；
- [ ] collision off；
- [ ] CUDA Graph on；
- [ ] warmup 次数；
- [ ] repeat 次数；
- [ ] graph capture 是否计时；
- [ ] memory allocation 是否计时；
- [ ] 是否使用同一外部 FK 复评。

### 验收标准

- [ ] 第 3 节同时说明 cuRobo-K1 和 cuRobo-K16；
- [ ] 后文所有 cuRobo 表格、图和正文表述与第 3 节协议一致；
- [ ] 不再把 cuRobo-K1 写成公平质量上限；
- [ ] 不再出现“cuRobo 对比仅使用 num_seeds=1”的孤立表述。

---

## P0-4. 扩展 cuRobo-K16 对比表

### 当前问题

表 6 只给 `N=1000` 代表性数据。  
如果正文说“同等 K=16 公平对比”，审稿人会要求更多 N 的结果。

### 修改要求

至少补充：

```text
N = 100, 500, 1000
```

建议补充：

```text
N = 100, 200, ..., 1000
```

表格字段至少包括：

```text
N
method
K
Strict SR
throughput_targets_per_s
p95_all_mm
p95_success_mm
effective_throughput = throughput × Strict SR
```

方法至少包括：

```text
OPT4C-K16
cuRobo-Graph-K16
cuRobo-Graph-K1
```

建议包括：

```text
OPT4C-K1
```

### 数据文件要求

新增或更新：

```text
results/fair_curobo_k16_summary.csv
```

字段至少包含：

```text
N, method, K, strict_sr, throughput_targets_per_s, p95_all_mm, p95_success_mm, effective_throughput
```

### 验收标准

- [ ] 论文至少有 `N=100,500,1000` 的 cuRobo-K16 对比；
- [ ] CSV 能追溯表格；
- [ ] 表格中的数值与摘要、正文、图 7 一致；
- [ ] 正文若只保留 N=1000 主表，必须说明“代表性数据”，并在附表或补充表中给其他 N；
- [ ] 不允许只保留单点数据却写泛化结论。

---

## P0-5. 轨迹连续性实验必须重新检查并降级解释

### 当前问题

当前轨迹实验：

```text
原始 best p95(Δq) = 14.6–14.9 rad
smoothness rerank 后 p95(Δq) = 9.5–10.7 rad
```

这个数值非常大，说明：

1. 轨迹连续性仍然很差；
2. 可能没有考虑关节角周期等价；
3. 不能把该实验写成明显优势。

### 必须检查

Codex 必须检查轨迹 Δq 的计算方式：

- [ ] 是否对 revolute joint 做 `wrapToPi(q_t - q_{t-1})`；
- [ ] UR10 关节限位是否允许多圈；
- [ ] 是否存在 `+π` 到 `-π` 的等价跳变被误算；
- [ ] `delta_q` 是否使用欧氏范数；
- [ ] `joint_jump_count` 阈值是多少；
- [ ] smoothness rerank 是否使用上一时刻最佳解；
- [ ] 是否只是离线候选重排序，而非真正 warm-start。

### 修改要求

如果当前代码未做 wrap，则添加：

```python
def angular_diff(a, b):
    return atan2(sin(a - b), cos(a - b))
```

并重新生成：

```text
results/trajectory_continuity_summary.csv
figures/fig_trajectory_delta_q.pdf
```

如果 UR10 关节不应 wrap，则 README 和正文必须说明原因。

### 正文必须改成局限性表述

推荐写法：

```text
轨迹实验表明，当前单点 pose cost 候选选择会导致 IK 分支切换。候选级 smoothness rerank 能降低 p95(Δq)，但残余跳变仍然较大，说明该方法当前更适合独立目标批量 IK 或规划前端候选生成；若要直接输出可执行连续轨迹，需要引入上一时刻解 warm-start、连续性约束或轨迹级优化。
```

### 验收标准

- [ ] 已检查并说明 Δq 是否使用 wrap；
- [ ] 若修改了 Δq 计算，CSV 和图已重新生成；
- [ ] 正文不再把轨迹实验写成强优势；
- [ ] 正文明确承认当前轨迹连续性不足；
- [ ] `joint_jump_count` 阈值在正文或表注中说明；
- [ ] 轨迹实验至少报告 `mean_delta_q`、`p95_delta_q`、`max_delta_q`、`jump_count`、`point_success_rate`。

---

## P0-6. 将 Barrier 从核心贡献降级为辅助正则项

### 当前问题

当前近限位实验显示：

```text
Barrier ON:  Strict SR = 0.940, near-limit ratio = 0.015
Barrier OFF: Strict SR = 0.943, near-limit ratio = 0.017
```

并且 `wlimit=0.03` 使 p95 从 9.45 mm 增至 11.90 mm。

这说明 Barrier 收益很弱，甚至可能带来误差副作用。

### 修改要求

论文中所有相关表述必须调整：

从：

```text
Barrier 是关键贡献 / 显著提升鲁棒性 / 改善收敛
```

改为：

```text
Barrier 是安全裕度正则项，主要用于轻微降低近限位解比例；其对成功率贡献有限，在部分 near-limit 目标上可能增加失败样本尾部误差。
```

### 贡献列表修改

贡献中不得把 Barrier 单列为主要贡献。  
可以放在方法细节或候选排序细节中。

### 验收标准

- [ ] 摘要不强调 Barrier；
- [ ] 贡献列表不把 Barrier 写成核心创新；
- [ ] 近限位章节明确写出 Barrier 效果有限；
- [ ] 结论不把 Barrier 作为成功率来源；
- [ ] 表述与图 10、图 11 的数据一致。

---

## P0-7. 降低 Nsight Compute 解释强度

### 当前问题

当前正文将：

```text
Warp Stall Long Scoreboard = 83.2%
```

直接解释为：

```text
FP64 管线发射延迟是性能瓶颈
```

这个解释过强。Long Scoreboard 表示长延迟依赖，可能来自 FP64、小矩阵依赖链、超越函数、寄存器依赖、spill/local memory 等组合，不宜单一归因。

### 修改要求

替换为：

```text
Long Scoreboard 占比较高表明核函数受长延迟依赖限制。结合 Memory Throughput 较低和共享内存冲突率较低，可排除全局/共享内存带宽作为主瓶颈；瓶颈更可能来自 FP64 小矩阵求解、超越函数、寄存器依赖和局部标量控制流的组合。
```

表 11 中“含义”列修改：

| 原含义 | 修改后 |
|---|---|
| FP64 管线延迟主导 | 长延迟依赖主导 |
| 指令发射大量空闲 | 受依赖链限制，发射槽利用不足 |
| 非访存瓶颈 | 带宽不是主瓶颈 |

### 验收标准

- [ ] 不再把 Long Scoreboard 单独等同于 FP64 管线延迟；
- [ ] 保留“非内存带宽瓶颈”判断；
- [ ] 写成“FP64 + 超越函数 + 依赖链 + 标量控制流”的组合瓶颈；
- [ ] 混合精度实验解释也同步调整；
- [ ] 表 11 含义列与正文一致。

---

## P0-8. 删除结论中的 `O(10^-6) -> O(10^-16)` 精度表述

### 当前问题

方法部分已经正确写成：

```text
解析雅可比优势在于减少 FK 调用、避免差分步长调参、改善梯度方向稳定性。
```

但结论中如果仍写：

```text
精度从 O(10^-6) 提升至 O(10^-16)
```

会引起质疑，因为最终 IK 误差仍是毫米级，不应把雅可比局部数值精度直接等同于最终解精度。

### 修改要求

删除所有类似：

```text
O(10^-6) -> O(10^-16)
机器精度提升
```

替换为：

```text
解析雅可比避免了有限差分步长选择带来的数值敏感性，并将每迭代 FK 调用从 13 次降至 1 次。
```

### 验收标准

- [ ] 全文搜索不到 `10^{-16}` 或 `O(10` 等不当精度提升表述，除非是在非常明确的局部雅可比数值误差语境中；
- [ ] 结论中不再把解析雅可比写成最终 IK 精度提升到机器精度；
- [ ] 解析雅可比贡献定位为计算量和梯度稳定性。

---

## P0-9. 增加或补充误差阈值扫描

### 当前问题

论文定义了 Strict，并提到 Loose/Medium 单调性，但缺少独立阈值扫描。  
当前结论容易被理解为只对 `5 mm / 1°` 单一阈值有效。

### 修改要求

新增小表或图，最低限度为表格：

| 阈值等级 | 位置阈值 | 姿态阈值 | OPT4C-K16 SR | OPT4C-K1 SR | cuRobo-K16 SR | cuRobo-K1 SR |
|---|---:|---:|---:|---:|---:|---:|
| Loose | 30 mm | 10° |  |  |  |  |
| Medium | 10 mm | 5° |  |  |  |  |
| Strict | 5 mm | 1° |  |  |  |  |
| Ultra，可选 | 2 mm | 0.5° |  |  |  |  |

至少使用：

```text
N = 1000
```

数据文件：

```text
results/threshold_scan.csv
```

### 正文解释

必须写清楚：

```text
OPT4C 的优势主要在满足工程阈值下的吞吐；当阈值进一步收紧时，cuRobo-K16 的成功样本精度优势会更明显。
```

### 验收标准

- [ ] 存在阈值扫描表或图；
- [ ] 至少包含 Loose、Medium、Strict；
- [ ] 数据来自 CSV；
- [ ] 正文不把 `5 mm / 1°` 泛化为所有工业任务要求；
- [ ] 如果 Ultra 结果较差，必须如实说明。

---

## P0-10. 重画关键图

### 当前问题

当前图表可读性不足，尤其是：

- 图 1 线程映射图过于简陋；
- 图 3 时间组成图 H2D/D2H/launch 几乎不可见；
- 图 7 Pareto 图是核心图，但排版还不够正式；
- 图 8、图 11、图 13 字号和风格不统一。

### 必须重画

#### 图 1：Target-Block 线程映射架构图

必须包含：

```text
Grid: N blocks
Block i -> Target i
lane 0-15 -> Sobol seed 0-15
lane 16-31 -> inactive/reserved
Shared memory candidate buffer
lane 0 final selection
Global memory best output
```

验收：

- [ ] 不再是简陋文字框；
- [ ] 箭头清晰；
- [ ] 字体与论文一致；
- [ ] 能在单栏宽度下看清。

#### 图 3：时间组成图

建议改为：

1. 表格列出百分比；
2. 或使用 broken axis / inset；
3. 或只画非 kernel 部分占比。

验收：

- [ ] H2D、D2H、launch 可以看清；
- [ ] 不再只有一整块黑色 kernel；
- [ ] 图注说明 kernel 占比极高，非 kernel 项很小。

#### 图 7：Pareto 图

必须包含：

```text
OPT4C-K1
OPT4C-K16
cuRobo-K1
cuRobo-K16
```

验收：

- [ ] 坐标轴单位完整；
- [ ] 点标注清楚；
- [ ] 图例不遮挡；
- [ ] 图中数据与表 6 或 CSV 一致；
- [ ] 作为核心结果图，视觉质量需高于其他辅助图。

### 图表通用验收

- [ ] 所有图由脚本从 CSV 生成；
- [ ] 字号在 PDF 中可读；
- [ ] 黑白打印可区分；
- [ ] 线型、marker、字体统一；
- [ ] 坐标轴单位完整；
- [ ] 图题与正文引用一致；
- [ ] 图中不能出现英文乱码或中文缺字。

---

## P0-11. CPU baseline 名称降级

### 当前问题

当前 CPU baseline 是：

```text
单进程 Python/NumPy CPU-LM
```

不是 KDL/TRAC-IK 实测，也不是优化后的 C++ CPU baseline。

### 修改要求

所有相关标题必须写成：

```text
Python/NumPy CPU-LM 量级对照
```

不要写成：

```text
CPU baseline 证明优于 KDL/TRAC-IK
```

正文必须明确：

```text
该基线仅用于量级对照，不代表经过 C++/SIMD 优化的 KDL、TRAC-IK 或工业 CPU IK 实现。
```

### 验收标准

- [ ] 表 7 标题体现 Python/NumPy CPU-LM；
- [ ] 正文明确“不代表 KDL/TRAC-IK”；
- [ ] 引言中提到 KDL/TRAC-IK 时，不暗示已经实测它们；
- [ ] 结论不使用 CPU baseline 过度推广。

---

## P0-12. 近奇异与近限位目标生成规则必须补清楚

### 当前问题

论文已有近奇异和近限位实验，但生成规则不够细。审稿人可能无法复现。

### 修改要求

在实验设置或 4.5 节补充：

#### 近奇异目标

至少说明：

```text
wrist singular 如何构造；
elbow singular 如何构造；
shoulder singular 如何构造；
每类 N；
随机种子；
是否由 FK 生成；
是否保证可达；
```

#### 近限位目标

至少说明：

```text
至少一个关节距离 qmin/qmax 小于 0.087 rad；
目标是否由 FK 生成；
选择哪个关节是否随机；
随机种子；
N；
```

### 验收标准

- [ ] 近奇异三类目标定义清楚；
- [ ] 近限位目标定义清楚；
- [ ] 生成脚本或 README 中有对应命令；
- [ ] 论文中的实验描述足以复现目标集；
- [ ] 不只写“构造近奇异目标”这种泛泛表述。

---

## P0-13. 修正主结论与贡献列表

### 当前问题

当前论文已经比上一版收敛，但结论仍可能有过强表达。

### 修改要求

最终贡献建议写成：

1. **结构感知单核融合 CUDA 实现**  
   将 FK、解析雅可比、LM 迭代和候选选择融合进单个 target-block kernel。

2. **Sobol 多起点成功率机制验证**  
   K 扫描说明成功率主要来自多起点覆盖，而非单种子 LM 局部优化能力。

3. **与 cuRobo 的 Pareto 边界分析**  
   cuRobo-K16 质量更高，OPT4C-K16 在满足阈值时吞吐更高。

4. **鲁棒性与边界实验**  
   近奇异、近限位、轨迹连续性实验说明适用边界。

5. **微架构分析**  
   Nsight Systems/Compute 表明核心执行路径为单 kernel，性能受长延迟依赖和小矩阵标量计算限制。

### 结论必须承认

- [ ] 不处理碰撞；
- [ ] 固定 UR10/6DOF；
- [ ] 目标主要来自 FK 可达集；
- [ ] 轨迹连续性不足；
- [ ] cuRobo-K16 精度和成功率更高；
- [ ] 对精密装配、插接、接触丰富任务不够；
- [ ] Barrier 只是辅助正则；
- [ ] 混合精度收益有限。

### 验收标准

- [ ] 贡献列表不夸大；
- [ ] 结论不说全面优于 cuRobo；
- [ ] 结论明确应用边界；
- [ ] 结论与实验中不利结果一致；
- [ ] 结论能经得起“为什么不用 cuRobo-K16？”的质疑。

---

# 3. 数据与复现验收标准

## 3.1 CSV 追溯

所有新增或修改图表必须有 CSV 数据来源。

至少需要：

```text
results/fair_curobo_k16_summary.csv
results/trajectory_continuity_summary.csv
results/threshold_scan.csv
results/kernel_time_breakdown.csv
results/seed_count_scan.csv
results/barrier_weight_scan.csv
```

验收：

- [ ] 表 6 来自 `fair_curobo_k16_summary.csv`；
- [ ] 图 7 来自 `fair_curobo_k16_summary.csv`；
- [ ] 图 12 来自 `trajectory_continuity_summary.csv`；
- [ ] 阈值扫描表来自 `threshold_scan.csv`；
- [ ] 图 3 来自 `kernel_time_breakdown.csv`；
- [ ] 图 8 来自 `seed_count_scan.csv`；
- [ ] 图 11 来自 `barrier_weight_scan.csv`。

## 3.2 绘图脚本

必须存在统一绘图入口：

```bash
python scripts/plot_all.py
```

或 README 中明确等效命令。

验收：

- [ ] 一条命令可重新生成所有论文新增图；
- [ ] 图路径与 `paper.tex` 中引用路径一致；
- [ ] 不依赖手动截图，Nsight timeline 除外；
- [ ] 图片格式优先 `.pdf`，Nsight timeline 可用 `.png`。

## 3.3 LaTeX 编译

必须通过：

```bash
cd paper
latexmk -xelatex paper.tex
```

或 README 指定命令。

验收：

- [ ] 无缺图；
- [ ] 无未定义引用 `??`；
- [ ] 无未定义文献 `[?]`；
- [ ] 无明显 overfull 导致表格出界；
- [ ] PDF 中图表编号连续；
- [ ] 标题、摘要、图表、结论均为最新版本。

---

# 4. 正文一致性检查

Codex 完成后必须执行全文搜索，确保以下危险表达不存在或已降级。

## 4.1 必须删除或改写的表达

全文不得出现以下意思：

```text
全面优于 cuRobo
cuRobo 质量不可用（除非明确指 cuRobo-K1 全样本 p95 受失败尾部影响）
消除 warp divergence
Bank 冲突为零（除非有明确计数器和定义）
FP64 管线延迟被 Long Scoreboard 直接证实
精度从 O(10^-6) 提升至 O(10^-16)
Barrier 显著提升成功率
轨迹连续性问题已解决
5 mm 是所有工业抓取要求
```

## 4.2 推荐替代表达

| 危险表达 | 推荐表达 |
|---|---|
| 全面优于 cuRobo | 与 cuRobo 形成 Pareto 取舍 |
| 消除 warp divergence | 降低控制流复杂度 |
| Bank 冲突为零 | 共享内存访问不是主瓶颈 |
| FP64 管线延迟主导 | 长延迟依赖主导，可能来自 FP64、超越函数、寄存器依赖和标量控制流 |
| Barrier 提升成功率 | Barrier 轻微降低 near-limit 解比例，但收益有限 |
| 轨迹连续性已解决 | smoothness rerank 只能缓解跳变，仍需 warm-start 或轨迹级优化 |
| 工业抓取 5 mm | 粗定位抓取、规划前端或中等精度任务阈值 |

---

# 5. 最低可接受验收标准

如果时间紧，至少完成以下 10 项，否则不建议投稿：

1. [ ] 标题修改；
2. [ ] 摘要补量化数据并保持克制；
3. [ ] 第 3 节 cuRobo-K1/K16 协议统一；
4. [ ] 表 6 扩展到至少 `N=100,500,1000`；
5. [ ] 轨迹 Δq 检查 wrap，并将轨迹连续性写成局限性；
6. [ ] Barrier 降级为辅助正则；
7. [ ] Nsight Long Scoreboard 解释降级；
8. [ ] 删除结论中 `O(10^-6) -> O(10^-16)`；
9. [ ] 重画图 1、图 3、图 7；
10. [ ] LaTeX 编译通过，无 `??`、无 `[?]`、无缺图。

---

# 6. 最终交付物

Codex 完成后必须交付：

```text
paper/paper.tex
paper/paper.pdf
figures/*.pdf 或 *.png
results/*.csv
scripts/plot_all.py
README.md
CHANGELOG.md 或修改摘要
```

其中 `CHANGELOG.md` 至少写：

```text
1. 修改了哪些正文段落；
2. 新增或更新了哪些图；
3. 新增或更新了哪些表；
4. 哪些实验数据来自已有结果，哪些重新生成；
5. 哪些计划项未完成及原因。
```

---

# 7. 最终验收模板

Codex 修改完成后，用下面模板检查：

```text
验收结论：通过 / 有条件通过 / 不通过

一、P0 修改
- 标题：
- 摘要：
- cuRobo 协议：
- cuRobo-K16 表：
- 轨迹连续性：
- Barrier：
- Nsight：
- 解析雅可比表述：
- 阈值扫描：
- 图 1/3/7：

二、数据追溯
- CSV：
- 图：
- 表：
- 摘要数值：

三、编译与复现
- latexmk：
- plot_all：
- README：

四、仍需人工判断
1.
2.
3.

五、是否可以投稿
- 可以 / 不建议 / 需再补实验
```

---

## 8. 推荐最终主结论

最终论文应收束到如下表述：

```text
本文提出了一种面向固定 6-DOF 中小批量 IK 的结构感知单核融合 CUDA 实现。其优势不在于比 cuRobo 获得更高极限精度，而在于将 FK、解析雅可比、LM 迭代和候选选择融合到单个 target-block kernel 中，在满足 5 mm / 1° 阈值的前提下提供更高吞吐和更确定的执行路径。实验表明，Sobol 多起点是成功率的主要来源；与 cuRobo-K16 相比，本文方法牺牲部分成功率和成功样本精度，换取更高吞吐。近奇异、近限位和轨迹实验进一步表明，该方法适合作为规划前端或独立目标批量 IK 候选生成器，而非直接替代带碰撞检测和轨迹连续性约束的通用运动生成框架。
```
