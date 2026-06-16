# CUDA 机械臂批量 IK 论文第二轮修改与补充实验计划

> 适用对象：当前稿件《基于 CUDA 小矩阵加速的机械臂批量逆运动学求解》新版 PDF。  
> 执行目标：先复核关键异常数据，必要时补充小规模实验；然后依据真实结果重构摘要、结论、表格和叙事口径。  
> 重要原则：不编造数据；没有日志或实验结果支撑的结论必须删除或改为“未来工作”。

---

## 0. 总目标

当前稿件已经完成了三层计时口径、cuRobo CUDA Graph on/off、有效吞吐率、误差分布和 Nsight 分析等关键改进，但仍存在两个风险点：

1. **部分结论与表格数据不一致**：例如摘要声称“低批量 N≤1000 下延迟降低 52%”，但表 5 只支持 N=100 这一点。
2. **少量实验数据存在异常或内部矛盾**：尤其是 cuRobo-Graph 在 N=5000 时出现极端高吞吐，以及 A7/A8 CUDA Graph 对比中正文说差异 <1%，表格却显示 N=500/1000 有 6%–11% 差异。

因此，本轮修改不是单纯润色，而是执行如下闭环：

```text
数据复核/补测 → 结果判定 → 表格修正 → 论文叙事重构 → 图表清单更新 → 参考文献核查
```

---

## 1. 执行前准备

### 1.1 建立工作分支或备份

执行前必须创建备份，避免覆盖当前稿件。

建议操作：

```bash
git status
git checkout -b paper_revision_curobo_fairness_v2
# 或者至少复制当前 paper 目录
cp -r docs/paper docs/paper_backup_before_revision_v2
```

如果当前目录不是 git 仓库，也要手动备份以下内容：

- 当前论文源文件：`.md` / `.tex` / `.docx` / `.pdf`
- 实验脚本
- 原始数据 CSV / JSON / log
- 图表脚本
- benchmark 输出文件

### 1.2 先定位项目结构

请自动搜索并记录以下文件位置：

```bash
find . -iname '*paper*' -o -iname '*.md' -o -iname '*.tex'
find . -iname '*curobo*' -o -iname '*benchmark*' -o -iname '*ik*'
find . -iname '*.csv' -o -iname '*.json' -o -iname '*.log'
find . -iname '*plot*' -o -iname '*figure*'
```

输出一个简短的 `revision_file_inventory.md`，记录：

| 类别 | 路径 | 用途 |
|---|---|---|
| 论文源文件 | TODO | 需要修改 |
| CUDA benchmark 脚本 | TODO | 需要复核/补跑 |
| cuRobo benchmark 脚本 | TODO | 需要复核/补跑 |
| 原始数据 | TODO | 判断是否可复用 |
| 图表脚本 | TODO | 暂不重绘，只记录 |

---

## 2. 必须复核/补测的实验

本轮不要求重新做全量大实验，但必须复核两个关键风险点。若复核失败，则必须补跑对应数据。

---

### 2.1 实验 A：复核 cuRobo-Graph 的 N=5000 异常高吞吐

#### 2.1.1 当前风险

当前表 5 中 cuRobo-Graph 数据如下：

| N | GPU 流时间/ms | raw throughput/(targets/s) |
|---:|---:|---:|
| 4000 | 27.50 | 145,437 |
| 5000 | 4.73 | 1,056,171 |
| 10000 | 61.29 | 163,160 |

N=5000 的 4.73 ms 和 1,056k targets/s 极端异常。它可能是真实的 cuRobo 内部批处理/Graph 行为，也可能是实验代码问题。

#### 2.1.2 必须检查的项目

请检查 cuRobo-Graph benchmark 脚本，确认：

1. 是否确实创建了 `N=5000` 个目标位姿；
2. 每个目标是否不同，而不是重复同一个目标；
3. Graph replay 前输入 tensor 是否正确更新；
4. Graph capture 是否固定了 batch shape，导致 replay 时没有替换输入；
5. 输出结果数量是否确实为 5000；
6. success rate 是否按 5000 个目标逐一统计；
7. `torch.cuda.synchronize()` 是否放在计时区间之后；
8. warmup/capture 时间是否错误计入或错误排除；
9. 是否误用了缓存结果、旧输出或未更新的 result 对象；
10. 是否对每个 N 都重新初始化 solver，还是复用 solver；如果复用，必须说明。

#### 2.1.3 增加校验日志

在 benchmark 中增加以下日志输出，至少保存为 `logs/curobo_graph_validation_N5000_YYYYMMDD_HHMMSS.log`：

```text
N = 5000
input target tensor shape = TODO
output solution tensor shape = TODO
target position checksum before solve = TODO
output q checksum after solve = TODO
number of unique target positions = TODO
success count = TODO / 5000
GPU event elapsed ms = TODO
E2E elapsed ms = TODO
first 5 target poses = TODO
first 5 output q = TODO
last 5 output q = TODO
```

checksum 可使用：

```python
float_tensor_sum = tensor.detach().float().sum().item()
float_tensor_mean = tensor.detach().float().mean().item()
float_tensor_std = tensor.detach().float().std().item()
```

如果目标或输出是 SE(3) 矩阵，也可以分别统计 position 和 rotation matrix 的 sum/mean/std。

#### 2.1.4 必须补跑的 N 点

最小补跑集合：

```text
N = 4000, 5000, 10000
重复次数：30
预热次数：至少 5；如果 CUDA Graph capture 需要额外 warmup，则 capture 不计入正式结果
计时口径：GPU Stream + End-to-End
```

如果 N=5000 仍然异常高，继续补跑邻近点：

```text
N = 4500, 5000, 5500
```

目的不是增加主实验复杂度，而是判断 N=5000 是否为真实的 batch-size 特殊点。

#### 2.1.5 判定标准

- 如果复核后 N=5000 仍然稳定在约 4–6 ms，并且输出数量、success rate、输入更新均正确，则保留该结果，并在正文明确说明：
  - 这是 cuRobo-Graph 的 batch-size 特殊高性能点；
  - 该点不可外推；
  - 本文不以该点作为趋势结论依据。

- 如果复核后 N=5000 消失或变为约 30–60 ms，则修正表 5、图 2–4、摘要和结论。

---

### 2.2 实验 B：复核 A7 Direct 与 A8 CUDA Graph 对本文单 kernel 方案的影响

#### 2.2.1 当前风险

当前表 7 中：

| N | A7 Direct/(t/s) | A8 Graph/(t/s) | 实际差异 |
|---:|---:|---:|---:|
| 100 | 80,757 | 81,240 | +0.6% |
| 500 | 128,404 | 142,805 | 约 +11.2% |
| 1000 | 138,719 | 147,381 | 约 +6.2% |
| 5000 | 157,481 | 157,841 | +0.2% |

正文却写 “A7 与 A8 吞吐差异 <1%”。这与表格不一致。

#### 2.2.2 必须补跑或复核

请对以下 N 重新执行 A7/A8 对比：

```text
N = 100, 500, 1000, 5000
重复次数 = 30
预热次数 >= 5
记录 GPU 温度/功耗状态，如可行
```

必须输出：

| N | A7 mean | A7 std | A7 p50 | A7 p95 | A8 mean | A8 std | A8 p50 | A8 p95 | relative diff/% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

#### 2.2.3 判定标准

- 如果 A8 相比 A7 差异稳定 <1%：保留“单 kernel 方案基本不依赖 CUDA Graph”的结论。
- 如果 A8 相比 A7 在 N=500/1000 仍有 5%–11% 提升：正文必须改为：

```text
CUDA Graph 对本文单 kernel 方案仍有小到中等幅度影响，尤其在中小批量下可进一步降低 launch/同步固定开销；但该影响显著小于 cuRobo-NoGraph 到 cuRobo-Graph 的数量级提升。本文单 kernel 结构仍降低了对 CUDA Graph 的依赖。
```

- 如果波动主要来自热状态或重复实验标准差过高：必须报告 mean±std，不得只报告单次结果。

---

### 2.3 实验 C：是否补跑 ε = 10^-5 的 Strict 阈值恢复实验

此实验不是强制项，但取决于正文怎么写。

#### 2.3.1 如果正文保留以下表述，就必须补跑

如果论文正文中写到：

```text
Strict 精度任务可切换到 ε = 10^-5 或全 FP64 路径。
```

这句话目前只是推断。若要作为技术建议，至少应补一个小实验。

#### 2.3.2 推荐最小实验

```text
方法：CUDA-Mixed ε=10^-6、CUDA-Mixed ε=10^-5、FP64 自适应阻尼
N = 500, 1000
阈值：Strict = 5 mm / 1°
重复次数：30
指标：success rate、valid throughput、pos p95、rot p95、GPU stream ms
```

表格模板：

| N | Method | ε | GPU ms | raw t/s | Strict SR | valid t/s | pos p95/mm | rot p95/deg |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 500 | CUDA-Mixed | 1e-6 | TODO | TODO | TODO | TODO | TODO | TODO |
| 500 | CUDA-Mixed | 1e-5 | TODO | TODO | TODO | TODO | TODO | TODO |
| 500 | FP64-Damping | 1e-6 or FP64 | TODO | TODO | TODO | TODO | TODO | TODO |
| 1000 | CUDA-Mixed | 1e-6 | TODO | TODO | TODO | TODO | TODO | TODO |
| 1000 | CUDA-Mixed | 1e-5 | TODO | TODO | TODO | TODO | TODO | TODO |
| 1000 | FP64-Damping | 1e-6 or FP64 | TODO | TODO | TODO | TODO | TODO | TODO |

#### 2.3.3 如果不补跑，则必须改写

如果不补跑这个实验，正文只能写：

```text
ε = 10^-5 或全 FP64 路径可能改善 Strict 阈值下的收敛表现，相关验证将作为后续工作。
```

不能写成已经验证的结论。

---

## 3. 必须修改的论文叙事

当前实验结果已经显示：cuRobo-Graph 开启后非常强，在 N=500、N=1000、N=5000 和 N=10000 上通常不弱于本文方法，甚至明显更快。因此论文不能再采用“本文全面优于 cuRobo”的叙事。

### 3.1 摘要必须修改

#### 当前问题

摘要中类似表述：

```text
低批量（N≤1000）下 GPU 流延迟较 cuRobo（CUDA Graph 开启）降低 52%。
```

这与表 5 不一致。表 5 仅支持 N=100 延迟降低 52%。

#### 推荐改法

将摘要改为类似：

```text
实验表明：当 N≥500 时，本文方法吞吐稳定在约 128k–157k targets/s，GPU 时间随 N 近似线性增长（R²>0.999）；在 N=100 的极小批量场景下，本文方法较 cuRobo（CUDA Graph 开启）GPU 流延迟降低约 52%。当 N≥500 时，cuRobo-Graph 依靠 CUDA Graph 和内部批处理优化通常获得更高吞吐。本文方法的主要优势在于单 kernel 结构简单、kernel launch 数恒为 1、批量扩展行为更可预测。Jacobian 精度分析表明，ε=10^-6 下 FP32 混合精度对应约 4–5 mm 的 IK 收敛精度边界，适用于中等精度批量采样和候选解筛选场景。
```

注意：如果复核后 N=5000 数据改变，应同步调整该段。

---

### 3.2 结论必须改写

#### 当前问题

结论中不能再写：

```text
低至中等批量规模下表现出较低端到端延迟。
```

表 5 不支持这个泛化结论。

#### 推荐结论骨架

```text
在统一 UR10、单 seed、关闭碰撞检测的批量 IK 查询任务中，本文提出的 CUDA 小矩阵 DLS 求解器通过 1 block/target 映射、单 kernel 全迭代封装、寄存器级 LDLT 求解和 FP32/FP64 混合精度策略，实现了稳定的线性批量扩展。实验表明，本文方法在 N=100 的极小批量场景下较 cuRobo-Graph 具有更低 GPU 流延迟；当 N≥500 时，cuRobo-Graph 依靠 CUDA Graph 和内部批处理优化通常获得更高吞吐。与其相比，本文方法的优势主要体现在实现结构简单、kernel launch 数恒为 1、性能随 N 线性增长且波动较小。

精度实验表明，FP32 Jacobian 在 ε=10^-6 下对应约 4–5 mm 的 IK 收敛精度边界。因此，本文方法更适合中等精度批量采样、候选目标筛选和粗 IK 初始化；对于 Strict 精度任务，应切换至 ε=10^-5 或全 FP64 路径，并在未来工作中进一步验证。
```

如果补跑了 ε=10^-5 Strict 实验，并且结果支持恢复成功率，可以将最后一句改为实证结论。

---

### 3.3 主贡献表述必须收敛

不要写：

```text
系统性优于 cuRobo。
全面加速 cuRobo。
N≤1000 均具有更低延迟。
混合精度无精度退化。
```

应改成：

```text
在 N=100 极小批量下延迟更低。
在 N≥500 时吞吐通常不超过 cuRobo-Graph，但线性扩展更稳定。
混合精度在 Medium 阈值下收敛率保持 0.998+，但 Strict 阈值下成功率下降，说明存在明确精度边界。
本文方法适合中等精度批量 IK、候选解筛选、粗 IK 初始化，不宜被描述为高精度最终 IK 求解器的完全替代。
```

---

## 4. 必须修正的表格

### 4.1 表 5：主性能对比

表 5 需要保留，但建议改成更严谨的形式：

| N | 方法 | Graph | GPU ms mean±std | E2E ms mean±std | raw t/s | Medium SR | valid t/s | Strict SR |
|---:|---|---|---:|---:|---:|---:|---:|---:|

要求：

1. 所有 30 次重复结果必须给 mean±std；
2. 如果篇幅不足，主表给 mean，附表给 std/p50/p95；
3. N=5000 cuRobo-Graph 必须在复核后再写入；
4. raw t/s 和 valid t/s 必须使用同一计时口径，建议基于 GPU Stream 时间；
5. End-to-End 只作为工程延迟，不用于 raw throughput 主排名，除非明确说明。

---

### 4.2 表 7：A7 vs A8 CUDA Graph 对比

表 7 必须重做。推荐格式：

| N | A7 Direct mean±std/(t/s) | A8 Graph mean±std/(t/s) | diff/% | 结论 |
|---:|---:|---:|---:|---|
| 100 | TODO | TODO | TODO | TODO |
| 500 | TODO | TODO | TODO | TODO |
| 1000 | TODO | TODO | TODO | TODO |
| 5000 | TODO | TODO | TODO | TODO |

禁止在表中用 “–” 代替差异值。即使标准差偏高，也要计算相对均值差异，并在注释中说明统计不确定性。

---

### 4.3 表 6：误差分布

当前表 6 很有价值，但必须在正文中正确解释。建议增加 max 或 p99，如果篇幅允许：

| 方法 | pos mean/mm | pos p50/mm | pos p95/mm | pos max/mm | rot mean/° | rot p50/° | rot p95/° | rot max/° |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

正文必须明确承认：

```text
cuRobo-Graph 的求解精度显著高于 CUDA-Mixed；CUDA-Mixed 的意义在于 Medium 阈值下的中等精度批量求解，不是高精度最终 IK 求解器的完全替代。
```

---

### 4.4 表 9：Nsight Systems 机制分析

表 9 建议补充：

| N | 方法 | Graph | kernel launches | API calls | GPU active/ms | GPU idle/ms | cudaEventRecord | cudaStreamWaitEvent | cudaMalloc time/% |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|

如果部分指标暂时没有，可以保留已有指标，但不要在正文中讨论未统计的指标。

---

## 5. 图像绘制清单

本阶段**暂不要求重绘最终图**，但必须在 `figure_plan_v2.md` 中列出后续需要绘制的图。不要把临时 Python 图直接当最终期刊图。

### 图 1：CUDA block/target 映射结构图

目的：说明每个 target 对应一个 block，block 内完成 FK、Jacobian、H/g、LDLT、update，所有迭代在单 kernel 内完成。

要求：

- 二维结构示意图；
- 中文标注；
- 不要使用手绘风格；
- 不要在图内部写“图 1”，图号只放 caption；
- 标出 `Grid(N)`、`Block(128 threads)`、`thread 0`、`thread 0–5`、`thread 0–35`、`shared memory`、`register LDLT`。

### 图 2：Raw/Valid Throughput 对比

目的：展示 CUDA-Mixed、cuRobo-NoGraph、cuRobo-Graph 随 N 变化的吞吐。

要求：

- 横轴：批量规模 N；
- 纵轴：吞吐 targets/s；
- 同时展示 raw throughput 和 valid throughput，或拆成两张图；
- 对 N=5000 cuRobo-Graph 异常点加注释，前提是复核确认真实。

### 图 3：End-to-End 延迟对比

目的：展示完整工程调用延迟。

要求：

- 横轴：批量规模 N；
- 纵轴：End-to-End time/ms；
- 含 CUDA-Mixed、cuRobo-NoGraph、cuRobo-Graph；
- 如果 E2E 包含 H2D/D2H，caption 中必须说明。

### 图 4：GPU Stream 时间对比

目的：展示纯设备侧执行时间。

要求：

- 横轴：批量规模 N；
- 纵轴：GPU stream time/ms；
- 该图是性能主图之一；
- 与图 3 保持坐标风格统一。

### 图 5：误差分布对比

目的：展示 CUDA-Mixed 和 cuRobo-Graph 的求解质量差异。

要求：

- 推荐使用箱线图或 p50/p95 柱状图；
- 分别展示 position error/mm 和 rotation error/deg；
- 明确显示 cuRobo 精度显著更高；
- 不要用图像掩盖 CUDA-Mixed 接近 Medium 阈值边界的问题。

### 图 6：A7/A8 CUDA Graph 对比

目的：展示 CUDA Graph 对本文单 kernel 方案的影响。

要求：

- 必须基于复核后的 mean±std；
- 如果差异非 <1%，图题和正文不能写“几乎无影响”。

### 图 7：消融实验图

目的：展示 A0 → A5 → A7 的吞吐和收敛率变化。

要求：

- 可用双轴图，但需清晰；
- 左轴：success rate；
- 右轴：valid throughput；
- 不要过度堆叠 A1–A4，主图只展示关键节点即可。

### 图 8：cuRobo 批量退化点分析

目的：展示 cuRobo-NoGraph 在特定 N 下的退化。

要求：

- 横轴：N；
- 纵轴：GPU stream time 或 kernel launch count；
- 标出退化点；
- 图内标题不要写“图 5”之类编号，避免与 caption 冲突。

### 图 9：Nsight Compute 剖析对比

目的：展示计算吞吐、显存吞吐、Bank 冲突、寄存器、kernel 时间等指标。

要求：

- 可以使用表格为主，图为辅；
- 如果图太拥挤，保留表 10 即可。

---

## 6. 参考文献核查

必须逐条核查参考文献，不要保留无法确认的文献。

重点核查：

```text
[7] HJCD-IK: arXiv 2510.07514，但访问日期写 2025-06-14。
[10] SPaSM: arXiv 2510.07674，但访问日期写 2025-06-14。
```

问题：arXiv 编号 `2510` 通常对应 2025 年 10 月，与 2025-06-14 的访问日期不一致。

处理要求：

1. 联网核查文献是否真实存在；
2. 若文献不存在或信息不匹配，删除或替换；
3. 若文献真实存在，修正访问日期和引用格式；
4. 所有中文参考文献按期刊要求补充英文翻译；
5. 所有 URL 类文献必须确保访问日期晚于发布日期。

---

## 7. 格式和写作规范修改

### 7.1 章节编号

当前稿件从 `0 引言` 开始，不符合多数中文期刊习惯。改为：

```text
1 引言
2 批量 IK 问题建模与 DLS 方法
3 CUDA 小矩阵加速设计
4 实验设计
5 实验结果与分析
6 结论
```

所有交叉引用同步更新。

### 7.2 代码 Listing 压缩

当前 Listing 仍偏多。建议：

- 保留 Listing 1，但继续压缩为伪代码；
- Listing 2 可改为文字描述或公式，不一定保留代码块；
- 期刊正文避免出现过多 C/CUDA 风格代码；
- 实现细节可放补充材料或开源仓库说明。

### 7.3 图表编号统一

检查所有图：

- 图内不要写“图 X”；
- caption 统一为“图 X：……”；
- 正文引用与 caption 一致；
- 图中文字、坐标轴、图例语言统一；
- 中文期刊优先中文坐标轴；
- 临时 Python 图不要作为最终图。

### 7.4 避免过度宣传词

禁止或慎用：

```text
全面优于
碾压
显著优于所有场景
无精度退化
完全替代 cuRobo
工业级高精度最终解
```

推荐使用：

```text
在本文设定的单 seed、关闭碰撞检测、Medium 阈值批量 IK 查询任务中
在 N=100 极小批量场景下
在 Medium 阈值下保持较高成功率
在批量扩展上表现出更强可预测性
适合中等精度批量采样、候选解筛选和粗 IK 初始化
```

---

## 8. 修改后的论文核心结论边界

本轮修改后，论文应收敛到以下边界内：

### 可以主张

1. 本文方法在 N=100 极小批量下 GPU stream 延迟低于 cuRobo-Graph；
2. 本文方法 kernel launch 数恒为 1，结构简单；
3. 本文方法 GPU 时间随 N 近似线性增长，性能可预测；
4. 本文方法在 Medium 阈值下成功率保持 0.998+；
5. FP32 Jacobian 在 ε=10^-6 下存在约 4–5 mm 收敛精度边界；
6. 本文方法适合中等精度批量 IK、候选解筛选和粗 IK 初始化。

### 不可以主张

1. 本文方法全面快于 cuRobo-Graph；
2. N≤1000 都比 cuRobo-Graph 快；
3. 混合精度无精度损失；
4. 本文方法可以直接替代 cuRobo 的高精度优化求解；
5. ε=10^-5 已经恢复 Strict 成功率，除非补跑并证明；
6. 带碰撞检测或多 seed 场景已经被本文解决。

---

## 9. 最终交付物要求

执行完成后，请输出以下文件：

```text
1. revision_file_inventory.md
   - 项目文件定位说明

2. experiment_validation_report.md
   - cuRobo N=5000 复核结果
   - A7/A8 复核结果
   - 是否补跑 ε=10^-5 Strict 实验及结果
   - 是否发现实验代码问题

3. updated_tables/
   - table5_main_performance.csv
   - table6_error_distribution.csv
   - table7_a7_a8_graph.csv
   - table9_nsight_systems.csv
   - 如有：strict_epsilon_validation.csv

4. figure_plan_v2.md
   - 后续需要正式绘制的图清单
   - 本阶段不要求绘制最终图

5. revised_paper.md 或 revised_paper.tex
   - 修改后的论文正文
   - 摘要、实验、结论必须与复核后数据一致

6. reference_audit.md
   - 每条参考文献是否核查
   - 有问题的参考文献如何处理
```

---

## 10. 给执行 AI 的硬性要求

1. 不允许编造实验数据。
2. 不允许把单次运行结果写成 30 次平均。
3. 不允许用旧图覆盖新数据。
4. 不允许忽略 N=5000 异常点。
5. 不允许继续写 “N≤1000 延迟降低 52%”。
6. 不允许写 “A7/A8 差异 <1%”，除非复核后所有 N 都支持。
7. 不允许把 ε=10^-5 的 Strict 恢复写成已验证结论，除非补跑实验。
8. 不允许保留时间逻辑不成立的参考文献。
9. 所有 TODO 必须显式保留，不得用虚构数值填补。
10. 最终论文叙事必须服从真实数据，而不是服从原先“加速 cuRobo”的预期。

---

## 11. 最小执行顺序

建议严格按以下顺序执行：

```text
Step 1：备份当前论文和实验数据
Step 2：定位论文源文件、benchmark 脚本、原始日志
Step 3：复核 cuRobo-Graph N=5000
Step 4：复核 A7/A8 Direct vs Graph
Step 5：根据正文需求决定是否补跑 ε=10^-5 Strict
Step 6：生成 experiment_validation_report.md
Step 7：更新表 5、表 6、表 7、表 9
Step 8：重写摘要、实验分析、结论
Step 9：整理 figure_plan_v2.md，不绘制最终图
Step 10：核查参考文献
Step 11：输出 revised_paper.md/tex
```

---

## 12. 本轮修改完成后的判断标准

修改完成后，论文应满足：

- 摘要中的每一句性能结论都能在表格中找到对应数据；
- 表 5 中每个关键值都有原始日志支撑；
- N=5000 cuRobo-Graph 异常点已被确认或修正；
- A7/A8 CUDA Graph 差异已用 mean±std 解释；
- 结论不再夸大为“全面优于 cuRobo”；
- 误差质量差异被正面承认；
- 参考文献不存在明显时间逻辑错误；
- 图表编号和图题不混乱；
- 后续图像绘制已有清单。

只要这些条件满足，论文就从“性能宣传稿”转为“实验口径相对严谨的工程型 CUDA+机器人论文”。
