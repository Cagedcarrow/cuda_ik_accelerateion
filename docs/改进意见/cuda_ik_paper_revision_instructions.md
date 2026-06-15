# 《基于 CUDA 小矩阵加速的机械臂批量逆运动学求解》论文修改说明

> 用途：本说明文档用于指导 AI 对当前论文稿件进行结构性修改、实验补充设计、表格重构和后续绘图规划。  
> 重要限制：**本阶段不绘制图片，只在文档中说明后期需要绘制哪些图、每张图表达什么、需要哪些数据。**  
> 修改原则：严谨、保守、可复现。不要夸大“加速比”，不要回避 cuRobo 对比公平性问题，不要虚构实验数据和参考文献。

---

## 0. 当前论文的总体判断

当前论文已经形成了一个成立的工程型创新点：

> 面向机械臂批量逆运动学中固定规模 6×6 小矩阵反复求解的问题，将 DLS 迭代映射为 CUDA 小矩阵并行计算，通过 1 block/target、单 kernel 全迭代封装、寄存器级 LDLT 求解器、共享内存 padding 和 FP32/FP64 混合精度提升批量 IK 求解吞吐。

但是当前版本存在几个关键风险：

1. 与 cuRobo 的对比计时口径不完全一致；
2. 摘要、正文、表格中的部分数据存在不一致；
3. “kernel launch 从 O(N·K) 降到 O(1)”等表述容易被误解为计算复杂度降低；
4. “6 自由度机械臂通常无闭式解”等机器人基础表述不够严谨；
5. FP32 数值 Jacobian 的稳定性论证不足；
6. LDLT “86 次标量运算”的表达过于绝对；
7. 缺少误差分布、成功吞吐率、统计显著性；
8. 参考文献有待逐条核查；
9. 代码 Listing 占比偏大，不像正式期刊论文；
10. 缺少必要图示，目前表格太多。

修改目标不是把论文写得更“夸张”，而是让论文更像正式期刊论文：**结论边界清楚、实验口径公平、数据可复现、表述不被外审轻易抓住。**

---

## 1. 核心修改目标

请围绕以下主线修改论文：

> 本文不是提出新的 IK 算法，而是针对固定规模 6×6 DLS 小矩阵迭代的 CUDA 底层实现优化，重点证明该实现方式在低至中等批量规模下具有更低固定调度开销、更高工程吞吐和更稳定的批量扩展性。

避免使用以下不严谨表达：

- “全面碾压 cuRobo”
- “完全优于 cuRobo”
- “kernel launch 从 O(N·K) 降到 O(1)，所以复杂度降低”
- “6 自由度机械臂通常无闭式解”
- “86 次标量运算”作为绝对、唯一统计口径
- “Bank 冲突被消除”
- “混合精度完全无误差影响”

推荐使用以下更稳妥表达：

- “在本文设定的单 seed、关闭碰撞检测、统一目标位姿和收敛阈值的批量 IK 查询任务中……”
- “本文方法在小批量场景下显著降低固定调度开销”
- “本文方法的 kernel launch 数为常数级，但总体计算量仍随目标数和迭代次数线性增长”
- “对于部分具有解析结构的 6R 工业机械臂，可存在闭式 IK 解；数值 IK 仍因通用性和扩展性广泛使用”
- “LDLT 被完全展开为常量规模的寄存器级直线代码，算术规模为百级标量操作”
- “PaddedMat6×8 显著降低 Bank 冲突，而非完全消除”
- “混合精度在当前 benchmark 下未观察到显著收敛退化，但仍需通过误差分布验证”

---

## 2. 与 cuRobo 公平对比的补充实验设计

这是最重要的修改内容。原论文中存在如下风险：

- 本文 CUDA 使用 `cudaEventElapsedTime` 测量 kernel device 时间；
- cuRobo 使用 `time.perf_counter()` 测量完整调用栈；
- 两者计时口径不同，却直接给出“加速比”；
- cuRobo 当前设置 `use_cuda_graph=False`，可能被认为人为削弱 cuRobo。

因此需要将对比分为三层，而不是只给一个速度比。

### 2.1 三层计时口径

请在实验设计中新增“计时口径”小节。

#### Level 1：End-to-End 工程时间

用于回答：实际工程调用中，一个 batch 从输入到输出需要多久。

对于本文 CUDA 求解器：

```text
t0
H2D 拷贝目标位姿
调用 CUDA IK kernel
D2H 拷贝求解结果
cudaDeviceSynchronize
t1
```

如果目标和结果本来就在 GPU 上，则另外给出“不含 H2D/D2H”的版本，不要混在一起。

对于 cuRobo：

```text
t0
调用 cuRobo IK solve
torch.cuda.synchronize()
t1
```

注意：

- 不计入模型加载；
- 不计入 URDF 解析；
- 不计入 solver 初始化；
- 不计入第一次 JIT 或 CUDA Graph capture；
- 必须预热后再计时。

#### Level 2：GPU Stream 时间

用于回答：输入已经在 GPU 上时，GPU 设备侧完整执行负载是多少。

cuRobo / PyTorch 侧建议用：

```python
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
result = solver.solve_batch(targets)
end.record()

torch.cuda.synchronize()
gpu_ms = start.elapsed_time(end)
```

本文 CUDA 侧使用：

```cpp
cudaEventRecord(start);
ik_batch_solve<<<N, 128>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
```

注意：cuRobo 可能包含多个 kernel，因此不能只测某一个 kernel。应当用 start/end event 包住整个 cuRobo solve 调用，使其成为 GPU stream 范围内的完整执行时间。

#### Level 3：Nsight Systems 机制分析

用于解释为什么出现速度差异。该部分不是主性能表，而是机制分析表。

建议选取：

```text
N = 100
N = 1000
N = 4000
N = 5000
N = 10000
```

统计：

| 指标 | 含义 |
|---|---|
| CUDA kernel launch 数 | 是否存在大量 kernel launch |
| CUDA API 总时间 | host 侧 CUDA API 调度开销 |
| cudaEventRecord 次数 | 事件记录开销 |
| cudaStreamWaitEvent 次数 | stream 等待开销 |
| cudaMalloc/cudaFree 时间占比 | 排除内存分配是否为主因 |
| GPU active time | GPU 真正执行 kernel 的时间 |
| GPU idle gap | kernel 之间空隙 |

如果继续使用原稿中 N=4000 的 cuRobo 退化数据，则必须给出 Nsight Systems 截图、统计表或脚本统计结果，不能只在正文中描述。

---

### 2.2 cuRobo 必须增加 CUDA Graph 开/关两组

当前论文只写 `use_cuda_graph=False`。这容易被外审认为不公平。应当把它变成实验变量。

至少比较：

| 方法 | 配置 |
|---|---|
| CUDA-Ours | 本文单 kernel 方案 |
| cuRobo-NoGraph | `use_cuda_graph=False` |
| cuRobo-Graph | `use_cuda_graph=True` |

可选：

| 方法 | 配置 |
|---|---|
| CUDA-Ours-Graph | 将本文 kernel 也包装为 CUDA Graph，作为补充实验 |

如果开启 CUDA Graph 后 cuRobo 明显变快，不要回避。论文结论应收缩为：

> 在低至中等批量规模下，本文单 kernel DLS 求解器仍具有较低延迟和较高吞吐；在大批量或 CUDA Graph 优化充分的条件下，cuRobo 的性能可能接近或超过本文方法，但其性能对内部子批次策略和 batch size 更敏感。

---

### 2.3 统一 benchmark 输入

所有方法必须使用同一批目标。不能让不同求解器各自随机生成目标。

建议目标生成方式：

```text
seed = 42
随机采样合法关节角 q_gt ∈ [q_min, q_max]
用 UR10 FK 生成目标位姿 T_target
保存 q_gt、T_target、q_seed
统一初始种子 q_seed = [0,0,0,0,0,0]
```

这样目标天然可达。论文中需明确说明：

> 所有目标均由合法关节空间采样后经 FK 生成，因此理论上存在可行 IK 解。

需要保存的数据字段：

```text
target_id
q_gt_1 ... q_gt_6
target_px target_py target_pz
target_qx target_qy target_qz target_qw 或 target_R_00 ... target_R_22
q_seed_1 ... q_seed_6
```

---

### 2.4 统一求解设置

请在实验设计中用表格写清楚：

| 项目 | 统一设置 |
|---|---|
| 机器人模型 | UR10，统一 URDF / tool0 |
| 目标位姿 | 同一批 `T_target` |
| 初始种子 | zero_seed |
| 碰撞检测 | 关闭 |
| self-collision | 关闭 |
| 最大迭代次数 | 尽量统一；如无法统一，需说明 |
| 收敛阈值 | 主阈值 10 mm / 5°；可补充 5 mm / 1° |
| 重复次数 | 30 次 |
| 预热次数 | 建议 10 次 |
| 统计方式 | mean、std、median、p95 |

必须强调：

> 本文比较的是相同任务条件下不同 GPU IK 实现的工程吞吐与延迟，而不是严格等价的算法内部步骤对比。cuRobo 和本文 DLS 的优化目标、内部迭代过程和实现框架不同，因此加速比应解释为工程吞吐比，而不是算法复杂度比。

---

## 3. 主实验表格重构

### 3.1 主性能表

建议新增或替代表 5：

| N | Method | Graph | E2E ms | GPU stream ms | raw targets/s | success rate | valid targets/s | pos p95/mm | rot p95/deg |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | CUDA-Mixed | - | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 100 | cuRobo | Off | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 100 | cuRobo | On | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 500 | CUDA-Mixed | - | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 500 | cuRobo | Off | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 500 | cuRobo | On | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

其中：

```text
valid targets/s = raw targets/s × success rate
```

这个指标很重要，因为它避免只看速度不看求解成功率。

---

### 3.2 统计显著性表

建议新增：

| N | Method | Timing type | mean/ms | std/ms | median/ms | p95/ms | min/ms | max/ms |
|---:|---|---|---:|---:|---:|---:|---:|---:|

说明 30 次重复实验的波动情况。尤其 cuRobo 如果存在 batch size 退化，std 和 p95 可以反映性能不稳定性。

---

### 3.3 误差分布表

建议新增：

| Method | N | success rate | pos mean/mm | pos median/mm | pos p95/mm | pos max/mm | rot mean/deg | rot median/deg | rot p95/deg | rot max/deg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

该表用于证明：

- CUDA-Mixed 不只是快，而且解的误差没有明显退化；
- 与 FP64 相比，混合精度没有导致精度不可接受；
- 与 cuRobo 相比，本文方法在统一阈值下求解质量可比。

---

### 3.4 Nsight Systems 机制分析表

建议新增：

| N | Method | Graph | kernel launches | CUDA API calls | cudaEventRecord | cudaStreamWaitEvent | cudaMalloc/free time % | GPU active time/ms | total elapsed/ms |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|

注意：这张表不是用来证明“谁一定更好”，而是解释：

- 本文单 kernel 方案为什么小批量固定开销低；
- cuRobo 某些 batch size 为什么出现非单调退化；
- CUDA Graph 是否缓解了大量 kernel launch 的开销。

---

## 4. 摘要和结论必须修改

### 4.1 摘要中的数据一致性问题

当前摘要中存在风险表述：

> N = 100 → 10,000 全量程上吞吐稳定在 148k–174k targets/s。

但原表中 N=100 的 CUDA-Mixed 约为 112k targets/s，不在 148k–174k 范围内。

需要改为：

> 当 N ≥ 500 时，CUDA-Mixed 吞吐稳定在约 148k–174k targets/s；N=100 时受 block 数不足和固定开销影响，吞吐约为 112k targets/s。

或者根据新的补充实验数据重新写摘要，不要保留旧数据矛盾。

---

### 4.2 摘要建议改写方向

摘要应从“单一加速比”改为“分层计时 + 工程意义”：

建议结构：

1. 问题背景：批量 IK 中 6×6 小矩阵反复求解和多 kernel 调度带来固定开销；
2. 方法：1 block/target、单 kernel 全迭代、寄存器 LDLT、PaddedMat6×8、混合精度；
3. 实验：统一 UR10 benchmark，同目标、同初始种子、同收敛阈值，对比 CUDA-Mixed、cuRobo Graph on/off；
4. 结果：分别报告 end-to-end、GPU stream、valid throughput、误差分布；
5. 结论：本文在低至中等批量规模下具有更低调度开销和更稳定扩展性。

不要只写“比 cuRobo 快 36.1 倍”。

---

### 4.3 结论建议改写方向

结论需要更加保守：

错误倾向：

> 本文全面证明 CUDA 小矩阵方案优于 cuRobo。

推荐写法：

> 在本文设定的 UR10 单 seed 批量 IK 查询任务中，自定义 CUDA 小矩阵求解器在低至中等批量规模下表现出较低端到端延迟和较稳定的批量扩展性。与框架型 GPU IK 实现相比，本文方法的优势主要来自单 kernel 全迭代封装和固定规模寄存器级小矩阵求解，从而降低了 kernel launch、跨 kernel 同步和框架调度开销。需要指出的是，本文未考虑碰撞检测、多 seed 全局搜索和轨迹级优化，后续工作将扩展至 7 自由度冗余机械臂、多初始种子并行化以及带碰撞约束的运动规划场景。

---

## 5. 方法部分的关键表述修改

### 5.1 “O(N·K) 到 O(1)”问题

当前表达容易被误解为总体计算复杂度降低。应修改为：

> 本文并未改变 DLS 本身随目标数 N 和迭代次数 K 线性增长的计算量，而是将 host 侧 kernel launch 和跨 kernel 同步次数压缩到常数级。对于小批量任务，固定调度开销在总延迟中占比较高，因此单 kernel 全迭代封装能够显著降低端到端延迟。

---

### 5.2 “6 自由度通常无闭式解”问题

请把原文类似表述改为：

> 尽管部分具有球腕或特定几何结构的 6R 工业机械臂可以推导解析 IK，数值 IK 仍因模型适应性强、易处理关节限位和扩展约束，在批量采样、轨迹优化和通用机器人软件框架中被广泛采用。

这样更符合机器人学常识。

---

### 5.3 “86 次标量运算”问题

请弱化绝对说法。改为：

> 对于固定 n=6 的阻尼 Hessian，LDLT 分解和三角回代可在编译期完全展开为常量规模的寄存器级直线代码，其算术规模为百级标量操作。相比调用通用批量线性代数库，该方式避免了句柄调度、批量 kernel 分发和额外数据搬运开销。

如果保留“86 次”，必须明确统计口径：

- FMA 计为 1 条指令还是 2 FLOP；
- DIV 是否计入；
- load/store 是否计入；
- 类型转换是否计入；
- `a*b*c` 是否拆成多个乘法。

否则不要把“86”作为核心宣传点。

---

### 5.4 FP32 数值 Jacobian 风险

原论文声称 FP32 FK/Jacobian 可接受，但当前论证不足。必须增加验证实验或至少在修改稿中添加实验计划。

建议新增实验：

#### Jacobian 精度验证

比较：

```text
FP64 Jacobian
FP32 Jacobian
不同差分步长 ε = 1e-4, 1e-5, 1e-6, 1e-7
```

统计：

| ε | mean relative error | median relative error | p95 relative error | max relative error | success rate | throughput |
|---:|---:|---:|---:|---:|---:|---:|

相对误差定义：

```text
||J_FP32 - J_FP64||_F / ||J_FP64||_F
```

目的：

1. 证明 ε=1e-6 在 FP32 下不是拍脑袋；
2. 检查是否存在消减误差；
3. 确认混合精度收益没有以 Jacobian 严重失真为代价。

如果实验结果显示 ε=1e-5 更稳，应当调整论文设置。

---

### 5.5 Bank 冲突表达

避免写“消除 Bank 冲突”。应写：

> PaddedMat6×8 通过改变行步长降低了特定访问模式下的 Bank 冲突。Nsight Compute 结果显示，Bank conflict 相关指标下降约 63%，但仍存在来自其他访问路径的残余冲突。

---

### 5.6 代码 Listing 过长问题

第 3、4 页代码 Listing 占比过高。建议：

- 正文保留一个精简伪代码；
- 删除大量 C/CUDA 风格细节；
- 把具体实现细节放到附录；
- 用后续图 1 展示 block/target 映射和数据流。

正文中的算法只保留：

```text
输入：目标位姿 T_target，初始关节 q0
输出：关节解 q*
for each target block in parallel:
    initialize q
    for k = 1 ... Kmax:
        compute FK and pose error
        if converged: break
        compute numerical Jacobian
        update damping λ
        construct H and g
        solve H Δq = -g by register-level LDLT
        clamp step and update q
    write result and status
```

---

## 6. 实验部分应新增或修订的小节结构

建议第 4 节改为：

```text
4 实验设计与评价方法
4.1 统一 Benchmark 与目标生成
4.2 对比方法与参数配置
4.3 计时口径：End-to-End、GPU Stream 与 Nsight Systems
4.4 评价指标：吞吐、成功率、有效吞吐、误差分布
4.5 消融实验设计
4.6 统计方法与重复实验
```

建议第 5 节改为：

```text
5 实验结果与分析
5.1 主性能对比：End-to-End 与 GPU Stream
5.2 CUDA Graph 对 cuRobo 的影响
5.3 求解质量与误差分布
5.4 消融实验：阻尼、混合精度与内存布局
5.5 Nsight Systems 机制分析
5.6 Nsight Compute kernel 剖析
5.7 批量规模扩展性与退化点分析
```

---

## 7. 后续需要绘制的图像清单

本阶段不要绘图，只在论文中预留图题和图像说明。后续根据实验数据绘制。

### 图 1：CUDA block/target 映射与单 kernel 全迭代结构图

目的：解释本文核心并行映射。

应包含：

- Grid 中每个 block 对应一个 IK target；
- block 内 128 threads；
- thread 0 负责 FK/误差/LDLT；
- thread 0–5 负责 Jacobian 列和 g 向量；
- thread 0–35 负责 H 矩阵 36 个元素；
- shared memory 保存 q、J、H、g、误差；
- register 保存 LDLT 中间量；
- 单 kernel 内完成 Kmax 次迭代。

图题建议：

> 图 1 1 block/target 的批量 IK CUDA 并行映射结构

---

### 图 2：主性能吞吐-N 曲线

目的：展示 CUDA-Mixed、cuRobo-NoGraph、cuRobo-Graph 在不同 N 下的 raw throughput 和 valid throughput。

横轴：

```text
N = 100, 500, 1000, 5000, 10000
```

纵轴：

```text
targets/s
```

建议画两张或一张双指标图：

- raw throughput；
- valid throughput。

图题建议：

> 图 2 不同批量规模下的批量 IK 吞吐对比

---

### 图 3：End-to-End 延迟-N 曲线

目的：展示实际工程调用延迟。

横轴：

```text
N
```

纵轴：

```text
End-to-End latency / ms
```

曲线：

- CUDA-Mixed；
- cuRobo-NoGraph；
- cuRobo-Graph。

图题建议：

> 图 3 不同批量规模下的端到端延迟对比

---

### 图 4：GPU Stream 时间-N 曲线

目的：区分工程调用开销和 GPU 设备侧执行时间。

横轴：

```text
N
```

纵轴：

```text
GPU stream time / ms
```

图题建议：

> 图 4 不同批量规模下的 GPU stream 执行时间对比

---

### 图 5：cuRobo 退化点与 kernel launch 数关系图

目的：解释为什么 cuRobo 在某些 N 值出现非单调退化。

横轴：

```text
N = 100, 1000, 4000, 5000, 7000, 8000, 9000, 10000
```

左纵轴：

```text
total elapsed time / ms
```

右纵轴：

```text
kernel launch count
```

图题建议：

> 图 5 cuRobo 批量规模退化点与 kernel launch 数关系

---

### 图 6：消融实验柱状图

目的：展示 FP64 基线、FP64 自适应阻尼、CUDA-Mixed 的性能提升来源。

横轴：

```text
FP64 Baseline
FP64 Adaptive Damping
CUDA Mixed Precision
```

纵轴：

```text
targets/s 或 normalized speedup
```

建议按 N=100、500、5000 分组。

图题建议：

> 图 6 不同优化阶段的吞吐提升对比

---

### 图 7：收敛率与有效吞吐对比图

目的：避免只看 raw throughput。

横轴：

```text
Method
```

纵轴 1：

```text
success rate
```

纵轴 2：

```text
valid targets/s
```

图题建议：

> 图 7 不同方法的收敛率与有效吞吐对比

---

### 图 8：误差分布箱线图或分位数图

目的：证明 CUDA-Mixed 没有明显牺牲求解精度。

应分别画：

- 位置误差；
- 姿态误差。

方法：

- CUDA-FP64；
- CUDA-Mixed；
- cuRobo-Graph；
- cuRobo-NoGraph。

图题建议：

> 图 8 不同求解器的位置与姿态误差分布

---

### 图 9：Nsight Compute kernel 剖析柱状图

目的：展示计算密集型特征和 bank conflict 优化效果。

指标：

- compute throughput；
- memory throughput；
- register/thread；
- bank conflicts；
- local memory spill；
- kernel time。

图题建议：

> 图 9 CUDA kernel 的 Nsight Compute 剖析指标对比

---

## 8. 参考文献核查要求

必须逐条核查参考文献，尤其是以下风险：

1. arXiv 编号与访问日期是否矛盾；
2. 论文是否真实存在；
3. 作者名是否拼写正确；
4. 期刊名、卷号、页码是否正确；
5. 访问日期是否晚于论文发布日期；
6. 中文文献格式是否符合《系统工程与电子技术》要求；
7. 不要引用不存在或 AI 幻觉生成的论文。

特别注意：

- 若 arXiv 编号类似 `2510.xxxxx`，则通常代表 2025 年 10 月，不应出现 2025 年 6 月访问日期；
- 如果论文当前尚未公开，不要引用；
- 如果只是项目 GitHub，应明确标为 `[EB/OL]`；
- CUDA Programming Guide 应引用官方文档，并使用实际访问日期；
- cuRobo 应引用正式 arXiv 或官方论文页面。

---

## 9. 作者信息与模板占位

当前稿件中存在占位内容，正式投稿前必须删除或替换：

- “刘小明”
- “某某某”
- “XXXX 大学 XXXX 学院”
- “XXXX 研究所”
- “XX XXXXX”
- 作者简介中的手机号、身份证号等占位
- DOI、文章编号、收稿日期等预留字段

如果当前只是内部匿名稿，可保留匿名，但不能混用“匿名”和“占位符”。

---

## 10. 需要新增的数据文件清单

建议让实验脚本最终输出以下 CSV，便于后续绘图和写论文。

### 10.1 `benchmark_targets.csv`

```text
target_id
N_group
q_gt_1 ... q_gt_6
target_px target_py target_pz
target_qx target_qy target_qz target_qw
q_seed_1 ... q_seed_6
```

### 10.2 `performance_summary.csv`

```text
method
graph
N
timing_type        # e2e 或 gpu_stream
repeat_id
time_ms
raw_targets_per_s
success_rate
valid_targets_per_s
```

### 10.3 `error_summary.csv`

```text
method
graph
N
target_id
success
position_error_mm
rotation_error_deg
num_iterations
timeout
joint_limit_violation
```

### 10.4 `nsight_systems_summary.csv`

```text
method
graph
N
kernel_launches
cuda_api_calls
cuda_event_record_count
cuda_stream_wait_event_count
cuda_malloc_free_time_percent
gpu_active_time_ms
total_elapsed_ms
```

### 10.5 `nsight_compute_summary.csv`

```text
method
N
compute_throughput_percent
dram_throughput_percent
registers_per_thread
occupancy_percent
bank_conflict_count
l1_hit_rate_percent
local_memory_spill
kernel_time_us
```

### 10.6 `jacobian_precision_summary.csv`

```text
epsilon
N
mean_relative_error
median_relative_error
p95_relative_error
max_relative_error
success_rate
throughput_targets_per_s
```

---

## 11. 具体修改任务清单

AI 修改论文时，请按以下顺序执行。

### 第一阶段：先修硬伤

1. 检查摘要、正文、表格所有数据是否一致；
2. 修正 N=100 吞吐不在 148k–174k 范围内的问题；
3. 删除或弱化“全面优于 cuRobo”的表达；
4. 删除或改写“O(N·K) 降到 O(1)”；
5. 改写“6 自由度通常无闭式解”；
6. 弱化“86 次标量运算”的绝对说法；
7. 将“Bank 冲突消除”改为“Bank 冲突降低”；
8. 删除作者、单位、简介中的占位符或统一匿名化；
9. 标记所有缺失实验数据为 `TODO: 待补实验数据`，不要编造。

### 第二阶段：重写实验设计

1. 增加统一 benchmark 目标生成方法；
2. 增加对比方法配置；
3. 增加 End-to-End、GPU Stream、Nsight Systems 三类计时口径；
4. 增加 cuRobo Graph on/off 配置；
5. 增加 success rate、valid throughput、误差分布指标；
6. 增加重复实验统计方法；
7. 增加需要输出的 CSV 数据格式。

### 第三阶段：重写实验结果框架

在没有新实验数据前，不要编造结果。可以写成：

```text
本节将在补充实验完成后报告……
```

或者保留原有结果，但必须说明：

```text
以下结果为 device-only 初步结果，完整 end-to-end 和 CUDA Graph 对照将在补充实验中给出。
```

但最终投稿版不能保留这种“待补”表述，必须填入真实数据。

### 第四阶段：压缩代码 Listing

1. 删除过长 CUDA/C++ 风格代码；
2. 正文只保留抽象伪代码；
3. 把实现细节转为文字说明；
4. 预留图 1 作为后续并行结构图。

### 第五阶段：准备图表位置

本阶段不画图，但在正文中预留：

```text
图 1 1 block/target 的批量 IK CUDA 并行映射结构
图 2 不同批量规模下的批量 IK 吞吐对比
图 3 不同批量规模下的端到端延迟对比
图 4 不同批量规模下的 GPU stream 执行时间对比
图 5 cuRobo 批量规模退化点与 kernel launch 数关系
图 6 不同优化阶段的吞吐提升对比
图 7 不同方法的收敛率与有效吞吐对比
图 8 不同求解器的位置与姿态误差分布
图 9 CUDA kernel 的 Nsight Compute 剖析指标对比
```

---

## 12. 最终论文应形成的核心结论

修改后论文的核心结论建议为：

> 本文面向 UR10 单 seed 批量 IK 查询任务，提出了一种基于 CUDA 小矩阵定制计算的 DLS 求解器。该方法通过 1 block/target 映射、单 kernel 全迭代封装和寄存器级 6×6 LDLT 求解，降低了小批量任务中的固定调度开销；通过 FP32/FP64 混合精度在保持收敛质量的前提下提升 FK 与 Jacobian 计算吞吐。统一 benchmark 结果应从端到端时间、GPU stream 时间、有效吞吐和误差分布四个维度报告。与 cuRobo 相比，本文方法的优势主要体现在低至中等批量规模下的低延迟和扩展稳定性，而不是在所有机器人运动生成任务中替代 cuRobo。本文尚未处理碰撞检测、多 seed 全局搜索和轨迹优化，后续工作将扩展至 7 自由度机械臂、多初始种子并行化和带碰撞约束的 GPU 运动规划。

---

## 13. 禁止事项

修改论文时禁止：

1. 编造补充实验数据；
2. 编造参考文献；
3. 把 device-only 结果写成 end-to-end 结果；
4. 把 raw throughput 写成 valid throughput；
5. 混淆 cuRobo Graph on/off；
6. 在没有误差分布的情况下声称“精度完全不变”；
7. 在没有 Nsight 证据的情况下声称 cuRobo 退化原因已经完全确定；
8. 把本文方法描述成完整运动规划器；
9. 把无碰撞 IK 结果推广到工业完整运动规划场景；
10. 用“碾压”“吊打”“全面领先”等非学术表述。

---

## 14. 明天交给 AI 的执行提示词

可以直接把下面这段作为提示词发给代码/写作 AI：

```text
请读取当前论文稿件，并按照本 md 文档的要求进行结构性修改。

修改重点不是润色语言，而是解决论文当前的严谨性问题：
1. 重构与 cuRobo 的公平对比实验设计；
2. 将计时口径拆分为 End-to-End、GPU Stream 和 Nsight Systems 三层；
3. 增加 cuRobo CUDA Graph on/off 两组对比；
4. 补充 success rate、valid throughput、误差分布、统计显著性等指标；
5. 修正摘要、正文和表格中的数据不一致；
6. 弱化不严谨或过度宣传的表述；
7. 压缩过长代码 Listing；
8. 预留后续需要绘制的图，但本阶段不要绘图；
9. 不要编造任何实验结果和参考文献，缺失处统一标注 TODO；
10. 保持中文期刊论文风格，重点体现“CUDA 小矩阵底层实现优化”和“批量 IK 工程吞吐/扩展稳定性”。

请先输出修改计划，再逐节修改论文正文。
```
