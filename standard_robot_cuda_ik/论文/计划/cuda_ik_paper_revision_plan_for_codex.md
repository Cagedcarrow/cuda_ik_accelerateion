# CUDA 机械臂批量 IK 论文补实验与正文修正任务书

> 目标：在现有论文 `paper.tex / paper.pdf` 基础上，补齐关键实验、重绘必要图表、修正正文中容易被审稿人质疑的表述，使论文从“系统级实现展示”提升为“实验支撑充分、边界明确、可投稿的大修版”。

---

## 0. 总体原则

### 0.1 论文定位必须收敛

不要把论文写成“全面超过 cuRobo”。

应改成：

> 本文面向固定 6-DOF、固定小矩阵规模、N≤1000 的中小批量 IK 场景，通过运动学结构编译期特化、Sobol 多起点并行和单 kernel fusion，实现成功率优先、延迟确定的批量 IK 求解框架。

核心主张：

1. **不是通用 IK 最优器**；
2. **不是全面替代 cuRobo**；
3. **是在固定结构、中小批量、成功率优先场景下的一条专用 CUDA 设计路线**；
4. **与 cuRobo 的关系应写成 Pareto trade-off，而不是绝对优劣**。

### 0.2 所有新增实验统一输出格式

建议所有实验统一输出为 CSV：

```text
results/
├── fair_curobo_k16_summary.csv
├── near_singular_summary.csv
├── near_limit_barrier_summary.csv
├── trajectory_continuity_summary.csv
├── cpu_baseline_summary.csv
├── nsight_systems_summary.csv
├── kernel_time_breakdown.csv
├── barrier_weight_scan.csv
├── seed_count_scan.csv
├── lm_iter_scan.csv
└── threshold_scan.csv
```

每个 CSV 至少包含：

```text
method
N
K
gpu_time_ms
throughput_targets_per_s
strict_sr
medium_sr
loose_sr
pos_p95_all_mm
pos_p95_success_mm
rot_p95_all_deg
rot_p95_success_deg
mean_iter
p95_iter
near_limit_ratio
nan_count
inf_count
fail_count
repeat_id 或 mean/std
```

如果某项不适用，用空值，不要删除列。

### 0.3 计时协议统一

所有 GPU 实验必须写清楚：

```text
warmup = 10
repeat = 30
timing = CUDA event
exclude = first-run compilation / memory allocation / graph capture
include = kernel execution + necessary device-side work
```

cuRobo 实验必须额外说明：

```text
CUDA Graph 是否开启
是否包含 graph capture
是否关闭碰撞检测
num_seeds
maximum iteration
position/orientation threshold
是否使用同一外部 FK 重新评估
```

---

# 1. 必须补充实验

## 1.1 cuRobo K=16 公平对比实验

### 目的

当前论文主要比较：

```text
OPT4C K=16
vs
cuRobo K=1 default
```

该比较可以作为系统默认配置对比，但不能作为公平质量上限对比。

必须新增：

```text
OPT4C K=16
vs
cuRobo K=16
```

### 实验设置

规模：

```text
N = 100, 200, 300, ..., 1000
```

方法：

```text
OPT4C-K16
cuRobo-Graph-K1
cuRobo-Graph-K16
可选：OPT4C-K1
```

cuRobo 配置必须固定并写入论文：

```text
collision_check = false
num_seeds = 16
CUDA Graph = enabled
warmup = 10
repeat = 30
timing = CUDA event 或 torch.cuda.Event
external FK reevaluation = true
```

### 输出文件

```text
results/fair_curobo_k16_summary.csv
```

### 表格

新增表：

**表 X 同等 K=16 条件下 OPT4C 与 cuRobo-Graph 对比**

列：

| N | Method | K | Time/ms | Throughput | Strict SR | Success p95/mm | All p95/mm |
|---|---|---:|---:|---:|---:|---:|---:|

### 图

新增 Pareto 图：

```text
x-axis = throughput_targets_per_s
y-axis = strict_sr
points = OPT4C-K1, OPT4C-K16, cuRobo-K1, cuRobo-K16
```

保存：

```text
figures/fig_pareto_throughput_success.pdf
```

### 正文结论写法

推荐写法：

```text
在同等 K=16 条件下，cuRobo 通常取得更高的成功样本精度和更强的全局搜索能力；OPT4C 的优势主要体现在中小批量下的固定延迟、单 kernel 执行路径和较低调度复杂度。因此，本文方法与 cuRobo 并非简单替代关系，而是在吞吐量、成功率和工程确定性之间形成不同的 Pareto 取舍。
```

避免写：

```text
本方法全面优于 cuRobo。
```

---

## 1.2 Near Singular 近奇异位姿实验

### 目的

当前目标全部由随机关节角 FK 生成，难度偏低。需要验证近奇异位形下的稳定性。

### 建议生成方式

构造三类目标：

```text
A. wrist singular: q5 ≈ 0
B. elbow singular: q2 或 q3 接近伸直/折叠位置
C. shoulder singular: q1/q2 组合导致末端接近肩部奇异区域
```

具体实现建议：

```python
# 示例，仅供 Codex 实现时参考
q = random_uniform(qmin, qmax)
q[4] = uniform(-0.03, 0.03)  # wrist singular
target = FK(q)
```

每类至少：

```text
N = 100, 500, 1000
```

方法：

```text
OPT4C-K16
OPT4C-K1
cuRobo-K16
可选：cuRobo-K1
```

### 输出文件

```text
results/near_singular_summary.csv
```

### 指标

```text
Strict SR
Medium SR
Loose SR
pos_p95_all_mm
pos_p95_success_mm
rot_p95_success_deg
mean_iter
p95_iter
fail_count
nan_count
```

### 图

```text
figures/fig_near_singular_sr.pdf
figures/fig_near_singular_error.pdf
```

图建议：

1. grouped bar：不同方法 Strict SR；
2. line/bar：p95 error；
3. 可选：不同奇异类型分组。

### 正文结论写法

如果结果好：

```text
近奇异目标下，OPT4C-K16 的成功率较随机可达目标有所下降，但仍明显优于 K=1 配置，说明 Sobol 多起点对奇异附近的局部极小和条件数恶化具有缓解作用。
```

如果结果不好：

```text
近奇异目标下成功率明显下降，表明当前 LM 阻尼策略和固定 Sobol 种子仍不足以完全处理奇异区域。该结果限定了本文方法的适用边界，也为后续引入奇异鲁棒阻尼和任务相关 seed warm-start 提供依据。
```

不要隐瞒失败。

---

## 1.3 Near Limit 与 Barrier ON/OFF 实验

### 目的

论文使用了 limit barrier：

```text
wlimit = 0.03
margin = 0.087 rad
```

但当前没有实验证明其必要性。必须补。

### 实验设计

生成接近关节限位的目标：

```text
至少一个关节距离 qmin 或 qmax 小于 5°
```

构造方式：

```python
q = random_uniform(qmin, qmax)
selected_joint = random choice from 0..5
q[selected_joint] = qmin[selected_joint] + uniform(0, 0.087)
# 或 qmax - uniform(0, 0.087)
target = FK(q)
```

方法：

```text
OPT4C-K16-BarrierON
OPT4C-K16-BarrierOFF
OPT4C-K1-BarrierON
OPT4C-K1-BarrierOFF
```

规模：

```text
N = 100, 500, 1000
```

### 输出文件

```text
results/near_limit_barrier_summary.csv
```

### 指标

重点指标：

```text
Strict SR
near_limit_ratio
joint_violation_count
pos_p95_all_mm
success_p95_mm
mean_iter
```

其中：

```text
near_limit_ratio = 输出解中任一关节距离限位 < margin 的比例
joint_violation_count = clamp 前或最终输出超出限位的数量
```

### 表格

新增表：

**表 X 近限位目标下 Barrier ON/OFF 对比**

| Method | N | Strict SR | near-limit ratio | violation count | p95/mm |
|---|---:|---:|---:|---:|---:|

### 图

```text
figures/fig_near_limit_barrier.pdf
```

建议双 y 轴或两个子图：

1. Strict SR；
2. near-limit ratio。

### 正文结论写法

```text
Barrier 的作用不是显著提高成功率，而是在尽量保持成功率的同时降低近限位解比例。因此应将其定位为解质量与安全裕度约束，而非主要收敛加速手段。
```

---

## 1.4 Trajectory 连续性实验

### 目的

机器人真实使用中 IK 通常不是孤立目标，而是连续轨迹。需要证明多起点候选选择不会造成严重关节跳变。

### 轨迹生成

建议至少三类轨迹：

```text
line_50: 直线轨迹 50 点
arc_50: 圆弧轨迹 50 点
random_local_50: 局部随机扰动轨迹 50 点
```

每类生成多条：

```text
num_traj = 20 或 50
points_per_traj = 50
```

方法：

```text
OPT4C-K16-no-rerank
OPT4C-K16-smoothness-rerank
cuRobo-K16
可选：OPT4C-K1
```

如果当前代码已有 smoothness rerank，打开并测试。

### 指标

对每条轨迹计算：

```text
trajectory_success_rate = 全部点成功的轨迹比例
point_success_rate = 点级成功率
mean_delta_q = mean(||q_t - q_{t-1}||)
p95_delta_q = p95(||q_t - q_{t-1}||)
max_delta_q = max(||q_t - q_{t-1}||)
joint_jump_count = ||q_t - q_{t-1}|| > threshold 的次数
```

建议 threshold：

```text
joint_jump_threshold = 0.5 rad 或 1.0 rad
```

### 输出文件

```text
results/trajectory_continuity_summary.csv
```

### 图

```text
figures/fig_trajectory_delta_q.pdf
figures/fig_trajectory_success.pdf
```

图建议：

1. boxplot：不同方法的 `delta_q` 分布；
2. bar：trajectory success rate；
3. 可选：一条典型轨迹的 joint curve。

### 正文结论写法

```text
轨迹实验表明，单点最优的 pose cost 选择可能导致相邻时刻解分支切换；加入 smoothness rerank 后，平均关节跳变量和 p95 跳变量显著降低，而 Strict SR 仅小幅变化。这说明本文框架可通过候选级重排序兼顾单点精度与轨迹连续性。
```

如果当前没有 smoothness rerank，至少补一个简单版本：

```text
candidate_score = pose_cost + beta * ||q_candidate - q_prev||^2
```

---

## 1.5 CPU Baseline 实验

### 目的

论文引言引用了 KDL、TRAC-IK、MoveIt，但实验没有 CPU baseline。审稿人会认为引用和实验脱节。

### 方法

至少选择：

```text
KDL
TRAC-IK
可选：MoveIt IK
```

如果环境搭建困难，最低限度使用一个 CPU DLS/LM baseline：

```text
CPU-LM-K1
CPU-LM-K16
```

### 规模

不必全量跑：

```text
N = 100, 500, 1000
```

如果 CPU 太慢：

```text
N = 100, 200
```

并说明：

```text
CPU baseline 仅用于量级对照。
```

### 输出文件

```text
results/cpu_baseline_summary.csv
```

### 表格

**表 X CPU 与 GPU IK 基准对比**

| Method | Device | N | Time/ms | Throughput | Strict SR | p95/mm |
|---|---|---:|---:|---:|---:|---:|

### 正文结论写法

```text
CPU baseline 的目的不是证明 GPU 对单目标 IK 总是更优，而是说明当目标数量达到百级以上时，批量并行结构能够显著摊薄调度和迭代开销。
```

---

## 1.6 Nsight Systems 单 Kernel Timeline 实验

### 目的

论文强调 kernel fusion 和单次 launch，最好用 Nsight Systems 时间线直接证明。

### 操作

对 N=1000 做一次 Nsight Systems 采样：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  -o reports/nsys_opt4c_n1000 \
  ./your_benchmark_binary --N 1000 --K 16
```

如果有 cuRobo：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  -o reports/nsys_curobo_k16_n1000 \
  python run_curobo_k16.py --N 1000
```

### 输出

```text
reports/nsys_opt4c_n1000.qdrep
reports/nsys_opt4c_n1000.sqlite
reports/nsys_opt4c_n1000_summary.txt
figures/fig_nsys_timeline_opt4c.pdf 或 .png
```

### 论文使用

新增图：

```text
图 X Nsight Systems 时间线：OPT4C 单 kernel launch 执行路径
```

图中标注：

```text
H2D
IK_LM_Multiseed_Block_Target kernel
D2H
```

### 正文结论写法

```text
Nsight Systems 时间线显示，OPT4C 的核心求解阶段由单个长 kernel 构成，未出现多阶段 kernel 往返和 host-device 同步链。这从系统层面验证了 kernel fusion 的执行路径。
```

---

## 1.7 Kernel Time Breakdown 实验

### 目的

当前论文写了：

```text
kernel 99.76%
H2D 0.20%
D2H 0.04%
launch <0.01%
```

建议系统化输出并作图，而不是只写文字。

### 输出文件

```text
results/kernel_time_breakdown.csv
```

列：

```text
N
h2d_ms
kernel_ms
d2h_ms
launch_ms
total_ms
h2d_percent
kernel_percent
d2h_percent
launch_percent
```

规模：

```text
N = 100, 500, 1000
```

### 图

```text
figures/fig_kernel_time_breakdown.pdf
```

图类型：

```text
stacked bar
```

### 正文结论写法

```text
当 kernel fusion 后，OPT4C 的总时间几乎完全由核心计算 kernel 贡献，主机-设备传输和 launch 时间均处于次要地位。这说明后续优化方向应从减少调度转向降低单迭代算术代价。
```

---

## 1.8 Barrier 权重扫描实验

### 目的

说明为什么选择：

```text
wlimit = 0.03
```

### 扫描范围

```text
wlimit = 0
wlimit = 0.005
wlimit = 0.01
wlimit = 0.03
wlimit = 0.05
wlimit = 0.1
```

规模：

```text
N = 1000
```

目标：

```text
random reachable
near-limit reachable
```

最好两个场景都测。

### 输出文件

```text
results/barrier_weight_scan.csv
```

### 指标

```text
Strict SR
near_limit_ratio
p95_all_mm
mean_iter
```

### 图

```text
figures/fig_barrier_weight_scan.pdf
```

建议：

1. x-axis: wlimit；
2. left y-axis: Strict SR；
3. right y-axis: near_limit_ratio；
4. 或者两个子图，不使用双轴也可以。

### 正文结论写法

```text
wlimit 过小无法有效降低近限位解比例，过大则可能牺牲收敛成功率。wlimit=0.03 在成功率和近限位比例之间取得折中，因此作为默认参数。
```

---

## 1.9 Seed 数量扫描实验

### 目的

当前只有：

```text
K=1 vs K=16
```

不足以说明多起点收益何时饱和。

### 扫描范围

```text
K = 1, 2, 4, 8, 16, 32
```

如果当前 kernel 固定 K=16，至少实现：

```text
K = 1, 2, 4, 8, 16
```

K=32 可选，可以用两个 block 或 32 lane 全用，但要注意修改选择逻辑。

### 规模

```text
N = 100, 500, 1000
```

### 输出文件

```text
results/seed_count_scan.csv
```

### 指标

```text
Strict SR
throughput
gpu_time_ms
p95_all_mm
mean_iter
```

### 图

```text
figures/fig_seed_count_scan.pdf
```

图建议：

1. x-axis: K；
2. y1: Strict SR；
3. y2 或第二图：throughput；
4. 标出 K=16 为默认点。

### 正文结论写法

```text
随着 K 增大，成功率显著提升但吞吐下降；当 K 从 8 增至 16 时，成功率收益趋于饱和，而吞吐损失仍可接受。因此 K=16 被选为默认配置。
```

---

## 1.10 LM 最大迭代次数扫描实验

### 目的

说明为什么选择：

```text
max_iter = 60
```

### 扫描范围

```text
max_iter = 20, 40, 60, 80, 100
```

规模：

```text
N = 1000
K = 16
```

可选再测：

```text
near_singular N=1000
```

### 输出文件

```text
results/lm_iter_scan.csv
```

### 指标

```text
Strict SR
throughput
mean_iter
p95_iter
p95_all_mm
```

### 图

```text
figures/fig_lm_iter_scan.pdf
```

### 正文结论写法

```text
max_iter=60 在成功率和计算时间之间取得折中；继续增加迭代次数带来的成功率增益较小，但线性增加 kernel 执行时间。
```

---

## 1.11 不同误差阈值扫描实验

### 目的

避免论文只依赖一个 Strict 阈值。

### 阈值建议

```text
Loose: 30 mm / 10 deg
Medium: 10 mm / 5 deg
Strict: 5 mm / 1 deg
Ultra: 2 mm / 0.5 deg
```

如果 Ultra 太苛刻，可选。

### 方法

```text
OPT4C-K16
OPT4C-K1
cuRobo-K16
cuRobo-K1
```

规模：

```text
N = 1000
```

### 输出文件

```text
results/threshold_scan.csv
```

### 图

```text
figures/fig_threshold_scan.pdf
```

图建议：

```text
x-axis = threshold level
y-axis = success rate
method = line color / marker
```

### 正文结论写法

```text
阈值扫描显示，OPT4C 的优势主要体现在 Strict 附近的成功率保障；当阈值进一步收紧到 Ultra 时，cuRobo 的成功样本精度优势更明显。这进一步说明本文方法适用于工程可用精度阈值下的高成功率批量求解，而非追求最小残差的高精度局部优化。
```

---

# 2. 必须新增或重绘的图

## 2.1 Pareto 图：吞吐量-成功率

文件：

```text
figures/fig_pareto_throughput_success.pdf
```

包含：

```text
OPT4C-K1
OPT4C-K16
cuRobo-K1
cuRobo-K16
```

横轴：

```text
throughput_targets_per_s
```

纵轴：

```text
Strict SR
```

用途：

说明本文方法与 cuRobo 的 trade-off。

---

## 2.2 Target-Block 线程映射示意图

文件：

```text
figures/fig_thread_mapping_redraw.pdf
```

应画清楚：

```text
Grid: N blocks
Block i -> Target i
Lane 0-15 -> Sobol seed 0-15
Lane 16-31 -> inactive or reserved
Shared memory -> candidate buffer
Lane 0 -> final selection
Global memory -> best solution
```

注意：

不要写“经线”。

统一术语：

```text
lane
thread lane
线程通道
```

---

## 2.3 Algorithm Pipeline 图

文件：

```text
figures/fig_algorithm_pipeline_redraw.pdf
```

流程：

```text
Input targets
↓
Load Sobol seeds
↓
Fused FK
↓
Analytical Jacobian
↓
LM update
↓
Limit barrier
↓
Shared-memory candidate selection
↓
Best IK output
```

不要写“无分支消除 divergence”。

改成：

```text
reduced control-flow complexity
```

或者中文：

```text
降低控制流复杂度
```

---

## 2.4 Kernel Time Breakdown 堆叠柱状图

文件：

```text
figures/fig_kernel_time_breakdown.pdf
```

组成：

```text
H2D
kernel
D2H
launch
```

规模：

```text
N=100, 500, 1000
```

---

## 2.5 Seed 数量扫描图

文件：

```text
figures/fig_seed_count_scan.pdf
```

建议两个子图：

```text
(a) K vs Strict SR
(b) K vs Throughput
```

---

## 2.6 Barrier 权重扫描图

文件：

```text
figures/fig_barrier_weight_scan.pdf
```

建议两个子图：

```text
(a) wlimit vs Strict SR
(b) wlimit vs near-limit ratio
```

---

## 2.7 Trajectory 连续性图

文件：

```text
figures/fig_trajectory_delta_q.pdf
```

建议：

```text
boxplot or violin plot
```

比较：

```text
no rerank
smoothness rerank
cuRobo-K16
```

---

## 2.8 Nsight Systems Timeline 图

文件：

```text
figures/fig_nsys_timeline_opt4c.pdf
```

要求：

图中至少标注：

```text
H2D
single IK kernel
D2H
```

---

# 3. 正文必须修正的地方

## 3.1 摘要修改

当前摘要中存在问题：

1. “硬件固化”偏夸张；
2. “共享内存 Bank 冲突为零”过强；
3. 与 cuRobo 默认 K=1 比较容易被认为不公平；
4. “新范式”说法过满。

建议修改摘要核心句：

原倾向：

```text
提出一种硬件固化的 Sobol 多起点并行 IK 加速方法。
```

改为：

```text
提出一种面向固定 6-DOF 批量 IK 的结构感知单核融合 CUDA 实现方法。
```

原倾向：

```text
与默认单种子 cuRobo-Graph 对比，本方法成功率领先。
```

改为：

```text
在默认单种子 cuRobo-Graph 对比中，本方法因 K=16 多起点策略获得更高成功率；在同等 K=16 条件下，本文进一步分析两者在吞吐量、成功率和成功样本精度之间的 Pareto 取舍。
```

原倾向：

```text
Bank 冲突为零。
```

改为：

```text
Nsight Compute 结果显示共享内存访问不是主要瓶颈。
```

---

## 3.2 引言修改

当前问题：

```text
N≤1000 场景下，GPU 调度开销是主要瓶颈。
```

但论文后面又写：

```text
kernel 执行占 99.76%
```

应区分：

1. 通用多阶段 pipeline 的调度开销；
2. 本文单 kernel fusion 后的主要瓶颈。

建议替换为：

```text
对于通用多阶段 GPU pipeline，小批量 IK 中 kernel launch、host-device 同步和全局内存中转会显著影响端到端延迟。本文通过单 kernel fusion 将上述固定调度成本压缩到可忽略水平；在 fusion 之后，主要瓶颈转移为 FP64 小矩阵求解、三角函数和标量寄存器运算的指令延迟。
```

---

## 3.3 “硬件固化”统一改词

建议全文替换：

```text
硬件固化
```

为：

```text
编译期特化
结构感知特化
固定 K 单核融合
```

保留一次“硬件固化”也可以，但要解释：

```text
本文所谓“固化”并非 FPGA/ASIC 层面的硬件固化，而是指将机器人结构参数、K 值和线程映射作为编译期常量编码进 CUDA kernel。
```

---

## 3.4 “经线”统一改为 “lane / 线程通道”

全文替换：

```text
16 条经线
每经线
经线内
```

改为：

```text
16 个 lane
每个线程通道
warp 内
```

中文建议：

```text
lane（线程通道）
```

第一次出现时写：

```text
lane（即 warp 内线程通道）
```

后文统一用 lane。

---

## 3.5 无分支 LM 表述修正

当前正文写：

```text
消除 warp divergence
```

但伪代码仍有：

```text
if rho > 0
if converged break
```

因此必须改。

建议替换为：

```text
本文采用总是接受 trial 的 LM 更新方式，避免了传统 acceptance-rejection 中回滚候选解造成的复杂控制流和额外状态保存。该设计降低了控制流复杂度，但不同 seed 的 rho 判定和收敛迭代次数仍可能导致 warp 内分支分化。因此本文不声称完全消除 warp divergence，而是将其限制在轻量级阻尼更新和收敛判定分支中。
```

算法标题也改：

原：

```text
无分支 LM 迭代
```

改：

```text
低控制流复杂度 LM 迭代
```

---

## 3.6 Bank Conflict 段落重写

当前有明显错误：

```text
32 与 16 互质
```

这是错误的，因为 gcd(32,16)=16。

必须删除整段推导。

建议改为：

```text
共享内存候选缓存仅用于块内 16 个候选解的暂存和最终选择，总容量约 2 KB。由于候选写入和读取只发生在每个 block 内，且访问次数相对 LM 主循环很少，Nsight Compute 采样显示共享内存访问并非主要瓶颈。本文后续不将共享内存 bank conflict 作为主要优化目标，而将优化重点放在降低 FP64 小矩阵求解和三角函数计算开销上。
```

如果有准确 Nsight 数据，再写：

```text
bank conflict rate < 0.1%
```

不要写：

```text
Bank 冲突为零
```

除非有明确计数器证明，并且要说明计数器名称。

---

## 3.7 Kernel Launch 数量论证修正

当前写：

```text
传统方案启动次数 NKW+N
```

这容易被 CUDA 审稿人质疑，因为正常 batched GPU 实现不会对每个 target 单独 launch。

建议改成：

```text
朴素逐目标/逐种子实现的启动次数可能随 N 和 K 增长，但更常见的多阶段 batched GPU pipeline 启动次数主要随优化阶段数和迭代阶段数增长。本文的优势不是相对一个极差的逐目标 baseline，而是将 FK、Jacobian、LM update 和候选选择融合进单个 target-block kernel，减少阶段间全局内存中转和 host-device 同步。
```

表格也修改。

原表：

```text
传统多 Kernel: NKW+N
本文: 1
```

改成：

| 实现方式 | Kernel stage | Global memory round-trip | Host sync |
|---|---:|---:|---:|
| 朴素逐目标实现 | O(NK) | 多次 | 多次 |
| 多阶段 batched pipeline | O(stage × iter) | 多次 | 少量 |
| 本文 OPT4C | 1 core kernel | 最少 | 无中间同步 |

---

## 3.8 解析雅可比精度表述修正

当前写：

```text
差分方案截断误差 1e-6 已接近或超过 Strict 阈值量级，可能将收敛解误判为失败。
```

这个表述不严谨。

5 mm = 5e-3 m，和 1e-6 并不是一个量级。

建议改成：

```text
解析雅可比相较有限差分的优势主要体现在避免差分步长调参、减少 FK 调用次数和提高梯度方向稳定性，而不是直接因为差分截断误差接近 5 mm 阈值。有限差分在奇异附近或尺度不一致时可能产生不稳定梯度，进而影响 LM 收敛。
```

---

## 3.9 “工业抓取 5mm”表述修正

不要泛化说：

```text
5 mm 满足工业抓取要求。
```

改成：

```text
5 mm / 1° 阈值可作为粗定位抓取、采样式规划前端或中等精度任务中的工程可用判据；对于精密装配、插接和接触丰富任务，该阈值仍不足，需要更高精度的局部优化或视觉伺服闭环。
```

---

## 3.10 可推广性段落修正

当前写：

```text
不依赖 UR10，任意 n-DOF 都可推广。
```

这个说法过强。

改成：

```text
本文框架可推广到其他串联机械臂，但需要重新生成结构参数、解析雅可比计算路径、线程映射和寄存器资源配置。对于 7-DOF 冗余机械臂，解空间维度增加且零空间优化需求更强，不能直接认为 UR10 上的成功率和吞吐量可平移到 Franka Panda。
```

如果补了 Franka 小实验，再写：

```text
初步 Franka 实验验证了框架可迁移性，但性能和成功率会随 DOF、关节限位和冗余目标选择策略变化。
```

---

## 3.11 结论修正

当前结论：

```text
确立了成功率优先、延迟确定的批量 IK 加速新范式。
```

建议收敛为：

```text
在固定 6-DOF、N≤1000、中小批量可达目标场景下，本文方法提供了一种成功率优先、执行路径确定的 CUDA IK 实现方案。其优势来自 Sobol 多起点与单 kernel fusion 的协同，而非单个 LM 优化器的局部收敛能力。对于更大批量、更高精度或复杂碰撞约束场景，cuRobo 等通用 GPU 规划框架仍具有重要优势。
```

---

# 4. 论文结构建议

建议将实验章节改为如下结构：

```text
4 实验结果与分析
  4.1 实验协议与公平性设置
  4.2 静态随机可达目标性能
  4.3 与 cuRobo 的系统级与同等 K 对比
  4.4 种子数量扫描与 K=16 选择依据
  4.5 近奇异与近限位鲁棒性实验
  4.6 轨迹连续性实验
  4.7 CPU baseline 对比
  4.8 系统级时间分解与 Nsight Systems 验证
  4.9 微架构分析与优化方向
```

讨论章节：

```text
5 讨论
  5.1 OPT4C 与 cuRobo 的 Pareto 边界
  5.2 Sobol 多起点的收益与代价
  5.3 适用边界
  5.4 局限性与后续工作
```

---

# 5. Codex 执行清单

## 5.1 代码层任务

- [ ] 增加 cuRobo K=16 benchmark 脚本；
- [ ] 增加 near singular target generator；
- [ ] 增加 near limit target generator；
- [ ] 增加 barrier ON/OFF 配置；
- [ ] 增加 trajectory target generator；
- [ ] 增加 smoothness rerank 开关；
- [ ] 增加 CPU baseline runner；
- [ ] 增加 barrier weight scan；
- [ ] 增加 seed count scan；
- [ ] 增加 max_iter scan；
- [ ] 增加 threshold scan；
- [ ] 增加 kernel timing breakdown 输出；
- [ ] 增加 Nsight Systems 命令和结果解析脚本；
- [ ] 所有实验统一输出 CSV；
- [ ] 所有图由脚本自动从 CSV 生成。

## 5.2 绘图任务

- [ ] `fig_pareto_throughput_success.pdf`
- [ ] `fig_thread_mapping_redraw.pdf`
- [ ] `fig_algorithm_pipeline_redraw.pdf`
- [ ] `fig_kernel_time_breakdown.pdf`
- [ ] `fig_seed_count_scan.pdf`
- [ ] `fig_barrier_weight_scan.pdf`
- [ ] `fig_trajectory_delta_q.pdf`
- [ ] `fig_near_singular_sr.pdf`
- [ ] `fig_near_limit_barrier.pdf`
- [ ] `fig_nsys_timeline_opt4c.pdf`

## 5.3 论文正文任务

- [ ] 摘要重写；
- [ ] 引言中调度开销逻辑修正；
- [ ] “硬件固化”术语降级；
- [ ] “经线”改为 lane；
- [ ] “无分支 LM”改为低控制流复杂度 LM；
- [ ] 删除 “消除 warp divergence”；
- [ ] 删除错误 Bank Conflict 推导；
- [ ] 删除或改写 `NKW+N` kernel launch 论证；
- [ ] 修正解析雅可比误差量级说法；
- [ ] 修正 “5 mm 工业抓取” 泛化；
- [ ] 加入 cuRobo K=16 公平对比；
- [ ] 加入近奇异、近限位、轨迹连续性实验；
- [ ] 加入 CPU baseline；
- [ ] 加入 Nsight Systems 图；
- [ ] 结论收敛，不写全面优于 cuRobo。

---

# 6. 最低验收标准

完成后，论文至少应满足：

1. `paper.tex` 可从干净目录一次编译通过；
2. 所有表格数据可从 CSV 追溯；
3. 所有图可由脚本从 CSV 重新生成；
4. cuRobo-K16 公平对比存在；
5. Near Singular、Near Limit、Trajectory 三类鲁棒性实验存在；
6. `Bank conflict`、`warp divergence`、`NKW+N` 三处危险表述已修正；
7. 摘要和结论不再声称全面超过 cuRobo；
8. README 中写清楚实验复现命令。

---

# 7. 推荐最终论文主结论

可以把最终论文收束为：

```text
本文提出了一种面向固定 6-DOF 中小批量 IK 的结构感知单核融合 CUDA 实现。其核心不是增强单个 LM 优化器，而是通过 Sobol 多起点并行、解析雅可比和 target-block kernel fusion，在单次 kernel 内完成候选生成与选择。实验表明，在 N≤1000 的可达目标场景下，该方法能够以确定的执行路径获得约 0.94–0.96 的 Strict 成功率。消融实验说明，成功率主要来自多起点覆盖，而非单种子 LM 本身。与 cuRobo 的对比表明，OPT4C 和 cuRobo 分别代表成功率优先的特化 kernel 路线与高吞吐通用优化路线，二者在吞吐量、成功率和成功样本精度之间形成不同 Pareto 取舍。
```

这是相对安全、可信、审稿人较难反驳的版本。
