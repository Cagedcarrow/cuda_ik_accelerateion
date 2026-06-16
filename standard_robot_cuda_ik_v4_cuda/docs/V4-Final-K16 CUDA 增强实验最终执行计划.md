# V4-Final-K16 CUDA 增强实验最终执行计划

## 0. 总目标

当前基线已经完成：

```text
V4-Final-K16 CUDA Port
correctness pass
static benchmark complete
cuRobo comparison complete
Nsight profiling complete
final_paper_readiness_report complete
```

本轮目标不是重做主线，也不是推翻已有结果，而是在不破坏现有论文闭环的前提下，用剩余 Codex 额度完成三个高价值增强方向：

```text
A. Kernel optimization：提升 CUDA-V4 性能，解释并缓解 Nsight 暴露的瓶颈；
B. Adaptive-K：在保持接近 K16 解质量的前提下降低平均计算量；
C. Full ablation：补齐论文消融实验，证明每个模块的贡献。
```

本轮工作路径仍为：

```bash
/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik_v4_cuda
```

必须遵守：

```text
1. 不破坏已有 V4-Final-K16 baseline。
2. 不覆盖已有 final_paper_readiness_report.md。
3. 不修改 Loose / Medium / Strict 阈值。
4. 不改变已有 success 判定逻辑。
5. 不做碰撞检测。
6. 不做 V5。
7. 不做完整 Motion Generation。
8. 不重新搜索 w_limit。
9. 所有新增结果必须单独写入 opt/adaptive/ablation 对应 CSV 和 Markdown。
10. 若优化版本性能更好但 correctness 不通过，不得进入论文主结果。
```

---

# 1. 当前基线锁定

## 1.1 基线结果

当前已经确认的 baseline：

```text
CUDA-V4-Final-K16:
Analytical Jacobian
+ LM
+ Sobol-K16
+ Limit Barrier(w=0.03, margin=0.087)
+ Smoothness Candidate Reranking
```

关键指标：

```text
N=1000 Strict SR = 0.954
N=1000 pos_p95_all = 4.509 mm
N=1000 near_limit = 0.007
CUDA vs Python speedup = 259.7x
Nsight:
  registers/thread = 184
  achieved occupancy ≈ 13%
  DRAM throughput < 1%
bottleneck:
  FP64 scalar LM
  register pressure
  low occupancy
  one-thread-per-block mapping
```

## 1.2 Baseline 文件保护

新增目录：

```bash
mkdir -p data/results/opt
mkdir -p data/results/adaptive
mkdir -p data/results/ablation
mkdir -p docs/opt
mkdir -p docs/adaptive
mkdir -p docs/ablation
mkdir -p logs/opt
mkdir -p logs/adaptive
mkdir -p logs/ablation
```

禁止覆盖：

```text
docs/final_paper_readiness_report.md
data/results/final_summary.csv
data/results/cuda_v4_static_benchmark.csv
data/results/cuda_v4_curobo_compare.csv
data/results/nsight_summary.csv
```

本轮所有增强结果写入：

```text
data/results/opt/
data/results/adaptive/
data/results/ablation/
docs/opt/
docs/adaptive/
docs/ablation/
logs/opt/
logs/adaptive/
logs/ablation/
```

---

# 2. 总体验收标准

本轮增强实验最终必须生成：

```text
docs/opt/kernel_optimization_report.md
docs/adaptive/adaptive_k_report.md
docs/ablation/full_ablation_report.md
docs/enhancement_final_summary.md
```

总汇总表：

```text
data/results/enhancement_final_summary.csv
```

最终报告必须明确：

```text
1. 哪些优化成功；
2. 哪些优化失败；
3. 哪些结果可以进入论文主文；
4. 哪些结果只能放 appendix；
5. 哪些结果只能作为 future work；
6. 是否需要更新论文主线；
7. 是否仍保持“不能主张全面超过 cuRobo”的边界。
```

---

# 3. Phase A：Kernel Optimization

## 3.1 目标

当前 Nsight 显示：

```text
registers/thread = 184
achieved occupancy ≈ 13%
DRAM throughput < 1%
```

说明当前 CUDA-V4 不是内存带宽瓶颈，而是：

```text
1. FP64 标量 LM 计算；
2. 寄存器压力；
3. occupancy 低；
4. one-thread-per-block 映射不充分；
5. 每个 block 的实际并行度不足。
```

Kernel Optimization 的目标：

```text
在不破坏 V4 算法质量的前提下，提高吞吐，降低寄存器压力，提高 occupancy。
```

---

## 3.2 优化版本命名

保留 baseline：

```text
CUDA-V4-Final-K16-baseline
```

新增优化候选：

```text
CUDA-V4-OPT0-baseline-rebuild
CUDA-V4-OPT1-analytic-limit-gradient
CUDA-V4-OPT2-register-reduced
CUDA-V4-OPT3-soa-candidate-layout
CUDA-V4-OPT4-warp-per-seed-prototype
CUDA-V4-OPT5-best-combined
```

每个版本必须有独立编译宏或 runner 参数：

```bash
--variant baseline
--variant opt1_limit_grad
--variant opt2_reg_reduce
--variant opt3_soa
--variant opt4_warp_seed
--variant opt5_best
```

---

## 3.3 OPT0：Baseline Rebuild

### 目的

确认新实验框架没有破坏原 baseline。

### 执行

```bash
./build/cuda_v4_runner --mode v4_static --variant baseline --N 1000 --K 16 --warmup 10 --repeat 30
```

### 输出

```text
data/results/opt/opt0_baseline_rebuild.csv
```

### 通过标准

必须接近原结果：

```text
Strict SR ≥ 93%
pos_p95_all ≤ 8 mm
near_limit ≤ 4%
no NaN/Inf
```

如果 OPT0 不通过，停止所有优化，先修 baseline runner。

---

## 3.4 OPT1：Analytical / Piecewise Limit Gradient

### 背景

当前 Limit Barrier 第一版可能使用 finite-difference gradient。该方法有利于 correctness，但会增加计算量。Limit Barrier 本身是分段二次函数，适合实现解析梯度。

### 数学定义

Limit loss：

```text
L = Σ_j w * max(0, margin - (q_j - q_min_j))^2
  + Σ_j w * max(0, margin - (q_max_j - q_j))^2
```

对下限项：

```text
d = q_j - q_min_j
if d < margin:
    grad_j += -2 * w * (margin - d)
```

对上限项：

```text
d = q_max_j - q_j
if d < margin:
    grad_j +=  2 * w * (margin - d)
```

固定：

```text
w_limit = 0.03
margin = 0.087
```

### 实现要求

新增函数：

```cpp
__device__ void limit_barrier_loss_grad_analytic_v4(
    const double q[6],
    double* loss,
    double grad[6]
);
```

runner 支持：

```bash
--limit-gradient finite_diff
--limit-gradient analytic
```

### correctness 验收

对比 finite-difference baseline：

```text
limit_loss diff < 1e-8
grad max abs diff < 1e-5
N=100 Strict SR diff ≤ 1 pp
N=100 pos_p95_all diff ≤ 1 mm
```

### performance 验收

在 N=1000 和 N=5000 上比较：

```text
gpu_stream_ms
e2e_ms
throughput
registers/thread
occupancy
```

### 输出

```text
data/results/opt/opt1_limit_gradient_correctness.csv
data/results/opt/opt1_limit_gradient_benchmark.csv
docs/opt/opt1_limit_gradient_report.md
```

### 成功标准

```text
correctness pass
N=1000 speedup vs baseline ≥ 1.05x
Strict SR drop ≤ 1 pp
pos_p95_all ≤ 8 mm
```

如果速度没有提升但 correctness 通过，可作为工程说明，不进入主结果。

---

## 3.5 OPT2：Register Reduction

### 目标

当前 `184 registers/thread` 偏高。目标是尝试降低到：

```text
primary target: <160 regs/thread
strong target: <128 regs/thread
excellent target: <96 regs/thread
```

### 允许优化手段

Codex 需要逐项尝试，不要一次性乱改。

#### OPT2-A：缩短变量 live range

要求：

```text
1. 减少长期保存在局部数组里的 T/J/H/e/g/q 临时变量；
2. 能局部计算就局部计算；
3. 每轮 LM 结束后不再需要的变量尽快离开作用域；
4. 拆分函数减少寄存器常驻。
```

#### OPT2-B：J/H 存储策略

尝试：

```text
1. J 不完整保存，只在构建 H=J^T J 和 g=J^T e 时流式计算；
2. 或 J 保存到 shared memory，减少每线程寄存器；
3. H 使用 6x6 packed symmetric storage；
4. g/dq 使用长度 6，不再 padded 到 8，除非 padding 明显更快。
```

#### OPT2-C：编译参数实验

尝试但必须谨慎：

```text
-Xptxas -v
--maxrregcount=160
--maxrregcount=128
```

必须记录：

```text
registers/thread
spill stores
spill loads
local memory
performance
```

如果 `maxrregcount` 导致 spill 明显增加或速度变慢，立即放弃。

### 输出

```text
data/results/opt/opt2_register_reduction.csv
docs/opt/opt2_register_reduction_report.md
logs/opt/ptxas_registers.log
```

CSV 字段：

```text
variant
maxrregcount
registers_per_thread
spill_stores
spill_loads
local_memory_bytes
achieved_occupancy
gpu_stream_ms_N1000
gpu_stream_ms_N5000
strict_sr_N1000
pos_p95_all_N1000
near_limit_N1000
pass_quality
pass_perf
notes
```

### 成功标准

质量：

```text
Strict SR drop ≤ 1 pp
pos_p95_all ≤ 8 mm
near_limit ≤ 4%
```

性能：

```text
N=1000 speedup vs baseline ≥ 1.10x
or
N=5000 speedup vs baseline ≥ 1.10x
or
registers/thread 降低 ≥ 15% 且性能不下降
```

如果寄存器下降但性能下降，写入 report，不进入 best combined。

---

## 3.6 OPT3：Candidate Result SoA Layout

### 背景

如果 CandidateResult 是 AoS：

```cpp
struct CandidateResult {
    double q[6];
    double pos_err;
    double rot_err;
    ...
};
```

selection kernel 访问时可能读入不必要字段。改为 SoA 可能提升 selection 阶段和 D2H 后处理效率。

### 实现

新增可选 SoA buffers：

```text
q_candidates[N*K*6]
pos_err[N*K]
rot_err[N*K]
pose_cost[N*K]
limit_score[N*K]
near_limit[N*K]
success_rank[N*K]
iters[N*K]
```

runner 参数：

```bash
--candidate-layout aos
--candidate-layout soa
```

### 测试

N：

```text
100
1000
5000
```

比较：

```text
candidate generation time
selection time
total gpu_stream_ms
D2H time
```

### 输出

```text
data/results/opt/opt3_candidate_layout.csv
docs/opt/opt3_candidate_layout_report.md
```

### 成功标准

```text
selection kernel speedup ≥ 1.20x
or
total gpu_stream_ms speedup ≥ 1.05x
quality unchanged
```

如果 selection 占比很小，总体不提升，也要记录，作为“selection 不是瓶颈”的证据。

---

## 3.7 OPT4：Warp-per-seed Prototype

### 目标

当前 one-thread-per-block 映射导致 GPU 并行利用率低。尝试 warp-per-seed 映射：

```text
1 warp = 1 target-seed pair
32 lanes 协同完成 FK/Jacobian/H/g/solve
```

注意：这是高风险实验，不要求必然成功。

### 实现原则

新增独立 kernel，不要替换 baseline：

```cpp
ik_lm_multiseed_kernel_v4_warpseed
```

grid 设计建议：

```text
blockDim = 128 or 256
每个 block 包含 4 或 8 个 warps
每个 warp 处理一个 (target_id, seed_id)
global warp id -> candidate id = target_id * K + seed_id
```

映射：

```text
warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32
lane = threadIdx.x % 32
candidate_id = warp_global
target_id = candidate_id / K
seed_id = candidate_id % K
```

### 分工建议

为了降低复杂度，第一版可以只并行部分：

```text
lane 0-5: 负责 6 个关节相关计算；
lane 0-5: 负责 H/g 行列构建；
lane 0: 负责 Cholesky/LDLT solve；
全 warp broadcast q/dq/loss。
```

第一版目标不是极致性能，而是验证：

```text
warp-per-seed 是否能降低 register pressure 或提高 occupancy。
```

### correctness

先跑：

```text
N=10 K=16
N=100 K=16
```

通过标准：

```text
Strict SR diff ≤ 2 pp
pos_p95_all diff ≤ 2 mm
no NaN/Inf
```

### performance

跑：

```text
N=100
N=1000
N=5000
```

比较：

```text
gpu_stream_ms
registers/thread
occupancy
warp execution efficiency
branch divergence
```

### 输出

```text
data/results/opt/opt4_warp_per_seed_correctness.csv
data/results/opt/opt4_warp_per_seed_benchmark.csv
docs/opt/opt4_warp_per_seed_report.md
```

### 成功标准

强成功：

```text
quality pass
N=1000 speedup ≥ 1.25x
occupancy 明显提升
registers/thread 明显下降
```

弱成功：

```text
quality pass
性能接近 baseline
Nsight 解释出下一步优化方向
```

失败也有价值：

```text
若 warp divergence 或协同开销导致性能下降，报告中说明 one-thread mapping 虽低占用但控制流简单，warp-per-seed 第一版不进入主结果。
```

---

## 3.8 OPT5：Best Combined Variant

### 目的

把 OPT1-OPT4 中真正有效的优化组合成一个最终候选。

### 组合规则

只能组合已经单独通过 correctness 的优化。

例如：

```text
OPT5 = analytic limit gradient + register reduced + SoA
```

不要把未通过 correctness 的 warp-per-seed 合进去。

### 测试

跑完整：

```text
N=100/500/1000/5000
K=16
warmup=10
repeat=30
```

再跑 cuRobo 对比可选：

```text
N=100/500/1000/5000
```

### 输出

```text
data/results/opt/opt5_best_combined_static_benchmark.csv
data/results/opt/opt5_best_combined_vs_baseline.csv
docs/opt/opt5_best_combined_report.md
```

### 成功标准

进入论文主表的条件：

```text
1. correctness pass；
2. N=1000 Strict SR ≥ 93%；
3. N=1000 pos_p95_all ≤ 8 mm；
4. near_limit ≤ 4%；
5. N=1000 speedup vs baseline ≥ 1.10x；
6. N=5000 speedup vs baseline ≥ 1.10x；
7. Nsight 指标至少一项明显改善：
   - registers/thread 降低；
   - occupancy 提升；
   - spill 降低；
   - branch divergence 降低；
```

如果没有达到 1.10x，不替换 baseline，只作为优化探索写入 discussion。

---

# 4. Phase B：Adaptive-K

## 4.1 目标

K16 全量多种子质量高，但每个目标固定计算 16 个 seed。Adaptive-K 的思想是：

```text
easy target 少算；
hard target 多算；
用接近 K16 的成功率换更低平均计算量。
```

该实验不得替代 V4-Final-K16 主结果，除非质量和性能都明显更好。

---

## 4.2 Adaptive-K 版本

实现 3 个版本：

```text
AK-8-only
AK-8+8-rescue
AK-4+4+8-rescue
```

### AK-8-only

```text
只跑前 8 个 Sobol seeds
选择 best
```

目的：

```text
质量下限和速度上限。
```

### AK-8+8-rescue

流程：

```text
Stage 1: 跑 K=8
if target strict success:
    停止
else:
    Stage 2: 对失败 target 跑剩余 K=8
最后在最多 16 个候选中选择 best
```

### AK-4+4+8-rescue

流程：

```text
Stage 1: K=4
if strict success stop
else Stage 2: K=4
if strict success stop
else Stage 3: K=8
```

### 可选：AK-threshold rescue

严格规则：

```text
如果 Stage 1 没达到 Strict，则 rescue。
```

可选规则：

```text
如果 Stage 1 pos_err < 6mm 且 rot_err < 1.2deg，可以不 rescue。
```

注意：可选规则容易影响质量，必须单独报告，不进入主线。

---

## 4.3 CUDA 实现方式

第一版可以采用多 kernel：

```text
kernel_stage1_K8
select_stage1
compact_failed_targets
kernel_stage2_K8
final_select
```

如果 compact 复杂，可以先使用简单 mask：

```text
Stage 2 kernel 仍遍历 N，但只对 failed target 执行，其他 target return。
```

第一版优先 correctness，不强求完美调度。

---

## 4.4 评价指标

输出：

```text
data/results/adaptive/adaptive_k_benchmark.csv
```

字段：

```text
method
N
K_max
avg_seeds_evaluated
stage1_success_rate
stage2_rescue_rate
stage3_rescue_rate
strict_sr
medium_sr
loose_sr
pos_p95_all_mm
pos_p95_suc_mm
rot_p95_all_deg
near_limit_ratio
gpu_stream_ms_mean
e2e_ms_mean
raw_throughput
effective_seed_reduction
speedup_vs_K16
quality_drop_vs_K16
pass_quality
pass_perf
```

---

## 4.5 测试矩阵

N：

```text
100
500
1000
5000
```

Methods：

```text
K16 baseline
K8 only
AK-8+8
AK-4+4+8
```

repeat：

```text
warmup=10
repeat=30
```

---

## 4.6 验收标准

### 主质量标准

Adaptive-K 若要进入论文主文，必须满足：

```text
N=1000 Strict SR ≥ K16 baseline - 1.0 pp
N=1000 pos_p95_all ≤ K16 baseline + 2 mm
near_limit ≤ 4%
monotonic pass
no NaN/Inf
```

### 性能标准

至少满足：

```text
avg_seeds_evaluated ≤ 12
N=1000 speedup vs K16 ≥ 1.20x
N=5000 speedup vs K16 ≥ 1.20x
```

### 强成功

```text
Strict SR drop ≤ 0.5 pp
avg seeds ≤ 10
speedup ≥ 1.30x
```

### 失败判定

如果：

```text
Strict SR drop > 2 pp
or pos_p95_all > 8 mm
or speedup < 1.10x
```

则 Adaptive-K 不进入主文，只作为探索性实验或 future work。

---

## 4.7 报告要求

生成：

```text
docs/adaptive/adaptive_k_report.md
```

报告必须回答：

```text
1. K8-only 损失多少成功率？
2. AK-8+8 是否接近 K16？
3. AK-4+4+8 是否更省但质量是否掉太多？
4. rescue target 占比是多少？
5. 平均 seed 数是多少？
6. speedup 是否来自少算 seed，而不是计时口径变化？
7. 是否值得进入论文主文？
```

结论模板：

```text
若成功：
Adaptive-K 在保持接近 K16 解质量的同时减少平均 seed 计算量，可作为批量 IK 的计算自适应策略。

若失败：
Adaptive-K 降低了计算量，但严格成功率或误差质量下降明显，因此本文仍采用固定 K16 作为主配置。
```

---

# 5. Phase C：Full Ablation

## 5.1 目标

补齐论文最重要的消融实验，证明：

```text
1. 解析 Jacobian 的作用；
2. LM 的作用；
3. Sobol-K16 的作用；
4. Limit Barrier 的作用；
5. Smoothness Reranking 的作用；
6. CUDA Port 的工程作用；
7. cuRobo 对比边界。
```

---

## 5.2 Ablation 组别

至少跑以下组：

```text
A0: Python V1 / CUDA V1 DLS baseline, if available
A1: V2 Analytical Jacobian + DLS
A2: V3 Analytical Jacobian + LM + Random-K16
A3: V3 Analytical Jacobian + LM + Sobol-K8
A4: V3 Analytical Jacobian + LM + Sobol-K16
A5: V4-Final-K16 without Limit Barrier
A6: V4-Final-K16 with Limit Barrier(w=0.03)
A7: V4-Final-K16 with Limit + Smoothness Rerank
A8: CUDA-V4-Final-K16
A9: cuRobo-Graph
```

如果某些历史版本无法直接在 CUDA runner 中复现，可以从已有 CSV 读取，但报告必须标注：

```text
source = historical_csv
source = rerun
source = python_reference
source = cuda_current
```

不要混淆来源。

---

## 5.3 Static Ablation 指标

N：

```text
N=1000
```

可选：

```text
N=100/500/5000
```

指标：

```text
method
source
N
K
strict_sr
medium_sr
loose_sr
pos_p50_all_mm
pos_p95_all_mm
pos_p99_all_mm
pos_max_all_mm
pos_p95_suc_mm
rot_p95_all_deg
near_limit_ratio
joint_violation_count
gpu_stream_ms
e2e_ms
raw_throughput
valid_throughput_strict
notes
```

输出：

```text
data/results/ablation/static_ablation_N1000.csv
```

---

## 5.4 Module Contribution Table

生成：

```text
data/results/ablation/module_contribution_table.csv
```

字段：

```text
transition
from_method
to_method
strict_sr_delta_pp
pos_p95_all_delta_mm
near_limit_delta_pp
speedup_x
main_interpretation
```

建议 transitions：

```text
V1 -> V2: 数值 Jacobian 到解析 Jacobian
V2 -> V3 Random-K16: 单 seed/传统 seed 到多 seed
Random-K16 -> Sobol-K16: seed coverage 改善
V3 -> V4 Limit: near-limit 控制
Independent -> Smoothness Rerank: trajectory continuity
Python V4 -> CUDA V4: 工程加速
CUDA V4 -> cuRobo: 性能边界
```

---

## 5.5 Trajectory Ablation

轨迹：

```text
line_50
arc_50
local_random_50
```

方法：

```text
independent selection
smoothness rerank
```

指标：

```text
trajectory_type
method
strict_sr
pos_p95_all_mm
mean_delta_q_rad
p95_delta_q_rad
max_delta_q_rad
jump_count_linf_0p5
jerk_cost
```

输出：

```text
data/results/ablation/trajectory_ablation.csv
```

---

## 5.6 cuRobo Boundary Table

生成：

```text
data/results/ablation/curobo_boundary_table.csv
```

字段：

```text
N
cuda_v4_throughput
curobo_graph_throughput
cuda_v4_strict_sr
curobo_strict_sr
cuda_v4_pos_p95
curobo_pos_p95
winner_throughput
winner_quality
interpretation
```

报告中必须写：

```text
cuRobo-Graph 吞吐更强；
CUDA-V4 解质量更强或更稳；
本文不主张全面超过 cuRobo。
```

---

## 5.7 绘图输出

生成以下图的数据和图片：

```text
figures/fig_v1_to_v4_pipeline.png
figures/fig_static_ablation_sr_pos.png
figures/fig_limit_barrier_effect.png
figures/fig_smoothness_rerank_effect.png
figures/fig_curobo_boundary.png
figures/fig_nsight_bottleneck.png
```

如果不画图，也必须生成画图数据 CSV。

---

## 5.8 Ablation 验收标准

Ablation 成功不是要求所有版本都更好，而是要求能回答论文问题：

```text
1. 哪个模块提升 Strict SR？
2. 哪个模块降低 pos_p95_all？
3. 哪个模块降低 near_limit？
4. 哪个模块改善 trajectory smoothness？
5. CUDA Port 带来多少加速？
6. cuRobo 的优势和本文优势分别在哪里？
```

若报告能清楚回答以上问题，则 Full Ablation 通过。

---

# 6. 总最终输出

本轮结束必须生成：

```text
docs/enhancement_final_summary.md
data/results/enhancement_final_summary.csv
```

报告结构：

```markdown
# V4 CUDA Enhancement Final Summary

## 1. Baseline Recap

## 2. Kernel Optimization
### 2.1 OPT1 Analytical Limit Gradient
### 2.2 OPT2 Register Reduction
### 2.3 OPT3 Candidate Layout
### 2.4 OPT4 Warp-per-Seed
### 2.5 OPT5 Best Combined

## 3. Adaptive-K
### 3.1 K8-only
### 3.2 AK-8+8
### 3.3 AK-4+4+8
### 3.4 Whether Adaptive-K Enters Main Paper

## 4. Full Ablation
### 4.1 Static Ablation
### 4.2 Module Contribution
### 4.3 Trajectory Ablation
### 4.4 cuRobo Boundary

## 5. Updated Paper Claims
### 5.1 Claims Supported
### 5.2 Claims Not Supported
### 5.3 Main-text Results
### 5.4 Appendix Results
### 5.5 Future Work

## 6. Final Decision
```

---

# 7. Claims Decision Rules

## 7.1 Kernel Optimization 进入主文的条件

```text
OPT5 quality pass
and speedup vs baseline ≥ 1.10x
and Nsight 至少一个瓶颈指标改善
```

否则：

```text
放 Discussion / Future Work。
```

## 7.2 Adaptive-K 进入主文的条件

```text
Strict SR drop ≤ 1 pp
pos_p95_all ≤ baseline + 2 mm
avg seeds ≤ 12
speedup ≥ 1.20x
```

否则：

```text
放 Appendix 或 Future Work。
```

## 7.3 Ablation 必须进入主文

Full ablation 是论文说服力核心。只要数据完整，必须进入主文。

---

# 8. 执行优先级

如果 Codex 额度不足，按这个顺序执行：

```text
Priority 1:
  OPT1 Analytical Limit Gradient
  OPT2 Register Reduction
  Full Ablation

Priority 2:
  Adaptive-K AK-8+8
  cuRobo Boundary Table
  Nsight OPT comparison

Priority 3:
  OPT3 SoA Candidate Layout
  OPT4 Warp-per-Seed Prototype
  AK-4+4+8
  plotting
```

最小可接受闭环：

```text
1. OPT1 completed
2. OPT2 completed
3. AK-8+8 completed
4. Full static ablation completed
5. enhancement_final_summary.md generated
```

---

# 9. 严格禁止事项

```text
1. 禁止改 V4-Final-K16 baseline 定义。
2. 禁止改 success threshold。
3. 禁止为了性能跳过 correctness。
4. 禁止用失败优化覆盖原 final reports。
5. 禁止把未通过 correctness 的优化写进主结果。
6. 禁止写“全面超过 cuRobo”。
7. 禁止因为 Adaptive-K 成功就删除 K16 baseline。
8. 禁止把 CPU trajectory rerank 伪装成 GPU kernel 性能。
```

---

# 10. Codex 执行提示词

请严格执行本增强实验计划。

目标是在已经完成 V4-Final-K16 CUDA Port 验收的基础上，继续完成三类增强实验：

```text
A. Kernel optimization；
B. Adaptive-K；
C. Full ablation。
```

本轮实验不得破坏既有 V4-Final-K16 baseline，不得覆盖已有 final_paper_readiness_report，不得修改阈值和 success 判定，不得做碰撞、V5 或 Motion Generation。

执行顺序：

```text
1. 锁定 baseline；
2. 完成 kernel optimization；
3. 完成 Adaptive-K；
4. 完成 full ablation；
5. 生成 enhancement_final_summary.md；
6. 明确哪些结果进入论文主文、哪些进入 appendix、哪些只能作为 future work。
```

所有实验必须输出 CSV 和 Markdown。若任一优化 correctness 不通过，停止该优化分支，不得进入主结果。

# END
