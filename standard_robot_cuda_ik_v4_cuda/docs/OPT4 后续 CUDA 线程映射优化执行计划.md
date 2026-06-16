# OPT4 后续 CUDA 线程映射优化执行计划

## 0. 目标

当前 `V4-Final-K16 CUDA Port` 已经通过主验收，`Adaptive-K` 已经成功，`OPT5` 有约 6% 提升但未达到替换主表的 1.10x 门槛，原始 `OPT4 warp-per-seed` 已记录为高风险失败/未来工作。

本轮继续使用 Codex 的目标不是推翻主线，而是：

```text
把 OPT4 从“失败一句话”扩展成完整的 CUDA 线程映射边界实验。
```

具体目标：

```text
1. 解释原始 warp-per-seed 为什么失败；
2. 尝试更合理的 warp/block 映射；
3. 判断是否存在比当前 baseline 更好的 target-seed 并行结构；
4. 若成功，作为 CUDA 优化增强结果；
5. 若失败，形成论文 Discussion / Future Work 的硬证据。
```

本轮实验路径：

```bash
/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik_v4_cuda
```

---

# 1. 严格边界

## 1.1 禁止事项

```text
1. 禁止修改 V4-Final-K16 数学定义。
2. 禁止修改 Loose / Medium / Strict 阈值。
3. 禁止修改 success 判定逻辑。
4. 禁止重新搜索 w_limit。
5. 禁止重新搜索 K。
6. 禁止用未通过 correctness 的优化替换 baseline。
7. 禁止覆盖已有 final_paper_readiness_report.md。
8. 禁止覆盖已有 baseline benchmark CSV。
9. 禁止写“全面超过 cuRobo”。
```

## 1.2 允许事项

```text
1. 新增独立 kernel variant；
2. 新增独立 runner 参数；
3. 新增独立 benchmark CSV；
4. 新增独立 Nsight CSV；
5. 新增 postmortem 报告；
6. 若优化失败，保留为 negative result。
```

## 1.3 所有新增结果写入

```bash
mkdir -p docs/opt/opt4_followup
mkdir -p data/results/opt/opt4_followup
mkdir -p logs/opt/opt4_followup
```

---

# 2. 当前 baseline 锁定

当前主 baseline：

```text
CUDA-V4-Final-K16 baseline
```

已知关键指标：

```text
N=1000 Strict SR ≈ 0.954
pos_p95_all ≈ 4.5 mm
near_limit 很低
Python-to-CUDA speedup ≈ 259.7x
Nsight:
  registers/thread ≈ 184
  achieved occupancy ≈ 13%
  DRAM throughput < 1%
主要瓶颈:
  FP64 scalar LM
  register pressure
  low occupancy
  one-thread-per-block / low useful lane utilization
```

本轮优化判断不能只看速度，必须同时看：

```text
correctness
Strict SR
pos_p95_all
near_limit
registers/thread
occupancy
spill
branch divergence
gpu_stream_ms
```

---

# 3. OPT4 总体路线

不要继续原始 `warp-per-seed` 方向。

原始方向：

```text
1 warp = 1 target-seed candidate
```

问题：

```text
1. 单个 6x6 IK 候选太小；
2. LM 迭代强串行；
3. 6x6 solve 难以被 32 lanes 高效切分；
4. 可能很多 lane 空闲；
5. shuffle/sync 协作成本高；
6. 寄存器压力未必降低；
7. branch / lambda update / convergence 判断会增加控制复杂度。
```

本轮改为三个更合理的方向：

```text
OPT4-0: 原始 warp-per-seed postmortem
OPT4B: warp-per-target / lane-per-seed
OPT4C: block-per-target / thread-per-seed
OPT4D: fused candidate generation + selection
```

优先级：

```text
Priority 1: OPT4-0 postmortem
Priority 2: OPT4C block-per-target/thread-per-seed
Priority 3: OPT4B warp-per-target/lane-per-seed
Priority 4: OPT4D fused selection
```

---

# 4. OPT4-0：warp-per-seed Postmortem

## 4.1 目的

把原始 `warp-per-seed` 的失败从“结论”变成“证据”。

必须回答：

```text
1. 它是否通过 correctness？
2. 它是否降低 registers/thread？
3. 它是否提高 achieved occupancy？
4. 它是否增加 branch divergence？
5. 它是否引入 local memory spill？
6. 它的 gpu_stream_ms 是否快于 baseline？
7. 它为什么不进入主结果？
```

## 4.2 输入

读取已有结果：

```text
data/results/opt/*warp*
docs/opt/*warp*
logs/opt/*warp*
```

如果已有 OPT4 数据不完整，则补跑：

```bash
./build/cuda_v4_runner --mode v4_static --variant opt4_warp_per_seed --N 100 --K 16 --warmup 5 --repeat 10
./build/cuda_v4_runner --mode v4_static --variant opt4_warp_per_seed --N 1000 --K 16 --warmup 5 --repeat 10
```

如果该 variant 当前无法编译或无法 correctness pass，也要记录。

## 4.3 Nsight

至少对 N=1000 跑一次：

```bash
ncu --set full \
  --kernel-name regex:.*ik.*warp.* \
  --target-processes all \
  --csv \
  ./build/cuda_v4_runner --mode v4_static --variant opt4_warp_per_seed --N 1000 --K 16 --warmup 2 --repeat 3 \
  > logs/opt/opt4_followup/opt4_warp_per_seed_ncu.csv
```

如果 ncu 太慢，可用较小 repeat。

## 4.4 输出

```text
docs/opt/opt4_followup/opt4_warp_per_seed_postmortem.md
data/results/opt/opt4_followup/opt4_warp_per_seed_postmortem.csv
logs/opt/opt4_followup/opt4_warp_per_seed_ncu.csv
```

## 4.5 CSV 字段

```text
variant
N
correctness_pass
strict_sr
pos_p95_all_mm
near_limit
gpu_stream_ms
speedup_vs_baseline
registers_per_thread
achieved_occupancy
sm_utilization
dram_throughput
local_memory_spill
branch_divergence
warp_execution_efficiency
main_failure_reason
paper_usage
```

## 4.6 报告结论模板

如果失败：

```text
原始 warp-per-seed 未进入主结果。原因是单个 target-seed IK 候选包含强串行 LM 迭代和 6x6 小矩阵求解，简单地将一个候选拆给一个 warp 后，协作开销和控制流复杂度抵消了潜在并行收益。
```

如果 correctness 通过但速度不快：

```text
warp-per-seed correctness 通过，但未获得稳定速度优势。该结果说明当前瓶颈不能仅通过粗粒度 warp 协作解决。
```

---

# 5. OPT4C：block-per-target / thread-per-seed

## 5.1 这是本轮最推荐尝试的方向

相比原始 warp-per-seed，`block-per-target/thread-per-seed` 更符合你的 K16 多种子结构。

设计：

```text
1 block = 1 target
thread 0~15 = 16 seeds
每个 thread 独立跑一个 seed 的 LM
block 内完成 best selection
```

映射：

```cpp
int target_id = blockIdx.x;
int seed_id = threadIdx.x;  // 0..15
```

block size：

```text
建议先用 32 threads/block
thread 0~15 有效
thread 16~31 可用于 reduction 或空闲
```

后续可试：

```text
64 threads/block
128 threads/block
```

但第一版先用 32。

## 5.2 为什么它比 warp-per-seed 更合理

```text
1. 不强行拆单个 6x6 LM；
2. 每个 seed 仍由单线程完整求解，保留 baseline 数学路径；
3. 1 target 的 K16 seeds 在一个 block 内完成；
4. 可以在 shared memory 内直接选 best；
5. 可以减少 candidate global write；
6. 可以减少单独 selection kernel；
7. 对应你的问题天然结构：target -> K seeds -> best seed。
```

## 5.3 实现文件

建议不要改 baseline kernel，新增：

```text
src/cuda/cuda_v4_opt4c_block_target.cu
```

或者在现有：

```text
src/cuda/cuda_v4_runner.cu
```

中新增独立 variant：

```text
--variant opt4c_block_target
```

新增 kernel：

```cpp
__global__ void ik_lm_multiseed_kernel_v4_block_target(
    const TargetPose* targets,
    const double* seed_bank,
    BestResultV4* best_results,
    int N,
    int K,
    SolverConfigV4 cfg
);
```

## 5.4 Shared memory 设计

每个 block 内需要保存 16 个 seed 的候选摘要。

不要把完整大结构全塞 shared memory，优先保存选择所需字段：

```cpp
__shared__ double s_q[16][6];
__shared__ double s_pos_err[16];
__shared__ double s_rot_err[16];
__shared__ double s_pose_cost[16];
__shared__ int    s_success_rank[16];
__shared__ int    s_near_limit[16];
__shared__ int    s_iters[16];
```

候选解 q 只有 16×6 doubles：

```text
16 * 6 * 8 = 768 bytes
```

总体 shared memory 很小。

## 5.5 每个 thread 内部逻辑

```cpp
if (seed_id < K) {
    q = seed_bank[target_id, seed_id]
    run_lm_v4_same_as_baseline()
    write candidate summary to shared memory
}
__syncthreads()

if (threadIdx.x == 0) {
    select best seed using:
    success_rank -> near_limit -> pose_cost
    write BestResultV4[target_id]
}
```

注意：

```text
LM 数学逻辑必须复现 baseline：
q_trial 总是接受；
lambda 仅按 loss_new < loss_old 调整；
max_iter = 60；
dq clamp = 0.35；
w_limit = 0.03；
margin = 0.087。
```

## 5.6 正确性测试

先跑：

```bash
./build/cuda_v4_runner --mode v4_check --variant opt4c_block_target --N 100 --K 16
```

验收：

```text
Strict SR diff vs baseline <= 2 pp
pos_p95_all diff <= 2 mm
near_limit diff <= 2 pp
no NaN/Inf
```

失败时必须输出：

```text
data/results/opt/opt4_followup/opt4c_failure_cases.csv
```

字段：

```text
target_id
baseline_best_seed
opt4c_best_seed
baseline_pos_err
opt4c_pos_err
baseline_rot_err
opt4c_rot_err
baseline_success_rank
opt4c_success_rank
baseline_near_limit
opt4c_near_limit
reason
```

## 5.7 性能测试

通过 N=100 后跑：

```bash
./build/cuda_v4_runner --mode v4_static --variant opt4c_block_target --N 100 --K 16 --warmup 10 --repeat 30
./build/cuda_v4_runner --mode v4_static --variant opt4c_block_target --N 1000 --K 16 --warmup 10 --repeat 30
./build/cuda_v4_runner --mode v4_static --variant opt4c_block_target --N 5000 --K 16 --warmup 10 --repeat 30
```

## 5.8 Nsight

```bash
ncu --set full \
  --kernel-name regex:.*block_target.* \
  --target-processes all \
  --csv \
  ./build/cuda_v4_runner --mode v4_static --variant opt4c_block_target --N 1000 --K 16 --warmup 2 --repeat 3 \
  > logs/opt/opt4_followup/opt4c_block_target_ncu.csv
```

## 5.9 输出

```text
data/results/opt/opt4_followup/opt4c_block_target_correctness.csv
data/results/opt/opt4_followup/opt4c_block_target_benchmark.csv
data/results/opt/opt4_followup/opt4c_block_target_nsight.csv
docs/opt/opt4_followup/opt4c_block_target_report.md
```

## 5.10 成功标准

进入主优化结果的条件：

```text
correctness pass
N=1000 Strict SR >= 93%
N=1000 pos_p95_all <= 8 mm
near_limit <= 4%
N=1000 speedup vs baseline >= 1.10x
N=5000 speedup vs baseline >= 1.10x
```

弱成功：

```text
correctness pass
speedup between 1.03x and 1.10x
or selection kernel time clearly reduced
```

弱成功只能放 Appendix / Discussion。

失败：

```text
correctness fail
or speedup < 1.03x
or quality drop > threshold
```

失败则写成 mapping boundary，不进入主结果。

---

# 6. OPT4B：warp-per-target / lane-per-seed

## 6.1 设计

```text
1 warp = 1 target
lane 0~15 = 16 seeds
lane 16~31 空闲或参与 reduction
```

映射：

```cpp
int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
int lane = threadIdx.x % 32;

int target_id = global_warp_id;
int seed_id = lane;
```

每个 lane 处理一个 seed：

```cpp
if (lane < K) {
    run_lm_v4_for_seed(target_id, seed_id)
}
```

然后 warp 内 reduction 选择 best seed。

## 6.2 为什么它值得试

```text
1. 一个 warp 正好覆盖 K16；
2. 不需要一个 block 只服务一个 target；
3. 可以让一个 block 包含多个 targets，例如 4 warps/block；
4. 比 block-per-target 有更高 block 内有效线程比例；
5. 可以融合 candidate generation 和 selection。
```

## 6.3 风险

```text
1. 不同 seed 迭代次数不同，会造成 warp divergence；
2. lane 内运行完整 LM，寄存器压力仍高；
3. warp reduction 对复杂 key 需要小心实现；
4. 如果 q/H/J 全是 lane-private，register 不一定下降。
```

因此它优先级低于 OPT4C。

## 6.4 实现文件

新增 variant：

```text
--variant opt4b_warp_target
```

新增 kernel：

```cpp
__global__ void ik_lm_multiseed_kernel_v4_warp_target(
    const TargetPose* targets,
    const double* seed_bank,
    BestResultV4* best_results,
    int N,
    int K,
    SolverConfigV4 cfg
);
```

建议 blockDim：

```text
128 threads/block = 4 targets/block
256 threads/block = 8 targets/block
```

先实现：

```text
128 threads/block
```

## 6.5 Warp-level best selection

每个 lane 有 key：

```text
success_rank
near_limit
pose_cost
seed_id
```

需要实现 warp reduction：

```cpp
for (int offset = 16; offset > 0; offset /= 2) {
    compare with __shfl_down_sync(...)
}
```

比较逻辑必须是：

```text
success_rank smaller better
near_limit smaller better
pose_cost smaller better
seed_id smaller only for tie-break
```

最终 lane 0 写出 `BestResultV4[target_id]`。

## 6.6 correctness

```bash
./build/cuda_v4_runner --mode v4_check --variant opt4b_warp_target --N 100 --K 16
```

标准：

```text
Strict SR diff <= 2 pp
pos_p95_all diff <= 2 mm
near_limit diff <= 2 pp
no NaN/Inf
```

## 6.7 benchmark

```bash
./build/cuda_v4_runner --mode v4_static --variant opt4b_warp_target --N 100 --K 16 --warmup 10 --repeat 30
./build/cuda_v4_runner --mode v4_static --variant opt4b_warp_target --N 1000 --K 16 --warmup 10 --repeat 30
./build/cuda_v4_runner --mode v4_static --variant opt4b_warp_target --N 5000 --K 16 --warmup 10 --repeat 30
```

## 6.8 Nsight

```bash
ncu --set full \
  --kernel-name regex:.*warp_target.* \
  --target-processes all \
  --csv \
  ./build/cuda_v4_runner --mode v4_static --variant opt4b_warp_target --N 1000 --K 16 --warmup 2 --repeat 3 \
  > logs/opt/opt4_followup/opt4b_warp_target_ncu.csv
```

## 6.9 输出

```text
data/results/opt/opt4_followup/opt4b_warp_target_correctness.csv
data/results/opt/opt4_followup/opt4b_warp_target_benchmark.csv
data/results/opt/opt4_followup/opt4b_warp_target_nsight.csv
docs/opt/opt4_followup/opt4b_warp_target_report.md
```

## 6.10 成功标准

强成功：

```text
correctness pass
N=1000 speedup vs baseline >= 1.15x
N=5000 speedup vs baseline >= 1.15x
```

弱成功：

```text
correctness pass
speedup 1.05x~1.15x
or Nsight 显示 occupancy / launch / selection 开销改善
```

失败：

```text
divergence 明显增加
registers/thread 不降
speedup < 1.03x
or correctness fail
```

---

# 7. OPT4D：Fused Candidate Generation + Selection

## 7.1 目标

如果 OPT4B/OPT4C 的主要收益来自减少 candidate global write 和 selection kernel，那么可以单独做一个更保守的融合版本。

目标：

```text
保留 baseline target-seed candidate generation；
减少或融合 select_best_per_target kernel；
降低 global memory candidate write/read。
```

## 7.2 两种实现路线

### 路线 D1：使用 block-per-target 自带 fused selection

如果 OPT4C 已经实现，那么 D1 等价于 OPT4C 的一部分。

### 路线 D2：两阶段轻量压缩

第一 kernel 仍输出每个 candidate 的最小摘要：

```text
success_rank
near_limit
pose_cost
seed_id
q[6]
```

不再输出所有中间字段。

第二 kernel 只读摘要做 selection。

这种方案保守，风险低。

## 7.3 输出

```text
data/results/opt/opt4_followup/opt4d_fused_selection.csv
docs/opt/opt4_followup/opt4d_fused_selection_report.md
```

## 7.4 成功标准

```text
quality unchanged
selection phase speedup >= 1.20x
or total gpu_stream_ms speedup >= 1.05x
```

如果 total 不明显提升，说明 selection 不是主瓶颈，写入报告。

---

# 8. 统一 benchmark 汇总

完成 OPT4B/C/D 后，生成总表：

```text
data/results/opt/opt4_followup/opt4_followup_summary.csv
docs/opt/opt4_followup/opt4_followup_summary.md
```

## 8.1 Summary CSV 字段

```text
variant
correctness_pass
N1000_strict_sr
N1000_pos_p95_all_mm
N1000_near_limit
N1000_gpu_stream_ms
N1000_speedup_vs_baseline
N5000_gpu_stream_ms
N5000_speedup_vs_baseline
registers_per_thread
achieved_occupancy
sm_utilization
dram_throughput
branch_divergence
warp_execution_efficiency
spill
paper_decision
notes
```

## 8.2 paper_decision 枚举

```text
main_result
appendix
discussion
future_work
discard
```

---

# 9. 最终决策规则

## 9.1 替换 baseline 的条件

只有满足以下全部条件，才能作为新的性能主表：

```text
1. correctness pass
2. N=1000 Strict SR >= 93%
3. N=1000 pos_p95_all <= 8 mm
4. near_limit <= 4%
5. N=1000 speedup vs baseline >= 1.10x
6. N=5000 speedup vs baseline >= 1.10x
7. Nsight 至少一个核心指标改善：
   - registers/thread 降低
   - achieved occupancy 提升
   - spill 降低
   - branch divergence 降低
   - selection kernel overhead 降低
```

## 9.2 进入 Appendix 的条件

```text
correctness pass
但 speedup < 1.10x
```

## 9.3 进入 Discussion/Future Work 的条件

```text
correctness fail
或 speedup 下降
但 Nsight 说明了瓶颈原因
```

## 9.4 直接丢弃的条件

```text
无法编译
无法稳定运行
无有效数据
```

---

# 10. 最终报告必须给出的结论

`docs/opt/opt4_followup/opt4_followup_summary.md` 必须明确回答：

```text
1. 原始 warp-per-seed 为什么失败？
2. block-per-target/thread-per-seed 是否更好？
3. warp-per-target/lane-per-seed 是否更好？
4. fused selection 是否有意义？
5. 当前 CUDA-V4 的线程映射瓶颈到底是什么？
6. 继续优化应该走 kernel mapping，还是 Adaptive-K？
7. 哪些结果进入主文？
8. 哪些结果进入 appendix？
9. 哪些结果进入 future work？
```

建议结论模板：

如果 OPT4C/OPT4B 成功：

```text
OPT4 后续实验表明，围绕 K16 多种子结构设计的 target-level cooperative mapping 能够在保持 V4-Final-K16 解质量的同时进一步提升 CUDA throughput。该结果说明，比原始 warp-per-seed 更粗粒度的 target-level seed 并行更适合本问题。
```

如果全部失败：

```text
OPT4 后续实验表明，简单调整 warp/block 映射无法显著突破当前 V4 IK kernel 的瓶颈。由于每个 seed 内部包含强串行 LM 迭代、FP64 小矩阵求解和较高寄存器压力，粗粒度线程协作无法稳定带来收益。相比之下，Adaptive-K 通过减少平均 seed 计算量实现了约 2.94x 加速，是当前更有效的算法-系统协同优化方向。
```

---

# 11. 执行顺序

严格按以下顺序：

```text
Step 1: 创建 opt4_followup 输出目录。
Step 2: 锁定 baseline 指标，写 opt4_baseline_snapshot.csv。
Step 3: 生成原始 warp-per-seed postmortem。
Step 4: 实现 OPT4C block-per-target/thread-per-seed。
Step 5: OPT4C N=100 correctness。
Step 6: OPT4C N=1000/N=5000 benchmark。
Step 7: OPT4C Nsight。
Step 8: 实现 OPT4B warp-per-target/lane-per-seed。
Step 9: OPT4B N=100 correctness。
Step 10: OPT4B N=1000/N=5000 benchmark。
Step 11: OPT4B Nsight。
Step 12: 实现 OPT4D fused selection 或从 OPT4C/OPT4B 中抽取 selection 对比。
Step 13: 生成 opt4_followup_summary.csv。
Step 14: 生成 opt4_followup_summary.md。
Step 15: 明确主文 / 附录 / discussion / future work 决策。
```

---

# 12. 最小闭环

如果 Codex 额度或时间不足，至少完成：

```text
1. 原始 warp-per-seed postmortem；
2. OPT4C block-per-target/thread-per-seed；
3. OPT4C correctness + N=1000 benchmark；
4. opt4_followup_summary.md。
```

这是最小有价值闭环。

---

# 13. Codex 执行提示词

请继续执行 OPT4 后续 CUDA 线程映射优化实验。

本轮不要继续死磕原始 warp-per-seed，而是完成：

```text
1. 原始 warp-per-seed postmortem；
2. OPT4C block-per-target/thread-per-seed；
3. OPT4B warp-per-target/lane-per-seed；
4. OPT4D fused candidate selection；
5. OPT4 follow-up summary。
```

所有新增实验必须保持 V4-Final-K16 数学逻辑不变，不得修改阈值，不得修改 success 判定，不得覆盖已有 baseline 和 final reports。

每个 variant 必须先做 N=100 correctness，未通过则停止该分支，并输出 failure cases。通过后再跑 N=1000/N=5000 benchmark 和 Nsight。

最终必须生成：

```text
docs/opt/opt4_followup/opt4_warp_per_seed_postmortem.md
docs/opt/opt4_followup/opt4c_block_target_report.md
docs/opt/opt4_followup/opt4b_warp_target_report.md
docs/opt/opt4_followup/opt4d_fused_selection_report.md
docs/opt/opt4_followup/opt4_followup_summary.md

data/results/opt/opt4_followup/opt4_warp_per_seed_postmortem.csv
data/results/opt/opt4_followup/opt4c_block_target_correctness.csv
data/results/opt/opt4_followup/opt4c_block_target_benchmark.csv
data/results/opt/opt4_followup/opt4b_warp_target_correctness.csv
data/results/opt/opt4_followup/opt4b_warp_target_benchmark.csv
data/results/opt/opt4_followup/opt4d_fused_selection.csv
data/results/opt/opt4_followup/opt4_followup_summary.csv
```

如果 OPT4C 或 OPT4B 达到：

```text
correctness pass
N=1000 speedup >= 1.10x
N=5000 speedup >= 1.10x
quality pass
```

则可作为新的 optimization result。

否则，必须明确写入 Discussion：当前问题更适合 Adaptive-K 这类减少 seed 计算量的算法-系统协同优化，而不是简单 warp/block 映射重排。

# END
