# V4-Final-K16 Adaptive-K Report

## 1. Scope

本报告对应增强计划 Phase B：Adaptive-K。

目标是在不改变 V4-Final-K16 算法评价阈值和 success 判定逻辑的前提下，评估是否可以通过“easy target 少算，hard target 多算”的策略减少平均 seed 数，同时保持接近 K16 的解质量。

本轮 Adaptive-K 不替代已锁定 baseline，除非同时满足质量和性能门槛。

## 2. Methods

本轮评估 4 种方法：

| method | description |
|---|---|
| K16 baseline | 固定计算 16 个 Sobol seeds，作为质量上限和主 baseline |
| K8-only | 只计算前 8 个 seeds，不 rescue |
| AK-8+8 | 第一阶段 K=8，若 strict 失败则计划 rescue 剩余 K=8 |
| AK-4+4+8 | 第一阶段 K=4，失败后 K=4，再失败后 K=8 |

实现口径：

- Stage 2 / Stage 3 使用 compact failed-target rerun，而不是对所有目标重复全量计算。
- 最终选择仍使用 V4-Final-K16 的 candidate ranking 逻辑。
- 不改变 Strict / Medium / Loose 阈值。
- 不改变 success 判定逻辑。
- 不重新搜索 seed bank。
- benchmark 使用 `warmup=10`、`repeat=30`。

输出文件：

- `data/results/adaptive/adaptive_k_benchmark.csv`
- `docs/adaptive/adaptive_k_report.md`

## 3. Acceptance Gates

Adaptive-K 若要进入论文主文，需要满足：

质量门槛：

- N=1000 Strict SR ≥ K16 baseline - 1.0 pp。
- N=1000 pos_p95_all ≤ K16 baseline + 2 mm。
- near_limit ≤ 4%。
- monotonic pass。
- no NaN/Inf。

性能门槛：

- avg_seeds_evaluated ≤ 12。
- N=1000 speedup vs K16 ≥ 1.20x。
- N=5000 speedup vs K16 ≥ 1.20x。

## 4. Full Result Table

| method | N | avg seeds | stage1 success | stage2 rescue | stage3 rescue | Strict SR | pos_p95_all mm | near_limit | gpu_stream_ms | speedup vs K16 | pass_quality | pass_perf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K16 baseline | 100 | 16.000 |  |  |  | 0.960 | 4.384514 | 0.010 | 73.666339 | 1.000000 | 1 | 1 |
| K8-only | 100 | 8.000 | 0.930 | 0.000 | 0.000 | 0.930 | 15.550133 | 0.040 | 33.446266 | 2.202528 | 0 | 1 |
| AK-8+8 | 100 | 8.000 | 0.930 | 0.000 | 0.000 | 0.930 | 15.550133 | 0.040 | 33.667164 | 2.188077 | 0 | 1 |
| AK-4+4+8 | 100 | 5.200 | 0.840 | 0.160 | 0.070 | 0.960 | 4.794823 | 0.040 | 26.843561 | 2.744283 | 1 | 1 |
| K16 baseline | 500 | 16.000 |  |  |  | 0.954 | 4.339606 | 0.004 | 319.969279 | 1.000000 | 1 | 1 |
| K8-only | 500 | 8.000 | 0.922 | 0.000 | 0.000 | 0.922 | 37.596819 | 0.018 | 152.002557 | 2.105026 | 0 | 1 |
| AK-8+8 | 500 | 8.000 | 0.922 | 0.000 | 0.000 | 0.922 | 37.596819 | 0.018 | 151.594709 | 2.110689 | 0 | 1 |
| AK-4+4+8 | 500 | 5.240 | 0.846 | 0.154 | 0.078 | 0.954 | 4.823029 | 0.020 | 116.220616 | 2.753120 | 1 | 1 |
| K16 baseline | 1000 | 16.000 |  |  |  | 0.954 | 4.508959 | 0.007 | 639.158346 | 1.000000 | 1 | 1 |
| K8-only | 1000 | 8.000 | 0.924 | 0.000 | 0.000 | 0.924 | 36.093694 | 0.021 | 301.640782 | 2.118939 | 0 | 1 |
| AK-8+8 | 1000 | 8.000 | 0.924 | 0.000 | 0.000 | 0.924 | 36.093694 | 0.021 | 300.642978 | 2.125971 | 0 | 1 |
| AK-4+4+8 | 1000 | 5.192 | 0.854 | 0.146 | 0.076 | 0.954 | 4.828379 | 0.027 | 217.313307 | 2.941184 | 1 | 1 |
| K16 baseline | 5000 | 16.000 |  |  |  | 0.954 | 4.508959 | 0.007 | 3186.436120 | 1.000000 | 1 | 1 |
| K8-only | 5000 | 8.000 | 0.924 | 0.000 | 0.000 | 0.924 | 36.093694 | 0.021 | 1495.135810 | 2.131202 | 0 | 1 |
| AK-8+8 | 5000 | 8.000 | 0.924 | 0.000 | 0.000 | 0.924 | 36.093694 | 0.021 | 1490.175450 | 2.138296 | 0 | 1 |
| AK-4+4+8 | 5000 | 5.192 | 0.854 | 0.146 | 0.076 | 0.954 | 4.828379 | 0.027 | 1054.137250 | 3.022791 | 1 | 1 |

## 5. Method-Level Interpretation

### 5.1 K8-only

K8-only 的速度明显提升：

- N=1000 speedup = 2.118939x。
- N=5000 speedup = 2.131202x。
- avg seeds = 8。

但是质量不达标：

- N=1000 Strict SR 从 0.954 降到 0.924，下降 3.0 pp。
- N=1000 pos_p95_all 从 4.508959 mm 增加到 36.093694 mm。
- `pass_quality=0`。

结论：K8-only 只能作为速度上限和质量下限对照，不能进入论文主配置。

### 5.2 AK-8+8

AK-8+8 的结果与 K8-only 基本一致：

- N=1000 Strict SR = 0.924。
- N=1000 pos_p95_all = 36.093694 mm。
- N=1000 speedup = 2.125971x。
- `pass_quality=0`。

解释：

- 当前 rescue 触发和最终候选合并逻辑没有恢复到 K16 质量。
- 对这批目标，前 8 seeds 失败样本中存在需要更细粒度阶段或更严格 rescue 条件的情况。
- 该结果说明“只做 8+8 rescue”不足以支撑论文主文。

结论：AK-8+8 不进入主文，可作为 negative ablation。

### 5.3 AK-4+4+8

AK-4+4+8 是本轮 Adaptive-K 中唯一同时通过质量和性能门槛的方法。

N=1000 结果：

- Strict SR = 0.954，与 K16 baseline 相同。
- pos_p95_all = 4.828379 mm，比 baseline 高 0.319420 mm，低于 baseline + 2 mm 门槛。
- near_limit = 0.027，低于 4%。
- avg seeds = 5.192，低于 12。
- speedup vs K16 = 2.941184x。
- pass_quality = 1。
- pass_perf = 1。

N=5000 结果：

- Strict SR = 0.954。
- pos_p95_all = 4.828379 mm。
- avg seeds = 5.192。
- speedup vs K16 = 3.022791x。

解释：

- 分成 4+4+8 后，Stage 1 更容易筛出 easy targets。
- Stage 2 / Stage 3 只处理失败目标，因此平均 seed 数显著低于固定 K16。
- 质量恢复到 K16 水平，说明 rescue 分层对该目标集有效。

结论：AK-4+4+8 满足主文进入条件。建议论文中作为“可选计算自适应策略”呈现，而不是替代已经锁定的 V4-Final-K16 baseline。

## 6. Answers Required by Plan

1. K8-only 损失多少成功率？

N=1000 Strict SR 从 0.954 降到 0.924，损失 3.0 pp；pos_p95_all 增加到 36.093694 mm，因此质量不可接受。

2. AK-8+8 是否接近 K16？

不接近。N=1000 Strict SR 仍为 0.924，pos_p95_all 仍为 36.093694 mm，未恢复 K16 质量。

3. AK-4+4+8 是否更省且质量可接受？

是。N=1000 avg seeds = 5.192，speedup = 2.941184x，Strict SR = 0.954，pos_p95_all = 4.828379 mm。

4. rescue target 占比是多少？

N=1000 AK-4+4+8：

- Stage 1 success rate = 0.854。
- Stage 2 rescue rate = 0.146。
- Stage 3 rescue rate = 0.076。

5. 平均 seed 数是多少？

N=1000 AK-4+4+8 平均 seed 数为 5.192。

6. speedup 是否来自少算 seed，而不是计时口径变化？

是。K16 和 Adaptive-K 均按 `warmup=10`、`repeat=30` 记录，Adaptive-K 的 speedup 与 `avg_seeds_evaluated` 的下降一致。

7. 是否值得进入论文主文？

AK-4+4+8 值得作为可选主文结果；K8-only 和 AK-8+8 只能作为 ablation / appendix。

## 7. Paper Claim Decision

可以写：

```text
AK-4+4+8 在该目标集上以平均约 5.19 个 seeds 达到固定 K16 的 Strict SR，并在 N=1000 / N=5000 上获得约 2.94x / 3.02x speedup。
```

必须保守写：

```text
Adaptive-K 是 V4-Final-K16 的可选计算自适应扩展，不替代 baseline 的主要正确性闭环。
```

不能写：

```text
所有 Adaptive-K 策略都保持 K16 质量。
```

最终判断：

```text
Adaptive-K package complete.
AK-4+4+8 passes quality and performance gates.
K8-only and AK-8+8 fail quality gates and should remain ablation/appendix.
```
