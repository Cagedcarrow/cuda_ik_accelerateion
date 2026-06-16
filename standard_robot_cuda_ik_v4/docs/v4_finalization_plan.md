# V4 成型方案：冻结 Constraint-Aware Batch IK 算法版本

## 0. 当前任务目标

当前项目路径：

```text
standard_robot_cuda_ik_v4/
```

本阶段目标不是继续无限扩展 V5，也不是立刻重写 CUDA，而是让 V4 算法版本正式成型。

V4 的最终定位：

```text
Analytical Jacobian
+ LM Solver
+ Sobol Multi-Seed
+ Joint-Limit Barrier
+ Smoothness Candidate Reranking
```

目标论文定位：

```text
GPU-oriented / GPU-native constraint-aware batch IK solver
```

当前应完成的是：

```text
V4-Algorithm-Final
```

而不是：

```text
V4-CUDA-Final
```

CUDA Port 是下一阶段。

---

# 1. 当前 V4 状态判断

## 1.1 已经成立的部分

### V3 Freeze 已成立

V3 最终主线已经确认：

```text
Analytical Jacobian + LM + Sobol-K16/K32
```

该版本已经解决了早期 Medium/Strict 不一致问题，统一采用：

```text
统一求解 + 多阈值评价
```

最终应保留：

```text
V3-Sobol-K16 = 主平衡配置
V3-Sobol-K32 = 高成功率 / 高精度配置
```

### Smoothness Reranking 已成立

旧方案：

```text
smoothness penalty 直接加入 LM residual
```

已经废弃，因为会导致 Strict SR 接近崩溃。

新方案：

```text
success_rank → near_limit → smoothness → pose_cost
```

采用字典序候选重排序，已经在 line / arc / local_random 三类轨迹上证明有效。

结论：

```text
Smoothness 不应作为主优化残差，而应作为候选解选择准则。
```

该模块可以进入 V4 最终方法。

---

## 1.2 尚未最终冻结的部分

### Limit Barrier 尚未最终冻结

当前 `w_limit=1.0` 的结果：

```text
near_limit ratio 显著下降
Strict SR 下降在可接受范围内
但 pos_p95_all 明显恶化
```

因此不能直接冻结：

```text
V4-Limit-w1.0
```

更合理的判断是：

```text
Limit Barrier 方向成立，但权重需要进一步扫参。
```

本阶段核心任务是找到一个折中权重，使其满足：

```text
near_limit 明显下降
Strict SR 基本不下降
pos_p95_all 不恶化
```

---

# 2. V4 是否成型的判定标准

V4 成型分为三个层级。

## 2.1 V4-Algorithm Candidate

满足以下条件即可认为“V4 方向成立”：

| 项目                            | 标准 |
| ----------------------------- | -- |
| V3-Sobol-K16/K32 已冻结          | 是  |
| Limit Barrier 能降低 near-limit  | 是  |
| Smoothness Rerank 能降低 mean_Δq | 是  |
| Loose ≥ Medium ≥ Strict       | 必须 |
| 所有失败旧方案有记录                    | 必须 |

当前状态：

```text
已经达到 V4-Algorithm Candidate。
```

---

## 2.2 V4-Algorithm Final

满足以下条件才能认为“V4 算法成型”：

| 项目                  | 标准                           |
| ------------------- | ---------------------------- |
| Final seed 策略       | Sobol-K16/K32                |
| Final limit 权重      | 已确定                          |
| Final smoothness 策略 | candidate reranking          |
| N=1000 IK 结果        | 完整                           |
| 轨迹结果                | line / arc / local_random 完整 |
| Limit 不造成长尾恶化       | pos_p95_all 不显著劣化            |
| 最终表格                | 已生成                          |
| 旧结果                 | 已标记废弃                        |

当前状态：

```text
尚未完全达到。
```

缺口：

```text
Limit Barrier 权重未最终冻结。
```

---

## 2.3 V4-System Final

满足以下条件才能认为“V4 系统成型”：

| 项目                            | 标准  |
| ----------------------------- | --- |
| CUDA kernel                   | 已实现 |
| Python-CUDA correctness check | 已完成 |
| Nsight profiling              | 已完成 |
| cuRobo benchmark              | 已完成 |
| GPU stream / E2E 时间           | 已报告 |
| valid throughput              | 已报告 |

当前状态：

```text
尚未开始。
```

因此当前结论应写成：

```text
V4 算法方向已经成型，V4 算法最终版还差 Limit 权重冻结，V4 CUDA 系统版尚未开始。
```

---

# 3. 本阶段最终目标

本阶段目标是生成：

```text
V4-Algorithm-Final
```

最终方法建议命名为：

```text
V4-Final-K16
V4-Final-K32
```

其中：

```text
V4-Final-K16 = Sobol-K16 + tuned Limit Barrier + Smooth Rerank
V4-Final-K32 = Sobol-K32 + tuned Limit Barrier + Smooth Rerank
```

如果 Limit Barrier 最终无法同时满足 near-limit 和 pos_p95_all，则最终命名为：

```text
V4-SmoothOnly-K16
V4-SmoothOnly-K32
```

并将 Limit Barrier 作为 ablation，而不是主方法。

---

# 4. 必须执行的实验

## 4.1 Experiment A：Limit Barrier 权重细扫

### 目的

找出最终 limit 权重，避免 `w_limit=1.0` 造成长尾误差恶化。

### 固定条件

```text
N = 1000
Solver = LM
Jacobian = Analytical
Seed = Sobol
Evaluation = unified solve + multi-threshold
```

### 测试 K16

| 配置       |  K | w_limit |
| -------- | -: | ------: |
| baseline | 16 |       0 |
| limit    | 16 |    0.03 |
| limit    | 16 |    0.05 |
| limit    | 16 |     0.1 |
| limit    | 16 |     0.2 |
| limit    | 16 |     0.3 |
| limit    | 16 |     0.5 |
| limit    | 16 |     1.0 |

### 测试 K32

如果时间允许：

| 配置       |  K | w_limit |
| -------- | -: | ------: |
| baseline | 32 |       0 |
| limit    | 32 |    0.05 |
| limit    | 32 |     0.1 |
| limit    | 32 |     0.3 |
| limit    | 32 |     0.5 |

### 输出指标

每组输出：

| 指标                    | 说明          |
| --------------------- | ----------- |
| Loose SR              | 30mm / 10°  |
| Medium SR             | 10mm / 5°   |
| Strict SR             | 5mm / 1°    |
| pos_p50_all           | 全部目标        |
| pos_p95_all           | 全部目标        |
| pos_p99_all           | 全部目标        |
| pos_max_all           | 全部目标        |
| pos_p95_suc           | Strict 成功样本 |
| rot_p95_all           | 全部目标        |
| rot_p95_suc           | Strict 成功样本 |
| near_limit_ratio      | 近限位比例       |
| joint_violation_count | 限位违反数量      |
| iter_mean             | 平均迭代        |
| time_s                | 总耗时         |
| monotonic_pass        | 单调性         |

### 冻结标准

Limit 权重进入 V4-Final 的标准：

| 指标                    |                要求 |
| --------------------- | ----------------: |
| Strict SR 下降          |          ≤ 1.0 pp |
| near_limit_ratio      |            ≤ 3.0% |
| joint_violation_count |                 0 |
| pos_p95_all           | ≤ baseline + 2 mm |
| pos_p99_all           |             不严重恶化 |
| monotonic_pass        |              True |

如果没有任何权重满足，则 Limit 不进入主方法，只作为 ablation。

---

## 4.2 Experiment B：Limit 失败样本诊断

### 目的

解释为什么 `w_limit=1.0` 造成 pos_p95_all 恶化。

### 对比

比较：

```text
K16-w0
K16-w0.1
K16-w0.3
K16-w1.0
```

### 需要输出失败样本表

```text
data/results/v4_limit_failure_cases.csv
```

字段：

| 字段             |
| -------------- |
| target_id      |
| w_limit        |
| pos_err        |
| rot_err        |
| strict_success |
| best_seed_id   |
| near_limit     |
| limit_score    |
| q_min_distance |
| q_max_distance |
| iter_count     |
| failure_type   |

### 诊断问题

重点回答：

1. pos_p95_all 恶化是否来自少数极端失败样本？
2. 这些样本是否接近关节限位？
3. Limit penalty 是否把解推离正确收敛盆地？
4. w=0.1 或 w=0.3 是否可以避免该问题？
5. 是否需要 adaptive limit weight？

---

## 4.3 Experiment C：Smoothness Rerank 稳定性复核

### 目的

确认 smoothness rerank 不是偶然结果。

### 轨迹类型

保留三类：

```text
line
arc
local_random
```

每类至少：

```text
50 waypoints
```

如果时间允许，扩展到：

```text
100 waypoints
```

### 对比方法

| 方法                        | 说明                   |
| ------------------------- | -------------------- |
| V3-independent-K16        | 独立 pose-loss 最小      |
| V4-rerank-K16             | 字典序重排序               |
| V4-final-limit-rerank-K16 | tuned limit + 字典序重排序 |
| V3-independent-K32        | 独立 pose-loss 最小      |
| V4-rerank-K32             | 字典序重排序               |

如果 Limit 权重没有冻结，则不要做 `limit-rerank`。

### 评价指标

| 指标                        | 说明            |   |    |   |              |
| ------------------------- | ------------- | - | -- | - | ------------ |
| waypoint Loose SR         | 单点 Loose      |   |    |   |              |
| waypoint Medium SR        | 单点 Medium     |   |    |   |              |
| waypoint Strict SR        | 单点 Strict     |   |    |   |              |
| pos_p95_all               | 全部 waypoint   |   |    |   |              |
| pos_p95_suc               | 成功 waypoint   |   |    |   |              |
| rot_p95_all               | 姿态            |   |    |   |              |
| mean_delta_q              | 平均相邻关节变化      |   |    |   |              |
| p95_delta_q               | p95 相邻关节变化    |   |    |   |              |
| max_delta_q               | 最大相邻关节变化      |   |    |   |              |
| jump_count_linf           | `             |   | Δq |   | ∞ > 0.5 rad` |
| jump_count_l2             | `             |   | Δq |   | 2 > 1.0 rad` |
| jerk_cost                 | 二阶差分或三阶差分代价   |   |    |   |              |
| trajectory_success_strict | 所有点 Strict 成功 |   |    |   |              |

### Smoothness 冻结标准

Smoothness rerank 进入 V4-Final 的标准：

| 指标                    |      要求 |
| --------------------- | ------: |
| waypoint Strict SR 下降 |  ≤ 2 pp |
| mean_delta_q 下降       |   ≥ 20% |
| p95_delta_q 下降        |   ≥ 15% |
| jump_count 至少部分下降     |       是 |
| pos_p95_all           | 仍在可接受范围 |
| monotonic_pass        |    True |

当前初步数据已经满足 mean_delta_q 下降，但 arc 的 jump_count 未改善，因此最终报告中不能夸大为“所有跳变均降低”。

---

# 5. 最终 V4 版本选择规则

## 5.1 Final-K16

优先选择：

```text
Sobol-K16 + best w_limit + smooth rerank
```

若 Limit 无法满足长尾要求，则选择：

```text
Sobol-K16 + smooth rerank
```

## 5.2 Final-K32

优先选择：

```text
Sobol-K32 + best w_limit + smooth rerank
```

若 K32 计算成本过高，则 K32 作为 high-accuracy mode，不作为默认主方法。

---

# 6. 最终报告输出要求

请生成：

```text
standard_robot_cuda_ik_v4/docs/v4_finalization_report.md
standard_robot_cuda_ik_v4/data/results/v4_limit_weight_sweep.csv
standard_robot_cuda_ik_v4/data/results/v4_limit_failure_cases.csv
standard_robot_cuda_ik_v4/data/results/v4_smooth_rerank_final.csv
standard_robot_cuda_ik_v4/data/results/v4_final_summary.csv
```

报告结构：

```markdown
# V4 Finalization Report

## 1. 当前 V4 定位

## 2. V3 Freeze 结果摘要

## 3. Limit Barrier 权重扫描

### 3.1 实验配置
### 3.2 结果表
### 3.3 最终权重选择
### 3.4 失败样本诊断

## 4. Smoothness Candidate Reranking

### 4.1 方法说明
### 4.2 轨迹数据
### 4.3 对比结果
### 4.4 是否进入最终方法

## 5. V4-Final 方法定义

### 5.1 V4-Final-K16
### 5.2 V4-Final-K32

## 6. V4 是否成型判断

## 7. 后续 CUDA Port 建议
```

---

# 7. 最终 V4 成型判断模板

报告最后必须给出明确判断，不要含糊。

## 情况 A：V4 成型

如果满足：

```text
Limit 找到不恶化长尾的权重
Smoothness rerank 保持成功率并降低 Δq
K16/K32 最终表完整
```

则写：

```text
结论：V4-Algorithm-Final 已成型。

最终版本为：
V4-Final-K16 = Analytical Jacobian + LM + Sobol-K16 + Limit Barrier(w=...) + Smoothness Candidate Reranking
V4-Final-K32 = Analytical Jacobian + LM + Sobol-K32 + Limit Barrier(w=...) + Smoothness Candidate Reranking

V4 相比 V3 的增量为：
1. 在基本保持 Strict SR 的前提下降低 near-limit ratio；
2. 在基本保持 waypoint success rate 的前提下降低相邻关节变化；
3. 从 IK-only solver 升级为 constraint-aware IK front-end。

下一阶段应进入 CUDA Port，而不是继续增加算法模块。
```

## 情况 B：V4 部分成型

如果 Limit 权重仍导致长尾恶化，但 Smoothness 成功，则写：

```text
结论：V4 部分成型。

最终保留：
V4-Smooth-K16/K32 = Analytical Jacobian + LM + Sobol + Smoothness Candidate Reranking

Limit Barrier 不进入主方法，仅作为 ablation：
它能降低 near-limit ratio，但会放大全体误差长尾，因此暂不冻结为最终模块。

下一阶段可以：
1. 直接进入 CUDA Port；
2. 或继续研究 adaptive limit weight。
```

## 情况 C：V4 未成型

如果 Smoothness 复核失败，Limit 也无法稳定，则写：

```text
结论：V4 未成型。

保留 V3-Sobol-K16/K32 作为最终 IK-only 方法。
V4 约束模块仅作为探索性实验，不进入论文主方法。
下一阶段应停止扩展约束模块，优先进行 V3 CUDA Port。
```

---

# 8. 后续 CUDA Port 的进入条件

只有当 V4 成型或部分成型后，才进入 CUDA Port。

进入条件：

| 条件              | 标准  |
| --------------- | --- |
| V4-Final 方法定义明确 | 是   |
| Limit 是否保留      | 已决定 |
| Smoothness 是否保留 | 已决定 |
| Final CSV       | 已生成 |
| Final report    | 已生成 |
| 不再继续改算法主结构      | 是   |

进入 CUDA Port 后，优先实现：

```text
cuda/
├── fk_analytic.cu
├── jacobian_analytic.cu
├── lm_multiseed_kernel.cu
├── limit_barrier.cu
├── candidate_select.cu
└── benchmark_v4.cu
```

第一阶段 CUDA 只实现：

```text
V4-Final-K16
```

不要一开始同时实现 K16/K32、smooth、collision 全套。

---

# 9. 当前执行顺序

严格按以下顺序执行：

1. 读取已有 M0、M1/M2 报告；
2. 补做 Limit weight sweep；
3. 诊断 w=1.0 长尾恶化原因；
4. 选择最终 limit 权重，或决定不保留 Limit；
5. 复核 Smoothness Rerank；
6. 生成 V4-Final-K16 / K32 最终汇总表；
7. 在报告中明确判断 V4 是否成型；
8. 如果 V4 成型，停止继续加模块，准备 CUDA Port；
9. 如果 V4 部分成型，只保留成功模块；
10. 如果 V4 未成型，回退到 V3-Sobol 并准备 V3 CUDA Port。

---

# 10. 当前最可能的结果预判

根据已有结果，最可能形成：

```text
V4 部分成型或完全成型。
```

更具体地说：

```text
Smoothness Rerank 大概率进入最终方法。
Limit Barrier 需要重新选择较小权重，w=1.0 不宜直接作为最终主配置。
```

最可能的最终版本：

```text
V4-Final-K16 =
Analytical Jacobian
+ LM
+ Sobol-K16
+ Limit Barrier(w=0.1 或 0.3)
+ Smoothness Candidate Reranking
```

如果小权重 Limit 仍然导致 pos_p95_all 恶化，则最终版本改为：

```text
V4-Smooth-K16 =
Analytical Jacobian
+ LM
+ Sobol-K16
+ Smoothness Candidate Reranking
```

此时 Limit Barrier 作为消融实验保留，不进入主方法。

# END
