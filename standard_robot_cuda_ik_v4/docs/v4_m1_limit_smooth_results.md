# V4 M1/M2 实验报告

生成时间: 2026-06-15 23:40:18

## 1. 实验目的

M1: Limit Barrier 在 N=1000 下复核。M2: Smoothness 改为候选重排序方案。

## 2. M1 Limit Barrier N=1000 复核

| 配置 | K | w_limit | Loose SR | Medium SR | Strict SR | 单调 | pos_p95_all | pos_p95_suc | near_lim | iters | time/s |
|------|---|---------|----------|-----------|-----------|---|-------------|-------------|----------|-------|--------|
| K16-w0 | 16 | 0 | 0.9650 | 0.9570 | 0.9510 | ✓ | 4.69 | 2.88 | 0.0930 | 24.0 | 186 |
| K16-w1.0 | 16 | 1.0 | 0.9550 | 0.9460 | 0.9440 | ✓ | 18.80 | 2.83 | 0.0120 | 24.4 | 207 |

### M1 判定
- Strict SR 下降: 0.0070 (标准 ≤0.01)
- near_limit 下降: 0.0810 (9.3% → 1.2%)
- **✅ 冻结 V4-Limit-w1.0**

## 3. M2 Smoothness Candidate Reranking

### 3.1 旧方案失败原因

M0 中 smoothness penalty 直接加入 LM 残差 → Strict SR≈2%。位姿优化被平滑项主导。

### 3.2 新方案

字典序候选重排序: success_rank → near_limit → smoothness → pose_cost。位姿优先，平滑其次。

### 3.3 结果

| 轨迹 | 方法 | Strict SR | pos_p95_all | mean_Δq/rad | max_Δq/rad | jumps |
|------|------|-----------|-------------|-------------|------------|-------|
| line | independent | 0.920 | 5.2 | 2.4906 | 3.0985 | 46 |
| line | rerank | 0.940 | 4.3 | 1.3158 | 2.7015 | 39 |
| arc | independent | 1.000 | 1.8 | 2.3940 | 3.1091 | 44 |
| arc | rerank | 1.000 | 3.1 | 1.5512 | 3.0739 | 45 |
| local_random | independent | 1.000 | 1.2 | 2.5937 | 3.1314 | 46 |
| local_random | rerank | 1.000 | 2.4 | 1.1472 | 2.6494 | 36 |

**line**: dq_mean 2.4906→1.3158 (+47%), Strict SR 0.920→0.940 (+0.020)

**arc**: dq_mean 2.3940→1.5512 (+35%), Strict SR 1.000→1.000 (+0.000)

**local_random**: dq_mean 2.5937→1.1472 (+56%), Strict SR 1.000→1.000 (+0.000)

## 4. 结论
- **M1 Limit Barrier**: ✅ 冻结 V4-Limit-w1.0 (near_limit ratio 9.3%→1.2%)
- **M2 Smoothness Reranking**: 见上表 pairwise 对比。若 dq 下降≥20%且 SR 下降≤2pp则保留。

## 5. 下一步
- 若 M1/M2 通过: 推进 V4 Module D 简化碰撞检测
- 若 M2 未通过: 暂不进入 smoothness，优先碰撞检测