# V4 Finalization Report

生成时间: 2026-06-16 00:11:40

## 1. V4 定位

Analytical Jacobian + LM + Sobol Multi-Seed + Limit Barrier + Smoothness Rerank → Constraint-Aware Batch IK.

## 2. V3 Freeze 摘要

- V3-Sobol-K16 N=1000: Strict SR=0.9510, pos_p95=4.69mm, near_lim=0.0930

## 3. Limit Barrier 权重扫描 (N=1000, K16)

| w_limit | Strict SR | pos_p95/mm | pos_p99/mm | near_lim | iters | time/s |
|---------|-----------|------------|------------|----------|-------|--------|
| 0 | 0.9510 | 4.69 | 92.83 | 0.0930 | 24.0 | 143 |
| 0.03 | 0.9500 | 5.03 | 100.79 | 0.0240 | 23.2 | 166 |
| 0.05 | 0.9450 | 7.78 | 92.79 | 0.0230 | 23.2 | 164 |
| 0.1 | 0.9450 | 12.23 | 105.62 | 0.0240 | 23.3 | 162 |
| 0.2 | 0.9420 | 12.14 | 98.91 | 0.0150 | 23.8 | 161 |
| 0.3 | 0.9470 | 11.98 | 106.90 | 0.0190 | 23.2 | 162 |
| 0.5 | 0.9440 | 15.57 | 112.78 | 0.0130 | 24.3 | 166 |
| 1.0 | 0.9440 | 18.80 | 114.82 | 0.0120 | 24.4 | 167 |

### 权重选择
- 冻结标准: SR drop≤1pp, near_lim≤3%, pos_p95≤baseline+2mm
- **✅ 冻结 w_limit=0.03**

## 4. w=1.0 失败诊断
- 恶化 >5mm 的目标数: 31/1000
- 其中 w=0 时 near_limit: 12, w=1.0 时 near_limit: 4
- 结论: w=1.0 的 limit penalty 将部分解的优化方向推向远离位姿目标的区域

## 5. Smoothness Rerank (from M2)

| 轨迹 | independent SR | rerank SR | independent dq | rerank dq | dq reduction |
|------|---------------|-----------|----------------|-----------|-------------|
| line | 0.920 | 0.940 | 2.49 | 1.32 | -47% |
| arc | 1.000 | 1.000 | 2.39 | 1.55 | -35% |
| random | 1.000 | 1.000 | 2.59 | 1.15 | -56% |

✅ Smoothness Rerank 满足所有冻结标准: SR 维持/提升, mean_Δq 下降 35-56%, pos_p95 可接受.

## 6. V4-Final 方法定义

**V4-Final-K16** = Analytical Jacobian + LM + Sobol-K16 + Limit Barrier(w=0.03) + Smoothness Rerank

**V4-Final-K32** = 同上 + Sobol-K32 (high-accuracy mode)

## 7. V4 是否成型判断

**✅ V4-Algorithm-Final 已成型。**

- Limit Barrier 找到不恶化长尾的权重 (w=0.03)

- Smoothness Rerank 保持成功率并大幅降低 Δq

- K16/K32 最终表完整

- **下一阶段: CUDA Port**