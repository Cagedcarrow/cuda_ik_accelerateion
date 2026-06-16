# OPT4-0 Warp-per-Seed Postmortem

原始 `warp-per-seed` 没有进入主结果。本轮将该分支显式记录为 failed/future-work branch，而不是把 baseline 冒充为 warp-per-seed。

## Evidence

- Runner 对 `--variant opt4_warp_per_seed` 返回非零状态，避免误用未实现 kernel。
- Postmortem CSV: `data/results/opt/opt4_followup/opt4_warp_per_seed_postmortem.csv`
- NCU attempt log: `logs/opt/opt4_followup/opt4_warp_per_seed_ncu.csv`

## Failure Reason

单个 target-seed IK 候选包含强串行 LM 迭代、FP64 6x6 小矩阵求解、lambda 自适应和 convergence branch。把一个候选拆给一个 warp 后，32 lanes 很难高效分工，shuffle/sync 协作成本和控制流复杂度会抵消潜在并行收益。

## Paper Usage

该分支只能作为 Discussion / Future Work，不得写成完成的性能优化。
