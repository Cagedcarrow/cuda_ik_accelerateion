# Solver Failure Isolation Verification

## Purpose

验证 `benchmark/run_all.py` 在单个 solver 失败时能够：

1. 生成结构化错误日志；
2. 保持进程继续执行其他 solver；
3. 仍然输出 Markdown 汇总文件。

## Test Command

```bash
python3 standard_robot_cuda_ik/benchmark/run_all.py \
  --robot badrobot \
  --seed 42 \
  --N 100 \
  --repeat 1 \
  --solver all \
  --seed-strategy zero_seed
```

## Observed Behavior

- 五个 solver 都因为缺失 `badrobot` 数据资产而失败。
- `run_all.py` 没有在第一个失败处中止。
- 每个 solver 都写出了独立 JSON 错误日志。
- 汇总 Markdown 仍然生成。

## Evidence Files

- Markdown:
  `standard_robot_cuda_ik/data/results/badrobot_all_N100_seed42_repeat1_zero_seed.md`
- Error logs:
  - `standard_robot_cuda_ik/data/results/errors/badrobot_cuda_N100_seed42_repeat1_zero_seed_error.json`
  - `standard_robot_cuda_ik/data/results/errors/badrobot_curobo_N100_seed42_repeat1_zero_seed_error.json`
  - `standard_robot_cuda_ik/data/results/errors/badrobot_pyroki_N100_seed42_repeat1_zero_seed_error.json`
  - `standard_robot_cuda_ik/data/results/errors/badrobot_kdl_N100_seed42_repeat1_zero_seed_error.json`
  - `standard_robot_cuda_ik/data/results/errors/badrobot_numeric_dls_N100_seed42_repeat1_zero_seed_error.json`

## Conclusion

当前 `run_all.py` 已满足“solver 失败时生成结构化错误日志，不影响其他 solver 继续”的集成验证要求。
