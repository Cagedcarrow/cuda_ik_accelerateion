#!/usr/bin/env python3
from __future__ import annotations

import subprocess

from common_metrics import COMMON_FIELDS, EXPERIMENTS, RESULTS, RUNNER, read_csv, row_from_runner_summary, write_csv

N = 1000
K = 16
MAX_ITERS = [20, 40, 60, 80, 100]
WARMUP = 10
REPEAT = 30


def run_one(max_iter: int) -> dict[str, object]:
    target = EXPERIMENTS / "inputs" / f"targets_N{N}_T4x4_f64.raw"
    seeds = EXPERIMENTS / "inputs" / f"seeds_N{N}_K{K}_q_f64.raw"
    best = RESULTS / f"lm_iter_best_N{N}_K{K}_I{max_iter}.csv"
    summary = RESULTS / f"lm_iter_summary_N{N}_K{K}_I{max_iter}.csv"
    timing = RESULTS / f"lm_iter_timing_N{N}_K{K}_I{max_iter}.csv"
    if not summary.exists():
        cmd = [
            str(RUNNER),
            "--mode", "v4_static",
            "--variant", "opt4c_block_target",
            "--limit-gradient", "analytic",
            "--graph-mode", "off",
            "--precision-mode", "fp64",
            "--fallback-mode", "none",
            "--targets", str(target),
            "--seeds", str(seeds),
            "--N", str(N),
            "--K", str(K),
            "--max-iter", str(max_iter),
            "--repeat", str(REPEAT),
            "--warmup", str(WARMUP),
            "--best-csv", str(best),
            "--summary-csv", str(summary),
            "--timing-csv", str(timing),
        ]
        print("running lm_iter", max_iter, flush=True)
        subprocess.run(cmd, check=True, cwd=RUNNER.parents[1])
    row = read_csv(summary)[0]
    normalized = row_from_runner_summary(row, "OPT4C-K16", "lm_iter_scan", "random_reachable")
    normalized["max_iter"] = max_iter
    normalized["notes"] = "max_iter scan"
    return normalized


def main() -> int:
    rows = [run_one(max_iter) for max_iter in MAX_ITERS]
    out = RESULTS / "lm_iter_scan.csv"
    write_csv(out, rows, COMMON_FIELDS)
    print(f"wrote {out} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

