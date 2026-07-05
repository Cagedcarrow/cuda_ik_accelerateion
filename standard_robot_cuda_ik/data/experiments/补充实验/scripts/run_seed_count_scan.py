#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from common_metrics import COMMON_FIELDS, EXPERIMENTS, INPUTS, RESULTS, RUNNER, read_csv, row_from_runner_summary, write_csv

N_VALUES = [100, 500, 1000]
K_VALUES = [1, 2, 4, 8, 16]
WARMUP = 10
REPEAT = 30
MAX_ITER = 60


def make_seed_file(n: int, k: int) -> Path:
    out = INPUTS / "seed_scan" / f"seeds_N{n}_K{k}_q_f64.raw"
    if out.exists():
        return out
    src = EXPERIMENTS / "inputs" / f"seeds_N{n}_K16_q_f64.raw"
    seeds = np.fromfile(src, dtype=np.float64).reshape(n, 16, 6)[:, :k, :].copy()
    out.parent.mkdir(parents=True, exist_ok=True)
    seeds.astype(np.float64).tofile(out)
    return out


def run_one(n: int, k: int) -> dict[str, object]:
    target = EXPERIMENTS / "inputs" / f"targets_N{n}_T4x4_f64.raw"
    seeds = make_seed_file(n, k)
    best = RESULTS / f"seed_scan_best_N{n}_K{k}.csv"
    summary = RESULTS / f"seed_scan_summary_N{n}_K{k}.csv"
    timing = RESULTS / f"seed_scan_timing_N{n}_K{k}.csv"
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
            "--N", str(n),
            "--K", str(k),
            "--max-iter", str(MAX_ITER),
            "--repeat", str(REPEAT),
            "--warmup", str(WARMUP),
            "--best-csv", str(best),
            "--summary-csv", str(summary),
            "--timing-csv", str(timing),
        ]
        print("running", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=RUNNER.parents[1])
    row = read_csv(summary)[0]
    normalized = row_from_runner_summary(row, f"OPT4C-K{k}", "seed_count_scan", "random_reachable")
    normalized["K"] = k
    return normalized


def main() -> int:
    if not RUNNER.exists():
        raise FileNotFoundError(f"runner not found: {RUNNER}")
    rows = []
    for n in N_VALUES:
        for k in K_VALUES:
            rows.append(run_one(n, k))
    out = RESULTS / "seed_count_scan.csv"
    write_csv(out, rows, COMMON_FIELDS)
    print(f"wrote {out} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

