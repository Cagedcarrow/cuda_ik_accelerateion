#!/usr/bin/env python3
from __future__ import annotations

import subprocess

from common_metrics import COMMON_FIELDS, EXPERIMENTS, INPUTS, RESULTS, ROOT, SUPP, read_csv, row_from_runner_summary, write_csv

N = 1000
K = 16
WEIGHTS = [0.0, 0.005, 0.01, 0.03, 0.05, 0.1]
TARGETS = {
    "random_reachable": EXPERIMENTS / "inputs" / f"targets_N{N}_T4x4_f64.raw",
    "near_limit": INPUTS / "near_limit" / f"targets_N{N}_T4x4_f64.raw",
}
SEEDS = {
    "random_reachable": EXPERIMENTS / "inputs" / f"seeds_N{N}_K{K}_q_f64.raw",
    "near_limit": INPUTS / "near_limit" / f"seeds_N{N}_K{K}_q_f64.raw",
}
WARMUP = 10
REPEAT = 30
MAX_ITER = 60
RUNNER = SUPP / "runner_variant" / "build" / "limit_weight_runner"


def label_weight(weight: float) -> str:
    return str(weight).replace(".", "p")


def run_one(target_type: str, weight: float) -> dict[str, object]:
    label = label_weight(weight)
    best = RESULTS / f"barrier_scan_best_{target_type}_w{label}.csv"
    summary = RESULTS / f"barrier_scan_summary_{target_type}_w{label}.csv"
    timing = RESULTS / f"barrier_scan_timing_{target_type}_w{label}.csv"
    if not summary.exists():
        cmd = [
            str(RUNNER),
            "--mode", "v4_static",
            "--variant", "opt4c_block_target",
            "--limit-gradient", "analytic",
            "--graph-mode", "off",
            "--precision-mode", "fp64",
            "--fallback-mode", "none",
            "--targets", str(TARGETS[target_type]),
            "--seeds", str(SEEDS[target_type]),
            "--N", str(N),
            "--K", str(K),
            "--max-iter", str(MAX_ITER),
            "--repeat", str(REPEAT),
            "--warmup", str(WARMUP),
            "--limit-weight", str(weight),
            "--best-csv", str(best),
            "--summary-csv", str(summary),
            "--timing-csv", str(timing),
        ]
        print("running barrier weight", target_type, weight, flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
    row = read_csv(summary)[0]
    normalized = row_from_runner_summary(
        row,
        "OPT4C-K16",
        "barrier_weight_scan",
        target_type,
        wlimit=str(weight),
        barrier_enabled="true" if weight > 0.0 else "false",
    )
    normalized["notes"] = "limit-weight runner variant under supplementary experiment directory"
    return normalized


def main() -> int:
    if not RUNNER.exists():
        raise FileNotFoundError(f"missing {RUNNER}; run prepare_limit_weight_runner.py")
    rows = []
    for target_type in TARGETS:
        for weight in WEIGHTS:
            rows.append(run_one(target_type, weight))
    out = RESULTS / "barrier_weight_scan.csv"
    write_csv(out, rows, COMMON_FIELDS)
    print(f"wrote {out} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

