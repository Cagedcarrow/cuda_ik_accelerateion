#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys

import numpy as np

from common_metrics import COMMON_FIELDS, INPUTS, RESULTS, ROOT, SUPP, read_csv, row_from_runner_summary, write_csv

sys.path.insert(0, str(ROOT / "tools"))
from robot_model import load_robot_model  # noqa: E402

N_VALUES = [100, 500, 1000]
K_VALUES = [1, 16]
CONFIGS = [("BarrierON", 0.03), ("BarrierOFF", 0.0)]
WARMUP = 10
REPEAT = 30
MAX_ITER = 60
RUNNER = SUPP / "runner_variant" / "build" / "limit_weight_runner"
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def joint_limits() -> tuple[np.ndarray, np.ndarray]:
    model = load_robot_model(
        ROOT / "urdf" / "ur10_official.urdf",
        base_link="base_link",
        tip_link="tool0",
        expected_active_joint_names=JOINT_NAMES,
    )
    arr = model.limits_array().reshape(6, 2)
    return arr[:, 0], arr[:, 1]


def violation_count(best_csv) -> int:
    qmin, qmax = joint_limits()
    count = 0
    for row in read_csv(best_csv):
        q = np.array([float(row[f"q{i}"]) for i in range(6)], dtype=np.float64)
        if np.any(q < qmin - 1e-9) or np.any(q > qmax + 1e-9):
            count += 1
    return count


def run_one(n: int, k: int, label: str, weight: float) -> dict[str, object]:
    target = INPUTS / "near_limit" / f"targets_N{n}_T4x4_f64.raw"
    seeds = INPUTS / "near_limit" / f"seeds_N{n}_K{k}_q_f64.raw"
    best = RESULTS / f"near_limit_best_N{n}_K{k}_{label}.csv"
    summary = RESULTS / f"near_limit_summary_N{n}_K{k}_{label}.csv"
    timing = RESULTS / f"near_limit_timing_N{n}_K{k}_{label}.csv"
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
            "--limit-weight", str(weight),
            "--best-csv", str(best),
            "--summary-csv", str(summary),
            "--timing-csv", str(timing),
        ]
        print("running near_limit", "N", n, "K", k, label, flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
    row = read_csv(summary)[0]
    normalized = row_from_runner_summary(
        row,
        f"OPT4C-K{k}-{label}",
        "near_limit_barrier",
        "near_limit",
        wlimit=str(weight),
        barrier_enabled="true" if weight > 0.0 else "false",
    )
    normalized["K"] = k
    normalized["joint_violation_count"] = violation_count(best)
    normalized["notes"] = "limit-weight runner variant under supplementary experiment directory"
    return normalized


def main() -> int:
    if not RUNNER.exists():
        raise FileNotFoundError(f"missing {RUNNER}; run prepare_limit_weight_runner.py")
    rows = []
    for n in N_VALUES:
        for k in K_VALUES:
            for label, weight in CONFIGS:
                rows.append(run_one(n, k, label, weight))
    out = RESULTS / "near_limit_barrier_summary.csv"
    write_csv(out, rows, COMMON_FIELDS)
    print(f"wrote {out} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

