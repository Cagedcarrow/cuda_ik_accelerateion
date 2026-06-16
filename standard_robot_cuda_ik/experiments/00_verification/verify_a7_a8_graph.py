#!/usr/bin/env python3
"""A7 vs A8 CUDA Graph Verification (Step 4).

Re-runs both binaries at N=100,500,1000,5000 with 30 repeats,
outputs mean±std for proper statistical reporting.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = PROJECT_ROOT / "build"
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent

CUDA_A7 = BUILD_DIR / "standard_robot_cuda_runner_A7"
CUDA_A8 = BUILD_DIR / "standard_robot_cuda_runner_A8"
N_VALUES = [100, 500, 1000, 5000]
REPEAT = 30
WARMUP = 5
ROBOT = "ur10"
SEED = 42
POS_TOL = 0.01
ROT_TOL = 0.0872664626


def run_binary(binary: Path, N: int) -> dict | None:
    targets_bin = DATA_DIR / "targets" / f"{ROBOT}_seed{SEED}_N{N}.bin"
    seeds_bin = DATA_DIR / "seeds" / f"{ROBOT}_seed{SEED}_zero_seed_N{N}.bin"
    cmd = [
        str(binary), "--targets", str(targets_bin), "--seeds", str(seeds_bin),
        "--repeat", str(REPEAT), "--warmup", str(WARMUP),
        "--pos-tol", str(POS_TOL), "--rot-tol", str(ROT_TOL), "--max-iter", "160",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"  ERROR: {r.stderr[:200]}")
        return None
    m = {}
    for line in r.stdout.strip().split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            try:
                m[k] = float(v)
            except ValueError:
                m[k] = v
    return m


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "a7_a8_graph_verification.csv"
    rows = []

    for N in N_VALUES:
        print(f"\nN={N}:", flush=True)
        for label, binary in [("A7-Direct", CUDA_A7), ("A8-Graph", CUDA_A8)]:
            m = run_binary(binary, N)
            if m:
                a7_or_a8 = "A7" if label == "A7-Direct" else "A8"
                row = {
                    "method": label, "N": N,
                    "kernel_ms_mean": m.get("kernel_time_only_ms_mean", 0),
                    "kernel_ms_std": m.get("kernel_time_only_ms_std", 0),
                    "kernel_ms_min": m.get("kernel_time_only_ms_min", 0),
                    "kernel_ms_max": m.get("kernel_time_only_ms_max", 0),
                    "kernel_ms_p50": m.get("kernel_time_only_ms_p50", 0),
                    "kernel_ms_p95": m.get("kernel_time_only_ms_p95", 0),
                    "e2e_ms_mean": m.get("e2e_time_ms_mean", 0),
                    "h2d_ms": m.get("h2d_time_ms", 0),
                    "d2h_ms_mean": m.get("d2h_time_ms_mean", 0),
                    "raw_tps": m.get("throughput_targets_per_s", 0),
                    "valid_tps": m.get("valid_throughput_targets_per_s", 0),
                    "success_rate_medium": m.get("convergence_rate_medium", 0),
                    "success_rate_strict": m.get("convergence_rate_strict", 0),
                    "avg_iterations": m.get("avg_iterations", 0),
                }
                rows.append(row)
                print(f"  {label}: {row['kernel_ms_mean']:.3f}±{row['kernel_ms_std']:.3f}ms "
                      f"→ {row['raw_tps']:.0f} t/s", flush=True)

        # Compute A7 vs A8 diff
        a7_rows = [r for r in rows if r["method"] == "A7-Direct" and r["N"] == N]
        a8_rows = [r for r in rows if r["method"] == "A8-Graph" and r["N"] == N]
        if a7_rows and a8_rows:
            a7_tps = a7_rows[0]["raw_tps"]
            a8_tps = a8_rows[0]["raw_tps"]
            diff = (a8_tps - a7_tps) / a7_tps * 100 if a7_tps > 0 else 0
            print(f"  → A8 vs A7 diff: {diff:+.2f}%", flush=True)

    # Write CSV
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV saved to {csv_path}")


if __name__ == "__main__":
    main()
