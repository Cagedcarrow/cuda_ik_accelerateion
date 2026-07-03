#!/usr/bin/env python3
"""CUDA Graph Comparison: A7 (direct launch) vs A8 (CUDA Graph replay).

Measures the overhead/benefit of CUDA Graph for our single-kernel design.

Output: cuda_graph_comparison.csv
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
WARMUP = 3
ROBOT = "ur10"
SEED = 42
POS_TOL = 0.01
ROT_TOL = 0.0872664626


def run(binary: Path, N: int) -> dict | None:
    targets_bin = DATA_DIR / "targets" / f"{ROBOT}_seed{SEED}_N{N}.bin"
    seeds_bin = DATA_DIR / "seeds" / f"{ROBOT}_seed{SEED}_zero_seed_N{N}.bin"
    cmd = [
        str(binary), "--targets", str(targets_bin), "--seeds", str(seeds_bin),
        "--repeat", str(REPEAT), "--warmup", str(WARMUP),
        "--pos-tol", str(POS_TOL), "--rot-tol", str(ROT_TOL), "--max-iter", "160",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
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
    out_path = OUT_DIR / "cuda_graph_comparison.csv"
    rows = []

    for N in N_VALUES:
        print(f"N={N}: ", end="", flush=True)
        for label, binary in [("A7-Direct", CUDA_A7), ("A8-Graph", CUDA_A8)]:
            m = run(binary, N)
            if m:
                rows.append({
                    "method": label, "N": N,
                    "kernel_time_ms_mean": m.get("kernel_time_only_ms_mean", 0),
                    "kernel_time_ms_std": m.get("kernel_time_only_ms_std", 0),
                    "kernel_time_ms_min": m.get("kernel_time_only_ms_min", 0),
                    "kernel_time_ms_max": m.get("kernel_time_only_ms_max", 0),
                    "e2e_time_ms": m.get("e2e_time_ms_mean", 0),
                    "h2d_time_ms": m.get("h2d_time_ms", 0),
                    "d2h_time_ms_mean": m.get("d2h_time_ms_mean", 0),
                    "throughput_targets_per_s": m.get("throughput_targets_per_s", 0),
                    "valid_throughput": m.get("valid_throughput_targets_per_s", 0),
                    "success_rate": m.get("convergence_rate_medium", 0),
                })
                print(f"{label}={m['throughput_targets_per_s']:.0f}t/s ", end="", flush=True)
        print()

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
