#!/usr/bin/env python3
"""Full Range Scaling Experiment: N=100 to 10000 sweep.

Runs CUDA-Mixed (A7) and cuRobo-NoGraph across all 12 N values
to analyze batch scalability, linearity, and degradation points.

Output: full_range_scaling.csv
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
from common import THRESHOLD_TIERS
from bench_curobo import run_curobo_benchmark

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = PROJECT_ROOT / "build"
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent

CUDA_BINARY = BUILD_DIR / "standard_robot_cuda_runner_A7"
N_VALUES = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
REPEAT = 10  # reduced for time
WARMUP = 3
ROBOT = "ur10"
SEED = 42
POS_TOL = THRESHOLD_TIERS["medium"]["pos"]
ROT_TOL = THRESHOLD_TIERS["medium"]["rot"]


def run_cuda(N: int) -> dict | None:
    targets_bin = DATA_DIR / "targets" / f"{ROBOT}_seed{SEED}_N{N}.bin"
    seeds_bin = DATA_DIR / "seeds" / f"{ROBOT}_seed{SEED}_zero_seed_N{N}.bin"
    cmd = [
        str(CUDA_BINARY), "--targets", str(targets_bin), "--seeds", str(seeds_bin),
        "--repeat", str(REPEAT), "--warmup", str(WARMUP),
        "--pos-tol", str(POS_TOL), "--rot-tol", str(ROT_TOL), "--max-iter", "160",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        return None
    metrics = {}
    for line in r.stdout.strip().split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            try:
                metrics[k] = float(v)
            except ValueError:
                metrics[k] = v
    return metrics


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "full_range_scaling.csv"
    rows = []

    for N in N_VALUES:
        print(f"N={N}: ", end="", flush=True)

        # CUDA-Mixed
        m = run_cuda(N)
        if m:
            rows.append({
                "method": "CUDA-Mixed", "N": N,
                "kernel_time_ms": m.get("kernel_time_only_ms_mean", 0),
                "kernel_time_ms_std": m.get("kernel_time_only_ms_std", 0),
                "e2e_time_ms": m.get("e2e_time_ms_mean", 0),
                "throughput_targets_per_s": m.get("throughput_targets_per_s", 0),
                "valid_throughput_targets_per_s": m.get("valid_throughput_targets_per_s", 0),
                "success_rate": m.get("convergence_rate_medium", 0),
                "avg_iterations": m.get("avg_iterations", 0),
            })
            print(f"CUDA={m['throughput_targets_per_s']:.0f}t/s ", end="", flush=True)

        # cuRobo-NoGraph
        try:
            r = run_curobo_benchmark(
                robot=ROBOT, seed=SEED, N=N, repeat=REPEAT,
                use_cuda_graph=False, warmup=WARMUP,
                pos_tol=POS_TOL, rot_tol=ROT_TOL,
            )
            rows.append({
                "method": "cuRobo-NoGraph", "N": N,
                "gpu_stream_time_ms": float(np.mean(r.gpu_stream_time_ms)) if r.gpu_stream_time_ms else 0,
                "host_time_ms": float(np.mean(r.host_api_total_time_ms)),
                "throughput_targets_per_s": r.throughput_targets_per_s,
                "valid_throughput_targets_per_s": r.valid_throughput_targets_per_s,
                "success_rate": r.convergence_rate,
            })
            print(f"cuRobo={r.throughput_targets_per_s:.0f}t/s", flush=True)
        except Exception as e:
            print(f"cuRobo FAILED: {e}", flush=True)

        # Incremental save
        if rows:
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
