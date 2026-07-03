#!/usr/bin/env python3
"""cuRobo CUDA Graph On/Off Comparison.

Runs cuRobo with and without CUDA Graph at key N values,
measuring GPU stream time, host time, and convergence.

Output: curobo_graph_comparison.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
from common import THRESHOLD_TIERS
from bench_curobo import run_curobo_benchmark

OUT_DIR = Path(__file__).resolve().parent
N_VALUES = [100, 500, 1000, 4000, 5000, 10000]
REPEAT = 30
WARMUP = 3
ROBOT = "ur10"
SEED = 42
POS_TOL = THRESHOLD_TIERS["medium"]["pos"]
ROT_TOL = THRESHOLD_TIERS["medium"]["rot"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "curobo_graph_comparison.csv"
    rows = []

    for N in N_VALUES:
        print(f"N={N}: ", end="", flush=True)
        for use_graph in [False, True]:
            label = "cuRobo-Graph" if use_graph else "cuRobo-NoGraph"
            try:
                r = run_curobo_benchmark(
                    robot=ROBOT, seed=SEED, N=N, repeat=REPEAT,
                    use_cuda_graph=use_graph, warmup=WARMUP,
                    pos_tol=POS_TOL, rot_tol=ROT_TOL,
                )
                gpu_mean = float(np.mean(r.gpu_stream_time_ms)) if r.gpu_stream_time_ms else 0
                gpu_std = float(np.std(r.gpu_stream_time_ms)) if r.gpu_stream_time_ms else 0
                host_mean = float(np.mean(r.host_api_total_time_ms))
                rows.append({
                    "method": label, "N": N,
                    "use_cuda_graph": use_graph,
                    "gpu_stream_time_ms_mean": gpu_mean,
                    "gpu_stream_time_ms_std": gpu_std,
                    "host_time_ms_mean": host_mean,
                    "host_time_ms_std": r.std_host_api_total_ms,
                    "throughput_targets_per_s": r.throughput_targets_per_s,
                    "valid_throughput_targets_per_s": r.valid_throughput_targets_per_s,
                    "success_rate": r.convergence_rate,
                    "success_rate_loose": r.convergence_rate_loose,
                    "success_rate_strict": r.convergence_rate_strict,
                    "pos_error_p50_m": r.pos_error_p50_m,
                    "pos_error_p95_m": r.pos_error_p95_m,
                    "rot_error_p50_rad": r.rot_error_p50_rad,
                    "rot_error_p95_rad": r.rot_error_p95_rad,
                })
                print(f"{label}={r.throughput_targets_per_s:.0f}t/s(sr={r.convergence_rate:.2f}) ",
                      end="", flush=True)
            except Exception as e:
                print(f"{label}=FAILED({e}) ", end="", flush=True)
        print()

    if rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
