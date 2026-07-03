#!/usr/bin/env python3
"""Main Experiment Runner: Full benchmark matrix for paper revision.

Runs all GPU methods at all N values with 30 repeats, collecting:
- Three-tier timing (kernel-only, GPU stream, E2E)
- Multi-threshold convergence (loose/medium/strict)
- Per-target error distribution
- Statistical significance

Output CSVs:
  performance_summary.csv  — aggregated metrics per (method, graph, N)
  error_summary.csv        — per-target errors for error distribution analysis
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Add benchmark directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
from common import (
    THRESHOLD_TIERS, compute_pose_error, compute_error_percentiles,
    load_robot_records, load_seed_values, mark_convergence,
)
from bench_curobo import run_curobo_benchmark

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = PROJECT_ROOT / "build"
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent

# CUDA binary paths
CUDA_BINARY = BUILD_DIR / "standard_robot_cuda_runner_A7"  # Mixed precision
CUDA_BINARY_A8 = BUILD_DIR / "standard_robot_cuda_runner_A8"  # + CUDA Graph

# Experiment matrix
N_VALUES = [100, 500, 1000, 4000, 5000, 10000]
REPEAT_COUNT = 30
WARMUP_COUNT = 3
ROBOT = "ur10"
SEED = 42
SEED_STRATEGY = "zero_seed"
POS_TOL = THRESHOLD_TIERS["medium"]["pos"]   # 0.01
ROT_TOL = THRESHOLD_TIERS["medium"]["rot"]   # 0.0873


def run_cuda_binary(binary: Path, N: int, error_log: Path | None = None) -> dict:
    """Run CUDA binary and parse its key=value stdout output."""
    targets_bin = DATA_DIR / "targets" / f"{ROBOT}_seed{SEED}_N{N}.bin"
    seeds_bin = DATA_DIR / "seeds" / f"{ROBOT}_seed{SEED}_{SEED_STRATEGY}_N{N}.bin"

    cmd = [
        str(binary),
        "--targets", str(targets_bin),
        "--seeds", str(seeds_bin),
        "--repeat", str(REPEAT_COUNT),
        "--warmup", str(WARMUP_COUNT),
        "--pos-tol", str(POS_TOL),
        "--rot-tol", str(ROT_TOL),
        "--max-iter", "160",
    ]
    if error_log:
        cmd += ["--error-log", str(error_log)]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  ERROR running {binary.name} N={N}: {result.stderr}")
        return {}

    metrics = {}
    for line in result.stdout.strip().split("\n"):
        if "=" in line:
            key, val = line.split("=", 1)
            try:
                metrics[key] = float(val)
            except ValueError:
                metrics[key] = val
    return metrics


def run_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    perf_path = OUT_DIR / "performance_summary.csv"
    error_dir = OUT_DIR / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)

    perf_rows = []

    for N in N_VALUES:
        print(f"\n{'='*60}")
        print(f"N = {N}")
        print(f"{'='*60}")

        # --- CUDA-Mixed (A7, no Graph) ---
        print("  CUDA-Mixed (A7)...")
        error_log = error_dir / f"cuda_mixed_N{N}_errors.csv"
        metrics = run_cuda_binary(CUDA_BINARY, N, error_log)
        if metrics:
            perf_rows.append({
                "method": "CUDA-Mixed", "graph": "N/A", "N": N,
                "timing_type": "kernel",
                "time_ms_mean": metrics.get("kernel_time_only_ms_mean", 0),
                "time_ms_std": metrics.get("kernel_time_only_ms_std", 0),
                "time_ms_min": metrics.get("kernel_time_only_ms_min", 0),
                "time_ms_max": metrics.get("kernel_time_only_ms_max", 0),
                "time_ms_p50": metrics.get("kernel_time_only_ms_p50", 0),
                "time_ms_p95": metrics.get("kernel_time_only_ms_p95", 0),
                "h2d_time_ms": metrics.get("h2d_time_ms", 0),
                "d2h_time_ms_mean": metrics.get("d2h_time_ms_mean", 0),
                "e2e_time_ms": metrics.get("e2e_time_ms_mean", 0),
                "raw_targets_per_s": metrics.get("throughput_targets_per_s", 0),
                "valid_targets_per_s": metrics.get("valid_throughput_targets_per_s", 0),
                "success_rate": metrics.get("convergence_rate_medium", 0),
                "success_rate_loose": metrics.get("convergence_rate_loose", 0),
                "success_rate_strict": metrics.get("convergence_rate_strict", 0),
                "avg_pos_error_m": metrics.get("avg_pos_error_m", 0),
                "pos_error_p50_m": metrics.get("pos_error_p50_m", 0),
                "pos_error_p95_m": metrics.get("pos_error_p95_m", 0),
                "avg_rot_error_rad": metrics.get("avg_rot_error_rad", 0),
                "rot_error_p50_rad": metrics.get("rot_error_p50_rad", 0),
                "rot_error_p95_rad": metrics.get("rot_error_p95_rad", 0),
                "avg_iterations": metrics.get("avg_iterations", 0),
            })
            print(f"    throughput={metrics.get('throughput_targets_per_s',0):.0f} t/s, "
                  f"success={metrics.get('convergence_rate_medium',0):.3f}, "
                  f"e2e={metrics.get('e2e_time_ms_mean',0):.3f}ms")

        # --- CUDA-Mixed Graph (A8) ---
        print("  CUDA-Mixed-Graph (A8)...")
        error_log_a8 = error_dir / f"cuda_mixed_graph_N{N}_errors.csv"
        metrics_a8 = run_cuda_binary(CUDA_BINARY_A8, N, error_log_a8)
        if metrics_a8:
            perf_rows.append({
                "method": "CUDA-Mixed-Graph", "graph": "On", "N": N,
                "timing_type": "kernel",
                "time_ms_mean": metrics_a8.get("kernel_time_only_ms_mean", 0),
                "time_ms_std": metrics_a8.get("kernel_time_only_ms_std", 0),
                "time_ms_min": metrics_a8.get("kernel_time_only_ms_min", 0),
                "time_ms_max": metrics_a8.get("kernel_time_only_ms_max", 0),
                "time_ms_p50": metrics_a8.get("kernel_time_only_ms_p50", 0),
                "time_ms_p95": metrics_a8.get("kernel_time_only_ms_p95", 0),
                "h2d_time_ms": metrics_a8.get("h2d_time_ms", 0),
                "d2h_time_ms_mean": metrics_a8.get("d2h_time_ms_mean", 0),
                "e2e_time_ms": metrics_a8.get("e2e_time_ms_mean", 0),
                "raw_targets_per_s": metrics_a8.get("throughput_targets_per_s", 0),
                "valid_targets_per_s": metrics_a8.get("valid_throughput_targets_per_s", 0),
                "success_rate": metrics_a8.get("convergence_rate_medium", 0),
                "success_rate_loose": metrics_a8.get("convergence_rate_loose", 0),
                "success_rate_strict": metrics_a8.get("convergence_rate_strict", 0),
                "avg_pos_error_m": metrics_a8.get("avg_pos_error_m", 0),
                "pos_error_p50_m": metrics_a8.get("pos_error_p50_m", 0),
                "pos_error_p95_m": metrics_a8.get("pos_error_p95_m", 0),
                "avg_rot_error_rad": metrics_a8.get("avg_rot_error_rad", 0),
                "rot_error_p50_rad": metrics_a8.get("rot_error_p50_rad", 0),
                "rot_error_p95_rad": metrics_a8.get("rot_error_p95_rad", 0),
                "avg_iterations": metrics_a8.get("avg_iterations", 0),
            })
            print(f"    throughput={metrics_a8.get('throughput_targets_per_s',0):.0f} t/s, "
                  f"e2e={metrics_a8.get('e2e_time_ms_mean',0):.3f}ms")

        # --- cuRobo-NoGraph ---
        print("  cuRobo-NoGraph...")
        try:
            error_log_cn = error_dir / f"curobo_nograph_N{N}_errors.csv"
            r_cn = run_curobo_benchmark(
                robot=ROBOT, seed=SEED, N=N, repeat=REPEAT_COUNT,
                seed_strategy=SEED_STRATEGY,
                pos_tol=POS_TOL, rot_tol=ROT_TOL,
                use_cuda_graph=False, warmup=WARMUP_COUNT,
                error_log_path=str(error_log_cn),
            )
            perf_rows.append({
                "method": "cuRobo", "graph": "Off", "N": N,
                "timing_type": "gpu_stream",
                "time_ms_mean": float(np.mean(r_cn.gpu_stream_time_ms)) if r_cn.gpu_stream_time_ms else 0,
                "time_ms_std": float(np.std(r_cn.gpu_stream_time_ms)) if r_cn.gpu_stream_time_ms else 0,
                "time_ms_p50": r_cn.p50_latency_ms,
                "time_ms_p95": r_cn.p95_latency_ms,
                "e2e_time_ms": float(np.mean(r_cn.host_api_total_time_ms)),
                "raw_targets_per_s": r_cn.throughput_targets_per_s,
                "valid_targets_per_s": r_cn.valid_throughput_targets_per_s,
                "success_rate": r_cn.convergence_rate,
                "success_rate_loose": r_cn.convergence_rate_loose,
                "success_rate_strict": r_cn.convergence_rate_strict,
                "avg_pos_error_m": r_cn.avg_pos_error_m,
                "pos_error_p50_m": r_cn.pos_error_p50_m,
                "pos_error_p95_m": r_cn.pos_error_p95_m,
                "avg_rot_error_rad": r_cn.avg_rot_error_rad,
                "rot_error_p50_rad": r_cn.rot_error_p50_rad,
                "rot_error_p95_rad": r_cn.rot_error_p95_rad,
            })
            print(f"    throughput={r_cn.throughput_targets_per_s:.0f} t/s, "
                  f"success={r_cn.convergence_rate:.3f}")
        except Exception as e:
            print(f"    FAILED: {e}")

        # --- cuRobo-Graph ---
        print("  cuRobo-Graph...")
        try:
            error_log_cg = error_dir / f"curobo_graph_N{N}_errors.csv"
            r_cg = run_curobo_benchmark(
                robot=ROBOT, seed=SEED, N=N, repeat=REPEAT_COUNT,
                seed_strategy=SEED_STRATEGY,
                pos_tol=POS_TOL, rot_tol=ROT_TOL,
                use_cuda_graph=True, warmup=WARMUP_COUNT,
                error_log_path=str(error_log_cg),
            )
            perf_rows.append({
                "method": "cuRobo", "graph": "On", "N": N,
                "timing_type": "gpu_stream",
                "time_ms_mean": float(np.mean(r_cg.gpu_stream_time_ms)) if r_cg.gpu_stream_time_ms else 0,
                "time_ms_std": float(np.std(r_cg.gpu_stream_time_ms)) if r_cg.gpu_stream_time_ms else 0,
                "time_ms_p50": r_cg.p50_latency_ms,
                "time_ms_p95": r_cg.p95_latency_ms,
                "e2e_time_ms": float(np.mean(r_cg.host_api_total_time_ms)),
                "raw_targets_per_s": r_cg.throughput_targets_per_s,
                "valid_targets_per_s": r_cg.valid_throughput_targets_per_s,
                "success_rate": r_cg.convergence_rate,
                "success_rate_loose": r_cg.convergence_rate_loose,
                "success_rate_strict": r_cg.convergence_rate_strict,
                "avg_pos_error_m": r_cg.avg_pos_error_m,
                "pos_error_p50_m": r_cg.pos_error_p50_m,
                "pos_error_p95_m": r_cg.pos_error_p95_m,
                "avg_rot_error_rad": r_cg.avg_rot_error_rad,
                "rot_error_p50_rad": r_cg.rot_error_p50_rad,
                "rot_error_p95_rad": r_cg.rot_error_p95_rad,
            })
            print(f"    throughput={r_cg.throughput_targets_per_s:.0f} t/s, "
                  f"success={r_cg.convergence_rate:.3f}")
        except Exception as e:
            print(f"    FAILED: {e}")

        # Write incremental CSV after each N (resilience against crashes)
        if perf_rows:
            with open(perf_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=perf_rows[0].keys())
                writer.writeheader()
                writer.writerows(perf_rows)

    print(f"\n{'='*60}")
    print(f"Done. {len(perf_rows)} rows written to {perf_path}")
    return perf_rows


if __name__ == "__main__":
    run_all()
