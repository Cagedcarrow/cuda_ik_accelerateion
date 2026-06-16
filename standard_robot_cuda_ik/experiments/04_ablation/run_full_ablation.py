#!/usr/bin/env python3
"""Full Ablation Experiment: A0 through A7 at key N values.

Runs all ablation levels to measure the performance contributions of
each optimization: constant memory, PaddedMat, register LDLT,
kernel fusion, adaptive damping, step clamp/branch align, mixed precision.

Output: ablation_full.csv
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

# Ablation level -> binary name
ABLATION_BINARIES = {
    "A0-GlobalMem":    BUILD_DIR / "standard_robot_cuda_runner_A0",
    "A1-ConstantMem":  BUILD_DIR / "standard_robot_cuda_runner_A1",
    "A2-PaddedMat":    BUILD_DIR / "standard_robot_cuda_runner_A2",
    "A3-RegLDLT":      BUILD_DIR / "standard_robot_cuda_runner_A3",
    "A4-KernelFusion": BUILD_DIR / "standard_robot_cuda_runner_A4",
    "A5-AdapDamp":     BUILD_DIR / "standard_robot_cuda_runner_A5",
    "A6-FullOptimized": BUILD_DIR / "standard_robot_cuda_runner_A6",
    "A7-MixedPrec":    BUILD_DIR / "standard_robot_cuda_runner_A7",
}

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
    out_path = OUT_DIR / "ablation_full.csv"
    rows = []

    for N in N_VALUES:
        print(f"\nN={N}:")
        for label, binary in ABLATION_BINARIES.items():
            m = run(binary, N)
            if m:
                rows.append({
                    "ablation_level": label, "N": N,
                    "kernel_time_ms_mean": m.get("kernel_time_only_ms_mean", 0),
                    "kernel_time_ms_std": m.get("kernel_time_only_ms_std", 0),
                    "e2e_time_ms": m.get("e2e_time_ms_mean", 0),
                    "throughput_targets_per_s": m.get("throughput_targets_per_s", 0),
                    "valid_throughput": m.get("valid_throughput_targets_per_s", 0),
                    "success_rate": m.get("convergence_rate_medium", 0),
                    "success_rate_loose": m.get("convergence_rate_loose", 0),
                    "success_rate_strict": m.get("convergence_rate_strict", 0),
                    "avg_pos_error_m": m.get("avg_pos_error_m", 0),
                    "avg_rot_error_rad": m.get("avg_rot_error_rad", 0),
                    "avg_iterations": m.get("avg_iterations", 0),
                })
                imp = ""
                if rows and len(rows) > 1:
                    prev_tp = rows[-2]["throughput_targets_per_s"]
                    curr_tp = rows[-1]["throughput_targets_per_s"]
                    if prev_tp > 0:
                        imp = f" (+{(curr_tp/prev_tp - 1)*100:.1f}%)"
                print(f"  {label}: {m['throughput_targets_per_s']:.0f} t/s, "
                      f"sr={m.get('convergence_rate_medium',0):.3f}{imp}")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
