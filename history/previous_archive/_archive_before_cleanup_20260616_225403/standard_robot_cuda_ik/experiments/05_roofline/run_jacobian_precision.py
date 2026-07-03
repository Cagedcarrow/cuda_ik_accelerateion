#!/usr/bin/env python3
"""Jacobian Precision Validation Experiment.

Compares FP32 vs FP64 numerical Jacobian accuracy at different
finite-difference epsilon values (1e-4 to 1e-7).

Output: jacobian_precision_summary.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from robot_model import load_robot_model

URDF = Path(__file__).resolve().parents[2] / "urdf" / "ur10_official.urdf"
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# UR10 joint limits (radians)
JOINT_LIMITS = np.array([
    [-6.2832, 6.2832],
    [-2.0944, 2.0944],
    [-3.1416, 3.1416],
    [-6.2832, 6.2832],
    [-3.1416, 3.1416],
    [-6.2832, 6.2832],
])

EPSILONS = [1e-4, 1e-5, 1e-6, 1e-7]
NUM_SAMPLES = 500
SEED = 42


def fk_fp64(model, q: np.ndarray) -> np.ndarray:
    """Forward kinematics in FP64. Returns 4x4 transform matrix."""
    T = model.fk(q)
    return T.reshape(4, 4)


def numerical_jacobian_fp64(model, q: np.ndarray, eps: float) -> np.ndarray:
    """Compute 6x6 numerical Jacobian in FP64 using central differences."""
    J = np.zeros((6, 6), dtype=np.float64)
    T_nom = fk_fp64(model, q)
    p_nom = T_nom[:3, 3].copy()
    R_nom = T_nom[:3, :3].copy()

    for j in range(6):
        q_plus = q.copy().astype(np.float64)
        q_plus[j] += eps
        T_plus = fk_fp64(model, q_plus)
        p_plus = T_plus[:3, 3]
        R_diff = T_plus[:3, :3] @ R_nom.T
        # Small-angle approximation for angular error
        w = np.array([
            R_diff[2, 1] - R_diff[1, 2],
            R_diff[0, 2] - R_diff[2, 0],
            R_diff[1, 0] - R_diff[0, 1],
        ]) * 0.5

        q_minus = q.copy().astype(np.float64)
        q_minus[j] -= eps
        T_minus = fk_fp64(model, q_minus)
        p_minus = T_minus[:3, 3]
        R_diff_m = T_minus[:3, :3] @ R_nom.T
        w_m = np.array([
            R_diff_m[2, 1] - R_diff_m[1, 2],
            R_diff_m[0, 2] - R_diff_m[2, 0],
            R_diff_m[1, 0] - R_diff_m[0, 1],
        ]) * 0.5

        J[:3, j] = (p_plus - p_minus) / (2.0 * eps)
        J[3:, j] = (w - w_m) / (2.0 * eps)

    return J


def numerical_jacobian_fp32(model, q: np.ndarray, eps: float) -> np.ndarray:
    """Compute 6x6 numerical Jacobian in FP32 using central differences.
    All internal computations done in float32, result cast back to float64
    for error comparison.
    """
    q32 = q.astype(np.float32)
    eps32 = np.float32(eps)
    J32 = np.zeros((6, 6), dtype=np.float32)

    T_nom = fk_fp64(model, q32.astype(np.float64))
    p_nom = T_nom[:3, 3].astype(np.float32)
    R_nom = T_nom[:3, :3].astype(np.float32)

    for j in range(6):
        q_plus = q32.copy()
        q_plus[j] += eps32
        T_plus = fk_fp64(model, q_plus.astype(np.float64))
        p_plus = T_plus[:3, 3].astype(np.float32)
        R_plus = T_plus[:3, :3].astype(np.float32)

        q_minus = q32.copy()
        q_minus[j] -= eps32
        T_minus = fk_fp64(model, q_minus.astype(np.float64))
        p_minus = T_minus[:3, 3].astype(np.float32)
        R_minus = T_minus[:3, :3].astype(np.float32)

        # Position Jacobian
        J32[:3, j] = (p_plus - p_minus) / (2.0 * eps32)

        # Rotation Jacobian (small-angle approximation in FP32)
        R_diff_p = np.dot(R_plus, R_nom.T)
        w_p = np.array([
            R_diff_p[2, 1] - R_diff_p[1, 2],
            R_diff_p[0, 2] - R_diff_p[2, 0],
            R_diff_p[1, 0] - R_diff_p[0, 1],
        ], dtype=np.float32) * np.float32(0.5)

        R_diff_m = np.dot(R_minus, R_nom.T)
        w_m = np.array([
            R_diff_m[2, 1] - R_diff_m[1, 2],
            R_diff_m[0, 2] - R_diff_m[2, 0],
            R_diff_m[1, 0] - R_diff_m[0, 1],
        ], dtype=np.float32) * np.float32(0.5)

        J32[3:, j] = (w_p - w_m) / (2.0 * eps32)

    return J32.astype(np.float64)


def sample_joint_configs(n: int, seed: int) -> np.ndarray:
    """Sample n random joint configurations within UR10 limits."""
    rng = np.random.default_rng(seed)
    configs = np.zeros((n, 6), dtype=np.float64)
    for j in range(6):
        lo, hi = JOINT_LIMITS[j]
        configs[:, j] = rng.uniform(lo, hi, size=n)
    return configs


def relative_frobenius_error(J_fp32: np.ndarray, J_fp64: np.ndarray) -> float:
    """||J_FP32 - J_FP64||_F / ||J_FP64||_F"""
    diff_norm = np.linalg.norm(J_fp32 - J_fp64, 'fro')
    ref_norm = np.linalg.norm(J_fp64, 'fro')
    if ref_norm < 1e-30:
        return 0.0
    return float(diff_norm / ref_norm)


def main():
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "jacobian_precision_summary.csv"

    print(f"Loading UR10 model from {URDF}...")
    model = load_robot_model(URDF, "base_link", "tool0", JOINT_NAMES)

    print(f"Sampling {NUM_SAMPLES} random joint configurations (seed={SEED})...")
    q_configs = sample_joint_configs(NUM_SAMPLES, SEED)

    rows = []
    for eps in EPSILONS:
        print(f"  epsilon = {eps:.0e} ...")
        rel_errors = np.zeros(NUM_SAMPLES, dtype=np.float64)
        success_count = 0

        for i in range(NUM_SAMPLES):
            q = q_configs[i]
            try:
                J64 = numerical_jacobian_fp64(model, q, eps)
                J32 = numerical_jacobian_fp32(model, q, eps)
                rel_errors[i] = relative_frobenius_error(J32, J64)
                success_count += 1
            except Exception:
                rel_errors[i] = np.nan

        valid = rel_errors[~np.isnan(rel_errors)]
        rows.append({
            "epsilon": eps,
            "N": len(valid),
            "mean_relative_error": float(np.mean(valid)),
            "median_relative_error": float(np.median(valid)),
            "p95_relative_error": float(np.percentile(valid, 95)),
            "max_relative_error": float(np.max(valid)),
            "success_rate": success_count / NUM_SAMPLES,
            "throughput_targets_per_s": 0.0,  # NA for pure Jacobian analysis
        })
        print(f"    mean={rows[-1]['mean_relative_error']:.6e} "
              f"median={rows[-1]['median_relative_error']:.6e} "
              f"p95={rows[-1]['p95_relative_error']:.6e} "
              f"max={rows[-1]['max_relative_error']:.6e}")

    # Write CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epsilon", "N", "mean_relative_error", "median_relative_error",
            "p95_relative_error", "max_relative_error",
            "success_rate", "throughput_targets_per_s",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
