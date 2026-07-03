#!/usr/bin/env python3
"""Panda 7DOF FK + CPU 数值 DLS IK 参考实现

用途：
1. 验证 CUDA 7DOF kernel 的 FK 正确性
2. 作为 CPU baseline 对比 CUDA DLS 收敛率
3. 生成 CUDA 常量头文件

关键差异（vs UR10 6DOF）：
- 7 个主动关节
- Jacobian 6×7（非方阵）
- Hessian 7×7（SPD）
- 7×7 LDL^T 求解器
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from robot_model import load_robot_model

URDF_PATH = Path(__file__).resolve().parents[2] / "urdf" / "panda_7dof.urdf"
BASE_LINK = "panda_link0"
TIP_LINK = "panda_link8"
EXPECTED_JOINTS = [
    "panda_joint1", "panda_joint2", "panda_joint3",
    "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7",
]

# Damping parameters (matching CUDA A5)
LAMBDA_BASE = 2e-4
LAMBDA_FAR = 5e-2
LAMBDA_FLOOR = 1e-4
LAMBDA_SCALE = 8e-2
WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)  # 6 error DOF (pos x,y,z + rot x,y,z)


def load_panda_model():
    return load_robot_model(URDF_PATH, BASE_LINK, TIP_LINK, EXPECTED_JOINTS)


def numerical_jacobian_7dof(model, q, eps=1e-6):
    """Compute 6×7 numerical Jacobian for 7DOF manipulator.

    J[i,j] = ∂err_i / ∂q_j
    where err is 6-DOF pose error (position + orientation).
    """
    T0 = model.fk(q)
    J = np.zeros((6, 7), dtype=np.float64)

    for j in range(7):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[j] += eps
        q_minus[j] -= eps

        T_p = model.fk(q_plus)
        T_m = model.fk(q_minus)
        inv_2eps = 0.5 / eps

        # Position columns (translation)
        J[0, j] = (T_p[0, 3] - T_m[0, 3]) * inv_2eps
        J[1, j] = (T_p[1, 3] - T_m[1, 3]) * inv_2eps
        J[2, j] = (T_p[2, 3] - T_m[2, 3]) * inv_2eps

        # Rotation columns (angular velocity from R_diff = R0^T @ R)
        # Same formulation as CUDA: dR_skew * 0.5 / eps
        r00, r01, r02 = T0[0, 0], T0[0, 1], T0[0, 2]
        r10, r11, r12 = T0[1, 0], T0[1, 1], T0[1, 2]
        r20, r21, r22 = T0[2, 0], T0[2, 1], T0[2, 2]

        # Compute R_cur^T * R at q±eps for each
        def r_dot(R, T):
            return np.array([
                [r00 * T[0, 0] + r10 * T[1, 0] + r20 * T[2, 0],
                 r00 * T[0, 1] + r10 * T[1, 1] + r20 * T[2, 1],
                 r00 * T[0, 2] + r10 * T[1, 2] + r20 * T[2, 2]],
                [r01 * T[0, 0] + r11 * T[1, 0] + r21 * T[2, 0],
                 r01 * T[0, 1] + r11 * T[1, 1] + r21 * T[2, 1],
                 r01 * T[0, 2] + r11 * T[1, 2] + r21 * T[2, 2]],
                [r02 * T[0, 0] + r12 * T[1, 0] + r22 * T[2, 0],
                 r02 * T[0, 1] + r12 * T[1, 1] + r22 * T[2, 1],
                 r02 * T[0, 2] + r12 * T[1, 2] + r22 * T[2, 2]],
            ])

        dR = (r_dot(T_p, T_p) - r_dot(T_m, T_m)) * inv_2eps
        J[3, j] = (dR[2, 1] - dR[1, 2]) * 0.5
        J[4, j] = (dR[0, 2] - dR[2, 0]) * 0.5
        J[5, j] = (dR[1, 0] - dR[0, 1]) * 0.5

    return J


def pose_error_6dof(T_cur, T_tgt):
    """6-DOF pose error (position + orientation), matching CUDA formulation."""
    err = np.zeros(6, dtype=np.float64)
    err[0:3] = T_tgt[:3, 3] - T_cur[:3, 3]
    # Rotation error: skew of R_cur^T @ R_tgt
    R_cur = T_cur[:3, :3]
    R_tgt = T_tgt[:3, :3]
    R_err = R_cur.T @ R_tgt
    err[3] = 0.5 * (R_err[2, 1] - R_err[1, 2])
    err[4] = 0.5 * (R_err[0, 2] - R_err[2, 0])
    err[5] = 0.5 * (R_err[1, 0] - R_err[0, 1])
    return err


def rotation_geodesic_distance(T_cur, T_tgt):
    """Geodesic distance on SO(3)."""
    R_cur = T_cur[:3, :3]
    R_tgt = T_tgt[:3, :3]
    trace = np.trace(R_cur.T @ R_tgt)
    return np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))


def ldlt_solve_7x7(H, g):
    """7×7 LDL^T Cholesky solve: H * x = g (SPD matrix)."""
    L = np.zeros((7, 7))
    D = np.zeros(7)

    # LDL^T decomposition
    for j in range(7):
        d = H[j, j]
        for k in range(j):
            d -= L[j, k] * L[j, k] * D[k]
        D[j] = d

        for i in range(j + 1, 7):
            s = H[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k] * D[k]
            L[i, j] = s / D[j]
        L[j, j] = 1.0

    # Forward substitution: L * y = g
    y = np.zeros(7)
    for i in range(7):
        s = g[i]
        for k in range(i):
            s -= L[i, k] * y[k]
        y[i] = s

    # Diagonal scaling: x = y / D
    x = y / D

    # Backward substitution: L^T * x_final = x
    x_final = np.zeros(7)
    for i in range(6, -1, -1):
        s = x[i]
        for k in range(i + 1, 7):
            s -= L[k, i] * x_final[k]
        x_final[i] = s

    return x_final


def cuda_clamp(x, lo, hi):
    return max(min(x, hi), lo)


def ik_solve_one(model, q_seed, T_tgt, max_iter=160,
                 pos_tol=0.03, orient_tol=0.5236):
    """Solve IK for one target using DLS (matching CUDA A5 logic)."""
    q = q_seed.copy()
    q_ref = q_seed.copy()
    q_best = q_seed.copy()
    best_pos_err = 1e100
    stagnation = 0
    lambda_base = LAMBDA_BASE
    lambda_far = LAMBDA_FAR
    lambda_floor = LAMBDA_FLOOR
    lambda_scale = LAMBDA_SCALE

    for iteration in range(max_iter):
        # FK
        T_cur = model.fk(q)

        # Pose error
        err = pose_error_6dof(T_cur, T_tgt)
        pos_err = np.linalg.norm(err[0:3])
        rot_err = rotation_geodesic_distance(T_cur, T_tgt)

        # Convergence check
        if pos_err <= pos_tol and rot_err <= orient_tol:
            return q, iteration + 1, pos_err, rot_err, True

        # Best tracking
        if pos_err < best_pos_err:
            best_pos_err = pos_err
            q_best = q.copy()
            stagnation = 0
        else:
            stagnation += 1

        # Divergence recovery
        if stagnation > 25:
            q = q_best.copy()
            T_cur = model.fk(q)
            err = pose_error_6dof(T_cur, T_tgt)
            pos_err = np.linalg.norm(err[0:3])
            rot_err = rotation_geodesic_distance(T_cur, T_tgt)
            return q, iteration + 1, pos_err, rot_err, False

        # Adaptive damping
        if pos_err > 0.1:
            lam = max(lambda_base, lambda_far * (pos_err / lambda_scale))
            lam = min(lam, lambda_far * 3.0)
        else:
            lam = lambda_floor + lambda_base * (pos_err / lambda_scale)
        if stagnation > 5:
            lam *= 1.0 + 0.3 * (stagnation - 5)
            lam = min(lam, 0.5)

        # Numerical Jacobian (6×7)
        J = numerical_jacobian_7dof(model, q)

        # Hessian: H = J^T * W^2 * J + λ * I  (7×7)
        W = np.diag(WEIGHTS)
        H = J.T @ (W @ W) @ J + lam * np.eye(7)

        # Gradient: g = J^T * W^2 * err
        g = J.T @ (W @ W) @ err

        # LDL^T Solve
        dq = ldlt_solve_7x7(H, g)

        # Step clamp (disabled — matching A5)
        step_norm = np.linalg.norm(dq)
        if step_norm <= 1e-8:
            break

        # Apply step with joint limits
        for i in range(7):
            lo = model.active_joints[i].lower or -np.pi
            hi = model.active_joints[i].upper or np.pi
            q[i] = cuda_clamp(q[i] + dq[i], lo, hi)

    # Fallback: use best solution
    T_best = model.fk(q_best)
    err = pose_error_6dof(T_best, T_tgt)
    pos_err = np.linalg.norm(err[0:3])
    rot_err = rotation_geodesic_distance(T_best, T_tgt)
    return q_best, max_iter, pos_err, rot_err, False


def main():
    model = load_panda_model()
    print(f"Panda model loaded: {model.dof} DOF")

    # Load or generate test targets
    data_dir = Path(__file__).resolve().parent
    seeds_path = data_dir / "panda_test_seeds_N10.bin"
    targets_path = data_dir / "panda_test_targets_N10.bin"
    gt_path = data_dir / "panda_test_q_target_N10.npy"

    if not all(p.exists() for p in [seeds_path, targets_path, gt_path]):
        print("Test data not found. Run test_fk.py first.")
        return

    seeds = np.fromfile(seeds_path, dtype=np.float64).reshape(-1, 7)
    targets = np.fromfile(targets_path, dtype=np.float64).reshape(-1, 4, 4)
    q_gt = np.load(str(gt_path))

    N = seeds.shape[0]
    print(f"\nRunning CPU DLS IK on {N} targets...")
    converged = 0
    for i in range(N):
        q_sol, iters, pos_err, rot_err, conv = ik_solve_one(
            model, seeds[i], targets[i]
        )
        if conv:
            converged += 1
        status = "✅" if conv else "❌"
        print(f"  [{i}] {status} iters={iters:3d} pos_err={pos_err:.6f} rot_err={rot_err:.6f}")

    print(f"\nResults: {converged}/{N} converged ({converged/N*100:.1f}%)")

    # Export CUDA constants header
    print("\nExporting CUDA constants...")
    origins = model.origins_array()
    axes = model.axes_array()
    tool = model.tool_offset_from_last_joint().reshape(-1)
    limits = model.limits_array()

    header = f"""// Auto-generated from Panda URDF. DO NOT HAND-EDIT.
#pragma once

static const double k_origins[{len(origins)}] = {{"""

    for i in range(0, len(origins), 4):
        header += "\n    "
        header += ", ".join(f"{x:.16e}" for x in origins[i:i+4])
        if i + 4 < len(origins):
            header += ","

    header += f"""
}};

static const double k_axes[{len(axes)}] = {{"""
    for i in range(0, len(axes), 3):
        header += "\n    "
        header += ", ".join(f"{x:.16e}" for x in axes[i:i+3])
        if i + 3 < len(axes):
            header += ","

    header += f"""
}};

static const int k_q_index[{model.dof}] = {{ {', '.join(str(i) for i in range(model.dof))} }};

static const double k_T_tcp[{len(tool)}] = {{"""
    for i in range(0, len(tool), 4):
        header += "\n    "
        header += ", ".join(f"{x:.16e}" for x in tool[i:i+4])
        if i + 4 < len(tool):
            header += ","

    header += f"""
}};

static const double k_joint_limits[{len(limits)}] = {{"""
    for i in range(0, len(limits), 2):
        header += "\n    "
        header += ", ".join(f"{x:.16e}" for x in limits[i:i+2])
        if i + 2 < len(limits):
            header += ","

    header += """
};

static const double k_weights[28] = {
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 0.6, 0.6, 0.6, 0.6,
    1.0, 1.0, 1.0, 0.3, 0.3, 0.3, 0.3,
    1.0, 1.0, 1.0, 0.15, 0.15, 0.15, 0.15,
};

static const double k_lambda_params[4] = { 2e-4, 5e-2, 1e-4, 8e-2 };
"""

    header_path = data_dir / "panda_model_constants.h"
    header_path.write_text(header)
    print(f"  Written: {header_path}")

    # Also export model verification JSON
    import json
    verif = {
        "model_name": model.name,
        "dof": model.dof,
        "joints": model.joint_table(),
        "tool_offset_mm": tool.tolist(),
        "convergence_cpu_dls": float(converged / N),
    }
    verif_path = data_dir / "panda_model_verification.json"
    with open(verif_path, "w") as f:
        json.dump(verif, f, indent=2)
    print(f"  Written: {verif_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
