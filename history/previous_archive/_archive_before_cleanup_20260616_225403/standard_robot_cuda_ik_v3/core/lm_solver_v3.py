#!/usr/bin/env python3
"""V3 LM Solver: LM + Analytical Jacobian + Line Search + Trust Region + Multi-Seed.

Compares: V1 (DLS+Numerical), V2 (DLS+Analytical), V3 (LM+Analytical+LineSearch+MultiSeed)
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "standard_robot_cuda_ik" / "tools"))
from robot_model import load_robot_model, rotation_about_axis

URDF = Path(__file__).resolve().parents[2] / "standard_robot_cuda_ik" / "urdf" / "ur10_official.urdf"
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
JOINT_LIMITS = np.array([
    [-6.2832, 6.2832], [-6.2832, 6.2832], [-3.1416, 3.1416],
    [-6.2832, 6.2832], [-6.2832, 6.2832], [-6.2832, 6.2832],
])

OUT_DIR = Path(__file__).resolve().parents[1] / "experiments"
EPS = 1e-6
SEED = 42


# ===========================================================================
# FK with per-joint frames (same as V2)
# ===========================================================================
def fk_with_frames(model, q):
    T = np.eye(4); p = np.zeros((6, 3)); z = np.zeros((6, 3))
    for i, joint in enumerate(model.active_joints):
        origin = joint.origin_matrix
        axis_local = np.asarray(joint.axis, dtype=np.float64)
        T = T @ origin
        p[i] = T[:3, 3]
        z[i] = T[:3, :3] @ axis_local
        T = T @ rotation_about_axis(axis_local, float(q[i]))
    for joint in model.fixed_tail_joints:
        T = T @ joint.origin_matrix
    return T, p, z


def analytical_jacobian(T_ee, p, z):
    """J = [z_i × (p_ee - p_i); z_i]"""
    p_ee = T_ee[:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        cross = np.cross(z[i], p_ee - p[i])
        J[0:3, i] = cross
        J[3:6, i] = z[i]
    return J


def numerical_jacobian(model, q, eps=EPS):
    """Central difference numerical Jacobian (matching CUDA kernel)."""
    J = np.zeros((6, 6), dtype=np.float64)
    q0 = q.copy()
    T_nom, _, _ = fk_with_frames(model, q0)
    R0 = T_nom[:3, :3]
    for j in range(6):
        qp, qm = q0.copy(), q0.copy()
        qp[j] += eps; qm[j] -= eps
        Tp, _, _ = fk_with_frames(model, qp)
        Tm, _, _ = fk_with_frames(model, qm)
        inv = 0.5 / eps
        J[0, j] = (Tp[0,3]-Tm[0,3])*inv
        J[1, j] = (Tp[1,3]-Tm[1,3])*inv
        J[2, j] = (Tp[2,3]-Tm[2,3])*inv
        dR = R0.T @ (Tp[:3,:3] - Tm[:3,:3])
        J[3, j] = (dR[2,1]-dR[1,2])*0.5*inv
        J[4, j] = (dR[0,2]-dR[2,0])*0.5*inv
        J[5, j] = (dR[1,0]-dR[0,1])*0.5*inv
    return J


def pose_error(T_cur, T_tgt):
    """Position + rotation error (matching CUDA skew-symmetric extraction)."""
    p_err = T_cur[:3, 3] - T_tgt[:3, 3]
    R_rel = T_cur[:3, :3].T @ T_tgt[:3, :3]
    r_err = np.array([
        0.5*(R_rel[2,1]-R_rel[1,2]),
        0.5*(R_rel[0,2]-R_rel[2,0]),
        0.5*(R_rel[1,0]-R_rel[0,1]),
    ])
    return np.concatenate([p_err, r_err])


def loss(e, W=None):
    """Weighted squared error."""
    if W is None:
        return 0.5 * np.dot(e, e)
    return 0.5 * e.T @ W @ e


# ===========================================================================
# V1: DLS + Numerical Jacobian
# ===========================================================================
def solve_v1_dls_numerical(model, q_seed, T_tgt, max_iter=160,
                             pos_tol=0.01, rot_tol=0.0873):
    q = q_seed.copy().astype(np.float64)
    lamb = 5e-4
    w_pos, w_rot = 1.0, 1.0
    W = np.diag([w_pos]*3 + [w_rot]*3)
    prev_pos_err = 1e9

    for k in range(max_iter):
        T_cur, _, _ = fk_with_frames(model, q)
        e = pose_error(T_cur, T_tgt)
        pos_err = float(np.linalg.norm(e[:3]))
        rot_err = float(np.linalg.norm(e[3:]))

        if pos_err < pos_tol and rot_err < rot_tol:
            return {"converged": True, "iterations": k+1,
                    "pos_error_m": pos_err, "rot_error_rad": rot_err, "q": q.copy()}

        J = numerical_jacobian(model, q)

        # DLS adaptive damping (same as V1 CUDA)
        if k == 0:
            if pos_err > 0.5: lamb = 0.1
            elif pos_err > 0.1: lamb = 0.1 + (5e-4-0.1)*(0.5-pos_err)/0.4
        else:
            r = pos_err / max(prev_pos_err, 1e-12)
            if r < 0.9: lamb *= 0.7
            elif r > 1.1: lamb *= 2.0
        lamb = np.clip(lamb, 1e-4, 0.5)
        prev_pos_err = pos_err

        H = J.T @ W @ W @ J + lamb * np.eye(6)
        g_vec = J.T @ W @ W @ e
        try:
            L = np.linalg.cholesky(H)
            dq = -np.linalg.solve(L.T, np.linalg.solve(L, g_vec))
        except np.linalg.LinAlgError:
            dq = -np.linalg.solve(H, g_vec)

        nrm = np.max(np.abs(dq))
        if nrm > 0.35: dq *= 0.35 / nrm
        q += dq
        q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    T_cur, _, _ = fk_with_frames(model, q)
    e = pose_error(T_cur, T_tgt)
    return {"converged": False, "iterations": max_iter,
            "pos_error_m": float(np.linalg.norm(e[:3])),
            "rot_error_rad": float(np.linalg.norm(e[3:])), "q": q.copy()}


# ===========================================================================
# V2: DLS + Analytical Jacobian
# ===========================================================================
def solve_v2_dls_analytical(model, q_seed, T_tgt, max_iter=160,
                              pos_tol=0.01, rot_tol=0.0873):
    q = q_seed.copy().astype(np.float64)
    lamb = 5e-4
    w_pos, w_rot = 1.0, 1.0
    W = np.diag([w_pos]*3 + [w_rot]*3)
    prev_pos_err = 1e9

    for k in range(max_iter):
        T_cur, p, z = fk_with_frames(model, q)
        e = pose_error(T_cur, T_tgt)
        pos_err = float(np.linalg.norm(e[:3]))
        rot_err = float(np.linalg.norm(e[3:]))

        if pos_err < pos_tol and rot_err < rot_tol:
            return {"converged": True, "iterations": k+1,
                    "pos_error_m": pos_err, "rot_error_rad": rot_err, "q": q.copy()}

        J = analytical_jacobian(T_cur, p, z)

        if k == 0:
            if pos_err > 0.5: lamb = 0.1
            elif pos_err > 0.1: lamb = 0.1 + (5e-4-0.1)*(0.5-pos_err)/0.4
        else:
            r = pos_err / max(prev_pos_err, 1e-12)
            if r < 0.9: lamb *= 0.7
            elif r > 1.1: lamb *= 2.0
        lamb = np.clip(lamb, 1e-4, 0.5)
        prev_pos_err = pos_err

        H = J.T @ W @ W @ J + lamb * np.eye(6)
        g_vec = J.T @ W @ W @ e
        try:
            L = np.linalg.cholesky(H)
            dq = -np.linalg.solve(L.T, np.linalg.solve(L, g_vec))
        except np.linalg.LinAlgError:
            dq = -np.linalg.solve(H, g_vec)

        nrm = np.max(np.abs(dq))
        if nrm > 0.35: dq *= 0.35 / nrm
        q_new = q + dq
        q_new = np.clip(q_new, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        q = q_new

    T_cur, _, _ = fk_with_frames(model, q)
    e = pose_error(T_cur, T_tgt)
    return {"converged": False, "iterations": max_iter,
            "pos_error_m": float(np.linalg.norm(e[:3])),
            "rot_error_rad": float(np.linalg.norm(e[3:])), "q": q.copy()}


# ===========================================================================
# V3: LM + Analytical Jacobian + Line Search + Trust Region
# ===========================================================================
def solve_v3_lm(model, q_seed, T_tgt, max_iter=160,
                pos_tol=0.01, rot_tol=0.0873):
    """V3 LM solver with analytical Jacobian, backtracking line search, trust region."""
    q = q_seed.copy().astype(np.float64)
    lamb = 1e-3          # initial damping (smaller than DLS for more Gauss-Newton)
    nu = 2.0             # damping adjustment factor
    delta = 0.5          # trust region radius (rad)
    w_pos, w_rot = 1.0, 1.0
    W = np.diag([w_pos]*3 + [w_rot]*3)

    T_cur, p, z = fk_with_frames(model, q)
    e_cur = pose_error(T_cur, T_tgt)
    loss_cur = loss(e_cur, W)
    prev_pos_err = float(np.linalg.norm(e_cur[:3]))

    for k in range(max_iter):
        pos_err = float(np.linalg.norm(e_cur[:3]))
        rot_err = float(np.linalg.norm(e_cur[3:]))

        if pos_err < pos_tol and rot_err < rot_tol:
            return {"converged": True, "iterations": k+1,
                    "pos_error_m": pos_err, "rot_error_rad": rot_err, "q": q.copy()}

        T_cur, p, z = fk_with_frames(model, q)
        e_cur = pose_error(T_cur, T_tgt)
        J = analytical_jacobian(T_cur, p, z)

        # LM system: (J^T W^2 J + λI) Δq = -J^T W^2 e
        H = J.T @ W @ W @ J + lamb * np.eye(6)
        g_vec = J.T @ W @ W @ e_cur

        # Solve
        try:
            L = np.linalg.cholesky(H)
            dq = -np.linalg.solve(L.T, np.linalg.solve(L, g_vec))
        except np.linalg.LinAlgError:
            dq = -np.linalg.solve(H, g_vec)

        # Trust region: ||Δq||_∞ ≤ δ
        nrm = np.max(np.abs(dq))
        if nrm > delta:
            dq *= delta / nrm

        # Predicted reduction (linear model)
        pred_red = -np.dot(g_vec, dq) - 0.5 * dq.T @ (H - lamb * np.eye(6)) @ dq
        pred_red = max(pred_red, 1e-12)

        # Backtracking line search
        alpha = 1.0
        q_new = q + alpha * dq
        q_new = np.clip(q_new, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        T_new, _, _ = fk_with_frames(model, q_new)
        e_new = pose_error(T_new, T_tgt)
        loss_new = loss(e_new, W)
        actual_red = loss_cur - loss_new

        # Backtrack if insufficient reduction
        bt_count = 0
        while actual_red < 0.1 * alpha * pred_red and bt_count < 8:
            alpha *= 0.5
            if alpha < 1e-4:
                break
            q_new = q + alpha * dq
            q_new = np.clip(q_new, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
            T_new, _, _ = fk_with_frames(model, q_new)
            e_new = pose_error(T_new, T_tgt)
            loss_new = loss(e_new, W)
            actual_red = loss_cur - loss_new
            bt_count += 1

        # Gain ratio: actual / predicted reduction
        rho = actual_red / pred_red if pred_red > 0 else 0.0

        if rho > 0.75:
            # Good step: accept, reduce damping (more Gauss-Newton)
            q = q_new
            e_cur = e_new
            loss_cur = loss_new
            lamb *= 0.5
            delta = min(delta * 1.5, 0.5)  # grow trust region
        elif rho > 0.25:
            # Moderate step: accept, keep damping
            q = q_new
            e_cur = e_new
            loss_cur = loss_new
            # lambda unchanged
        else:
            # Poor step: reject, increase damping (more gradient descent)
            lamb *= nu
            nu *= 2.0  # accelerate damping growth on repeated failures
            delta *= 0.5  # shrink trust region

        lamb = np.clip(lamb, 1e-6, 1.0)
        nu = min(nu, 16.0)
        delta = np.clip(delta, 0.01, 0.5)

        prev_pos_err = pos_err

    T_cur, _, _ = fk_with_frames(model, q)
    e = pose_error(T_cur, T_tgt)
    return {"converged": False, "iterations": max_iter,
            "pos_error_m": float(np.linalg.norm(e[:3])),
            "rot_error_rad": float(np.linalg.norm(e[3:])), "q": q.copy()}


# ===========================================================================
# V3-M: LM + Multi-Seed
# ===========================================================================
def solve_v3_lm_multiseed(model, T_tgt, K=4, max_iter=160,
                           pos_tol=0.01, rot_tol=0.0873):
    """V3 with K seeds, pick best solution."""
    best_result = None
    best_loss = float("inf")
    rng = np.random.default_rng(SEED)

    for s in range(K):
        if s == 0:
            q_seed = np.zeros(6)  # zero seed
        elif s == 1:
            q_seed = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])  # random
        else:
            # Jitter around random base
            q_base = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
            q_seed = q_base + rng.uniform(-0.3, 0.3, 6)
            q_seed = np.clip(q_seed, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

        result = solve_v3_lm(model, q_seed, T_tgt, max_iter, pos_tol, rot_tol)
        T_end, _, _ = fk_with_frames(model, result["q"])
        e_end = pose_error(T_end, T_tgt)
        l = loss(e_end)
        if l < best_loss:
            best_loss = l
            best_result = result
            best_result["seed_idx"] = s
    return best_result


# ===========================================================================
# Main Benchmark
# ===========================================================================
def main():
    print("=" * 65)
    print("V3 LM Solver Benchmark: V1 vs V2 vs V3 vs V3-MultiSeed")
    print("=" * 65)

    model = load_robot_model(URDF, "base_link", "tool0", JOINT_NAMES)
    rng = np.random.default_rng(SEED)
    N_test = 200

    # Generate reachable targets (same methodology as V1 benchmark)
    targets, seeds = [], []
    for _ in range(N_test):
        q_gt = np.array([rng.uniform(JOINT_LIMITS[j,0], JOINT_LIMITS[j,1]) for j in range(6)])
        T_gt, _, _ = fk_with_frames(model, q_gt)
        targets.append(T_gt)
        seeds.append(np.zeros(6))

    thresholds = {
        "Loose(30mm/10deg)": (0.030, 0.1745),
        "Medium(10mm/5deg)": (0.010, 0.0873),
        "Strict(5mm/1deg)":  (0.005, 0.01745),
    }

    methods = [
        ("V1-DLS-Numerical",  lambda m,s,t,pt,rt: solve_v1_dls_numerical(m,s,t,pos_tol=pt,rot_tol=rt)),
        ("V2-DLS-Analytical", lambda m,s,t,pt,rt: solve_v2_dls_analytical(m,s,t,pos_tol=pt,rot_tol=rt)),
        ("V3-LM",             lambda m,s,t,pt,rt: solve_v3_lm(m,s,t,pos_tol=pt,rot_tol=rt)),
        ("V3-LM-MultiSeed4",  lambda m,s,t,pt,rt: solve_v3_lm_multiseed(m,t,K=4,pos_tol=pt,rot_tol=rt)),
    ]

    all_rows = []
    for method_name, solver_fn in methods:
        print(f"\n{'─'*50}")
        print(f"Method: {method_name}")
        print(f"{'─'*50}")
        for thresh_label, (pos_t, rot_t) in thresholds.items():
            t0 = time.perf_counter()
            conv = 0
            pos_errs, rot_errs, iters_list = [], [], []
            for i in range(N_test):
                r = solver_fn(model, seeds[i], targets[i], pos_t, rot_t)
                if r["converged"]: conv += 1
                pos_errs.append(r["pos_error_m"])
                rot_errs.append(r["rot_error_rad"])
                iters_list.append(r["iterations"])
            dt = time.perf_counter() - t0

            sr = conv / N_test
            row = {
                "method": method_name,
                "threshold": thresh_label,
                "N": N_test,
                "success_rate": f"{sr:.3f}",
                "pos_p50_mm": f"{np.median(pos_errs)*1000:.2f}",
                "pos_p95_mm": f"{np.percentile(pos_errs,95)*1000:.2f}",
                "rot_p50_deg": f"{np.median(rot_errs)*57.3:.3f}",
                "rot_p95_deg": f"{np.percentile(rot_errs,95)*57.3:.3f}",
                "avg_iters": f"{np.mean(iters_list):.1f}",
                "time_s": f"{dt:.1f}",
            }
            all_rows.append(row)
            print(f"  {thresh_label}: SR={sr:.3f}  "
                  f"pos_p50={np.median(pos_errs)*1000:.2f}mm  "
                  f"iters={np.mean(iters_list):.1f}  "
                  f"time={dt:.1f}s")

    # ================================================================
    # Summary table
    # ================================================================
    print(f"\n{'='*65}")
    print("SUMMARY: Strict Threshold (5mm/1deg)")
    print(f"{'='*65}")
    print(f"{'Method':<25} {'SR':>6} {'pos_p50/mm':>10} {'pos_p95/mm':>10} {'iters':>7} {'time/s':>7}")
    print("-"*65)
    for row in all_rows:
        if row["threshold"] == "Strict(5mm/1deg)":
            print(f"{row['method']:<25} {row['success_rate']:>6} {row['pos_p50_mm']:>10} {row['pos_p95_mm']:>10} {row['avg_iters']:>7} {row['time_s']:>7}")

    # Save CSV
    csv_path = OUT_DIR / "v3_lm_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nData saved to {csv_path}")


if __name__ == "__main__":
    main()
