#!/usr/bin/env python3
"""V3 LM Solver v2: Proper LM with Trust Region (no line search).

Standard Levenberg-Marquardt algorithm:
  (J^T J + λI) Δq = -J^T e
  if loss_new < loss_old: accept, λ /= 3
  else: reject, λ *= 2

Trust region: ||Δq||_∞ ≤ δ, adjusted via gain ratio ρ.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

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
SEED = 42


def fk_with_frames(model, q):
    T = np.eye(4); p = np.zeros((6, 3)); z = np.zeros((6, 3))
    for i, joint in enumerate(model.active_joints):
        T = T @ joint.origin_matrix
        p[i] = T[:3, 3]
        z[i] = T[:3, :3] @ np.asarray(joint.axis, dtype=np.float64)
        T = T @ rotation_about_axis(np.asarray(joint.axis, dtype=np.float64), float(q[i]))
    for joint in model.fixed_tail_joints:
        T = T @ joint.origin_matrix
    return T, p, z


def analytical_jacobian(T_ee, p, z):
    p_ee = T_ee[:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        cross = np.cross(z[i], p_ee - p[i])
        J[0:3, i] = cross; J[3:6, i] = z[i]
    return J


def pose_error(T_cur, T_tgt):
    p_err = T_cur[:3, 3] - T_tgt[:3, 3]
    R_rel = T_cur[:3, :3].T @ T_tgt[:3, :3]
    r_err = np.array([0.5*(R_rel[2,1]-R_rel[1,2]),
                       0.5*(R_rel[0,2]-R_rel[2,0]),
                       0.5*(R_rel[1,0]-R_rel[0,1])])
    return np.concatenate([p_err, r_err])


def loss_fn(e):
    return 0.5 * np.dot(e, e)


# =========================================================================
# Proper LM with Trust Region
# =========================================================================
def solve_lm_trust_region(model, q_seed, T_tgt, max_iter=160,
                           pos_tol=0.01, rot_tol=0.0873):
    """Standard LM: trust region only, no line search."""
    q = q_seed.copy().astype(np.float64)
    lamb = 1e-2          # initial damping
    delta = 0.35          # trust region radius (rad)
    eta1, eta2 = 0.25, 0.75  # gain ratio thresholds

    T_cur, p, z = fk_with_frames(model, q)
    e_cur = pose_error(T_cur, T_tgt)
    loss_cur = loss_fn(e_cur)

    for k in range(max_iter):
        pos_err = float(np.linalg.norm(e_cur[:3]))
        rot_err = float(np.linalg.norm(e_cur[3:]))

        if pos_err < pos_tol and rot_err < rot_tol:
            return {"converged": True, "iterations": k+1,
                    "pos_error_m": pos_err, "rot_error_rad": rot_err, "q": q.copy()}

        J = analytical_jacobian(T_cur, p, z)

        # Build LM system
        H = J.T @ J + lamb * np.eye(6)
        g_vec = J.T @ e_cur

        # Solve
        try:
            L = np.linalg.cholesky(H)
            dq = -np.linalg.solve(L.T, np.linalg.solve(L, g_vec))
        except np.linalg.LinAlgError:
            # LDLT fallback: add more damping
            dq = -np.linalg.solve(J.T @ J + (lamb*10) * np.eye(6), g_vec)

        # Trust region: clamp step
        dq_nrm = np.max(np.abs(dq))
        if dq_nrm > delta:
            dq *= delta / dq_nrm

        # Trial step
        q_trial = q + dq
        q_trial = np.clip(q_trial, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        dq_actual = q_trial - q  # actual step after clipping

        T_trial, p_trial, z_trial = fk_with_frames(model, q_trial)
        e_trial = pose_error(T_trial, T_tgt)
        loss_trial = loss_fn(e_trial)

        # Compute gain ratio
        actual_red = loss_cur - loss_trial
        # Predicted reduction from quadratic model:
        # pred = -g^T Δq - 0.5 Δq^T (J^T J) Δq
        pred_red = -np.dot(g_vec, dq_actual) - 0.5 * dq_actual.T @ (J.T @ J) @ dq_actual
        rho = actual_red / max(pred_red, 1e-12)

        if rho > eta2:
            # Very successful: accept, reduce damping, grow trust region
            q = q_trial
            T_cur, p, z = T_trial, p_trial, z_trial
            e_cur = e_trial
            loss_cur = loss_trial
            lamb = max(lamb / 3.0, 1e-6)
            delta = min(delta * 1.5, 0.5)
        elif rho > eta1:
            # Moderately successful: accept, keep damping
            q = q_trial
            T_cur, p, z = T_trial, p_trial, z_trial
            e_cur = e_trial
            loss_cur = loss_trial
            # λ unchanged
        else:
            # Poor: reject step, increase damping, shrink trust region
            lamb *= 3.0
            delta *= 0.5
            delta = max(delta, 0.01)

        lamb = np.clip(lamb, 1e-6, 1.0)

    T_cur, _, _ = fk_with_frames(model, q)
    e = pose_error(T_cur, T_tgt)
    return {"converged": False, "iterations": max_iter,
            "pos_error_m": float(np.linalg.norm(e[:3])),
            "rot_error_rad": float(np.linalg.norm(e[3:])), "q": q.copy()}


# =========================================================================
# V3-M: LM Trust Region + Multi-Seed
# =========================================================================
def solve_lm_multiseed(model, T_tgt, K=4, max_iter=160,
                        pos_tol=0.01, rot_tol=0.0873):
    rng = np.random.default_rng(SEED)
    best_result = None
    best_loss = float("inf")
    for s in range(K):
        if s == 0:
            q_seed = np.zeros(6)
        else:
            q_seed = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        result = solve_lm_trust_region(model, q_seed, T_tgt, max_iter, pos_tol, rot_tol)
        T_end, _, _ = fk_with_frames(model, result["q"])
        l = loss_fn(pose_error(T_end, T_tgt))
        if l < best_loss:
            best_loss = l
            best_result = result
    return best_result


# =========================================================================
# Old V2 DLS for comparison
# =========================================================================
def numerical_jacobian(model, q, eps=1e-6):
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


def solve_dls_analytical(model, q_seed, T_tgt, max_iter=160,
                          pos_tol=0.01, rot_tol=0.0873):
    """V2 DLS with analytical Jacobian."""
    q = q_seed.copy().astype(np.float64)
    lamb = 5e-4
    W = np.diag([1.0]*3 + [1.0]*3)
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
        q += dq
        q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    T_cur, _, _ = fk_with_frames(model, q)
    e = pose_error(T_cur, T_tgt)
    return {"converged": False, "iterations": max_iter,
            "pos_error_m": float(np.linalg.norm(e[:3])),
            "rot_error_rad": float(np.linalg.norm(e[3:])), "q": q.copy()}


# =========================================================================
def main():
    print("=" * 65)
    print("V3 LM Trust-Region Benchmark: V2-DLS vs V3-LM-TR vs V3-LM-TR-MultiSeed")
    print("=" * 65)

    model = load_robot_model(URDF, "base_link", "tool0", JOINT_NAMES)
    rng = np.random.default_rng(SEED)
    N_test = 200

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
        ("V2-DLS-Analytical", lambda m,s,t,pt,rt: solve_dls_analytical(m,s,t,pos_tol=pt,rot_tol=rt)),
        ("V3-LM-TR",          lambda m,s,t,pt,rt: solve_lm_trust_region(m,s,t,pos_tol=pt,rot_tol=rt)),
        ("V3-LM-TR-Multi4",   lambda m,s,t,pt,rt: solve_lm_multiseed(m,t,K=4,pos_tol=pt,rot_tol=rt)),
        ("V3-LM-TR-Multi8",   lambda m,s,t,pt,rt: solve_lm_multiseed(m,t,K=8,pos_tol=pt,rot_tol=rt)),
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
                "method": method_name, "threshold": thresh_label, "N": N_test,
                "success_rate": f"{sr:.3f}",
                "pos_p50_mm": f"{np.median(pos_errs)*1000:.2f}",
                "pos_p95_mm": f"{np.percentile(pos_errs,95)*1000:.2f}",
                "rot_p50_deg": f"{np.median(rot_errs)*57.3:.3f}",
                "avg_iters": f"{np.mean(iters_list):.1f}",
                "time_s": f"{dt:.1f}",
            }
            all_rows.append(row)
            print(f"  {thresh_label}: SR={sr:.3f}  pos_p50={np.median(pos_errs)*1000:.2f}mm  "
                  f"iters={np.mean(iters_list):.1f}  time={dt:.1f}s")

    print(f"\n{'='*65}")
    print("SUMMARY: Strict Threshold (5mm/1deg)")
    print(f"{'='*65}")
    print(f"{'Method':<25} {'SR':>6} {'pos_p50/mm':>10} {'pos_p95/mm':>10} {'iters':>7} {'time/s':>7}")
    print("-"*65)
    for row in all_rows:
        if row["threshold"] == "Strict(5mm/1deg)":
            print(f"{row['method']:<25} {row['success_rate']:>6} {row['pos_p50_mm']:>10} {row['pos_p95_mm']:>10} {row['avg_iters']:>7} {row['time_s']:>7}")

    csv_path = OUT_DIR / "v3_lm_trust_region_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
