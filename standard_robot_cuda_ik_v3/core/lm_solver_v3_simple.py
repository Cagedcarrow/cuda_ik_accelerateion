#!/usr/bin/env python3
"""V3 Simplified LM: Adaptive damping + analytical Jacobian, no step rejection.

Key insight: For IK from zero_seed (far from target), the quadratic model
is too inaccurate for trust-region step rejection. A simpler approach:
  - Always accept the step (like DLS)
  - Use LM-style λ update based on loss reduction ratio
  - Works because IK always improves with small enough damping

Compares: V2-DLS-Analytical vs V3-LM-Simple vs V3-LM-Simple-MultiSeed
"""
from __future__ import annotations

import csv, sys, time
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


def pose_error_log(T_cur, T_tgt):
    """Pose error using SO(3) log map (proper metric for LM)."""
    p_err = T_cur[:3, 3] - T_tgt[:3, 3]
    R_rel = T_tgt[:3, :3].T @ T_cur[:3, :3]
    # Matrix logarithm for SO(3)
    tr = np.trace(R_rel)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if abs(theta) < 1e-10:
        r_err = np.zeros(3)
    else:
        r_err = theta / (2.0 * np.sin(theta)) * np.array([
            R_rel[2,1] - R_rel[1,2],
            R_rel[0,2] - R_rel[2,0],
            R_rel[1,0] - R_rel[0,1],
        ])
    return np.concatenate([p_err, r_err])


def pose_error_skew(T_cur, T_tgt):
    """Pose error using skew-symmetric extraction (matches CUDA kernel)."""
    p_err = T_cur[:3, 3] - T_tgt[:3, 3]
    R_rel = T_cur[:3, :3].T @ T_tgt[:3, :3]
    r_err = np.array([0.5*(R_rel[2,1]-R_rel[1,2]),
                       0.5*(R_rel[0,2]-R_rel[2,0]),
                       0.5*(R_rel[1,0]-R_rel[0,1])])
    return np.concatenate([p_err, r_err])


# =========================================================================
# V3-LM-Simple: Adaptive LM, always accept, no trust region
# =========================================================================
def solve_lm_simple(model, q_seed, T_tgt, max_iter=160,
                     pos_tol=0.01, rot_tol=0.0873, use_log_map=False):
    """Simplified LM: adaptive λ, always accept steps. Suitable for far-from-solution IK."""
    q = q_seed.copy().astype(np.float64)
    lamb = 1e-2     # initial damping (conservative, more gradient-descent)
    nu = 2.0
    err_fn = pose_error_log if use_log_map else pose_error_skew
    prev_loss = 1e9

    for k in range(max_iter):
        T_cur, p, z = fk_with_frames(model, q)
        e = err_fn(T_cur, T_tgt)
        pos_err = float(np.linalg.norm(e[:3]))
        rot_err = float(np.linalg.norm(e[3:]))
        loss_cur = 0.5 * np.dot(e, e)

        if pos_err < pos_tol and rot_err < rot_tol:
            return {"converged": True, "iterations": k+1,
                    "pos_error_m": pos_err, "rot_error_rad": rot_err, "q": q.copy()}

        J = analytical_jacobian(T_cur, p, z)

        # Build LM system
        H = J.T @ J + lamb * np.eye(6)
        g_vec = J.T @ e

        # Solve
        try:
            L = np.linalg.cholesky(H)
            dq = -np.linalg.solve(L.T, np.linalg.solve(L, g_vec))
        except np.linalg.LinAlgError:
            dq = -np.linalg.solve(J.T @ J + (lamb*10)*np.eye(6), g_vec)

        # Step clamp (always 0.35 rad, same as CUDA DLS)
        nrm = np.max(np.abs(dq))
        if nrm > 0.35: dq *= 0.35 / nrm

        # Trial step
        q_trial = q + dq
        q_trial = np.clip(q_trial, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        dq_actual = q_trial - q

        T_trial, _, _ = fk_with_frames(model, q_trial)
        e_trial = err_fn(T_trial, T_tgt)
        loss_trial = 0.5 * np.dot(e_trial, e_trial)

        # Gain ratio (for damping adjustment only, always accept step)
        actual_red = loss_cur - loss_trial
        pred_red = -np.dot(g_vec, dq_actual) - 0.5 * dq_actual.T @ (J.T @ J) @ dq_actual
        rho = actual_red / max(pred_red, 1e-12)

        # Always accept the step
        q = q_trial

        # Adjust damping based on gain ratio
        if rho > 0.75:
            lamb *= 0.5          # good reduction → more Gauss-Newton
        elif rho > 0.25:
            lamb *= 0.7          # moderate → slightly less damping
        else:
            lamb *= 2.0          # poor reduction → more gradient descent

        lamb = np.clip(lamb, 1e-6, 0.5)
        prev_loss = loss_cur

    T_cur, _, _ = fk_with_frames(model, q)
    e = pose_error_skew(T_cur, T_tgt)
    return {"converged": False, "iterations": max_iter,
            "pos_error_m": float(np.linalg.norm(e[:3])),
            "rot_error_rad": float(np.linalg.norm(e[3:])), "q": q.copy()}


# =========================================================================
# Multi-seed variant
# =========================================================================
def solve_lm_simple_multiseed(model, T_tgt, K=4, max_iter=160,
                               pos_tol=0.01, rot_tol=0.0873, use_log_map=False):
    rng = np.random.default_rng(SEED)
    best_result = None; best_loss = float("inf")
    for s in range(K):
        if s == 0:
            q_seed = np.zeros(6)
        else:
            q_seed = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        result = solve_lm_simple(model, q_seed, T_tgt, max_iter, pos_tol, rot_tol, use_log_map)
        T_end, _, _ = fk_with_frames(model, result["q"])
        e = pose_error_skew(T_end, T_tgt)
        l = 0.5 * np.dot(e, e)
        if l < best_loss:
            best_loss = l; best_result = result
    return best_result


# =========================================================================
# V2 DLS baseline
# =========================================================================
def solve_dls_analytical(model, q_seed, T_tgt, max_iter=160,
                          pos_tol=0.01, rot_tol=0.0873):
    q = q_seed.copy().astype(np.float64)
    lamb = 5e-4; W = np.diag([1.0]*3 + [1.0]*3); prev_pos_err = 1e9

    for k in range(max_iter):
        T_cur, p, z = fk_with_frames(model, q)
        e = pose_error_skew(T_cur, T_tgt)
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
        lamb = np.clip(lamb, 1e-4, 0.5); prev_pos_err = pos_err

        H = J.T @ W @ W @ J + lamb * np.eye(6); g_vec = J.T @ W @ W @ e
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
    e = pose_error_skew(T_cur, T_tgt)
    return {"converged": False, "iterations": max_iter,
            "pos_error_m": float(np.linalg.norm(e[:3])),
            "rot_error_rad": float(np.linalg.norm(e[3:])), "q": q.copy()}


def main():
    print("=" * 65)
    print("V3 Simplified LM: V2-DLS vs V3-LM-Simple vs V3-LM-Simple-MultiSeed")
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
        ("V2-DLS-Analytical",     lambda m,s,t,pt,rt: solve_dls_analytical(m,s,t,pos_tol=pt,rot_tol=rt)),
        ("V3-LM-Simple(skew)",    lambda m,s,t,pt,rt: solve_lm_simple(m,s,t,pos_tol=pt,rot_tol=rt,use_log_map=False)),
        ("V3-LM-Simple(log)",     lambda m,s,t,pt,rt: solve_lm_simple(m,s,t,pos_tol=pt,rot_tol=rt,use_log_map=True)),
        ("V3-LM-Simple-Multi4",   lambda m,s,t,pt,rt: solve_lm_simple_multiseed(m,t,K=4,pos_tol=pt,rot_tol=rt)),
        ("V3-LM-Simple-Multi8",   lambda m,s,t,pt,rt: solve_lm_simple_multiseed(m,t,K=8,pos_tol=pt,rot_tol=rt)),
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
    print(f"{'Method':<28} {'SR':>6} {'pos_p50/mm':>10} {'pos_p95/mm':>10} {'iters':>7} {'time/s':>7}")
    print("-"*65)
    for row in all_rows:
        if row["threshold"] == "Strict(5mm/1deg)":
            print(f"{row['method']:<28} {row['success_rate']:>6} {row['pos_p50_mm']:>10} {row['pos_p95_mm']:>10} {row['avg_iters']:>7} {row['time_s']:>7}")

    csv_path = OUT_DIR / "v3_lm_simple_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
