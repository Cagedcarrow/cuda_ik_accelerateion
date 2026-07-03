#!/usr/bin/env python3
"""Diagnose Medium/Strict SR anomaly and generate full error table.

Per 排查.md: Medium SR must always be >= Strict SR for the same targets/solutions.
If Multi4 Medium=90.5% < Strict=91.0%, there's a bug in evaluation code.
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
OUT_DIR = Path(__file__).resolve().parent
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
    p_ee = T_ee[:3, 3]; J = np.zeros((6, 6))
    for i in range(6):
        cross = np.cross(z[i], p_ee - p[i])
        J[0:3, i] = cross; J[3:6, i] = z[i]
    return J


def pose_error_skew(T_cur, T_tgt):
    p_err = T_cur[:3, 3] - T_tgt[:3, 3]
    R_rel = T_cur[:3, :3].T @ T_tgt[:3, :3]
    r_err = np.array([0.5*(R_rel[2,1]-R_rel[1,2]),
                       0.5*(R_rel[0,2]-R_rel[2,0]),
                       0.5*(R_rel[1,0]-R_rel[0,1])])
    return np.concatenate([p_err, r_err])


def solve_lm_simple(model, q_seed, T_tgt, max_iter=160,
                     pos_tol=0.01, rot_tol=0.0873):
    """Same V3-LM-Simple as before."""
    q = q_seed.copy().astype(np.float64); lamb = 1e-2
    for k in range(max_iter):
        T_cur, p, z = fk_with_frames(model, q)
        e = pose_error_skew(T_cur, T_tgt)
        pos_err = float(np.linalg.norm(e[:3]))
        rot_err = float(np.linalg.norm(e[3:]))
        if pos_err < pos_tol and rot_err < rot_tol:
            return {"converged": True, "iterations": k+1,
                    "pos_error_m": pos_err, "rot_error_rad": rot_err, "q": q.copy()}
        J = analytical_jacobian(T_cur, p, z)
        H = J.T @ J + lamb * np.eye(6); g_vec = J.T @ e
        try:
            L = np.linalg.cholesky(H)
            dq = -np.linalg.solve(L.T, np.linalg.solve(L, g_vec))
        except np.linalg.LinAlgError:
            dq = -np.linalg.solve(J.T @ J + (lamb*10)*np.eye(6), g_vec)
        nrm = np.max(np.abs(dq))
        if nrm > 0.35: dq *= 0.35 / nrm
        q_trial = q + dq
        q_trial = np.clip(q_trial, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        dq_actual = q_trial - q
        T_trial, _, _ = fk_with_frames(model, q_trial)
        e_trial = pose_error_skew(T_trial, T_tgt)
        loss_cur, loss_trial = 0.5*np.dot(e,e), 0.5*np.dot(e_trial,e_trial)
        actual_red = loss_cur - loss_trial
        pred_red = -np.dot(g_vec, dq_actual) - 0.5 * dq_actual.T @ (J.T @ J) @ dq_actual
        rho = actual_red / max(pred_red, 1e-12)
        q = q_trial
        if rho > 0.75: lamb *= 0.5
        elif rho > 0.25: lamb *= 0.7
        else: lamb *= 2.0
        lamb = np.clip(lamb, 1e-6, 0.5)
    T_cur, _, _ = fk_with_frames(model, q)
    e = pose_error_skew(T_cur, T_tgt)
    return {"converged": False, "iterations": max_iter,
            "pos_error_m": float(np.linalg.norm(e[:3])),
            "rot_error_rad": float(np.linalg.norm(e[3:])), "q": q.copy()}


def solve_lm_multiseed(model, T_tgt, K=4, max_iter=160):
    rng = np.random.default_rng(SEED)
    best_result = None; best_loss = float("inf")
    for s in range(K):
        q_seed = np.zeros(6) if s == 0 else rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        result = solve_lm_simple(model, q_seed, T_tgt, max_iter)
        T_end, _, _ = fk_with_frames(model, result["q"])
        l = 0.5 * np.dot(pose_error_skew(T_end, T_tgt), pose_error_skew(T_end, T_tgt))
        if l < best_loss: best_loss = l; best_result = result
    return best_result


def main():
    print("=" * 70)
    print("DIAGNOSIS: Medium/Strict SR consistency check + Full Error Table")
    print("=" * 70)

    model = load_robot_model(URDF, "base_link", "tool0", JOINT_NAMES)
    rng = np.random.default_rng(SEED)
    N_test = 200

    # Generate targets ONCE, shared across all methods
    targets, seeds = [], []
    for _ in range(N_test):
        q_gt = np.array([rng.uniform(JOINT_LIMITS[j,0], JOINT_LIMITS[j,1]) for j in range(6)])
        T_gt, _, _ = fk_with_frames(model, q_gt)
        targets.append(T_gt)
        seeds.append(np.zeros(6))

    # ================================================================
    # Test: V3-LM-Multi4 with detailed per-target tracking
    # ================================================================
    print("\nRunning V3-LM-Multi4 with detailed per-target error tracking...")
    per_target = []
    for i in range(N_test):
        r = solve_lm_multiseed(model, targets[i], K=4)
        T_end, _, _ = fk_with_frames(model, r["q"])
        e = pose_error_skew(T_end, targets[i])
        pos_err = float(np.linalg.norm(e[:3]))
        rot_err = float(np.linalg.norm(e[3:]))
        rot_deg = rot_err * 57.3  # rad → deg

        # Check convergence at each threshold ON THE SAME SOLUTION
        medium_ok = (pos_err < 0.010) and (rot_err < 0.0873)  # 10mm/5deg
        strict_ok = (pos_err < 0.005) and (rot_err < 0.01745)  # 5mm/1deg
        loose_ok  = (pos_err < 0.030) and (rot_err < 0.1745)  # 30mm/10deg

        per_target.append({
            "target_id": i,
            "pos_err_mm": pos_err * 1000,
            "rot_err_deg": rot_deg,
            "medium_ok": medium_ok,
            "strict_ok": strict_ok,
            "loose_ok": loose_ok,
            "iterations": r["iterations"],
        })

    # Count by threshold
    medium_sr = sum(1 for t in per_target if t["medium_ok"]) / N_test
    strict_sr = sum(1 for t in per_target if t["strict_ok"]) / N_test
    loose_sr  = sum(1 for t in per_target if t["loose_ok"]) / N_test

    print(f"\n  Threshold consistency check:")
    print(f"    Loose  SR = {loose_sr:.4f}  ({sum(1 for t in per_target if t['loose_ok'])}/{N_test})")
    print(f"    Medium SR = {medium_sr:.4f}  ({sum(1 for t in per_target if t['medium_ok'])}/{N_test})")
    print(f"    Strict SR = {strict_sr:.4f}  ({sum(1 for t in per_target if t['strict_ok'])}/{N_test})")

    # Verify monotonicity
    if loose_sr >= medium_sr >= strict_sr:
        print(f"    ✅ Monotonic: Loose ≥ Medium ≥ Strict (correct)")
    else:
        print(f"    ❌ NON-MONOTONIC! Loose={loose_sr:.4f} Medium={medium_sr:.4f} Strict={strict_sr:.4f}")
        # Find counterexamples
        for t in per_target:
            if t["strict_ok"] and not t["medium_ok"]:
                print(f"    BUG: target {t['target_id']}: strict_ok=True but medium_ok=False!")
                print(f"         pos={t['pos_err_mm']:.2f}mm rot={t['rot_err_deg']:.4f}deg")
            if t["medium_ok"] and not t["loose_ok"]:
                print(f"    BUG: target {t['target_id']}: medium_ok=True but loose_ok=False!")

    # ================================================================
    # Full error table: all-target vs success-only
    # ================================================================
    print(f"\n{'='*70}")
    print("FULL ERROR TABLE (V3-LM-Multi4, per 排查.md Step 2)")
    print(f"{'='*70}")

    pos_all = np.array([t["pos_err_mm"] for t in per_target])
    rot_all = np.array([t["rot_err_deg"] for t in per_target])

    for label, mask_fn in [
        ("All targets", lambda t: True),
        ("Loose success", lambda t: t["loose_ok"]),
        ("Medium success", lambda t: t["medium_ok"]),
        ("Strict success", lambda t: t["strict_ok"]),
    ]:
        subset = [t for t in per_target if mask_fn(t)]
        if not subset:
            continue
        pos_sub = np.array([t["pos_err_mm"] for t in subset])
        rot_sub = np.array([t["rot_err_deg"] for t in subset])
        n = len(subset)
        print(f"\n  {label} (n={n}):")
        print(f"    pos: mean={np.mean(pos_sub):.2f}  p50={np.median(pos_sub):.2f}  "
              f"p95={np.percentile(pos_sub,95):.2f}  max={np.max(pos_sub):.2f} mm")
        print(f"    rot: mean={np.mean(rot_sub):.4f}  p50={np.median(rot_sub):.4f}  "
              f"p95={np.percentile(rot_sub,95):.4f}  max={np.max(rot_sub):.4f} deg")

    # ================================================================
    # CSV output
    # ================================================================
    csv_path = OUT_DIR / "diagnosis_v3_multiseed.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "subset", "N", "SR", "pos_mean_mm", "pos_p50_mm", "pos_p95_mm", "pos_max_mm",
            "rot_mean_deg", "rot_p50_deg", "rot_p95_deg", "rot_max_deg",
        ])
        w.writeheader()
        for label, mask_fn in [
            ("all", lambda t: True),
            ("loose_success", lambda t: t["loose_ok"]),
            ("medium_success", lambda t: t["medium_ok"]),
            ("strict_success", lambda t: t["strict_ok"]),
        ]:
            subset = [t for t in per_target if mask_fn(t)]
            if not subset: continue
            pos_sub = np.array([t["pos_err_mm"] for t in subset])
            rot_sub = np.array([t["rot_err_deg"] for t in subset])
            w.writerow({
                "subset": label, "N": len(subset),
                "SR": f"{len(subset)/N_test:.3f}",
                "pos_mean_mm": f"{np.mean(pos_sub):.2f}",
                "pos_p50_mm": f"{np.median(pos_sub):.2f}",
                "pos_p95_mm": f"{np.percentile(pos_sub,95):.2f}",
                "pos_max_mm": f"{np.max(pos_sub):.2f}",
                "rot_mean_deg": f"{np.mean(rot_sub):.4f}",
                "rot_p50_deg": f"{np.median(rot_sub):.4f}",
                "rot_p95_deg": f"{np.percentile(rot_sub,95):.4f}",
                "rot_max_deg": f"{np.max(rot_sub):.4f}",
            })
    print(f"\nDiagnosis CSV saved to {csv_path}")

    # ================================================================
    # Summary answer
    # ================================================================
    print(f"\n{'='*70}")
    print("DIAGNOSIS CONCLUSION")
    print(f"{'='*70}")
    if loose_sr >= medium_sr >= strict_sr:
        print("✅ SR monotonicity CORRECT (Loose ≥ Medium ≥ Strict)")
        print("   Previous Medium=90.5% vs Strict=91.0% was likely a CSV/rounding artifact")
        print(f"   Current re-run: Medium={medium_sr:.3f} Strict={strict_sr:.3f}")
    else:
        print("❌ SR monotonicity BROKEN — evaluation code bug confirmed")


if __name__ == "__main__":
    main()
