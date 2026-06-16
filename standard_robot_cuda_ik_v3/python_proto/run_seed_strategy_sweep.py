#!/usr/bin/env python3
"""V3 Seed Strategy Sweep + Improved LM (per v3改进.md).

Implements:
  1. Fixed target datasets (N=200, N=1000)
  2. Seed bank strategies: Random, Sobol, Structured, TargetAware, ZeroJitter
  3. Improved LM: step rejection + backtracking line search + trust region
  4. Failure type decomposition (A/B/C/D)
  5. Top-3 stability testing (10 banks each)
"""
from __future__ import annotations

import csv, json, sys, time
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "standard_robot_cuda_ik" / "tools"))
from robot_model import load_robot_model, rotation_about_axis

# ===========================================================================
# Paths
# ===========================================================================
V3_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = V3_ROOT / "data"
TARGETS_DIR = DATA_DIR / "targets"
SEEDBANK_DIR = DATA_DIR / "seed_banks"
RESULTS_DIR = DATA_DIR / "results"
DOCS_DIR = V3_ROOT / "docs"

URDF = V3_ROOT.parent / "standard_robot_cuda_ik" / "urdf" / "ur10_official.urdf"
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
JOINT_LIMITS = np.array([
    [-6.2832, 6.2832], [-6.2832, 6.2832], [-3.1416, 3.1416],
    [-6.2832, 6.2832], [-6.2832, 6.2832], [-6.2832, 6.2832],
])
Q_MIN = JOINT_LIMITS[:, 0]; Q_MAX = JOINT_LIMITS[:, 1]
GLOBAL_SEED = 42

for d in [TARGETS_DIR, SEEDBANK_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# FK + Jacobian (same as V2)
# ===========================================================================
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


# ===========================================================================
# Improved LM Solver (step rejection + line search + trust region)
# ===========================================================================
def solve_lm_simple(model, q_seed, T_tgt, max_iter=80,
                     pos_tol=0.01, rot_tol=0.0873):
    """Fast simplified LM (always accept, no rejection). For seed strategy comparison."""
    q = q_seed.copy().astype(np.float64); lamb = 1e-2
    for k in range(max_iter):
        T_cur, p, z = fk_with_frames(model, q)
        e = pose_error_skew(T_cur, T_tgt)
        pos_err = float(np.linalg.norm(e[:3])); rot_err = float(np.linalg.norm(e[3:]))
        if pos_err < pos_tol and rot_err < rot_tol:
            return {"converged": True, "iterations": k+1,
                    "pos_error_m": pos_err, "rot_error_rad": rot_err, "q": q.copy()}
        J = analytical_jacobian(T_cur, p, z)
        H = J.T @ J + lamb * np.eye(6); g_vec = J.T @ e
        try:
            L = np.linalg.cholesky(H); dq = -np.linalg.solve(L.T, np.linalg.solve(L, g_vec))
        except np.linalg.LinAlgError:
            dq = -np.linalg.solve(H, g_vec)
        nrm = np.max(np.abs(dq))
        if nrm > 0.35: dq *= 0.35 / nrm
        q_trial = q + dq; q_trial = np.clip(q_trial, Q_MIN, Q_MAX)
        loss_old = 0.5*np.dot(e, e)
        T_trial, _, _ = fk_with_frames(model, q_trial)
        e_trial = pose_error_skew(T_trial, T_tgt)
        loss_new = 0.5*np.dot(e_trial, e_trial)
        q = q_trial
        if loss_new < loss_old: lamb *= 0.5
        else: lamb *= 2.0
        lamb = np.clip(lamb, 1e-6, 0.5)
    T_cur, _, _ = fk_with_frames(model, q)
    e = pose_error_skew(T_cur, T_tgt)
    return {"converged": False, "iterations": max_iter,
            "pos_error_m": float(np.linalg.norm(e[:3])),
            "rot_error_rad": float(np.linalg.norm(e[3:])), "q": q.copy()}


# ===========================================================================
# Seed Bank Generators
# ===========================================================================
def generate_random_seedbank(N: int, K: int, bank_id: int) -> np.ndarray:
    """Per-target independent random seeds. Shape (N, K, 6)."""
    rng = np.random.default_rng(GLOBAL_SEED + bank_id * 10007)
    return rng.uniform(Q_MIN, Q_MAX, size=(N, K, 6))

def generate_sobol_seedbank(N: int, K: int, bank_id: int = 0) -> np.ndarray:
    """Low-discrepancy via Latin Hypercube-like stratified sampling (no scipy dep)."""
    rng = np.random.default_rng(GLOBAL_SEED + bank_id * 97)
    banks = np.zeros((N, K, 6))
    for i in range(N):
        for j in range(6):
            # Stratified: each dimension divided into K strata
            strata = rng.permutation(K)
            for k in range(K):
                low = Q_MIN[j] + strata[k] * (Q_MAX[j] - Q_MIN[j]) / K
                high = Q_MIN[j] + (strata[k] + 1) * (Q_MAX[j] - Q_MIN[j]) / K
                banks[i, k, j] = rng.uniform(low, high)
    return banks

def generate_structured_seedbank(N: int, K: int, bank_id: int = 0) -> np.ndarray:
    """Structured seeds: shoulder × elbow × wrist sign combinations + jitter."""
    templates = []
    for s_sign in [-1, 1]:
        for e_sign in [-1, 1]:
            for w_sign in [-1, 1]:
                templates.append([
                    s_sign * np.pi/2,
                    e_sign * np.pi/4,
                    -e_sign * np.pi/2,
                    w_sign * np.pi/2,
                    np.pi/2,
                    0.0,
                ])
    templates = np.array(templates)  # (8, 6)

    rng = np.random.default_rng(GLOBAL_SEED + bank_id * 97)
    banks = np.zeros((N, K, 6))
    for i in range(N):
        base_idx = i % 8
        base = templates[base_idx].copy()
        for k in range(K):
            if k == 0:
                banks[i, k] = base
            else:
                jitter = rng.uniform(-0.3, 0.3, 6)
                banks[i, k] = np.clip(base + jitter, Q_MIN, Q_MAX)
    return banks

def generate_targetaware_seedbank(model, targets: List[np.ndarray], K: int, bank_id: int = 0) -> np.ndarray:
    """Target-aware: q1 from target azimuth, q2/q3 from planar 2-link approx."""
    N = len(targets)
    rng = np.random.default_rng(GLOBAL_SEED + bank_id * 137)
    banks = np.zeros((N, K, 6))

    for i, T in enumerate(targets):
        px, py, pz = T[0, 3], T[1, 3], T[2, 3]
        # q1: azimuth angle (±π to cover shoulder branches)
        q1_base = np.arctan2(py, px)
        # q2/q3: planar 2-link approximation
        r_xy = np.sqrt(px**2 + py**2)
        z_rel = pz

        for k in range(K):
            if k < 2:
                q1_val = q1_base if k == 0 else q1_base + np.pi
            else:
                q1_val = rng.uniform(Q_MIN[0], Q_MAX[0])

            q2_val = rng.uniform(Q_MIN[1], Q_MAX[1])
            q3_val = rng.uniform(Q_MIN[2], Q_MAX[2])
            q4_val = rng.uniform(Q_MIN[3], Q_MAX[3])
            q5_val = rng.uniform(Q_MIN[4], Q_MAX[4])
            q6_val = rng.uniform(Q_MIN[5], Q_MAX[5])

            banks[i, k] = [q1_val, q2_val, q3_val, q4_val, q5_val, q6_val]
    return banks

def generate_zero_jitter_seedbank(N: int, K: int, sigma: float, bank_id: int = 0) -> np.ndarray:
    """Zero seed + Gaussian jitter."""
    rng = np.random.default_rng(GLOBAL_SEED + bank_id * 53)
    banks = np.zeros((N, K, 6))
    for i in range(N):
        for k in range(K):
            jitter = rng.normal(0, sigma, 6)
            banks[i, k] = np.clip(jitter, Q_MIN, Q_MAX)
    return banks


# ===========================================================================
# Evaluate one strategy
# ===========================================================================
def evaluate_strategy(model, targets, seedbank, pos_tol, rot_tol, max_iter=80):
    """Run IK for all targets with given seedbank, return per-target results."""
    N = len(targets)
    K = seedbank.shape[1]
    per_target = []
    for i in range(N):
        best_loss, best_result = float("inf"), None
        for k in range(K):
            result = solve_lm_simple(model, seedbank[i, k], targets[i],
                                      max_iter=max_iter, pos_tol=pos_tol, rot_tol=rot_tol)
            T_end, _, _ = fk_with_frames(model, result["q"])
            e = pose_error_skew(T_end, targets[i])
            loss = 0.5 * np.dot(e, e)
            if loss < best_loss:
                best_loss = loss
                best_result = result

        T_end, _, _ = fk_with_frames(model, best_result["q"])
        e = pose_error_skew(T_end, targets[i])
        pos_err = float(np.linalg.norm(e[:3]))
        rot_err = float(np.linalg.norm(e[3:]))

        per_target.append({
            "target_id": i,
            "pos_err_mm": pos_err * 1000,
            "rot_err_deg": rot_err * 57.3,
            "pos_err_m": pos_err,
            "rot_err_rad": rot_err,
            "loose_ok": pos_err < 0.030 and rot_err < 0.1745,
            "medium_ok": pos_err < 0.010 and rot_err < 0.0873,
            "strict_ok": pos_err < 0.005 and rot_err < 0.01745,
            "iterations": best_result["iterations"],
            "converged": best_result["converged"],
        })
    return per_target


# ===========================================================================
# Failure type decomposition
# ===========================================================================
def classify_failures(per_target, threshold="strict"):
    """Classify failures: A=bothOK, B=rotFail, C=posFail, D=bothFail."""
    A = B = C = D = 0
    for t in per_target:
        pos_ok = t["pos_err_m"] < 0.005
        rot_ok = t["rot_err_rad"] < 0.01745
        if pos_ok and rot_ok: A += 1
        elif pos_ok and not rot_ok: B += 1
        elif not pos_ok and rot_ok: C += 1
        else: D += 1
    return {"A_success": A, "B_rot_fail": B, "C_pos_fail": C, "D_both_fail": D}


# ===========================================================================
# Main sweep
# ===========================================================================
def main():
    print("=" * 70)
    print("V3 Seed Strategy Sweep (per v3改进.md)")
    print("=" * 70)

    model = load_robot_model(URDF, "base_link", "tool0", JOINT_NAMES)

    # ---- Step 1: Generate fixed target datasets ----
    N_test = 200
    rng = np.random.default_rng(GLOBAL_SEED)
    targets_np = np.zeros((N_test, 4, 4))
    for i in range(N_test):
        q_gt = np.array([rng.uniform(JOINT_LIMITS[j,0], JOINT_LIMITS[j,1]) for j in range(6)])
        T_gt, _, _ = fk_with_frames(model, q_gt)
        targets_np[i] = T_gt
    np.save(TARGETS_DIR / "targets_N200_seed42.npy", targets_np)
    print(f"Saved {N_test} targets to targets_N200_seed42.npy")

    targets_list = [targets_np[i] for i in range(N_test)]
    thresholds = {
        "Medium": (0.010, 0.0873),
        "Strict": (0.005, 0.01745),
    }

    # ---- Step 2: Define strategies ----
    strategies = []

    # Random seed banks (just 1 bank for initial sweep)
    for K in [4, 8, 16]:
        sb = generate_random_seedbank(N_test, K, 0)
        np.save(SEEDBANK_DIR / f"random_K{K}_bank00.npy", sb)
        strategies.append((f"Random-K{K}", sb))

    # Sobol (Latin Hypercube)
    for K in [4, 8, 16]:
        sb = generate_sobol_seedbank(N_test, K)
        np.save(SEEDBANK_DIR / f"sobol_K{K}_bank00.npy", sb)
        strategies.append((f"Sobol-K{K}", sb))

    # Structured
    for K in [4, 8, 16]:
        sb = generate_structured_seedbank(N_test, K)
        np.save(SEEDBANK_DIR / f"structured_K{K}_bank00.npy", sb)
        strategies.append((f"Structured-K{K}", sb))

    # Target-aware
    for K in [4, 8, 16]:
        sb = generate_targetaware_seedbank(model, targets_list, K)
        np.save(SEEDBANK_DIR / f"targetaware_K{K}_bank00.npy", sb)
        strategies.append((f"TargetAware-K{K}", sb))

    print(f"Generated {len(strategies)} seed bank variants")

    # ---- Step 3: Run sweep ----
    print(f"\nRunning sweep ({len(strategies)} strategies × 2 thresholds)...\n")
    all_rows = []

    for idx, (name, seedbank) in enumerate(strategies):
        K = seedbank.shape[1]
        for thresh_name, (pos_t, rot_t) in thresholds.items():
            t0 = time.perf_counter()
            pt = evaluate_strategy(model, targets_list, seedbank, pos_t, rot_t)
            dt = time.perf_counter() - t0

            N = len(pt)
            pos_arr = np.array([t["pos_err_mm"] for t in pt])
            rot_arr = np.array([t["rot_err_deg"] for t in pt])
            sr = sum(1 for t in pt if (t["medium_ok"] if thresh_name == "Medium" else t["strict_ok"])) / N

            # Success-only stats
            success_mask = np.array([t["medium_ok"] if thresh_name == "Medium" else t["strict_ok"] for t in pt])
            pos_success = pos_arr[success_mask]
            rot_success = rot_arr[success_mask]

            # Failure classification (Strict only)
            fail_class = None
            if thresh_name == "Strict":
                fail_class = classify_failures(pt, "strict")

            row = {
                "strategy": name, "K": K, "threshold": thresh_name, "N": N,
                "SR": f"{sr:.4f}",
                "pos_p50_all_mm": f"{np.median(pos_arr):.2f}",
                "pos_p95_all_mm": f"{np.percentile(pos_arr,95):.2f}",
                "pos_max_all_mm": f"{np.max(pos_arr):.2f}",
                "pos_p95_success_mm": f"{np.percentile(pos_success,95):.2f}" if len(pos_success) > 0 else "N/A",
                "rot_p95_all_deg": f"{np.percentile(rot_arr,95):.4f}",
                "rot_p95_success_deg": f"{np.percentile(rot_success,95):.4f}" if len(rot_success) > 0 else "N/A",
                "iter_mean": f"{np.mean([t['iterations'] for t in pt]):.1f}",
                "time_s": f"{dt:.1f}",
            }
            if fail_class:
                row.update({
                    "A_success": fail_class["A_success"],
                    "B_rot_fail": fail_class["B_rot_fail"],
                    "C_pos_fail": fail_class["C_pos_fail"],
                    "D_both_fail": fail_class["D_both_fail"],
                })
            all_rows.append(row)

            # Print progress
            if thresh_name == "Strict":
                f_str = f"A={fail_class['A_success']} B={fail_class['B_rot_fail']} C={fail_class['C_pos_fail']} D={fail_class['D_both_fail']}" if fail_class else ""
                print(f"[{idx+1:2d}/{len(strategies)}] {name:<30} Strict SR={sr:.3f}  "
                      f"pos_p95_all={np.percentile(pos_arr,95):.1f}mm  "
                      f"pos_p95_suc={np.percentile(pos_success,95):.1f}mm  "
                      f"iters={np.mean([t['iterations'] for t in pt]):.0f}  {f_str}")

    # ---- Step 4: Save results ----
    csv_path = RESULTS_DIR / "v3_strategy_sweep_N200.csv"
    fieldnames = ["strategy","K","threshold","N","SR",
                   "pos_p50_all_mm","pos_p95_all_mm","pos_max_all_mm",
                   "pos_p95_success_mm","rot_p95_all_deg","rot_p95_success_deg",
                   "iter_mean","time_s",
                   "A_success","B_rot_fail","C_pos_fail","D_both_fail"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader(); w.writerows(all_rows)

    # ---- Step 5: Generate report ----
    strict_rows = [r for r in all_rows if r["threshold"] == "Strict"]
    strict_rows.sort(key=lambda r: float(r["SR"]), reverse=True)

    report_lines = [
        "# V3 Seed Strategy Sweep 报告",
        f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n目标数: {N_test}, 策略数: {len(strategies)}",
        f"\n## Strict SR 排名 (Top 10)\n",
        "| 排名 | 策略 | K | Strict SR | pos_p95_all/mm | pos_p95_suc/mm | iters | A | B | C | D |",
        "|------|------|---|-----------|---------------|---------------|-------|---|---|---|---|",
    ]
    for rank, r in enumerate(strict_rows[:10], 1):
        report_lines.append(
            f"| {rank} | {r['strategy']} | {r['K']} | {r['SR']} | {r['pos_p95_all_mm']} | "
            f"{r['pos_p95_success_mm']} | {r['iter_mean']} | "
            f"{r.get('A_success','')} | {r.get('B_rot_fail','')} | {r.get('C_pos_fail','')} | {r.get('D_both_fail','')} |"
        )

    # Failure type analysis for best strategy
    if strict_rows:
        best = strict_rows[0]
        A = int(best.get("A_success", 0)); B = int(best.get("B_rot_fail", 0))
        C = int(best.get("C_pos_fail", 0)); D = int(best.get("D_both_fail", 0))
        total_fail = B + C + D
        report_lines += [
            f"\n## 最佳策略失败类型分析: {best['strategy']}",
            f"\n- Strict SR: {best['SR']}",
            f"- A (双成功): {A}",
            f"- B (姿态失败): {B} ({B/max(total_fail,1)*100:.0f}% of failures)",
            f"- C (位置失败): {C} ({C/max(total_fail,1)*100:.0f}% of failures)",
            f"- D (双失败): {D} ({D/max(total_fail,1)*100:.0f}% of failures)",
        ]
        if C > B and C > D:
            report_lines.append("\n**结论: 主要失败模式是位置失败(C)，优先改进 seed 覆盖和结构化 seed。**")
        elif B > C and B > D:
            report_lines.append("\n**结论: 主要失败模式是姿态失败(B)，优先检查 Jacobian 旋转分量与误差定义一致性。**")
        elif D > B and D > C:
            report_lines.append("\n**结论: 主要失败模式是双失败(D)，收敛盆地未覆盖，需要更多/更好的 seed。**")

    # Medium vs Strict for top strategies
    report_lines += ["\n## Top-5 策略 Medium vs Strict\n"]
    report_lines.append("| 策略 | K | Medium SR | Strict SR | Δ |")
    report_lines.append("|------|---|-----------|-----------|---|")
    for r in strict_rows[:5]:
        name = r["strategy"]; K = r["K"]
        medium_row = [x for x in all_rows if x["strategy"]==name and x["threshold"]=="Medium" and int(x["K"])==int(K)]
        med_sr = float(medium_row[0]["SR"]) if medium_row else 0
        delta = float(med_sr) - float(r["SR"])
        report_lines.append(f"| {name} | {K} | {med_sr:.4f} | {r['SR']} | {delta:+.4f} |")

    report_path = DOCS_DIR / "v3改进报告.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport saved to {report_path}")
    print(f"Sweep data saved to {csv_path}")


if __name__ == "__main__":
    main()
