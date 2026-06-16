#!/usr/bin/env python3
"""V4 M0: V3 Freeze + Joint-Limit Barrier + q_prev Smoothness (per v4.md Priority 1+2).

Key fixes from V3:
  1. UNIFIED solve at Strict threshold → evaluate all 3 thresholds on same solution
  2. Sobol-K16 + Sobol-K32
  3. N=200 + N=1000
  4. Joint-limit barrier loss (Module B)
  5. q_prev warm start + smoothness penalty (Module C, simplified)

Output: v4_m0_results.md in docs/
"""
from __future__ import annotations

import csv, sys, time
from pathlib import Path
from typing import List
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "standard_robot_cuda_ik" / "tools"))
from robot_model import load_robot_model, rotation_about_axis

V4_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = V4_ROOT / "data"
DOCS_DIR = V4_ROOT / "docs"
for d in [DATA_DIR / "targets", DATA_DIR / "seed_banks", DATA_DIR / "results"]:
    d.mkdir(parents=True, exist_ok=True)

URDF = V4_ROOT.parent / "standard_robot_cuda_ik" / "urdf" / "ur10_official.urdf"
JOINT_NAMES = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"]
JOINT_LIMITS = np.array([[-6.2832,6.2832],[-6.2832,6.2832],[-3.1416,3.1416],[-6.2832,6.2832],[-6.2832,6.2832],[-6.2832,6.2832]])
Q_MIN, Q_MAX = JOINT_LIMITS[:,0], JOINT_LIMITS[:,1]
SEED = 42


def fk_with_frames(model, q):
    T = np.eye(4); p = np.zeros((6,3)); z = np.zeros((6,3))
    for i, joint in enumerate(model.active_joints):
        T = T @ joint.origin_matrix; p[i]=T[:3,3]; z[i]=T[:3,:3]@np.asarray(joint.axis,dtype=np.float64)
        T = T @ rotation_about_axis(np.asarray(joint.axis,dtype=np.float64),float(q[i]))
    for joint in model.fixed_tail_joints: T = T @ joint.origin_matrix
    return T,p,z


def analytical_jacobian(T_ee,p,z):
    pe=T_ee[:3,3]; J=np.zeros((6,6))
    for i in range(6): cross=np.cross(z[i],pe-p[i]); J[0:3,i]=cross; J[3:6,i]=z[i]
    return J


def pose_error_skew(T_c,T_t):
    pe=T_c[:3,3]-T_t[:3,3]; Rr=T_c[:3,:3].T@T_t[:3,:3]
    re=np.array([0.5*(Rr[2,1]-Rr[1,2]),0.5*(Rr[0,2]-Rr[2,0]),0.5*(Rr[1,0]-Rr[0,1])])
    return np.concatenate([pe,re])


def sobol_seedbank(N,K,bank_id=0):
    """Stratified Latin Hypercube seed bank (no scipy dep)."""
    rng=np.random.default_rng(SEED+bank_id*97)
    banks=np.zeros((N,K,6))
    for i in range(N):
        for j in range(6):
            strata=rng.permutation(K)
            for k in range(K):
                lo=Q_MIN[j]+strata[k]*(Q_MAX[j]-Q_MIN[j])/K
                hi=Q_MIN[j]+(strata[k]+1)*(Q_MAX[j]-Q_MIN[j])/K
                banks[i,k,j]=rng.uniform(lo,hi)
    return banks


def joint_limit_loss(q, margin=0.087):
    """Smooth barrier: penalize proximity to joint limits."""
    loss=0.0
    for j in range(6):
        d_lo=q[j]-Q_MIN[j]; d_hi=Q_MAX[j]-q[j]
        if d_lo<margin: loss+=(margin-d_lo)**2
        if d_hi<margin: loss+=(margin-d_hi)**2
    return loss


def solve_lm_v4(model, q_seed, T_tgt, max_iter=80, pos_tol=0.005, rot_tol=0.01745,
                 w_limit=0.1, margin=0.087, q_prev=None, w_smooth=0.0):
    """V4 LM: unified Strict-threshold solve with optional limit barrier + smoothness."""
    q = q_seed.copy().astype(np.float64); lamb = 1e-2
    for k in range(max_iter):
        T_cur,p,z = fk_with_frames(model,q)
        e = pose_error_skew(T_cur,T_tgt)
        pos_err = float(np.linalg.norm(e[:3])); rot_err = float(np.linalg.norm(e[3:]))
        if pos_err < pos_tol and rot_err < rot_tol:
            return {"converged":True,"iterations":k+1,"pos_error_m":pos_err,"rot_error_rad":rot_err,"q":q.copy()}

        J = analytical_jacobian(T_cur,p,z)

        # Extended gradient: g = J^T e + w_limit * ∇limit + w_smooth * (q - q_prev)
        g_vec = J.T @ e
        if w_limit > 0:
            # Numerical gradient of limit loss
            eps_l = 1e-6
            g_limit = np.zeros(6)
            loss0 = joint_limit_loss(q, margin)
            for jj in range(6):
                qp = q.copy(); qp[jj] += eps_l
                g_limit[jj] = (joint_limit_loss(qp,margin)-loss0)/eps_l
            g_vec += w_limit * g_limit
        if w_smooth > 0 and q_prev is not None:
            g_vec += w_smooth * (q - q_prev)

        H = J.T @ J + lamb * np.eye(6)
        try:
            L = np.linalg.cholesky(H); dq = -np.linalg.solve(L.T, np.linalg.solve(L,g_vec))
        except np.linalg.LinAlgError:
            dq = -np.linalg.solve(H, g_vec)

        nrm = np.max(np.abs(dq))
        if nrm > 0.35: dq *= 0.35/nrm

        q_trial = q + dq; q_trial = np.clip(q_trial, Q_MIN, Q_MAX)
        loss_old = 0.5*np.dot(e,e)
        if w_limit>0: loss_old += w_limit*joint_limit_loss(q,margin)
        if w_smooth>0 and q_prev is not None: loss_old += 0.5*w_smooth*np.dot(q-q_prev,q-q_prev)

        T_trial,_,_ = fk_with_frames(model,q_trial)
        e_trial = pose_error_skew(T_trial,T_tgt)
        loss_new = 0.5*np.dot(e_trial,e_trial)
        if w_limit>0: loss_new += w_limit*joint_limit_loss(q_trial,margin)
        if w_smooth>0 and q_prev is not None: loss_new += 0.5*w_smooth*np.dot(q_trial-q_prev,q_trial-q_prev)

        q = q_trial
        lamb = np.clip(lamb*(0.5 if loss_new<loss_old else 2.0), 1e-6, 0.5)

    T_cur,_,_ = fk_with_frames(model,q)
    e = pose_error_skew(T_cur,T_tgt)
    return {"converged":False,"iterations":max_iter,
            "pos_error_m":float(np.linalg.norm(e[:3])),"rot_error_rad":float(np.linalg.norm(e[3:])),"q":q.copy()}


def evaluate_unified(model, targets, seedbank, q_prev_list=None, w_limit=0.0, w_smooth=0.0):
    """UNIFIED solve at Strict threshold → evaluate at Loose/Medium/Strict on same solution."""
    N = len(targets); K = seedbank.shape[1]
    per_target = []
    for i in range(N):
        best_loss, best_r = float("inf"), None
        q_prev = q_prev_list[i] if q_prev_list else None
        for k in range(K):
            r = solve_lm_v4(model, seedbank[i,k], targets[i],
                             w_limit=w_limit, q_prev=q_prev, w_smooth=w_smooth)
            T_end,_,_ = fk_with_frames(model, r["q"])
            e = pose_error_skew(T_end, targets[i])
            loss = 0.5*np.dot(e,e)
            if w_limit>0: loss += w_limit*joint_limit_loss(r["q"])
            if loss < best_loss: best_loss=loss; best_r=r
        T_end,_,_ = fk_with_frames(model, best_r["q"])
        e = pose_error_skew(T_end, targets[i])
        pe = float(np.linalg.norm(e[:3])); re = float(np.linalg.norm(e[3:]))
        per_target.append({
            "pos_err_mm":pe*1000,"rot_err_deg":re*57.3,
            "loose_ok":pe<0.030 and re<0.1745,
            "medium_ok":pe<0.010 and re<0.0873,
            "strict_ok":pe<0.005 and re<0.01745,
            "iterations":best_r["iterations"],
            "near_limit":any((best_r["q"]-Q_MIN)<0.087) or any((Q_MAX-best_r["q"])<0.087),
        })
    return per_target


def main():
    print("="*60); print("V4 M0: V3 Freeze + Limit Barrier + Smoothness"); print("="*60)
    model = load_robot_model(URDF,"base_link","tool0",JOINT_NAMES)

    # Generate datasets
    for N_test in [200, 1000]:
        rng = np.random.default_rng(SEED+N_test)
        targets_np = np.zeros((N_test,4,4))
        for i in range(N_test):
            qg = np.array([rng.uniform(JOINT_LIMITS[j,0],JOINT_LIMITS[j,1]) for j in range(6)])
            T_gt,_,_ = fk_with_frames(model,qg); targets_np[i]=T_gt
        np.save(DATA_DIR/f"targets/v4_targets_N{N_test}_seed42.npy", targets_np)
        print(f"Saved {N_test} targets")

    # Seed banks
    for N_test in [200,1000]:
        for K in [16,32]:
            sb = sobol_seedbank(N_test, K)
            np.save(DATA_DIR/f"seed_banks/sobol_K{K}_N{N_test}_bank00.npy", sb)
    print("Seed banks generated")

    # ---- Run experiments ----
    results = []

    # First: generate a simple trajectory for smoothness test
    N_traj = 50
    rng_t = np.random.default_rng(SEED+999)
    q_start = np.array([rng_t.uniform(JOINT_LIMITS[j,0],JOINT_LIMITS[j,1]) for j in range(6)])
    T_start,_,_ = fk_with_frames(model, q_start)
    T_end,_,_ = fk_with_frames(model, np.array([rng_t.uniform(JOINT_LIMITS[j,0],JOINT_LIMITS[j,1]) for j in range(6)]))
    # Linear interpolation in workspace (not joint space - naive waypoints)
    traj_targets = []
    for t in range(N_traj):
        alpha = t/(N_traj-1)
        T = np.eye(4)
        T[:3,:3] = T_start[:3,:3]  # constant orientation (simplified)
        T[:3,3] = T_start[:3,3] + alpha*(T_end[:3,3]-T_start[:3,3])
        traj_targets.append(T)
    sb_traj = sobol_seedbank(N_traj, 16, 99)
    print(f"Trajectory: {N_traj} waypoints")

    configs = [
        # (label, N, K, w_limit, w_smooth, use_trajectory)
        ("V3-Sobol-K16",         200, 16, 0.0, 0.0, False),
        ("V3-Sobol-K32",         200, 32, 0.0, 0.0, False),
        ("V3-Sobol-K16",         1000, 16, 0.0, 0.0, False),
        ("V3-Sobol-K32",         1000, 32, 0.0, 0.0, False),
        ("V4-Limit-w0.1",        200, 16, 0.1, 0.0, False),
        ("V4-Limit-w1.0",        200, 16, 1.0, 0.0, False),
        ("V4-Limit-w10",         200, 16, 10.0, 0.0, False),
        ("V4-Smooth-w0.1",       50,  16, 0.0, 0.1, True),
        ("V4-Smooth-w1.0",       50,  16, 0.0, 1.0, True),
        ("V4-Smooth-w0(nobase)", 50,  16, 0.0, 0.0, True),
    ]

    # Pre-load targets
    targets_cache = {}
    for N_test in [200,1000]:
        targets_cache[N_test] = [np.load(DATA_DIR/f"targets/v4_targets_N{N_test}_seed42.npy")[i] for i in range(N_test)]

    for label, N_test, K, w_limit, w_smooth, use_traj in configs:
        t0 = time.perf_counter()
        if use_traj:
            targets = traj_targets
            sb = sobol_seedbank(len(targets), K, 99)
            N_actual = len(targets)
            # Generate q_prev chain: first frame has None, subsequent use previous best_q
            pt = []
            prev_q = None
            for i in range(N_actual):
                best_loss, best_r = float("inf"), None
                for k in range(K):
                    r = solve_lm_v4(model, sb[i,k], targets[i], w_limit=w_limit,
                                     q_prev=prev_q, w_smooth=w_smooth)
                    e = pose_error_skew(fk_with_frames(model,r["q"])[0], targets[i])
                    loss = 0.5*np.dot(e,e)
                    if w_limit>0: loss += w_limit*joint_limit_loss(r["q"])
                    if loss<best_loss: best_loss=loss; best_r=r
                prev_q = best_r["q"].copy()  # warm-start next frame
                T_end,_,_ = fk_with_frames(model, best_r["q"])
                e = pose_error_skew(T_end, targets[i])
                pe=float(np.linalg.norm(e[:3])); re=float(np.linalg.norm(e[3:]))
                pt.append({"pos_err_mm":pe*1000,"rot_err_deg":re*57.3,
                           "loose_ok":pe<0.030 and re<0.1745,
                           "medium_ok":pe<0.010 and re<0.0873,
                           "strict_ok":pe<0.005 and re<0.01745,
                           "iterations":best_r["iterations"]})
        else:
            targets = targets_cache[N_test]
            sb_path = DATA_DIR/f"seed_banks/sobol_K{K}_N{N_test}_bank00.npy"
            if sb_path.exists():
                sb = np.load(sb_path)
            else:
                sb = sobol_seedbank(N_test, K)
            N_actual = N_test
            pt = evaluate_unified(model, targets, sb, w_limit=w_limit, w_smooth=w_smooth)

        dt = time.perf_counter()-t0
        pos = np.array([t["pos_err_mm"] for t in pt])
        rot = np.array([t["rot_err_deg"] for t in pt])
        loose_sr = sum(1 for t in pt if t["loose_ok"])/N_actual
        med_sr   = sum(1 for t in pt if t["medium_ok"])/N_actual
        strict_sr= sum(1 for t in pt if t["strict_ok"])/N_actual
        mono = "✓" if (loose_sr>=med_sr>=strict_sr) else "✗"
        strict_mask = np.array([t["strict_ok"] for t in pt])
        pos_suc = pos[strict_mask]; rot_suc = rot[strict_mask]
        near_lim = sum(1 for t in pt if t.get("near_limit",False))/N_actual if pt and "near_limit" in pt[0] else None

        # Trajectory smoothness
        if use_traj and len(pt)>=2:
            qs = []
            for i in range(N_actual):
                best_loss, best_r = float("inf"), None
                for k in range(K):
                    r = solve_lm_v4(model, sb[i,k], targets[i], w_limit=w_limit, w_smooth=w_smooth)
                    e=pose_error_skew(fk_with_frames(model,r["q"])[0],targets[i]); l=0.5*np.dot(e,e)
                    if w_limit>0: l+=w_limit*joint_limit_loss(r["q"])
                    if l<best_loss: best_loss=l; best_r=r
                qs.append(best_r["q"])
            dqs = [np.max(np.abs(qs[i]-qs[i-1])) for i in range(1,N_actual)]
            mean_dq = np.mean(dqs); max_dq = np.max(dqs)
        else:
            mean_dq = max_dq = None

        row = {
            "config":label, "N":N_actual, "K":K, "w_limit":w_limit, "w_smooth":w_smooth,
            "Loose_SR":f"{loose_sr:.4f}", "Medium_SR":f"{med_sr:.4f}", "Strict_SR":f"{strict_sr:.4f}",
            "monotonic":mono,
            "pos_p50_all":f"{np.median(pos):.2f}", "pos_p95_all":f"{np.percentile(pos,95):.2f}",
            "pos_p95_suc":f"{np.percentile(pos_suc,95):.2f}" if len(pos_suc)>0 else "N/A",
            "rot_p95_all":f"{np.percentile(rot,95):.4f}",
            "rot_p95_suc":f"{np.percentile(rot_suc,95):.4f}" if len(rot_suc)>0 else "N/A",
            "iter_mean":f"{np.mean([t['iterations'] for t in pt]):.1f}",
            "near_limit":f"{near_lim:.3f}" if near_lim is not None else "N/A",
            "mean_dq":f"{mean_dq:.4f}" if mean_dq is not None else "N/A",
            "max_dq":f"{max_dq:.4f}" if max_dq is not None else "N/A",
            "time_s":f"{dt:.1f}",
        }
        results.append(row)

        traj_str = f" dq_mean={mean_dq:.4f} dq_max={max_dq:.4f}" if mean_dq else ""
        lim_str = f" near_lim={near_lim:.3f}" if near_lim else ""
        print(f"  {label:<25} N={N_actual:>4} K={K:>2} "
              f"L={loose_sr:.3f} M={med_sr:.3f} S={strict_sr:.3f} {mono} "
              f"p95a={np.percentile(pos,95):.1f}mm p95s={np.percentile(pos_suc,95):.1f}mm "
              f"iters={np.mean([t['iterations'] for t in pt]):.0f}{lim_str}{traj_str}")

    # ---- Save CSV ----
    csv_path = DATA_DIR / "results" / "v4_m0_results.csv"
    with open(csv_path,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    # ---- Generate MD report ----
    lines = [
        "# V4 M0 实验报告：V3 Freeze + Limit Barrier + Smoothness",
        f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n## 总体结果\n",
        "| 配置 | N | K | Loose SR | Medium SR | Strict SR | 单调 | pos_p95_all | pos_p95_suc | iters | time/s |",
        "|------|---|---|----------|-----------|-----------|---|-------------|-------------|-------|--------|",
    ]
    for r in results:
        lines.append(f"| {r['config']} | {r['N']} | {r['K']} | {r['Loose_SR']} | {r['Medium_SR']} | {r['Strict_SR']} | {r['monotonic']} | {r['pos_p95_all']} | {r['pos_p95_suc']} | {r['iter_mean']} | {r['time_s']} |")

    # V3 freeze analysis
    lines += ["\n## V3 Freeze: Sobol-K16 vs Sobol-K32\n"]
    v3_rows = [r for r in results if r["w_limit"]=="0.0" and r["w_smooth"]=="0.0" and "traj" not in r["config"].lower() and "Smooth" not in r["config"]]
    for r in v3_rows:
        lines.append(f"- **{r['config']}** (N={r['N']}): Loose={r['Loose_SR']}, Medium={r['Medium_SR']}, Strict={r['Strict_SR']}, pos_p95_all={r['pos_p95_all']}mm, pos_p95_suc={r['pos_p95_suc']}mm")

    # Limit barrier analysis
    lines += ["\n## Limit Barrier (Module B)\n"]
    lines.append("| w_limit | N | Strict SR | near_limit_ratio | pos_p95_all | pos_p95_suc |")
    lines.append("|---------|---|-----------|-----------------|-------------|-------------|")
    for r in results:
        if r["w_limit"] != "0.0" and r["w_smooth"]=="0.0":
            lines.append(f"| {r['w_limit']} | {r['N']} | {r['Strict_SR']} | {r['near_limit']} | {r['pos_p95_all']} | {r['pos_p95_suc']} |")

    # Smoothness analysis
    lines += ["\n## Trajectory Smoothness (Module C)\n"]
    lines.append("| w_smooth | N_waypoints | Strict SR | mean_Δq/rad | max_Δq/rad | pos_p95_all |")
    lines.append("|----------|------------|-----------|-------------|------------|-------------|")
    for r in results:
        if r["w_smooth"] != "0.0" or ("Smooth" in r["config"] and r["w_smooth"]=="0.0"):
            lines.append(f"| {r['w_smooth']} | {r['N']} | {r['Strict_SR']} | {r['mean_dq']} | {r['max_dq']} | {r['pos_p95_all']} |")

    # Conclusion
    lines += ["\n## 结论\n"]
    v3_k16_200 = [r for r in results if r["config"]=="V3-Sobol-K16" and int(r["N"])==200]
    v3_k16_1000 = [r for r in results if r["config"]=="V3-Sobol-K16" and int(r["N"])==1000]
    if v3_k16_200 and v3_k16_1000:
        lines.append(f"- **V3 Sobol-K16 N=200**: Strict SR={v3_k16_200[0]['Strict_SR']}, pos_p95_all={v3_k16_200[0]['pos_p95_all']}mm, monotonic={v3_k16_200[0]['monotonic']}")
        lines.append(f"- **V3 Sobol-K16 N=1000**: Strict SR={v3_k16_1000[0]['Strict_SR']}, pos_p95_all={v3_k16_1000[0]['pos_p95_all']}mm, monotonic={v3_k16_1000[0]['monotonic']}")

    report_path = DOCS_DIR / "v4_m0_results.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to {report_path}")
    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
