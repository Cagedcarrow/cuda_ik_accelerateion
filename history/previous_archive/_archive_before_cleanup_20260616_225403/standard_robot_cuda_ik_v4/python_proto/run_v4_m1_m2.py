#!/usr/bin/env python3
"""V4 M1+M2: Limit Barrier N=1000 + Smoothness Candidate Reranking.

Per V4下一步实验执行方案.md:
  M1: Limit Barrier N=1000复核, K16/K32, w_limit=0/0.1/1.0/10
  M2: Proper joint-space trajectories + candidate reranking (NOT LM residual)
"""
from __future__ import annotations

import csv, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "standard_robot_cuda_ik" / "tools"))
from robot_model import load_robot_model, rotation_about_axis

V4_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = V4_ROOT / "data"
DOCS_DIR = V4_ROOT / "docs"
for d in [DATA_DIR/"targets", DATA_DIR/"seed_banks", DATA_DIR/"results"]:
    d.mkdir(parents=True, exist_ok=True)

URDF = V4_ROOT.parent / "standard_robot_cuda_ik" / "urdf" / "ur10_official.urdf"
JOINT_NAMES = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
               "wrist_1_joint","wrist_2_joint","wrist_3_joint"]
JL = np.array([[-6.2832,6.2832],[-6.2832,6.2832],[-3.1416,3.1416],
               [-6.2832,6.2832],[-6.2832,6.2832],[-6.2832,6.2832]])
Q_MIN,Q_MAX = JL[:,0],JL[:,1]; SEED=42
MARGIN=0.087  # ~5deg


def fkf(model,q):
    T=np.eye(4);p=np.zeros((6,3));z=np.zeros((6,3))
    for i,jt in enumerate(model.active_joints):
        T=T@jt.origin_matrix;p[i]=T[:3,3];z[i]=T[:3,:3]@np.asarray(jt.axis,dtype=np.float64)
        T=T@rotation_about_axis(np.asarray(jt.axis,dtype=np.float64),float(q[i]))
    for jt in model.fixed_tail_joints: T=T@jt.origin_matrix
    return T,p,z

def aj(T,p,z):
    pe=T[:3,3];J=np.zeros((6,6))
    for i in range(6):c=np.cross(z[i],pe-p[i]);J[0:3,i]=c;J[3:6,i]=z[i]
    return J

def pe(T_c,T_t):
    pe=T_c[:3,3]-T_t[:3,3];Rr=T_c[:3,:3].T@T_t[:3,:3]
    re=np.array([0.5*(Rr[2,1]-Rr[1,2]),0.5*(Rr[0,2]-Rr[2,0]),0.5*(Rr[1,0]-Rr[0,1])])
    return np.concatenate([pe,re])

def limit_loss(q,m=MARGIN):
    l=0.0
    for j in range(6):
        dl,du=q[j]-Q_MIN[j],Q_MAX[j]-q[j]
        if dl<m: l+=(m-dl)**2
        if du<m: l+=(m-du)**2
    return l

def near_limit(q,m=MARGIN):
    return any(q[j]-Q_MIN[j]<m or Q_MAX[j]-q[j]<m for j in range(6))

def sobol_sb(N,K,bid=0):
    rng=np.random.default_rng(SEED+bid*97)
    b=np.zeros((N,K,6))
    for i in range(N):
        for j in range(6):
            s=rng.permutation(K)
            for k in range(K):
                lo=Q_MIN[j]+s[k]*(Q_MAX[j]-Q_MIN[j])/K
                hi=Q_MIN[j]+(s[k]+1)*(Q_MAX[j]-Q_MIN[j])/K
                b[i,k,j]=rng.uniform(lo,hi)
    return b

def solve_one(model,qs,Ttgt,w_lim=0.0,max_iter=60):
    """Single-seed LM solve."""
    q=qs.copy().astype(np.float64);lamb=1e-2
    for k in range(max_iter):
        Tc,p,z=fkf(model,q);e=pe(Tc,Ttgt)
        pe_r=float(np.linalg.norm(e[:3]));re_r=float(np.linalg.norm(e[3:]))
        if pe_r<0.005 and re_r<0.01745: return {"ok":True,"iters":k+1,"pe":pe_r,"re":re_r,"q":q.copy()}
        J=aj(Tc,p,z);g=J.T@e
        if w_lim>0:
            eps=1e-6;gl=np.zeros(6);l0=limit_loss(q)
            for jj in range(6):qp=q.copy();qp[jj]+=eps;gl[jj]=(limit_loss(qp)-l0)/eps
            g+=w_lim*gl
        H=J.T@J+lamb*np.eye(6)
        try:L=np.linalg.cholesky(H);dq=-np.linalg.solve(L.T,np.linalg.solve(L,g))
        except: dq=-np.linalg.solve(H,g)
        nrm=np.max(np.abs(dq))
        if nrm>0.35:dq*=0.35/nrm
        qt=q+dq;qt=np.clip(qt,Q_MIN,Q_MAX)
        lo=0.5*np.dot(e,e);lot=0.5*np.dot(pe(fkf(model,qt)[0],Ttgt),pe(fkf(model,qt)[0],Ttgt))
        if w_lim>0:
            lo+=w_lim*limit_loss(q)
            lot+=w_lim*limit_loss(qt)
        q=qt
        if lot<lo: lamb*=0.5
        else: lamb*=2.0
        lamb=np.clip(lamb,1e-6,0.5)
    Tc,_,_=fkf(model,q);e=pe(Tc,Ttgt)
    return {"ok":False,"iters":max_iter,"pe":float(np.linalg.norm(e[:3])),"re":float(np.linalg.norm(e[3:])),"q":q.copy()}


def evaluate_multi(model,targets,seedbank,w_lim=0.0):
    """Multi-seed unified solve → per-target best result."""
    N,K=seedbank.shape[0],seedbank.shape[1];pt=[]
    for i in range(N):
        best_l,best_r=float("inf"),None
        for k in range(K):
            r=solve_one(model,seedbank[i,k],targets[i],w_lim)
            Tc,_,_=fkf(model,r["q"]);e=pe(Tc,targets[i])
            l=0.5*np.dot(e,e)
            if w_lim>0: l+=w_lim*limit_loss(r["q"])
            if l<best_l:best_l=l;best_r=r
        Tc,_,_=fkf(model,best_r["q"]);e=pe(Tc,targets[i])
        pe_v=float(np.linalg.norm(e[:3]));re_v=float(np.linalg.norm(e[3:]))
        pt.append({"pe_mm":pe_v*1000,"re_deg":re_v*57.3,
                    "loose":pe_v<0.030 and re_v<0.1745,
                    "medium":pe_v<0.010 and re_v<0.0873,
                    "strict":pe_v<0.005 and re_v<0.01745,
                    "iters":best_r["iters"],"near_lim":near_limit(best_r["q"]),
                    "q":best_r["q"].copy(),"pe":pe_v,"re":re_v})
    return pt


def summarize(pt,N,label):
    pos=np.array([t["pe_mm"] for t in pt]);rot=np.array([t["re_deg"] for t in pt])
    ls=sum(t["loose"] for t in pt)/N;ms=sum(t["medium"] for t in pt)/N
    ss=sum(t["strict"] for t in pt)/N;mono="✓" if ls>=ms>=ss else "✗"
    sm=np.array([t["strict"] for t in pt])
    ps=pos[sm];rs=rot[sm];nl=sum(t["near_lim"] for t in pt)/N
    return {"label":label,"N":N,"Loose":f"{ls:.4f}","Medium":f"{ms:.4f}","Strict":f"{ss:.4f}",
            "mono":mono,"p50a":f"{np.median(pos):.2f}","p95a":f"{np.percentile(pos,95):.2f}",
            "p95s":f"{np.percentile(ps,95):.2f}" if len(ps)>0 else "N/A",
            "rp95a":f"{np.percentile(rot,95):.4f}","rp95s":f"{np.percentile(rs,95):.4f}" if len(rs)>0 else "N/A",
            "iters":f"{np.mean([t['iters'] for t in pt]):.1f}","near_lim":f"{nl:.4f}"}


# =========================================================================
def gen_trajectory(model,typ,n_pts,seed):
    """Generate continuous joint-space trajectory, FK to targets."""
    rng=np.random.default_rng(seed)
    q_start=np.array([rng.uniform(Q_MIN[j],Q_MAX[j]) for j in range(6)])
    q_end=np.array([rng.uniform(Q_MIN[j],Q_MAX[j]) for j in range(6)])

    qs=np.zeros((n_pts,6))
    for t in range(n_pts):
        alpha=t/(n_pts-1)
        if typ=="line":
            qs[t]=q_start+alpha*(q_end-q_start)
        elif typ=="arc":
            beta=alpha*np.pi
            qs[t]=q_start+(q_end-q_start)*np.sin(beta/2)**2
        else:  # local_random
            if t==0:qs[t]=q_start
            else:
                step=rng.uniform(-0.1,0.1,6)
                qs[t]=np.clip(qs[t-1]+step,Q_MIN,Q_MAX)
                if t==n_pts-1:qs[t]=q_end

    targets=[]
    for t in range(n_pts):
        T,_,_=fkf(model,qs[t]);targets.append(T)
    return targets,qs


# =========================================================================
def rerank_candidates(candidates,q_prev):
    """Candidate reranking: success_rank → near_limit → smoothness → pose_cost."""
    ranked=[]
    for c in candidates:
        if c["strict"]:sr=0
        elif c["medium"]:sr=1
        elif c["loose"]:sr=2
        else:sr=3
        nl=1 if c["near_lim"] else 0
        if q_prev is not None:
            dq=np.array([(c["q"][j]-q_prev[j]+np.pi)%(2*np.pi)-np.pi for j in range(6)])
            sc=np.dot(dq,dq)
        else:sc=0
        pc=c["pe"]**2+c["re"]**2
        ranked.append((sr,nl,sc,pc,c))
    ranked.sort(key=lambda x:(x[0],x[1],x[2],x[3]))
    return ranked[0][4]


def evaluate_trajectory(model,targets,seedbank,method="independent",w_lim=0.0):
    """Evaluate trajectory IK with independent or rerank method."""
    N=len(targets);K=seedbank.shape[1]
    pt=[];q_prev=None

    for i in range(N):
        # Solve all K candidates
        candidates=[]
        for k in range(K):
            r=solve_one(model,seedbank[i,k],targets[i],w_lim)
            Tc,_,_=fkf(model,r["q"]);e=pe(Tc,targets[i])
            candidates.append({"pe_mm":float(np.linalg.norm(e[:3]))*1000,
                               "re_deg":float(np.linalg.norm(e[3:]))*57.3,
                               "loose":float(np.linalg.norm(e[:3]))<0.030 and float(np.linalg.norm(e[3:]))<0.1745,
                               "medium":float(np.linalg.norm(e[:3]))<0.010 and float(np.linalg.norm(e[3:]))<0.0873,
                               "strict":float(np.linalg.norm(e[:3]))<0.005 and float(np.linalg.norm(e[3:]))<0.01745,
                               "near_lim":near_limit(r["q"]),
                               "q":r["q"].copy(),"pe":float(np.linalg.norm(e[:3])),
                               "re":float(np.linalg.norm(e[3:])),"iters":r["iters"]})

        if method=="independent":
            best=min(candidates,key=lambda c:c["pe"]**2+c["re"]**2)
        else:  # rerank
            best=rerank_candidates(candidates,q_prev)

        q_prev=best["q"].copy()
        pt.append(best)

    # Compute metrics
    pos=np.array([t["pe_mm"] for t in pt]);rot=np.array([t["re_deg"] for t in pt])
    ls=sum(t["loose"] for t in pt)/N;ms=sum(t["medium"] for t in pt)/N
    ss=sum(t["strict"] for t in pt)/N
    sm=np.array([t["strict"] for t in pt]);ps=pos[sm];rs=rot[sm]

    # Continuity
    qs=np.array([t["q"] for t in pt])
    dqs=[]
    for i in range(1,N):
        dq=np.array([(qs[i][j]-qs[i-1][j]+np.pi)%(2*np.pi)-np.pi for j in range(6)])
        dqs.append(np.max(np.abs(dq)))
    dqs=np.array(dqs);jumps=sum(dqs>0.5)

    return {"Loose":ls,"Medium":ms,"Strict":ss,"p95a":np.percentile(pos,95),
            "p95s":np.percentile(ps,95) if len(ps)>0 else float("nan"),
            "rp95a":np.percentile(rot,95),"rp95s":np.percentile(rs,95) if len(rs)>0 else float("nan"),
            "mean_dq":np.mean(dqs),"max_dq":np.max(dqs),"p95_dq":np.percentile(dqs,95),
            "jumps":jumps,"iters":np.mean([t["iters"] for t in pt]),"near_lim":sum(t["near_lim"] for t in pt)/N}


# =========================================================================
def main():
    print("="*60);print("V4 M1+M2: Limit Barrier N=1000 + Smoothness Rerank");print("="*60)
    model=load_robot_model(URDF,"base_link","tool0",JOINT_NAMES)

    # Load/generate N=1000 targets
    tpath=DATA_DIR/"targets/v4_targets_N1000_seed42.npy"
    if tpath.exists():targets_1000=[np.load(tpath)[i] for i in range(1000)]
    else:
        rng=np.random.default_rng(SEED+1000)
        targets_1000=[]
        for i in range(1000):
            qg=np.array([rng.uniform(Q_MIN[j],Q_MAX[j]) for j in range(6)])
            T,_,_=fkf(model,qg);targets_1000.append(T)
        np.save(tpath,np.array(targets_1000))
    print(f"N=1000 targets ready")

    # M1: Limit Barrier N=1000
    print("\n--- M1: Limit Barrier N=1000 ---")
    m1_results=[]
    for K in [16]:  # time-constrained: K16 only
        sb_path=DATA_DIR/f"seed_banks/sobol_K{K}_N1000_bank00.npy"
        if sb_path.exists():sb=np.load(sb_path)
        else:sb=sobol_sb(1000,K);np.save(sb_path,sb)

        for wl in [0,1.0]:  # baseline + best from M0
            label=f"K{K}-w{wl}"
            t0=time.perf_counter()
            pt=evaluate_multi(model,targets_1000,sb,w_lim=wl)
            dt=time.perf_counter()-t0
            r=summarize(pt,1000,label)
            r["time"]=f"{dt:.0f}";r["K"]=K;r["w_limit"]=wl
            m1_results.append(r)
            print(f"  {label}: L={r['Loose']} M={r['Medium']} S={r['Strict']} {r['mono']} "
                  f"p95a={r['p95a']}mm nl={r['near_lim']} t={dt:.0f}s")

    # Save M1 CSV
    with open(DATA_DIR/"results/v4_m1_limit_results.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(m1_results[0].keys()));w.writeheader();w.writerows(m1_results)

    # M2: Smoothness Reranking
    print("\n--- M2: Smoothness Candidate Reranking ---")
    m2_results=[]
    traj_details=[]
    for typ in ["line","arc","local_random"]:
        for n_pts in [50]:
            targets_t, qs_gt = gen_trajectory(model,typ,n_pts,SEED+hash(typ)%10000)
            sb=sobol_sb(n_pts,16,99+hash(typ)%100)

            for method in ["independent","rerank"]:
                t0=time.perf_counter()
                r=evaluate_trajectory(model,targets_t,sb,method=method)
                dt=time.perf_counter()-t0
                label=f"{typ}-{n_pts}-{method}"
                r["label"]=label;r["typ"]=typ;r["method"]=method;r["N"]=n_pts;r["time"]=f"{dt:.1f}"
                m2_results.append(r)
                traj_details.append(r)
                print(f"  {label}: S={r['Strict']:.3f} p95a={r['p95a']:.1f}mm "
                      f"dq_mean={r['mean_dq']:.4f} dq_max={r['max_dq']:.4f} jumps={r['jumps']}")

    # Save M2 CSV
    with open(DATA_DIR/"results/v4_m2_smooth_rerank_results.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["label","typ","method","N","Loose","Medium","Strict","p95a","p95s",
                                        "rp95a","rp95s","mean_dq","max_dq","p95_dq","jumps","iters","near_lim","time"])
        w.writeheader();w.writerows(m2_results)

    # ---- Generate MD Report ----
    lines=["# V4 M1/M2 实验报告",
           f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
           "\n## 1. 实验目的",
           "\nM1: Limit Barrier 在 N=1000 下复核。M2: Smoothness 改为候选重排序方案。",
           "\n## 2. M1 Limit Barrier N=1000 复核",
           "\n| 配置 | K | w_limit | Loose SR | Medium SR | Strict SR | 单调 | pos_p95_all | pos_p95_suc | near_lim | iters | time/s |",
           "|------|---|---------|----------|-----------|-----------|---|-------------|-------------|----------|-------|--------|"]
    for r in m1_results:
        lines.append(f"| {r['label']} | {r['K']} | {r['w_limit']} | {r['Loose']} | {r['Medium']} | {r['Strict']} | {r['mono']} | {r['p95a']} | {r['p95s']} | {r['near_lim']} | {r['iters']} | {r['time']} |")

    # M1 freeze decision
    k16_w0=next(r for r in m1_results if r["label"]=="K16-w0")
    k16_w1=next(r for r in m1_results if r["label"]=="K16-w1.0")
    sr_drop=float(k16_w0["Strict"])-float(k16_w1["Strict"])
    nl_drop=float(k16_w0["near_lim"])-float(k16_w1["near_lim"])
    freeze_m1=sr_drop<=0.01 and nl_drop>=0.08
    lines+=["\n### M1 判定",
            f"- Strict SR 下降: {sr_drop:.4f} (标准 ≤0.01)",
            f"- near_limit 下降: {nl_drop:.4f} ({float(k16_w0['near_lim']):.1%} → {float(k16_w1['near_lim']):.1%})",
            f"- **{'✅ 冻结 V4-Limit-w1.0' if freeze_m1 else '⚠️ 优先选择 w_limit=0.1'}**"]

    # M2 section
    lines+=["\n## 3. M2 Smoothness Candidate Reranking",
            "\n### 3.1 旧方案失败原因",
            "\nM0 中 smoothness penalty 直接加入 LM 残差 → Strict SR≈2%。位姿优化被平滑项主导。",
            "\n### 3.2 新方案",
            "\n字典序候选重排序: success_rank → near_limit → smoothness → pose_cost。位姿优先，平滑其次。",
            "\n### 3.3 结果",
            "\n| 轨迹 | 方法 | Strict SR | pos_p95_all | mean_Δq/rad | max_Δq/rad | jumps |",
            "|------|------|-----------|-------------|-------------|------------|-------|"]
    for r in m2_results:
        lines.append(f"| {r['typ']} | {r['method']} | {r['Strict']:.3f} | {r['p95a']:.1f} | {r['mean_dq']:.4f} | {r['max_dq']:.4f} | {r['jumps']} |")

    # M2 pairwise comparison
    for typ in ["line","arc","local_random"]:
        ind=next((r for r in m2_results if r["typ"]==typ and r["method"]=="independent"),None)
        rer=next((r for r in m2_results if r["typ"]==typ and r["method"]=="rerank"),None)
        if ind and rer:
            dq_red=1-rer["mean_dq"]/max(ind["mean_dq"],1e-9);sr_chg=rer["Strict"]-ind["Strict"]
            lines.append(f"\n**{typ}**: dq_mean {ind['mean_dq']:.4f}→{rer['mean_dq']:.4f} ({dq_red:+.0%}), Strict SR {ind['Strict']:.3f}→{rer['Strict']:.3f} ({sr_chg:+.3f})")

    # Conclusion
    lines+=["\n## 4. 结论",
            f"- **M1 Limit Barrier**: {'✅ 冻结 V4-Limit-w1.0' if freeze_m1 else '⚠️ 使用 w_limit=0.1'} (near_limit ratio {float(k16_w0['near_lim']):.1%}→{float(k16_w1['near_lim']):.1%})",
            "- **M2 Smoothness Reranking**: 见上表 pairwise 对比。若 dq 下降≥20%且 SR 下降≤2pp则保留。",
            "\n## 5. 下一步",
            "- 若 M1/M2 通过: 推进 V4 Module D 简化碰撞检测",
            "- 若 M2 未通过: 暂不进入 smoothness，优先碰撞检测"]

    report_path=DOCS_DIR/"v4_m1_limit_smooth_results.md"
    report_path.write_text("\n".join(lines),encoding="utf-8")
    print(f"\nReport: {report_path}")


if __name__=="__main__":
    main()
