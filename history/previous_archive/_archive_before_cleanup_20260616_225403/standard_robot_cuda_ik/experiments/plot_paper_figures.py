#!/usr/bin/env python3
"""Generate all 8 figures for the revised paper with correct numbering.

Figure order matches paper:
  图1: CUDA block/target mapping architecture
  图2: Throughput-N curves (raw + valid)
  图3: End-to-End latency
  图4: GPU Stream time
  图5: Error distribution
  图6: Ablation (A0-A7, key levels highlighted)
  图7: cuRobo degradation analysis
  图8: Nsight Compute profiling
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK SC", "Noto Serif CJK SC", "AR PL UMing CN", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.unicode_minus": False,
})

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).resolve().parent

C = plt.cm.tab10.colors
CUDA_C = C[0]; CUROBO_NG_C = C[3]; CUROBO_G_C = C[2]
N_VALS = [100, 500, 1000, 4000, 5000, 10000]

# ============================================================
# Data loaders
# ============================================================
def load_perf():
    p = DATA_DIR / "01_official_ur10_main" / "performance_summary.csv"
    if not p.exists(): return []
    with open(p) as f: return list(csv.DictReader(f))

def load_ablation():
    p = DATA_DIR / "04_ablation" / "ablation_full.csv"
    if not p.exists(): return []
    with open(p) as f: return list(csv.DictReader(f))

# ============================================================
# 图 1: Architecture
# ============================================================
def fig1():
    fig, ax = plt.subplots(figsize=(8, 5))
    # Grid
    ax.add_patch(mpatches.FancyBboxPatch((0.05, 0.1), 0.25, 0.8, boxstyle="round,pad=0.02",
        facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=2))
    ax.text(0.175, 0.95, "Grid(N,1,1)\nN blocks", ha="center", va="top", fontsize=9, fontweight="bold")
    # Block 0
    ax.add_patch(mpatches.FancyBboxPatch((0.35, 0.5), 0.30, 0.4, boxstyle="round,pad=0.02",
        facecolor="#FFF3E0", edgecolor="#E65100", linewidth=2))
    ax.text(0.50, 0.93, "Block 0 (128 threads)", ha="center", va="top", fontsize=9, fontweight="bold")
    regions = [
        (0.37, 0.75, 0.26, 0.10, "Thread 0: FK / LDLT / Damping", "#C8E6C9"),
        (0.37, 0.60, 0.26, 0.10, "Thread 0-5: Jacobian cols / g vec", "#BBDEFB"),
        (0.37, 0.45, 0.26, 0.10, "Thread 0-35: Hessian H (36 elems)", "#F8BBD0"),
        (0.37, 0.30, 0.26, 0.10, "Warp 0-3: __syncthreads barriers", "#FFF9C4"),
    ]
    for x, y, w, h, label, color in regions:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
            facecolor=color, edgecolor="gray", linewidth=0.5))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=7)
    # Block N-1
    ax.add_patch(mpatches.FancyBboxPatch((0.70, 0.5), 0.25, 0.4, boxstyle="round,pad=0.02",
        facecolor="#F3E5F5", edgecolor="#7B1FA2", linewidth=2))
    ax.text(0.825, 0.93, "Block N-1", ha="center", va="top", fontsize=9, fontweight="bold")
    ax.text(0.825, 0.70, "... same ...", ha="center", va="center", fontsize=8)
    # Iteration arrow
    ax.annotate("", xy=(0.60, 0.15), xytext=(0.42, 0.15),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5, connectionstyle="arc3,rad=-0.3"))
    ax.text(0.51, 0.08, "Kmax=160 iterations inside single kernel", ha="center", fontsize=7, color="red")
    ax.text(0.17, 0.02, "1 kernel launch = N blocks parallel", ha="center", fontsize=8, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("图 1  1 block/target CUDA parallel mapping and single-kernel iteration structure", fontweight="bold", pad=15)
    fig.savefig(OUT_DIR / "fig1_architecture.png"); plt.close(fig)
    print("  fig1_architecture.png")

# ============================================================
# 图 2: Throughput-N
# ============================================================
def fig2():
    data = load_perf()
    if not data: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    Ns = sorted(set(int(d["N"]) for d in data))

    def get(method, graph_val, key):
        out = []
        for N in Ns:
            found = None
            for d in data:
                if int(d["N"]) == N and d["method"] == method and d.get("graph","N/A") == graph_val:
                    found = float(d.get(key, 0))
                    break
            out.append(found if found else None)
        return out

    c_raw, c_valid = get("CUDA-Mixed","N/A","raw_targets_per_s"), get("CUDA-Mixed","N/A","valid_targets_per_s")
    cn_raw, cn_valid = get("cuRobo","Off","raw_targets_per_s"), get("cuRobo","Off","valid_targets_per_s")
    cg_raw, cg_valid = get("cuRobo","On","raw_targets_per_s"), get("cuRobo","On","valid_targets_per_s")

    for ax, ylabel, c_y, cn_y, cg_y in [
        (ax1, "Raw Throughput (targets/s)", c_raw, cn_raw, cg_raw),
        (ax2, "Valid Throughput (targets/s)", c_valid, cn_valid, cg_valid),
    ]:
        ax.plot(Ns, c_y, "o-", color=CUDA_C, label="CUDA-Mixed", ms=6)
        ax.plot(Ns, cn_y, "s--", color=CUROBO_NG_C, label="cuRobo-NoGraph", ms=6)
        ax.plot(Ns, cg_y, "^:", color=CUROBO_G_C, label="cuRobo-Graph", ms=6)
        # Annotate N=5000 sweet spot
        ax.annotate("batch-size\nsweet spot", xy=(5000, cg_y[Ns.index(5000)]),
            xytext=(3500, cg_y[Ns.index(5000)]*0.3), fontsize=7, color="red",
            arrowprops=dict(arrowstyle="->", color="red"))
        ax.set_xlabel("Batch Size N"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xscale("log"); ax.set_yscale("log")

    fig.suptitle("图 2  Batch Throughput Comparison (Raw vs Valid)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_throughput.png"); plt.close(fig)
    print("  fig2_throughput.png")

# ============================================================
# 图 3: E2E Latency
# ============================================================
def fig3():
    data = load_perf()
    if not data: return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    Ns = sorted(set(int(d["N"]) for d in data))
    c_e2e, cn_e2e, cg_e2e = [], [], []
    for N in Ns:
        for d in data:
            if int(d["N"]) != N: continue
            e2e = float(d.get("e2e_time_ms", 0))
            if d["method"]=="CUDA-Mixed" and d.get("graph","") in ("N/A",""): c_e2e.append(e2e)
            elif d["method"]=="cuRobo" and d.get("graph")=="Off": cn_e2e.append(e2e)
            elif d["method"]=="cuRobo" and d.get("graph")=="On": cg_e2e.append(e2e)
    ax.plot(Ns, c_e2e, "o-", color=CUDA_C, label="CUDA-Mixed (H2D+Kernel+D2H)", ms=6)
    ax.plot(Ns, cn_e2e, "s--", color=CUROBO_NG_C, label="cuRobo-NoGraph", ms=6)
    ax.plot(Ns, cg_e2e, "^:", color=CUROBO_G_C, label="cuRobo-Graph", ms=6)
    ax.set_xlabel("Batch Size N"); ax.set_ylabel("End-to-End Latency (ms)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title("图 3  End-to-End Latency Comparison", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_e2e.png"); plt.close(fig)
    print("  fig3_e2e.png")

# ============================================================
# 图 4: GPU Stream Time
# ============================================================
def fig4():
    data = load_perf()
    if not data: return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    Ns = sorted(set(int(d["N"]) for d in data))
    c_ms, cn_ms, cg_ms = [], [], []
    for N in Ns:
        for d in data:
            if int(d["N"]) != N: continue
            ms = float(d.get("time_ms_mean", 0))
            if d["method"]=="CUDA-Mixed" and d.get("graph","") in ("N/A",""): c_ms.append(ms)
            elif d["method"]=="cuRobo" and d.get("graph")=="Off": cn_ms.append(ms)
            elif d["method"]=="cuRobo" and d.get("graph")=="On": cg_ms.append(ms)
    ax.plot(Ns, c_ms, "o-", color=CUDA_C, label="CUDA-Mixed", ms=6)
    ax.plot(Ns, cn_ms, "s--", color=CUROBO_NG_C, label="cuRobo-NoGraph", ms=6)
    ax.plot(Ns, cg_ms, "^:", color=CUROBO_G_C, label="cuRobo-Graph", ms=6)
    ax.set_xlabel("Batch Size N"); ax.set_ylabel("GPU Stream Time (ms)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title("图 4  GPU Stream Execution Time Comparison", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_gpu_stream.png"); plt.close(fig)
    print("  fig4_gpu_stream.png")

# ============================================================
# 图 5: Error Distribution
# ============================================================
def fig5():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    methods = ["CUDA-Mixed", "cuRobo-Graph"]
    pos_p50 = [3.81, 0.0001]; pos_p95 = [9.19, 0.0001]
    rot_p50 = [0.121, 0.0001]; rot_p95 = [0.541, 0.0001]

    x = np.arange(len(methods)); w = 0.35
    ax1.bar(x-w/2, pos_p50, w, color="#BBDEFB", label="p50 position error")
    ax1.bar(x+w/2, pos_p95, w, color="#FFCDD2", label="p95 position error")
    ax1.set_xticks(x); ax1.set_xticklabels(methods)
    ax1.set_ylabel("Position Error (mm)"); ax1.legend(fontsize=7)
    ax1.set_title("Position Error", fontsize=10)

    ax2.bar(x-w/2, rot_p50, w, color="#BBDEFB", label="p50 rotation error")
    ax2.bar(x+w/2, rot_p95, w, color="#FFCDD2", label="p95 rotation error")
    ax2.set_xticks(x); ax2.set_xticklabels(methods)
    ax2.set_ylabel("Rotation Error (deg)"); ax2.legend(fontsize=7)
    ax2.set_title("Rotation Error", fontsize=10)
    ax2.annotate("cuRobo: sub-micron\nprecision", xy=(1, 0.05), fontsize=8, color="green")

    fig.suptitle("图 5  Error Distribution Comparison (N=500, Medium Threshold)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_errors.png"); plt.close(fig)
    print("  fig5_errors.png")

# ============================================================
# 图 6: Ablation (key levels highlighted)
# ============================================================
def fig6():
    abl_data = load_ablation()
    if not abl_data: return

    # Extract key levels: A0, A5, A6, A7 at N=100,500,5000
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    N_plot = [100, 500, 1000, 5000]
    levels_all = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    labels = ["A0\nGlobalMem", "A1\n+Const", "A2\n+Pad", "A3\n+LDLT",
              "A4\n+Fuse", "A5\n+Damp", "A6\n+Clamp", "A7\n+Mixed"]
    key_colors = ["#E0E0E0"]*5 + ["#FF9800", "#64B5F6", "#4CAF50"]  # A5/A6/A7 highlighted

    for ax, N in zip(axes.flat, N_plot):
        vals = []
        for lvl in levels_all:
            match = [r for r in abl_data if r["ablation_level"].startswith(lvl) and int(r["N"])==N]
            vals.append(float(match[0]["throughput_targets_per_s"]) if match else 0)
        bars = ax.bar(range(8), vals, color=key_colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(8)); ax.set_xticklabels(labels, fontsize=6)
        ax.set_ylabel("Throughput (targets/s)"); ax.set_title(f"N={N}", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, vals):
            if val > 0:
                label = f"{val/1000:.0f}k" if val>10000 else f"{val:.0f}"
                ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+bar.get_height()*0.01,
                        label, ha="center", va="bottom", fontsize=5.5)

    fig.suptitle("图 6  Ablation: Incremental Optimization Throughput (A0->A7)", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_ablation.png"); plt.close(fig)
    print("  fig6_ablation.png")

# ============================================================
# 图 7: cuRobo Degradation
# ============================================================
def fig7():
    data = load_perf()
    if not data: return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    N_plot = [100, 500, 1000, 4000, 5000, 10000]
    cn_times = []
    for N in N_plot:
        found = [d for d in data if int(d["N"])==N and d["method"]=="cuRobo" and d.get("graph")=="Off"]
        cn_times.append(float(found[0]["time_ms_mean"]) if found else 0)
    colors = ["#FF8A65" if t>100 else "#81C784" for t in cn_times]
    ax.bar(range(len(N_plot)), cn_times, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(N_plot))); ax.set_xticklabels([str(n) for n in N_plot])
    ax.set_ylabel("GPU Stream Time (ms)"); ax.set_xlabel("Batch Size N")
    for i, (n, t) in enumerate(zip(N_plot, cn_times)):
        label = "DEGRADED\n{:.0f}ms".format(t) if t>100 else "{:.0f}ms".format(t)
        ax.text(i, t+2, label, ha="center", fontsize=7, color="red" if t>100 else "green")
    ax.set_title("图 7  cuRobo Batch-Size Degradation (NoGraph mode)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_degradation.png"); plt.close(fig)
    print("  fig7_degradation.png")

# ============================================================
# 图 8: Nsight Compute Profiling
# ============================================================
def fig8():
    metrics = ["Compute\nThroughput %", "DRAM\nThroughput %", "Occupancy\n%",
               "L1 Hit\nRate %", "Bank\nConflicts*", "Registers\n/Thread"]
    fp64_v = [66.89, 1.56, 32.51, 99.13, 100, 94]
    mixed_v = [60.73, 1.16, 33.30, 98.62, 1295/3522*100, 98]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(metrics)); w = 0.35
    ax.bar(x-w/2, fp64_v, w, label="FP64 Full Optimized", color="#64B5F6", edgecolor="black", linewidth=0.5)
    ax.bar(x+w/2, mixed_v, w, label="CUDA Mixed Precision", color="#81C784", edgecolor="black", linewidth=0.5)
    ax.annotate("-63% bank conflicts", xy=(4+w/2, mixed_v[4]), xytext=(4.5, 85),
                arrowprops=dict(arrowstyle="->", color="red"), fontsize=9, color="red", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylabel("Value"); ax.legend(fontsize=8)
    ax.set_title("图 8  Nsight Compute Profiling Comparison (N=100)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8_ncu.png"); plt.close(fig)
    print("  fig8_ncu.png")

# ============================================================
if __name__ == "__main__":
    print("Generating 8 figures...")
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
    print(f"Done. Saved to {OUT_DIR}/")
