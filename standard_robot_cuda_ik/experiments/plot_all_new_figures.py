#!/usr/bin/env python3
"""Generate all 9 figures for the revised paper.

Reads experimental CSV data and produces publication-quality figures
using matplotlib + seaborn. Output saved to experiments/figures/.

Requirements: pip install matplotlib seaborn numpy
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# Style configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK SC", "Noto Serif CJK SC", "AR PL UMing CN", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.unicode_minus": False,
})

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tableau 10 color palette
C = plt.cm.tab10.colors
CUDA_COLOR = C[0]      # blue
CUROBO_NG_COLOR = C[3]  # red
CUROBO_G_COLOR = C[2]   # green

# ---- Load Data ----
def load_perf_data() -> list[dict]:
    path = Path(__file__).resolve().parent / "01_official_ur10_main" / "performance_summary.csv"
    if not path.exists():
        print(f"WARNING: {path} not found, using placeholder data")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def load_jacobian_data() -> list[dict]:
    path = Path(__file__).resolve().parent / "05_roofline" / "jacobian_precision_summary.csv"
    if not path.exists():
        print(f"WARNING: {path} not found")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


# ---- Figure 1: CUDA Block/Target Mapping Architecture ----
def plot_fig1_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Grid representation
    grid_box = mpatches.FancyBboxPatch((0.05, 0.1), 0.25, 0.8, boxstyle="round,pad=0.02",
                                        facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=2)
    ax.add_patch(grid_box)
    ax.text(0.175, 0.95, "Grid(N,1,1)\nN blocks", ha="center", va="top", fontsize=9, fontweight="bold")

    # Block 0 detail
    block_box = mpatches.FancyBboxPatch((0.35, 0.5), 0.30, 0.4, boxstyle="round,pad=0.02",
                                         facecolor="#FFF3E0", edgecolor="#E65100", linewidth=2)
    ax.add_patch(block_box)
    ax.text(0.50, 0.93, "Block 0 (128 threads)", ha="center", va="top", fontsize=9, fontweight="bold")

    # Thread layout inside block
    regions = [
        (0.37, 0.75, 0.26, 0.10, "Thread 0: FK / LDLT / Damping", "#C8E6C9"),
        (0.37, 0.60, 0.26, 0.10, "Thread 0-5: Jacobian cols / g vec", "#BBDEFB"),
        (0.37, 0.45, 0.26, 0.10, "Thread 0-35: Hessian H (36 elems)", "#F8BBD0"),
        (0.37, 0.30, 0.26, 0.10, "Warp 0-3: __syncthreads barriers", "#FFF9C4"),
    ]
    for x, y, w, h, label, color in regions:
        r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                     facecolor=color, edgecolor="gray", linewidth=0.5)
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=7)

    # Block N-1
    block_n = mpatches.FancyBboxPatch((0.70, 0.5), 0.25, 0.4, boxstyle="round,pad=0.02",
                                       facecolor="#F3E5F5", edgecolor="#7B1FA2", linewidth=2)
    ax.add_patch(block_n)
    ax.text(0.825, 0.93, "Block N-1", ha="center", va="top", fontsize=9, fontweight="bold")
    ax.text(0.825, 0.70, "... same structure ...", ha="center", va="center", fontsize=8)

    # Iteration loop arrow
    ax.annotate("", xy=(0.60, 0.15), xytext=(0.42, 0.15),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5, connectionstyle="arc3,rad=-0.3"))
    ax.text(0.51, 0.08, "K_max=160 iterations\ninside single kernel", ha="center", fontsize=7, color="red")

    # Labels
    ax.text(0.17, 0.02, "1 kernel launch = N blocks parallel", ha="center", fontsize=8, style="italic")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("图 1  1 block/target CUDA 并行映射与单 kernel 全迭代结构", fontweight="bold", pad=15)

    fig.savefig(OUT_DIR / "fig1_block_target_mapping.png")
    plt.close(fig)
    print("  Figure 1 saved.")


# ---- Figure 2: Throughput-N Curve ----
def plot_fig2_throughput():
    data = load_perf_data()
    if not data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    N_vals = sorted(set(int(d["N"]) for d in data))

    # Helper to extract per-method data
    cuda_raw, cuda_valid = [], []
    cn_raw, cn_valid, cg_raw, cg_valid = [], [], [], []

    for N in N_vals:
        for d in data:
            if int(d["N"]) != N:
                continue
            tp = float(d.get("raw_targets_per_s", 0))
            vt = float(d.get("valid_targets_per_s", 0))
            if d["method"] == "CUDA-Mixed" and d.get("graph", "") in ("N/A", ""):
                cuda_raw.append(tp)
                cuda_valid.append(vt)
            elif d["method"] == "cuRobo" and d.get("graph") == "Off":
                cn_raw.append(tp)
                cn_valid.append(vt)
            elif d["method"] == "cuRobo" and d.get("graph") == "On":
                cg_raw.append(tp)
                cg_valid.append(vt)

    for ax, ylabel, cuda_y, cn_y, cg_y in [
        (ax1, "Raw Throughput (targets/s)", cuda_raw, cn_raw, cg_raw),
        (ax2, "Valid Throughput (targets/s)", cuda_valid, cn_valid, cg_valid),
    ]:
        ax.plot(N_vals[:len(cuda_y)], cuda_y, "o-", color=CUDA_COLOR, label="CUDA-Mixed", ms=6)
        ax.plot(N_vals[:len(cn_y)], cn_y, "s--", color=CUROBO_NG_COLOR, label="cuRobo-NoGraph", ms=6)
        ax.plot(N_vals[:len(cg_y)], cg_y, "^:", color=CUROBO_G_COLOR, label="cuRobo-Graph", ms=6)
        ax.set_xlabel("Batch Size N")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")

    fig.suptitle("图 2  批量吞吐对比（Raw vs Valid Throughput）", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_throughput_n_curve.png")
    plt.close(fig)
    print("  Figure 2 saved.")


# ---- Figure 3: End-to-End Latency ----
def plot_fig3_e2e_latency():
    data = load_perf_data()
    if not data:
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    N_vals = sorted(set(int(d["N"]) for d in data))

    cuda_e2e, cn_e2e, cg_e2e = [], [], []
    for N in N_vals:
        for d in data:
            if int(d["N"]) != N:
                continue
            e2e = float(d.get("e2e_time_ms", 0))
            if d["method"] == "CUDA-Mixed" and d.get("graph", "") in ("N/A", ""):
                cuda_e2e.append(e2e)
            elif d["method"] == "cuRobo" and d.get("graph") == "Off":
                cn_e2e.append(e2e)
            elif d["method"] == "cuRobo" and d.get("graph") == "On":
                cg_e2e.append(e2e)

    ax.plot(N_vals[:len(cuda_e2e)], cuda_e2e, "o-", color=CUDA_COLOR, label="CUDA-Mixed (H2D+Kernel+D2H)", ms=6)
    ax.plot(N_vals[:len(cn_e2e)], cn_e2e, "s--", color=CUROBO_NG_COLOR, label="cuRobo-NoGraph", ms=6)
    ax.plot(N_vals[:len(cg_e2e)], cg_e2e, "^:", color=CUROBO_G_COLOR, label="cuRobo-Graph", ms=6)
    ax.set_xlabel("Batch Size N")
    ax.set_ylabel("End-to-End Latency (ms)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("图 3  端到端延迟对比", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_e2e_latency.png")
    plt.close(fig)
    print("  Figure 3 saved.")


# ---- Figure 4: GPU Stream Time ----
def plot_fig4_gpu_stream():
    data = load_perf_data()
    if not data:
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    N_vals = sorted(set(int(d["N"]) for d in data))

    cuda_ms, cn_ms, cg_ms = [], [], []
    for N in N_vals:
        for d in data:
            if int(d["N"]) != N:
                continue
            ms = float(d.get("time_ms_mean", 0))
            if d["method"] == "CUDA-Mixed" and d.get("graph", "") in ("N/A", ""):
                cuda_ms.append(ms)
            elif d["method"] == "cuRobo" and d.get("graph") == "Off":
                cn_ms.append(ms)
            elif d["method"] == "cuRobo" and d.get("graph") == "On":
                cg_ms.append(ms)

    ax.plot(N_vals[:len(cuda_ms)], cuda_ms, "o-", color=CUDA_COLOR, label="CUDA-Mixed", ms=6)
    ax.plot(N_vals[:len(cn_ms)], cn_ms, "s--", color=CUROBO_NG_COLOR, label="cuRobo-NoGraph", ms=6)
    ax.plot(N_vals[:len(cg_ms)], cg_ms, "^:", color=CUROBO_G_COLOR, label="cuRobo-Graph", ms=6)
    ax.set_xlabel("Batch Size N")
    ax.set_ylabel("GPU Stream Time (ms)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("图 4  GPU Stream 执行时间对比", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_gpu_stream_time.png")
    plt.close(fig)
    print("  Figure 4 saved.")


# ---- Figure 5: cuRobo Degradation vs Kernel Launch Count ----
def plot_fig5_curobo_degradation():
    """Uses Nsight Systems summary (or placeholder) to show kernel launch vs time."""
    # Placeholder: extract from performance_summary for cuRobo-NoGraph
    data = load_perf_data()
    if not data:
        return

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 4.5))

    N_vals = [100, 500, 1000, 4000, 5000, 10000]
    cn_times = []
    for N in N_vals:
        for d in data:
            if int(d["N"]) == N and d["method"] == "cuRobo" and d.get("graph") == "Off":
                cn_times.append(float(d.get("time_ms_mean", 0)))
                break

    # Qualitative: annotate degradation points
    ax1.bar(range(len(N_vals)), cn_times, color=[CUROBO_NG_COLOR if t > 100 else C[1] for t in cn_times])
    ax1.set_xticks(range(len(N_vals)))
    ax1.set_xticklabels([str(n) for n in N_vals])
    ax1.set_ylabel("GPU Stream Time (ms)")
    ax1.set_xlabel("Batch Size N")
    ax1.set_title("图 5  cuRobo 批量规模退化点分析（无Graph模式）", fontweight="bold")

    # Annotate
    for i, (n, t) in enumerate(zip(N_vals, cn_times)):
        label = "DEGRADED" if t > 100 else "normal"
        color = "red" if t > 100 else "green"
        ax1.text(i, t + 2, f"{t:.0f}ms\n{label}", ha="center", fontsize=7, color=color)

    ax1.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_curobo_degradation.png")
    plt.close(fig)
    print("  Figure 5 saved.")


# ---- Figure 6: Ablation Bar Chart ----
def plot_fig6_ablation():
    """Manual ablation data from the experiment output."""
    # Data from run_full_ablation.py output
    levels = ["A0\nGlobalMem", "A1\n+ConstMem", "A2\n+PaddedMat",
              "A3\n+RegLDLT", "A4\n+KernelFus", "A5\n+AdapDamp",
              "A6\n+StepClamp", "A7\n+MixedPrec"]
    N_groups = ["N=100", "N=500", "N=1000", "N=5000"]

    # throughput data from ablation experiment
    tp = {
        "N=100":  [112705, 10635, 10625, 10664, 10672, 61773, 47121, 144188],
        "N=500":  [90274, 10199, 10221, 10216, 10231, 77074, 47123, 131249],
        "N=1000": [108792, 9166, 9128, 9167, 9195, 57761, 44335, 146817],
        "N=5000": [122987, 12642, 12690, 12737, 12750, 61151, 47343, 158905],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ["#E0E0E0"] * 5 + ["#FF9800", "#2196F3", "#4CAF50"]

    for ax, N_label in zip(axes.flat, N_groups):
        vals = tp[N_label]
        bars = ax.bar(range(len(levels)), vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels, fontsize=7)
        ax.set_ylabel("Throughput (targets/s)")
        ax.set_title(N_label, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on top
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            label = f"{val/1000:.0f}k" if val > 10000 else f"{val:.0f}"
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    label, ha="center", va="bottom", fontsize=6)

    fig.suptitle("图 6  逐级消融吞吐提升对比", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_ablation.png")
    plt.close(fig)
    print("  Figure 6 saved.")


# ---- Figure 7: Convergence Rate vs Valid Throughput ----
def plot_fig7_convergence():
    data = load_perf_data()
    if not data:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    methods = []
    sr_vals = []
    vt_vals = []
    for d in data:
        name = d["method"]
        if d.get("graph") == "On":
            name += "-Graph"
        elif d.get("graph") == "Off":
            name += "-NoGraph"
        methods.append(name)
        sr_vals.append(float(d.get("success_rate", 0)))
        vt_vals.append(float(d.get("valid_targets_per_s", 0)))

    x = range(len(methods))
    width = 0.35
    bars1 = ax.bar([i - width/2 for i in x], sr_vals, width, label="Success Rate", color="#42A5F5")
    ax.set_ylabel("Success Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.1)

    ax2 = ax.twinx()
    bars2 = ax2.bar([i + width/2 for i in x], [v/1000 for v in vt_vals], width,
                     label="Valid Throughput (k t/s)", color="#FF7043", alpha=0.7)
    ax2.set_ylabel("Valid Throughput (k targets/s)")

    ax.set_title("图 7  收敛率与有效吞吐对比", fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_convergence_vs_valid_throughput.png")
    plt.close(fig)
    print("  Figure 7 saved.")


# ---- Figure 8: Error Distribution Box Plot ----
def plot_fig8_error_distribution():
    """Simplified box plot from performance summary percentiles."""
    data = load_perf_data()
    if not data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Extract per-method error percentiles at N=500
    pos_data = {}
    rot_data = {}
    for d in data:
        N = int(d["N"])
        if N not in [100, 500, 1000]:
            continue
        name = d["method"]
        if d.get("graph") == "On":
            name += "-G"
        elif d.get("graph") == "Off":
            name += "-NG"
        key = f"{name}_N{N}"
        p50 = float(d.get("pos_error_p50_m", 0)) * 1000  # to mm
        p95 = float(d.get("pos_error_p95_m", 0)) * 1000
        pmax = float(d.get("avg_pos_error_m", 0)) * 1000
        if p50 > 0:
            pos_data[key] = [p50, p95, pmax]
        r50 = float(d.get("rot_error_p50_rad", 0)) * 57.3  # to deg
        r95 = float(d.get("rot_error_p95_rad", 0)) * 57.3
        if r50 > 0:
            rot_data[key] = [r50, r95]

    # Simple bar chart of p50/p95 error
    labels = list(pos_data.keys())[:8]
    x = range(len(labels))

    p50_vals = [pos_data.get(l, [0, 0, 0])[0] for l in labels]
    p95_vals = [pos_data.get(l, [0, 0, 0])[1] for l in labels]

    ax1.bar(x, p95_vals, color="#FFCDD2", label="p95 pos error")
    ax1.bar(x, p50_vals, color="#BBDEFB", label="p50 pos error")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax1.set_ylabel("Position Error (mm)")
    ax1.legend(fontsize=7)
    ax1.set_title("Position Error Distribution", fontsize=10)

    # Rotation errors
    r50_vals = [rot_data.get(l, [0, 0])[0] for l in labels]
    r95_vals = [rot_data.get(l, [0, 0])[1] for l in labels] if any(len(rot_data.get(l, [0])) > 1 for l in labels) else [0] * len(labels)
    ax2.bar(x, r50_vals, color="#BBDEFB", label="p50 rot error")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax2.set_ylabel("Rotation Error (deg)")
    ax2.legend(fontsize=7)
    ax2.set_title("Rotation Error Distribution", fontsize=10)

    fig.suptitle("图 8  误差分布对比（p50/p95）", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8_error_distribution.png")
    plt.close(fig)
    print("  Figure 8 saved.")


# ---- Figure 9: Nsight Compute Profiling ----
def plot_fig9_ncu():
    """NCU profiling metrics comparison."""
    metrics = ["Compute\nThroughput %", "DRAM\nThroughput %", "Occupancy\n%",
               "L1 Hit\nRate %", "Bank\nConflicts*", "Registers\n/Thread"]

    fp64_vals = [66.89, 1.56, 32.51, 99.13, 3522, 94]
    mixed_vals = [60.73, 1.16, 33.30, 98.62, 1295, 98]

    # Normalize bank conflicts for display
    fp64_vals[4] = 100
    mixed_vals[4] = 1295 / 3522 * 100

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, fp64_vals, width, label="FP64 Full Optimized", color="#64B5F6")
    bars2 = ax.bar(x + width/2, mixed_vals, width, label="CUDA Mixed Precision", color="#81C784")

    # Annotate bank conflicts reduction
    ax.annotate("-63%", xy=(4 + width/2, mixed_vals[4]), xytext=(4.5, 85),
                arrowprops=dict(arrowstyle="->", color="red"), fontsize=9, color="red", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)
    ax.set_title("图 9  Nsight Compute 剖析指标对比（N=100）", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig9_ncu_profiling.png")
    plt.close(fig)
    print("  Figure 9 saved.")


# ---- Main ----
def main():
    print("Generating figures...")
    plot_fig1_architecture()
    plot_fig2_throughput()
    plot_fig3_e2e_latency()
    plot_fig4_gpu_stream()
    plot_fig5_curobo_degradation()
    plot_fig6_ablation()
    plot_fig7_convergence()
    plot_fig8_error_distribution()
    plot_fig9_ncu()
    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
