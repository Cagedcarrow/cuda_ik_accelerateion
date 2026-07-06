#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate final revision figures for the CUDA IK paper.

The script writes publication PDFs and quick PNG previews.  In the source
project it reads ``data/experiments/补充实验/results`` and writes
``论文/figures`` plus ``论文/figures_preview``.  In the review package it reads
``数据/CSV`` and writes ``LaTeX工程/figures`` plus ``图片/PNG预览``.  It only
derives secondary matrices from existing experiment rows.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESULTS = ROOT / "data" / "experiments" / "补充实验" / "results"
if SOURCE_RESULTS.exists():
    RESULTS = SOURCE_RESULTS
    FIGURES = ROOT / "论文" / "figures"
    PREVIEW = ROOT / "论文" / "figures_preview"
else:
    RESULTS = ROOT / "数据" / "CSV"
    FIGURES = ROOT / "LaTeX工程" / "figures"
    PREVIEW = ROOT / "图片" / "PNG预览"

FAIR = RESULTS / "fair_curobo_k16_summary.csv"
THRESHOLD = RESULTS / "threshold_scan.csv"
TIME = RESULTS / "kernel_time_breakdown.csv"
SEED_SCAN = RESULTS / "seed_count_scan.csv"
NEAR_SINGULAR = RESULTS / "near_singular_summary.csv"
NEAR_LIMIT = RESULTS / "near_limit_barrier_summary.csv"
BARRIER_SCAN = RESULTS / "barrier_weight_scan.csv"
TRAJ_SUMMARY = RESULTS / "trajectory_continuity_summary.csv"
LM_SCAN = RESULTS / "lm_iter_scan.csv"
TRAJ_CANDIDATES = RESULTS / "trajectory_dump_candidates_random_local_50_N1000_K16.csv"
TRAJ_BEST = RESULTS / "trajectory_dump_best_random_local_50_N1000_K16.csv"


def configure_style() -> None:
    cjk_font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    cjk_family = "Noto Sans CJK SC"
    if cjk_font.exists():
        fm.fontManager.addfont(str(cjk_font))
        cjk_family = fm.FontProperties(fname=str(cjk_font)).get_name()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [cjk_family, "Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.75,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, name: str) -> None:
    pdf = FIGURES / f"{name}.pdf"
    png = PREVIEW / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[write] {pdf.relative_to(ROOT)}")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"empty csv: {path}")
    return df


def pct(v: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=float) * 100.0


def method_label(method: str, k: int | float | str | None = None) -> str:
    base = str(method).replace("OPT4C", "本文方法").replace("cuRobo-Graph", "对比方法")
    base = base.replace("-K16", "-16起点").replace("-K1", "-1起点")
    if "起点" in base:
        return base
    if k is None or pd.isna(k):
        return base
    return f"{base}-{int(k)}起点"


def annotate_panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(
        0.0,
        1.03,
        f"({letter}) {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def draw_thread_mapping() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 3.25))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)

    def box(x, y, w, h, text, fc="#f7f7f7", ec="#333333", lw=0.9, fs=7.3):
        patch = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="square,pad=0.035",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
        return patch

    def arrow(x1, y1, x2, y2, text=None, rad=0.0):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=0.85,
                color="#2f2f2f",
                shrinkA=4,
                shrinkB=4,
                connectionstyle=f"arc3,rad={rad}",
            ),
        )
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, text, ha="center", va="bottom", fontsize=7.2)

    box(0.35, 4.95, 2.3, 0.72, "Grid: N blocks", "#f2f2f2", fs=7.4)
    box(3.15, 4.95, 2.15, 0.72, "Block i", "#f2f2f2", fs=7.4)
    box(5.85, 4.95, 2.35, 0.72, "Target $T_i$", "#ffffff", fs=7.4)
    box(8.85, 4.95, 2.35, 0.72, "Best output $q^*$", "#ffffff", fs=7.4)
    arrow(2.65, 5.65, 3.15, 5.65)
    arrow(5.30, 5.65, 5.85, 5.65)
    arrow(8.20, 5.65, 8.85, 5.65)

    box(0.35, 3.32, 3.15, 0.9, "Lane 0--15\nSobol seeds", "#e6e6e6", fs=7.4)
    box(0.35, 2.20, 3.15, 0.9, "Lane 16--31\ninactive", "#f7f7f7", fs=7.4)
    box(4.25, 3.32, 3.2, 0.9, "Fused FK + Jacobian + LM", "#eeeeee", fs=7.4)
    box(4.25, 2.20, 3.2, 0.9, "Shared candidate buffer", "#f7f7f7", fs=7.4)
    box(8.3, 2.78, 2.5, 1.0, "Lane-0 selection", "#e6e6e6", fs=7.4)
    arrow(3.50, 4.02, 4.25, 4.02)
    arrow(7.45, 4.02, 8.30, 3.70)
    arrow(7.45, 2.80, 8.30, 3.22)
    arrow(9.55, 4.10, 9.70, 5.25, rad=-0.15)
    arrow(4.95, 5.25, 5.25, 4.50, rad=0.15)
    box(0.55, 0.70, 10.15, 0.9, "Candidate generation -> damped LM -> limit regularization -> FK recheck", "#ffffff", "#666666", fs=7.2)
    arrow(1.95, 3.55, 2.10, 1.80, rad=-0.08)
    arrow(5.85, 2.35, 5.85, 1.80)
    arrow(9.55, 3.05, 9.55, 1.80)

    ax.text(0.15, 6.15, "Target-Block kernel mapping", fontsize=9.5, fontweight="bold", ha="left", va="center")
    save_fig(fig, "fig1_thread_mapping")


def draw_time_breakdown() -> None:
    df = read_csv(TIME).sort_values("N")
    show = df[df["N"].isin([100, 500, 1000])].copy()
    y = np.arange(len(show))
    kernel = show["kernel_percent"].to_numpy(dtype=float)
    non_kernel = 100.0 - kernel
    labels = [f"目标数={int(n)}" for n in show["N"]]
    fig, ax = plt.subplots(figsize=(6.2, 2.45), constrained_layout=True)
    ax.barh(y, kernel, color="#bdbdbd", edgecolor="white", linewidth=0.6, label="核心求解")
    ax.barh(y, non_kernel, left=kernel, color="#4d4d4d", edgecolor="white", linewidth=0.6, label="非求解开销")
    for idx, nk in enumerate(non_kernel):
        ax.text(100.35, idx, f"{nk:.3f}%", va="center", ha="left", fontsize=7.5)
    ax.text(100.35, len(show) - 0.25, "非求解开销小于0.3%", va="bottom", ha="left", fontsize=8.0, color="#333333")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 103.2)
    ax.set_xlabel("流时间占比/%")
    ax.set_title("核心求解占主要时间", fontweight="bold")
    ax.grid(axis="x")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.34))
    save_fig(fig, "fig2_time_breakdown")


def draw_static_performance() -> None:
    df = read_csv(FAIR)
    series = [
        ("OPT4C-K16", "#4e79a7", "o", "本文方法-16起点"),
        ("cuRobo-Graph-K16", "#7f7f7f", "s", "对比方法-16起点"),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.95), constrained_layout=True)
    annotate_panel(ax1, "a", "吞吐率")
    annotate_panel(ax2, "b", "严格成功率")
    for method, color, marker, label in series:
        sub = df[df["method"] == method].sort_values("N")
        if sub.empty:
            continue
        ax1.plot(sub["N"], sub["throughput_targets_per_s_mean"] / 1e4, marker=marker, lw=1.5, color=color, label=label)
        ax2.plot(sub["N"], pct(sub["strict_sr"]), marker=marker, lw=1.5, color=color, label=label)
    for ax in (ax1, ax2):
        ax.set_xticks([100, 500, 1000])
        ax.grid(True)
        ax.set_xlabel("目标数量/个")
    ax1.set_ylabel("吞吐率/(万目标/s)")
    ax2.set_ylabel("严格成功率/%")
    ax2.set_ylim(0, 104)
    ax1.legend(frameon=False, loc="best")
    save_fig(fig, "fig3_static_performance")


def draw_pareto_front() -> None:
    df = read_csv(FAIR)
    n1000 = df[(df["N"] == 1000) & (df["method"].isin(["OPT4C-K1", "OPT4C-K16", "cuRobo-Graph-K1", "cuRobo-Graph-K16"]))].copy()
    n1000["label"] = [method_label(m, None) for m in n1000["method"]]
    order = ["本文方法-1起点", "本文方法-16起点", "对比方法-1起点", "对比方法-16起点"]
    n1000["label"] = pd.Categorical(n1000["label"], order, ordered=True)
    n1000 = n1000.sort_values("label")

    fig, ax = plt.subplots(figsize=(5.4, 3.45), constrained_layout=True)
    markers = {"本文方法-1起点": "o", "本文方法-16起点": "s", "对比方法-1起点": "^", "对比方法-16起点": "D"}
    colors = {"本文方法-1起点": "#9ecae1", "本文方法-16起点": "#3182bd", "对比方法-1起点": "#bdbdbd", "对比方法-16起点": "#636363"}
    offsets = {"本文方法-1起点": (-1.05, 5.2), "本文方法-16起点": (0.12, 2.4), "对比方法-1起点": (-1.35, -4.2), "对比方法-16起点": (0.15, -6.0)}
    for _, row in n1000.iterrows():
        x = row["throughput_targets_per_s_mean"] / 1e4
        y = row["strict_sr"] * 100.0
        label = str(row["label"])
        ax.scatter(
            x,
            y,
            s=95,
            marker=markers[label],
            color=colors[label],
            edgecolor="#222222",
            linewidth=0.75,
            zorder=3,
        )
        dx, dy = offsets[label]
        ax.text(x + dx, y + dy, f"95分位={row['pos_p95_all_mm']:.2f} mm", fontsize=7.2, ha="left", va="center")
    ax.annotate("", xy=(7.1, 55), xytext=(5.7, 55), arrowprops=dict(arrowstyle="->", lw=0.8, color="#9e9e9e"))
    ax.text(5.7, 56.6, "吞吐率增大", fontsize=7.0, color="#777777", ha="left")
    ax.annotate("", xy=(0.95, 96), xytext=(0.95, 82), arrowprops=dict(arrowstyle="->", lw=0.8, color="#9e9e9e"))
    ax.text(1.05, 91, "成功率增大", fontsize=7.0, color="#777777", ha="left", rotation=90, va="center")
    ax.set_xlabel("吞吐率/(万目标/s)")
    ax.set_ylabel("严格成功率/%")
    ax.set_title("目标数为1000时的折中关系", fontweight="bold")
    ax.grid(True)
    ax.set_xlim(0.55, 7.9)
    ax.set_ylim(49.8, 101.2)
    handles = [
        Line2D([0], [0], marker=markers[label], color="none", markerfacecolor=colors[label],
               markeredgecolor="#222222", markersize=6.5, label=label)
        for label in order
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    save_fig(fig, "fig4_pareto_front")


def draw_threshold_scan() -> None:
    df = read_csv(THRESHOLD)
    df = df[df["N"] == 1000].copy()
    order = ["Loose", "Medium", "Strict", "Ultra"]
    order_cn = ["宽松", "中等", "严格", "超严"]
    df["threshold_level"] = pd.Categorical(df["threshold_level"], order, ordered=True)
    methods = [
        ("OPT4C-K16", 16, "#4e79a7", "o"),
        ("OPT4C-K1", 1, "#76b7b2", "s"),
        ("cuRobo-Graph-K16", 16, "#b07aa1", "^"),
        ("cuRobo-Graph-K1", 1, "#e15759", "D"),
    ]
    fig, ax = plt.subplots(figsize=(5.9, 3.25), constrained_layout=True)
    x = np.arange(len(order))
    for method, k, color, marker in methods:
        sub = df[df["method"] == method].sort_values("threshold_level")
        if sub.empty:
            continue
        ax.plot(x, pct(sub["success_rate"]), color=color, marker=marker, lw=1.5, label=method_label(method, None))
    ax.axvspan(1.72, 2.28, color="#f2f2f2", zorder=0)
    ax.text(2, 102.5, "主阈值", ha="center", va="bottom", fontsize=7.5, color="#555555")
    ax.set_xticks(x, order_cn)
    ax.set_ylim(0, 106)
    ax.set_ylabel("成功率/%")
    ax.set_xlabel("阈值等级/类")
    ax.set_title("目标数为1000时的阈值扫描", fontweight="bold")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="lower left", ncol=2)
    save_fig(fig, "fig5_threshold_scan")


def build_candidate_success_matrix() -> pd.DataFrame:
    out = RESULTS / "candidate_success_matrix.csv"
    df = read_csv(TRAJ_CANDIDATES)
    df["rank"] = np.select(
        [df["success_strict"].astype(bool), df["success_medium"].astype(bool), df["success_loose"].astype(bool)],
        [3, 2, 1],
        default=0,
    )
    diff = (
        df.groupby("target_id")
        .agg(strict_seed_count=("success_strict", "sum"), best_pose_cost=("pose_cost", "min"), min_loss=("total_loss", "min"))
        .reset_index()
        .sort_values(["strict_seed_count", "best_pose_cost", "target_id"], ascending=[True, True, True])
    )
    sample_idx = np.linspace(0, len(diff) - 1, 100, dtype=int)
    selected = diff.iloc[sample_idx].copy()
    selected["sorted_target_rank"] = np.arange(len(selected))
    ranked = df.merge(selected[["target_id", "sorted_target_rank"]], on="target_id", how="inner")
    ranked = ranked.sort_values(["sorted_target_rank", "seed_id"])
    keep = [
        "target_id",
        "sorted_target_rank",
        "seed_id",
        "rank",
        "pos_err_mm",
        "rot_err_deg",
        "pose_cost",
        "total_loss",
        "success_loose",
        "success_medium",
        "success_strict",
    ]
    ranked[keep].to_csv(out, index=False)
    print(f"[write] {out.relative_to(ROOT)}")
    return ranked[keep]


def draw_seed_success_heatmap() -> None:
    df = build_candidate_success_matrix()
    matrix = df.pivot(index="sorted_target_rank", columns="seed_id", values="rank").sort_index()
    fig, ax = plt.subplots(figsize=(4.9, 4.15), constrained_layout=True)
    cmap = mcolors.ListedColormap(["#efefef", "#efe6aa", "#b9d7ea", "#2f6f9f"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    im = ax.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_title("目标与起点的成功等级", fontweight="bold")
    ax.set_xlabel("起点编号/个")
    ax.set_ylabel("目标排序/个")
    ax.set_xticks(np.arange(matrix.shape[1]), [str(int(c)) for c in matrix.columns])
    ax.set_yticks(np.arange(0, matrix.shape[0], 20), [str(i) for i in range(0, matrix.shape[0], 20)])
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.035)
    cbar.ax.set_yticklabels(["失败", "宽松", "中等", "严格"])
    save_fig(fig, "fig5_seed_success_heatmap")


def wrap_delta(q1: np.ndarray, q0: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(q1 - q0), np.cos(q1 - q0))


def smooth_select(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prev_q = None
    for target_id in sorted(candidates["target_id"].unique()):
        group = candidates[candidates["target_id"] == target_id].copy()
        if prev_q is None:
            selected = group.sort_values(["success_strict", "total_loss"], ascending=[False, True]).iloc[0]
        else:
            qs = group[[f"q{i}" for i in range(6)]].to_numpy(dtype=float)
            dq = wrap_delta(qs, prev_q)
            smooth_cost = np.linalg.norm(dq, axis=1)
            group = group.assign(smooth_cost=smooth_cost, combined_loss=group["total_loss"].to_numpy(dtype=float) + 0.05 * smooth_cost)
            selected = group.sort_values(["success_strict", "combined_loss"], ascending=[False, True]).iloc[0]
        prev_q = selected[[f"q{i}" for i in range(6)]].to_numpy(dtype=float)
        rows.append(selected)
    return pd.DataFrame(rows)


def build_trajectory_deltaq_matrix() -> pd.DataFrame:
    out = RESULTS / "trajectory_deltaq_matrix.csv"
    candidates = read_csv(TRAJ_CANDIDATES).copy()
    best = read_csv(TRAJ_BEST).copy()
    methods = [("no-rerank", best), ("smoothness-rerank", smooth_select(candidates))]
    records = []
    for method, df in methods:
        df = df.sort_values("target_id").reset_index(drop=True)
        q = df[[f"q{i}" for i in range(6)]].to_numpy(dtype=float)
        for traj_id in range(20):
            start = traj_id * 50
            end = start + 50
            qseg = q[start:end]
            if qseg.shape[0] < 50:
                continue
            d = wrap_delta(qseg[1:], qseg[:-1])
            norms = np.linalg.norm(d, axis=1)
            for step, val in enumerate(norms, start=1):
                records.append(
                    {
                        "trajectory_type": "random_local_50",
                        "trajectory_id": traj_id,
                        "step": step,
                        "method": method,
                        "delta_q_norm": float(val),
                        "is_jump": int(val > 0.5),
                    }
                )
    result = pd.DataFrame(records)
    result.to_csv(out, index=False)
    print(f"[write] {out.relative_to(ROOT)}")
    return result


def draw_trajectory_deltaq_heatmap() -> None:
    df = build_trajectory_deltaq_matrix()
    methods = ["no-rerank", "smoothness-rerank"]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.25), sharey=True, constrained_layout=True)
    vmax = 4.0
    last_im = None
    cmap = mcolors.LinearSegmentedColormap.from_list("soft_blues", ["#ffffff", "#c6dbef", "#2171b5"])
    for ax, method, title in zip(axes, methods, ["未重排序", "平滑重排序"]):
        sub = df[df["method"] == method]
        mat = sub.pivot(index="trajectory_id", columns="step", values="delta_q_norm").sort_index()
        last_im = ax.imshow(np.clip(mat.to_numpy(), 0, vmax), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
        p95 = float(np.percentile(sub["delta_q_norm"].to_numpy(dtype=float), 95))
        jump_ratio = float(sub["is_jump"].mean() * 100.0)
        ax.text(
            0.02,
            0.96,
            f"95分位={p95:.2f} rad\n跳变占比={jump_ratio:.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            bbox=dict(facecolor="white", edgecolor="#bdbdbd", linewidth=0.4, alpha=0.86),
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("相邻步序号/步")
        ax.set_xticks([0, 9, 19, 29, 39, 48], ["1", "10", "20", "30", "40", "49"])
        ax.grid(False)
    axes[0].set_ylabel("轨迹编号/条")
    axes[0].set_yticks([0, 5, 10, 15, 19], ["0", "5", "10", "15", "19"])
    cbar = fig.colorbar(last_im, ax=axes, pad=0.02, fraction=0.045)
    cbar.set_label("关节环绕差分范数/rad")
    save_fig(fig, "fig7_trajectory_deltaq_heatmap")


def draw_robustness_boundary() -> None:
    singular = read_csv(NEAR_SINGULAR)
    near_limit = read_csv(NEAR_LIMIT)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.95), constrained_layout=True)
    ax = axes[0]
    annotate_panel(ax, "a", "近奇异成功率")
    sub = singular[(singular["N"] == 1000) & (singular["K"].isin([1, 16]))].copy()
    types = ["wrist_singular", "elbow_singular", "shoulder_singular"]
    x = np.arange(len(types))
    width = 0.34
    for offset, k, color in [(-width / 2, 1, "#bdbdbd"), (width / 2, 16, "#4e79a7")]:
        vals = [float(sub[(sub["target_type"] == t) & (sub["K"] == k)]["strict_sr"].iloc[0] * 100.0) for t in types]
        ax.bar(x + offset, vals, width=width, color=color, edgecolor="white", label=f"{k}起点")
    ax.set_xticks(x, ["腕部", "肘部", "肩部"])
    ax.set_ylabel("严格成功率/%")
    ax.set_ylim(0, 105)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left", fontsize=7.0)

    ax = axes[1]
    annotate_panel(ax, "b", "近限位比例")
    sub = near_limit[(near_limit["N"] == 1000) & (near_limit["K"] == 16)].copy()
    modes = ["OPT4C-K16-BarrierOFF", "OPT4C-K16-BarrierON"]
    x = np.arange(len(modes))
    near = [float(sub[sub["method"] == m]["near_limit_ratio"].iloc[0] * 100.0) for m in modes]
    ax.bar(x, near, width=0.48, color=["#bdbdbd", "#4e79a7"], edgecolor="white")
    for idx, val in enumerate(near):
        ax.text(idx, val + 0.08, f"{val:.2f}%", ha="center", va="bottom", fontsize=7.4)
    ax.set_xticks(x, ["关闭", "开启"])
    ax.set_ylabel("近限位比例/%")
    ax.set_ylim(0, max(near) * 1.45)
    ax.grid(axis="y")
    save_fig(fig, "fig6_robustness_boundary")


def draw_iteration_tradeoff() -> None:
    df = read_csv(LM_SCAN).sort_values("max_iter")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.05), constrained_layout=True)
    annotate_panel(ax1, "a", "迭代次数与精度")
    ax1.plot(df["max_iter"], pct(df["strict_sr"]), marker="o", color="#4e79a7", lw=1.5, label="严格成功率")
    ax1.set_xlabel("最大迭代次数/次")
    ax1.set_ylabel("严格成功率/%")
    ax1.grid(True)
    ax1b = ax1.twinx()
    ax1b.spines["right"].set_visible(True)
    ax1b.plot(df["max_iter"], df["pos_p95_all_mm"], marker="s", color="#59a14f", lw=1.3, label="位置误差95分位")
    ax1b.set_ylabel("位置误差95分位/mm")
    lines = ax1.get_lines() + ax1b.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], frameon=False, loc="upper center", ncol=2)
    ax1.axvline(60, color="#333333", lw=0.8, ls="--")

    annotate_panel(ax2, "b", "迭代次数与吞吐")
    ax2.plot(df["max_iter"], df["throughput_targets_per_s_mean"] / 1e4, marker="o", color="#e15759", lw=1.5)
    ax2.set_xlabel("最大迭代次数/次")
    ax2.set_ylabel("吞吐率/(万目标/s)")
    ax2.grid(True)
    ax2.axvline(60, color="#333333", lw=0.8, ls="--")
    save_fig(fig, "fig9_iteration_tradeoff")


def main() -> None:
    configure_style()
    ensure_dirs()
    draw_thread_mapping()
    draw_time_breakdown()
    draw_static_performance()
    draw_pareto_front()
    draw_seed_success_heatmap()
    draw_robustness_boundary()
    draw_trajectory_deltaq_heatmap()


if __name__ == "__main__":
    main()
