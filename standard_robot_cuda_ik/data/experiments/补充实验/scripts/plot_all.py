#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

from common_metrics import FIGURES, RESULTS, f, read_csv

FONT_CANDIDATES = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
]
for font_path in FONT_CANDIDATES:
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        CJK_FONT = font_manager.FontProperties(fname=str(font_path)).get_name()
        break
else:
    CJK_FONT = "DejaVu Sans"

plt.rcParams.update({
    "font.family": CJK_FONT,
    "font.sans-serif": [CJK_FONT, "Noto Sans CJK SC", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "figure.dpi": 220,
    "savefig.dpi": 300,
})

COLORS = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "cyan": "#17becf",
    "yellow": "#f2c94c",
    "gray": "#7f7f7f",
    "light_blue": "#dceeff",
    "light_orange": "#ffe8d4",
    "light_green": "#dcf2e3",
    "light_purple": "#eadff7",
}


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_pareto() -> None:
    path = RESULTS / "fair_curobo_k16_summary.csv"
    if not path.exists():
        print(f"skip pareto missing {path}")
        return
    rows = [row for row in read_csv(path) if int(float(row["N"])) == 1000]
    if not rows:
        return
    order = ["cuRobo-Graph-K1", "OPT4C-K1", "OPT4C-K16", "cuRobo-Graph-K16"]
    rows.sort(key=lambda r: order.index(r["method"]) if r["method"] in order else 99)
    fig, ax = plt.subplots(figsize=(6.9, 4.4))
    styles = {
        "cuRobo-Graph-K1": ("o", COLORS["red"]),
        "OPT4C-K1": ("s", COLORS["orange"]),
        "OPT4C-K16": ("^", COLORS["blue"]),
        "cuRobo-Graph-K16": ("D", COLORS["green"]),
    }
    offsets = {
        "cuRobo-Graph-K1": (-68, 8),
        "OPT4C-K1": (-18, 10),
        "OPT4C-K16": (8, -18),
        "cuRobo-Graph-K16": (8, -12),
    }
    for row in rows:
        marker, color = styles.get(row["method"], ("o", "white"))
        x = f(row, "throughput_targets_per_s_mean")
        y = f(row, "strict_sr")
        p95 = f(row, "pos_p95_all_mm")
        ax.scatter(x, y, s=92, marker=marker, facecolor=color, edgecolor="white",
                   linewidth=0.8, label=f"{row['method']}，p95={p95:.1f} mm", zorder=3)
        dx, dy = offsets.get(row["method"], (6, 6))
        ax.annotate(row["method"].replace("cuRobo-Graph-", "cu-"),
                    xy=(x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5,
                    arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#555555"})
    ax.set_xlabel("吞吐量/(targets·s$^{-1}$)")
    ax.set_ylabel("Strict 成功率")
    ax.set_ylim(0.48, 1.035)
    ax.margins(x=0.04)
    ax.set_title("吞吐量-成功率帕累托对比（N=1000）")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.92)
    save(fig, "fig_pareto_throughput_success.pdf")


def plot_kernel_time() -> None:
    path = RESULTS / "kernel_time_breakdown.csv"
    if not path.exists():
        print(f"skip kernel breakdown missing {path}")
        return
    rows = read_csv(path)
    labels = [row["N"] for row in rows]
    h2d = [f(row, "h2d_percent") for row in rows]
    launch = [f(row, "launch_percent") for row in rows]
    kernel = [f(row, "kernel_percent") for row in rows]
    d2h = [f(row, "d2h_percent") for row in rows]
    x = range(len(rows))
    h2d_ms = [f(row, "h2d_ms") for row in rows]
    launch_ms = [f(row, "launch_ms") for row in rows]
    d2h_ms = [f(row, "d2h_ms") for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw={"width_ratios": [1.15, 1.0]})
    bottom = [0.0] * len(rows)
    for values, label, color, hatch in (
        (h2d, "H2D", COLORS["cyan"], ""),
        (launch, "Launch", COLORS["orange"], ""),
        (kernel, "Core kernel", COLORS["blue"], ""),
        (d2h, "D2H", COLORS["green"], ""),
    ):
        axes[0].bar(x, values, bottom=bottom, label=label, color=color, hatch=hatch,
                    edgecolor="black", linewidth=0.45)
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[0].set_xticks(list(x), labels)
    axes[0].set_xlabel("目标数量 N")
    axes[0].set_ylabel("时间占比/%")
    axes[0].set_title("总时间组成")
    axes[0].legend(fontsize=7, ncol=2)

    bottom = [0.0] * len(rows)
    for values, label, color, hatch in (
        (h2d_ms, "H2D", COLORS["cyan"], ""),
        (launch_ms, "Launch", COLORS["orange"], ""),
        (d2h_ms, "D2H", COLORS["green"], ""),
    ):
        axes[1].bar(x, values, bottom=bottom, label=label, color=color, hatch=hatch,
                    edgecolor="black", linewidth=0.45)
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[1].set_xticks(list(x), labels)
    axes[1].set_xlabel("目标数量 N")
    axes[1].set_ylabel("非 kernel 时间/ms")
    axes[1].set_title("H2D/D2H/Launch 放大图")
    axes[1].grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    save(fig, "fig_kernel_time_breakdown.pdf")


def plot_threshold() -> None:
    path = RESULTS / "threshold_scan.csv"
    if not path.exists():
        print(f"skip threshold missing {path}")
        return
    rows = read_csv(path)
    if not rows:
        return
    levels = ["Loose", "Medium", "Strict", "Ultra"]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    methods = ["OPT4C-K16", "OPT4C-K1", "cuRobo-Graph-K16", "cuRobo-Graph-K1"]
    markers = {"OPT4C-K16": "o", "OPT4C-K1": "s", "cuRobo-Graph-K16": "^", "cuRobo-Graph-K1": "D"}
    linestyles = {"OPT4C-K16": "-", "OPT4C-K1": "--", "cuRobo-Graph-K16": "-.", "cuRobo-Graph-K1": ":"}
    colors = {"OPT4C-K16": COLORS["blue"], "OPT4C-K1": COLORS["orange"],
              "cuRobo-Graph-K16": COLORS["green"], "cuRobo-Graph-K1": COLORS["red"]}
    for method in [m for m in methods if any(row["method"] == m for row in rows)]:
        mr = [row for row in rows if row["method"] == method]
        values = []
        for level in levels:
            hit = next((row for row in mr if row["threshold_level"] == level), None)
            values.append(f(hit, "success_rate") if hit else math.nan)
        ax.plot(levels, values, marker=markers.get(method, "o"),
                linestyle=linestyles.get(method, "-"), color=colors.get(method, COLORS["gray"]),
                linewidth=1.8, label=method)
    ax.set_xlabel("阈值等级")
    ax.set_ylabel("成功率")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("不同误差阈值下的成功率")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=7, ncol=2, loc="lower left", frameon=True, framealpha=0.9)
    save(fig, "fig_threshold_scan.pdf")


def plot_seed_scan() -> None:
    path = RESULTS / "seed_count_scan.csv"
    if not path.exists():
        print(f"skip seed scan missing {path}")
        return
    rows = [row for row in read_csv(path) if int(float(row["N"])) == 1000]
    if not rows:
        return
    rows.sort(key=lambda r: int(float(r["K"])))
    k = [int(float(row["K"])) for row in rows]
    sr = [f(row, "strict_sr") for row in rows]
    tp = [f(row, "throughput_targets_per_s_mean") for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    axes[0].plot(k, sr, marker="o", color=COLORS["blue"], linewidth=1.8)
    axes[0].set_xlabel("种子数 K")
    axes[0].set_ylabel("Strict 成功率")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axes[1].plot(k, tp, marker="s", color=COLORS["orange"], linewidth=1.8)
    axes[1].set_xlabel("种子数 K")
    axes[1].set_ylabel("吞吐量/(targets·s$^{-1}$)")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    save(fig, "fig_seed_count_scan.pdf")


def plot_near_singular() -> None:
    path = RESULTS / "near_singular_summary.csv"
    if not path.exists():
        print(f"skip near singular missing {path}")
        return
    rows = [row for row in read_csv(path) if int(float(row["N"])) == 1000]
    if not rows:
        return
    target_types = ["wrist_singular", "elbow_singular", "shoulder_singular"]
    methods = ["OPT4C-K1", "OPT4C-K16"]
    x = range(len(target_types))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for mi, method in enumerate(methods):
        values = []
        for target_type in target_types:
            hit = next((row for row in rows if row["target_type"] == target_type and row["method"] == method), None)
            values.append(f(hit, "strict_sr") if hit else 0.0)
        offset = (mi - 0.5) * width
        ax.bar([v + offset for v in x], values, width=width, label=method,
               color=COLORS["blue"] if mi else COLORS["orange"], edgecolor="white", linewidth=0.6)
    ax.set_xticks(list(x), ["腕部奇异", "肘部奇异", "肩部奇异"])
    ax.set_xlabel("近奇异类型")
    ax.set_ylabel("Strict 成功率")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("近奇异目标成功率（N=1000）")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    save(fig, "fig_near_singular_sr.pdf")


def plot_trajectory() -> None:
    path = RESULTS / "trajectory_continuity_summary.csv"
    if not path.exists():
        print(f"skip trajectory missing {path}")
        return
    rows = read_csv(path)
    traj_types = ["line_50", "arc_50", "random_local_50"]
    labels = ["直线", "圆弧", "局部随机"]
    methods = ["OPT4C-K16-no-rerank", "OPT4C-K16-smoothness-rerank"]
    x = range(len(traj_types))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for mi, method in enumerate(methods):
        values = []
        for traj_type in traj_types:
            hit = next((row for row in rows if row["trajectory_type"] == traj_type and row["method"] == method), None)
            values.append(f(hit, "p95_delta_q") if hit else 0.0)
        offset = (mi - 0.5) * width
        ax.bar([v + offset for v in x], values, width=width, label="无重排序" if mi == 0 else "平滑重排序",
               color=COLORS["orange"] if mi == 0 else COLORS["blue"], edgecolor="white", linewidth=0.6)
    ax.set_xticks(list(x), labels)
    ax.set_xlabel("轨迹类型")
    ax.set_ylabel("$p95(\\Delta q)$/rad")
    ax.set_title("轨迹关节跳变量")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    save(fig, "fig_trajectory_delta_q.pdf")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for mi, method in enumerate(methods):
        values = []
        for traj_type in traj_types:
            hit = next((row for row in rows if row["trajectory_type"] == traj_type and row["method"] == method), None)
            values.append(f(hit, "trajectory_success_rate") if hit else 0.0)
        offset = (mi - 0.5) * width
        ax.bar([v + offset for v in x], values, width=width, label="无重排序" if mi == 0 else "平滑重排序",
               color=COLORS["orange"] if mi == 0 else COLORS["blue"], edgecolor="white", linewidth=0.6)
    ax.set_xticks(list(x), labels)
    ax.set_xlabel("轨迹类型")
    ax.set_ylabel("轨迹成功率")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("轨迹级成功率")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    save(fig, "fig_trajectory_success.pdf")


def plot_near_limit() -> None:
    path = RESULTS / "near_limit_barrier_summary.csv"
    if not path.exists():
        print(f"skip near limit missing {path}")
        return
    rows = [row for row in read_csv(path) if int(float(row["N"])) == 1000 and int(float(row["K"])) == 16]
    if not rows:
        return
    methods = ["OPT4C-K16-BarrierON", "OPT4C-K16-BarrierOFF"]
    labels = ["Barrier ON", "Barrier OFF"]
    sr = []
    near = []
    for method in methods:
        hit = next((row for row in rows if row["method"] == method), None)
        sr.append(f(hit, "strict_sr") if hit else 0.0)
        near.append(f(hit, "near_limit_ratio") if hit else 0.0)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    axes[0].bar(labels, sr, color=[COLORS["blue"], COLORS["orange"]], edgecolor="white", linewidth=0.6)
    axes[0].set_ylabel("Strict 成功率")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("成功率")
    axes[0].grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    axes[1].bar(labels, near, color=[COLORS["blue"], COLORS["orange"]], edgecolor="white", linewidth=0.6)
    axes[1].set_ylabel("近限位比例")
    axes[1].set_ylim(0.0, max(near + [0.05]) * 1.2)
    axes[1].set_title("近限位解比例")
    axes[1].grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    save(fig, "fig_near_limit_barrier.pdf")


def plot_barrier_weight_scan() -> None:
    path = RESULTS / "barrier_weight_scan.csv"
    if not path.exists():
        print(f"skip barrier scan missing {path}")
        return
    rows = [row for row in read_csv(path) if row["target_type"] == "near_limit"]
    if not rows:
        return
    rows.sort(key=lambda r: float(r["wlimit"]))
    weights = [float(row["wlimit"]) for row in rows]
    sr = [f(row, "strict_sr") for row in rows]
    near = [f(row, "near_limit_ratio") for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    axes[0].plot(weights, sr, marker="o", color=COLORS["blue"], linewidth=1.8)
    axes[0].set_xlabel("$w_{limit}$")
    axes[0].set_ylabel("Strict 成功率")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axes[1].plot(weights, near, marker="s", color=COLORS["orange"], linewidth=1.8)
    axes[1].set_xlabel("$w_{limit}$")
    axes[1].set_ylabel("近限位比例")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    save(fig, "fig_barrier_weight_scan.pdf")


def plot_lm_iter_scan() -> None:
    path = RESULTS / "lm_iter_scan.csv"
    if not path.exists():
        print(f"skip lm iter missing {path}")
        return
    rows = read_csv(path)
    rows.sort(key=lambda r: int(float(r["max_iter"])))
    iters = [int(float(row["max_iter"])) for row in rows]
    sr = [f(row, "strict_sr") for row in rows]
    tp = [f(row, "throughput_targets_per_s_mean") for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    axes[0].plot(iters, sr, marker="o", color=COLORS["blue"], linewidth=1.8)
    axes[0].set_xlabel("最大迭代次数")
    axes[0].set_ylabel("Strict 成功率")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axes[1].plot(iters, tp, marker="s", color=COLORS["orange"], linewidth=1.8)
    axes[1].set_xlabel("最大迭代次数")
    axes[1].set_ylabel("吞吐量/(targets·s$^{-1}$)")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    save(fig, "fig_lm_iter_scan.pdf")


def placeholder_diagram_thread() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    ax.axis("off")
    boxes = [
        ("Grid: N blocks", 0.04, 0.77, 0.20, 0.13, COLORS["light_blue"]),
        ("Block 0", 0.28, 0.82, 0.12, 0.08, COLORS["light_blue"]),
        ("Block i -> Target i", 0.44, 0.77, 0.22, 0.13, COLORS["light_blue"]),
        ("Block N-1", 0.70, 0.82, 0.14, 0.08, COLORS["light_blue"]),
        ("Lane 0-15\nSobol seeds", 0.11, 0.51, 0.24, 0.16, COLORS["light_green"]),
        ("Lane 16-31\ninactive/reserved", 0.11, 0.29, 0.24, 0.16, "#eeeeee"),
        ("FK + Jacobian + LM\nper active lane", 0.43, 0.51, 0.25, 0.16, COLORS["light_orange"]),
        ("Shared candidate buffer\ns_cand[16][16]", 0.43, 0.29, 0.25, 0.16, COLORS["light_purple"]),
        ("Lane 0 selection\nmin pose cost", 0.74, 0.42, 0.20, 0.16, "#ffe1e1"),
        ("Global best output\nq*, error, flags", 0.42, 0.08, 0.28, 0.13, COLORS["light_blue"]),
    ]
    for text, x0, y0, w, h, fc in boxes:
        hatch = "///" if "inactive" in text else None
        ax.add_patch(plt.Rectangle((x0, y0), w, h, facecolor=fc, hatch=hatch,
                                   edgecolor="#333333", linewidth=1.0))
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=9)
    for x1, y1, x2, y2 in (
        (0.24, 0.835, 0.28, 0.86),
        (0.40, 0.86, 0.44, 0.835),
        (0.66, 0.835, 0.70, 0.86),
        (0.35, 0.59, 0.43, 0.59),
        (0.55, 0.51, 0.55, 0.45),
        (0.68, 0.37, 0.74, 0.48),
        (0.84, 0.42, 0.58, 0.21),
    ):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "lw": 1.0})
    save(fig, "fig_thread_mapping_redraw.pdf")


def placeholder_diagram_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(4.5, 5.8))
    ax.axis("off")
    steps = ["输入目标位姿", "加载 Sobol 种子", "融合正运动学", "解析雅可比",
             "低控制流复杂度 LM", "关节限位 barrier", "块内候选选择", "输出最佳 IK"]
    y = 0.90
    for step in steps:
        fc = COLORS["light_blue"] if step in {"输入目标位姿", "输出最佳 IK"} else COLORS["light_green"]
        ax.add_patch(plt.Rectangle((0.22, y - 0.045), 0.56, 0.07, facecolor=fc,
                                   edgecolor="#333333", linewidth=1.1))
        ax.text(0.50, y - 0.01, step, ha="center", va="center", fontsize=9)
        if step != steps[-1]:
            ax.annotate("", xy=(0.50, y - 0.10), xytext=(0.50, y - 0.05),
                        arrowprops={"arrowstyle": "->", "lw": 1.0})
        y -= 0.12
    save(fig, "fig_algorithm_pipeline_redraw.pdf")


def main() -> int:
    plot_pareto()
    plot_kernel_time()
    plot_threshold()
    plot_seed_scan()
    plot_near_singular()
    plot_trajectory()
    plot_near_limit()
    plot_barrier_weight_scan()
    plot_lm_iter_scan()
    placeholder_diagram_thread()
    placeholder_diagram_pipeline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
