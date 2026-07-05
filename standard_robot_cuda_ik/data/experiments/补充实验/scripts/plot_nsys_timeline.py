#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common_metrics import FIGURES, REPORTS, ensure_dirs

SQLITE = REPORTS / "nsys_opt4c_n1000.sqlite"
SUMMARY = REPORTS / "nsys_opt4c_n1000_summary.txt"
FIGURE = FIGURES / "fig_nsys_timeline_opt4c.pdf"


def rows(con: sqlite3.Connection, query: str) -> list[tuple]:
    return list(con.execute(query))


def main() -> int:
    ensure_dirs()
    if not SQLITE.exists():
        raise FileNotFoundError(f"missing Nsight sqlite: {SQLITE}")
    con = sqlite3.connect(SQLITE)
    kernels = rows(con, "select start, end, streamId, demangledName, gridX, blockX, registersPerThread from CUPTI_ACTIVITY_KIND_KERNEL order by start")
    memcpys = rows(con, "select start, end, streamId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start")
    memsets = rows(con, "select start, end, streamId, bytes from CUPTI_ACTIVITY_KIND_MEMSET order by start")
    strings = {sid: value for sid, value in rows(con, "select id, value from StringIds")}
    if not kernels:
        raise RuntimeError("Nsight sqlite contains no CUDA kernel events")

    t0 = min([r[0] for r in kernels] + [r[0] for r in memcpys] + [r[0] for r in memsets])
    kernel_durations_ms = [(end - start) / 1e6 for start, end, *_ in kernels]
    memcpy_durations_ms = [(end - start) / 1e6 for start, end, *_ in memcpys]
    memset_durations_ms = [(end - start) / 1e6 for start, end, *_ in memsets]

    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    y_positions = {"memcpy": 2.0, "memset": 1.2, "kernel": 0.4}
    for start, end, stream, _, _, _, _ in kernels:
        ax.broken_barh([((start - t0) / 1e6, (end - start) / 1e6)], (y_positions["kernel"], 0.55), facecolors="#2563eb")
    for start, end, stream, nbytes, copy_kind in memcpys:
        color = "#f59e0b" if copy_kind == 1 else "#10b981"
        ax.broken_barh([((start - t0) / 1e6, max((end - start) / 1e6, 0.02))], (y_positions["memcpy"], 0.45), facecolors=color)
    for start, end, stream, nbytes in memsets:
        ax.broken_barh([((start - t0) / 1e6, max((end - start) / 1e6, 0.02))], (y_positions["memset"], 0.35), facecolors="#6b7280")
    ax.set_yticks([2.225, 1.375, 0.675])
    ax.set_yticklabels(["H2D/D2H", "memset", "IK kernel"])
    ax.set_xlabel("time from first CUDA event / ms")
    ax.set_title("Nsight Systems timeline: OPT4C N=1000 K=16")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.6)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURE)
    plt.close(fig)

    kernel_name = strings.get(kernels[0][3], str(kernels[0][3]))
    text = [
        "# Nsight Systems OPT4C N=1000 K=16 摘要",
        "",
        f"- report: `{SQLITE.name}`",
        f"- kernel_name: `{kernel_name}`",
        f"- kernel_instances: {len(kernels)}",
        f"- kernel_avg_ms: {sum(kernel_durations_ms) / len(kernel_durations_ms):.6f}",
        f"- kernel_min_ms: {min(kernel_durations_ms):.6f}",
        f"- kernel_max_ms: {max(kernel_durations_ms):.6f}",
        f"- memcpy_count: {len(memcpys)}",
        f"- memcpy_total_ms: {sum(memcpy_durations_ms):.6f}",
        f"- memset_count: {len(memsets)}",
        f"- memset_total_ms: {sum(memset_durations_ms):.6f}",
        f"- figure: `{FIGURE.name}`",
        "",
        "说明：本次采样命令包含 `warmup=10, repeat=30`，因此核心 IK kernel 共 40 次。采样时系统不允许 CPU context switch tracing；CUDA kernel、memcpy 和 memset timeline 已成功采集。",
    ]
    SUMMARY.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"wrote {SUMMARY}")
    print(f"wrote {FIGURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
