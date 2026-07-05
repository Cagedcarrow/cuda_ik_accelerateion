#!/usr/bin/env python3
from __future__ import annotations

from common_metrics import EXPERIMENTS, RESULTS, f, mean, read_csv, std, write_csv

FIELDS = [
    "N",
    "h2d_ms",
    "kernel_ms",
    "d2h_ms",
    "launch_ms",
    "total_ms",
    "h2d_percent",
    "kernel_percent",
    "d2h_percent",
    "launch_percent",
    "h2d_ms_std",
    "kernel_ms_std",
    "d2h_ms_std",
    "launch_ms_std",
]


def main() -> int:
    rows = []
    for n in (100, 500, 1000):
        path = EXPERIMENTS / "results" / f"cuda_opt4c_timing_N{n}.csv"
        if not path.exists():
            print(f"missing {path}")
            continue
        data = read_csv(path)
        h2d = [f(row, "h2d_ms") for row in data]
        kernel = [f(row, "gpu_stream_ms") for row in data]
        d2h = [f(row, "d2h_ms") for row in data]
        launch = [f(row, "graph_launch_or_kernel_launch_ms") for row in data]
        h2d_m = mean(h2d)
        kernel_m = mean(kernel)
        d2h_m = mean(d2h)
        launch_m = mean(launch)
        total = h2d_m + kernel_m + d2h_m + launch_m
        denom = total if total > 0.0 else 1.0
        rows.append({
            "N": n,
            "h2d_ms": h2d_m,
            "kernel_ms": kernel_m,
            "d2h_ms": d2h_m,
            "launch_ms": launch_m,
            "total_ms": total,
            "h2d_percent": 100.0 * h2d_m / denom,
            "kernel_percent": 100.0 * kernel_m / denom,
            "d2h_percent": 100.0 * d2h_m / denom,
            "launch_percent": 100.0 * launch_m / denom,
            "h2d_ms_std": std(h2d),
            "kernel_ms_std": std(kernel),
            "d2h_ms_std": std(d2h),
            "launch_ms_std": std(launch),
        })
    out = RESULTS / "kernel_time_breakdown.csv"
    write_csv(out, rows, FIELDS)
    print(f"wrote {out} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

