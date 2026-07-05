#!/usr/bin/env python3
from __future__ import annotations

from common_metrics import COMMON_FIELDS, FIGURES, REPORTS, RESULTS, read_csv

REQUIRED_P0_RESULTS = [
    "fair_curobo_k16_summary.csv",
    "near_singular_summary.csv",
    "near_limit_barrier_summary.csv",
    "trajectory_continuity_summary.csv",
    "seed_count_scan.csv",
    "kernel_time_breakdown.csv",
    "cpu_baseline_summary.csv",
    "threshold_scan.csv",
]

REQUIRED_P0_FIGURES = [
    "fig_pareto_throughput_success.pdf",
    "fig_near_singular_sr.pdf",
    "fig_near_limit_barrier.pdf",
    "fig_trajectory_delta_q.pdf",
    "fig_seed_count_scan.pdf",
    "fig_kernel_time_breakdown.pdf",
    "fig_threshold_scan.pdf",
    "fig_thread_mapping_redraw.pdf",
    "fig_nsys_timeline_opt4c.pdf",
]

REQUIRED_REPORTS = [
    "nsys_opt4c_n1000_summary.txt",
    "paper_review_report.md",
    "final_revision_changelog.md",
    "final_revision_acceptance_report.md",
]


def check_csv(path):
    if not path.exists():
        return False, "missing"
    rows = read_csv(path)
    if not rows:
        return False, "empty"
    return True, f"rows={len(rows)}"


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    lines = ["# 补充实验总体验收报告", ""]
    ok_all = True
    lines.append("## CSV 产物")
    for name in REQUIRED_P0_RESULTS:
        ok, detail = check_csv(RESULTS / name)
        ok_all = ok_all and ok
        lines.append(f"- [{'x' if ok else ' '}] `{name}`: {detail}")

    lines.append("")
    lines.append("## 图产物")
    for name in REQUIRED_P0_FIGURES:
        path = FIGURES / name
        ok = path.exists() and path.stat().st_size > 0
        ok_all = ok_all and ok
        detail = f"{path.stat().st_size} bytes" if path.exists() else "missing"
        lines.append(f"- [{'x' if ok else ' '}] `{name}`: {detail}")

    lines.append("")
    lines.append("## 报告产物")
    for name in REQUIRED_REPORTS:
        path = REPORTS / name
        ok = path.exists() and path.stat().st_size > 0
        ok_all = ok_all and ok
        detail = f"{path.stat().st_size} bytes" if path.exists() else "missing"
        lines.append(f"- [{'x' if ok else ' '}] `{name}`: {detail}")

    lines.append("")
    lines.append("## 通用字段检查")
    for name in REQUIRED_P0_RESULTS:
        path = RESULTS / name
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        missing = [field for field in COMMON_FIELDS if field not in header]
        if name in {"kernel_time_breakdown.csv", "trajectory_continuity_summary.csv", "threshold_scan.csv"}:
            lines.append(f"- `{name}`: specialized schema, header columns={len(header)}")
        elif missing:
            ok_all = False
            lines.append(f"- [ ] `{name}` missing common fields: {', '.join(missing)}")
        else:
            lines.append(f"- [x] `{name}` common fields complete")

    lines.append("")
    lines.append("## 结论")
    if ok_all:
        lines.append("总体验收：通过。补充实验、CPU baseline、Nsight Systems timeline、论文审稿报告和正文合并产物均已生成。")
    else:
        lines.append("总体验收：未通过。缺失项必须补齐或在论文局限性中明确说明后，才能考虑有条件合并。")

    out = REPORTS / "acceptance_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
