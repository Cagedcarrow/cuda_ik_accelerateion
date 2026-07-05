#!/usr/bin/env python3
from __future__ import annotations

from common_metrics import COMMON_FIELDS, EXPERIMENTS, RESULTS, f, i, read_csv, row_from_runner_summary, write_csv


def main() -> int:
    rows: list[dict[str, object]] = []

    dense = EXPERIMENTS / "results" / "dense_static_summary.csv"
    if dense.exists():
        for row in read_csv(dense):
            rows.append(row_from_runner_summary(row, "OPT4C-K16", "fair_curobo_k16", "random_reachable"))

    k1 = RESULTS / "k1_static_summary.csv"
    if k1.exists():
        for row in read_csv(k1):
            rows.append(row_from_runner_summary(row, "OPT4C-K1", "fair_curobo_k16", "random_reachable"))

    cu1 = EXPERIMENTS / "results" / "dense_curobo_summary.csv"
    if cu1.exists():
        for row in read_csv(cu1):
            n = i(row, "N")
            strict = f(row, "strict_sr")
            rows.append({
                "experiment": "fair_curobo_k16",
                "method": "cuRobo-Graph-K1",
                "robot": "UR10",
                "N": n,
                "K": 1,
                "seed_type": "curobo_default",
                "max_iter": "",
                "wlimit": "",
                "barrier_enabled": "",
                "target_type": "random_reachable",
                "repeat": "",
                "gpu_time_ms_mean": f(row, "gpu_ms_mean"),
                "gpu_time_ms_std": "",
                "throughput_targets_per_s_mean": f(row, "throughput"),
                "throughput_targets_per_s_std": "",
                "strict_sr": strict,
                "medium_sr": "",
                "loose_sr": "",
                "pos_p95_all_mm": f(row, "pos_p95_all_mm"),
                "pos_p95_success_mm": f(row, "pos_p95_success_only_mm"),
                "rot_p95_all_deg": f(row, "rot_p95_all_deg"),
                "rot_p95_success_deg": "",
                "mean_iter": "",
                "p95_iter": "",
                "near_limit_ratio": "",
                "joint_violation_count": "",
                "nan_count": "",
                "inf_count": "",
                "fail_count": i(row, "strict_failure_count"),
                "notes": "CUDA Graph enabled; collision disabled; external FK reevaluation from source script",
            })

    cu16 = RESULTS / "curobo_k16_summary.csv"
    if cu16.exists():
        for row in read_csv(cu16):
            n = i(row, "N")
            strict = f(row, "strict_sr")
            rows.append({
                "experiment": "fair_curobo_k16",
                "method": "cuRobo-Graph-K16",
                "robot": "UR10",
                "N": n,
                "K": i(row, "num_seeds"),
                "seed_type": "curobo_16",
                "max_iter": "",
                "wlimit": "",
                "barrier_enabled": "",
                "target_type": "random_reachable",
                "repeat": "",
                "gpu_time_ms_mean": f(row, "gpu_ms"),
                "gpu_time_ms_std": "",
                "throughput_targets_per_s_mean": f(row, "throughput"),
                "throughput_targets_per_s_std": "",
                "strict_sr": strict,
                "medium_sr": "",
                "loose_sr": "",
                "pos_p95_all_mm": f(row, "pos_p95_all_mm"),
                "pos_p95_success_mm": f(row, "pos_p95_success_mm"),
                "rot_p95_all_deg": f(row, "rot_p95_all_deg"),
                "rot_p95_success_deg": "",
                "mean_iter": "",
                "p95_iter": "",
                "near_limit_ratio": "",
                "joint_violation_count": "",
                "nan_count": "",
                "inf_count": "",
                "fail_count": n - round(strict * n),
                "notes": "CUDA Graph enabled; collision disabled; num_seeds=16; current source covers N=100,500,1000",
            })

    rows.sort(key=lambda r: (int(r["N"]), str(r["method"])))
    out = RESULTS / "fair_curobo_k16_summary.csv"
    write_csv(out, rows, COMMON_FIELDS)
    print(f"wrote {out} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

