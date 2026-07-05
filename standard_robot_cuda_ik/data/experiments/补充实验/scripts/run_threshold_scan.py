#!/usr/bin/env python3
from __future__ import annotations

from common_metrics import (
    EXPERIMENTS,
    RESULTS,
    LOOSE_POS_MM,
    LOOSE_ROT_DEG,
    MEDIUM_POS_MM,
    MEDIUM_ROT_DEG,
    STRICT_POS_MM,
    STRICT_ROT_DEG,
    ULTRA_POS_MM,
    ULTRA_ROT_DEG,
    f,
    pass_threshold,
    percentile,
    read_csv,
    write_csv,
)

FIELDS = [
    "experiment",
    "method",
    "N",
    "K",
    "threshold_level",
    "pos_threshold_mm",
    "rot_threshold_deg",
    "success_rate",
    "pos_p95_all_mm",
    "rot_p95_all_deg",
    "source_csv",
]

LEVELS = [
    ("Loose", LOOSE_POS_MM, LOOSE_ROT_DEG),
    ("Medium", MEDIUM_POS_MM, MEDIUM_ROT_DEG),
    ("Strict", STRICT_POS_MM, STRICT_ROT_DEG),
    ("Ultra", ULTRA_POS_MM, ULTRA_ROT_DEG),
]


def scan_best(path, method: str, n: int, k: int) -> list[dict[str, object]]:
    rows = read_csv(path)
    pos = [f(row, "pos_err_mm") for row in rows]
    rot = [f(row, "rot_err_deg") for row in rows]
    return scan_errors(path, method, n, k, pos, rot)


def scan_curobo_rows(path, method: str, n: int, k: int) -> list[dict[str, object]]:
    rows = read_csv(path)
    pos = [f(row, "our_reeval_pos_err_mm") for row in rows]
    rot = [f(row, "our_reeval_rot_err_deg") for row in rows]
    return scan_errors(path, method, n, k, pos, rot)


def scan_errors(path, method: str, n: int, k: int, pos: list[float], rot: list[float]) -> list[dict[str, object]]:
    out = []
    total = max(1, len(pos))
    for level, p_lim, r_lim in LEVELS:
        ok = sum(1 for p, r in zip(pos, rot) if pass_threshold(p, r, p_lim, r_lim))
        out.append({
            "experiment": "threshold_scan",
            "method": method,
            "N": n,
            "K": k,
            "threshold_level": level,
            "pos_threshold_mm": p_lim,
            "rot_threshold_deg": r_lim,
            "success_rate": ok / total,
            "pos_p95_all_mm": percentile(pos, 95),
            "rot_p95_all_deg": percentile(rot, 95),
            "source_csv": str(path.relative_to(EXPERIMENTS.parents[1])),
        })
    return out


def main() -> int:
    rows: list[dict[str, object]] = []
    n = 1000
    k16 = EXPERIMENTS / "results" / f"cuda_opt4c_best_N{n}.csv"
    if k16.exists():
        rows.extend(scan_best(k16, "OPT4C-K16", n, 16))
    k1 = RESULTS / f"cuda_opt4c_k1_best_N{n}.csv"
    if k1.exists():
        rows.extend(scan_best(k1, "OPT4C-K1", n, 1))
    for k, method in ((16, "cuRobo-Graph-K16"), (1, "cuRobo-Graph-K1")):
        curobo_rows = RESULTS / f"curobo_threshold_N{n}_K{k}.rows.csv"
        if curobo_rows.exists():
            rows.extend(scan_curobo_rows(curobo_rows, method, n, k))
    out = RESULTS / "threshold_scan.csv"
    write_csv(out, rows, FIELDS)
    print(f"wrote {out} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
