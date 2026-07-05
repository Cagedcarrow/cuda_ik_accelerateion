#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[4]
EXPERIMENTS = ROOT / "data" / "experiments"
SUPP = EXPERIMENTS / "补充实验"
RESULTS = SUPP / "results"
FIGURES = SUPP / "figures"
REPORTS = SUPP / "reports"
INPUTS = SUPP / "inputs"
RUNNER = ROOT / "build" / "standard_robot_cuda_v4_runner"

STRICT_POS_MM = 5.0
STRICT_ROT_DEG = 1.0
MEDIUM_POS_MM = 10.0
MEDIUM_ROT_DEG = 5.0
LOOSE_POS_MM = 30.0
LOOSE_ROT_DEG = 10.0
ULTRA_POS_MM = 2.0
ULTRA_ROT_DEG = 0.5

COMMON_FIELDS = [
    "experiment",
    "method",
    "robot",
    "N",
    "K",
    "seed_type",
    "max_iter",
    "wlimit",
    "barrier_enabled",
    "target_type",
    "repeat",
    "gpu_time_ms_mean",
    "gpu_time_ms_std",
    "throughput_targets_per_s_mean",
    "throughput_targets_per_s_std",
    "strict_sr",
    "medium_sr",
    "loose_sr",
    "pos_p95_all_mm",
    "pos_p95_success_mm",
    "rot_p95_all_deg",
    "rot_p95_success_deg",
    "mean_iter",
    "p95_iter",
    "near_limit_ratio",
    "joint_violation_count",
    "nan_count",
    "inf_count",
    "fail_count",
    "notes",
]


def ensure_dirs() -> None:
    for path in (RESULTS, FIGURES, REPORTS, INPUTS):
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def f(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None, "NA"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def i(row: Mapping[str, object], key: str, default: int = 0) -> int:
    return int(round(f(row, key, float(default))))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def std(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def percentile(values: Iterable[float], q: float) -> float:
    values = sorted(float(v) for v in values if not math.isnan(float(v)))
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def pass_threshold(pos_mm: float, rot_deg: float, pos_limit_mm: float, rot_limit_deg: float) -> bool:
    return pos_mm < pos_limit_mm and rot_deg < rot_limit_deg


def summarize_best_rows(rows: list[dict[str, str]], method: str, experiment: str, n: int, k: int,
                        target_type: str, max_iter: int = 60, wlimit: str = "0.03",
                        barrier_enabled: str = "true", seed_type: str = "sobol") -> dict[str, object]:
    pos = [f(row, "pos_err_mm") for row in rows]
    rot = [f(row, "rot_err_deg") for row in rows]
    iters = [f(row, "iters") for row in rows]
    strict_flags = [pass_threshold(p, r, STRICT_POS_MM, STRICT_ROT_DEG) for p, r in zip(pos, rot)]
    medium_flags = [pass_threshold(p, r, MEDIUM_POS_MM, MEDIUM_ROT_DEG) for p, r in zip(pos, rot)]
    loose_flags = [pass_threshold(p, r, LOOSE_POS_MM, LOOSE_ROT_DEG) for p, r in zip(pos, rot)]
    pos_suc = [p for p, ok in zip(pos, strict_flags) if ok]
    rot_suc = [r for r, ok in zip(rot, strict_flags) if ok]
    near_limit = [i(row, "near_limit") for row in rows]
    nan_count = sum(1 for p, r in zip(pos, rot) if math.isnan(p) or math.isnan(r))
    inf_count = sum(1 for p, r in zip(pos, rot) if math.isinf(p) or math.isinf(r))
    strict_count = sum(1 for ok in strict_flags if ok)
    total = max(1, len(rows))
    return {
        "experiment": experiment,
        "method": method,
        "robot": "UR10",
        "N": n,
        "K": k,
        "seed_type": seed_type,
        "max_iter": max_iter,
        "wlimit": wlimit,
        "barrier_enabled": barrier_enabled,
        "target_type": target_type,
        "repeat": "",
        "gpu_time_ms_mean": "",
        "gpu_time_ms_std": "",
        "throughput_targets_per_s_mean": "",
        "throughput_targets_per_s_std": "",
        "strict_sr": strict_count / total,
        "medium_sr": sum(1 for ok in medium_flags if ok) / total,
        "loose_sr": sum(1 for ok in loose_flags if ok) / total,
        "pos_p95_all_mm": percentile(pos, 95),
        "pos_p95_success_mm": percentile(pos_suc, 95) if pos_suc else "",
        "rot_p95_all_deg": percentile(rot, 95),
        "rot_p95_success_deg": percentile(rot_suc, 95) if rot_suc else "",
        "mean_iter": mean(iters),
        "p95_iter": percentile(iters, 95),
        "near_limit_ratio": sum(near_limit) / total,
        "joint_violation_count": "",
        "nan_count": nan_count,
        "inf_count": inf_count,
        "fail_count": total - strict_count,
        "notes": "",
    }


def row_from_runner_summary(row: Mapping[str, object], method: str, experiment: str, target_type: str,
                            seed_type: str = "sobol", wlimit: str = "0.03",
                            barrier_enabled: str = "true") -> dict[str, object]:
    n = i(row, "N")
    strict = f(row, "strict_sr")
    return {
        "experiment": experiment,
        "method": method,
        "robot": "UR10",
        "N": n,
        "K": i(row, "K"),
        "seed_type": seed_type,
        "max_iter": 60,
        "wlimit": wlimit,
        "barrier_enabled": barrier_enabled,
        "target_type": target_type,
        "repeat": i(row, "repeat"),
        "gpu_time_ms_mean": f(row, "gpu_stream_ms_mean"),
        "gpu_time_ms_std": f(row, "gpu_stream_ms_std"),
        "throughput_targets_per_s_mean": f(row, "raw_throughput_mean"),
        "throughput_targets_per_s_std": f(row, "raw_throughput_std"),
        "strict_sr": strict,
        "medium_sr": f(row, "medium_sr"),
        "loose_sr": f(row, "loose_sr"),
        "pos_p95_all_mm": f(row, "pos_p95_all_mm"),
        "pos_p95_success_mm": f(row, "pos_p95_suc_mm"),
        "rot_p95_all_deg": f(row, "rot_p95_all_deg"),
        "rot_p95_success_deg": f(row, "rot_p95_suc_deg"),
        "mean_iter": f(row, "iter_mean"),
        "p95_iter": f(row, "iter_p95"),
        "near_limit_ratio": f(row, "near_limit_ratio"),
        "joint_violation_count": "",
        "nan_count": i(row, "nan_count"),
        "inf_count": i(row, "inf_count"),
        "fail_count": n - round(strict * n),
        "notes": "",
    }
