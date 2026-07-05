#!/usr/bin/env python3
from __future__ import annotations

import math
import subprocess
from collections import defaultdict

import numpy as np

from common_metrics import INPUTS, RESULTS, RUNNER, f, read_csv, write_csv

TRAJ_TYPES = ["line_50", "arc_50", "random_local_50"]
POINTS_PER_TRAJ = 50
NUM_TRAJ = 20
N = POINTS_PER_TRAJ * NUM_TRAJ
K = 16
WARMUP = 10
REPEAT = 30
MAX_ITER = 60
BETA = 0.02
JUMP_THRESHOLD_RAD = 0.5

FIELDS = [
    "experiment",
    "method",
    "robot",
    "trajectory_type",
    "num_traj",
    "points_per_traj",
    "N",
    "K",
    "point_success_rate",
    "trajectory_success_rate",
    "mean_delta_q",
    "p95_delta_q",
    "max_delta_q",
    "joint_jump_count",
    "strict_sr",
    "gpu_time_ms_mean",
    "throughput_targets_per_s_mean",
    "beta",
    "jump_threshold_rad",
    "source_csv",
    "notes",
]


def run_runner(traj_type: str) -> tuple:
    target = INPUTS / "trajectory" / f"targets_{traj_type}_N{N}_T4x4_f64.raw"
    seeds = INPUTS / "trajectory" / f"seeds_{traj_type}_N{N}_K{K}_q_f64.raw"
    best = RESULTS / f"trajectory_dump_best_{traj_type}_N{N}_K{K}.csv"
    candidates = RESULTS / f"trajectory_dump_candidates_{traj_type}_N{N}_K{K}.csv"
    summary = RESULTS / f"trajectory_dump_summary_{traj_type}_N{N}_K{K}.csv"
    timing = RESULTS / f"trajectory_dump_timing_{traj_type}_N{N}_K{K}.csv"
    if not summary.exists() or not candidates.exists():
        cmd = [
            str(RUNNER),
            "--mode", "v4_static",
            "--variant", "baseline",
            "--limit-gradient", "analytic",
            "--graph-mode", "off",
            "--precision-mode", "fp64",
            "--fallback-mode", "none",
            "--targets", str(target),
            "--seeds", str(seeds),
            "--N", str(N),
            "--K", str(K),
            "--max-iter", str(MAX_ITER),
            "--repeat", str(REPEAT),
            "--warmup", str(WARMUP),
            "--best-csv", str(best),
            "--candidates-csv", str(candidates),
            "--summary-csv", str(summary),
            "--timing-csv", str(timing),
        ]
        print("running trajectory candidate dump", traj_type, flush=True)
        subprocess.run(cmd, check=True, cwd=RUNNER.parents[1])
    return best, candidates, summary


def q_from_row(row) -> np.ndarray:
    return np.array([f(row, f"q{i}") for i in range(6)], dtype=np.float64)


def angular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def trajectory_deltas(q: np.ndarray) -> np.ndarray:
    shaped = q.reshape(NUM_TRAJ, POINTS_PER_TRAJ, 6)
    dq = angular_diff(shaped[:, 1:, :], shaped[:, :-1, :])
    return np.linalg.norm(dq, axis=2).reshape(-1)


def summarize_sequence(rows: list[dict], method: str, traj_type: str, summary_row: dict, source_csv: str) -> dict[str, object]:
    q = np.stack([q_from_row(row) for row in rows])
    strict = np.array([f(row, "success_strict") > 0.5 for row in rows], dtype=bool)
    deltas = trajectory_deltas(q)
    traj_success = strict.reshape(NUM_TRAJ, POINTS_PER_TRAJ).all(axis=1)
    return {
        "experiment": "trajectory_continuity",
        "method": method,
        "robot": "UR10",
        "trajectory_type": traj_type,
        "num_traj": NUM_TRAJ,
        "points_per_traj": POINTS_PER_TRAJ,
        "N": N,
        "K": K,
        "point_success_rate": float(strict.mean()),
        "trajectory_success_rate": float(traj_success.mean()),
        "mean_delta_q": float(np.mean(deltas)) if len(deltas) else 0.0,
        "p95_delta_q": float(np.percentile(deltas, 95)) if len(deltas) else 0.0,
        "max_delta_q": float(np.max(deltas)) if len(deltas) else 0.0,
        "joint_jump_count": int(np.sum(deltas > JUMP_THRESHOLD_RAD)),
        "strict_sr": f(summary_row, "strict_sr"),
        "gpu_time_ms_mean": f(summary_row, "gpu_stream_ms_mean"),
        "throughput_targets_per_s_mean": f(summary_row, "raw_throughput_mean"),
        "beta": BETA if "smoothness" in method else "",
        "jump_threshold_rad": JUMP_THRESHOLD_RAD,
        "source_csv": source_csv,
        "notes": "wrapped revolute delta via atan2(sin(dq),cos(dq)); offline rerank from baseline candidate dump; fused OPT4C source unchanged" if "smoothness" in method else "wrapped revolute delta via atan2(sin(dq),cos(dq)); baseline candidate dump best selection",
    }


def smoothness_rerank(candidate_rows: list[dict]) -> list[dict]:
    by_target = defaultdict(list)
    for row in candidate_rows:
        by_target[int(float(row["target_id"]))].append(row)
    selected = []
    prev_q = None
    for target_id in range(N):
        cand = by_target[target_id]
        strict = [row for row in cand if f(row, "success_strict") > 0.5]
        pool = strict if strict else cand
        best_row = None
        best_score = math.inf
        for row in pool:
            q = q_from_row(row)
            smooth = 0.0 if prev_q is None else float(np.sum(angular_diff(q, prev_q) ** 2))
            score = f(row, "pose_cost") + BETA * smooth
            if score < best_score:
                best_score = score
                best_row = row
        selected.append(best_row)
        prev_q = q_from_row(best_row)
        if (target_id + 1) % POINTS_PER_TRAJ == 0:
            prev_q = None
    return selected


def main() -> int:
    rows = []
    for traj_type in TRAJ_TYPES:
        best_csv, cand_csv, summary_csv = run_runner(traj_type)
        best_rows = read_csv(best_csv)
        cand_rows = read_csv(cand_csv)
        summary_row = read_csv(summary_csv)[0]
        rows.append(summarize_sequence(best_rows, "OPT4C-K16-no-rerank", traj_type, summary_row, str(best_csv)))
        reranked = smoothness_rerank(cand_rows)
        rows.append(summarize_sequence(reranked, "OPT4C-K16-smoothness-rerank", traj_type, summary_row, str(cand_csv)))
    out = RESULTS / "trajectory_continuity_summary.csv"
    write_csv(out, rows, FIELDS)
    print(f"wrote {out} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
