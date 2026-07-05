#!/usr/bin/env python3
from __future__ import annotations

import math
import re
import sys
import time
from pathlib import Path

import numpy as np

from common_metrics import (
    COMMON_FIELDS,
    EXPERIMENTS,
    RESULTS,
    ROOT,
    STRICT_POS_MM,
    STRICT_ROT_DEG,
    MEDIUM_POS_MM,
    MEDIUM_ROT_DEG,
    LOOSE_POS_MM,
    LOOSE_ROT_DEG,
    ensure_dirs,
    percentile,
    write_csv,
)

N_VALUES = [100, 500, 1000]
K_VALUES = [1, 16]
MAX_ITER = 60
LIMIT_WEIGHT = 0.03
LIMIT_MARGIN = 0.087
REPEAT = 1


def parse_array(header_text: str, name: str, dtype: type[float] | type[int]) -> np.ndarray:
    pattern = rf"static const [^=]+ {name}\[[^\]]+\] = \{{(.*?)\}};"
    match = re.search(pattern, header_text, flags=re.S)
    if not match:
        raise ValueError(f"Cannot find {name}")
    raw = [x.strip() for x in match.group(1).replace("\n", " ").split(",") if x.strip()]
    return np.asarray([dtype(x) for x in raw])


def load_constants() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    header = (ROOT / "include" / "standard_robot_cuda_ik" / "generated" / "ur10_model_constants.h").read_text(encoding="utf-8")
    origins = parse_array(header, "k_origins", float).reshape(6, 4, 4)
    axes = parse_array(header, "k_axes", float).reshape(6, 3)
    limits = parse_array(header, "k_joint_limits", float).reshape(6, 2)
    tool = parse_array(header, "k_T_wrist3_to_tcp", float).reshape(4, 4)
    return origins, axes, limits, tool


ORIGINS, AXES, LIMITS, TOOL = load_constants()


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    ax = axis / np.linalg.norm(axis)
    x, y, z = ax
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )
    return out


def fk_with_frames(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = np.eye(4, dtype=np.float64)
    p = np.zeros((6, 3), dtype=np.float64)
    z = np.zeros((6, 3), dtype=np.float64)
    for seg in range(6):
        T = T @ ORIGINS[seg]
        p[seg] = T[:3, 3]
        z[seg] = T[:3, :3] @ AXES[seg]
        T = T @ rotation_about_axis(AXES[seg], float(q[seg]))
    T = T @ TOOL
    return T, p, z


def pose_error(T_cur: np.ndarray, T_tgt: np.ndarray) -> np.ndarray:
    err = np.empty(6, dtype=np.float64)
    err[:3] = T_cur[:3, 3] - T_tgt[:3, 3]
    R_cur = T_cur[:3, :3]
    R_tgt = T_tgt[:3, :3]
    E = R_cur.T @ R_tgt
    err[3] = 0.5 * (E[2, 1] - E[1, 2])
    err[4] = 0.5 * (E[0, 2] - E[2, 0])
    err[5] = 0.5 * (E[1, 0] - E[0, 1])
    return err


def analytical_jacobian(T: np.ndarray, p: np.ndarray, z: np.ndarray) -> np.ndarray:
    pe = T[:3, 3]
    J = np.empty((6, 6), dtype=np.float64)
    for j in range(6):
        J[:3, j] = np.cross(z[j], pe - p[j])
        J[3:, j] = z[j]
    return J


def limit_loss(q: np.ndarray) -> float:
    below = np.maximum(0.0, LIMIT_MARGIN - (q - LIMITS[:, 0]))
    above = np.maximum(0.0, LIMIT_MARGIN - (LIMITS[:, 1] - q))
    return float(np.sum(below * below + above * above))


def limit_grad(q: np.ndarray) -> np.ndarray:
    grad = np.zeros(6, dtype=np.float64)
    for j in range(6):
        lo, hi = LIMITS[j]
        lower_gap = q[j] - lo
        upper_gap = hi - q[j]
        if lower_gap < LIMIT_MARGIN:
            grad[j] += -2.0 * (LIMIT_MARGIN - lower_gap)
        if upper_gap < LIMIT_MARGIN:
            grad[j] += 2.0 * (LIMIT_MARGIN - upper_gap)
    return grad


def near_limit(q: np.ndarray) -> bool:
    return bool(np.any((q - LIMITS[:, 0] < LIMIT_MARGIN) | (LIMITS[:, 1] - q < LIMIT_MARGIN)))


def solve_one(T_tgt: np.ndarray, q_seed: np.ndarray) -> dict[str, object]:
    q = q_seed.astype(np.float64).copy()
    lam = 1e-2
    iters = MAX_ITER
    for it in range(MAX_ITER):
        T, p, z = fk_with_frames(q)
        e = pose_error(T, T_tgt)
        pos = float(np.linalg.norm(e[:3]))
        rot = float(np.linalg.norm(e[3:]))
        if pos < STRICT_POS_MM / 1000.0 and rot < math.radians(STRICT_ROT_DEG):
            iters = it + 1
            break
        J = analytical_jacobian(T, p, z)
        g = J.T @ e + LIMIT_WEIGHT * limit_grad(q)
        H = J.T @ J + lam * np.eye(6, dtype=np.float64)
        try:
            dq = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            dq = np.linalg.lstsq(H, -g, rcond=None)[0]
        max_abs = float(np.max(np.abs(dq)))
        if max_abs > 0.35:
            dq *= 0.35 / max_abs
        q_trial = np.clip(q + dq, LIMITS[:, 0], LIMITS[:, 1])
        loss_old = 0.5 * float(e @ e) + LIMIT_WEIGHT * limit_loss(q)
        T_trial, _, _ = fk_with_frames(q_trial)
        e_trial = pose_error(T_trial, T_tgt)
        loss_new = 0.5 * float(e_trial @ e_trial) + LIMIT_WEIGHT * limit_loss(q_trial)
        q = q_trial
        lam = min(0.5, max(1e-6, lam * (0.5 if loss_new < loss_old else 2.0)))

    T, _, _ = fk_with_frames(q)
    e = pose_error(T, T_tgt)
    pos_mm = float(np.linalg.norm(e[:3]) * 1000.0)
    rot_deg = float(np.linalg.norm(e[3:]) * 180.0 / math.pi)
    return {
        "q": q,
        "pos_err_mm": pos_mm,
        "rot_err_deg": rot_deg,
        "pose_cost": float(np.linalg.norm(e[:3]) ** 2 + np.linalg.norm(e[3:]) ** 2),
        "limit_score": limit_loss(q),
        "iters": iters,
        "near_limit": near_limit(q),
    }


def strict_ok(pos_mm: float, rot_deg: float) -> bool:
    return pos_mm < STRICT_POS_MM and rot_deg < STRICT_ROT_DEG


def threshold_ok(pos_mm: float, rot_deg: float, pos_lim: float, rot_lim: float) -> bool:
    return pos_mm < pos_lim and rot_deg < rot_lim


def choose_best(candidates: list[dict[str, object]]) -> tuple[int, dict[str, object]]:
    def key(item: tuple[int, dict[str, object]]) -> tuple[int, int, float]:
        idx, row = item
        pos = float(row["pos_err_mm"])
        rot = float(row["rot_err_deg"])
        if threshold_ok(pos, rot, STRICT_POS_MM, STRICT_ROT_DEG):
            rank = 0
        elif threshold_ok(pos, rot, MEDIUM_POS_MM, MEDIUM_ROT_DEG):
            rank = 1
        elif threshold_ok(pos, rot, LOOSE_POS_MM, LOOSE_ROT_DEG):
            rank = 2
        else:
            rank = 3
        return (rank, 1 if row["near_limit"] else 0, float(row["pose_cost"]) + LIMIT_WEIGHT * float(row["limit_score"]))

    return min(enumerate(candidates), key=key)


def run_case(n: int, k: int) -> dict[str, object]:
    targets = np.fromfile(EXPERIMENTS / "inputs" / f"targets_N{n}_T4x4_f64.raw", dtype=np.float64).reshape(n, 4, 4)
    seed_path = EXPERIMENTS / "inputs" / f"seeds_N{n}_K16_q_f64.raw"
    seeds = np.fromfile(seed_path, dtype=np.float64).reshape(n, 16, 6)[:, :k, :]
    best_rows: list[dict[str, object]] = []
    t0 = time.perf_counter()
    for target_id in range(n):
        candidates = [solve_one(targets[target_id], seeds[target_id, seed_id]) for seed_id in range(k)]
        seed_id, best = choose_best(candidates)
        out = {"target_id": target_id, "best_seed_id": seed_id, **best}
        best_rows.append(out)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    pos = [float(row["pos_err_mm"]) for row in best_rows]
    rot = [float(row["rot_err_deg"]) for row in best_rows]
    iters = [float(row["iters"]) for row in best_rows]
    strict = [threshold_ok(p, r, STRICT_POS_MM, STRICT_ROT_DEG) for p, r in zip(pos, rot)]
    medium = [threshold_ok(p, r, MEDIUM_POS_MM, MEDIUM_ROT_DEG) for p, r in zip(pos, rot)]
    loose = [threshold_ok(p, r, LOOSE_POS_MM, LOOSE_ROT_DEG) for p, r in zip(pos, rot)]
    pos_suc = [p for p, ok in zip(pos, strict) if ok]
    rot_suc = [r for r, ok in zip(rot, strict) if ok]
    method = f"CPU-LM-K{k}"
    best_csv = RESULTS / f"cpu_baseline_best_N{n}_K{k}.csv"
    write_csv(
        best_csv,
        best_rows,
        ["target_id", "best_seed_id", "q", "pos_err_mm", "rot_err_deg", "pose_cost", "limit_score", "iters", "near_limit"],
    )
    return {
        "experiment": "cpu_baseline",
        "method": method,
        "robot": "UR10",
        "N": n,
        "K": k,
        "seed_type": "sobol",
        "max_iter": MAX_ITER,
        "wlimit": LIMIT_WEIGHT,
        "barrier_enabled": "true",
        "target_type": "random_reachable",
        "repeat": REPEAT,
        "gpu_time_ms_mean": "",
        "gpu_time_ms_std": "",
        "throughput_targets_per_s_mean": 1000.0 * n / elapsed_ms if elapsed_ms > 0 else 0.0,
        "throughput_targets_per_s_std": "",
        "strict_sr": sum(strict) / n,
        "medium_sr": sum(medium) / n,
        "loose_sr": sum(loose) / n,
        "pos_p95_all_mm": percentile(pos, 95),
        "pos_p95_success_mm": percentile(pos_suc, 95) if pos_suc else "",
        "rot_p95_all_deg": percentile(rot, 95),
        "rot_p95_success_deg": percentile(rot_suc, 95) if rot_suc else "",
        "mean_iter": float(np.mean(iters)),
        "p95_iter": percentile(iters, 95),
        "near_limit_ratio": sum(1 for row in best_rows if row["near_limit"]) / n,
        "joint_violation_count": 0,
        "nan_count": sum(1 for p, r in zip(pos, rot) if math.isnan(p) or math.isnan(r)),
        "inf_count": sum(1 for p, r in zip(pos, rot) if math.isinf(p) or math.isinf(r)),
        "fail_count": n - sum(strict),
        "notes": "Python NumPy CPU-LM baseline; single process; used for order-of-magnitude comparison only",
    }


def main(argv: list[str]) -> int:
    ensure_dirs()
    n_values = N_VALUES
    if len(argv) > 1:
        n_values = [int(x) for x in argv[1].split(",")]
    rows = []
    for n in n_values:
        for k in K_VALUES:
            print(f"running CPU baseline N={n} K={k}", flush=True)
            rows.append(run_case(n, k))
    out = RESULTS / "cpu_baseline_summary.csv"
    write_csv(out, rows, COMMON_FIELDS)
    print(f"wrote {out} rows={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
