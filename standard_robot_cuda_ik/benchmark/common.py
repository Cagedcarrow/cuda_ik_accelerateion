from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = DATA_ROOT / "results"
ERRORS_ROOT = RESULTS_ROOT / "errors"

# 三档收敛阈值（按 docs/修改意见/8.md）
# Loose:  30mm / 10°  — 宽松任务/候选生成
# Medium: 10mm /  5°  — 主 benchmark（推荐）
# Strict:  5mm /  1°  — 严格阈值压力验证
THRESHOLD_TIERS = {
    "loose":  {"pos": 0.030, "rot": 0.1745329252},
    "medium": {"pos": 0.010, "rot": 0.0872664626},
    "strict": {"pos": 0.005, "rot": 0.0174532925},
}

# 默认使用 Medium 作为主 benchmark 阈值
POS_THRESHOLD_M = THRESHOLD_TIERS["medium"]["pos"]
ROT_THRESHOLD_RAD = THRESHOLD_TIERS["medium"]["rot"]


@dataclass
class BenchmarkResult:
    solver_name: str
    robot_model: str
    num_targets: int
    repeat_count: int
    uses_gpu: bool
    seed_strategy: str
    includes_jit_compile: bool = False
    includes_data_copy: bool = True
    notes: list[str] = field(default_factory=list)

    kernel_time_only_ms: list[float] = field(default_factory=list)
    gpu_end_to_end_time_ms: list[float] = field(default_factory=list)
    host_api_total_time_ms: list[float] = field(default_factory=list)

    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_targets_per_s: float = 0.0
    convergence_rate: float = 0.0
    failure_count: int = 0
    avg_pos_error_m: float = 0.0
    max_pos_error_m: float = 0.0
    avg_rot_error_rad: float = 0.0
    max_rot_error_rad: float = 0.0
    avg_iterations: float = 0.0
    max_iterations: float = 0.0
    std_host_api_total_ms: float = 0.0
    min_host_api_total_ms: float = 0.0
    max_host_api_total_ms: float = 0.0
    ci95_host_api_total_ms: float = 0.0

    def finalize(self) -> "BenchmarkResult":
        host = np.asarray(self.host_api_total_time_ms if self.host_api_total_time_ms else [0.0], dtype=np.float64)
        self.p50_latency_ms = float(np.percentile(host, 50))
        self.p95_latency_ms = float(np.percentile(host, 95))
        self.p99_latency_ms = float(np.percentile(host, 99))
        self.std_host_api_total_ms = float(np.std(host))
        self.min_host_api_total_ms = float(np.min(host))
        self.max_host_api_total_ms = float(np.max(host))
        self.ci95_host_api_total_ms = float(1.96 * self.std_host_api_total_ms / math.sqrt(len(host))) if len(host) > 0 else 0.0
        return self


def load_robot_records(robot: str, seed: int, N: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    stem = f"{robot}_seed{seed}_N{N}"
    records = json.loads((DATA_ROOT / "targets" / f"{stem}.json").read_text(encoding="utf-8"))
    mats = np.fromfile(DATA_ROOT / "targets" / f"{stem}.bin", dtype=np.float64).reshape(N, 16)
    return mats, records


def load_seed_values(robot: str, seed: int, N: int, strategy: str) -> np.ndarray:
    stem = f"{robot}_seed{seed}_{strategy}_N{N}"
    return np.fromfile(DATA_ROOT / "seeds" / f"{stem}.bin", dtype=np.float64).reshape(N, -1)


def compute_pose_error(target_flat: np.ndarray, achieved_flat: np.ndarray) -> tuple[float, float]:
    target = target_flat.reshape(4, 4)
    achieved = achieved_flat.reshape(4, 4)
    pos_err = float(np.linalg.norm(target[:3, 3] - achieved[:3, 3]))
    diff = target[:3, :3] @ achieved[:3, :3].T
    cos_angle = float(np.clip((np.trace(diff) - 1.0) / 2.0, -1.0, 1.0))
    rot_err = float(abs(math.acos(cos_angle)))
    return pos_err, rot_err


def mark_convergence(pos_err: float, rot_err: float) -> bool:
    return pos_err < POS_THRESHOLD_M and rot_err < ROT_THRESHOLD_RAD


def save_summary(result: BenchmarkResult, robot: str, seed: int, N: int) -> tuple[Path, Path]:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{robot}_{result.solver_name}_N{N}_seed{seed}_repeat{result.repeat_count}_"
        f"{result.seed_strategy}"
    )
    json_path = RESULTS_ROOT / f"{stem}_summary.json"
    csv_path = RESULTS_ROOT / f"{stem}.csv"
    json_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    csv_path.write_text(
        "metric,value\n"
        + "\n".join(
            [
                f"p50_latency_ms,{result.p50_latency_ms}",
                f"p95_latency_ms,{result.p95_latency_ms}",
                f"p99_latency_ms,{result.p99_latency_ms}",
                f"throughput_targets_per_s,{result.throughput_targets_per_s}",
                f"convergence_rate,{result.convergence_rate}",
                f"avg_pos_error_m,{result.avg_pos_error_m}",
                f"avg_rot_error_rad,{result.avg_rot_error_rad}",
                f"avg_iterations,{result.avg_iterations}",
            ]
        ),
        encoding="utf-8",
    )
    return json_path, csv_path


def save_error_log(
    *,
    solver: str,
    robot: str,
    seed: int,
    N: int,
    repeat: int,
    seed_strategy: str,
    exc: BaseException,
) -> Path:
    ERRORS_ROOT.mkdir(parents=True, exist_ok=True)
    stem = f"{robot}_{solver}_N{N}_seed{seed}_repeat{repeat}_{seed_strategy}_error"
    path = ERRORS_ROOT / f"{stem}.json"
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "solver": solver,
        "robot": robot,
        "seed": seed,
        "N": N,
        "repeat": repeat,
        "seed_strategy": seed_strategy,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def format_markdown_table(results: list[BenchmarkResult]) -> str:
    lines = [
        "| Solver | p50 ms | p95 ms | p99 ms | Throughput targets/s | ConvRate | Avg Pos Err m | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.solver_name} | {r.p50_latency_ms:.3f} | {r.p95_latency_ms:.3f} | {r.p99_latency_ms:.3f} | "
            f"{r.throughput_targets_per_s:.1f} | {r.convergence_rate:.3f} | {r.avg_pos_error_m:.6f} | {'; '.join(r.notes)} |"
        )
    return "\n".join(lines)
