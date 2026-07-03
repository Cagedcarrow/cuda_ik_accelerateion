from __future__ import annotations

import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BenchmarkResult, load_seed_values, load_robot_records

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from robot_model import load_robot_model


def run_cuda_6dof_benchmark(robot: str, seed: int, N: int, repeat: int,
                             seed_strategy: str = "zero_seed",
                             ablation_level: int | None = None,
                             pos_tol: float | None = None,
                             rot_tol: float | None = None) -> BenchmarkResult:
    load_robot_records(robot, seed, N)
    load_seed_values(robot, seed, N, seed_strategy)
    urdf_path = Path(__file__).resolve().parents[1] / "urdf" / "ur10_official.urdf"
    load_robot_model(
        urdf_path,
        "base_link",
        "tool0",
        [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
    )
    # Select binary based on ablation level
    if ablation_level is None:
        bin_name = "standard_robot_cuda_runner"
        level_flag: list[str] = []
    else:
        bin_name = f"standard_robot_cuda_runner_A{ablation_level}"
        level_flag = ["--ablation-level", str(ablation_level)]
    binary = Path(__file__).resolve().parents[1] / "build" / bin_name
    targets = Path(__file__).resolve().parents[1] / "data" / "targets" / f"{robot}_seed{seed}_N{N}.bin"
    seeds = Path(__file__).resolve().parents[1] / "data" / "seeds" / f"{robot}_seed{seed}_{seed_strategy}_N{N}.bin"
    proc = subprocess.run(
        [
            str(binary),
            "--targets",
            str(targets),
            "--seeds",
            str(seeds),
            "--max-iter",
            "160",
            "--weight-level",
            "2",
            "--repeat",
            str(repeat),
            *level_flag,
            *(["--pos-tol", str(pos_tol)] if pos_tol is not None else []),
            *(["--rot-tol", str(rot_tol)] if rot_tol is not None else []),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metrics = {}
    for line in proc.stdout.splitlines():
      if "=" in line:
        k, v = line.strip().split("=", 1)
        metrics[k] = v

    solver_label = f"cuda_A{ablation_level}" if ablation_level is not None else "cuda"
    result = BenchmarkResult(
        solver_name=solver_label,
        robot_model=robot,
        num_targets=N,
        repeat_count=repeat,
        uses_gpu=True,
        seed_strategy=seed_strategy,
    )
    result.kernel_time_only_ms = [float(metrics["kernel_time_only_ms_mean"])]
    result.gpu_end_to_end_time_ms = [float(metrics["gpu_end_to_end_ms_mean"])]
    result.host_api_total_time_ms = [float(metrics["host_api_total_ms_mean"])]
    result.throughput_targets_per_s = float(metrics["throughput_targets_per_s"])
    result.convergence_rate = float(metrics["convergence_rate"])
    result.failure_count = N - int(metrics["converged"])
    result.avg_pos_error_m = float(metrics["avg_pos_error_m"])
    result.avg_rot_error_rad = float(metrics["avg_rot_error_rad"])
    result.avg_iterations = float(metrics["avg_iterations"])
    result.notes.append(f"target/seed assets loaded and official UR10 URDF parsed before runner launch; max_iter=160; weight_level=2; ablation_level={ablation_level if ablation_level is not None else 'A6'}")
    return result.finalize()
