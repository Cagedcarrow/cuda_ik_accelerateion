from __future__ import annotations

import time
import csv
from pathlib import Path
import sys

import numpy as np
import torch
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo.types import GoalToolPose, Pose

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    BenchmarkResult, THRESHOLD_TIERS, compute_pose_error,
    compute_error_percentiles,
    load_robot_records, load_seed_values, mark_convergence,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from robot_model import load_robot_model, quaternion_from_matrix


def _robot_cfg(urdf_path: Path) -> dict:
    return {
        "robot_cfg": {
            "kinematics": {
                "asset_root_path": str(urdf_path.parent.resolve()),
                "urdf_path": str(urdf_path.resolve()),
                "base_link": "base_link",
                "tool_frames": ["tool0"],
                "collision_link_names": [
                    "shoulder_link", "upper_arm_link", "forearm_link",
                    "wrist_1_link", "wrist_2_link", "wrist_3_link", "tool0",
                ],
                "mesh_link_names": [],
                "collision_spheres": {},
                "self_collision_ignore": {},
                "self_collision_buffer": {},
                "collision_sphere_buffer": 0.0,
                "cspace": {
                    "joint_names": [
                        "shoulder_pan_joint", "shoulder_lift_joint",
                        "elbow_joint", "wrist_1_joint",
                        "wrist_2_joint", "wrist_3_joint",
                    ],
                    "max_acceleration": 12.0,
                    "max_jerk": 500.0,
                    "cspace_distance_weight": [1.0] * 6,
                    "null_space_weight": [1.0] * 6,
                    "position_limit_clip": 0.1,
                    "default_joint_position": [0.0, -1.57, 1.57, 0.0, 1.57, 0.0],
                },
                "lock_joints": None,
            },
            "dynamics": {"payload_joint": "wrist_3_joint", "payload_mass_range": [0.0, 10.0]},
        }
    }


def run_curobo_benchmark(
    robot: str,
    seed: int,
    N: int,
    repeat: int,
    seed_strategy: str = "zero_seed",
    pos_tol: float | None = None,
    rot_tol: float | None = None,
    use_cuda_graph: bool = False,
    warmup: int = 3,
    error_log_path: str | None = None,
) -> BenchmarkResult:
    """Run cuRobo IK benchmark with configurable CUDA Graph and GPU event timing.

    Parameters
    ----------
    use_cuda_graph : bool
        Enable cuRobo CUDA Graph mode (use_cuda_graph=True in InverseKinematicsCfg).
    warmup : int
        Number of warm-up (untimed) iterations before the timed loop.
    error_log_path : str | None
        If provided, write per-target error CSV to this path.
    """
    targets, _ = load_robot_records(robot, seed, N)
    seeds = load_seed_values(robot, seed, N, seed_strategy)
    urdf_path = Path(__file__).resolve().parents[1] / "urdf" / "ur10_official.urdf"
    model = load_robot_model(
        urdf_path, "base_link", "tool0",
        [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ],
    )

    _pos_tol = pos_tol if pos_tol is not None else 0.01
    _rot_tol = rot_tol if rot_tol is not None else 0.0872664626

    config = InverseKinematicsCfg.create(
        robot=_robot_cfg(urdf_path),
        num_seeds=1,
        seed_solver_num_seeds=1,
        self_collision_check=False,
        max_batch_size=N,
        position_tolerance=_pos_tol,
        orientation_tolerance=_rot_tol,
        use_cuda_graph=use_cuda_graph,
        load_collision_spheres=False,
    )
    solver = InverseKinematics(config)
    target_link = solver.tool_frames[0]

    positions = []
    quats = []
    for T_flat in targets:
        T = T_flat.reshape(4, 4)
        positions.append(T[:3, 3].astype(np.float32))
        quats.append(quaternion_from_matrix(T[:3, :3]).astype(np.float32))

    pos_tensor = torch.tensor(np.stack(positions), device="cuda", dtype=torch.float32)
    quat_tensor = torch.tensor(np.stack(quats), device="cuda", dtype=torch.float32)
    seed_tensor = torch.tensor(seeds[:, None, :], device="cuda", dtype=torch.float32)
    pose = Pose(position=pos_tensor, quaternion=quat_tensor)
    current_state = JointState.from_position(
        torch.tensor(seeds, device="cuda", dtype=torch.float32),
        joint_names=model.active_joint_names(),
    )

    # Warm-up iterations (excluded from timing)
    for _ in range(warmup):
        _ = solver.solve_pose(
            GoalToolPose.from_poses({target_link: pose}, num_goalset=1),
            current_state=current_state,
            seed_config=seed_tensor,
            return_seeds=1,
        )
        torch.cuda.synchronize()

    result = BenchmarkResult(
        solver_name="curobo",
        robot_model=robot,
        num_targets=N,
        repeat_count=repeat,
        uses_gpu=True,
        seed_strategy=seed_strategy,
        includes_jit_compile=False,
        use_cuda_graph=use_cuda_graph,
    )

    # CUDA events for GPU stream timing
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for rep in range(repeat):
        torch.cuda.synchronize()

        # Host wall-clock timing
        host_start = time.perf_counter()

        # GPU stream timing (CUDA events)
        start_ev.record()
        solution = solver.solve_pose(
            GoalToolPose.from_poses({target_link: pose}, num_goalset=1),
            current_state=current_state,
            seed_config=seed_tensor,
            return_seeds=1,
        )
        end_ev.record()
        torch.cuda.synchronize()

        host_dt_ms = (time.perf_counter() - host_start) * 1000.0
        gpu_dt_ms = start_ev.elapsed_time(end_ev)

        result.host_api_total_time_ms.append(host_dt_ms)
        result.gpu_stream_time_ms.append(gpu_dt_ms)

        # Per-target error collection on first repeat
        if rep == 0 and solution.js_solution is not None:
            q_solutions = solution.js_solution.position.detach().cpu().numpy()[:, 0, :]
            pos_errs = []
            rot_errs = []
            successes_loose = 0
            successes_medium = 0
            successes_strict = 0
            for i in range(N):
                fk = model.fk(q_solutions[i])
                pos_err, rot_err = compute_pose_error(targets[i], fk.reshape(-1))
                pos_errs.append(pos_err)
                rot_errs.append(rot_err)
                result.per_target_errors.append({
                    "target_id": i,
                    "pos_error_m": float(pos_err),
                    "rot_error_rad": float(rot_err),
                    "converged_loose": int(mark_convergence(pos_err, rot_err, "loose")),
                    "converged_medium": int(mark_convergence(pos_err, rot_err, "medium")),
                    "converged_strict": int(mark_convergence(pos_err, rot_err, "strict")),
                })
                if mark_convergence(pos_err, rot_err, "loose"):
                    successes_loose += 1
                if mark_convergence(pos_err, rot_err, "medium"):
                    successes_medium += 1
                if mark_convergence(pos_err, rot_err, "strict"):
                    successes_strict += 1

            result.avg_pos_error_m = float(np.mean(pos_errs))
            result.max_pos_error_m = float(np.max(pos_errs))
            result.pos_error_p50_m = float(np.median(pos_errs))
            result.pos_error_p95_m = float(np.percentile(pos_errs, 95))
            result.avg_rot_error_rad = float(np.mean(rot_errs))
            result.max_rot_error_rad = float(np.max(rot_errs))
            result.rot_error_p50_rad = float(np.median(rot_errs))
            result.rot_error_p95_rad = float(np.percentile(rot_errs, 95))
            result.convergence_rate = successes_medium / N
            result.convergence_rate_loose = successes_loose / N
            result.convergence_rate_strict = successes_strict / N
            result.failure_count = N - successes_medium

    # Compute throughput from GPU stream time (fairer GPU-to-GPU comparison)
    gpu_mean = float(np.mean(result.gpu_stream_time_ms)) if result.gpu_stream_time_ms else float(np.mean(result.host_api_total_time_ms))
    result.throughput_targets_per_s = N / (gpu_mean / 1000.0)

    graph_tag = "Graph" if use_cuda_graph else "NoGraph"
    result.notes.append(f"cuRobo {graph_tag}: use_cuda_graph={use_cuda_graph}")
    result.notes.append("custom robot dict used to force same URDF/TCP as project")
    result.notes.append("single external seed per target; warm-up solves excluded")
    result.notes.append("GPU stream timing via torch.cuda.Event")

    finalized = result.finalize()

    # Write per-target error CSV if requested
    if error_log_path and result.per_target_errors:
        Path(error_log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(error_log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "target_id", "pos_error_m", "rot_error_rad",
                "converged_loose", "converged_medium", "converged_strict",
            ])
            writer.writeheader()
            writer.writerows(result.per_target_errors)

    return finalized


def run_curobo_benchmark_nograph(
    robot: str, seed: int, N: int, repeat: int,
    seed_strategy: str = "zero_seed",
    pos_tol: float | None = None, rot_tol: float | None = None,
    warmup: int = 3, error_log_path: str | None = None,
) -> BenchmarkResult:
    """Convenience wrapper: cuRobo without CUDA Graph."""
    return run_curobo_benchmark(
        robot=robot, seed=seed, N=N, repeat=repeat,
        seed_strategy=seed_strategy,
        pos_tol=pos_tol, rot_tol=rot_tol,
        use_cuda_graph=False, warmup=warmup,
        error_log_path=error_log_path,
    )


def run_curobo_benchmark_graph(
    robot: str, seed: int, N: int, repeat: int,
    seed_strategy: str = "zero_seed",
    pos_tol: float | None = None, rot_tol: float | None = None,
    warmup: int = 3, error_log_path: str | None = None,
) -> BenchmarkResult:
    """Convenience wrapper: cuRobo with CUDA Graph enabled."""
    return run_curobo_benchmark(
        robot=robot, seed=seed, N=N, repeat=repeat,
        seed_strategy=seed_strategy,
        pos_tol=pos_tol, rot_tol=rot_tol,
        use_cuda_graph=True, warmup=warmup,
        error_log_path=error_log_path,
    )
