from __future__ import annotations

import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BenchmarkResult, compute_pose_error, load_robot_records, load_seed_values, mark_convergence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from robot_model import load_robot_model


def _solve_numeric_dls(model, target_flat: np.ndarray, q0: np.ndarray, max_iter: int = 100) -> tuple[np.ndarray, int]:
    q = q0.astype(np.float64).copy()
    target = target_flat.reshape(4, 4)
    lambda_value = 1e-3
    for it in range(max_iter):
        cur = model.fk(q)
        pos_err = target[:3, 3] - cur[:3, 3]
        R_diff = cur[:3, :3].T @ target[:3, :3]
        rot_vec = np.array(
            [
                R_diff[2, 1] - R_diff[1, 2],
                R_diff[0, 2] - R_diff[2, 0],
                R_diff[1, 0] - R_diff[0, 1],
            ],
            dtype=np.float64,
        ) * 0.5
        err = np.concatenate([pos_err, rot_vec])
        if np.linalg.norm(pos_err) < 0.03 and np.linalg.norm(rot_vec) < 0.5235987755982988:
            return q, it + 1
        J = np.zeros((6, model.dof), dtype=np.float64)
        eps = 1e-6
        for j in range(model.dof):
            qp = q.copy()
            qm = q.copy()
            qp[j] += eps
            qm[j] -= eps
            Tp = model.fk(qp)
            Tm = model.fk(qm)
            J[:3, j] = (Tp[:3, 3] - Tm[:3, 3]) * (0.5 / eps)
            R_delta = Tp[:3, :3] - Tm[:3, :3]
            J[3:, j] = np.array([R_delta[2, 1] - R_delta[1, 2], R_delta[0, 2] - R_delta[2, 0], R_delta[1, 0] - R_delta[0, 1]]) * (0.25 / eps)
        H = J.T @ J + lambda_value * np.eye(model.dof)
        g = J.T @ err
        dq = np.linalg.solve(H, g)
        q += dq
    return q, max_iter


def run_numeric_dls_benchmark(robot: str, seed: int, N: int, repeat: int, seed_strategy: str = "zero_seed") -> BenchmarkResult:
    targets, _ = load_robot_records(robot, seed, N)
    seeds = load_seed_values(robot, seed, N, seed_strategy)
    model = load_robot_model(
        Path(__file__).resolve().parents[1] / "urdf" / "ur10_official.urdf",
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
    result = BenchmarkResult(
        solver_name="numeric_dls",
        robot_model=robot,
        num_targets=N,
        repeat_count=repeat,
        uses_gpu=False,
        seed_strategy=seed_strategy,
        includes_data_copy=False,
    )
    for rep in range(repeat):
        run_begin = time.perf_counter()
        pos_errs = []
        rot_errs = []
        iters = []
        successes = 0
        for i in range(N):
            q_sol, used = _solve_numeric_dls(model, targets[i], seeds[i])
            fk = model.fk(q_sol).reshape(-1)
            pos_err, rot_err = compute_pose_error(targets[i], fk)
            pos_errs.append(pos_err)
            rot_errs.append(rot_err)
            iters.append(used)
            if mark_convergence(pos_err, rot_err):
                successes += 1
        if rep == 0:
            result.avg_pos_error_m = float(np.mean(pos_errs))
            result.max_pos_error_m = float(np.max(pos_errs))
            result.avg_rot_error_rad = float(np.mean(rot_errs))
            result.max_rot_error_rad = float(np.max(rot_errs))
            result.avg_iterations = float(np.mean(iters))
            result.max_iterations = float(np.max(iters))
            result.convergence_rate = successes / N
            result.failure_count = N - successes
        result.host_api_total_time_ms.append((time.perf_counter() - run_begin) * 1000.0)
    result.throughput_targets_per_s = N / (np.mean(result.host_api_total_time_ms) / 1000.0)
    result.notes.append("numpy numerical DLS baseline")
    return result.finalize()
