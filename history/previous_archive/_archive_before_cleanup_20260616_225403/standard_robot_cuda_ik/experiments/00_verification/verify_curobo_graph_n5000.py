#!/usr/bin/env python3
"""cuRobo-Graph N=5000 Anomaly Verification Script.

Per revision plan §2.1: Verify the anomalous 1,056k t/s at N=5000.
Must check: input targets, output count, graph replay correctness,
success rate per-target, GPU timing, unique target validation.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo.types import GoalToolPose, Pose

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from common import load_robot_records, load_seed_values, compute_pose_error, mark_convergence
from robot_model import load_robot_model, quaternion_from_matrix

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_VALUES = [4000, 5000, 10000]
ROBOT = "ur10"
SEED = 42
REPEAT = 30
WARMUP = 5
STRATEGY = "zero_seed"
POS_TOL = 0.01
ROT_TOL = 0.0872664626
URDF_PATH = Path(__file__).resolve().parents[2] / "urdf" / "ur10_official.urdf"
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def validate_run(N: int) -> dict:
    """Run cuRobo-Graph with thorough validation logging."""
    targets, _ = load_robot_records(ROBOT, SEED, N)
    seeds = load_seed_values(ROBOT, SEED, N, STRATEGY)
    model = load_robot_model(URDF_PATH, "base_link", "tool0", JOINT_NAMES)

    config = InverseKinematicsCfg.create(
        robot={
            "robot_cfg": {
                "kinematics": {
                    "asset_root_path": str(URDF_PATH.parent.resolve()),
                    "urdf_path": str(URDF_PATH.resolve()),
                    "base_link": "base_link", "tool_frames": ["tool0"],
                    "collision_link_names": [
                        "shoulder_link", "upper_arm_link", "forearm_link",
                        "wrist_1_link", "wrist_2_link", "wrist_3_link", "tool0",
                    ],
                    "mesh_link_names": [], "collision_spheres": {},
                    "self_collision_ignore": {}, "self_collision_buffer": {},
                    "collision_sphere_buffer": 0.0,
                    "cspace": {
                        "joint_names": JOINT_NAMES,
                        "max_acceleration": 12.0, "max_jerk": 500.0,
                        "cspace_distance_weight": [1.0] * 6,
                        "null_space_weight": [1.0] * 6,
                        "position_limit_clip": 0.1,
                        "default_joint_position": [0.0, -1.57, 1.57, 0.0, 1.57, 0.0],
                    },
                    "lock_joints": None,
                },
                "dynamics": {"payload_joint": "wrist_3_joint", "payload_mass_range": [0.0, 10.0]},
            }
        },
        num_seeds=1, seed_solver_num_seeds=1,
        self_collision_check=False, max_batch_size=N,
        position_tolerance=POS_TOL, orientation_tolerance=ROT_TOL,
        use_cuda_graph=True, load_collision_spheres=False,
    )
    solver = InverseKinematics(config)
    target_link = solver.tool_frames[0]

    # Build tensors
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

    # Validation checks BEFORE solving
    pos_checksum = pos_tensor.detach().float().sum().item()
    pos_mean = pos_tensor.detach().float().mean().item()
    pos_std = pos_tensor.detach().float().std().item()

    # Count unique target positions
    pos_np = pos_tensor.detach().cpu().numpy()
    unique_count = len(np.unique(pos_np.round(decimals=5), axis=0))

    # Warmup + graph capture
    for _ in range(WARMUP):
        _ = solver.solve_pose(
            GoalToolPose.from_poses({target_link: pose}, num_goalset=1),
            current_state=current_state, seed_config=seed_tensor, return_seeds=1,
        )
        torch.cuda.synchronize()

    # Timed repeats
    gpu_times = []
    host_times = []
    first_solution = None

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for rep in range(REPEAT):
        torch.cuda.synchronize()
        host_start = time.perf_counter()

        start_ev.record()
        solution = solver.solve_pose(
            GoalToolPose.from_poses({target_link: pose}, num_goalset=1),
            current_state=current_state, seed_config=seed_tensor, return_seeds=1,
        )
        end_ev.record()
        torch.cuda.synchronize()

        host_ms = (time.perf_counter() - host_start) * 1000.0
        gpu_ms = start_ev.elapsed_time(end_ev)
        gpu_times.append(gpu_ms)
        host_times.append(host_ms)

        if rep == 0 and solution.js_solution is not None:
            first_solution = solution.js_solution.position.detach().cpu().numpy()[:, 0, :]

    # Validation checks AFTER solving
    q_shape = first_solution.shape if first_solution is not None else "None"
    q_checksum = float(first_solution.sum()) if first_solution is not None else 0.0
    q_mean = float(first_solution.mean()) if first_solution is not None else 0.0
    q_std = float(first_solution.std()) if first_solution is not None else 0.0

    # Per-target convergence
    successes_medium = 0
    if first_solution is not None:
        for i in range(N):
            fk = model.fk(first_solution[i])
            pos_err, rot_err = compute_pose_error(targets[i], fk.reshape(-1))
            if mark_convergence(pos_err, rot_err, "medium"):
                successes_medium += 1

    gpu_mean = float(np.mean(gpu_times))
    gpu_std = float(np.std(gpu_times))
    host_mean = float(np.mean(host_times))
    raw_tps = N / (gpu_mean / 1000.0) if gpu_mean > 0 else 0

    result = {
        "N": N,
        "input_tensor_shape": str(pos_tensor.shape),
        "output_solution_shape": str(q_shape),
        "pos_checksum": float(pos_checksum),
        "pos_mean": float(pos_mean),
        "pos_std": float(pos_std),
        "unique_target_positions": unique_count,
        "q_checksum": float(q_checksum),
        "q_mean": float(q_mean),
        "q_std": float(q_std),
        "success_count": successes_medium,
        "success_rate": successes_medium / N,
        "gpu_stream_ms_mean": gpu_mean,
        "gpu_stream_ms_std": gpu_std,
        "gpu_stream_ms_p50": float(np.percentile(gpu_times, 50)),
        "gpu_stream_ms_p95": float(np.percentile(gpu_times, 95)),
        "gpu_stream_ms_min": float(np.min(gpu_times)),
        "gpu_stream_ms_max": float(np.max(gpu_times)),
        "host_ms_mean": host_mean,
        "raw_throughput_tps": raw_tps,
        "valid_throughput_tps": raw_tps * successes_medium / N,
        "first_5_positions": pos_np[:5].tolist(),
        "first_5_q": first_solution[:5].tolist() if first_solution is not None else [],
        "last_5_q": first_solution[-5:].tolist() if first_solution is not None else [],
    }
    return result


def main():
    import json

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = OUT_DIR / f"curobo_graph_validation_{timestamp}.log"
    csv_path = OUT_DIR / "curobo_graph_n5000_verification.csv"

    results = []
    for N in N_VALUES:
        print(f"\n{'='*60}")
        print(f"Verifying cuRobo-Graph N={N} ...")
        print(f"{'='*60}")
        torch.cuda.empty_cache()  # fresh GPU state
        time.sleep(2)  # thermal stabilization

        r = validate_run(N)
        results.append(r)

        print(f"  Input tensor shape: {r['input_tensor_shape']}")
        print(f"  Unique target positions: {r['unique_target_positions']} / {N}")
        print(f"  Output solution shape: {r['output_solution_shape']}")
        print(f"  Position checksum: {r['pos_checksum']:.4f}")
        print(f"  Q checksum: {r['q_checksum']:.4f}")
        print(f"  Success: {r['success_count']} / {N} (rate={r['success_rate']:.4f})")
        print(f"  GPU stream: {r['gpu_stream_ms_mean']:.3f} ± {r['gpu_stream_ms_std']:.3f} ms "
              f"(p50={r['gpu_stream_ms_p50']:.3f}, p95={r['gpu_stream_ms_p95']:.3f})")
        print(f"  Host E2E: {r['host_ms_mean']:.3f} ms")
        print(f"  Raw throughput: {r['raw_throughput_tps']:.0f} t/s")
        print(f"  Valid throughput: {r['valid_throughput_tps']:.0f} t/s")

        # Write log
        with open(log_path, "a") as f:
            f.write(f"\n{'='*60}\nN={N} VERIFICATION\n{'='*60}\n")
            for k, v in r.items():
                if k not in ("first_5_positions", "first_5_q", "last_5_q"):
                    f.write(f"{k} = {v}\n")
            f.write(f"first_5_positions = {r['first_5_positions']}\n")
            f.write(f"first_5_q = {r['first_5_q']}\n")
            f.write(f"last_5_q = {r['last_5_q']}\n")

    # Write CSV summary
    fieldnames = ["N", "gpu_stream_ms_mean", "gpu_stream_ms_std", "gpu_stream_ms_p50",
                  "gpu_stream_ms_p95", "gpu_stream_ms_min", "gpu_stream_ms_max",
                  "host_ms_mean", "raw_throughput_tps", "valid_throughput_tps",
                  "success_rate", "unique_target_positions"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nLog saved to {log_path}")
    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
