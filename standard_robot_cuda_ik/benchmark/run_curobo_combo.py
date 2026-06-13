#!/usr/bin/env python3
"""Run cuRobo benchmark at three combo threshold tiers, repeat=30, zero_seed.
   Uses the EXACT same API as bench_curobo.py."""

import sys, time, math
from pathlib import Path
import numpy as np
import torch
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo.types import GoalToolPose, Pose

PROJ = Path(__file__).resolve().parents[1] / "standard_robot_cuda_ik"
sys.path.insert(0, str(PROJ / "benchmark"))
sys.path.insert(0, str(PROJ / "tools"))

from common import compute_pose_error, load_robot_records, load_seed_values, mark_convergence
from robot_model import load_robot_model, quaternion_from_matrix

THRESHOLD_TIERS = {
    "loose":  {"pos": 0.030, "rot": 0.1745329252},
    "medium": {"pos": 0.010, "rot": 0.0872664626},
    "strict": {"pos": 0.005, "rot": 0.0174532925},
}

N_VALUES = [100, 500, 1000, 5000]
REPEAT = 30
SEED = 42
ROBOT = "ur10"
OUT_DIR = Path(__file__).resolve().parent / "results" / "combo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

URDF_PATH = PROJ / "urdf" / "ur10_official.urdf"
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


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
                    "joint_names": JOINT_NAMES,
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


def run_curobo_tier(tier_name, pos_tol, rot_tol, N):
    print(f"\n=== cuRobo {tier_name} N={N} (pos={pos_tol}m, rot={rot_tol:.6f}rad) ===", flush=True)

    targets, _ = load_robot_records(ROBOT, SEED, N)
    seeds = load_seed_values(ROBOT, SEED, N, "zero_seed")
    model = load_robot_model(URDF_PATH, "base_link", "tool0", JOINT_NAMES)

    config = InverseKinematicsCfg.create(
        robot=_robot_cfg(URDF_PATH),
        num_seeds=1,
        seed_solver_num_seeds=1,
        self_collision_check=False,
        max_batch_size=N,
        position_tolerance=pos_tol,
        orientation_tolerance=rot_tol,
        use_cuda_graph=False,
        load_collision_spheres=False,
    )
    solver = InverseKinematics(config)
    target_link = solver.tool_frames[0]

    positions, quats = [], []
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

    # Warm-up
    warm = solver.solve_pose(
        GoalToolPose.from_poses({target_link: pose}, num_goalset=1),
        current_state=current_state, seed_config=seed_tensor, return_seeds=1,
    )
    torch.cuda.synchronize()
    _ = warm

    # Timed repeats
    host_times_ms = []
    for rep in range(REPEAT):
        start = time.perf_counter()
        solution = solver.solve_pose(
            GoalToolPose.from_poses({target_link: pose}, num_goalset=1),
            current_state=current_state, seed_config=seed_tensor, return_seeds=1,
        )
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - start) * 1000.0
        host_times_ms.append(dt_ms)

        if rep == 0 and solution.js_solution is not None:
            q_solutions = solution.js_solution.position.detach().cpu().numpy()[:, 0, :]
            p_errs, r_errs, successes = [], [], 0
            for i in range(N):
                fk = model.fk(q_solutions[i])
                p_err, r_err = compute_pose_error(targets[i], fk.reshape(-1))
                p_errs.append(p_err)
                r_errs.append(r_err)
                if mark_convergence(p_err, r_err):
                    successes += 1

    host_arr = np.array(host_times_ms)
    throughput = N / (host_arr.mean() / 1000.0)

    result = {
        "tier": tier_name, "pos_tol": pos_tol, "rot_tol": rot_tol, "N": N,
        "repeat": REPEAT,
        "host_api_total_ms_mean": float(host_arr.mean()),
        "host_api_total_ms_p50": float(np.percentile(host_arr, 50)),
        "host_api_total_ms_p95": float(np.percentile(host_arr, 95)),
        "host_api_total_ms_p99": float(np.percentile(host_arr, 99)),
        "throughput_targets_per_s": float(throughput),
        "converged": successes, "convergence_rate": successes / N,
        "avg_pos_error_m": float(np.mean(p_errs)),
        "avg_rot_error_rad": float(np.mean(r_errs)),
    }

    log_path = OUT_DIR / f"curobo_{tier_name}_N{N}.log"
    with open(log_path, "w") as f:
        for k, v in result.items():
            f.write(f"{k}={v}\n")

    print(f"  TP={throughput:.1f}  ConvRate={successes/N:.4f}  HostMs={host_arr.mean():.3f}  pErr={np.mean(p_errs):.4f}", flush=True)
    return result


def main():
    all_results = []
    for tier_name, tols in THRESHOLD_TIERS.items():
        for N in N_VALUES:
            r = run_curobo_tier(tier_name, tols["pos"], tols["rot"], N)
            all_results.append(r)

    csv_path = OUT_DIR / "curobo_summary.csv"
    with open(csv_path, "w") as f:
        keys = ["tier", "N", "throughput_targets_per_s", "convergence_rate",
                "host_api_total_ms_mean", "avg_pos_error_m", "avg_rot_error_rad"]
        f.write(",".join(keys) + "\n")
        for r in all_results:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    print(f"\n=== All cuRobo done → {OUT_DIR} ===")


if __name__ == "__main__":
    main()
