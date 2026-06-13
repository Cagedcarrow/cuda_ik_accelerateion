from __future__ import annotations

import time
from pathlib import Path
import sys

import numpy as np
import torch
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo.types import GoalToolPose, Pose

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BenchmarkResult, compute_pose_error, load_robot_records, load_seed_values, mark_convergence

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
                    "shoulder_link",
                    "upper_arm_link",
                    "forearm_link",
                    "wrist_1_link",
                    "wrist_2_link",
                    "wrist_3_link",
                    "tool0",
                ],
                "mesh_link_names": [],
                "collision_spheres": {},
                "self_collision_ignore": {},
                "self_collision_buffer": {},
                "collision_sphere_buffer": 0.0,
                "cspace": {
                    "joint_names": [
                        "shoulder_pan_joint",
                        "shoulder_lift_joint",
                        "elbow_joint",
                        "wrist_1_joint",
                        "wrist_2_joint",
                        "wrist_3_joint",
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


def run_curobo_benchmark(robot: str, seed: int, N: int, repeat: int, seed_strategy: str = "zero_seed",
                        pos_tol: float | None = None, rot_tol: float | None = None) -> BenchmarkResult:
    targets, _ = load_robot_records(robot, seed, N)
    seeds = load_seed_values(robot, seed, N, seed_strategy)
    urdf_path = Path(__file__).resolve().parents[1] / "urdf" / "ur10_official.urdf"
    model = load_robot_model(
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
    config = InverseKinematicsCfg.create(
        robot=_robot_cfg(urdf_path),
        num_seeds=1,
        seed_solver_num_seeds=1,
        self_collision_check=False,
        max_batch_size=N,
        position_tolerance=pos_tol if pos_tol is not None else 0.01,
        orientation_tolerance=rot_tol if rot_tol is not None else 0.0872664626,
        use_cuda_graph=False,
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

    warm = solver.solve_pose(
        GoalToolPose.from_poses({target_link: pose}, num_goalset=1),
        current_state=current_state,
        seed_config=seed_tensor,
        return_seeds=1,
    )
    torch.cuda.synchronize()
    _ = warm

    result = BenchmarkResult(
        solver_name="curobo",
        robot_model=robot,
        num_targets=N,
        repeat_count=repeat,
        uses_gpu=True,
        seed_strategy=seed_strategy,
        includes_jit_compile=False,
    )
    for rep in range(repeat):
        start = time.perf_counter()
        solution = solver.solve_pose(
            GoalToolPose.from_poses({target_link: pose}, num_goalset=1),
            current_state=current_state,
            seed_config=seed_tensor,
            return_seeds=1,
        )
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - start) * 1000.0
        result.host_api_total_time_ms.append(dt_ms)
        if rep == 0 and solution.js_solution is not None:
            q_solutions = solution.js_solution.position.detach().cpu().numpy()[:, 0, :]
            pos_errs = []
            rot_errs = []
            successes = 0
            for i in range(N):
                fk = model.fk(q_solutions[i])
                pos_err, rot_err = compute_pose_error(targets[i], fk.reshape(-1))
                pos_errs.append(pos_err)
                rot_errs.append(rot_err)
                if mark_convergence(pos_err, rot_err):
                    successes += 1
            result.avg_pos_error_m = float(np.mean(pos_errs))
            result.max_pos_error_m = float(np.max(pos_errs))
            result.avg_rot_error_rad = float(np.mean(rot_errs))
            result.max_rot_error_rad = float(np.max(rot_errs))
            result.convergence_rate = successes / N
            result.failure_count = N - successes
    result.throughput_targets_per_s = N / (np.mean(result.host_api_total_time_ms) / 1000.0)
    result.notes.append("custom robot dict used to force same URDF/TCP as project")
    result.notes.append("single external seed per target is passed through current_state + seed_config; LM seed stage is kept at one seed for fairness; warm-up solve excluded")
    return result.finalize()
