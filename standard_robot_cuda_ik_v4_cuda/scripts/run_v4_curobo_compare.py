#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from curobo._src.state.state_joint import JointState
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, Pose

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LOGS = ROOT / "logs"
URDF = ROOT.parent / "standard_robot_cuda_ik" / "urdf" / "ur10_official.urdf"
sys.path.insert(0, str(ROOT.parent / "standard_robot_cuda_ik" / "tools"))
from robot_model import load_robot_model, quaternion_from_matrix  # noqa: E402

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def robot_cfg() -> dict:
    return {
        "robot_cfg": {
            "kinematics": {
                "asset_root_path": str(URDF.parent.resolve()),
                "urdf_path": str(URDF.resolve()),
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


def load_targets(n: int) -> np.ndarray:
    base = np.load(ROOT / "data" / "targets" / "v4_targets_N1000_seed42.npy").astype(np.float64).reshape(1000, 16)
    if n <= 1000:
        return base[:n]
    reps = int(math.ceil(n / 1000))
    return np.tile(base, (reps, 1))[:n]


def pose_error(target_flat: np.ndarray, achieved_flat: np.ndarray) -> tuple[float, float]:
    target = target_flat.reshape(4, 4)
    achieved = achieved_flat.reshape(4, 4)
    pos = float(np.linalg.norm(target[:3, 3] - achieved[:3, 3]))
    diff = target[:3, :3] @ achieved[:3, :3].T
    rot = float(abs(math.acos(float(np.clip((np.trace(diff) - 1.0) / 2.0, -1.0, 1.0)))))
    return pos, rot


def mark(pos: float, rot: float, tier: str) -> bool:
    if tier == "strict":
        return pos < 0.005 and rot < 0.0174532925
    if tier == "medium":
        return pos < 0.010 and rot < 0.0872664626
    return pos < 0.030 and rot < 0.1745329252


def run_one(n: int, repeat: int, warmup: int) -> dict[str, object]:
    targets = load_targets(n)
    model = load_robot_model(URDF, "base_link", "tool0", JOINT_NAMES)
    cfg = InverseKinematicsCfg.create(
        robot=robot_cfg(),
        num_seeds=1,
        seed_solver_num_seeds=1,
        self_collision_check=False,
        max_batch_size=n,
        position_tolerance=0.010,
        orientation_tolerance=0.0872664626,
        use_cuda_graph=True,
        load_collision_spheres=False,
    )
    solver = InverseKinematics(cfg)
    target_link = solver.tool_frames[0]
    positions, quats = [], []
    for row in targets:
        T = row.reshape(4, 4)
        positions.append(T[:3, 3].astype(np.float32))
        quats.append(quaternion_from_matrix(T[:3, :3]).astype(np.float32))
    pose = Pose(
        position=torch.tensor(np.stack(positions), device="cuda", dtype=torch.float32),
        quaternion=torch.tensor(np.stack(quats), device="cuda", dtype=torch.float32),
    )
    seeds_np = np.zeros((n, 6), dtype=np.float32)
    seed_tensor = torch.tensor(seeds_np[:, None, :], device="cuda", dtype=torch.float32)
    current_state = JointState.from_position(torch.tensor(seeds_np, device="cuda", dtype=torch.float32), joint_names=JOINT_NAMES)
    for _ in range(warmup):
        _ = solver.solve_pose(
            GoalToolPose.from_poses({target_link: pose}, num_goalset=1),
            current_state=current_state,
            seed_config=seed_tensor,
            return_seeds=1,
        )
        torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    gpu_ms, host_ms = [], []
    solution = None
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start_ev.record()
        solution = solver.solve_pose(
            GoalToolPose.from_poses({target_link: pose}, num_goalset=1),
            current_state=current_state,
            seed_config=seed_tensor,
            return_seeds=1,
        )
        end_ev.record()
        torch.cuda.synchronize()
        host_ms.append((time.perf_counter() - t0) * 1000.0)
        gpu_ms.append(start_ev.elapsed_time(end_ev))
    pos_errs, rot_errs = [], []
    loose = medium = strict = 0
    if solution is not None and solution.js_solution is not None:
        qs = solution.js_solution.position.detach().cpu().numpy()[:, 0, :]
        for i in range(n):
            fk = model.fk(qs[i])
            p, r = pose_error(targets[i], fk.reshape(-1))
            pos_errs.append(p * 1000.0)
            rot_errs.append(r * 180.0 / math.pi)
            loose += int(mark(p, r, "loose"))
            medium += int(mark(p, r, "medium"))
            strict += int(mark(p, r, "strict"))
    gpu_mean = float(np.mean(gpu_ms))
    return {
        "method": "cuRobo-Graph",
        "N": n,
        "gpu_stream_ms_mean": gpu_mean,
        "gpu_stream_ms_std": float(np.std(gpu_ms)),
        "e2e_ms_mean": float(np.mean(host_ms)),
        "e2e_ms_std": float(np.std(host_ms)),
        "raw_throughput": 1000.0 * n / gpu_mean if gpu_mean > 0 else 0.0,
        "strict_sr": strict / n,
        "medium_sr": medium / n,
        "loose_sr": loose / n,
        "valid_throughput_strict": (1000.0 * n / gpu_mean) * (strict / n) if gpu_mean > 0 else 0.0,
        "pos_p50_all_mm": float(np.percentile(pos_errs, 50)) if pos_errs else 0.0,
        "pos_p95_all_mm": float(np.percentile(pos_errs, 95)) if pos_errs else 0.0,
        "pos_max_all_mm": float(np.max(pos_errs)) if pos_errs else 0.0,
        "rot_p95_all_deg": float(np.percentile(rot_errs, 95)) if rot_errs else 0.0,
        "near_limit_ratio": "",
        "speedup_vs_curobo_graph_stream": 1.0,
        "speedup_vs_curobo_graph_e2e": 1.0,
        "notes": "collision disabled; zero external seed; cuRobo internal optimizer/graph/seeding not equivalent to V4 Sobol-K16",
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in [100, 500, 1000, 5000]:
        t0 = time.perf_counter()
        try:
            row = run_one(n, repeat=5, warmup=2)
        except Exception as exc:
            row = {
                "method": "cuRobo-Graph",
                "N": n,
                "gpu_stream_ms_mean": "",
                "gpu_stream_ms_std": "",
                "e2e_ms_mean": "",
                "e2e_ms_std": "",
                "raw_throughput": "",
                "strict_sr": "",
                "medium_sr": "",
                "loose_sr": "",
                "valid_throughput_strict": "",
                "pos_p50_all_mm": "",
                "pos_p95_all_mm": "",
                "pos_max_all_mm": "",
                "rot_p95_all_deg": "",
                "near_limit_ratio": "",
                "speedup_vs_curobo_graph_stream": "",
                "speedup_vs_curobo_graph_e2e": "",
                "notes": f"FAILED: {type(exc).__name__}: {exc}",
            }
        rows.append(row)
        (LOGS / f"curobo_graph_N{n}.log").write_text(f"elapsed_s={time.perf_counter()-t0:.3f}\nnotes={row['notes']}\n", encoding="utf-8")
    out = RESULTS / "cuda_v4_curobo_compare.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
