#!/usr/bin/env python3
"""Single-N cuRobo benchmark for fresh-process isolation (Phase 3).

Usage:
    python bench_single_curobo_n.py --N 4000 --repeat 30 --warmup 3 --output results/fresh_N4000.json
    python bench_single_curobo_n.py --N 5000 --repeat 30 --warmup 3 --max_batch_size 10000 --output results/fixed_batch_N5000.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

# Point to the main project benchmark/tools
_PROJ = Path(__file__).resolve().parents[2] / "standard_robot_cuda_ik"
sys.path.insert(0, str(_PROJ / "benchmark"))
sys.path.insert(0, str(_PROJ / "tools"))

from common import compute_pose_error, load_robot_records, load_seed_values, mark_convergence
from robot_model import load_robot_model, quaternion_from_matrix
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo.types import GoalToolPose, Pose

URDF_PATH = _PROJ / "urdf" / "ur10_official.urdf"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Override max_batch_size (default: same as N)")
    parser.add_argument("--cache_policy", type=str, default="keep_cache",
                        choices=["keep_cache", "empty_cache_before_N"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default="single_n")
    parser.add_argument("--process_mode", type=str, default="single_process")
    parser.add_argument("--order_mode", type=str, default="single")
    args = parser.parse_args()

    ROBOT = "ur10"
    SEED = 42
    N = args.N
    max_batch = args.max_batch_size if args.max_batch_size is not None else N
    REPEAT = args.repeat
    WARMUP = args.warmup
    POS_TOL = 0.01
    ROT_TOL = 0.0872664626

    result = {
        "experiment_name": args.experiment_name,
        "process_mode": args.process_mode,
        "order_mode": args.order_mode,
        "cache_policy": args.cache_policy,
        "N": N,
        "max_batch_size": max_batch,
        "actual_batch_size": N,
        "repeat": REPEAT,
        "warmup": WARMUP,
        "use_cuda_graph": False,
        "host_wall_mean_ms": float("nan"),
        "host_wall_std_ms": float("nan"),
        "host_wall_min_ms": float("nan"),
        "host_wall_max_ms": float("nan"),
        "host_wall_median_ms": float("nan"),
        "throughput_mean_tps": float("nan"),
        "conv_rate": float("nan"),
        "memory_allocated_before": float("nan"),
        "memory_allocated_after": float("nan"),
        "memory_reserved_before": float("nan"),
        "memory_reserved_after": float("nan"),
        "max_memory_allocated": float("nan"),
        "max_memory_reserved": float("nan"),
        "num_cuda_malloc_events": float("nan"),
        "num_cuda_free_events": float("nan"),
        "num_cuda_sync_events": float("nan"),
        "notes": "",
        "error": None,
    }

    error_notes = []

    try:
        # --- Cache policy ---
        if args.cache_policy == "empty_cache_before_N":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            error_notes.append("empty_cache_before_N called")

        # --- Load data ---
        targets, _ = load_robot_records(ROBOT, SEED, N)
        seeds = load_seed_values(ROBOT, SEED, N, "zero_seed")
        model = load_robot_model(URDF_PATH, "base_link", "tool0", JOINT_NAMES)

        # --- Build solver ---
        config = InverseKinematicsCfg.create(
            robot=_robot_cfg(URDF_PATH),
            num_seeds=1,
            seed_solver_num_seeds=1,
            self_collision_check=False,
            max_batch_size=max_batch,
            position_tolerance=POS_TOL,
            orientation_tolerance=ROT_TOL,
            use_cuda_graph=False,
            load_collision_spheres=False,
        )
        solver = InverseKinematics(config)
        target_link = solver.tool_frames[0]

        # --- Prepare tensors ---
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

        # Warmup
        for wi in range(WARMUP):
            warm = solver.solve_pose(
                GoalToolPose.from_poses({target_link: pose}, num_goalset=1),
                current_state=current_state, seed_config=seed_tensor, return_seeds=1,
            )
            torch.cuda.synchronize()
            _ = warm

        # Memory before timed repeats
        mem_alloc_before = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_reserved_before = torch.cuda.memory_reserved() / (1024 ** 2)

        # Timed repeats
        host_times = []
        for rep in range(REPEAT):
            start = time.perf_counter()
            solution = solver.solve_pose(
                GoalToolPose.from_poses({target_link: pose}, num_goalset=1),
                current_state=current_state, seed_config=seed_tensor, return_seeds=1,
            )
            torch.cuda.synchronize()
            dt = (time.perf_counter() - start) * 1000.0
            host_times.append(dt)

            if rep == 0 and solution.js_solution is not None:
                q = solution.js_solution.position.detach().cpu().numpy()[:, 0, :]
                successes = 0
                for i in range(N):
                    fk = model.fk(q[i])
                    pe, re = compute_pose_error(targets[i], fk.reshape(-1))
                    if mark_convergence(pe, re):
                        successes += 1
                conv_rate = successes / N

        # Memory after timed repeats
        mem_alloc_after = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_reserved_after = torch.cuda.memory_reserved() / (1024 ** 2)
        max_mem_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        max_mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

        host_arr = np.array(host_times)
        result.update({
            "host_wall_mean_ms": float(host_arr.mean()),
            "host_wall_std_ms": float(host_arr.std()),
            "host_wall_min_ms": float(host_arr.min()),
            "host_wall_max_ms": float(host_arr.max()),
            "host_wall_median_ms": float(np.median(host_arr)),
            "throughput_mean_tps": float(N / (host_arr.mean() / 1000.0)),
            "conv_rate": float(conv_rate) if 'conv_rate' in dir() else float("nan"),
            "memory_allocated_before": float(mem_alloc_before),
            "memory_allocated_after": float(mem_alloc_after),
            "memory_reserved_before": float(mem_reserved_before),
            "memory_reserved_after": float(mem_reserved_after),
            "max_memory_allocated": float(max_mem_alloc),
            "max_memory_reserved": float(max_mem_reserved),
            "notes": "; ".join(error_notes) if error_notes else "",
            "error": None,
        })

    except torch.cuda.OutOfMemoryError as e:
        result["error"] = "OOM"
        result["notes"] = f"CUDA OOM at N={N}, max_batch={max_batch}: {e}"
    except Exception as e:
        result["error"] = type(e).__name__
        result["notes"] = f"Error at N={N}: {e}\n{traceback.format_exc()}"

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    # Also print to stdout for caller to capture
    if result["error"]:
        print(f"ERROR: {result['error']} - {result['notes']}")
    else:
        print(f"N={N} mean={result['host_wall_mean_ms']:.2f}ms "
              f"std={result['host_wall_std_ms']:.2f}ms "
              f"median={result['host_wall_median_ms']:.2f}ms "
              f"min={result['host_wall_min_ms']:.2f}ms "
              f"max={result['host_wall_max_ms']:.2f}ms "
              f"tp={result['throughput_mean_tps']:.1f} tps "
              f"conv={result['conv_rate']:.4f} "
              f"mem_alloc={mem_alloc_before:.1f}->{mem_alloc_after:.1f}MB "
              f"mem_resv={mem_reserved_before:.1f}->{mem_reserved_after:.1f}MB")


if __name__ == "__main__":
    main()
