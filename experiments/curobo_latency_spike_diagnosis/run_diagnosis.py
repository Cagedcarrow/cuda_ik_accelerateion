#!/usr/bin/env python3
"""cuRobo batch oscillation diagnosis – multi-phase experiment runner.

Phases:
  1  Ordered scan         → results/curobo_latency_scan_ordered.csv
  2  Randomized scan       → results/curobo_latency_scan_randomized.csv
  3  Fresh-process scan    → results/curobo_latency_scan_fresh_process.csv
  4  Fixed max_batch_size  → results/curobo_latency_fixed_max_batch.csv
  5  Memory stats          → results/curobo_memory_summary_logs/
  6  Config dump           → results/config_dump.json
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

# ── paths ──────────────────────────────────────────────────────────────────
_EXP_DIR = Path(__file__).resolve().parent
_RESULTS = _EXP_DIR / "results"
_PROJ = _EXP_DIR.parents[1] / "standard_robot_cuda_ik"
_BENCH_SINGLE = _EXP_DIR / "bench_single_curobo_n.py"

sys.path.insert(0, str(_PROJ / "benchmark"))
sys.path.insert(0, str(_PROJ / "tools"))

from common import (
    compute_pose_error, load_robot_records, load_seed_values, mark_convergence,
)
from robot_model import load_robot_model, quaternion_from_matrix
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo.types import GoalToolPose, Pose

# ── constants ──────────────────────────────────────────────────────────────
URDF_PATH = _PROJ / "urdf" / "ur10_official.urdf"
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
ROBOT = "ur10"
SEED = 42
REPEAT = 30
WARMUP = 3
POS_TOL = 0.01
ROT_TOL = 0.0872664626

# Ordered N list
N_LIST = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

# Random orders (Phase 2)
RANDOM_ORDER_1 = [8000, 4000, 1000, 7000, 5000, 9000, 3000, 10000, 2000, 6000, 500, 100]
RANDOM_ORDER_2 = [4000, 5000, 7000, 8000, 9000, 10000, 3000, 2000, 1000, 500, 100, 6000]
RANDOM_ORDER_3 = [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 500, 100]

# Fresh process N list (Phase 3)
FRESH_N_LIST = [4000, 5000, 7000, 8000]

# CSV field order
CSV_FIELDS = [
    "experiment_name", "run_id", "process_mode", "order_mode", "cache_policy",
    "N", "max_batch_size", "actual_batch_size", "repeat", "warmup",
    "use_cuda_graph",
    "host_wall_mean_ms", "host_wall_std_ms", "host_wall_min_ms",
    "host_wall_max_ms", "host_wall_median_ms",
    "throughput_mean_tps", "conv_rate",
    "memory_allocated_before", "memory_allocated_after",
    "memory_reserved_before", "memory_reserved_after",
    "max_memory_allocated", "max_memory_reserved",
    "num_cuda_malloc_events", "num_cuda_free_events", "num_cuda_sync_events",
    "notes",
]


# ── helpers ────────────────────────────────────────────────────────────────
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
    }


def _bench_single_n(N, max_batch_size, experiment_name, process_mode, order_mode,
                    cache_policy, run_id, warmup=None):
    """Benchmark a single N value – returns result dict."""
    warm = warmup if warmup is not None else WARMUP
    return {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "process_mode": process_mode,
        "order_mode": order_mode,
        "cache_policy": cache_policy,
        "N": N,
        "max_batch_size": max_batch_size,
        "actual_batch_size": N,
        "repeat": REPEAT,
        "warmup": warm,
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


def _solve_single_n(N, max_batch_size, cache_policy="keep_cache",
                     warmup=None, memory_log_dir=None):
    """Run a single N benchmark in-process. Returns (result_dict, raw_times_list)."""
    warm = warmup if warmup is not None else WARMUP
    result = _bench_single_n(N, max_batch_size, "unnamed", "single_process",
                              "ordered", cache_policy, 0, warmup=warm)
    raw_times = []
    error_notes = []

    try:
        if cache_policy == "empty_cache_before_N":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            error_notes.append("empty_cache_before_N")

        targets, _ = load_robot_records(ROBOT, SEED, N)
        seeds = load_seed_values(ROBOT, SEED, N, "zero_seed")
        model = load_robot_model(URDF_PATH, "base_link", "tool0", JOINT_NAMES)

        config = InverseKinematicsCfg.create(
            robot=_robot_cfg(URDF_PATH),
            num_seeds=1, seed_solver_num_seeds=1,
            self_collision_check=False,
            max_batch_size=max_batch_size,
            position_tolerance=POS_TOL, orientation_tolerance=ROT_TOL,
            use_cuda_graph=False, load_collision_spheres=False,
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

        # Memory summary before warmup
        if memory_log_dir:
            _write_memory_summary(memory_log_dir, N, "before")

        # Warmup
        for wi in range(warm):
            _ = solver.solve_pose(
                GoalToolPose.from_poses({target_link: pose}, num_goalset=1),
                current_state=current_state, seed_config=seed_tensor, return_seeds=1,
            )
            torch.cuda.synchronize()

        # Memory summary after warmup
        if memory_log_dir:
            _write_memory_summary(memory_log_dir, N, "after_warmup")

        # Memory before timed
        mem_alloc_before = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_reserved_before = torch.cuda.memory_reserved() / (1024 ** 2)

        # Timed repeats
        conv_rate = float("nan")
        for rep in range(REPEAT):
            start = time.perf_counter()
            solution = solver.solve_pose(
                GoalToolPose.from_poses({target_link: pose}, num_goalset=1),
                current_state=current_state, seed_config=seed_tensor, return_seeds=1,
            )
            torch.cuda.synchronize()
            dt = (time.perf_counter() - start) * 1000.0
            raw_times.append(dt)

            if rep == 0 and solution.js_solution is not None:
                q = solution.js_solution.position.detach().cpu().numpy()[:, 0, :]
                successes = 0
                for i in range(N):
                    fk = model.fk(q[i])
                    pe, re = compute_pose_error(targets[i], fk.reshape(-1))
                    if mark_convergence(pe, re):
                        successes += 1
                conv_rate = successes / N

        # Memory after timed
        mem_alloc_after = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_reserved_after = torch.cuda.memory_reserved() / (1024 ** 2)
        max_mem_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        max_mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

        # Memory summary after repeat
        if memory_log_dir:
            _write_memory_summary(memory_log_dir, N, "after_repeat")

        host_arr = np.array(raw_times)
        result.update({
            "host_wall_mean_ms": float(host_arr.mean()),
            "host_wall_std_ms": float(host_arr.std()),
            "host_wall_min_ms": float(host_arr.min()),
            "host_wall_max_ms": float(host_arr.max()),
            "host_wall_median_ms": float(np.median(host_arr)),
            "throughput_mean_tps": float(N / (host_arr.mean() / 1000.0)),
            "conv_rate": float(conv_rate),
            "memory_allocated_before": float(mem_alloc_before),
            "memory_allocated_after": float(mem_alloc_after),
            "memory_reserved_before": float(mem_reserved_before),
            "memory_reserved_after": float(mem_reserved_after),
            "max_memory_allocated": float(max_mem_alloc),
            "max_memory_reserved": float(max_mem_reserved),
            "notes": "; ".join(error_notes) if error_notes else "",
        })

    except torch.cuda.OutOfMemoryError as e:
        result["error"] = "OOM"
        result["notes"] = f"CUDA OOM at N={N}, max_batch={max_batch_size}: {e}"
    except Exception as e:
        result["error"] = type(e).__name__
        result["notes"] = f"{e}"

    return result, raw_times


def _write_memory_summary(log_dir, N, tag):
    """Write torch.cuda.memory_summary() to a file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"memory_summary_N{N}_{tag}.txt"
    try:
        summary = torch.cuda.memory_summary()
        path.write_text(summary, encoding="utf-8")
        print(f"  [memory] wrote {path}")
    except Exception as e:
        print(f"  [memory] WARNING: could not write memory summary for N={N} {tag}: {e}")


def _write_csv(path, rows, fields=None):
    """Write list-of-dict rows to CSV."""
    fields = fields or CSV_FIELDS
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"  [csv] wrote {len(rows)} rows → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Ordered scan
# ═══════════════════════════════════════════════════════════════════════════
def phase1_ordered_scan():
    print("\n" + "=" * 60)
    print("PHASE 1: Ordered Scan")
    print("=" * 60)
    rows = []
    for i, N in enumerate(N_LIST):
        print(f"\n--- N={N} ({i+1}/{len(N_LIST)}) ---")
        result, raw_times = _solve_single_n(
            N, max_batch_size=N, cache_policy="keep_cache",
            memory_log_dir=_RESULTS / "curobo_memory_summary_logs",
        )
        result["experiment_name"] = "phase1_ordered"
        result["run_id"] = i
        result["process_mode"] = "single_process"
        result["order_mode"] = "ordered"
        if result["error"]:
            print(f"  ERROR: {result['error']} - {result['notes']}")
        else:
            print(f"  mean={result['host_wall_mean_ms']:.2f}ms "
                  f"median={result['host_wall_median_ms']:.2f}ms "
                  f"min={result['host_wall_min_ms']:.2f}ms "
                  f"max={result['host_wall_max_ms']:.2f}ms "
                  f"std={result['host_wall_std_ms']:.2f}ms "
                  f"tp={result['throughput_mean_tps']:.1f}tps "
                  f"conv={result['conv_rate']:.4f}")
        rows.append(result)
    out = _RESULTS / "curobo_latency_scan_ordered.csv"
    _write_csv(out, rows)
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Randomized scan
# ═══════════════════════════════════════════════════════════════════════════
def phase2_randomized_scan():
    print("\n" + "=" * 60)
    print("PHASE 2: Randomized Scan")
    print("=" * 60)
    all_rows = []
    orders = [
        ("RANDOM_ORDER_1", RANDOM_ORDER_1),
        ("RANDOM_ORDER_2", RANDOM_ORDER_2),
        ("RANDOM_ORDER_3", RANDOM_ORDER_3),
    ]
    for order_name, order_list in orders:
        print(f"\n--- {order_name} ---")
        for i, N in enumerate(order_list):
            print(f"\n  N={N} ({i+1}/{len(order_list)})")
            result, raw_times = _solve_single_n(
                N, max_batch_size=N, cache_policy="keep_cache",
            )
            result["experiment_name"] = "phase2_randomized"
            result["run_id"] = order_name
            result["process_mode"] = "single_process"
            result["order_mode"] = order_name
            if result["error"]:
                print(f"    ERROR: {result['error']}")
            else:
                print(f"    mean={result['host_wall_mean_ms']:.2f}ms "
                      f"median={result['host_wall_median_ms']:.2f}ms "
                      f"min={result['host_wall_min_ms']:.2f}ms "
                      f"max={result['host_wall_max_ms']:.2f}ms")
            all_rows.append(result)
    out = _RESULTS / "curobo_latency_scan_randomized.csv"
    _write_csv(out, all_rows)
    return all_rows


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Fresh process scan
# ═══════════════════════════════════════════════════════════════════════════
def phase3_fresh_process():
    print("\n" + "=" * 60)
    print("PHASE 3: Fresh Process Scan")
    print("=" * 60)
    rows = []
    fresh_dir = _RESULTS / "fresh_process_json"
    fresh_dir.mkdir(parents=True, exist_ok=True)
    for N in FRESH_N_LIST:
        print(f"\n--- N={N} (fresh subprocess) ---")
        out_json = fresh_dir / f"fresh_N{N}.json"
        cmd = [
            sys.executable, str(_BENCH_SINGLE),
            "--N", str(N),
            "--repeat", str(REPEAT),
            "--warmup", str(WARMUP),
            "--max_batch_size", str(N),
            "--cache_policy", "keep_cache",
            "--experiment_name", "phase3_fresh_process",
            "--process_mode", "fresh_process",
            "--order_mode", "single",
            "--output", str(out_json),
        ]
        print(f"  Running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print(proc.stdout)
        if proc.stderr:
            print("  STDERR:", proc.stderr[:500])
        if proc.returncode != 0:
            print(f"  Subprocess returned non-zero: {proc.returncode}")
        # Load result JSON
        if out_json.exists():
            result = json.loads(out_json.read_text(encoding="utf-8"))
            rows.append(result)
            if result.get("error"):
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  mean={result['host_wall_mean_ms']:.2f}ms "
                      f"median={result['host_wall_median_ms']:.2f}ms")
        else:
            print(f"  WARNING: output file not found: {out_json}")
    out = _RESULTS / "curobo_latency_scan_fresh_process.csv"
    _write_csv(out, rows)
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Fixed max_batch_size
# ═══════════════════════════════════════════════════════════════════════════
def phase4_fixed_max_batch():
    print("\n" + "=" * 60)
    print("PHASE 4: Fixed max_batch_size = 10000")
    print("=" * 60)
    rows = []
    FIXED_MAX_BATCH = 10000
    for i, N in enumerate(N_LIST):
        print(f"\n--- actual_N={N}, max_batch={FIXED_MAX_BATCH} ({i+1}/{len(N_LIST)}) ---")
        result, raw_times = _solve_single_n(
            N, max_batch_size=FIXED_MAX_BATCH, cache_policy="keep_cache",
        )
        result["experiment_name"] = "phase4_fixed_max_batch"
        result["run_id"] = i
        result["process_mode"] = "single_process"
        result["order_mode"] = "ordered"
        if result["error"]:
            print(f"  ERROR: {result['error']} - {result['notes']}")
        else:
            print(f"  mean={result['host_wall_mean_ms']:.2f}ms "
                  f"median={result['host_wall_median_ms']:.2f}ms "
                  f"min={result['host_wall_min_ms']:.2f}ms "
                  f"max={result['host_wall_max_ms']:.2f}ms "
                  f"std={result['host_wall_std_ms']:.2f}ms "
                  f"tp={result['throughput_mean_tps']:.1f}tps")
        rows.append(result)
    out = _RESULTS / "curobo_latency_fixed_max_batch.csv"
    _write_csv(out, rows)
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: Memory stats (already collected inline, but also write summary)
# ═══════════════════════════════════════════════════════════════════════════
def phase5_memory_stats():
    print("\n" + "=" * 60)
    print("PHASE 5: Memory Stats (summary logging)")
    print("=" * 60)
    mem_dir = _RESULTS / "curobo_memory_summary_logs"
    existing = sorted(mem_dir.glob("memory_summary_*.txt")) if mem_dir.exists() else []
    print(f"  Memory summary logs in {mem_dir}: {len(existing)} files")
    for f in existing:
        size = f.stat().st_size
        print(f"    {f.name} ({size} bytes)")
    if not existing:
        print("  No memory summary logs found (they are written inline during Phase 1).")
    return mem_dir


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: Config dump
# ═══════════════════════════════════════════════════════════════════════════
def phase6_config_dump():
    print("\n" + "=" * 60)
    print("PHASE 6: Config Dump")
    print("=" * 60)
    import curobo
    import platform

    config = {
        "curobo_version": getattr(curobo, "__version__", "unknown"),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_toolkit_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_vram_gb": round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        ) if torch.cuda.is_available() else -1,
        "gpu_compute_capability": (
            f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
            if torch.cuda.is_available() else "N/A"
        ),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "benchmark_settings": {
            "robot": ROBOT,
            "tool": "tool0",
            "seed": SEED,
            "seed_strategy": "zero_seed",
            "repeat": REPEAT,
            "warmup": WARMUP,
            "pos_tolerance_m": POS_TOL,
            "rot_tolerance_rad": ROT_TOL,
            "threshold_name": "Medium (10mm/5°)",
            "use_cuda_graph": False,
            "self_collision_check": False,
            "num_seeds": 1,
            "seed_solver_num_seeds": 1,
            "urdf": str(URDF_PATH),
            "joint_names": JOINT_NAMES,
            "target_link": "tool0",
            "load_collision_spheres": False,
        },
        "n_list": N_LIST,
        "note": "Original full_range data collected on RTX 4090 (24GB); current data on RTX 4060 (8GB). OOM may differ.",
    }

    out = _RESULTS / "config_dump.json"
    out.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"  Config dumped to {out}")
    for k, v in config.items():
        if k != "benchmark_settings":
            print(f"    {k}: {v}")
    return config


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    _RESULTS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("cuRobo Batch Oscillation Diagnosis")
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    print(f"Results dir: {_RESULTS}")
    print("=" * 60)

    # Phase 6 first (config dump – fast, independent)
    phase6_config_dump()

    # Phase 1: Ordered scan
    phase1_ordered_scan()

    # Phase 5: Memory stats (summarize inline data from Phase 1)
    phase5_memory_stats()

    # Phase 2: Randomized scan
    phase2_randomized_scan()

    # Phase 3: Fresh process
    phase3_fresh_process()

    # Phase 4: Fixed max_batch_size
    phase4_fixed_max_batch()

    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETE")
    print(f"Results in: {_RESULTS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
