#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))

from robot_model import load_robot_model  # noqa: E402
from common_metrics import EXPERIMENTS, INPUTS

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def robot():
    return load_robot_model(
        ROOT / "urdf" / "ur10_official.urdf",
        base_link="base_link",
        tip_link="tool0",
        expected_active_joint_names=JOINT_NAMES,
    )


def limits(model) -> tuple[np.ndarray, np.ndarray]:
    arr = model.limits_array().reshape(6, 2)
    return arr[:, 0], arr[:, 1]


def write_targets(model, q: np.ndarray, out_path: Path) -> None:
    targets = np.stack([model.fk(row).reshape(-1) for row in q]).astype(np.float64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    targets.tofile(out_path)


def slice_seed_file(n: int, k: int, out_path: Path) -> None:
    src = EXPERIMENTS / "inputs" / f"seeds_N{n}_K16_q_f64.raw"
    seeds = np.fromfile(src, dtype=np.float64).reshape(n, 16, 6)[:, :k, :].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seeds.astype(np.float64).tofile(out_path)


def near_singular(model, n: int, target_type: str, rng: np.random.Generator) -> np.ndarray:
    qmin, qmax = limits(model)
    q = rng.uniform(qmin, qmax, size=(n, 6))
    if target_type == "wrist_singular":
        q[:, 4] = rng.uniform(-0.03, 0.03, size=n)
    elif target_type == "elbow_singular":
        q[:, 2] = rng.choice([qmin[2] + 0.05, qmax[2] - 0.05], size=n)
    elif target_type == "shoulder_singular":
        q[:, 0] = rng.uniform(-0.05, 0.05, size=n)
        q[:, 1] = rng.choice([qmin[1] + 0.05, qmax[1] - 0.05], size=n)
    else:
        raise ValueError(target_type)
    return q


def near_limit(model, n: int, rng: np.random.Generator) -> np.ndarray:
    qmin, qmax = limits(model)
    q = rng.uniform(qmin, qmax, size=(n, 6))
    joints = rng.integers(0, 6, size=n)
    side = rng.integers(0, 2, size=n)
    offsets = rng.uniform(0.0, 0.087, size=n)
    for row, joint, use_upper, offset in zip(q, joints, side, offsets):
        row[joint] = qmax[joint] - offset if use_upper else qmin[joint] + offset
    return q


def trajectory(model, traj_type: str, num_traj: int, points: int, rng: np.random.Generator) -> np.ndarray:
    qmin, qmax = limits(model)
    chunks = []
    for _ in range(num_traj):
        q0 = rng.uniform(qmin * 0.5, qmax * 0.5)
        if traj_type == "line_50":
            delta = rng.uniform(-0.25, 0.25, size=6)
            qs = np.linspace(q0, np.clip(q0 + delta, qmin, qmax), points)
        elif traj_type == "arc_50":
            axis_a = rng.uniform(-0.18, 0.18, size=6)
            axis_b = rng.uniform(-0.18, 0.18, size=6)
            t = np.linspace(0.0, np.pi, points)[:, None]
            qs = q0 + np.sin(t) * axis_a + (1.0 - np.cos(t)) * axis_b
            qs = np.clip(qs, qmin, qmax)
        elif traj_type == "random_local_50":
            steps = rng.normal(0.0, 0.035, size=(points, 6))
            qs = np.clip(q0 + np.cumsum(steps, axis=0), qmin, qmax)
        else:
            raise ValueError(traj_type)
        chunks.append(qs)
    return np.concatenate(chunks, axis=0)


def main() -> int:
    model = robot()
    rng = np.random.default_rng(42)
    for n in (100, 500, 1000):
        for kind in ("wrist_singular", "elbow_singular", "shoulder_singular"):
            q = near_singular(model, n, kind, rng)
            write_targets(model, q, INPUTS / "near_singular" / f"targets_{kind}_N{n}_T4x4_f64.raw")
            for k in (1, 16):
                slice_seed_file(n, k, INPUTS / "near_singular" / f"seeds_{kind}_N{n}_K{k}_q_f64.raw")
        q = near_limit(model, n, rng)
        write_targets(model, q, INPUTS / "near_limit" / f"targets_N{n}_T4x4_f64.raw")
        for k in (1, 16):
            slice_seed_file(n, k, INPUTS / "near_limit" / f"seeds_N{n}_K{k}_q_f64.raw")
    for kind in ("line_50", "arc_50", "random_local_50"):
        q = trajectory(model, kind, num_traj=20, points=50, rng=rng)
        n = q.shape[0]
        write_targets(model, q, INPUTS / "trajectory" / f"targets_{kind}_N{n}_T4x4_f64.raw")
        for k in (1, 16):
            slice_seed_file(n, k, INPUTS / "trajectory" / f"seeds_{kind}_N{n}_K{k}_q_f64.raw")
    print(f"generated special target inputs under {INPUTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
