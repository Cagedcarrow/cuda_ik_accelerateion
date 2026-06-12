#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from robot_model import (
    export_cuda_constants_header,
    file_md5,
    load_robot_model,
    quaternion_from_matrix,
    write_targets_csv,
)


ROBOTS = {
    "ur10": {
        "urdf": "urdf/ur10_official.urdf",
        "base_link": "base_link",
        "tip_link": "tool0",
        "active_joint_names": [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        "home_seed": [0.0, -1.5707963267948966, 1.5707963267948966, 0.0, 1.5707963267948966, 0.0],
        "sizes": [100, 500, 1000, 5000],
    },
    "panda": {
        "urdf": "urdf/panda_7dof.urdf",
        "base_link": "panda_link0",
        "tip_link": "panda_link8",
        "active_joint_names": [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ],
        "home_seed": [0.0, -1.3, 0.0, -2.5, 0.0, 1.5, 0.8],
        "sizes": [100, 500, 1000],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="ur10", choices=ROBOTS.keys())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--N", type=int, nargs="*", default=None)
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1], type=Path)
    return parser.parse_args()


def generate_seed_set(model, q_samples: np.ndarray, home_seed: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    zero_seed = np.zeros_like(q_samples)
    home = np.repeat(home_seed.reshape(1, -1), q_samples.shape[0], axis=0)
    random_seed = np.vstack([model.sample_joint_vector(rng) for _ in range(q_samples.shape[0])])
    perturb = rng.normal(loc=0.0, scale=0.015, size=q_samples.shape)
    near_gt = q_samples + perturb
    return {
        "zero_seed": zero_seed,
        "home_seed": home,
        "random_seed": random_seed,
        "near_ground_truth_seed": near_gt,
    }


def smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _workspace_ok(transform: np.ndarray) -> bool:
    xyz = transform[:3, 3]
    radius_xy = float(np.linalg.norm(xyz[:2]))
    return 0.20 <= xyz[2] <= 1.35 and 0.20 <= radius_xy <= 1.25 and xyz[0] >= -0.15


def _sample_anchor(model, rng: np.random.Generator, home_seed: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    for _ in range(256):
        if model.dof == 6:
            q1 = home_seed[0] + rng.normal(0.0, 0.35)
            q2 = home_seed[1] + rng.normal(0.0, 0.25)
            q3 = home_seed[2] + rng.normal(0.0, 0.35)
            q4 = -(q2 + q3) + rng.normal(0.0, 0.05)
            q5 = home_seed[4] + rng.normal(0.0, 0.05)
            q6 = home_seed[5] + rng.normal(0.0, 0.08)
            candidate = np.array([q1, q2, q3, q4, q5, q6], dtype=np.float64)
        else:
            span = np.minimum(0.35, 0.10 * (upper - lower))
            candidate = home_seed + rng.normal(loc=0.0, scale=span)
        candidate = np.clip(candidate, lower, upper)
        if _workspace_ok(model.fk(candidate)):
            return candidate
    return np.clip(home_seed, lower, upper)


def generate_reasonable_trajectory_q(
    model,
    N: int,
    home_seed: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    limits = model.limits_array().reshape(model.dof, 2)
    full_lower = limits[:, 0]
    full_upper = limits[:, 1]
    margin = 0.08 * (full_upper - full_lower)
    lower = full_lower + margin
    upper = full_upper - margin

    segment_len = min(96, max(32, N // 8))
    segment_count = max(3, math.ceil((N - 1) / segment_len))
    anchors = [np.clip(home_seed, lower, upper)]
    for _ in range(segment_count):
        anchors.append(_sample_anchor(model, rng, home_seed, lower, upper))
    anchors = np.asarray(anchors, dtype=np.float64)

    q_samples = []
    for seg in range(segment_count):
        q0 = anchors[seg]
        q1 = anchors[seg + 1]
        count = min(segment_len, N - len(q_samples))
        if count <= 0:
            break
        t = np.linspace(0.0, 1.0, count, endpoint=(seg == segment_count - 1), dtype=np.float64)
        s = smoothstep(t)
        interp = (1.0 - s[:, None]) * q0[None, :] + s[:, None] * q1[None, :]
        jitter = rng.normal(loc=0.0, scale=0.005, size=interp.shape)
        interp = np.clip(interp + jitter, lower, upper)
        for row in interp:
            if _workspace_ok(model.fk(row)):
                q_samples.append(row.copy())
            if len(q_samples) == N:
                break
        if len(q_samples) == N:
            break

    while len(q_samples) < N:
        q_samples.append(_sample_anchor(model, rng, home_seed, lower, upper))
    return np.asarray(q_samples[:N], dtype=np.float64)


def main() -> None:
    args = parse_args()
    root = args.project_root
    spec = ROBOTS[args.robot]
    urdf_path = root / spec["urdf"]
    model = load_robot_model(
        urdf_path,
        spec["base_link"],
        spec["tip_link"],
        spec["active_joint_names"],
    )

    include_dir = root / "include" / "standard_robot_cuda_ik" / "generated"
    include_dir.mkdir(parents=True, exist_ok=True)
    export_cuda_constants_header(include_dir / f"{args.robot}_model_constants.h", model)

    rng = np.random.default_rng(args.seed)
    sizes = args.N if args.N else spec["sizes"]
    targets_dir = root / "data" / "targets"
    seeds_dir = root / "data" / "seeds"
    targets_dir.mkdir(parents=True, exist_ok=True)
    seeds_dir.mkdir(parents=True, exist_ok=True)

    log_lines = [
        "# Target Generation Report",
        "",
        f"- Robot: `{args.robot}`",
        f"- URDF: `{urdf_path}`",
        f"- URDF MD5: `{file_md5(urdf_path)}`",
        f"- Random seed: `{args.seed}`",
        f"- Main benchmark seed strategy: `zero_seed`",
        "- Target generation mode: `smooth_joint_trajectory_from_home`",
        "- Workspace guard: `z in [0.20, 1.35] m`, `xy radius in [0.20, 1.25] m`, `x >= -0.15 m`",
        "",
    ]

    for N in sizes:
        q_samples = generate_reasonable_trajectory_q(
            model,
            N,
            np.asarray(spec["home_seed"], dtype=np.float64),
            rng,
        )
        transforms = np.stack([model.fk(q) for q in q_samples], axis=0)
        records = []
        for idx in range(N):
            T = transforms[idx]
            quat = quaternion_from_matrix(T[:3, :3])
            limits_ok = True
            for joint_value, joint in zip(q_samples[idx], model.active_joints):
                if joint.lower is not None and joint_value < joint.lower:
                    limits_ok = False
                if joint.upper is not None and joint_value > joint.upper:
                    limits_ok = False
            records.append(
                {
                    "target_id": idx,
                    "q_sampled": q_samples[idx].tolist(),
                    "position_xyz": T[:3, 3].tolist(),
                    "rotation_matrix": T[:3, :3].reshape(-1).tolist(),
                    "quaternion_wxyz": quat.tolist(),
                    "transform_row_major": T.reshape(-1).tolist(),
                    "within_joint_limits": limits_ok,
                    "fk_verified": True,
                    "target_profile": "smooth_joint_trajectory_from_home",
                }
            )

        xyz = transforms[:, :3, 3]
        radius_xy = np.linalg.norm(xyz[:, :2], axis=1)
        joint_std = np.std(q_samples, axis=0)

        stem = f"{args.robot}_seed{args.seed}_N{N}"
        json_path = targets_dir / f"{stem}.json"
        csv_path = targets_dir / f"{stem}.csv"
        bin_path = targets_dir / f"{stem}.bin"
        json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        write_targets_csv(csv_path, records)
        transforms.reshape(-1).astype(np.float64).tofile(bin_path)

        seed_sets = generate_seed_set(model, q_samples, np.asarray(spec["home_seed"], dtype=np.float64), rng)
        for strategy, values in seed_sets.items():
            seed_stem = f"{args.robot}_seed{args.seed}_{strategy}_N{N}"
            (seeds_dir / f"{seed_stem}.bin").write_bytes(values.astype(np.float64).tobytes())
            (seeds_dir / f"{seed_stem}.json").write_text(
                json.dumps(
                    {
                        "robot": args.robot,
                        "seed": args.seed,
                        "strategy": strategy,
                        "N": N,
                        "values": values.tolist(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        if args.robot == "ur10" and N == 1000:
            (targets_dir / "targets_273.bin").write_bytes(transforms.reshape(-1).astype(np.float64).tobytes())
            (seeds_dir / "seeds_273.bin").write_bytes(seed_sets["zero_seed"].astype(np.float64).tobytes())

        log_lines.extend(
            [
                f"## N = {N}",
                "",
                f"- Targets JSON: `{json_path}`",
                f"- Targets CSV: `{csv_path}`",
                f"- Targets BIN: `{bin_path}`",
                f"- Seed strategies: `{', '.join(seed_sets.keys())}`",
                f"- Position envelope xyz min (m): `{np.round(xyz.min(axis=0), 4).tolist()}`",
                f"- Position envelope xyz max (m): `{np.round(xyz.max(axis=0), 4).tolist()}`",
                f"- XY radius min/max/mean (m): `{round(float(radius_xy.min()), 4)}` / `{round(float(radius_xy.max()), 4)}` / `{round(float(radius_xy.mean()), 4)}`",
                f"- Joint std (rad): `{np.round(joint_std, 4).tolist()}`",
                "",
            ]
        )

    report_path = root.parent / "docs" / "logs" / f"target_generation_seed{args.seed}.md"
    report_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
