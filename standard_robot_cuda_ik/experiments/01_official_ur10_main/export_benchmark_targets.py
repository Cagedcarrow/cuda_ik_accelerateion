#!/usr/bin/env python3
"""Export benchmark target reference CSV from existing data.

Reads target pose data and generates benchmark_targets.csv with:
  target_id, N_group, q_gt_1..q_gt_6, target_px,py,pz, target_qx,qy,qz,qw, q_seed_1..q_seed_6

Output: benchmark_targets.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from common import load_robot_records, load_seed_values
from robot_model import quaternion_from_matrix

OUT_DIR = Path(__file__).resolve().parent

N_VALUES = [100, 500, 1000, 5000]
ROBOT = "ur10"
SEED = 42
STRATEGY = "zero_seed"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "benchmark_targets.csv"

    rows = []
    for N in N_VALUES:
        targets, records = load_robot_records(ROBOT, SEED, N)
        seeds = load_seed_values(ROBOT, SEED, N, STRATEGY)
        for i in range(N):
            T = targets[i].reshape(4, 4)
            px, py, pz = T[0, 3], T[1, 3], T[2, 3]
            qx, qy, qz, qw = quaternion_from_matrix(T[:3, :3])
            row = {
                "target_id": i, "N_group": N,
                "target_px": px, "target_py": py, "target_pz": pz,
                "target_qx": qx, "target_qy": qy, "target_qz": qz, "target_qw": qw,
            }
            for j in range(6):
                row[f"q_seed_{j+1}"] = seeds[i, j]
            rows.append(row)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Exported {len(rows)} targets to {out_path}")


if __name__ == "__main__":
    main()
