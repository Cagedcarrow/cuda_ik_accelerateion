#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--robot", default="ur10")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def collect_hashes(project_root: Path, robot: str, seed: int) -> dict[str, str]:
    data_root = project_root / "data"
    hashes: dict[str, str] = {}
    for path in sorted((data_root / "targets").glob(f"{robot}_seed{seed}_N*.*")):
        hashes[str(path.relative_to(project_root))] = md5(path)
    for path in sorted((data_root / "seeds").glob(f"{robot}_seed{seed}_*_*.*")):
        hashes[str(path.relative_to(project_root))] = md5(path)
    header = project_root / "include" / "standard_robot_cuda_ik" / "generated" / f"{robot}_model_constants.h"
    hashes[str(header.relative_to(project_root))] = md5(header)
    return hashes


def run_generation(project_root: Path, robot: str, seed: int) -> None:
    subprocess.run(
        ["python3", str(project_root / "tools" / "generate_standard_assets.py"), "--robot", robot, "--seed", str(seed)],
        cwd=project_root.parent,
        check=True,
        capture_output=True,
        text=True,
    )


def main() -> None:
    args = parse_args()
    project_root = args.project_root
    run_generation(project_root, args.robot, args.seed)
    hashes_a = collect_hashes(project_root, args.robot, args.seed)
    run_generation(project_root, args.robot, args.seed)
    hashes_b = collect_hashes(project_root, args.robot, args.seed)

    stable = hashes_a == hashes_b
    report = {
        "robot": args.robot,
        "seed": args.seed,
        "stable": stable,
        "hashes_first_run": hashes_a,
        "hashes_second_run": hashes_b,
    }
    json_path = project_root / "docs" / "logs" / f"{args.robot}_seed{args.seed}_reproducibility.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Seed 42 Reproducibility Verification",
        "",
        f"- Robot: `{args.robot}`",
        f"- Seed: `{args.seed}`",
        f"- Stable across two consecutive regenerations: `{stable}`",
        "",
        "## Asset Hashes",
        "",
    ]
    for rel_path, digest in hashes_a.items():
        marker = "match" if hashes_b.get(rel_path) == digest else "mismatch"
        md_lines.append(f"- `{rel_path}`: `{digest}` ({marker})")
    md_path = project_root.parent / "docs" / "logs" / f"{args.robot}_seed{args.seed}_reproducibility.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
