#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


OFFICIAL_REPO = "https://github.com/UniversalRobots/Universal_Robots_ROS2_Description"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--repo-url", default=OFFICIAL_REPO)
    parser.add_argument(
        "--ref",
        default="4.3.1",
        help="Official repo branch or tag to fetch. Default pins the latest verified release tag.",
    )
    return parser.parse_args()


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, check=True, text=True, capture_output=True)


def patch_find_substitutions(repo_root: Path) -> None:
    for path in repo_root.rglob("*.xacro"):
        text = path.read_text(encoding="utf-8")
        patched = text.replace("$(find ur_description)", str(repo_root))
        if patched != text:
            path.write_text(patched, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root
    target_urdf = project_root / "urdf" / "ur10_official.urdf"
    metadata_path = project_root / "urdf" / "ur10_official_source.json"

    xacro_bin = shutil.which("xacro") or str(Path.home() / ".local" / "bin" / "xacro")
    if not Path(xacro_bin).exists():
        raise FileNotFoundError("xacro not found. Install it first, e.g. `python3 -m pip install --user xacro`.")

    with tempfile.TemporaryDirectory(prefix="ur10_official_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        repo_root = tmp_root / "Universal_Robots_ROS2_Description"
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                args.ref,
                "--single-branch",
                args.repo_url,
                str(repo_root),
            ]
        )
        commit = run(["git", "rev-parse", "HEAD"], cwd=repo_root).stdout.strip()
        patch_find_substitutions(repo_root)

        cmd = [
            xacro_bin,
            str(repo_root / "urdf" / "ur.urdf.xacro"),
            "ur_type:=ur10",
            f"joint_limit_params:={repo_root / 'config' / 'ur10' / 'joint_limits.yaml'}",
            f"kinematics_params:={repo_root / 'config' / 'ur10' / 'default_kinematics.yaml'}",
            f"physical_params:={repo_root / 'config' / 'ur10' / 'physical_parameters.yaml'}",
            f"visual_params:={repo_root / 'config' / 'ur10' / 'visual_parameters.yaml'}",
            "name:=ur10",
        ]
        urdf_text = run(cmd).stdout
        target_urdf.write_text(urdf_text, encoding="utf-8")
        metadata = {
            "source_repo": args.repo_url,
            "source_ref": args.ref,
            "source_commit": commit,
            "generated_from": "urdf/ur.urdf.xacro",
            "ur_type": "ur10",
            "joint_limit_params": "config/ur10/joint_limits.yaml",
            "kinematics_params": "config/ur10/default_kinematics.yaml",
            "physical_params": "config/ur10/physical_parameters.yaml",
            "visual_params": "config/ur10/visual_parameters.yaml",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {target_urdf}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
