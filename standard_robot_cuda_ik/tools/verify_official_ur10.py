#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yourdfpy

from robot_model import file_md5, load_robot_model


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    urdf_path = project_root / "urdf" / "ur10_official.urdf"
    source_path = project_root / "urdf" / "ur10_official_source.json"
    source_info = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {}
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
    urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=False, build_collision_scene_graph=False)
    rng = np.random.default_rng(42)
    max_fk_err = 0.0
    sample_rows = []
    for idx in range(20):
        q = model.sample_joint_vector(rng)
        ours = model.fk(q)
        cfg = {name: float(v) for name, v in zip(model.active_joint_names(), q)}
        urdf.update_cfg(cfg)
        theirs = urdf.get_transform("tool0", "base_link")
        err = float(np.max(np.abs(ours - theirs)))
        max_fk_err = max(max_fk_err, err)
        sample_rows.append((idx, err))

    lines = [
        "# Official UR10 Model Verification",
        "",
        f"- URDF path: `{urdf_path}`",
        f"- URDF MD5: `{file_md5(urdf_path)}`",
        f"- Source repo: `{source_info.get('source_repo', 'unknown')}`",
        f"- Source ref: `{source_info.get('source_ref', 'unknown')}`",
        f"- Source commit: `{source_info.get('source_commit', 'unknown')}`",
        f"- Base link: `{model.base_link}`",
        f"- TCP link: `{model.tip_link}`",
        f"- Joint count: `{model.dof}`",
        f"- Active joints: `{', '.join(model.active_joint_names())}`",
        f"- CPU FK vs yourdfpy max abs error: `{max_fk_err:.3e}`",
        "",
        "## Joint Table",
        "",
    ]
    for row in model.joint_table():
        lines.append(
            f"- `{row['name']}`: parent=`{row['parent']}`, child=`{row['child']}`, "
            f"axis={row['axis']}, xyz={row['origin_xyz']}, rpy={row['origin_rpy']}, "
            f"limits=[{row['lower']}, {row['upper']}]"
        )
    lines.extend(
        [
            "",
            "## Solver Consistency Notes",
            "",
            "- `cuda`: 使用本项目 `urdf/ur10_official.urdf` 导出的常量表。",
            "- `pyroki`: 直接加载同一份 URDF，TCP 固定为 `tool0`。",
            "- `kdl`: 需要使用同一 active joint 顺序与 `tool0` 末端定义。",
            "- `curobo`: 当前 benchmark 使用自定义 robot dict 强制指向同一份 URDF，并通过 `current_state + seed_config` 接入共享外部 seed。",
            "- `hjcd_ik`: 当前默认不进入主公平对比表，除非确认模型链与 TCP 一致。",
            "",
            "## FK Spot Check",
            "",
        ]
    )
    for idx, err in sample_rows:
        lines.append(f"- sample `{idx}` max abs diff: `{err:.3e}`")

    out_path = project_root.parent / "docs" / "logs" / "official_ur10_model_verification.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "urdf_path": str(urdf_path),
        "urdf_md5": file_md5(urdf_path),
        "source_repo": source_info.get("source_repo"),
        "source_ref": source_info.get("source_ref"),
        "source_commit": source_info.get("source_commit"),
        "joint_count": model.dof,
        "active_joints": model.active_joint_names(),
        "tool_offset_row_major": model.tool_offset_from_last_joint().reshape(-1).tolist(),
        "fk_max_abs_error_vs_yourdfpy": max_fk_err,
    }
    (project_root / "docs" / "logs" / "official_ur10_model_verification.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
