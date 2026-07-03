#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def transform_from_xyz_rpy(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rpy_matrix(rpy[0], rpy[1], rpy[2])
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def rotation_about_axis(axis: Sequence[float], angle: float) -> np.ndarray:
    axis_arr = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis_arr)
    if norm == 0.0:
        return np.eye(4, dtype=np.float64)
    ax, ay, az = axis_arr / norm
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    R = np.array(
        [
            [t * ax * ax + c, t * ax * ay - s * az, t * ax * az + s * ay],
            [t * ax * ay + s * az, t * ay * ay + c, t * ay * az - s * ax],
            [t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def quaternion_from_matrix(R: np.ndarray) -> np.ndarray:
    tr = float(np.trace(R))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


@dataclass
class JointInfo:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None
    origin_matrix_override: np.ndarray | None = None

    @property
    def origin_matrix(self) -> np.ndarray:
        if self.origin_matrix_override is not None:
            return self.origin_matrix_override
        return transform_from_xyz_rpy(self.origin_xyz, self.origin_rpy)


@dataclass
class RobotModel:
    name: str
    urdf_path: Path
    base_link: str
    tip_link: str
    active_joints: List[JointInfo]
    fixed_tail_joints: List[JointInfo]

    @property
    def dof(self) -> int:
        return len(self.active_joints)

    def fk(self, q: Sequence[float]) -> np.ndarray:
        q_arr = np.asarray(q, dtype=np.float64)
        if q_arr.shape[0] != self.dof:
            raise ValueError(f"Expected {self.dof} joints, got {q_arr.shape[0]}")
        T = np.eye(4, dtype=np.float64)
        for i, joint in enumerate(self.active_joints):
            T = T @ joint.origin_matrix @ rotation_about_axis(joint.axis, float(q_arr[i]))
        for joint in self.fixed_tail_joints:
            T = T @ joint.origin_matrix
        return T

    def sample_joint_vector(self, rng: np.random.Generator) -> np.ndarray:
        q = np.zeros(self.dof, dtype=np.float64)
        for i, joint in enumerate(self.active_joints):
            lo = -math.pi if joint.lower is None else joint.lower
            hi = math.pi if joint.upper is None else joint.upper
            q[i] = rng.uniform(lo, hi)
        return q

    def limits_array(self) -> np.ndarray:
        values: List[float] = []
        for joint in self.active_joints:
            values.append(-math.pi if joint.lower is None else joint.lower)
            values.append(math.pi if joint.upper is None else joint.upper)
        return np.asarray(values, dtype=np.float64)

    def origins_array(self) -> np.ndarray:
        mats = [joint.origin_matrix.reshape(-1) for joint in self.active_joints]
        return np.concatenate(mats).astype(np.float64)

    def axes_array(self) -> np.ndarray:
        return np.concatenate([joint.axis for joint in self.active_joints]).astype(np.float64)

    def tool_offset_from_last_joint(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        for joint in self.fixed_tail_joints:
            T = T @ joint.origin_matrix
        return T

    def active_joint_names(self) -> List[str]:
        return [joint.name for joint in self.active_joints]

    def joint_table(self) -> List[Dict[str, object]]:
        rows = []
        for joint in self.active_joints:
            rows.append(
                {
                    "name": joint.name,
                    "parent": joint.parent,
                    "child": joint.child,
                    "axis": [float(x) for x in joint.axis],
                    "origin_xyz": [float(x) for x in joint.origin_xyz],
                    "origin_rpy": [float(x) for x in joint.origin_rpy],
                    "lower": joint.lower,
                    "upper": joint.upper,
                }
            )
        return rows


def load_robot_model(
    urdf_path: Path,
    base_link: str,
    tip_link: str,
    expected_active_joint_names: Sequence[str] | None = None,
) -> RobotModel:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    child_to_joint: Dict[str, JointInfo] = {}
    for joint_node in root.findall("joint"):
        origin_node = joint_node.find("origin")
        axis_node = joint_node.find("axis")
        limit_node = joint_node.find("limit")
        xyz = np.fromstring(origin_node.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float64) if origin_node is not None else np.zeros(3)
        rpy = np.fromstring(origin_node.attrib.get("rpy", "0 0 0"), sep=" ", dtype=np.float64) if origin_node is not None else np.zeros(3)
        axis = np.fromstring(axis_node.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float64) if axis_node is not None else np.zeros(3)
        lower = float(limit_node.attrib["lower"]) if limit_node is not None and "lower" in limit_node.attrib else None
        upper = float(limit_node.attrib["upper"]) if limit_node is not None and "upper" in limit_node.attrib else None
        joint = JointInfo(
            name=joint_node.attrib["name"],
            joint_type=joint_node.attrib["type"],
            parent=joint_node.find("parent").attrib["link"],
            child=joint_node.find("child").attrib["link"],
            origin_xyz=xyz,
            origin_rpy=rpy,
            axis=axis,
            lower=lower,
            upper=upper,
        )
        child_to_joint[joint.child] = joint

    path: List[JointInfo] = []
    current = tip_link
    while current != base_link:
        if current not in child_to_joint:
            raise ValueError(f"No parent joint found for link {current}")
        joint = child_to_joint[current]
        path.append(joint)
        current = joint.parent
    path.reverse()

    active: List[JointInfo] = []
    pending_fixed = np.eye(4, dtype=np.float64)
    for joint in path:
        if joint.joint_type == "fixed":
            pending_fixed = pending_fixed @ joint.origin_matrix
            continue
        origin_override = pending_fixed @ joint.origin_matrix
        active.append(
            JointInfo(
                name=joint.name,
                joint_type=joint.joint_type,
                parent=joint.parent,
                child=joint.child,
                origin_xyz=joint.origin_xyz.copy(),
                origin_rpy=joint.origin_rpy.copy(),
                axis=joint.axis.copy(),
                lower=joint.lower,
                upper=joint.upper,
                origin_matrix_override=origin_override,
            )
        )
        pending_fixed = np.eye(4, dtype=np.float64)

    if expected_active_joint_names is not None:
        names = [joint.name for joint in active]
        if list(expected_active_joint_names) != names:
            raise ValueError(f"Active joint mismatch: expected {expected_active_joint_names}, got {names}")

    fixed_tail: List[JointInfo] = []
    if not np.allclose(pending_fixed, np.eye(4, dtype=np.float64)):
        fixed_tail.append(
            JointInfo(
                name="__tail_fixed__",
                joint_type="fixed",
                parent=active[-1].child if active else base_link,
                child=tip_link,
                origin_xyz=np.zeros(3, dtype=np.float64),
                origin_rpy=np.zeros(3, dtype=np.float64),
                axis=np.zeros(3, dtype=np.float64),
                lower=None,
                upper=None,
                origin_matrix_override=pending_fixed,
            )
        )

    return RobotModel(
        name=root.attrib.get("name", urdf_path.stem),
        urdf_path=urdf_path,
        base_link=base_link,
        tip_link=tip_link,
        active_joints=active,
        fixed_tail_joints=fixed_tail,
    )


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def export_cuda_constants_header(path: Path, model: RobotModel) -> None:
    origins = ",\n    ".join(f"{x:.16e}" for x in model.origins_array())
    axes = ", ".join(f"{x:.16e}" for x in model.axes_array())
    q_index = ", ".join(str(i) for i in range(model.dof))
    tool = ",\n    ".join(f"{x:.16e}" for x in model.tool_offset_from_last_joint().reshape(-1))
    limits = ", ".join(f"{x:.16e}" for x in model.limits_array())
    weights = ", ".join(
        f"{x:.16e}"
        for x in np.array(
            [
                1.0, 1.0, 1.0, 1.00, 1.00, 1.00,
                1.0, 1.0, 1.0, 0.60, 0.60, 0.60,
                1.0, 1.0, 1.0, 0.30, 0.30, 0.30,
                1.0, 1.0, 1.0, 0.15, 0.15, 0.15,
            ],
            dtype=np.float64,
        )
    )
    lambda_params = ", ".join(f"{x:.16e}" for x in np.array([2e-4, 5e-2, 1e-4, 8e-2], dtype=np.float64))
    text = f"""#pragma once
// Auto-generated from {model.urdf_path.name}. Do not hand-edit.

static const double k_origins[96] = {{
    {origins}
}};

static const double k_axes[18] = {{ {axes} }};
static const int k_q_index[6] = {{ {q_index} }};
static const double k_T_wrist3_to_tcp[16] = {{
    {tool}
}};
static const double k_joint_limits[12] = {{ {limits} }};
static const double k_weights[24] = {{ {weights} }};
static const double k_lambda_params[4] = {{ {lambda_params} }};
"""
    path.write_text(text, encoding="utf-8")


def write_targets_csv(path: Path, records: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "target_id",
        "q_sampled",
        "position_xyz",
        "rotation_matrix",
        "quaternion_wxyz",
        "transform_row_major",
        "within_joint_limits",
        "fk_verified",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(row[key]) if isinstance(row[key], (list, dict)) else row[key] for key in fieldnames})
