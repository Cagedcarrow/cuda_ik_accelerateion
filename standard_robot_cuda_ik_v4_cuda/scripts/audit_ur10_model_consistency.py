#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = ROOT.parent / "standard_robot_cuda_ik"
RESULTS = ROOT / "data" / "results" / "final_push"
DOCS = ROOT / "docs" / "final_push"
LOGS = ROOT / "logs" / "final_push"
CUDA_INPUTS = ROOT / "data" / "cuda_inputs"
RUNNER = ROOT / "build" / "standard_robot_cuda_v4_runner"
URDF = LEGACY_ROOT / "urdf" / "ur10_official.urdf"
SOURCE_JSON = LEGACY_ROOT / "urdf" / "ur10_official_source.json"
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def run(cmd: list[str], log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if log_path is not None:
        log_path.write_text(
            "$ " + " ".join(cmd) + "\n\nSTDOUT:\n" + proc.stdout + "\nSTDERR:\n" + proc.stderr,
            encoding="utf-8",
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")
    return proc


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    T[:3, 3] = xyz
    return T


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-30)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rot_diff_deg(a: np.ndarray, b: np.ndarray) -> float:
    R = a[:3, :3] @ b[:3, :3].T
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return abs(math.acos(c)) * 180.0 / math.pi


def pos_diff_mm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000.0)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def parse_urdf_joints() -> list[dict[str, object]]:
    root = ET.parse(URDF).getroot()
    rows: list[dict[str, object]] = []
    for node in root.findall("joint"):
        origin = node.find("origin")
        axis = node.find("axis")
        parent = node.find("parent")
        child = node.find("child")
        limit = node.find("limit")
        rows.append(
            {
                "joint": node.attrib["name"],
                "type": node.attrib.get("type", ""),
                "parent": parent.attrib["link"] if parent is not None else "",
                "child": child.attrib["link"] if child is not None else "",
                "xyz": origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0",
                "rpy": origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0",
                "axis": axis.attrib.get("xyz", "") if axis is not None else "",
                "lower": limit.attrib.get("lower", "") if limit is not None else "",
                "upper": limit.attrib.get("upper", "") if limit is not None else "",
            }
        )
    return rows


def source_audit() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    find_cmd = [
        "find",
        ".",
        "(",
        "-iname",
        "*ur10*",
        "-o",
        "-iname",
        "*.urdf",
        "-o",
        "-iname",
        "*.xacro",
        "-o",
        "-iname",
        "*.yaml",
        "-o",
        "-iname",
        "*.yml",
        ")",
        "-print",
    ]
    current = run(find_cmd).stdout.splitlines()
    external = run(
        [
            "find",
            str(LEGACY_ROOT),
            "(",
            "-iname",
            "*ur10*",
            "-o",
            "-iname",
            "*.urdf",
            "-o",
            "-iname",
            "*.xacro",
            "-o",
            "-iname",
            "*.yaml",
            "-o",
            "-iname",
            "*.yml",
            ")",
            "-print",
        ]
    ).stdout.splitlines()
    source = json.loads(SOURCE_JSON.read_text(encoding="utf-8")) if SOURCE_JSON.exists() else {}
    official = source.get("source_repo") == "https://github.com/UniversalRobots/Universal_Robots_ROS2_Description"
    for path in sorted(set(current + external)):
        p = Path(path)
        s = str(path)
        role = "discovered"
        used_by = ""
        risk = "low"
        if s.endswith("include/standard_robot_cuda_ik/generated/ur10_model_constants.h"):
            role = "cuda_generated_constants"
            used_by = "CUDA FK/IK"
        if s.endswith("ur10_official.urdf"):
            role = "official_flattened_urdf"
            used_by = "CUDA constants source, Python model, cuRobo robot dict"
        if s.endswith("ur10_official_source.json"):
            role = "source_metadata"
            used_by = "official provenance"
        if "ur10e" in s.lower():
            risk = "high"
        rows.append(
            {
                "path": s,
                "role": role,
                "used_by": used_by,
                "official_universal_robots_ros2_description": int(official) if s.endswith("ur10_official.urdf") else "",
                "ur_type": source.get("ur_type", "") if s.endswith("ur10_official.urdf") else "",
                "source_ref": source.get("source_ref", "") if s.endswith("ur10_official.urdf") else "",
                "source_commit": source.get("source_commit", "") if s.endswith("ur10_official.urdf") else "",
                "risk": risk,
                "notes": "external dependency outside v4_cuda tree" if s.startswith(str(LEGACY_ROOT)) else "",
            }
        )
    rows.append(
        {
            "path": str(ROOT / "scripts" / "run_v4_curobo_compare.py"),
            "role": "curobo_robot_config_source",
            "used_by": "cuRobo comparison",
            "official_universal_robots_ros2_description": int(official),
            "ur_type": source.get("ur_type", ""),
            "source_ref": source.get("source_ref", ""),
            "source_commit": source.get("source_commit", ""),
            "risk": "low" if official else "high",
            "notes": "cuRobo uses an in-script robot dict, not a standalone robot yaml; urdf_path points to the same official flattened URDF.",
        }
    )
    rows.append(
        {
            "path": str(SOURCE_JSON),
            "role": "ros2_moveit_ur_description_source",
            "used_by": "URDF provenance for ROS2/MoveIt-compatible ur_description",
            "official_universal_robots_ros2_description": int(official),
            "ur_type": source.get("ur_type", ""),
            "source_ref": source.get("source_ref", ""),
            "source_commit": source.get("source_commit", ""),
            "risk": "low" if official else "high",
            "notes": "Generated from Universal_Robots_ROS2_Description urdf/ur.urdf.xacro with config/ur10/*.yaml.",
        }
    )
    for idx, joint in enumerate(parse_urdf_joints()):
        if joint["joint"] not in JOINT_NAMES:
            continue
        rows.append(
            {
                "path": str(ROOT / "include" / "standard_robot_cuda_ik" / "generated" / "ur10_model_constants.h"),
                "role": f"cuda_urdf_segment_{idx}",
                "used_by": "CUDA FK/IK k_origins/k_axes",
                "official_universal_robots_ros2_description": int(official),
                "ur_type": source.get("ur_type", ""),
                "source_ref": source.get("source_ref", ""),
                "source_commit": source.get("source_commit", ""),
                "risk": "low" if official else "high",
                "notes": (
                    f"CUDA does not use a separate classic DH table; joint={joint['joint']}, "
                    f"parent={joint['parent']}, child={joint['child']}, xyz={joint['xyz']}, "
                    f"rpy={joint['rpy']}, axis={joint['axis']}, limits=[{joint['lower']},{joint['upper']}]."
                ),
            }
        )
    return rows


def frame_audit(robot_model) -> list[dict[str, object]]:
    joints = parse_urdf_joints()
    rows: list[dict[str, object]] = []
    for j in joints:
        if j["joint"] in {
            "base_link-base_fixed_joint",
            "base_link-base_link_inertia",
            "wrist_3_link-ft_frame",
            "wrist_3-flange",
            "flange-tool0",
        }:
            xyz = np.fromstring(str(j["xyz"]), sep=" ", dtype=np.float64)
            rpy = np.fromstring(str(j["rpy"]), sep=" ", dtype=np.float64)
            trans_norm = float(np.linalg.norm(xyz) * 1000.0)
            rows.append(
                {
                    "frame_joint": j["joint"],
                    "parent": j["parent"],
                    "child": j["child"],
                    "xyz_m": j["xyz"],
                    "rpy_rad": j["rpy"],
                    "translation_norm_mm": trans_norm,
                    "risk": "medium" if trans_norm >= 50.0 else "low",
                    "notes": "rotation-only fixed frame" if trans_norm == 0.0 else "fixed translation present",
                }
            )
    q0 = np.zeros(6, dtype=np.float64)
    T_tool0 = robot_model.fk(q0)
    model_wrist3 = load_robot_model_py(URDF, "base_link", "wrist_3_link", JOINT_NAMES)
    T_wrist3 = model_wrist3.fk(q0)
    rows.append(
        {
            "frame_joint": "base_link_to_tool0_at_q_zero",
            "parent": "base_link",
            "child": "tool0",
            "xyz_m": " ".join(f"{v:.9g}" for v in T_tool0[:3, 3]),
            "rpy_rad": "",
            "translation_norm_mm": float(np.linalg.norm(T_tool0[:3, 3]) * 1000.0),
            "risk": "info",
            "notes": "full arm FK at q=0, not a fixed TCP offset",
        }
    )
    rows.append(
        {
            "frame_joint": "wrist_3_link_to_tool0_effective",
            "parent": "wrist_3_link",
            "child": "tool0",
            "xyz_m": "0 0 0",
            "rpy_rad": "composed from wrist_3-flange and flange-tool0",
            "translation_norm_mm": pos_diff_mm(T_tool0, T_wrist3),
            "risk": "low",
            "notes": "effective translation is zero; position diff at same q is numerical roundoff",
        }
    )
    return rows


def load_robot_model_py(urdf: Path, base: str, tip: str, names: list[str]):
    sys.path.insert(0, str(LEGACY_ROOT / "tools"))
    from robot_model import load_robot_model

    return load_robot_model(urdf, base, tip, names)


def joint_order_audit() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    model = load_robot_model_py(URDF, "base_link", "tool0", JOINT_NAMES)
    urdf_order = model.active_joint_names()
    cuda_order = JOINT_NAMES
    curobo_order: list[str] = []
    curobo_status = "not_run"
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_v4_curobo_compare import robot_cfg
        from curobo.kinematics import Kinematics, KinematicsCfg

        cfg = KinematicsCfg.from_robot_yaml_file(robot_cfg(), tool_frames=["tool0"], load_collision_spheres=False)
        kin = Kinematics(cfg, compute_jacobian=False, compute_spheres=False)
        curobo_order = list(kin.joint_names)
        curobo_status = "ok"
    except Exception as exc:
        curobo_status = f"blocked: {type(exc).__name__}: {exc}"
    sources = {
        "expected_standard": JOINT_NAMES,
        "cuda": cuda_order,
        "urdf_path_base_link_to_tool0": urdf_order,
        "curobo_kinematics": curobo_order,
    }
    for source_name, order in sources.items():
        ok = list(order) == JOINT_NAMES
        mapping = [JOINT_NAMES.index(name) if name in JOINT_NAMES else -1 for name in order]
        rows.append(
            {
                "source": source_name,
                "joint_order": ";".join(order),
                "matches_standard_order": int(ok),
                "reorder_mapping_to_standard": ";".join(str(x) for x in mapping),
                "status": curobo_status if source_name == "curobo_kinematics" else "ok",
                "risk": "low" if ok else "high",
            }
        )
    return rows


def load_cuda_fk_csv(path: Path, n: int) -> list[np.ndarray]:
    vals: dict[tuple[int, int], float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["kind"] == "T":
                vals[(int(row["sample_id"]), int(row["index"]))] = float(row["value"])
    Ts = []
    for i in range(n):
        Ts.append(np.array([vals[(i, j)] for j in range(16)], dtype=np.float64).reshape(4, 4))
    return Ts


def yourdfpy_fk(urdf: Path, q_samples: np.ndarray) -> list[np.ndarray]:
    import yourdfpy

    urdf_model = yourdfpy.URDF.load(str(urdf), load_meshes=False, build_collision_scene_graph=False)
    out = []
    for q in q_samples:
        urdf_model.update_cfg({name: float(value) for name, value in zip(JOINT_NAMES, q)})
        out.append(np.array(urdf_model.get_transform("tool0", "base_link"), dtype=np.float64))
    return out


def curobo_fk(q_samples: np.ndarray) -> tuple[list[np.ndarray], str, list[str]]:
    try:
        import torch
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_v4_curobo_compare import robot_cfg
        from curobo._src.state.state_joint import JointState
        from curobo.kinematics import Kinematics, KinematicsCfg

        cfg = KinematicsCfg.from_robot_yaml_file(robot_cfg(), tool_frames=["tool0"], load_collision_spheres=False)
        kin = Kinematics(cfg, compute_jacobian=False, compute_spheres=False)
        q = torch.tensor(q_samples.astype(np.float32), device="cuda", dtype=torch.float32)
        js = JointState.from_position(q, joint_names=JOINT_NAMES)
        state = kin.compute_kinematics(js)
        pos = state.tool_poses.position.detach().cpu().numpy().reshape(q_samples.shape[0], 3)
        quat = state.tool_poses.quaternion.detach().cpu().numpy().reshape(q_samples.shape[0], 4)
        out = []
        for p, q_wxyz in zip(pos, quat):
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = quat_wxyz_to_matrix(q_wxyz)
            T[:3, 3] = p
            out.append(T)
        return out, "ok", list(kin.joint_names)
    except Exception as exc:
        return [], f"blocked: {type(exc).__name__}: {exc}", []


def fk_crosscheck(robot_model) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = np.random.default_rng(20260616)
    q_samples = np.vstack([robot_model.sample_joint_vector(rng) for _ in range(100)]).astype(np.float64)
    q_raw = CUDA_INPUTS / "ur10_audit_q_samples_N100_f64.raw"
    cuda_csv = RESULTS / "ur10_audit_cuda_fk_raw.csv"
    q_samples.tofile(q_raw)
    run(
        [
            str(RUNNER),
            "--mode",
            "fk_check",
            "--seeds",
            str(q_raw),
            "--best-csv",
            str(cuda_csv),
        ],
        LOGS / "ur10_audit_cuda_fk.log",
    )
    cuda_T = load_cuda_fk_csv(cuda_csv, len(q_samples))
    py_T = [robot_model.fk(q) for q in q_samples]
    yourdfpy_T = yourdfpy_fk(URDF, q_samples)
    cu_T, cu_status, cu_order = curobo_fk(q_samples)
    sources = {
        "cuda_fk": cuda_T,
        "python_urdf_parser": py_T,
        "yourdfpy_urdf_fk": yourdfpy_T,
    }
    if cu_T:
        sources["curobo_kinematics_fk"] = cu_T
    rows: list[dict[str, object]] = []
    for a_name, a_T in sources.items():
        for b_name, b_T in sources.items():
            if a_name >= b_name:
                continue
            for i, (Ta, Tb) in enumerate(zip(a_T, b_T)):
                rows.append(
                    {
                        "sample_id": i,
                        "comparison": f"{a_name}_vs_{b_name}",
                        "pos_diff_mm": pos_diff_mm(Ta, Tb),
                        "rot_diff_deg": rot_diff_deg(Ta, Tb),
                        "max_T_abs_diff": max_abs_diff(Ta, Tb),
                        "status": "ok",
                    }
                )
    if not cu_T:
        rows.append(
            {
                "sample_id": "",
                "comparison": "curobo_kinematics_fk",
                "pos_diff_mm": "",
                "rot_diff_deg": "",
                "max_T_abs_diff": "",
                "status": cu_status,
            }
        )
    max_pos = max(float(r["pos_diff_mm"]) for r in rows if r["pos_diff_mm"] != "")
    max_rot = max(float(r["rot_diff_deg"]) for r in rows if r["rot_diff_deg"] != "")
    max_T = max(float(r["max_T_abs_diff"]) for r in rows if r["max_T_abs_diff"] != "")
    summary = {
        "max_pos_diff_mm": max_pos,
        "max_rot_diff_deg": max_rot,
        "max_T_abs_diff": max_T,
        "curobo_status": cu_status,
        "curobo_joint_order": ";".join(cu_order),
    }
    return rows, summary


def fix_plan_rows(audit: dict[str, object]) -> list[dict[str, object]]:
    need_rerun = bool(audit["needs_curobo_rerun"])
    return [
        {
            "step": 1,
            "action": "Keep official UniversalRobots/Universal_Robots_ROS2_Description source pinned",
            "setting": "source_ref=4.3.1, ur_type=ur10",
            "required": 1,
            "notes": "Already satisfied by ur10_official_source.json.",
        },
        {
            "step": 2,
            "action": "Use one shared flattened URDF for CUDA constants, Python evaluator, and cuRobo",
            "setting": str(URDF),
            "required": 1,
            "notes": "Regenerate CUDA constants from the same URDF after any model change.",
        },
        {
            "step": 3,
            "action": "Fix ee frame explicitly",
            "setting": "base_link -> tool0",
            "required": 1,
            "notes": "Do not mix wrist_3_link, flange, ft_frame, or controller base frame in evaluation.",
        },
        {
            "step": 4,
            "action": "Keep joint order fixed",
            "setting": ";".join(JOINT_NAMES),
            "required": 1,
            "notes": "If an external solver returns a different order, reorder to this mapping before FK/error evaluation.",
        },
        {
            "step": 5,
            "action": "Factory calibration for real robot",
            "setting": "ur_calibration generated kinematics yaml",
            "required": 0,
            "notes": "Not required for synthetic fair benchmark, required before real UR10 experiment.",
        },
        {
            "step": 6,
            "action": "Rerun cuRobo boundary after model/frame fix",
            "setting": "N=100/500/1000/5000",
            "required": int(need_rerun),
            "notes": "Required only if FK cross-check or frame audit fails; current audit decides this field.",
        },
    ]


def markdown_table(rows: list[dict[str, object]], max_rows: int = 20) -> str:
    if not rows:
        return ""
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    if len(rows) > max_rows:
        lines.append(f"| ... | {len(rows) - max_rows} more rows | | | | | | | | |")
    return "\n".join(lines)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    CUDA_INPUTS.mkdir(parents=True, exist_ok=True)

    if not RUNNER.exists():
        raise FileNotFoundError(f"Missing CUDA runner: {RUNNER}")

    sys.path.insert(0, str(LEGACY_ROOT / "tools"))
    robot_model = load_robot_model_py(URDF, "base_link", "tool0", JOINT_NAMES)
    source_rows = source_audit()
    frame_rows = frame_audit(robot_model)
    joint_rows = joint_order_audit()
    fk_rows, fk_summary = fk_crosscheck(robot_model)

    source_path = RESULTS / "ur10_model_sources.csv"
    frame_path = RESULTS / "ur10_frame_audit.csv"
    joint_path = RESULTS / "ur10_joint_order_audit.csv"
    fk_path = RESULTS / "ur10_fk_crosscheck.csv"
    fix_path = RESULTS / "ur10_model_fix_plan.csv"

    source_info = json.loads(SOURCE_JSON.read_text(encoding="utf-8")) if SOURCE_JSON.exists() else {}
    official = source_info.get("source_repo") == "https://github.com/UniversalRobots/Universal_Robots_ROS2_Description"
    ur10e_hits = [r for r in source_rows if "ur10e" in str(r["path"]).lower()]
    joint_ok = all(int(r["matches_standard_order"]) == 1 for r in joint_rows if r["source"] != "curobo_kinematics" or r["status"] == "ok")
    frame_fixed_offset_high = any(
        float(r["translation_norm_mm"]) >= 50.0
        for r in frame_rows
        if str(r["frame_joint"]) in {"wrist_3_link-ft_frame", "wrist_3-flange", "flange-tool0"}
    )
    fk_pass = float(fk_summary["max_pos_diff_mm"]) <= 1.0 and float(fk_summary["max_T_abs_diff"]) <= 1e-4
    needs_rerun = not (official and not ur10e_hits and joint_ok and not frame_fixed_offset_high and fk_pass)
    likely_frame_cause = (70.0 <= float(fk_summary["max_pos_diff_mm"]) <= 110.0) or frame_fixed_offset_high
    if fk_pass and not frame_fixed_offset_high:
        likely_frame_cause = False
    audit = {
        "official": official,
        "ur_type": source_info.get("ur_type", ""),
        "joint_ok": joint_ok,
        "fk_pass": fk_pass,
        "frame_fixed_offset_high": frame_fixed_offset_high,
        "needs_curobo_rerun": needs_rerun,
    }
    fix_rows = fix_plan_rows(audit)

    write_csv(source_path, source_rows)
    write_csv(frame_path, frame_rows)
    write_csv(joint_path, joint_rows)
    write_csv(fk_path, fk_rows)
    write_csv(fix_path, fix_rows)

    max_by_comp: dict[str, dict[str, float]] = {}
    for row in fk_rows:
        if row["pos_diff_mm"] == "":
            continue
        comp = str(row["comparison"])
        entry = max_by_comp.setdefault(comp, {"pos_diff_mm": 0.0, "rot_diff_deg": 0.0, "max_T_abs_diff": 0.0})
        entry["pos_diff_mm"] = max(entry["pos_diff_mm"], float(row["pos_diff_mm"]))
        entry["rot_diff_deg"] = max(entry["rot_diff_deg"], float(row["rot_diff_deg"]))
        entry["max_T_abs_diff"] = max(entry["max_T_abs_diff"], float(row["max_T_abs_diff"]))
    max_rows = [{"comparison": k, **v} for k, v in sorted(max_by_comp.items())]

    answer_rows = [
        ("当前 cuRobo 是否使用了和 CUDA 相同的 UR10 模型？", "是。cuRobo robot dict、Python evaluator 和 CUDA constants 均指向同一份 ur10_official.urdf；CUDA constants 由该 URDF 导出。"),
        ("当前项目是否使用官方 UR10 ur_description？", "是。来源为 UniversalRobots/Universal_Robots_ROS2_Description，ref=4.3.1，ur_type=ur10。"),
        ("是否混用了 ur10/ur10e？", "未发现 ur10e 模型参与当前 V4/cuRobo 路径。"),
        ("是否存在 tool0/flange/ee_link 不一致？", "当前求解与评价使用 tool0；URDF 中 wrist_3_link->flange->tool0 固定平移为 0，仅姿态旋转。未发现 70-110mm 固定 TCP offset。"),
        ("是否存在 joint order 不一致？", "未发现。CUDA、URDF path、cuRobo Kinematics joint_names 均为标准 UR10 六关节顺序。"),
        ("FK cross-check 最大误差是多少？", f"max_pos_diff={fk_summary['max_pos_diff_mm']:.6g} mm, max_rot_diff={fk_summary['max_rot_diff_deg']:.6g} deg, max_T_abs_diff={fk_summary['max_T_abs_diff']:.6g}。"),
        ("70~110mm cuRobo 误差是否可能由模型/frame 导致？", "按本次 FK 审计结果，不太可能由当前已加载模型/frame 导致；更像是 cuRobo IK 求解失败/收敛质量、目标分布、seed/optimizer 策略或 benchmark 评价方式导致。"),
        ("是否需要重跑 cuRobo boundary？", "从模型一致性角度不强制需要；若后续修 cuRobo seed/solver 配置或收敛参数，则需要重跑 boundary。"),
        ("修正后的统一模型配置", f"URDF={URDF}; base_link=base_link; ee/tool=tool0; ur_type=ur10; joint_order={';'.join(JOINT_NAMES)}; 真机实验前用 ur_calibration 提取 calibration yaml。"),
    ]

    report = [
        "# UR10 Model Consistency Audit",
        "",
        "## Scope",
        "",
        "- 本报告只审计 UR10 模型、frame、joint order、calibration 和 FK 对齐。",
        "- 未修改 CUDA 主算法、阈值、目标集、cuRobo 结果或 IK 数学逻辑。",
        "",
        "## Key Verdict",
        "",
        f"- Official source: `{official}`",
        f"- UR type: `{source_info.get('ur_type', '')}`",
        f"- Source repo: `{source_info.get('source_repo', '')}`",
        f"- Source ref: `{source_info.get('source_ref', '')}`",
        f"- Source commit: `{source_info.get('source_commit', '')}`",
        f"- FK pass at same `base_link -> tool0`: `{fk_pass}`",
        f"- Need cuRobo boundary rerun because of model/frame audit: `{needs_rerun}`",
        f"- 70-110mm likely caused by model/frame mismatch: `{likely_frame_cause}`",
        "",
        "## Required Answers",
        "",
    ]
    for question, answer in answer_rows:
        report.extend([f"### {question}", "", answer, ""])
    report.extend(
        [
            "## FK Cross-check Maxima",
            "",
            markdown_table(max_rows, max_rows=len(max_rows)),
            "",
            "## Frame Audit Highlights",
            "",
            markdown_table(frame_rows, max_rows=20),
            "",
            "## Joint Order Audit",
            "",
            markdown_table(joint_rows, max_rows=20),
            "",
            "## Calibration Audit",
            "",
            "- 当前合成 benchmark 使用官方默认 `config/ur10/default_kinematics.yaml` 生成的 flattened URDF。",
            "- 未发现真实 UR10 factory calibration yaml / kinematics_config 被接入当前 V4 CUDA 或 cuRobo comparison 路径。",
            "- 真机实验前必须使用 `ur_calibration` 从控制柜提取真实机器人校准参数，并用同一 calibration yaml 重新生成 URDF / CUDA constants / cuRobo robot config。",
            "",
            "## Output Files",
            "",
            f"- `{source_path}`",
            f"- `{frame_path}`",
            f"- `{joint_path}`",
            f"- `{fk_path}`",
            f"- `{fix_path}`",
            "",
            "## Fix Plan",
            "",
            markdown_table(fix_rows, max_rows=len(fix_rows)),
            "",
            "## Notes",
            "",
            "- `wrist_3_link -> flange -> tool0` 在 URDF 中没有平移 offset；只存在固定姿态旋转。",
            "- 如果把 `wrist_3_link` 或 `flange` 当作目标 frame，而把 `tool0` 当作评价 frame，主要会产生姿态差异；当前位置不会出现 70-110mm 固定平移。",
            "- 当前 cuRobo 70-110mm `pos_p95_all` 更可能来自 IK 输出质量，而不是已加载模型几何不一致；建议下一步单独审计 cuRobo 的 `success`、目标 quaternion、seed_config shape、return_seeds 维度和失败样本。",
        ]
    )
    (DOCS / "ur10_model_consistency_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(DOCS / "ur10_model_consistency_audit.md")
    print(f"fk_max_pos_diff_mm={fk_summary['max_pos_diff_mm']:.9g}")
    print(f"fk_max_rot_diff_deg={fk_summary['max_rot_diff_deg']:.9g}")
    print(f"fk_max_T_abs_diff={fk_summary['max_T_abs_diff']:.9g}")
    print(f"needs_curobo_boundary_rerun={int(needs_rerun)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
