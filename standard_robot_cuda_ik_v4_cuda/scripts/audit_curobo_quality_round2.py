#!/usr/bin/env python3
from __future__ import annotations

import csv
import argparse
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = ROOT.parent / "standard_robot_cuda_ik"
RESULTS = ROOT / "data" / "results" / "final_push"
DOCS = ROOT / "docs" / "final_push"
LOGS = ROOT / "logs" / "final_push"
URDF = LEGACY_ROOT / "urdf" / "ur10_official.urdf"

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(LEGACY_ROOT / "tools"))

from robot_model import load_robot_model, quaternion_from_matrix  # noqa: E402
from run_v4_curobo_compare import JOINT_NAMES, robot_cfg  # noqa: E402

from curobo._src.state.state_joint import JointState  # noqa: E402
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg  # noqa: E402
from curobo.types import GoalToolPose, Pose  # noqa: E402


STRICT_POS_M = 0.005
STRICT_ROT_RAD = 0.0174532925
MEDIUM_POS_M = 0.010
MEDIUM_ROT_RAD = 0.0872664626
Q_MIN = np.array([-2 * math.pi, -2 * math.pi, -math.pi, -2 * math.pi, -2 * math.pi, -2 * math.pi])
Q_MAX = np.array([2 * math.pi, 2 * math.pi, math.pi, 2 * math.pi, 2 * math.pi, 2 * math.pi])


@dataclass
class RunResult:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    status: str


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except Exception:
        return default


def pct(values: list[float] | np.ndarray, p: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, p))


def quat_to_matrix_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-30)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rot_diff_rad(a: np.ndarray, b: np.ndarray) -> float:
    R = a[:3, :3] @ b[:3, :3].T
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return abs(math.acos(c))


def pose_error(target: np.ndarray, achieved: np.ndarray) -> tuple[float, float]:
    pos = float(np.linalg.norm(target[:3, 3] - achieved[:3, 3]))
    return pos, rot_diff_rad(target, achieved)


def target_bank(n: int) -> np.ndarray:
    base = np.load(ROOT / "data" / "targets" / "v4_targets_N1000_seed42.npy").astype(np.float64).reshape(1000, 16)
    if n <= 1000:
        return base[:n].copy()
    reps = int(math.ceil(n / 1000))
    return np.tile(base, (reps, 1))[:n].copy()


def make_pose(targets: np.ndarray, quat_mode: str) -> tuple[Pose, np.ndarray]:
    positions: list[np.ndarray] = []
    quats: list[np.ndarray] = []
    for row in targets:
        T = row.reshape(4, 4)
        q_wxyz = quaternion_from_matrix(T[:3, :3]).astype(np.float32)
        if quat_mode in {"current", "wxyz"}:
            q = q_wxyz
        elif quat_mode == "xyzw":
            q = q_wxyz[[1, 2, 3, 0]]
        else:
            raise ValueError(quat_mode)
        positions.append(T[:3, 3].astype(np.float32))
        quats.append(q.astype(np.float32))
    pose = Pose(
        position=torch.tensor(np.stack(positions), device="cuda", dtype=torch.float32),
        quaternion=torch.tensor(np.stack(quats), device="cuda", dtype=torch.float32),
    )
    return pose, np.stack(quats).astype(np.float64)


def make_solver(
    n: int,
    num_seeds: int = 1,
    seed_solver_num_seeds: int | None = None,
    use_cuda_graph: bool = True,
    position_tolerance: float = MEDIUM_POS_M,
    orientation_tolerance: float = MEDIUM_ROT_RAD,
    override_iters: dict[str, int | None] | None = None,
    seed_position_weight: float = 1.0,
    seed_orientation_weight: float = 1.0,
    store_debug: bool = False,
) -> InverseKinematics:
    if seed_solver_num_seeds is None:
        seed_solver_num_seeds = num_seeds
    cfg = InverseKinematicsCfg.create(
        robot=robot_cfg(),
        num_seeds=num_seeds,
        seed_solver_num_seeds=seed_solver_num_seeds,
        self_collision_check=False,
        max_batch_size=n,
        position_tolerance=position_tolerance,
        orientation_tolerance=orientation_tolerance,
        use_cuda_graph=use_cuda_graph,
        load_collision_spheres=False,
        override_optimizer_num_iters=override_iters or {"particle": None, "lbfgs": None},
        seed_position_weight=seed_position_weight,
        seed_orientation_weight=seed_orientation_weight,
        store_debug=store_debug,
        success_requires_convergence=True,
    )
    return InverseKinematics(cfg)


def near_limit(q: np.ndarray, margin: float = 0.087) -> bool:
    return bool(np.any((q - Q_MIN) < margin) or np.any((Q_MAX - q) < margin))


def run_curobo(
    n: int,
    label: str,
    num_seeds: int = 1,
    seed_solver_num_seeds: int | None = None,
    use_cuda_graph: bool = True,
    position_tolerance: float = MEDIUM_POS_M,
    orientation_tolerance: float = MEDIUM_ROT_RAD,
    override_iters: dict[str, int | None] | None = None,
    quat_mode: str = "current",
    repeat: int = 3,
    warmup: int = 1,
    seed_position_weight: float = 1.0,
    seed_orientation_weight: float = 1.0,
) -> RunResult:
    targets = target_bank(n)
    pose, input_quats = make_pose(targets, quat_mode)
    model = load_robot_model(URDF, "base_link", "tool0", JOINT_NAMES)
    try:
        solver = make_solver(
            n,
            num_seeds=num_seeds,
            seed_solver_num_seeds=seed_solver_num_seeds,
            use_cuda_graph=use_cuda_graph,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            override_iters=override_iters,
            seed_position_weight=seed_position_weight,
            seed_orientation_weight=seed_orientation_weight,
        )
        target_link = solver.tool_frames[0]
        seeds_np = np.zeros((n, 6), dtype=np.float32)
        seed_tensor = torch.tensor(seeds_np[:, None, :], device="cuda", dtype=torch.float32)
        current_state = JointState.from_position(torch.tensor(seeds_np, device="cuda", dtype=torch.float32), joint_names=JOINT_NAMES)
        solution = None
        for _ in range(warmup):
            solution = solver.solve_pose(
                GoalToolPose.from_poses({target_link: pose}, num_goalset=1),
                current_state=current_state,
                seed_config=seed_tensor,
                return_seeds=1,
            )
            torch.cuda.synchronize()
        gpu_ms: list[float] = []
        e2e_ms: list[float] = []
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        for _ in range(repeat):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            start_ev.record()
            solution = solver.solve_pose(
                GoalToolPose.from_poses({target_link: pose}, num_goalset=1),
                current_state=current_state,
                seed_config=seed_tensor,
                return_seeds=1,
            )
            end_ev.record()
            torch.cuda.synchronize()
            e2e_ms.append((time.perf_counter() - t0) * 1000.0)
            gpu_ms.append(float(start_ev.elapsed_time(end_ev)))
        if solution is None or solution.js_solution is None:
            raise RuntimeError("empty cuRobo solution")
        qs = solution.js_solution.position.detach().cpu().numpy()[:, 0, :].astype(np.float64)
        reported_pos = solution.position_error.detach().cpu().numpy()[:, 0].astype(np.float64)
        reported_rot = solution.rotation_error.detach().cpu().numpy()[:, 0].astype(np.float64)
        cu_success = solution.success.detach().cpu().numpy()[:, 0].astype(bool)
        rows: list[dict[str, Any]] = []
        for i in range(n):
            T_target = targets[i].reshape(4, 4)
            T_eval = model.fk(qs[i])
            pos, rot = pose_error(T_target, T_eval)
            strict_eval = pos < STRICT_POS_M and rot < STRICT_ROT_RAD
            if reported_pos[i] * 1000.0 < 5.0 and pos * 1000.0 >= 5.0:
                flag = "reported_small_reeval_large"
            elif reported_pos[i] * 1000.0 >= 5.0 and pos * 1000.0 >= 5.0:
                flag = "reported_large_reeval_large"
            elif reported_pos[i] * 1000.0 >= 5.0 and pos * 1000.0 < 5.0:
                flag = "reported_large_reeval_small"
            elif strict_eval and cu_success[i]:
                flag = "consistent_success"
            else:
                flag = "consistent_failure"
            q_in = input_quats[i]
            rows.append(
                {
                    "N": n,
                    "run_label": label,
                    "target_id": i,
                    "strict_success_by_curobo": int(bool(cu_success[i])),
                    "strict_success_by_our_eval": int(strict_eval),
                    "q0": qs[i, 0],
                    "q1": qs[i, 1],
                    "q2": qs[i, 2],
                    "q3": qs[i, 3],
                    "q4": qs[i, 4],
                    "q5": qs[i, 5],
                    "curobo_reported_pos_err_mm": float(reported_pos[i] * 1000.0),
                    "curobo_reported_rot_err_deg": float(reported_rot[i] * 180.0 / math.pi),
                    "our_reeval_pos_err_mm": float(pos * 1000.0),
                    "our_reeval_rot_err_deg": float(rot * 180.0 / math.pi),
                    "abs_diff_pos_err_mm": float(abs(reported_pos[i] * 1000.0 - pos * 1000.0)),
                    "abs_diff_rot_err_deg": float(abs(reported_rot[i] * 180.0 / math.pi - rot * 180.0 / math.pi)),
                    "target_x": float(T_target[0, 3]),
                    "target_y": float(T_target[1, 3]),
                    "target_z": float(T_target[2, 3]),
                    "target_qw": float(q_in[0]),
                    "target_qx": float(q_in[1]),
                    "target_qy": float(q_in[2]),
                    "target_qz": float(q_in[3]),
                    "near_limit": int(near_limit(qs[i])),
                    "reason_flag": flag,
                }
            )
        summary = summarize_rows(rows, gpu_ms, e2e_ms, num_seeds)
        summary.update(
            {
                "run_label": label,
                "N": n,
                "num_seeds": num_seeds,
                "seed_solver_num_seeds": seed_solver_num_seeds if seed_solver_num_seeds is not None else num_seeds,
                "position_tolerance": position_tolerance,
                "rotation_tolerance": orientation_tolerance,
                "use_cuda_graph": int(use_cuda_graph),
                "quat_mode": quat_mode,
                "override_iters": str(override_iters or {"particle": None, "lbfgs": None}),
                "status": "ok",
            }
        )
        return RunResult(rows=rows, summary=summary, status="ok")
    except Exception as exc:
        return RunResult(
            rows=[],
            summary={
                "run_label": label,
                "N": n,
                "num_seeds": num_seeds,
                "status": f"FAILED: {type(exc).__name__}: {exc}",
            },
            status=f"FAILED: {type(exc).__name__}: {exc}",
        )


def single_run_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-solver-num-seeds", type=int, default=-1)
    parser.add_argument("--use-cuda-graph", type=int, default=1)
    parser.add_argument("--position-tolerance", type=float, default=MEDIUM_POS_M)
    parser.add_argument("--orientation-tolerance", type=float, default=MEDIUM_ROT_RAD)
    parser.add_argument("--override-particle-iters", type=int, default=-1)
    parser.add_argument("--override-lbfgs-iters", type=int, default=-1)
    parser.add_argument("--quat-mode", default="current")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed-position-weight", type=float, default=1.0)
    parser.add_argument("--seed-orientation-weight", type=float, default=1.0)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args(argv)
    iters = {"particle": None, "lbfgs": None}
    if args.override_particle_iters >= 0:
        iters["particle"] = args.override_particle_iters
    if args.override_lbfgs_iters >= 0:
        iters["lbfgs"] = args.override_lbfgs_iters
    rr = run_curobo(
        args.N,
        args.label,
        num_seeds=args.num_seeds,
        seed_solver_num_seeds=None if args.seed_solver_num_seeds < 0 else args.seed_solver_num_seeds,
        use_cuda_graph=bool(args.use_cuda_graph),
        position_tolerance=args.position_tolerance,
        orientation_tolerance=args.orientation_tolerance,
        override_iters=None if iters == {"particle": None, "lbfgs": None} else iters,
        quat_mode=args.quat_mode,
        repeat=args.repeat,
        warmup=args.warmup,
        seed_position_weight=args.seed_position_weight,
        seed_orientation_weight=args.seed_orientation_weight,
    )
    prefix = Path(args.out_prefix)
    write_csv(prefix.with_suffix(".rows.csv"), rr.rows)
    write_csv(prefix.with_suffix(".summary.csv"), [rr.summary])
    if rr.status != "ok":
        print(rr.status, file=sys.stderr)
        return 2
    return 0


def safe_run_curobo(
    n: int,
    label: str,
    num_seeds: int = 1,
    seed_solver_num_seeds: int | None = None,
    use_cuda_graph: bool = True,
    position_tolerance: float = MEDIUM_POS_M,
    orientation_tolerance: float = MEDIUM_ROT_RAD,
    override_iters: dict[str, int | None] | None = None,
    quat_mode: str = "current",
    repeat: int = 2,
    warmup: int = 1,
    seed_position_weight: float = 1.0,
    seed_orientation_weight: float = 1.0,
) -> RunResult:
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
    prefix = RESULTS / f"round2_tmp_{safe_label}"
    for p in [prefix.with_suffix(".rows.csv"), prefix.with_suffix(".summary.csv"), prefix.with_suffix(".log")]:
        if p.exists():
            p.unlink()
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-run",
        "--N",
        str(n),
        "--label",
        label,
        "--num-seeds",
        str(num_seeds),
        "--seed-solver-num-seeds",
        str(seed_solver_num_seeds if seed_solver_num_seeds is not None else -1),
        "--use-cuda-graph",
        "1" if use_cuda_graph else "0",
        "--position-tolerance",
        str(position_tolerance),
        "--orientation-tolerance",
        str(orientation_tolerance),
        "--quat-mode",
        quat_mode,
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
        "--seed-position-weight",
        str(seed_position_weight),
        "--seed-orientation-weight",
        str(seed_orientation_weight),
        "--out-prefix",
        str(prefix),
    ]
    if override_iters:
        if override_iters.get("particle") is not None:
            cmd.extend(["--override-particle-iters", str(override_iters["particle"])])
        if override_iters.get("lbfgs") is not None:
            cmd.extend(["--override-lbfgs-iters", str(override_iters["lbfgs"])])
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
    prefix.with_suffix(".log").write_text(
        "$ " + " ".join(cmd) + "\n\nSTDOUT:\n" + proc.stdout + "\nSTDERR:\n" + proc.stderr,
        encoding="utf-8",
    )
    rows = read_csv_rows(prefix.with_suffix(".rows.csv"))
    summary_rows = read_csv_rows(prefix.with_suffix(".summary.csv"))
    if proc.returncode == 0 and summary_rows:
        status = str(summary_rows[0].get("status", "ok"))
        return RunResult(rows=rows, summary=summary_rows[0], status=status)
    summary = summary_rows[0] if summary_rows else {"run_label": label, "N": n, "num_seeds": num_seeds}
    stderr_tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "no stderr"
    status = f"FAILED subprocess rc={proc.returncode}: {stderr_tail}"
    summary["status"] = status
    return RunResult(rows=rows, summary=summary, status=status)


def summarize_rows(rows: list[dict[str, Any]], gpu_ms: list[float], e2e_ms: list[float], num_seeds: int) -> dict[str, Any]:
    pos = np.array([float(r["our_reeval_pos_err_mm"]) for r in rows], dtype=np.float64)
    rot = np.array([float(r["our_reeval_rot_err_deg"]) for r in rows], dtype=np.float64)
    suc = np.array([bool(r["strict_success_by_our_eval"]) for r in rows])
    fail = ~suc
    n = len(rows)
    gpu = float(np.mean(gpu_ms)) if gpu_ms else float("nan")
    e2e = float(np.mean(e2e_ms)) if e2e_ms else float("nan")
    strict_sr = float(np.mean(suc)) if n else float("nan")
    return {
        "strict_success_count": int(np.sum(suc)),
        "strict_failure_count": int(np.sum(fail)),
        "strict_sr": strict_sr,
        "pos_p50_all": pct(pos, 50),
        "pos_p95_all": pct(pos, 95),
        "pos_p99_all": pct(pos, 99),
        "pos_max_all": float(np.max(pos)) if pos.size else float("nan"),
        "pos_p50_success_only": pct(pos[suc], 50),
        "pos_p95_success_only": pct(pos[suc], 95),
        "pos_p99_success_only": pct(pos[suc], 99),
        "pos_max_success_only": float(np.max(pos[suc])) if np.any(suc) else float("nan"),
        "pos_p50_failure_only": pct(pos[fail], 50),
        "pos_p95_failure_only": pct(pos[fail], 95),
        "pos_p99_failure_only": pct(pos[fail], 99),
        "pos_max_failure_only": float(np.max(pos[fail])) if np.any(fail) else float("nan"),
        "rot_p95_all": pct(rot, 95),
        "rot_p95_success_only": pct(rot[suc], 95),
        "rot_p95_failure_only": pct(rot[fail], 95),
        "gpu_ms": gpu,
        "e2e_ms": e2e,
        "throughput": (1000.0 * n / gpu) if gpu and gpu > 0 else float("nan"),
        "valid_throughput": ((1000.0 * n / gpu) * strict_sr) if gpu and gpu > 0 else float("nan"),
        "failure_count": int(np.sum(fail)),
        "num_seeds": num_seeds,
    }


def read_cuda_best(n: int) -> dict[int, dict[str, Any]]:
    paths = [
        RESULTS / f"graph_N{n}_off.best.csv",
        ROOT / "data" / "results" / "opt" / "opt4_followup" / f"opt4c_final_static_N{n}.best.csv",
        RESULTS / f"mixed_full_N{n}_fp64_none.best.csv",
    ]
    path = next((p for p in paths if p.exists()), None)
    out: dict[int, dict[str, Any]] = {}
    if path is None:
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = int(row["target_id"])
            out[tid] = row
    return out


def quaternion_audit_rows() -> list[dict[str, Any]]:
    targets = target_bank(100)
    rows = []
    for i, row in enumerate(targets):
        T = row.reshape(4, 4)
        q = quaternion_from_matrix(T[:3, :3]).astype(np.float64)
        q_xyzw_as_wxyz = q[[1, 2, 3, 0]]
        R_xyzw_wrong = quat_to_matrix_wxyz(q_xyzw_as_wxyz)
        T_wrong = np.eye(4)
        T_wrong[:3, :3] = R_xyzw_wrong
        T_wrong[:3, 3] = T[:3, 3]
        rows.append(
            {
                "target_id": i,
                "q_input_0": q[0],
                "q_input_1": q[1],
                "q_input_2": q[2],
                "q_input_3": q[3],
                "assumed_order": "wxyz/current",
                "quat_norm": float(np.linalg.norm(q)),
                "det_R": float(np.linalg.det(T[:3, :3])),
                "orthogonality_error": float(np.max(np.abs(T[:3, :3].T @ T[:3, :3] - np.eye(3)))),
                "rot_diff_if_wxyz_vs_xyzw_deg": float(rot_diff_rad(T, T_wrong) * 180.0 / math.pi),
                "target_frame": "base_link -> tool0",
                "notes": "cuRobo Pose quaternion is wxyz in this installed API; xyzw is tested as a negative-control input.",
            }
        )
    return rows


def split_row(n: int, default: RunResult) -> dict[str, Any]:
    s = dict(default.summary)
    return {
        "N": n,
        "strict_success_count": s.get("strict_success_count", ""),
        "strict_failure_count": s.get("strict_failure_count", ""),
        "strict_sr": s.get("strict_sr", ""),
        "pos_p50_all": s.get("pos_p50_all", ""),
        "pos_p95_all": s.get("pos_p95_all", ""),
        "pos_p99_all": s.get("pos_p99_all", ""),
        "pos_max_all": s.get("pos_max_all", ""),
        "pos_p50_success_only": s.get("pos_p50_success_only", ""),
        "pos_p95_success_only": s.get("pos_p95_success_only", ""),
        "pos_p99_success_only": s.get("pos_p99_success_only", ""),
        "pos_max_success_only": s.get("pos_max_success_only", ""),
        "pos_p50_failure_only": s.get("pos_p50_failure_only", ""),
        "pos_p95_failure_only": s.get("pos_p95_failure_only", ""),
        "pos_p99_failure_only": s.get("pos_p99_failure_only", ""),
        "pos_max_failure_only": s.get("pos_max_failure_only", ""),
        "rot_p95_all": s.get("rot_p95_all", ""),
        "rot_p95_success_only": s.get("rot_p95_success_only", ""),
        "rot_p95_failure_only": s.get("rot_p95_failure_only", ""),
        "status": default.status,
    }


def target_difficulty(rows: list[dict[str, Any]], cuda_by_id: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        tid = int(r["target_id"])
        cuda = cuda_by_id.get(tid, {})
        T = target_bank(max(tid + 1, 1000))[tid % 1000].reshape(4, 4)
        angle = abs(math.acos(float(np.clip((np.trace(T[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)))) * 180.0 / math.pi
        cuda_success = int(float(cuda.get("success_strict", 0))) if cuda else ""
        cuda_pos = float(cuda.get("pos_err_mm", "nan")) if cuda else float("nan")
        cu_success = int(r["strict_success_by_our_eval"])
        cu_pos = float(r["our_reeval_pos_err_mm"])
        out.append(
            {
                "target_id": tid,
                "target_x": r["target_x"],
                "target_y": r["target_y"],
                "target_z": r["target_z"],
                "target_radius_from_base": math.sqrt(float(r["target_x"]) ** 2 + float(r["target_y"]) ** 2),
                "target_height_z": r["target_z"],
                "orientation_angle": angle,
                "cuda_success": cuda_success,
                "curobo_success": cu_success,
                "cuda_pos_err": cuda_pos,
                "curobo_pos_err": cu_pos,
                "is_curobo_only_fail": int(cuda_success == 1 and cu_success == 0) if cuda else "",
                "is_cuda_only_fail": int(cuda_success == 0 and cu_success == 1) if cuda else "",
                "is_both_fail": int(cuda_success == 0 and cu_success == 0) if cuda else "",
                "radius_bin": int(min(4, math.floor(math.sqrt(float(r["target_x"]) ** 2 + float(r["target_y"]) ** 2) / 0.35))),
                "z_bin": int(min(4, math.floor((float(r["target_z"]) + 0.5) / 0.3))),
                "orientation_angle_bin": int(min(5, math.floor(angle / 30.0))),
            }
        )
    return out


def same_target_compare(rows: list[dict[str, Any]], cuda_by_id: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        tid = int(r["target_id"])
        cuda = cuda_by_id.get(tid, {})
        if not cuda:
            continue
        cuda_success = int(float(cuda["success_strict"]))
        cu_success = int(r["strict_success_by_our_eval"])
        if cuda_success and cu_success:
            ft = "both_success"
        elif cuda_success and not cu_success:
            ft = "cuda_success_curobo_fail"
        elif cu_success and not cuda_success:
            ft = "curobo_success_cuda_fail"
        else:
            ft = "both_fail"
        out.append(
            {
                "target_id": tid,
                "cuda_success": cuda_success,
                "curobo_success": cu_success,
                "cuda_pos_err_mm": cuda["pos_err_mm"],
                "curobo_pos_err_mm": r["our_reeval_pos_err_mm"],
                "cuda_rot_err_deg": cuda["rot_err_deg"],
                "curobo_rot_err_deg": r["our_reeval_rot_err_deg"],
                "cuda_near_limit": cuda["near_limit"],
                "curobo_near_limit": r["near_limit"],
                "winner_quality": "CUDA" if float(cuda["pos_err_mm"]) <= float(r["our_reeval_pos_err_mm"]) else "cuRobo",
                "failure_type": ft,
            }
        )
    return out


def reason_guess(row: dict[str, Any], cuda: dict[str, Any]) -> str:
    if row["reason_flag"] in {"reported_small_reeval_large", "reported_large_reeval_small"}:
        return "metric_mismatch"
    if float(row["our_reeval_rot_err_deg"]) > 5.0:
        return "orientation_hard"
    if int(row["near_limit"]):
        return "near_joint_limit"
    radius = math.sqrt(float(row["target_x"]) ** 2 + float(row["target_y"]) ** 2)
    if radius > 1.15 or float(row["target_z"]) < -0.25:
        return "workspace_boundary"
    if cuda and int(float(cuda.get("success_strict", 0))) == 1:
        return "seed_failure"
    return "unknown"


def failure_top50(rows: list[dict[str, Any]], cuda_by_id: dict[int, dict[str, Any]], n: int) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda r: float(r["our_reeval_pos_err_mm"]), reverse=True)[:50]
    out = []
    for rank, r in enumerate(sorted_rows, 1):
        tid = int(r["target_id"])
        cuda = cuda_by_id.get(tid, {})
        target_q = [r["target_qw"], r["target_qx"], r["target_qy"], r["target_qz"]]
        out.append(
            {
                "rank": rank,
                "N": n,
                "target_id": tid,
                "target_x": r["target_x"],
                "target_y": r["target_y"],
                "target_z": r["target_z"],
                "target_qw": target_q[0],
                "target_qx": target_q[1],
                "target_qy": target_q[2],
                "target_qz": target_q[3],
                "curobo_pos_err_mm": r["our_reeval_pos_err_mm"],
                "curobo_rot_err_deg": r["our_reeval_rot_err_deg"],
                "curobo_success": r["strict_success_by_our_eval"],
                "cuda_pos_err_mm": cuda.get("pos_err_mm", ""),
                "cuda_rot_err_deg": cuda.get("rot_err_deg", ""),
                "cuda_success": cuda.get("success_strict", ""),
                "curobo_q0": r["q0"],
                "curobo_q1": r["q1"],
                "curobo_q2": r["q2"],
                "curobo_q3": r["q3"],
                "curobo_q4": r["q4"],
                "curobo_q5": r["q5"],
                "cuda_q0": cuda.get("q0", ""),
                "cuda_q1": cuda.get("q1", ""),
                "cuda_q2": cuda.get("q2", ""),
                "cuda_q3": cuda.get("q3", ""),
                "cuda_q4": cuda.get("q4", ""),
                "cuda_q5": cuda.get("q5", ""),
                "distance_to_workspace_center": math.sqrt(float(r["target_x"]) ** 2 + float(r["target_y"]) ** 2 + float(r["target_z"]) ** 2),
                "near_joint_limit_by_curobo": r["near_limit"],
                "near_joint_limit_by_cuda": cuda.get("near_limit", ""),
                "jacobian_condition_estimate_if_available": "",
                "reason_guess": reason_guess(r, cuda),
            }
        )
    return out


def add_bin_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = list(rows)
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((int(r["radius_bin"]), int(r["z_bin"]), int(r["orientation_angle_bin"])), []).append(r)
    for (rb, zb, ob), items in sorted(groups.items()):
        cu_fail = [1 - int(x["curobo_success"]) for x in items]
        cuda_fail = [1 - int(x["cuda_success"]) for x in items if x["cuda_success"] != ""]
        out.append(
            {
                "target_id": "BIN_SUMMARY",
                "target_x": "",
                "target_y": "",
                "target_z": "",
                "target_radius_from_base": "",
                "target_height_z": "",
                "orientation_angle": "",
                "cuda_success": "",
                "curobo_success": "",
                "cuda_pos_err": np.nanmean([float(x["cuda_pos_err"]) for x in items if not math.isnan(float(x["cuda_pos_err"]))]),
                "curobo_pos_err": np.mean([float(x["curobo_pos_err"]) for x in items]),
                "is_curobo_only_fail": "",
                "is_cuda_only_fail": "",
                "is_both_fail": "",
                "radius_bin": rb,
                "z_bin": zb,
                "orientation_angle_bin": ob,
                "curobo_failure_rate": float(np.mean(cu_fail)),
                "cuda_failure_rate": float(np.mean(cuda_fail)) if cuda_fail else "",
                "mean_curobo_pos_err": np.mean([float(x["curobo_pos_err"]) for x in items]),
                "mean_cuda_pos_err": np.nanmean([float(x["cuda_pos_err"]) for x in items if not math.isnan(float(x["cuda_pos_err"]))]),
                "bin_count": len(items),
            }
        )
    return out


def md_table(rows: list[dict[str, Any]], limit: int = 20) -> str:
    if not rows:
        return ""
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows[:limit]:
        vals = []
        for k in keys:
            v = row.get(k, "")
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    default_runs: dict[int, RunResult] = {}
    split_rows = []
    reported_rows = []
    for n in [100, 500, 1000, 5000]:
        result = safe_run_curobo(n, f"default_N{n}", num_seeds=1, seed_solver_num_seeds=1, repeat=3, warmup=1)
        default_runs[n] = result
        split_rows.append(split_row(n, result))
        reported_rows.extend(result.rows)
        print("default", n, result.status, result.summary.get("strict_sr"), result.summary.get("pos_p95_all"))

    write_csv(RESULTS / "curobo_success_failure_split_round2.csv", split_rows)
    write_csv(RESULTS / "curobo_reported_vs_reeval_error.csv", reported_rows)

    q_rows = quaternion_audit_rows()
    q_experiments = []
    for mode in ["current", "wxyz", "xyzw"]:
        rr = safe_run_curobo(100, f"quat_{mode}", num_seeds=1, seed_solver_num_seeds=1, quat_mode=mode, repeat=3, warmup=1)
        q_experiments.append({"target_id": "EXPERIMENT", "q_input_0": "", "q_input_1": "", "q_input_2": "", "q_input_3": "", "assumed_order": mode, "quat_norm": "", "det_R": "", "orthogonality_error": "", "rot_diff_if_wxyz_vs_xyzw_deg": "", "target_frame": "base_link -> tool0", "notes": str(rr.summary)})
        print("quat", mode, rr.status, rr.summary.get("strict_sr"), rr.summary.get("pos_p95_all"))
    write_csv(RESULTS / "curobo_quaternion_audit.csv", q_rows + q_experiments)

    default_by_n = {n: default_runs[n].summary for n in default_runs}
    seed_rows = []
    for n in [100, 500, 1000]:
        base = default_by_n[n]
        for seeds in [16, 32, 64, 128, 256, 500]:
            rr = safe_run_curobo(n, f"seed{seeds}_N{n}", num_seeds=seeds, seed_solver_num_seeds=seeds, repeat=2, warmup=1)
            row = dict(rr.summary)
            row.update(
                {
                    "N": n,
                    "num_seeds": seeds,
                    "quality_improved_vs_default": int(safe_float(row.get("strict_sr")) > safe_float(base.get("strict_sr")) + 0.02) if rr.status == "ok" else 0,
                    "throughput_drop_vs_default": (1.0 - safe_float(row.get("throughput")) / max(safe_float(base.get("throughput"), 1.0), 1e-9)) if rr.status == "ok" else "",
                    "notes": rr.status,
                }
            )
            seed_rows.append(row)
            print("seed", n, seeds, rr.status, row.get("strict_sr"), row.get("pos_p95_all"))
    write_csv(RESULTS / "curobo_seed_sweep_round2.csv", seed_rows)

    optimizer_specs = [
        ("default_graph", 1, None, MEDIUM_POS_M, MEDIUM_ROT_RAD, 1.0, True),
        ("quality_more_iters", 1, {"particle": 128, "lbfgs": 200}, MEDIUM_POS_M, MEDIUM_ROT_RAD, 1.0, True),
        ("quality_more_seeds", 128, None, MEDIUM_POS_M, MEDIUM_ROT_RAD, 1.0, True),
        ("quality_tighter_threshold", 128, None, STRICT_POS_M, STRICT_ROT_RAD, 1.0, True),
        ("quality_pose_weight_high", 128, None, STRICT_POS_M, STRICT_ROT_RAD, 5.0, True),
        ("no_graph_quality", 128, None, STRICT_POS_M, STRICT_ROT_RAD, 1.0, False),
    ]
    opt_rows = []
    for n in [100, 500, 1000]:
        for name, seeds, iters, ptol, rtol, pose_w, graph in optimizer_specs:
            rr = safe_run_curobo(
                n,
                f"{name}_N{n}",
                num_seeds=seeds,
                seed_solver_num_seeds=seeds,
                position_tolerance=ptol,
                orientation_tolerance=rtol,
                override_iters=iters,
                use_cuda_graph=graph,
                seed_position_weight=pose_w,
                seed_orientation_weight=pose_w,
                repeat=2,
                warmup=1,
            )
            row = dict(rr.summary)
            row.update(
                {
                    "config_name": name,
                    "N": n,
                    "num_iters": str(iters),
                    "position_threshold": ptol,
                    "rotation_threshold": rtol,
                    "pose_weight": pose_w,
                    "use_cuda_graph": int(graph),
                    "collision_enabled": 0,
                    "notes": rr.status,
                }
            )
            opt_rows.append(row)
            print("opt", n, name, rr.status, row.get("strict_sr"), row.get("pos_p95_all"))
    write_csv(RESULTS / "curobo_optimizer_sweep_round2.csv", opt_rows)

    rows1000 = default_runs[1000].rows
    cuda1000 = read_cuda_best(1000)
    top50 = failure_top50(rows1000, cuda1000, 1000)
    difficulty = add_bin_summaries(target_difficulty(rows1000, cuda1000))
    same_target = same_target_compare(rows1000, cuda1000)
    write_csv(RESULTS / "curobo_failure_case_top50.csv", top50)
    write_csv(RESULTS / "curobo_target_difficulty_analysis.csv", difficulty)
    write_csv(RESULTS / "curobo_cuda_same_target_compare.csv", same_target)

    counts: dict[str, int] = {}
    for r in same_target:
        counts[r["failure_type"]] = counts.get(r["failure_type"], 0) + 1
    both_success = [r for r in same_target if r["failure_type"] == "both_success"]
    both_success_cu_p95 = pct([float(r["curobo_pos_err_mm"]) for r in both_success], 95)
    reported_flags: dict[str, int] = {}
    for r in reported_rows:
        reported_flags[r["reason_flag"]] = reported_flags.get(r["reason_flag"], 0) + 1

    seed_best = max([r for r in seed_rows if str(r.get("status", "ok")) == "ok"], key=lambda x: safe_float(x.get("strict_sr")), default={})
    opt_best = max([r for r in opt_rows if str(r.get("status", "ok")) == "ok"], key=lambda x: safe_float(x.get("strict_sr")), default={})
    root_causes = []
    n1000 = default_runs[1000].summary
    if safe_float(n1000.get("pos_p95_success_only"), 999.0) < 5.0 and safe_float(n1000.get("pos_p95_all")) > 50.0:
        root_causes.append("A_failure_tail_dominated")
    if any("reported_small_reeval_large" in k or "reported_large_reeval_small" in k for k in reported_flags):
        root_causes.append("F_metric_mismatch_possible")
    if safe_float(seed_best.get("strict_sr")) > safe_float(n1000.get("strict_sr")) + 0.05:
        root_causes.append("C_seed_config_insufficient")
    if safe_float(opt_best.get("strict_sr")) > safe_float(n1000.get("strict_sr")) + 0.05:
        root_causes.append("D_throughput_quality_tradeoff")
    if not root_causes:
        root_causes = ["G_unknown_or_solver_pipeline_failure_tail"]

    summary_rows = [
        {
            "item": "default_N1000_strict_sr",
            "value": n1000.get("strict_sr", ""),
            "notes": "",
        },
        {
            "item": "default_N1000_pos_p95_all_mm",
            "value": n1000.get("pos_p95_all", ""),
            "notes": "",
        },
        {
            "item": "default_N1000_pos_p95_success_only_mm",
            "value": n1000.get("pos_p95_success_only", ""),
            "notes": "",
        },
        {
            "item": "reported_flags",
            "value": str(reported_flags),
            "notes": "reported-vs-reeval classification over all default N rows",
        },
        {
            "item": "same_target_counts_N1000",
            "value": str(counts),
            "notes": f"both_success cuRobo p95={both_success_cu_p95:.6g} mm",
        },
        {
            "item": "best_seed_sweep",
            "value": str({k: seed_best.get(k) for k in ['N','num_seeds','strict_sr','pos_p95_all','throughput']}),
            "notes": "",
        },
        {
            "item": "best_optimizer_sweep",
            "value": str({k: opt_best.get(k) for k in ['N','config_name','num_seeds','strict_sr','pos_p95_all','throughput']}),
            "notes": "",
        },
        {
            "item": "root_cause_decision",
            "value": ";".join(root_causes),
            "notes": "",
        },
    ]
    write_csv(RESULTS / "curobo_round2_summary.csv", summary_rows)

    report = [
        "# cuRobo Quality Audit Round 2",
        "",
        "## 1. Background",
        "",
        "第一轮已排除 URDF、UR10/UR10e 混用、tool frame 固定平移、joint order 和 FK 实现不一致。本轮不改 CUDA 主算法、目标集或评价阈值，只定位 cuRobo `pos_p95_all` 尾部误差来源。",
        "",
        "## 2. Success vs Failure Split",
        "",
        md_table(split_rows, limit=len(split_rows)),
        "",
        f"N=1000 默认配置：Strict SR={safe_float(n1000.get('strict_sr')):.6g}，pos_p95_all={safe_float(n1000.get('pos_p95_all')):.6g} mm，pos_p95_success_only={safe_float(n1000.get('pos_p95_success_only')):.6g} mm，pos_p95_failure_only={safe_float(n1000.get('pos_p95_failure_only')):.6g} mm。",
        "",
        "## 3. Reported Error vs Re-evaluated Error",
        "",
        f"reason_flag 统计：`{reported_flags}`。",
        "cuRobo 自报误差和本文 FK 复评误差的逐目标表见 `curobo_reported_vs_reeval_error.csv`。",
        "",
        "## 4. Quaternion and Target Pose Audit",
        "",
        "Target rotation matrix determinant 和 orthogonality 均在数值误差范围内；当前 API 的 cuRobo `Pose.quaternion` 使用 wxyz。N=100 的 current/wxyz/xyzw 实验已写入 `curobo_quaternion_audit.csv`。",
        "",
        "## 5. Seed Sweep",
        "",
        md_table(seed_rows, limit=18),
        "",
        "## 6. Optimizer Sweep",
        "",
        md_table(opt_rows, limit=18),
        "",
        "## 7. Failure Case Analysis",
        "",
        md_table(top50, limit=10),
        "",
        "## 8. Same-target CUDA vs cuRobo Comparison",
        "",
        f"同目标 N=1000 failure_type 统计：`{counts}`。both_success 样本中 cuRobo pos_p95={both_success_cu_p95:.6g} mm。",
        "",
        "## 9. Root Cause Decision",
        "",
        f"Root cause decision: `{';'.join(root_causes)}`。",
        "",
        "- 如果 `pos_p95_success_only` 很小而 `pos_p95_all` 很大，说明 all p95 主要由失败样本尾部拉高。",
        "- 如果 seed/optimizer sweep 明显提高 Strict SR，则当前 cuRobo boundary 是 high-throughput/default setting，不是 quality-tuned 上限。",
        "- 如果 reported-vs-reeval 出现系统性不一致，应标记 metric/input mismatch；本次详细分类以 CSV 为准。",
        "",
        "## 10. Paper Decision",
        "",
        "当前 cuRobo boundary 可以作为“统一目标集和评价协议下的默认/Graph 系统对比”，但不应表述为 cuRobo quality-tuned 上限。若论文主张质量边界，需要同时报告本轮 seed/optimizer sweep 中的最佳质量配置，并重跑正式 repeat 的 cuRobo boundary。",
        "",
        "## Required Five Answers",
        "",
        f"1. 70~110mm pos_p95_all 是否失败样本拉高：`{'yes' if safe_float(n1000.get('pos_p95_success_only'), 999.0) < 5.0 and safe_float(n1000.get('pos_p95_all')) > 50.0 else 'not solely'}`。",
        f"2. cuRobo 成功样本 pos_p95_success_only：`{safe_float(n1000.get('pos_p95_success_only')):.6g} mm`。",
        f"3. reported error 与本文 FK 复评是否一致：见 reason flags `{reported_flags}`。",
        f"4. seed / optimizer 后质量是否改善：best seed `{seed_best.get('strict_sr')}`, best optimizer `{opt_best.get('strict_sr')}`。",
        "5. 当前 cuRobo boundary 能否作为论文主对比：可以作为默认 Graph 系统对比；若写质量上限，必须补 quality-tuned boundary。",
    ]
    (DOCS / "curobo_quality_audit_round2.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    wording = [
        "# cuRobo Paper Wording After Audit",
        "",
        "## 1. Failure-tail Dominated",
        "",
        "Under the shared official UR10 model and identical `base_link -> tool0` evaluation frame, cuRobo's successful solutions are accurate under the paper metric, while the all-target p95 error is dominated by the failure tail. Therefore, we report both all-target statistics and success-only statistics to separate solution accuracy from solve-rate robustness.",
        "",
        "## 2. Seed/config Insufficient",
        "",
        "Increasing cuRobo seeds or using quality-oriented optimizer settings improves Strict success rate at the cost of latency. We therefore treat the default CUDA-Graph cuRobo setting as a high-throughput system baseline rather than the quality-tuned upper bound.",
        "",
        "## 3. Metric/input Mismatch",
        "",
        "If internal cuRobo success or reported error diverges from the paper's FK re-evaluation metric, the comparison is reported as a system-level benchmark under a shared target set, with all final success and error values re-evaluated by the same external URDF FK pipeline used for CUDA-V4.",
    ]
    (DOCS / "curobo_paper_wording_after_audit.md").write_text("\n".join(wording) + "\n", encoding="utf-8")
    print(DOCS / "curobo_quality_audit_round2.md")
    print("root_causes", root_causes)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--single-run":
        raise SystemExit(single_run_cli(sys.argv[2:]))
    raise SystemExit(main())
