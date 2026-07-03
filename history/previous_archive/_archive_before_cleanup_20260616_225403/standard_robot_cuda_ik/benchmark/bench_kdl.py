from __future__ import annotations

import time
from pathlib import Path
import sys

import numpy as np
import PyKDL

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BenchmarkResult, compute_pose_error, load_robot_records, load_seed_values, mark_convergence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from robot_model import RobotModel, load_robot_model


def _rotation_from_matrix(R: np.ndarray) -> PyKDL.Rotation:
    return PyKDL.Rotation(
        float(R[0, 0]),
        float(R[0, 1]),
        float(R[0, 2]),
        float(R[1, 0]),
        float(R[1, 1]),
        float(R[1, 2]),
        float(R[2, 0]),
        float(R[2, 1]),
        float(R[2, 2]),
    )


def _frame_from_matrix(T: np.ndarray) -> PyKDL.Frame:
    return PyKDL.Frame(
        _rotation_from_matrix(T[:3, :3]),
        PyKDL.Vector(float(T[0, 3]), float(T[1, 3]), float(T[2, 3])),
    )


def _build_kdl_chain(model: RobotModel) -> PyKDL.Chain:
    chain = PyKDL.Chain()
    for joint in model.active_joints:
        T = joint.origin_matrix
        R0 = T[:3, :3]
        p = T[:3, 3]
        # URDF uses T_parent_child = Origin * Rot(axis_local, q). KDL's segment
        # pose is JointPose(q) * FrameToTip, so we rotate the axis into parent
        # frame and keep the full origin transform in the segment tip frame.
        axis_parent = R0 @ joint.axis
        kdl_joint = PyKDL.Joint(
            joint.name,
            PyKDL.Vector(float(p[0]), float(p[1]), float(p[2])),
            PyKDL.Vector(float(axis_parent[0]), float(axis_parent[1]), float(axis_parent[2])),
            PyKDL.Joint.RotAxis,
        )
        chain.addSegment(PyKDL.Segment(joint.child, kdl_joint, _frame_from_matrix(T)))

    for joint in model.fixed_tail_joints:
        chain.addSegment(
            PyKDL.Segment(
                joint.child,
                PyKDL.Joint(PyKDL.Joint.Fixed),
                _frame_from_matrix(joint.origin_matrix),
            )
        )
    return chain


def _jnt_array(values: np.ndarray) -> PyKDL.JntArray:
    arr = PyKDL.JntArray(int(values.shape[0]))
    for i, value in enumerate(values):
        arr[i] = float(value)
    return arr


def _frame_to_flat(frame: PyKDL.Frame) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    for r in range(3):
        for c in range(3):
            T[r, c] = frame.M[r, c]
        T[r, 3] = frame.p[r]
    return T.reshape(-1)


def run_kdl_benchmark(robot: str, seed: int, N: int, repeat: int, seed_strategy: str = "zero_seed") -> BenchmarkResult:
    targets, _ = load_robot_records(robot, seed, N)
    seeds = load_seed_values(robot, seed, N, seed_strategy)
    model = load_robot_model(
        Path(__file__).resolve().parents[1] / "urdf" / "ur10_official.urdf",
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
    chain = _build_kdl_chain(model)
    fk_solver = PyKDL.ChainFkSolverPos_recursive(chain)
    ik_vel_solver = PyKDL.ChainIkSolverVel_pinv(chain)
    limits = model.limits_array().reshape(model.dof, 2)
    q_min = _jnt_array(limits[:, 0])
    q_max = _jnt_array(limits[:, 1])
    ik_solver = PyKDL.ChainIkSolverPos_NR_JL(chain, q_min, q_max, fk_solver, ik_vel_solver, 160, 1e-6)

    result = BenchmarkResult(
        solver_name="kdl",
        robot_model=robot,
        num_targets=N,
        repeat_count=repeat,
        uses_gpu=False,
        seed_strategy=seed_strategy,
        includes_data_copy=False,
    )
    for rep in range(repeat):
        run_begin = time.perf_counter()
        pos_errs = []
        rot_errs = []
        successes = 0
        kdl_failures = 0
        for i in range(N):
            q_init = _jnt_array(seeds[i])
            q_out = PyKDL.JntArray(model.dof)
            target_frame = _frame_from_matrix(targets[i].reshape(4, 4))
            rc = ik_solver.CartToJnt(q_init, target_frame, q_out)
            if rc != 0:
                kdl_failures += 1
            achieved = PyKDL.Frame()
            fk_solver.JntToCart(q_out, achieved)
            pos_err, rot_err = compute_pose_error(targets[i], _frame_to_flat(achieved))
            pos_errs.append(pos_err)
            rot_errs.append(rot_err)
            if mark_convergence(pos_err, rot_err):
                successes += 1
        if rep == 0:
            result.avg_pos_error_m = float(np.mean(pos_errs))
            result.max_pos_error_m = float(np.max(pos_errs))
            result.avg_rot_error_rad = float(np.mean(rot_errs))
            result.max_rot_error_rad = float(np.max(rot_errs))
            result.convergence_rate = successes / N
            result.failure_count = N - successes
            if kdl_failures:
                result.notes.append(f"KDL solver returned non-zero code on {kdl_failures}/{N} targets, but final FK was still evaluated")
        result.host_api_total_time_ms.append((time.perf_counter() - run_begin) * 1000.0)
    result.throughput_targets_per_s = N / (np.mean(result.host_api_total_time_ms) / 1000.0)
    result.notes.append("PyKDL ChainIkSolverPos_NR_JL baseline on the same URDF/targets/seeds path")
    return result.finalize()
