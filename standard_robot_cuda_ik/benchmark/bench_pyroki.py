from __future__ import annotations

import time
import sys
from pathlib import Path

import jax_dataclasses as jdc
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BenchmarkResult, compute_pose_error, load_robot_records, load_seed_values, mark_convergence


@jdc.jit
def solve_ik(robot_model, target_link_index, target_quat, target_pos, init_q):
    joint_var = robot_model.joint_var_cls(0)
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot_model,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_quat), target_pos),
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_constraint(robot_model, joint_var),
    ]
    solution = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make([joint_var.with_value(init_q)]),
            verbose=False,
            linear_solver="dense_cholesky",
        )
    )
    return solution[joint_var]


def run_pyroki_benchmark(robot: str, seed: int, N: int, repeat: int, seed_strategy: str = "zero_seed") -> BenchmarkResult:
    targets, _ = load_robot_records(robot, seed, N)
    seeds = load_seed_values(robot, seed, N, seed_strategy)
    urdf = yourdfpy.URDF.load(
        str(__import__("pathlib").Path(__file__).resolve().parents[1] / "urdf" / "ur10_official.urdf"),
        load_meshes=False,
        build_collision_scene_graph=False,
    )
    robot_model = pk.Robot.from_urdf(urdf)
    target_link_name = "tool0"
    target_link_idx = robot_model.links.names.index(target_link_name)

    warm_pos = jnp.array(targets[0].reshape(4, 4)[:3, 3])
    warm_quat = jnp.array(jaxlie.SO3.from_matrix(targets[0].reshape(4, 4)[:3, :3]).wxyz)
    warm_seed = jnp.array(seeds[0], dtype=jnp.float32)
    _ = solve_ik(robot_model, jnp.array(target_link_idx), warm_quat, warm_pos, warm_seed)

    result = BenchmarkResult(
        solver_name="pyroki",
        robot_model=robot,
        num_targets=N,
        repeat_count=repeat,
        uses_gpu=True,
        seed_strategy=seed_strategy,
        includes_jit_compile=False,
    )
    for rep in range(repeat):
        run_begin = time.perf_counter()
        pos_errs = []
        rot_errs = []
        successes = 0
        for i in range(N):
            T = targets[i].reshape(4, 4)
            pos = jnp.array(T[:3, 3])
            quat = jnp.array(jaxlie.SO3.from_matrix(T[:3, :3]).wxyz)
            q_sol = np.array(
                solve_ik(
                    robot_model,
                    jnp.array(target_link_idx),
                    quat,
                    pos,
                    jnp.array(seeds[i], dtype=jnp.float32),
                )
            )
            fk = robot_model.forward_kinematics(q_sol)
            ee = np.array(jaxlie.SE3(fk[target_link_idx]).as_matrix()).reshape(-1)
            pos_err, rot_err = compute_pose_error(targets[i], ee)
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
        result.host_api_total_time_ms.append((time.perf_counter() - run_begin) * 1000.0)
    result.throughput_targets_per_s = N / (np.mean(result.host_api_total_time_ms) / 1000.0)
    result.notes.append("per-target JAX solve, warm JIT excluded; initial configuration comes from the shared external seed asset")
    return result.finalize()
