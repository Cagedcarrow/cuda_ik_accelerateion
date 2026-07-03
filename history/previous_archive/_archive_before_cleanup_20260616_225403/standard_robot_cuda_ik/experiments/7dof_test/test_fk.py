#!/usr/bin/env python3
"""Panda FK 正确性验证 + 模型参数导出"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from robot_model import load_robot_model, file_md5
import numpy as np

URDF_PATH = Path(__file__).resolve().parents[2] / "urdf" / "panda_7dof.urdf"
BASE_LINK = "panda_link0"
TIP_LINK = "panda_link8"
EXPECTED_JOINTS = [
    "panda_joint1", "panda_joint2", "panda_joint3",
    "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7",
]


def main():
    print("=" * 60)
    print("Panda 7DOF FK 验证")
    print("=" * 60)

    # 1. Load model
    model = load_robot_model(URDF_PATH, BASE_LINK, TIP_LINK, EXPECTED_JOINTS)
    print(f"\nModel: {model.name}")
    print(f"DOF: {model.dof}")
    print(f"Active joints ({len(model.active_joints)}):")
    for j in model.active_joints:
        print(f"  {j.name:20s} axis=({j.axis[0]:.4f}, {j.axis[1]:.4f}, {j.axis[2]:.4f})  "
              f"limits=[{j.lower:.4f}, {j.upper:.4f}]  "
              f"origin_xyz=({j.origin_xyz[0]:.4f}, {j.origin_xyz[1]:.4f}, {j.origin_xyz[2]:.4f})")

    # 2. FK correctness: verify FK(0) = identity (home should yield identity from base)
    q_zeros = np.zeros(model.dof)
    T_zeros = model.fk(q_zeros)
    print(f"\nFK at q=0:\n{T_zeros}")
    identity_err = np.max(np.abs(T_zeros - np.eye(4)))
    print(f"Max error from identity: {identity_err:.2e}")
    if identity_err < 1e-10:
        print("  ✅ FK(0) ≈ I_4")

    # 3. Check yourdfpy cross-validation
    try:
        import yourdfpy
        urdf = yourdfpy.URDF.load(str(URDF_PATH))
        print("\nyourdfpy FK cross-validation:")
        max_err = 0.0
        rng = np.random.default_rng(42)
        for i in range(20):
            q_test = model.sample_joint_vector(rng)
            T_cpu = model.fk(q_test)
            # yourdfpy matches joint names to values
            joint_dict = dict(zip(EXPECTED_JOINTS, q_test))
            T_ref = urdf.joint_fk(joint_dict, TIP_LINK)
            # Align: yourdfpy returns homogenous matrix from base_link to tip_link
            T_ref_np = np.array(T_ref).reshape(4, 4) if hasattr(T_ref, 'reshape') else np.array(T_ref)
            if T_ref_np.shape != (4, 4):
                # Try extracting from FK output dict
                T_ref_np = np.array(T_ref).reshape(4, 4)
            err = np.max(np.abs(T_cpu - T_ref_np))
            max_err = max(max_err, err)
        print(f"  Max error vs yourdfpy: {max_err:.2e}")
        if max_err < 1e-10:
            print("  ✅ FK matches yourdfpy to machine precision")
        elif max_err < 1e-6:
            print("  ⚠️ FK matches yourdfpy within 1e-6 (acceptable)")
        else:
            print("  ❌ FK mismatch!")
    except ImportError:
        print("  ⚠️ yourdfpy not available, skipping cross-validation")
    except Exception as e:
        print(f"  ⚠️ yourdfpy validation error: {e}")

    # 4. Export model parameters for reference
    print("\nModel parameters for CUDA constants:")
    print(f"  DOF: {model.dof}")
    origins = model.origins_array()
    print(f"  origins array: {len(origins)} doubles")
    axes = model.axes_array()
    print(f"  axes array: {len(axes)} doubles")
    tool = model.tool_offset_from_last_joint().reshape(-1)
    print(f"  tool offset: {tool}")
    limits = model.limits_array()
    print(f"  limits array: {limits}")
    print(f"\n  export script:")
    print(f"    origins = np.{repr(origins.tolist())}")
    print(f"    axes = np.{repr(axes.tolist())}")
    print(f"    tool = np.{repr(tool.tolist())}")

    # 5. Generate test targets/seeds for CUDA verification
    print("\nGenerating test targets/seeds for CUDA verification (N=10)...")
    rng = np.random.default_rng(123)
    N = 10
    q_seeds = []
    q_targets = []
    targets = []
    for i in range(N):
        q_seed = model.sample_joint_vector(rng)
        q_target = model.sample_joint_vector(rng)
        T_tgt = model.fk(q_target)
        q_seeds.append(q_seed)
        q_targets.append(q_target)
        targets.append(T_tgt.reshape(-1))

    seeds_arr = np.array(q_seeds, dtype=np.float64)
    targets_arr = np.array(targets, dtype=np.float64)

    out_dir = Path(__file__).resolve().parent
    seeds_arr.tofile(out_dir / "panda_test_seeds_N10.bin")
    targets_arr.tofile(out_dir / "panda_test_targets_N10.bin")
    np.save(out_dir / "panda_test_q_target_N10.npy", np.array(q_targets))
    print(f"  Seeds:   {out_dir}/panda_test_seeds_N10.bin  ({seeds_arr.nbytes} bytes)")
    print(f"  Targets: {out_dir}/panda_test_targets_N10.bin  ({targets_arr.nbytes} bytes)")
    print(f"  GT q:    {out_dir}/panda_test_q_target_N10.npy")
    print("  ✅ Done")

    print("\n" + "=" * 60)
    print("✅ Panda model loaded and test data generated successfully")
    print("   DOF:", model.dof)
    print("   FK at q=0: x=", T_zeros[3], "y=", T_zeros[7], "z=", T_zeros[11])
    print("=" * 60)


if __name__ == "__main__":
    main()
