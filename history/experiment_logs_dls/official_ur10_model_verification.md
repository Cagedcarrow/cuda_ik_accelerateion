# Official UR10 Model Verification

- URDF path: `/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/urdf/ur10_official.urdf`
- URDF MD5: `cc4a23af365aaec3050462b8528bb994`
- Source repo: `https://github.com/UniversalRobots/Universal_Robots_ROS2_Description`
- Source ref: `4.3.1`
- Source commit: `ae333289875f9ba5a9ea6649a54036efb5ccabee`
- Base link: `base_link`
- TCP link: `tool0`
- Joint count: `6`
- Active joints: `shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint`
- CPU FK vs yourdfpy max abs error: `3.331e-16`

## Joint Table

- `shoulder_pan_joint`: parent=`base_link_inertia`, child=`shoulder_link`, axis=[0.0, 0.0, 1.0], xyz=[0.0, 0.0, 0.1273], rpy=[0.0, 0.0, 0.0], limits=[-6.283185307179586, 6.283185307179586]
- `shoulder_lift_joint`: parent=`shoulder_link`, child=`upper_arm_link`, axis=[0.0, 0.0, 1.0], xyz=[0.0, 0.0, 0.0], rpy=[1.570796327, 0.0, 0.0], limits=[-6.283185307179586, 6.283185307179586]
- `elbow_joint`: parent=`upper_arm_link`, child=`forearm_link`, axis=[0.0, 0.0, 1.0], xyz=[-0.612, 0.0, 0.0], rpy=[0.0, 0.0, 0.0], limits=[-3.141592653589793, 3.141592653589793]
- `wrist_1_joint`: parent=`forearm_link`, child=`wrist_1_link`, axis=[0.0, 0.0, 1.0], xyz=[-0.5723, 0.0, 0.163941], rpy=[0.0, 0.0, 0.0], limits=[-6.283185307179586, 6.283185307179586]
- `wrist_2_joint`: parent=`wrist_1_link`, child=`wrist_2_link`, axis=[0.0, 0.0, 1.0], xyz=[0.0, -0.1157, -2.373046667922381e-11], rpy=[1.570796327, 0.0, 0.0], limits=[-6.283185307179586, 6.283185307179586]
- `wrist_3_joint`: parent=`wrist_2_link`, child=`wrist_3_link`, axis=[0.0, 0.0, 1.0], xyz=[0.0, 0.0922, -1.891053610911353e-11], rpy=[1.570796326589793, 3.141592653589793, 3.141592653589793], limits=[-6.283185307179586, 6.283185307179586]

## Solver Consistency Notes

- `cuda`: 使用本项目 `urdf/ur10_official.urdf` 导出的常量表。
- `pyroki`: 直接加载同一份 URDF，TCP 固定为 `tool0`。
- `kdl`: 需要使用同一 active joint 顺序与 `tool0` 末端定义。
- `curobo`: 当前 benchmark 使用自定义 robot dict 强制指向同一份 URDF，并通过 `current_state + seed_config` 接入共享外部 seed。
- `hjcd_ik`: 当前默认不进入主公平对比表，除非确认模型链与 TCP 一致。

## FK Spot Check

- sample `0` max abs diff: `2.220e-16`
- sample `1` max abs diff: `1.110e-16`
- sample `2` max abs diff: `3.331e-16`
- sample `3` max abs diff: `2.220e-16`
- sample `4` max abs diff: `2.220e-16`
- sample `5` max abs diff: `2.220e-16`
- sample `6` max abs diff: `1.110e-16`
- sample `7` max abs diff: `3.331e-16`
- sample `8` max abs diff: `1.388e-16`
- sample `9` max abs diff: `2.220e-16`
- sample `10` max abs diff: `2.220e-16`
- sample `11` max abs diff: `2.220e-16`
- sample `12` max abs diff: `2.220e-16`
- sample `13` max abs diff: `1.249e-16`
- sample `14` max abs diff: `1.110e-16`
- sample `15` max abs diff: `2.220e-16`
- sample `16` max abs diff: `1.110e-16`
- sample `17` max abs diff: `2.220e-16`
- sample `18` max abs diff: `2.220e-16`
- sample `19` max abs diff: `2.220e-16`