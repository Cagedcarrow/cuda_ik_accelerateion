# UR10 Model Consistency Audit

## Scope

- 本报告只审计 UR10 模型、frame、joint order、calibration 和 FK 对齐。
- 未修改 CUDA 主算法、阈值、目标集、cuRobo 结果或 IK 数学逻辑。

## Key Verdict

- Official source: `True`
- UR type: `ur10`
- Source repo: `https://github.com/UniversalRobots/Universal_Robots_ROS2_Description`
- Source ref: `4.3.1`
- Source commit: `ae333289875f9ba5a9ea6649a54036efb5ccabee`
- FK pass at same `base_link -> tool0`: `True`
- Need cuRobo boundary rerun because of model/frame audit: `False`
- 70-110mm likely caused by model/frame mismatch: `False`

## Required Answers

### 当前 cuRobo 是否使用了和 CUDA 相同的 UR10 模型？

是。cuRobo robot dict、Python evaluator 和 CUDA constants 均指向同一份 ur10_official.urdf；CUDA constants 由该 URDF 导出。

### 当前项目是否使用官方 UR10 ur_description？

是。来源为 UniversalRobots/Universal_Robots_ROS2_Description，ref=4.3.1，ur_type=ur10。

### 是否混用了 ur10/ur10e？

未发现 ur10e 模型参与当前 V4/cuRobo 路径。

### 是否存在 tool0/flange/ee_link 不一致？

当前求解与评价使用 tool0；URDF 中 wrist_3_link->flange->tool0 固定平移为 0，仅姿态旋转。未发现 70-110mm 固定 TCP offset。

### 是否存在 joint order 不一致？

未发现。CUDA、URDF path、cuRobo Kinematics joint_names 均为标准 UR10 六关节顺序。

### FK cross-check 最大误差是多少？

max_pos_diff=0.000336194 mm, max_rot_diff=2.94273e-05 deg, max_T_abs_diff=4.18401e-07。

### 70~110mm cuRobo 误差是否可能由模型/frame 导致？

按本次 FK 审计结果，不太可能由当前已加载模型/frame 导致；更像是 cuRobo IK 求解失败/收敛质量、目标分布、seed/optimizer 策略或 benchmark 评价方式导致。

### 是否需要重跑 cuRobo boundary？

从模型一致性角度不强制需要；若后续修 cuRobo seed/solver 配置或收敛参数，则需要重跑 boundary。

### 修正后的统一模型配置

URDF=/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/urdf/ur10_official.urdf; base_link=base_link; ee/tool=tool0; ur_type=ur10; joint_order=shoulder_pan_joint;shoulder_lift_joint;elbow_joint;wrist_1_joint;wrist_2_joint;wrist_3_joint; 真机实验前用 ur_calibration 提取 calibration yaml。

## FK Cross-check Maxima

| comparison | pos_diff_mm | rot_diff_deg | max_T_abs_diff |
| --- | --- | --- | --- |
| cuda_fk_vs_curobo_kinematics_fk | 0.0003361936491577532 | 2.9427337181133195e-05 | 4.1840141098981043e-07 |
| cuda_fk_vs_python_urdf_parser | 4.577566798522238e-13 | 2.4148365394514667e-06 | 4.440892098500626e-16 |
| cuda_fk_vs_yourdfpy_urdf_fk | 5.087681048627601e-13 | 2.4148365394514667e-06 | 5.551115123125783e-16 |
| curobo_kinematics_fk_vs_python_urdf_parser | 0.0003361936490659073 | 2.9427337181133195e-05 | 4.1840141098981043e-07 |
| curobo_kinematics_fk_vs_yourdfpy_urdf_fk | 0.00033619364894643217 | 2.9427337181133195e-05 | 4.1840141087878813e-07 |
| python_urdf_parser_vs_yourdfpy_urdf_fk | 4.4755865680201623e-13 | 2.4148365394514667e-06 | 4.440892098500626e-16 |

## Frame Audit Highlights

| frame_joint | parent | child | xyz_m | rpy_rad | translation_norm_mm | risk | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base_link-base_link_inertia | base_link | base_link_inertia | 0 0 0 | 0 0 3.141592653589793 | 0.0 | low | rotation-only fixed frame |
| wrist_3_link-ft_frame | wrist_3_link | ft_frame | 0 0 0 | 3.141592653589793 0 0 | 0.0 | low | rotation-only fixed frame |
| base_link-base_fixed_joint | base_link | base | 0 0 0 | 0 0 3.141592653589793 | 0.0 | low | rotation-only fixed frame |
| wrist_3-flange | wrist_3_link | flange | 0 0 0 | 0 -1.5707963267948966 -1.5707963267948966 | 0.0 | low | rotation-only fixed frame |
| flange-tool0 | flange | tool0 | 0 0 0 | 1.5707963267948966 0 1.5707963267948966 | 0.0 | low | rotation-only fixed frame |
| base_link_to_tool0_at_q_zero | base_link | tool0 | 1.1843 0.256141 0.0115999999 |  | 1211.7381160364098 | info | full arm FK at q=0, not a fixed TCP offset |
| wrist_3_link_to_tool0_effective | wrist_3_link | tool0 | 0 0 0 | composed from wrist_3-flange and flange-tool0 | 0.0 | low | effective translation is zero; position diff at same q is numerical roundoff |

## Joint Order Audit

| source | joint_order | matches_standard_order | reorder_mapping_to_standard | status | risk |
| --- | --- | --- | --- | --- | --- |
| expected_standard | shoulder_pan_joint;shoulder_lift_joint;elbow_joint;wrist_1_joint;wrist_2_joint;wrist_3_joint | 1 | 0;1;2;3;4;5 | ok | low |
| cuda | shoulder_pan_joint;shoulder_lift_joint;elbow_joint;wrist_1_joint;wrist_2_joint;wrist_3_joint | 1 | 0;1;2;3;4;5 | ok | low |
| urdf_path_base_link_to_tool0 | shoulder_pan_joint;shoulder_lift_joint;elbow_joint;wrist_1_joint;wrist_2_joint;wrist_3_joint | 1 | 0;1;2;3;4;5 | ok | low |
| curobo_kinematics | shoulder_pan_joint;shoulder_lift_joint;elbow_joint;wrist_1_joint;wrist_2_joint;wrist_3_joint | 1 | 0;1;2;3;4;5 | ok | low |

## Calibration Audit

- 当前合成 benchmark 使用官方默认 `config/ur10/default_kinematics.yaml` 生成的 flattened URDF。
- 未发现真实 UR10 factory calibration yaml / kinematics_config 被接入当前 V4 CUDA 或 cuRobo comparison 路径。
- 真机实验前必须使用 `ur_calibration` 从控制柜提取真实机器人校准参数，并用同一 calibration yaml 重新生成 URDF / CUDA constants / cuRobo robot config。

## Output Files

- `/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik_v4_cuda/data/results/final_push/ur10_model_sources.csv`
- `/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik_v4_cuda/data/results/final_push/ur10_frame_audit.csv`
- `/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik_v4_cuda/data/results/final_push/ur10_joint_order_audit.csv`
- `/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik_v4_cuda/data/results/final_push/ur10_fk_crosscheck.csv`
- `/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik_v4_cuda/data/results/final_push/ur10_model_fix_plan.csv`

## Fix Plan

| step | action | setting | required | notes |
| --- | --- | --- | --- | --- |
| 1 | Keep official UniversalRobots/Universal_Robots_ROS2_Description source pinned | source_ref=4.3.1, ur_type=ur10 | 1 | Already satisfied by ur10_official_source.json. |
| 2 | Use one shared flattened URDF for CUDA constants, Python evaluator, and cuRobo | /mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/urdf/ur10_official.urdf | 1 | Regenerate CUDA constants from the same URDF after any model change. |
| 3 | Fix ee frame explicitly | base_link -> tool0 | 1 | Do not mix wrist_3_link, flange, ft_frame, or controller base frame in evaluation. |
| 4 | Keep joint order fixed | shoulder_pan_joint;shoulder_lift_joint;elbow_joint;wrist_1_joint;wrist_2_joint;wrist_3_joint | 1 | If an external solver returns a different order, reorder to this mapping before FK/error evaluation. |
| 5 | Factory calibration for real robot | ur_calibration generated kinematics yaml | 0 | Not required for synthetic fair benchmark, required before real UR10 experiment. |
| 6 | Rerun cuRobo boundary after model/frame fix | N=100/500/1000/5000 | 0 | Required only if FK cross-check or frame audit fails; current audit decides this field. |

## Notes

- `wrist_3_link -> flange -> tool0` 在 URDF 中没有平移 offset；只存在固定姿态旋转。
- 如果把 `wrist_3_link` 或 `flange` 当作目标 frame，而把 `tool0` 当作评价 frame，主要会产生姿态差异；当前位置不会出现 70-110mm 固定平移。
- 当前 cuRobo 70-110mm `pos_p95_all` 更可能来自 IK 输出质量，而不是已加载模型几何不一致；建议下一步单独审计 cuRobo 的 `success`、目标 quaternion、seed_config shape、return_seeds 维度和失败样本。
