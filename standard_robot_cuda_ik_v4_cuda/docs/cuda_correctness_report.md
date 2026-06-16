# CUDA Correctness Report

## FK

- max_T_abs_diff: 4.440892e-16
- max_p_joint_diff: 2.220446e-16
- max_z_joint_diff: 3.330669e-16
- pass: True

## Analytical Jacobian

- max_abs_diff: 3.330669e-16
- fro_rel_error: 1.672617e-16
- pass: True

## Full K16 N=100 CUDA vs Python

- Strict SR diff pp: 0.000
- pos_p95_all diff mm: 0.000
- near_limit diff pp: 0.000
- failure_count: 0
- pass: True
