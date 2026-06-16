# cuRobo Paper Wording After Audit

## 1. Failure-tail Dominated

Under the shared official UR10 model and identical `base_link -> tool0` evaluation frame, cuRobo's successful solutions are accurate under the paper metric, while the all-target p95 error is dominated by the failure tail. Therefore, we report both all-target statistics and success-only statistics to separate solution accuracy from solve-rate robustness.

## 2. Seed/config Insufficient

Increasing cuRobo seeds or using quality-oriented optimizer settings improves Strict success rate at the cost of latency. We therefore treat the default CUDA-Graph cuRobo setting as a high-throughput system baseline rather than the quality-tuned upper bound.

## 3. Metric/input Mismatch

If internal cuRobo success or reported error diverges from the paper's FK re-evaluation metric, the comparison is reported as a system-level benchmark under a shared target set, with all final success and error values re-evaluated by the same external URDF FK pipeline used for CUDA-V4.
