# CUDA V4 Port Execution Log

- Generated: 2026-06-16 10:27:55
- Scope: V4-Final-K16 CUDA Port only; collision, V5, and Motion Generation excluded.
- Precision order: fp64_debug correctness first. mixed_fast is not promoted unless fp64 passes.
- Limit gradient: finite-difference gradient, matching Python prototype for correctness-first baseline.
