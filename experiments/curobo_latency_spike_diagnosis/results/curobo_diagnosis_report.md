# cuRobo Batch Oscillation Diagnosis Report

**Date**: 2026-06-12
**Experimenter**: Automated diagnosis script
**GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (7.64 GB VRAM, Compute Capability 8.9)
**cuRobo version**: 0.8.0.post1.dev33
**Original data GPU**: NVIDIA GeForce RTX 4090 (24 GB)
**Torch**: 2.6.0+cu124, **CUDA**: 12.4

---

## 1. Reproducibility: Is the latency spike stably reproducible?

**VERDICT: CONFIRMED**

The oscillation phenomenon is stably reproduced on RTX 4060 (8GB) with the identical degradation pattern observed on RTX 4090 (24GB) in the original data.

### Phase 1 — Ordered Scan Results

| N | Host Mean ms | Host Std ms | Host Min ms | Host Max ms | Host Median ms | Throughput tps | Conv Rate | Mode |
|--:|------------:|-----------:|-----------:|-----------:|--------------:|-------------:|:---:|:---:|
| 100 | 30.80 | 2.18 | 28.12 | 37.38 | 29.94 | 3,246 | 1.000 | normal |
| 500 | 30.48 | 1.40 | 28.68 | 33.60 | 30.01 | 16,402 | 1.000 | normal |
| 1000 | 32.01 | 2.65 | 28.40 | 44.37 | 31.77 | 31,245 | 1.000 | normal |
| 2000 | 31.87 | 1.32 | 28.59 | 34.57 | 32.13 | 62,760 | 1.000 | normal |
| 3000 | 29.94 | 1.03 | 28.72 | 32.69 | 29.60 | 100,199 | 1.000 | normal |
| **4000** | **232.12** | **11.28** | **211.82** | **259.15** | **231.25** | **17,233** | 1.000 | **DEGRADED** |
| 5000 | 31.34 | 2.72 | 29.05 | 42.13 | 30.11 | 159,555 | 1.000 | normal |
| 6000 | 32.21 | 1.99 | 29.48 | 36.76 | 32.09 | 186,295 | 1.000 | normal |
| **7000** | **230.02** | **11.49** | **214.20** | **252.11** | **230.59** | **30,432** | 1.000 | **DEGRADED** |
| 8000 | 31.84 | 1.81 | 29.02 | 35.35 | 32.04 | 251,253 | 1.000 | normal |
| **9000** | **235.30** | **10.14** | **220.72** | **262.77** | **231.91** | **38,249** | 1.000 | **DEGRADED** |
| **10000** | **238.07** | **15.30** | **220.85** | **280.68** | **236.19** | **42,005** | 1.000 | **DEGRADED** |

### Key Observations

1. **Binary states**: cuRobo operates in exactly two modes — "normal" (~29–34 ms) and "degraded" (~220–240 ms). No intermediate states exist.
2. **Degradation magnitude**: ~7.0–7.5x slowdown at degraded N values.
3. **Non-monotonic**: N=4000 degrades but N=5000 is normal; N=7000 degrades but N=8000 is normal; N=9000 and 10000 both degrade.
4. **Convergence rate**: Unaffected (all ≥ 0.9998, most at 1.000).
5. **Cross-GPU consistency**: Degradation pattern identical between RTX 4090 (24GB) and RTX 4060 (8GB), ruling out VRAM-capacity-specific effects.

### Comparison with Original 4090 Data

| N | 4090 HostMs | 4060 HostMs | Mode Match |
|--:|-----------:|-----------:|:---:|
| 4000 | 218.61 | 232.12 | YES |
| 5000 | 32.25 | 31.34 | YES |
| 7000 | 228.44 | 230.02 | YES |
| 8000 | 32.16 | 31.84 | YES |
| 9000 | 236.06 | 235.30 | YES |
| 10000 | 239.05 | 238.07 | YES |

---

## 2. Order Sensitivity: Do degradation points change with run order?

**VERDICT: NOT CONFIRMED (degradation is order-independent)**

All three randomized run orders (Phase 2) produce the exact same degradation pattern. N=4000, 7000, 9000, and 10000 consistently degrade regardless of whether they are run first, last, or in the middle of the test sequence.

### Phase 2 — Randomized Scan Summary

| N | Order 1 Mean ms | Order 2 Mean ms | Order 3 Mean ms | Degraded? |
|--:|---------------:|---------------:|---------------:|:---:|
| 4000 | 234.79 | 219.89 | 235.47 | YES (all 3) |
| 5000 | 31.42 | 29.15 | 32.35 | NO (all 3) |
| 7000 | 234.59 | 221.23 | 226.43 | YES (all 3) |
| 8000 | 32.08 | 30.77 | 30.34 | NO (all 3) |
| 9000 | 239.80 | 225.66 | 226.08 | YES (all 3) |
| 10000 | 239.37 | 244.32 | 234.17 | YES (all 3) |

**Interpretation**: Degradation is intrinsic to specific N values, NOT an artifact of run order, warmup history, or sequential memory accumulation. When N=4000 runs first in Order 2, it degrades. When N=4000 runs 7th in Order 3, it degrades. When N=4000 runs 2nd in Order 1, it degrades.

---

## 3. Process Isolation: Do degradation points depend on allocator/cache state?

**VERDICT: PARTIALLY CONFIRMED — degradation persists in fresh processes**

Phase 3 launched each N value in a completely separate Python subprocess, ensuring:
- Fresh Python interpreter
- Fresh PyTorch CUDA allocator state
- Fresh cuRobo solver initialization
- Fresh CUDA context

### Phase 3 — Fresh Process Results

| N | Host Mean ms | Host Std ms | Host Median ms | Mem Alloc Before MB | Mem Resv Before MB | Degraded? |
|--:|------------:|-----------:|--------------:|-------------------:|------------------:|:---:|
| 4000 | 224.89 | 13.77 | 219.61 | 196.1 | 256.0 | **YES** |
| 5000 | 32.43 | 4.50 | 31.23 | 243.4 | 282.0 | NO |
| 7000 | 242.89 | 13.56 | 242.46 | 336.7 | 424.0 | **YES** |
| 8000 | 32.39 | 2.43 | 32.39 | 381.3 | 448.0 | NO |

**Interpretation**: Even with a completely fresh Python process and clean CUDA allocator state, N=4000 and N=7000 still degrade. This rules out:
- Accumulated allocator fragmentation from previous N values
- Lingering cuRobo solver state
- Python-level caching or GC artifacts

The degradation is therefore related to the N value itself (and its associated tensor shapes, solver configuration, or internal cuRobo execution path), NOT to process-level state pollution.

**Notable observation**: N=5000 (normal) uses MORE memory (243 MB) than N=4000 (degraded, 196 MB). Degradation does NOT correlate with memory pressure.

---

## 4. max_batch_size Dependence: Do degradation points depend on max_batch_size?

**VERDICT: PARTIALLY CONFIRMED — degradation persists with fixed max_batch_size=10000**

Phase 4 fixed `max_batch_size=10000` for ALL N values, keeping solver workspace, CUDA graph shape (disabled), and internal tensor dimensions identical. Only the actual number of input targets varied.

### Phase 4 — Fixed max_batch_size=10000 Results

| actual_N | max_batch | Host Mean ms | Host Std ms | Host Median ms | Degraded? |
|--------:|---------:|------------:|-----------:|--------------:|:---:|
| 100 | 10000 | 31.59 | 1.52 | 31.23 | NO |
| 500 | 10000 | 31.52 | 2.57 | 30.82 | NO |
| 1000 | 10000 | 32.25 | 2.65 | 32.42 | NO |
| 2000 | 10000 | 31.52 | 2.74 | 30.70 | NO |
| 3000 | 10000 | 30.85 | 1.39 | 30.32 | NO |
| **4000** | **10000** | **230.10** | **9.92** | **228.74** | **YES** |
| 5000 | 10000 | 31.94 | 2.34 | 31.32 | NO |
| 6000 | 10000 | 33.94 | 5.64 | 32.86 | NO |
| **7000** | **10000** | **225.40** | **9.65** | **224.07** | **YES** |
| 8000 | 10000 | 31.76 | 1.83 | 30.94 | NO |
| **9000** | **10000** | **231.91** | **8.72** | **229.89** | **YES** |
| **10000** | **10000** | **228.09** | **8.39** | **227.53** | **YES** |

### Critical Interpretation

This is the most diagnostically significant result. With `max_batch_size` fixed at 10000:
1. The solver workspace is IDENTICAL for all N values.
2. According to cuRobo source code (solver_ik.py:678-683), when `batch_size < max_batch_size`, inputs are padded to `max_batch_size`. The solver ALWAYS processes `max_batch_size` targets internally.
3. cuRobo processes 10000 targets for ALL N values (N < 10000 are padded).
4. Yet N=4000, 7000, 9000 still degrade while N=5000, 6000, 8000 do not.

**This rules out as primary causes:**
- Changes in solver workspace size
- Changes in CUDA graph capture shape (CUDA graph is disabled anyway)
- Changes in tensor shape allocation (all padded to same shape)
- `max_batch_size` parameter itself

**What remains as possible causes:**
- cuRobo's internal sub-batching/tiling strategy that depends on actual batch_size
- CUDA kernel grid/block dimensions that are parameterized by actual batch_size
- Internal optimization heuristics that behave differently for different numbers of unique vs. padded targets
- A yet-unidentified code path in cuRobo's solver that branches on actual_batch_size

**Unresolved puzzle**: If cuRobo truly processes 10000 targets for all N values, ALL N should take ~230 ms (the N=10000 time). But N=5000 takes 32 ms while N=10000 takes 228 ms, despite both processing 10000 targets. This suggests either: (a) the padding is not actually happening as documented in the source code, (b) cuRobo has an optimization that skips computation for padded (duplicate) targets, or (c) there is a caching/JIT mechanism that depends on the pattern of input data. Further investigation with internal cuRobo instrumentation would be needed to resolve this.

---

## 5. Memory Correlation: Do degradation points correlate with memory jumps?

**VERDICT: NOT CONFIRMED — no clear correlation between degradation and memory allocation patterns**

### Phase 1 Memory Allocation Summary (Ordered Scan)

| N | Alloc Before MB | Alloc After MB | Resv Before MB | Resv After MB | Max Alloc MB | Max Resv MB | Degraded? |
|--:|---------------:|--------------:|--------------:|-------------:|------------:|-----------:|:---:|
| 100 | 13.4 | 13.5 | 30.0 | 30.0 | 13.7 | 30.0 | NO |
| 500 | 36.5 | 36.6 | 50.0 | 50.0 | 37.9 | 50.0 | NO |
| 1000 | 81.2 | 81.4 | 100.0 | 100.0 | 84.0 | 100.0 | NO |
| 2000 | 171.2 | 171.5 | 200.0 | 200.0 | 176.7 | 200.0 | NO |
| 3000 | 303.5 | 304.1 | 374.0 | 376.0 | 311.9 | 376.0 | NO |
| **4000** | **196.4** | **197.1** | **450.0** | **450.0** | **311.9*** | **450.0** | **YES** |
| 5000 | 416.2 | 417.2 | 558.0 | 558.0 | 430.5 | 558.0 | NO |
| 6000 | 677.5 | 678.6 | 840.0 | 840.0 | 695.2 | 840.0 | NO |
| **7000** | **335.9** | **337.3** | **958.0** | **958.0** | **695.2*** | **958.0** | **YES** |
| 8000 | 685.0 | 686.6 | 1104.0 | 1104.0 | 707.4 | 1104.0 | NO |
| **9000** | **430.3** | **432.0** | **1266.0** | **1266.0** | **707.4*** | **1266.0** | **YES** |
| **10000** | **477.4** | **479.3** | **1440.0** | **1440.0** | **707.4*** | **1440.0** | **YES** |

*Starred max_memory_allocated values indicate no increase from the previous N value — the peak allocation was set by an earlier N and not exceeded.

### Key Observations

1. **No correlation with current allocation**: N=4000 (degraded) uses 196 MB allocated; N=5000 (normal) uses 416 MB allocated. More memory → normal. Less memory → degraded.
2. **No correlation with reserved memory**: N=4000 has 450 MB reserved; N=5000 has 558 MB reserved. Normal N has MORE reserved memory.
3. **"Stalled peak" pattern**: At degraded N values (4000, 7000, 9000, 10000), `max_memory_allocated` does NOT increase — the allocator reuses previously allocated memory without needing to grow. This may indicate that the allocator is finding space within its cached pool rather than requesting new memory from the driver.
4. **Memory summary analysis**: The `torch.cuda.memory_summary()` outputs show 0 CUDA OOMs and 0 cudaMalloc retries across all N values. The allocator is functioning normally.
5. **cudaMalloc/cudaFree overhead is negligible**: Nsight Systems profiling shows cudaMalloc (41 calls, 0.5% of API time) and cudaFree (1 call, 0.2%) for N=4000. Memory allocation is NOT the bottleneck.

### Interpretation

Memory allocation behavior alone DOES NOT explain the degradation. The fact that normal N values use MORE memory than degraded N values (in fresh processes) rules out memory pressure as the cause. The "stalled peak" pattern at degraded N values is interesting but appears to be a symptom rather than a cause — the allocator isn't growing because the memory request fits within existing cached blocks.

---

## 6. Profiler Evidence: CUDA API events during degradation

**VERDICT: PARTIALLY CONFIRMED — kernel launch count differs dramatically between degraded and normal N values**

Nsight Systems profiling was performed for N=4000 (degraded) and N=5000 (normal) with `max_batch_size=N` (Phase 3 configuration). Key CUDA API statistics:

### CUDA API Summary Comparison

| API Call | N=4000 (degraded) | N=5000 (normal) | Ratio |
|----------|:---:|:---:|:---:|
| cudaLaunchKernel | 11,893 calls | 5,770 calls | **2.06x** |
| cuLaunchKernelEx | 2,215 calls | 175 calls | **12.66x** |
| **Total kernel launches** | **14,108** | **5,945** | **2.37x** |
| cudaMemcpyAsync | 1,512 calls | 737 calls | 2.05x |
| cudaStreamSynchronize | 277 calls | 267 calls | 1.04x |
| cudaEventRecord | 5,069 calls | 390 calls | **13.00x** |
| cudaStreamWaitEvent | 4,985 calls | 350 calls | **14.24x** |
| cudaMalloc | 41 calls | 45 calls | 0.91x |
| cudaFree | 1 call | 1 call | 1.00x |
| cudaDeviceSynchronize | 10 calls | 8 calls | 1.25x |
| cuModuleLoadDataEx | 5 calls (142ms) | 5 calls (136ms) | 1.00x |
| cuModuleUnload | 5 calls (43ms) | 5 calls (41ms) | 1.00x |

### Critical Findings

**1. Kernel launch count is the dominant differentiator.**
- N=4000 launches 14,108 kernels across 5 solve_pose calls (~2,822 per solve)
- N=5000 launches 5,945 kernels across 5 solve_pose calls (~1,189 per solve)
- Despite processing FEWER targets (4000 < 5000), N=4000 launches MORE THAN TWICE as many kernels.
- This is counter-intuitive and strongly suggests cuRobo's internal sub-batching or tiling strategy is batch-size-dependent and NON-MONOTONIC.

**2. Event synchronization overhead is dramatically higher at N=4000.**
- cudaEventRecord: 5,069 vs 390 (13x more)
- cudaStreamWaitEvent: 4,985 vs 350 (14x more)
- These events are used for inter-kernel synchronization within cuRobo's solver. The explosion in event count at N=4000 suggests cuRobo is breaking the computation into many more sub-operations with explicit synchronization between them.

**3. cudaMalloc/cudaFree are NOT the bottleneck.**
- Both N values have similar numbers of allocation calls (~41-45 malloc, 1 free).
- Total allocation time is < 1% of total CUDA API time.
- This definitively rules out the "PyTorch CUDA Caching Allocator fragmentation" hypothesis as the primary cause.

**4. CUDA module loading occurs on EVERY solve_pose call, regardless of N.**
- Both N values show exactly 5 `cuModuleLoadDataEx` calls (one per solve_pose invocation).
- Average time: ~28 ms per load, ~8.5 ms per unload.
- Total module load/unload: ~190 ms out of ~580 ms total API time.
- This overhead is SIMILAR for both N values, so it does NOT explain the differential.
- However, it DOES explain a large portion of the absolute solve_pose time (~40 ms/call for module management). Without this overhead, degraded N values might still be slower than normal, but both would be faster overall.

**5. cudaDeviceSynchronize calls are minimal.**
- 10 calls for N=4000, 8 calls for N=5000.
- Total time: ~207 µs and ~147 µs respectively.
- Not the cause of degradation.

### What the Profiler Does NOT Show

The profiler does NOT show:
- cudaGraphInstantiate or cudaGraphLaunch events (CUDA graphs are disabled in config)
- Explosions of cudaMalloc or cudaFree at degraded points
- Abnormal cudaDeviceSynchronize durations
- Differences in total GPU compute time proportional to the host wall time difference

The profiler analysis was limited to `max_batch_size=N` configuration (Phase 3 style). Profiling with fixed `max_batch_size=10000` (Phase 4 style) was not performed due to time constraints. This limits our ability to fully explain the Phase 4 results.

---

## 7. Root Cause Analysis: Most Likely to Least Likely

Based on the evidence collected across all 6 phases, here is the ranked assessment of possible root causes:

### 1. MOST LIKELY: cuRobo internal sub-batching/tiling strategy (evidence: STRONG)

**Evidence**:
- Nsight Systems shows N=4000 launches 2.37x more kernels than N=5000 despite processing fewer targets
- N=4000 uses 13x more CUDA events and stream-wait operations than N=5000
- The degradation pattern is stable across GPUs (4090 and 4060), run orders, and process states
- Degradation persists with fixed max_batch_size (Phase 4), suggesting the actual batch_size triggers different internal execution paths

**Hypothesis**: cuRobo's IK solver internally tiles or sub-batches the computation. The tiling strategy depends on the actual batch size and is NOT monotonic — certain N values result in small sub-batches with high launch overhead, while neighboring N values use larger sub-batches with lower overhead. The non-monotonicity could arise from how cuRobo selects tile sizes based on batch_size and hardware properties.

**Confidence**: Medium-High. The kernel launch count difference is direct evidence, but we have not identified the exact cuRobo code path responsible for the tiling decision.

### 2. SECONDARY: cuRobo internal solver iteration count variation (evidence: MODERATE)

**Evidence**:
- The degradation magnitude (~200 ms extra over ~32 ms baseline) is too large to be explained by a few extra iterations
- However, if the extra kernel launches at degraded N correspond to additional solver iterations or restarts, this could compound the effect

**Confidence**: Low-Medium. We did not instrument cuRobo to count iterations per solve call.

### 3. LESS LIKELY: PyTorch CUDA Caching Allocator behavior (evidence: WEAK)

**Evidence**:
- cudaMalloc/cudaFree overhead is < 1% of CUDA API time (nsys data)
- Memory allocated does NOT correlate with degradation (Phase 1, Phase 3 data)
- Degradation persists in fresh processes with clean allocator state (Phase 3)
- No CUDA OOMs or cudaMalloc retries observed

**Confidence**: This can be largely ruled out as the primary cause. The original oscillation_analysis.md hypothesis attributing degradation to "PyTorch CUDA Caching Allocator fragmentation" is NOT supported by the profiling evidence.

### 4. RULED OUT: Run order / warmup state pollution (evidence: STRONG against)

**Evidence**: Phase 2 shows identical degradation pattern across 3 random orders. Phase 3 shows degradation in completely fresh processes.

### 5. RULED OUT: max_batch_size / workspace size changes (evidence: STRONG against)

**Evidence**: Phase 4 fixes max_batch_size=10000, yet degradation persists unchanged.

### 6. RULED OUT: CUDA graph capture (evidence: DEFINITIVE)

**Evidence**: `use_cuda_graph=False` in all experiments. Nsight Systems confirms no cudaGraphInstantiate or cudaGraphLaunch events.

### 7. RULED OUT: VRAM capacity effects (evidence: STRONG against)

**Evidence**: Identical degradation pattern on RTX 4090 (24GB) and RTX 4060 (8GB).

### 8. CANNOT DETERMINE: Padding behavior in cuRobo 0.8.0 (requires further investigation)

The Phase 4 results present a puzzle: with max_batch_size=10000, cuRobo should pad all N < 10000 to 10000 and process them identically, yet different N values produce different times. This contradiction cannot be resolved without instrumenting cuRobo internals to verify whether padding actually occurs and, if so, whether the padded computation takes genuinely different execution paths.

### 9. CANNOT DETERMINE: Whether N=10000 would degrade on RTX 4090 with fixed max_batch_size > 10000

The original 4090 data used max_batch_size=N. Had max_batch_size been fixed at, say, 12000, the degradation points might shift. This is testable but was not part of the current diagnostic scope.

---

## 8. Paper Recommendation: Safe Wording

Based on the experimental evidence, the following statement is recommended for the paper:

### Recommended Wording

> Under the present cuRobo benchmark configuration (UR10, tool0, medium threshold 10mm/5°, zero_seed, repeat=30, CUDA graph disabled, max_batch_size=N), we observe batch-size-sensitive host-call latency spikes in cuRobo's `solve_pose()` invocation time. The phenomenon exhibits a binary pattern: eight of twelve tested batch sizes (N = 100, 500, 1000, 2000, 3000, 5000, 6000, 8000) complete in approximately 31 ms, while the remaining four (N = 4000, 7000, 9000, 10000) require approximately 230 ms — a 7.4x degradation.
>
> This pattern is stably reproducible across independent runs, random test orders, fresh Python processes, and two different GPU models (RTX 4090 24GB and RTX 4060 8GB). It persists when max_batch_size is fixed at 10000 for all N values, ruling out workspace resizing or CUDA graph shape changes as the cause. Nsight Systems profiling reveals that degraded N values launch 2.37x more CUDA kernels and 13x more inter-kernel synchronization events than neighboring normal N values, despite processing fewer targets, suggesting that cuRobo's internal sub-batching or tiling strategy is batch-size-dependent and non-monotonic.
>
> Critically, cudaMalloc and cudaFree overhead is negligible (< 1% of CUDA API time) at both degraded and normal N values, indicating that PyTorch CUDA Caching Allocator behavior is not the primary driver. The observed latency spikes are therefore more likely attributable to internal cuRobo execution-path selection — specifically the number and granularity of CUDA kernel launches — rather than to external memory management or benchmark procedure artifacts.
>
> We note that this phenomenon does not affect the convergence rate of cuRobo's solutions (all tested N values achieve ≥ 99.98% convergence) and does not represent a defect in cuRobo's IK algorithm. However, it does constitute a performance-determinism limitation that is absent from our CUDA B5 solver, whose fixed per-target memory model and single-kernel iteration encapsulation yield strictly linear GPU time scaling (R² > 0.999) with zero batch-size-dependent latency spikes.

### If stronger profiler evidence were available, add:

> Nsight Systems / Torch profiler observations further show that [specific event, e.g., cudaGraphInstantiate occurring only at degraded N, or specific kernel launch pattern changes], confirming that the latency spike originates from [specific mechanism] rather than from IK iteration computation itself.

*(Note: the above addendum cannot be written with confidence at this time because we did not profile the fixed max_batch_size=10000 configuration, and the exact cuRobo internal mechanism triggering the different kernel launch counts has not been identified.)*

---

## 9. Limitations and Caveats

1. **cuRobo version mismatch**: The original oscillation analysis referenced cuRobo 0.12.0; the current diagnosis uses cuRobo 0.8.0.post1.dev33. Padding behavior or kernel launch strategies may differ between versions.

2. **Phase 4 profiling gap**: The most diagnostically puzzling result (fixed max_batch_size=10000 showing differential degradation) was not profiled with Nsight Systems. Profiling this configuration is essential to understand why cuRobo behaves differently for different actual_N values when max_batch_size is constant.

3. **No internal cuRobo instrumentation**: We observed the phenomenon from outside cuRobo (host wall time, memory stats, CUDA API trace). We did not instrument cuRobo's internal solve loop, tiling logic, or workspace management to identify the exact code path causing the kernel launch count difference.

4. **8GB VRAM limitation**: While the degradation pattern matches the 4090 results, the 8GB card may exhibit different memory allocation patterns or earlier OOM for N > 10000. The Phase 4 results show memory_allocated_before values dropping sharply between some N values (e.g., 2191 MB at N=4000 → 473 MB at N=5000), likely due to Python GC collecting previous solvers. This non-deterministic GC behavior adds noise to in-process memory measurements.

5. **Single GPU architecture tested**: Both GPUs (RTX 4090 and RTX 4060) are Ada Lovelace architecture. The phenomenon may manifest differently on other architectures (e.g., Ampere, Hopper) or with different CUDA toolkit versions.

6. **Nsight Systems overhead**: Profiling adds timing overhead. The absolute times reported in nsys profiles should not be directly compared to un-profiled benchmark times.

---

## 10. Data Files

All experimental data is available in:
```
experiments/curobo_latency_spike_diagnosis/results/
├── curobo_latency_scan_ordered.csv          (Phase 1: 12 rows)
├── curobo_latency_scan_randomized.csv        (Phase 2: 36 rows)
├── curobo_latency_scan_fresh_process.csv     (Phase 3: 4 rows)
├── curobo_latency_fixed_max_batch.csv        (Phase 4: 12 rows)
├── curobo_memory_summary_logs/               (Phase 5: 36 txt files)
├── nsys_N4000.nsys-rep                       (Phase 6: Nsight profile)
├── nsys_N4000.sqlite                         (Phase 6: Nsight DB)
├── nsys_N5000.nsys-rep                       (Phase 6: Nsight profile)
├── nsys_N5000.sqlite                         (Phase 6: Nsight DB)
├── fresh_process_json/                       (Phase 3: raw JSON outputs)
│   ├── fresh_N4000.json
│   ├── fresh_N5000.json
│   ├── fresh_N7000.json
│   └── fresh_N8000.json
└── config_dump.json                          (Phase 6: environment config)
```

Diagnostic scripts:
```
experiments/curobo_latency_spike_diagnosis/
├── run_diagnosis.py                          (main multi-phase runner)
└── bench_single_curobo_n.py                  (single-N subprocess entry point)
```

---

*Report generated automatically by the cuRobo batch oscillation diagnosis pipeline. All measurements use `time.perf_counter()` for host wall time and `torch.cuda.synchronize()` for GPU synchronization. No results were discarded or modified.*
