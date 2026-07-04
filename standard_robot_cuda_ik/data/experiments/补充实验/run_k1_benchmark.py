#!/usr/bin/env python3
"""Run OPT4C with K=1 (single seed) and compare vs cuRobo default (also 1 seed).
This isolates the solver algorithm quality independent of seed count."""
import subprocess, csv, os, sys, time, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / 'build'
RUNNER = BUILD / 'standard_robot_cuda_v4_runner'
INPUTS = ROOT / '暂时修改' / '补充实验' / 'inputs'
RESULTS = ROOT / '暂时修改' / '补充实验' / 'results'
INPUTS_SRC = ROOT / '暂时修改' / 'inputs'  # existing N=100-1000 target/seed files

RESULTS.mkdir(parents=True, exist_ok=True)
INPUTS.mkdir(parents=True, exist_ok=True)

WARMUP = 10
REPEAT = 30
MAX_ITER = 60
N_VALUES = list(range(100, 1001, 100))

def generate_k1_seeds():
    """Take first seed of each target from K=16 files → K=1 files."""
    for N in N_VALUES:
        src = INPUTS_SRC / f'seeds_N{N}_K16_q_f64.raw'
        dst = INPUTS / f'seeds_N{N}_K1_q_f64.raw'
        if dst.exists():
            continue
        seeds16 = np.fromfile(src, dtype=np.float64).reshape(N, 16, 6)
        seeds1 = seeds16[:, 0:1, :].copy().reshape(-1)
        seeds1.astype(np.float64).tofile(dst)
        print(f"  K=1 seeds N={N}: {dst} ({dst.stat().st_size} bytes)")
    # Copy targets (same file, just symlink or copy)
    for N in N_VALUES:
        src = INPUTS_SRC / f'targets_N{N}_T4x4_f64.raw'
        dst = INPUTS / f'targets_N{N}_T4x4_f64.raw'
        if not dst.exists():
            dst.write_bytes(src.read_bytes())

def run_one(N):
    target_file = INPUTS / f'targets_N{N}_T4x4_f64.raw'
    seed_file = INPUTS / f'seeds_N{N}_K1_q_f64.raw'
    best_csv = RESULTS / f'cuda_opt4c_k1_best_N{N}.csv'
    summary_csv = RESULTS / f'cuda_opt4c_k1_summary_N{N}.csv'
    timing_csv = RESULTS / f'cuda_opt4c_k1_timing_N{N}.csv'

    cmd = [
        str(RUNNER), '--mode', 'v4_static',
        '--variant', 'opt4c_block_target',
        '--limit-gradient', 'analytic',
        '--graph-mode', 'off',
        '--precision-mode', 'fp64',
        '--fallback-mode', 'none',
        '--targets', str(target_file),
        '--seeds', str(seed_file),
        '--N', str(N), '--K', '1',
        '--max-iter', str(MAX_ITER),
        '--repeat', str(REPEAT), '--warmup', str(WARMUP),
        '--best-csv', str(best_csv),
        '--summary-csv', str(summary_csv),
        '--timing-csv', str(timing_csv),
    ]
    print(f"  N={N} K=1: running...", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"    FAILED: {result.stderr[-200:]}")
        return None
    print(f"    done in {elapsed:.1f}s", flush=True)
    if summary_csv.exists():
        with open(summary_csv, newline='') as f:
            rows = list(csv.DictReader(f))
            return rows[0] if rows else None
    return None

def main():
    print("=== Step 1: Generate K=1 seed files ===")
    generate_k1_seeds()

    print("\n=== Step 2: Run CUDA OPT4C K=1 benchmarks ===")
    summaries = []
    for N in N_VALUES:
        row = run_one(N)
        if row:
            summaries.append(row)
            sr = float(row.get('strict_sr', 0))
            tp = float(row.get('raw_throughput_mean', 0))
            p95 = float(row.get('pos_p95_all_mm', 0))
            gpu = float(row.get('gpu_stream_ms_mean', 0))
            print(f"    SR={sr:.4f}  tp={tp:.1f}  p95={p95:.1f}mm  GPU={gpu:.3f}ms")
        print()

    if summaries:
        combined = RESULTS / 'k1_static_summary.csv'
        with open(combined, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
        print(f"\nK=1 summary → {combined}")

    # === Step 3: Compare vs cuRobo (both single-seed!) ===
    curobo_path = ROOT / '暂时修改' / 'results' / 'dense_curobo_summary.csv'
    print(f"\n=== Step 3: K=1 vs cuRobo (both single-seed) ===")
    print(f"{'N':>5}  {'K=1 SR':>8}  {'K=1 tp':>10}  {'cuRobo SR':>10}  {'cuRobo tp':>12}  {'SR Δ(pp)':>9}  {'tp ratio':>8}")
    print("-" * 80)

    curobo = {}
    if curobo_path.exists():
        with open(curobo_path, newline='') as f:
            for row in csv.DictReader(f):
                curobo[int(row['N'])] = row

    for row in summaries:
        N = int(row['N'])
        k1_sr = float(row['strict_sr'])
        k1_tp = float(row['raw_throughput_mean'])
        k1_p95 = float(row['pos_p95_all_mm'])

        cu = curobo.get(N, {})
        cu_sr = float(cu.get('strict_sr', 0)) if cu else 0
        cu_tp = float(cu.get('throughput', 0)) if cu else 0
        cu_p95_suc = float(cu.get('pos_p95_success_only_mm', 0)) if cu else 0

        sr_delta = (k1_sr - cu_sr) * 100
        tp_ratio = k1_tp / cu_tp if cu_tp else 0

        print(f"{N:>5}  {k1_sr:8.4f}  {k1_tp:10.1f}  {cu_sr:10.4f}  {cu_tp:12.1f}  {sr_delta:9.1f}  {tp_ratio:8.2f}")

    # Write comparison CSV
    comp_path = RESULTS / 'k1_vs_curobo_comparison.csv'
    with open(comp_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['N','k1_strict_sr','k1_throughput','k1_p95_mm',
            'k1_gpu_ms','curobo_strict_sr','curobo_throughput','curobo_p95_suc_mm',
            'sr_delta_pp','throughput_ratio'])
        w.writeheader()
        for row in summaries:
            N = int(row['N'])
            cu = curobo.get(N, {})
            w.writerow({
                'N': N,
                'k1_strict_sr': float(row['strict_sr']),
                'k1_throughput': float(row['raw_throughput_mean']),
                'k1_p95_mm': float(row['pos_p95_all_mm']),
                'k1_gpu_ms': float(row['gpu_stream_ms_mean']),
                'curobo_strict_sr': float(cu.get('strict_sr', 0)) if cu else '',
                'curobo_throughput': float(cu.get('throughput', 0)) if cu else '',
                'curobo_p95_suc_mm': float(cu.get('pos_p95_success_only_mm', 0)) if cu else '',
                'sr_delta_pp': round((float(row['strict_sr']) - float(cu.get('strict_sr', 0))) * 100, 1) if cu else '',
                'throughput_ratio': round(float(row['raw_throughput_mean']) / float(cu.get('throughput', 1)), 3) if cu else '',
            })
    print(f"\nComparison CSV → {comp_path}")

    # Also show K=16 vs K=1 comparison
    k16_path = ROOT / '暂时修改' / 'results' / 'dense_static_summary.csv'
    if k16_path.exists():
        k16 = {}
        with open(k16_path, newline='') as f:
            for row in csv.DictReader(f):
                k16[int(row['N'])] = row

        print(f"\n=== Bonus: K=16 vs K=1 (same solver, different seeds) ===")
        print(f"{'N':>5}  {'K=16 SR':>8}  {'K=1 SR':>8}  {'K=16 tp':>10}  {'K=1 tp':>10}  {'K=16 p95':>9}  {'K=1 p95':>9}")
        print("-" * 70)
        for row in summaries:
            N = int(row['N'])
            k16_row = k16.get(N, {})
            k16_sr = float(k16_row.get('strict_sr', 0)) if k16_row else 0
            k16_tp = float(k16_row.get('raw_throughput_mean', 0)) if k16_row else 0
            k16_p95 = float(k16_row.get('pos_p95_all_mm', 0)) if k16_row else 0
            k1_sr = float(row['strict_sr'])
            k1_tp = float(row['raw_throughput_mean'])
            k1_p95 = float(row['pos_p95_all_mm'])
            print(f"{N:>5}  {k16_sr:8.4f}  {k1_sr:8.4f}  {k16_tp:10.1f}  {k1_tp:10.1f}  {k16_p95:9.2f}  {k1_p95:9.2f}")

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
