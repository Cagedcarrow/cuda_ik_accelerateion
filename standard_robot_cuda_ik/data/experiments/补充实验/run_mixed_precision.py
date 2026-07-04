#!/usr/bin/env python3
"""FP32 mixed precision ablation: Jacobian/Hessian in FP32, linear solve in FP64."""
import subprocess, csv, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'build' / 'standard_robot_cuda_v4_runner'
INPUTS = ROOT / '暂时修改' / 'inputs'
RESULTS_DIR = ROOT / '暂时修改' / '补充实验' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

K, WARMUP, REPEAT, MAX_ITER = 16, 10, 30, 60
N_VALUES = [100, 500, 1000]

# Configs: (precision_mode, fallback_mode, label)
CONFIGS = [
    ('fp64',             'none',                'fp64_baseline'),
    ('mixed_safe',       'none',                'mixed_safe'),
    ('mixed_aggressive', 'none',                'mixed_aggressive'),
    ('mixed_safe',       'strict_fail_to_fp64', 'mixed_safe_fallback'),
]

def run(N, precision, fallback, label):
    target = INPUTS / f'targets_N{N}_T4x4_f64.raw'
    seed = INPUTS / f'seeds_N{N}_K16_q_f64.raw'
    summary_csv = RESULTS_DIR / f'mixed_{label}_N{N}_summary.csv'
    timing_csv = RESULTS_DIR / f'mixed_{label}_N{N}_timing.csv'

    cmd = [str(RUNNER), '--mode', 'v4_static',
           '--variant', 'opt4c_block_target', '--limit-gradient', 'analytic',
           '--graph-mode', 'off',
           '--precision-mode', precision, '--fallback-mode', fallback,
           '--targets', str(target), '--seeds', str(seed),
           '--N', str(N), '--K', str(K),
           '--max-iter', str(MAX_ITER), '--repeat', str(REPEAT), '--warmup', str(WARMUP),
           '--summary-csv', str(summary_csv), '--timing-csv', str(timing_csv)]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        return None, dt, r.stderr[-200:]
    if summary_csv.exists():
        with open(summary_csv, newline='') as f:
            rows = list(csv.DictReader(f))
            return (rows[0] if rows else None), dt, None
    return None, dt, "no output"

def main():
    # Load FP64 baseline from existing data
    fp64_data = {}
    fp64_path = ROOT / '暂时修改' / 'results' / 'dense_static_summary.csv'
    if fp64_path.exists():
        with open(fp64_path, newline='') as f:
            for row in csv.DictReader(f):
                fp64_data[int(row['N'])] = row

    print("=" * 90)
    print("FP32 Mixed Precision Benchmark")
    print("=" * 90)

    results = {}
    for precision, fallback, label in CONFIGS:
        for N in N_VALUES:
            key = f"{label}_N{N}"
            print(f"\n[{precision}/{fallback}] N={N}...", flush=True)
            row, dt, err = run(N, precision, fallback, label)
            if row:
                sr = float(row['strict_sr']); tp = float(row['raw_throughput_mean'])
                p95 = float(row['pos_p95_all_mm']); gpu = float(row['gpu_stream_ms_mean'])
                fb = float(row.get('fallback_rate', 0))
                results[key] = {'N': N, 'precision': precision, 'fallback': fallback,
                    'sr': sr, 'tp': tp, 'p95': p95, 'gpu_ms': gpu, 'fallback_rate': fb}
                print(f"  OK: SR={sr:.4f} tp={tp:.1f} p95={p95:.1f}mm GPU={gpu:.3f}ms "
                      f"fb={fb:.3f} ({dt:.1f}s)")
            else:
                print(f"  FAIL: {err[:100] if err else 'unknown'}")

    # Comparison table
    print("\n" + "=" * 90)
    print("COMPARISON: FP64 vs FP32 Mixed Precision (K=16)")
    print("=" * 90)
    print(f"{'N':>5} | {'Config':<25} | {'SR':>7} | {'Throughput':>10} | {'p95':>7} | {'GPU ms':>8} | {'FB rate':>7} | {'vs FP64':>7}")
    print("-" * 96)

    for precision, fallback, label in CONFIGS:
        for N in N_VALUES:
            key = f"{label}_N{N}"
            r = results.get(key)
            if not r: continue
            fp64 = fp64_data.get(N, {})
            fp64_tp = float(fp64.get('raw_throughput_mean', r['tp']))
            speedup = r['tp'] / fp64_tp if fp64_tp > 0 else 1
            cfg_name = f"{precision}/{fallback}"
            print(f"{N:>5} | {cfg_name:<25} | {r['sr']:7.4f} | {r['tp']:10.1f} | "
                  f"{r['p95']:7.1f} | {r['gpu_ms']:8.3f} | {r['fallback_rate']:7.3f} | "
                  f"{speedup:6.2f}x")

    # Write summary CSV
    csv_path = RESULTS_DIR / 'mixed_precision_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        fields = ['N','precision','fallback','strict_sr','throughput','p95_mm','gpu_ms',
                  'fallback_rate','speedup_vs_fp64']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key, r in sorted(results.items()):
            fp64 = fp64_data.get(r['N'], {})
            fp64_tp = float(fp64.get('raw_throughput_mean', r['tp']))
            w.writerow({**r, 'speedup_vs_fp64': round(r['tp']/fp64_tp, 2)})
    print(f"\nSummary → {csv_path}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
