#!/usr/bin/env python3
"""Fair comparison: cuRobo with K=16 seeds vs OPT4C with K=16 seeds.
Both use identical seed count — isolates solver algorithm quality from seed strategy."""
import csv, os, sys, time, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / '暂时修改' / '补充实验' / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / 'scripts'))
from audit_curobo_quality_round2 import (
    run_curobo, safe_float, STRICT_POS_M, STRICT_ROT_RAD,
    MEDIUM_POS_M, MEDIUM_ROT_RAD
)

def main():
    N_VALUES = [100, 500, 1000]
    K = 16  # Match our method's seed count

    print("=== cuRobo K=16 Fair Comparison ===")
    print(f"num_seeds={K}, seed_solver_num_seeds={K}")
    print(f"CUDA Graph=on, collision=off")
    print()

    summaries = []
    for N in N_VALUES:
        label = f"curobo_k16_N{N}"
        print(f"N={N}: running cuRobo with {K} seeds...", flush=True)
        t0 = time.perf_counter()

        try:
            result = run_curobo(
                N, label,
                num_seeds=K,
                seed_solver_num_seeds=K,
                use_cuda_graph=True,
                position_tolerance=MEDIUM_POS_M,
                orientation_tolerance=MEDIUM_ROT_RAD,
                repeat=3, warmup=1,
            )
            elapsed = time.perf_counter() - t0

            if result.status == 'ok':
                s = result.summary
                row = {
                    'N': N, 'num_seeds': K,
                    'gpu_ms': safe_float(s.get('gpu_ms')),
                    'throughput': safe_float(s.get('throughput')),
                    'strict_sr': safe_float(s.get('strict_sr')),
                    'pos_p95_all_mm': safe_float(s.get('pos_p95_all')),
                    'pos_p95_success_mm': safe_float(s.get('pos_p95_success_only')),
                    'pos_p95_failure_mm': safe_float(s.get('pos_p95_failure_only')),
                    'rot_p95_all_deg': safe_float(s.get('rot_p95_all')),
                    'elapsed_sec': round(elapsed, 1),
                }
                summaries.append(row)
                print(f"  OK: SR={row['strict_sr']:.4f}  tp={row['throughput']:.1f}  "
                      f"p95_all={row['pos_p95_all_mm']:.1f}mm  "
                      f"p95_suc={row['pos_p95_success_mm']:.1f}mm  ({elapsed:.1f}s)")
            else:
                print(f"  FAILED: {result.status}")

        except Exception as e:
            import traceback
            print(f"  EXCEPTION: {e}")
            traceback.print_exc()
        print()

    # Save cuRobo K=16 results
    if summaries:
        path = RESULTS / 'curobo_k16_summary.csv'
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
        print(f"cuRobo K=16 summary → {path}")

    # === COMPARISON: OPT4C K=16 vs cuRobo K=16 (FAIR!) ===
    print(f"\n{'='*90}")
    print(f"FAIR COMPARISON: OPT4C K=16 vs cuRobo K=16 (same seed count)")
    print(f"{'='*90}")
    print(f"{'N':>5}  {'OPT4C SR':>9}  {'cuRobo SR':>9}  {'OPT4C tp':>10}  {'cuRobo tp':>10}  "
          f"{'OPT4C p95':>9}  {'cuRobo p95':>9}  {'SR Δ(pp)':>9}  {'tp ratio':>9}")
    print("-" * 90)

    # Load OPT4C K=16 data
    k16_path = ROOT / '暂时修改' / 'results' / 'dense_static_summary.csv'
    opt4c_k16 = {}
    if k16_path.exists():
        with open(k16_path, newline='') as f:
            for row in csv.DictReader(f):
                opt4c_k16[int(row['N'])] = row

    # Load cuRobo K=1 data (for comparison)
    curobo_k1_path = ROOT / '暂时修改' / 'results' / 'dense_curobo_summary.csv'
    curobo_k1 = {}
    if curobo_k1_path.exists():
        with open(curobo_k1_path, newline='') as f:
            for row in csv.DictReader(f):
                curobo_k1[int(row['N'])] = row

    comparison_rows = []
    for cu_row in summaries:
        N = cu_row['N']
        o4 = opt4c_k16.get(N, {})
        cu1 = curobo_k1.get(N, {})

        o4_sr = float(o4.get('strict_sr', 0)) if o4 else 0
        o4_tp = float(o4.get('raw_throughput_mean', 0)) if o4 else 0
        o4_p95 = float(o4.get('pos_p95_all_mm', 0)) if o4 else 0

        cu16_sr = cu_row['strict_sr']
        cu16_tp = cu_row['throughput']
        cu16_p95 = cu_row['pos_p95_all_mm']
        cu16_p95_suc = cu_row['pos_p95_success_mm']

        sr_delta = (o4_sr - cu16_sr) * 100
        tp_ratio = o4_tp / cu16_tp if cu16_tp else 0

        print(f"{N:>5}  {o4_sr:9.4f}  {cu16_sr:9.4f}  {o4_tp:10.1f}  {cu16_tp:10.1f}  "
              f"{o4_p95:9.1f}  {cu16_p95:9.1f}  {sr_delta:9.1f}  {tp_ratio:9.2f}")

        comparison_rows.append({
            'N': N, 'method': 'both K=16',
            'opt4c_k16_sr': o4_sr, 'opt4c_k16_tp': o4_tp, 'opt4c_k16_p95': o4_p95,
            'curobo_k16_sr': cu16_sr, 'curobo_k16_tp': cu16_tp, 'curobo_k16_p95': cu16_p95,
            'curobo_k16_p95_suc': cu16_p95_suc,
            'sr_delta_pp': round(sr_delta, 1),
            'throughput_ratio': round(tp_ratio, 3),
        })

    # Save fair comparison
    comp_path = RESULTS / 'fair_comparison_k16_vs_k16.csv'
    with open(comp_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        w.writeheader()
        w.writerows(comparison_rows)
    print(f"\nFair comparison → {comp_path}")

    # === THREE-WAY COMPARISON ===
    print(f"\n{'='*100}")
    print(f"THREE-WAY: OPT4C K=16 vs OPT4C K=1 vs cuRobo K=16 vs cuRobo K=1")
    print(f"{'='*100}")
    print(f"{'N':>5}  {'O4-K16':>7}  {'O4-K1':>7}  {'cu-K16':>7}  {'cu-K1':>7}  "
          f"{'O4-K16':>9}  {'cu-K16':>9}  {'cu-K1':>9}  {'cu-K16':>9}  {'cu-K1':>9}")
    print(f"{'':5}  {'SR':>7}  {'SR':>7}  {'SR':>7}  {'SR':>7}  "
          f"{'tp':>9}  {'tp':>9}  {'tp':>9}  {'p95':>9}  {'p95':>9}")
    print("-" * 100)

    # Load OPT4C K=1 data
    k1_path = ROOT / '暂时修改' / '补充实验' / 'results' / 'k1_static_summary.csv'
    opt4c_k1 = {}
    if k1_path.exists():
        with open(k1_path, newline='') as f:
            for row in csv.DictReader(f):
                opt4c_k1[int(row['N'])] = row

    for N in N_VALUES:
        o16 = opt4c_k16.get(N, {})
        o1 = opt4c_k1.get(N, {})
        cu16 = next((r for r in summaries if r['N'] == N), {})
        cu1 = curobo_k1.get(N, {})

        o16_sr = float(o16.get('strict_sr', 0)) if o16 else 0
        o1_sr = float(o1.get('strict_sr', 0)) if o1 else 0
        cu16_sr = cu16.get('strict_sr', 0)
        cu1_sr = float(cu1.get('strict_sr', 0)) if cu1 else 0

        o16_tp = float(o16.get('raw_throughput_mean', 0)) if o16 else 0
        cu16_tp = cu16.get('throughput', 0)
        cu1_tp = float(cu1.get('throughput', 0)) if cu1 else 0

        cu16_p95 = cu16.get('pos_p95_all_mm', 0)
        cu1_p95 = float(cu1.get('pos_p95_all_mm', 0)) if cu1 else 0

        print(f"{N:>5}  {o16_sr:7.4f}  {o1_sr:7.4f}  {cu16_sr:7.4f}  {cu1_sr:7.4f}  "
              f"{o16_tp:9.1f}  {cu16_tp:9.1f}  {cu1_tp:9.1f}  {cu16_p95:9.1f}  {cu1_p95:9.1f}")

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
