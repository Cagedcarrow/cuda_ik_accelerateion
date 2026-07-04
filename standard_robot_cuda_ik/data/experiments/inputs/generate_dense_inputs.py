#!/usr/bin/env python3
"""Slice N=200-900 target/seed raw files from N=1000 base files."""
import numpy as np
import os, sys

INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cuda_inputs')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load base N=1000 files
targets_1000 = np.fromfile(os.path.join(INPUT_DIR, 'targets_N1000_T4x4_f64.raw'), dtype=np.float64)
seeds_1000 = np.fromfile(os.path.join(INPUT_DIR, 'seeds_N1000_K16_q_f64.raw'), dtype=np.float64)

# Verify sizes
assert targets_1000.size == 1000 * 16, f"targets size mismatch: {targets_1000.size}"
assert seeds_1000.size == 1000 * 16 * 6, f"seeds size mismatch: {seeds_1000.size}"

targets_1000 = targets_1000.reshape(1000, 16)
seeds_1000 = seeds_1000.reshape(1000, 16, 6)

K = 16

# Generate all N values in 100-1000
for N in range(100, 1001, 100):
    target_path = os.path.join(OUTPUT_DIR, f'targets_N{N}_T4x4_f64.raw')
    seed_path = os.path.join(OUTPUT_DIR, f'seeds_N{N}_K{K}_q_f64.raw')

    # Slice first N
    t = targets_1000[:N].reshape(-1).astype(np.float64)
    s = seeds_1000[:N].reshape(-1).astype(np.float64)

    if os.path.exists(target_path):
        print(f"Skip existing: {os.path.basename(target_path)}")
    else:
        t.tofile(target_path)
        print(f"Created: {os.path.basename(target_path)} ({t.nbytes} bytes)")

    if os.path.exists(seed_path):
        print(f"Skip existing: {os.path.basename(seed_path)}")
    else:
        s.tofile(seed_path)
        print(f"Created: {os.path.basename(seed_path)} ({s.nbytes} bytes)")

# Verify byte sizes
print("\nVerification:")
for N in range(100, 1001, 100):
    tp = os.path.join(OUTPUT_DIR, f'targets_N{N}_T4x4_f64.raw')
    sp = os.path.join(OUTPUT_DIR, f'seeds_N{N}_K{K}_q_f64.raw')
    expected_t = N * 16 * 8
    expected_s = N * K * 6 * 8
    actual_t = os.path.getsize(tp)
    actual_s = os.path.getsize(sp)
    t_ok = "✓" if actual_t == expected_t else "✗"
    s_ok = "✓" if actual_s == expected_s else "✗"
    print(f"  N={N:4d}: targets {actual_t:8d}/{expected_t:8d} {t_ok}  seeds {actual_s:8d}/{expected_s:8d} {s_ok}")

print("\nDone. All files ready for benchmark.")
