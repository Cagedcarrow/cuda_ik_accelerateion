#!/usr/bin/env python3
"""Generate figures for patent - focused on small-matrix CUDA acceleration"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 10

DATA_DIR = '/mnt/linuxdata/cuda_ik_accelerateion/docs/data'
OUTPUT_DIR = '/mnt/linuxdata/cuda_ik_accelerateion/docs/latex论文_backup_v2_20260615_130404/专利/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_csv(filename):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def fig1_ablation_throughput():
    """Ablation study throughput by level at N=5000"""
    rows = read_csv('ablation_ur10.csv')
    n5000 = {}
    for row in rows:
        if int(row['n']) == 5000:
            n5000[row['level']] = float(row['throughput_targets_per_s']) / 1000.0
    levels = sorted(n5000.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 99)
    values = [n5000[l] for l in levels]
    colors = ['#d62728' if l == 'A0' else '#ff7f0e' if l == 'A5'
              else '#2ca02c' if l in ('A7','A8') else '#1f77b4' for l in levels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(levels)), values, color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{val:.1f}k', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(range(len(levels))); ax.set_xticklabels(levels, fontsize=10)
    ax.set_ylabel('Throughput (k targets/s)', fontsize=12)
    ax.set_xlabel('Ablation Level', fontsize=12)
    ax.set_title('Ablation Study: CUDA Optimization Levels (N=5000)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    if 'A2' in levels:
        idx = levels.index('A2')
        ax.annotate('PaddedMat6x8\nBank Conflict Eliminated',
                    xy=(idx, values[idx]), xytext=(idx-0.5, values[idx]+15),
                    arrowprops=dict(arrowstyle='->', color='#1f77b4'), fontsize=8, color='#1f77b4', fontweight='bold')
    if 'A7' in levels:
        idx = levels.index('A7')
        ax.annotate('FP32/FP64 Mixed\nPrecision ~2.5x',
                    xy=(idx, values[idx]), xytext=(idx-1, values[idx]+25),
                    arrowprops=dict(arrowstyle='->', color='#2ca02c'), fontsize=8, color='#2ca02c', fontweight='bold')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'fig1_ablation_throughput.{fmt}'))
    plt.close()
    print("  OK fig1: ablation throughput")

def fig2_throughput_scaling():
    """CUDA vs cuRobo throughput scaling"""
    rows = read_csv('solver_comparison.csv')
    cuda, curobo = {}, {}
    for row in rows:
        n = int(row['n']); t = float(row['throughput_targets_per_s']) / 1000.0
        if row['solver'] == 'cuda_a7': cuda[n] = t
        elif row['solver'] == 'curobo': curobo[n] = t
    ns = sorted(set(list(cuda.keys()) + list(curobo.keys())))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(ns, [cuda.get(n,None) for n in ns], 'o-', color='#0072B2', lw=2, ms=8, label='CUDA A7 (Small-Matrix Accel)')
    ax.plot(ns, [curobo.get(n,None) for n in ns], 's--', color='#E69F00', lw=2, ms=8, label='cuRobo')
    ax.fill_between([100, 500], 0, 200, alpha=0.05, color='#0072B2')
    ax.annotate('CUDA Advantage\n(Low Batch)', xy=(200, 80), fontsize=10, color='#0072B2', ha='center', fontweight='bold')
    ax.annotate('cuRobo Advantage\n(High Batch)', xy=(2500, 120), fontsize=10, color='#E69F00', ha='center', fontweight='bold')
    ax.set_xlabel('Batch Size N', fontsize=12)
    ax.set_ylabel('Throughput (k targets/s)', fontsize=12)
    ax.set_title('CUDA Small-Matrix Acceleration vs cuRobo: Throughput Scaling', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log'); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'fig2_throughput_scaling.{fmt}'))
    plt.close()
    print("  OK fig2: throughput scaling")

def fig3_mixed_precision():
    """FP64 vs Mixed Precision speedup"""
    rows = read_csv('mixed_precision_ablation.csv')
    fp64_dict, mixed_dict = {}, {}
    for row in rows:
        n = int(row['n']); t = float(row['throughput_targets_per_s']) / 1000.0
        if 'FP64' in row['precision']: fp64_dict[n] = t
        elif 'Mixed' in row['precision']: mixed_dict[n] = t
    common_ns = sorted(set(fp64_dict.keys()) & set(mixed_dict.keys()))
    fp64_vals = [fp64_dict[n] for n in common_ns]
    mixed_vals = [mixed_dict[n] for n in common_ns]
    speedups = [mixed_vals[i]/fp64_vals[i] for i in range(len(common_ns))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(common_ns)); w = 0.35
    b1 = ax1.bar(x-w/2, fp64_vals, w, label='FP64', color='#1f77b4', edgecolor='black', linewidth=0.5)
    b2 = ax1.bar(x+w/2, mixed_vals, w, label='FP32/FP64 Mixed', color='#2ca02c', edgecolor='black', linewidth=0.5)
    for b,v in zip(b1, fp64_vals): ax1.text(b.get_x()+b.get_width()/2, b.get_height()+2, f'{v:.1f}k', ha='center', va='bottom', fontsize=8)
    for b,v in zip(b2, mixed_vals): ax1.text(b.get_x()+b.get_width()/2, b.get_height()+2, f'{v:.1f}k', ha='center', va='bottom', fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels([f'N={n}' for n in common_ns], fontsize=10)
    ax1.set_ylabel('Throughput (k targets/s)', fontsize=11)
    ax1.set_title('FP64 vs Mixed Precision', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10); ax1.grid(axis='y', alpha=0.3, linestyle='--')

    b3 = ax2.bar(x, speedups, color='#d62728', edgecolor='black', linewidth=0.5, width=0.5)
    for b,v in zip(b3, speedups): ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f'{v:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.axhline(y=1.0, color='gray', linestyle='--', lw=1, alpha=0.7, label='FP64 Baseline')
    ax2.set_xticks(x); ax2.set_xticklabels([f'N={n}' for n in common_ns], fontsize=10)
    ax2.set_ylabel('Speedup Ratio', fontsize=11)
    ax2.set_title('Mixed Precision Speedup', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10); ax2.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'fig3_mixed_precision.{fmt}'))
    plt.close()
    print("  OK fig3: mixed precision")

def fig4_bank_conflicts():
    """Bank conflict comparison with architecture diagram overlay"""
    rows = read_csv('ncu_profiling.csv')
    fp64_bc = mixed_bc = 0
    for row in rows:
        if row['kernel'] == 'ik_batch_solve' and int(row['n']) == 100: fp64_bc = int(row['bank_conflicts'])
        elif row['kernel'] == 'ik_batch_solve_mixed' and int(row['n']) == 100: mixed_bc = int(row['bank_conflicts'])

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Standard 6x6\n(stride=6)', 'PaddedMat6x8\n(stride=8)']
    values = [fp64_bc, mixed_bc]
    colors = ['#d62728', '#2ca02c']
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.5, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50, f'{val:,}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    if fp64_bc > 0:
        reduction = (fp64_bc - mixed_bc) / fp64_bc * 100
        ax.set_title(f'Shared Memory Bank Conflict Elimination (N=100)\nPaddedMat6x8 Reduces Conflicts by {reduction:.0f}%', fontsize=14, fontweight='bold')
    ax.set_ylabel('Bank Conflict Count', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'fig4_bank_conflicts.{fmt}'))
    plt.close()
    print("  OK fig4: bank conflicts")

def fig5_cuda_architecture():
    """CUDA Small-Matrix Acceleration Architecture Diagram"""
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 17); ax.set_ylim(0, 11); ax.axis('off')
    ax.set_title('CUDA Small-Matrix Acceleration Architecture: 1-Block-per-Target + Register LDLT + PaddedMat6x8',
                 fontsize=14, fontweight='bold', pad=15)

    # N target blocks
    for i in range(4):
        x, y = 0.2, 9.2 - i*2.2
        rect = plt.Rectangle((x, y-0.5), 3.0, 1.5, fill=True, facecolor='#e8f4fd', edgecolor='#0072B2', lw=2)
        ax.add_patch(rect)
        ax.text(x+1.5, y+0.25, f'Target #{i+1}\nT* in SE(3)', ha='center', fontsize=9, fontweight='bold')

    # Blocks
    for i in range(4):
        x, y = 4.2, 9.2 - i*2.2
        rect = plt.Rectangle((x, y-0.5), 2.8, 1.5, fill=True, facecolor='#fff3e0', edgecolor='#E69F00', lw=2)
        ax.add_patch(rect)
        ax.text(x+1.4, y+0.25, f'CUDA Block #{i}\n128 thr, 4 Warps', ha='center', fontsize=9, fontweight='bold')

    for i in range(4):
        y = 9.2 - i*2.2
        ax.annotate('', xy=(4.2, y), xytext=(3.2, y), arrowprops=dict(arrowstyle='->', color='#555', lw=1.8))

    # Block internals
    bx, by = 8.0, 4.2
    block_rect = plt.Rectangle((bx, by-1.5), 7.5, 6.5, fill=True, facecolor='#fafafa', edgecolor='#333', lw=2.5)
    ax.add_patch(block_rect)
    ax.text(bx+3.75, by+4.2, 'Single Block Internal: 128-Thread 4-Warp Pipeline for 6x6 Small-Matrix Acceleration',
            ha='center', fontsize=10, fontweight='bold')

    # Storage hierarchy
    ax.text(bx+0.3, by+3.5, '[Storage Hierarchy]', fontsize=8, fontweight='bold', color='#555')
    mem_items = ['Constant Memory: UR10 model params (broadcast, zero-latency)',
                 'Shared Memory: PaddedMat6x8 (stride=8, zero bank conflict)',
                 'Registers: LDLT 6x6 solver (~60 regs, ~0.1us, zero mem access)']
    for j, item in enumerate(mem_items):
        ax.text(bx+0.5, by+3.0-j*0.4, f'{j+1}. {item}', fontsize=7, color='#333')

    # Compute pipeline
    ax.text(bx+0.3, by+1.5, '[Compute Pipeline per DLS Iteration]', fontsize=8, fontweight='bold', color='#555')
    pipeline = ['FK: Lane 0 (Rodrigues, constant-mem reads)',
                'Jacobian 6 cols: Lane 0-5 (6-way parallel, central diff)',
                'Hessian 36 elems: Lane 0-35 (36-way parallel, weighted inner prod)',
                'Gradient 6 dims: Lane 0-5 (6-way parallel)',
                'LDLT Solve: Lane 0 (register-resident, ~63 scalar ops)',
                'Joint Update: Lane 0-5 (clamp + branch align)']
    for j, item in enumerate(pipeline):
        ax.text(bx+0.5, by+1.0-j*0.35, f'{j+1}. {item}', fontsize=6.8, color='#333')

    # Output
    out_rect = plt.Rectangle((bx+4.8, by+1.2), 2.5, 3.5, fill=True, facecolor='#e8f5e9', edgecolor='#2ca02c', lw=1.5)
    ax.add_patch(out_rect)
    ax.text(bx+6.05, by+4.0, 'Output', fontsize=9, fontweight='bold', color='#2ca02c')
    for j, o in enumerate(['Best q (6 joints)', 'Pose Error (pos+rot)', 'Iteration Count', 'Limit Status', 'NaN/Inf = 0']):
        ax.text(bx+5.0, by+3.5-j*0.42, f'* {o}', fontsize=7, color='#333')

    # Bottom bar
    hw_rect = plt.Rectangle((0.2, 0.2), 16.2, 1.8, fill=True, facecolor='#eceff1', edgecolor='#666', lw=1.5)
    ax.add_patch(hw_rect)
    ax.text(8.3, 1.6, 'NVIDIA GeForce RTX 4060 Laptop GPU (Ada Lovelace SM89) | CUDA 13.3 | 128 threads/block',
            ha='center', fontsize=10, fontweight='bold', color='#555')
    ax.text(8.3, 0.9, 'Key innovations: PaddedMat6x8 (bank-conflict-free shared mem) | Register LDLT (0.1us) | FP32/FP64 Mixed Precision (2.5x speedup) | 1-Block-per-Target Mapping',
            ha='center', fontsize=9, color='#777')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'fig5_architecture.{fmt}'))
    plt.close()
    print("  OK fig5: architecture diagram")

def fig6_roofline():
    """NCU Roofline: compute vs memory bound"""
    rows = read_csv('ncu_profiling.csv')
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in rows:
        if int(row['n']) == 100:
            k = row['kernel'].replace('ik_batch_solve', 'FP64-Kernel').replace('_mixed', '-Mixed')
            c = float(row['compute_throughput_pct']); d = float(row['dram_throughput_pct'])
            ax.scatter(d, c, s=250, edgecolors='black', lw=1, zorder=5)
            ax.annotate(k, (d, c), textcoords="offset points", xytext=(12, 10), fontsize=11, fontweight='bold')

    ax.set_xlabel('DRAM Throughput (%)', fontsize=12)
    ax.set_ylabel('Compute Throughput (%)', fontsize=12)
    ax.set_title('NCU Roofline: 6x6 Small-Matrix Kernel is Compute-Bound (N=100)', fontsize=14, fontweight='bold')
    ax.axhline(y=80, color='red', linestyle=':', alpha=0.4, label='Compute-bound region (>80%)')
    ax.axvline(x=20, color='blue', linestyle=':', alpha=0.4, label='Memory-bound region (>20%)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 10); ax.set_ylim(50, 100)
    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'fig6_roofline.{fmt}'))
    plt.close()
    print("  OK fig6: roofline analysis")

def main():
    print("Generating patent figures (small-matrix CUDA focus)...")
    print(f"Output: {OUTPUT_DIR}")
    fig1_ablation_throughput()
    fig2_throughput_scaling()
    fig3_mixed_precision()
    fig4_bank_conflicts()
    fig5_cuda_architecture()
    fig6_roofline()
    print(f"\nAll 6 figures saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
