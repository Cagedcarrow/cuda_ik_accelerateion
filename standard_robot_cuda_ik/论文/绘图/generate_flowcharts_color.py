#!/usr/bin/env python3
"""Regenerate fig6 (thread architecture) and fig7 (algorithm flowchart) in COLOR.
These are flowcharts - keep original color scheme for visual clarity."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Noto Sans CJK SC', 'AR PL UKai CN', 'DejaVu Sans'],
    'font.size': 7.5, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

OUTDIR = os.path.dirname(os.path.abspath(__file__))
fig_width_full_in = 7.0

# Original color scheme
CUDA_COLOR = '#0072B2'
LIGHT_BLUE = '#56B4E9'
GREEN = '#009E73'
ORANGE = '#D55E00'
YELLOW = '#F0E442'
TEXT_COLOR = '#333333'
GRAY_BG = '#F0F8FF'

def save(fig, name):
    for fmt in ['pdf', 'svg']:
        path = os.path.join(OUTDIR, f'{name}.{fmt}')
        fig.savefig(path, format=fmt, bbox_inches='tight', pad_inches=0.05)
    print(f"  Saved: {name}.pdf, {name}.svg")

def draw_box(ax, x, y, w, h, text, color, fs=8, tc='white', bold=False):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor='#333333', linewidth=0.8, alpha=0.9)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=tc,
            weight='bold' if bold else 'normal')

# ====== Figure 6: Thread Architecture (COLOR) ======
print("Generating fig6 (thread architecture) in color...")
fig6, ax6 = plt.subplots(figsize=(fig_width_full_in, 4.5))
ax6.set_xlim(0, 10); ax6.set_ylim(0, 10); ax6.axis('off')

ax6.text(0.5, 9.5, 'CUDA Grid: N blocks', fontsize=9, fontweight='bold', color=TEXT_COLOR)
for i in range(8):
    draw_box(ax6, 1.0+i*1.1, 8.2, 0.9, 1.2, f'Block {i}', CUDA_COLOR, fs=6)
ax6.text(9.2, 8.2, '...', fontsize=10, ha='center', va='center')
ax6.annotate('', xy=(5, 8.8), xytext=(0.5, 9.2),
             arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0))
ax6.text(0.5, 7.0, 'Block 0 内部 (32 threads, K=16 seeds):', fontsize=9, fontweight='bold', color=TEXT_COLOR)

block_rect = FancyBboxPatch((0.3, 3.5), 9.4, 3.2, boxstyle="round,pad=0.15",
                            facecolor=GRAY_BG, edgecolor=CUDA_COLOR, linewidth=1.5, alpha=0.3)
ax6.add_patch(block_rect)
for i in range(16):
    draw_box(ax6, 0.7+i*0.55, 5.8, 0.45, 0.7, f'T{i}', LIGHT_BLUE, fs=5)
ax6.text(4.8, 5.2, '16 lanes × 1 seed/lane = K=16 Sobol seeds', fontsize=7, ha='center', color=TEXT_COLOR)
draw_box(ax6, 5.0, 4.1, 8.5, 0.7, 'Shared Memory: s_cand[16 × 16] doubles', GREEN, fs=7, bold=True)
for i in [2, 8, 14]:
    ax6.annotate('', xy=(0.7+i*0.55, 4.5), xytext=(0.7+i*0.55, 5.4),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=0.6))
draw_box(ax6, 5.0, 3.2, 4.0, 0.55, 'Lane 0: In-Block Best Selection', ORANGE, fs=7, bold=True)
ax6.annotate('', xy=(5.0, 3.5), xytext=(5.0, 3.75),
            arrowprops=dict(arrowstyle='->', color='#888888', lw=1.0))
draw_box(ax6, 5.0, 2.4, 5.0, 0.55, 'best[q,pos,rot,cost,rank] × N targets', YELLOW, fs=7, tc='#333333')
ax6.annotate('', xy=(5.0, 2.6), xytext=(5.0, 2.9),
            arrowprops=dict(arrowstyle='->', color='#888888', lw=1.0))

legend_items = [(CUDA_COLOR,'CUDA Block'),(LIGHT_BLUE,'Thread Lane'),(GREEN,'Shared Memory'),(ORANGE,'Selection')]
for i,(c,l) in enumerate(legend_items):
    ax6.add_patch(plt.Rectangle((0.3+i*2.4, 1.5), 0.3, 0.3, facecolor=c, edgecolor='#333333', lw=0.5))
    ax6.text(0.7+i*2.4, 1.65, l, fontsize=6.5, va='center', color=TEXT_COLOR)
save(fig6, 'fig6_thread_architecture'); plt.close()

# ====== Figure 7: Algorithm Flowchart (COLOR) ======
print("Generating fig7 (algorithm flowchart) in color...")
fig7, ax7 = plt.subplots(figsize=(fig_width_full_in, 5.5))
ax7.set_xlim(0, 10); ax7.set_ylim(0, 14); ax7.axis('off')

def draw_node(ax, x, y, w, h, text, color, fs=7, tc='white', shape='round', bold=False):
    weight = 'bold' if bold else 'normal'
    if shape == 'diamond':
        diamond = np.array([[x, y+h/2], [x+w/2, y], [x, y-h/2], [x-w/2, y]])
        ax.fill(diamond[:,0], diamond[:,1], facecolor=color, edgecolor='#333333', lw=0.8, alpha=0.9)
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=tc, weight=weight)
    else:
        box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.06",
                             facecolor=color, edgecolor='#333333', lw=0.8, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=tc, weight=weight)

def arrow(ax, x1, y1, x2, y2, label=''):
    ax.annotate(label, xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2),
                fontsize=6, ha='center', va='bottom', color='#555555')

draw_node(ax7, 5, 13.5, 3.0, 0.6, '输入: N个目标位姿 T*, K=16 Sobol种子', GREEN, fs=7)
draw_node(ax7, 5, 12.3, 3.5, 0.6, 'CUDA Grid Launch: N blocks, 32 threads/block', CUDA_COLOR, fs=7)
arrow(ax7, 5, 12.0, 5, 11.5)
draw_node(ax7, 5, 11.0, 4.0, 0.7, 'Block i: 加载 T*_i 和 seed[0..15] 到寄存器', LIGHT_BLUE, fs=7)
arrow(ax7, 5, 10.6, 5, 10.2)
draw_node(ax7, 5, 9.7, 4.5, 0.7, '16 lanes 并行: 每个lane执行LM迭代求解', CUDA_COLOR, fs=7, bold=True)
arrow(ax7, 5, 9.3, 5, 9.0)
draw_node(ax7, 2.2, 8.1, 3.5, 1.2,
    'LM迭代 (max 60):\n1. FK with frames → T,p,z\n2. 解析Jacobian: Jv=z×(p-p)\n3. H=JTJ+λI, g=JTe\n4. 6×6 Gauss Elim\n5. λ自适应缩放', CUDA_COLOR, fs=5.5)
draw_node(ax7, 7.8, 8.1, 3.5, 1.2,
    'Loss评估:\n- Pose cost=|ep|²+|eR|²\n- Limit barrier penalty\n- Total=0.5·pose+w·barrier\n- Trial accept: q←q+dq', CUDA_COLOR, fs=5.5)
arrow(ax7, 5, 9.4, 2.2, 8.7); arrow(ax7, 5, 9.4, 7.8, 8.7)
arrow(ax7, 2.2, 7.5, 5, 6.8); arrow(ax7, 7.8, 7.5, 5, 6.8)
draw_node(ax7, 5, 6.3, 4.5, 0.6, '收敛? (pos<5mm & rot<1°) 或 iter=max?', '#E69F00', fs=6.5, shape='diamond')
arrow(ax7, 8.0, 6.3, 9.0, 6.3, 'No')
draw_node(ax7, 9.5, 6.3, 1.0, 0.5, '继续', '#CCCCCC', fs=5)
arrow(ax7, 5, 5.9, 5, 5.0, 'Yes')
draw_node(ax7, 5, 4.5, 4.5, 0.6, '16个候选解写入Shared Memory s_cand[]', LIGHT_BLUE, fs=7)
arrow(ax7, 5, 4.2, 5, 3.8)
draw_node(ax7, 5, 3.2, 4.5, 0.8, 'Lane 0: 块内候选选择\nSuccess Rank→Near-Limit→Pose Cost', CUDA_COLOR, fs=7, bold=True)
arrow(ax7, 5, 2.8, 5, 2.4)
draw_node(ax7, 5, 1.8, 4.0, 0.6, 'best[i] 写入Global Memory', LIGHT_BLUE, fs=7)
arrow(ax7, 5, 1.5, 5, 1.0)
draw_node(ax7, 5, 0.4, 4.5, 0.6, '输出: N×18 best [q, pos, rot, cost, rank]', ORANGE, fs=7)

ax7.text(9.8, 11.0, '每个Block处理\n1个Target的\n16个种子', fontsize=6, ha='center',
         bbox=dict(boxstyle='round', facecolor='#FFF3CD', alpha=0.8, edgecolor='#D4A017'))
ax7.text(9.8, 3.5, '单Kernel\n零调度开销\n无额外筛选\nKernel Launch', fontsize=6, ha='center',
         bbox=dict(boxstyle='round', facecolor='#D4EDDA', alpha=0.8, edgecolor='#28A745'))

save(fig7, 'fig7_algorithm_flowchart'); plt.close()

print("Done. Color flowcharts regenerated.")
