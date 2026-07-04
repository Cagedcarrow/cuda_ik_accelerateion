#!/usr/bin/env python3
"""Generate all 7 figures as SVG + draw.io XML for flowcharts."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os, sys

# === Nature-style config (Chinese) ===
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Noto Sans CJK SC', 'AR PL UKai CN', 'DejaVu Sans'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'mathtext.fontset': 'dejavuserif',
})

CUDA_COLOR = '#0072B2'
CUROBO_COLOR = '#E69F00'
TEXT_COLOR = '#333333'

OUTDIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTDIR, exist_ok=True)

# ===== DATA =====
N_values = [100, 500, 1000, 5000]
cuda_throughput = [15539.2, 16344.0, 17817.9, 18490.4]
cuda_strict_sr = [0.960, 0.954, 0.954, 0.954]
cuda_pos_p95 = [4.385, 4.338, 4.563, 4.563]
cuda_gpu_ms = [6.435, 30.592, 56.123, 270.411]
curobo_throughput = [10508.7, 41815.6, 64928.2, 137148.3]
curobo_strict_sr = [0.870, 0.836, 0.840, 0.844]
curobo_pos_p95 = [74.32, 115.64, 98.92, 75.05]
h2d_mean, d2h_mean, kernel_mean, launch_overhead = 0.55, 0.097, 270.41, 0.007

fig_width_in = 3.35
fig_width_full_in = 7.0

def save(fig, name):
    for fmt in ['pdf', 'svg']:
        path = os.path.join(OUTDIR, f'{name}.{fmt}')
        fig.savefig(path, format=fmt, bbox_inches='tight', pad_inches=0.05)
    print(f"  Saved: {name}.pdf, {name}.svg")

# ====== Figure 1: Throughput Comparison ======
def fig1():
    fig, ax = plt.subplots(figsize=(fig_width_full_in, 3.2))
    x = np.arange(len(N_values)); w = 0.32
    b1 = ax.bar(x-w/2, cuda_throughput, w, color=CUDA_COLOR, edgecolor='white', lw=0.5, label='CUDA OPT4C')
    b2 = ax.bar(x+w/2, curobo_throughput, w, color=CUROBO_COLOR, edgecolor='white', lw=0.5, label='cuRobo-Graph')
    ax.set_xticks(x); ax.set_xticklabels([f'N={n}' for n in N_values])
    ax.set_ylabel('吞吐量 / (targets·s⁻¹)'); ax.set_yscale('log'); ax.set_ylim(5e3, 3e5)
    ax.legend(frameon=False, loc='upper left')
    for bar, val in zip(b1, cuda_throughput):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2000, f'{val:.0f}', ha='center', va='bottom', fontsize=7, color=CUDA_COLOR)
    for bar, val in zip(b2, curobo_throughput):
        offset = 8000 if val < 50000 else 15000
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+offset, f'{val:.0f}', ha='center', va='bottom', fontsize=7, color=CUROBO_COLOR)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title('CUDA OPT4C 与 cuRobo-Graph 批量吞吐量对比', fontsize=9, pad=6)
    save(fig, 'fig1_throughput'); plt.close()

# ====== Figure 2: Strict SR ======
def fig2():
    fig, ax = plt.subplots(figsize=(fig_width_in, 2.6))
    x = np.arange(len(N_values)); w = 0.30
    ax.bar(x-w/2, [s*100 for s in cuda_strict_sr], w, color=CUDA_COLOR, edgecolor='white', lw=0.5, label='CUDA OPT4C')
    ax.bar(x+w/2, [s*100 for s in curobo_strict_sr], w, color=CUROBO_COLOR, edgecolor='white', lw=0.5, label='cuRobo-Graph')
    ax.set_xticks(x); ax.set_xticklabels([f'N={n}' for n in N_values])
    ax.set_ylabel('Strict 成功率 / %'); ax.set_ylim(75, 102)
    ax.legend(frameon=False, loc='lower left', fontsize=7)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title('Strict 成功率对比 (5 mm / 1°)', fontsize=9, pad=6)
    for i in range(4):
        ax.text(i-w/2, cuda_strict_sr[i]*100+0.5, f'{cuda_strict_sr[i]:.3f}', ha='center', fontsize=7, color=CUDA_COLOR)
        ax.text(i+w/2, curobo_strict_sr[i]*100+0.5, f'{curobo_strict_sr[i]:.3f}', ha='center', fontsize=7, color=CUROBO_COLOR)
    save(fig, 'fig2_strict_sr'); plt.close()

# ====== Figure 3: Position Error p95 ======
def fig3():
    fig, ax = plt.subplots(figsize=(fig_width_in, 2.6))
    x = np.arange(len(N_values)); w = 0.30
    ax.bar(x-w/2, cuda_pos_p95, w, color=CUDA_COLOR, edgecolor='white', lw=0.5, label='CUDA OPT4C')
    ax.bar(x+w/2, curobo_pos_p95, w, color=CUROBO_COLOR, edgecolor='white', lw=0.5, label='cuRobo-Graph')
    ax.set_xticks(x); ax.set_xticklabels([f'N={n}' for n in N_values])
    ax.set_ylabel('位置误差 p95 / mm')
    ax.legend(frameon=False, loc='upper left', fontsize=7)
    ax.spines[['top','right']].set_visible(False)
    ax.axhline(y=5, color='red', linestyle='--', lw=0.8, alpha=0.5)
    ax.text(3.5, 5.5, 'Strict 阈值 (5 mm)', fontsize=6, color='red', ha='right', alpha=0.7)
    ax.set_title('全样本位置误差 p95 对比', fontsize=9, pad=6)
    for i in range(4):
        ax.text(i-w/2, cuda_pos_p95[i]+0.5, f'{cuda_pos_p95[i]:.2f}', ha='center', fontsize=7, color=CUDA_COLOR)
        ax.text(i+w/2, curobo_pos_p95[i]+3, f'{curobo_pos_p95[i]:.1f}', ha='center', fontsize=7, color=CUROBO_COLOR)
    save(fig, 'fig3_pos_error'); plt.close()

# ====== Figure 4: CUDA Scalability ======
def fig4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width_full_in, 2.8))
    ax1.plot(N_values, cuda_throughput, 'o-', color=CUDA_COLOR, lw=1.5, markersize=5)
    ax1.set_xlabel('批量规模 N'); ax1.set_ylabel('吞吐量 / (targets·s⁻¹)', color=CUDA_COLOR)
    ax1.tick_params(axis='y', labelcolor=CUDA_COLOR); ax1.set_ylim(14000, 20000)
    ax1.spines[['top','right']].set_visible(False)
    ax2.plot(N_values, cuda_gpu_ms, 's-', color='#D55E00', lw=1.5, markersize=5)
    ax2.set_xlabel('批量规模 N'); ax2.set_ylabel('GPU 流时间 / ms', color='#D55E00')
    ax2.tick_params(axis='y', labelcolor='#D55E00')
    ax2.spines[['top','right']].set_visible(False)
    fig.suptitle('CUDA OPT4C 批量扩展性分析', fontsize=9, y=1.02)
    save(fig, 'fig4_scalability'); plt.close()

# ====== Figure 5: Timing Breakdown ======
def fig5():
    fig, ax = plt.subplots(figsize=(fig_width_in, 2.4))
    labels = ['Kernel (LM+Jacobi)', 'H2D Transfer', 'D2H Transfer', 'Launch Overhead']
    sizes = [kernel_mean, h2d_mean, d2h_mean, launch_overhead]
    colors = [CUDA_COLOR, '#56B4E9', '#009E73', '#F0E442']
    explode = (0.04, 0, 0, 0)
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=None, colors=colors,
                                        autopct='%1.2f%%', startangle=90, pctdistance=0.75)
    for at in autotexts: at.set_fontsize(7)
    ax.legend(wedges, [f'{l}\n({s:.3f} ms)' for l, s in zip(labels, sizes)],
              loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7, frameon=False)
    ax.set_title('GPU 时间分解 (N=5000)', fontsize=9, pad=6)
    save(fig, 'fig5_timing'); plt.close()

print("Generating data figures (fig1-fig5)...")
fig1(); fig2(); fig3(); fig4(); fig5()

# =====================================================================
# Draw.io XML generation for flowcharts (fig6, fig7)
# =====================================================================

def write_drawio(path, mxfile_content):
    """Write a .drawio file (uncompressed XML format)."""
    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="drawio" modified="2026-07-04T00:00:00.000Z" agent="python" version="21.0.0" type="device">
  <diagram name="Page-1" id="page1">
    <mxGraphModel dx="1200" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="827" pageHeight="1169" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
{mxfile_content}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''
    with open(path, 'w', encoding='utf-8') as f:
        f.write(xml)
    print(f"  Saved: {os.path.basename(path)}")

def drawio_box(id, x, y, w, h, text, color, parent="1", font_color="#FFFFFF", bold=False, rounded=True, font_size=11):
    style = f"rounded={'1' if rounded else '0'};whiteSpace=wrap;html=1;fillColor={color};strokeColor=#333333;strokeWidth=1;fontColor={font_color};fontSize={font_size};"
    if bold: style += "fontStyle=1;"
    return f'        <mxCell id="{id}" value="{text}" style="{style}" vertex="1" parent="{parent}">\n          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>\n        </mxCell>\n'

def drawio_arrow(id, source, target, parent="1"):
    style = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#555555;strokeWidth=1.5;endArrow=block;endFill=1;"
    return f'        <mxCell id="{id}" style="{style}" edge="1" parent="{parent}" source="{source}" target="{target}">\n          <mxGeometry relative="1" as="geometry"/>\n        </mxCell>\n'

def drawio_diamond(id, x, y, w, h, text, color, parent="1", font_color="#FFFFFF", font_size=10):
    style = f"rhombus;whiteSpace=wrap;html=1;fillColor={color};strokeColor=#333333;strokeWidth=1;fontColor={font_color};fontSize={font_size};"
    return f'        <mxCell id="{id}" value="{text}" style="{style}" vertex="1" parent="{parent}">\n          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>\n        </mxCell>\n'

def drawio_text(id, x, y, w, h, text, parent="1", font_size=10, color="#333333", bold=False):
    style = f"text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize={font_size};fontColor={color};"
    if bold: style += "fontStyle=1;"
    return f'        <mxCell id="{id}" value="{text}" style="{style}" vertex="1" parent="{parent}">\n          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>\n        </mxCell>\n'

# ====== Figure 6: Thread Architecture (draw.io) ======
def fig6_drawio():
    cells = []
    nid = [0]
    def n(): nid[0] += 1; return f"cell{nid[0]}"
    BLUE = "#0072B2"; LIGHT_BLUE = "#56B4E9"; GREEN = "#009E73"; ORANGE = "#D55E00"; YELLOW = "#F0E442"
    GRAY = "#F0F8FF"; LIGHT_GRAY = "#E8F4FD"

    # Grid label
    cells.append(drawio_text(n(), 20, 20, 800, 30, "CUDA Grid: N blocks (<<<N, 32>>>)", bold=True, font_size=13))

    # Block row
    block_ids = []
    for i in range(8):
        bid = n(); block_ids.append(bid)
        cells.append(drawio_box(bid, 30 + i*95, 55, 80, 50, f"Block {i}", BLUE, bold=True))
    cells.append(drawio_text(n(), 790, 65, 30, 30, "...", font_size=14, bold=True))

    # Arrow: Grid -> Blocks
    cells.append(drawio_text(n(), 400, 38, 100, 20, "1 block / target", font_size=9, color="#555555"))

    # Section divider
    cells.append(drawio_text(n(), 20, 120, 800, 30, "Block 0 内部 (32 threads, K=16 seeds):", bold=True, font_size=12))

    # Block rectangle
    cells.append(drawio_box(n(), 20, 155, 800, 260, "", GRAY, font_color="#333333"))

    # Lanes (threads)
    lane_ids = []
    for i in range(16):
        lid = n(); lane_ids.append(lid)
        cells.append(drawio_box(lid, 40 + i*48, 210, 40, 50, f"T{i}", LIGHT_BLUE))

    cells.append(drawio_text(n(), 400, 180, 300, 25, "16 lanes × 1 seed/lane = K=16 Sobol seeds", font_size=10, color=TEXT_COLOR))

    # Per-lane: LM iteration
    for i in range(16):
        cells.append(drawio_text(n(), 42 + i*48, 265, 36, 40, "LM\n迭代\n求解", font_size=7, color=BLUE))

    # Shared memory
    smem_id = n()
    cells.append(drawio_box(smem_id, 60, 320, 720, 40, "Shared Memory: s_cand[16][kCandidateStride] doubles (16 × 16 = 256 doubles)", GREEN, bold=True, font_size=10))

    # Arrows lane -> smem
    for i in [2, 7, 13]:
        arrow_id = n()
        cells.append(drawio_arrow(arrow_id, lane_ids[i], smem_id))

    # Lane 0 selection
    sel_id = n()
    cells.append(drawio_box(sel_id, 200, 380, 400, 35, "Lane 0: 块内最佳候选选择 (Success Rank → Near-Limit → Pose Cost)", ORANGE, bold=True, font_size=10))

    arrow_id = n()
    cells.append(drawio_arrow(arrow_id, smem_id, sel_id))

    # Output
    out_id = n()
    cells.append(drawio_box(out_id, 200, 430, 400, 35, "best[q, pos, rot, cost, rank] × N targets → Global Memory", YELLOW, font_color="#333333", font_size=10))
    arrow_id = n()
    cells.append(drawio_arrow(arrow_id, sel_id, out_id))

    # Legend
    legend_y = 490
    cells.append(drawio_text(n(), 20, legend_y, 150, 20, "图例:", bold=True, font_size=10))
    for i, (color, label) in enumerate([
        (BLUE, "CUDA Block"), (LIGHT_BLUE, "Thread Lane"), (GREEN, "Shared Memory"), (ORANGE, "In-Block Selection")
    ]):
        cells.append(drawio_box(n(), 20 + i*190, legend_y + 25, 30, 20, "", color))
        cells.append(drawio_text(n(), 55 + i*190, legend_y + 25, 140, 20, label, font_size=9, color=TEXT_COLOR))

    # Notes
    cells.append(drawio_text(n(), 30, 545, 780, 50,
        "关键设计: (1) 每个 Block 独占处理 1 个 Target 的 16 个 Sobol 种子\n(2) 候选解全部存储在 Shared Memory, 无需 Global Memory 写回\n(3) Lane 0 在 Block 内完成三级层次化候选选择, 零额外 Kernel Launch\n(4) 单 Kernel 设计: Grid<<<N, 32>>> → 无多 Kernel 调度开销",
        font_size=9, color="#555555"))

    write_drawio(os.path.join(OUTDIR, 'fig6_thread_architecture.drawio'), ''.join(cells))

# ====== Figure 7: Algorithm Flowchart (draw.io) ======
def fig7_drawio():
    cells = []
    nid = [0]
    def n(): nid[0] += 1; return f"cell{nid[0]}"

    GREEN = "#009E73"; BLUE = "#0072B2"; ORANGE = "#E69F00"; RED = "#D55E00"; LIGHT_BLUE = "#56B4E9"

    # Start
    start_id = n()
    cells.append(drawio_box(start_id, 320, 20, 360, 45, "输入: N 个目标位姿 T*, K=16 Sobol 种子 q₀", GREEN, bold=True, font_size=11))

    # Grid Launch
    grid_id = n()
    cells.append(drawio_box(grid_id, 300, 85, 400, 40, "CUDA Grid Launch: <<<N, 32>>>", BLUE, bold=True, font_size=11))
    cells.append(drawio_arrow(n(), start_id, grid_id))

    # Block i loads data
    load_id = n()
    cells.append(drawio_box(load_id, 280, 150, 440, 40, "Block i: Global Memory → 加载 T*_i 和 seed[0..15]", LIGHT_BLUE, bold=True, font_size=10))
    cells.append(drawio_arrow(n(), grid_id, load_id))

    # 16 lanes parallel LM
    lm_id = n()
    cells.append(drawio_box(lm_id, 250, 215, 500, 45, "16 Lanes 并行: 每个 Lane 执行 LM 迭代求解 (max 60 iters)", BLUE, bold=True, font_size=11))
    cells.append(drawio_arrow(n(), load_id, lm_id))

    # LM iteration details (left + right)
    lm_left_id = n()
    cells.append(drawio_box(lm_left_id, 50, 285, 370, 90,
        "LM 迭代循环:\n1. FK with frames → T, p_joint, z_axis\n2. 解析 Jacobian: Jv = z×(p_ee−p_joint)\n3. Hessian H = JᵀJ + λI\n4. Gauss Elim 6×6 → Δq\n5. λ 自适应缩放 (×0.5 或 ×2)",
        LIGHT_BLUE, font_color="#333333", font_size=9))

    lm_right_id = n()
    cells.append(drawio_box(lm_right_id, 580, 285, 370, 90,
        "Loss 评估与更新:\n- Pose cost = ‖e_p‖² + ‖e_R‖²\n- Limit barrier penalty\n- Total loss = 0.5·pose + w·barrier\n- q ← q + Δq (总是接受 trial)",
        LIGHT_BLUE, font_color="#333333", font_size=9))
    cells.append(drawio_arrow(n(), lm_id, lm_left_id))
    cells.append(drawio_arrow(n(), lm_id, lm_right_id))

    # Convergence check
    conv_id = n()
    cells.append(drawio_diamond(conv_id, 350, 400, 280, 65, "收敛?\n(pos < 5mm & rot < 1°)", ORANGE, font_size=10))
    cells.append(drawio_arrow(n(), lm_left_id, conv_id))
    cells.append(drawio_arrow(n(), lm_right_id, conv_id))

    # Loop back
    loop_text_id = n()
    cells.append(drawio_text(loop_text_id, 660, 405, 80, 30, "No → 继续迭代", font_size=8, color="#D55E00"))
    cells.append(drawio_arrow(n(), conv_id, loop_text_id))

    # Write candidates to shared memory
    smem_id = n()
    cells.append(drawio_box(smem_id, 280, 495, 440, 40, "16 个候选解写入 Shared Memory s_cand[16][kCandidateStride]", GREEN, bold=True, font_size=10))
    cells.append(drawio_arrow(n(), conv_id, smem_id))

    # In-block selection
    sel_id = n()
    cells.append(drawio_box(sel_id, 280, 560, 440, 40, "Lane 0: 块内候选选择 (Success Rank → Near-Limit → Pose Cost)", ORANGE, bold=True, font_size=10))
    cells.append(drawio_arrow(n(), smem_id, sel_id))

    # Write best to global memory
    best_id = n()
    cells.append(drawio_box(best_id, 280, 625, 440, 40, "best[i] 写入 Global Memory [q₀..₅, pos, rot, cost, rank]", LIGHT_BLUE, font_color="#333333", font_size=10))
    cells.append(drawio_arrow(n(), sel_id, best_id))

    # Output
    out_id = n()
    cells.append(drawio_box(out_id, 300, 690, 400, 45, "输出: N × 18 best 解 [q, pos_err, rot_err, cost, rank]", RED, bold=True, font_size=11))
    cells.append(drawio_arrow(n(), best_id, out_id))

    # Side notes
    cells.append(drawio_text(n(), 830, 190, 150, 120,
        "每个 Block\n处理 1 个\nTarget 的\n16 个种子\n\n单 Kernel\n零调度开销\n无额外筛选\nKernel Launch",
        font_size=9, color=BLUE, bold=False))

    cells.append(drawio_text(n(), 830, 500, 150, 90,
        "FP64 精度\n寄存器级\n6×6 高斯消元\nShared Memory\n无 Bank 冲突",
        font_size=9, color=GREEN, bold=False))

    write_drawio(os.path.join(OUTDIR, 'fig7_algorithm_flowchart.drawio'), ''.join(cells))

print("Generating draw.io flowcharts (fig6, fig7)...")
fig6_drawio()
fig7_drawio()

# Also generate fig6 and fig7 as matplotlib SVGs for LaTeX
print("Generating SVG flowcharts (fig6, fig7)...")

# fig6 SVG
fig6, ax6 = plt.subplots(figsize=(fig_width_full_in, 4.5))
ax6.set_xlim(0, 10); ax6.set_ylim(0, 10); ax6.axis('off')
ax6.set_title('CUDA OPT4C Block-Target 线程映射架构', fontsize=9, pad=6)
def draw_box(ax, x, y, w, h, text, color, fs=8, tc='white', bold=False):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor='#333333', linewidth=0.8, alpha=0.9)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=tc, weight='bold' if bold else 'normal')

ax6.text(0.5, 9.5, 'CUDA Grid: N blocks', fontsize=9, fontweight='bold', color=TEXT_COLOR)
for i in range(8):
    draw_box(ax6, 1.0+i*1.1, 8.2, 0.9, 1.2, f'Block {i}', CUDA_COLOR, fs=6)
ax6.text(9.2, 8.2, '...', fontsize=10, ha='center', va='center')
ax6.annotate('', xy=(5, 8.8), xytext=(0.5, 9.2), arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0))
ax6.text(0.5, 7.0, 'Block 0 内部 (32 threads, K=16 seeds):', fontsize=9, fontweight='bold', color=TEXT_COLOR)
block_rect = FancyBboxPatch((0.3, 3.5), 9.4, 3.2, boxstyle="round,pad=0.15",
                            facecolor='#F0F8FF', edgecolor=CUDA_COLOR, linewidth=1.5, alpha=0.3)
ax6.add_patch(block_rect)
for i in range(16):
    draw_box(ax6, 0.7+i*0.55, 5.8, 0.45, 0.7, f'T{i}', '#56B4E9', fs=5)
ax6.text(4.8, 5.2, '16 lanes × 1 seed/lane = K=16 Sobol seeds', fontsize=7, ha='center', color=TEXT_COLOR)
draw_box(ax6, 5.0, 4.1, 8.5, 0.7, 'Shared Memory: s_cand[16 × 16] doubles', '#009E73', fs=7, bold=True)
for i in [2, 8, 14]:
    ax6.annotate('', xy=(0.7+i*0.55, 4.5), xytext=(0.7+i*0.55, 5.4), arrowprops=dict(arrowstyle='->', color='#888888', lw=0.6))
draw_box(ax6, 5.0, 3.2, 4.0, 0.55, 'Lane 0: In-Block Best Selection', '#D55E00', fs=7, bold=True)
ax6.annotate('', xy=(5.0, 3.5), xytext=(5.0, 3.75), arrowprops=dict(arrowstyle='->', color='#888888', lw=1.0))
draw_box(ax6, 5.0, 2.4, 5.0, 0.55, 'best[q,pos,rot,cost,rank] × N targets', '#F0E442', fs=7, tc='#333333')
ax6.annotate('', xy=(5.0, 2.6), xytext=(5.0, 2.9), arrowprops=dict(arrowstyle='->', color='#888888', lw=1.0))
legend_items = [(CUDA_COLOR,'CUDA Block'),('#56B4E9','Thread Lane'),('#009E73','Shared Memory'),('#D55E00','Selection')]
for i,(c,l) in enumerate(legend_items):
    ax6.add_patch(plt.Rectangle((0.3+i*2.4, 1.5), 0.3, 0.3, facecolor=c, edgecolor='#333333', lw=0.5))
    ax6.text(0.7+i*2.4, 1.65, l, fontsize=6.5, va='center', color=TEXT_COLOR)
save(fig6, 'fig6_thread_architecture'); plt.close()

# fig7 SVG
fig7, ax7 = plt.subplots(figsize=(fig_width_full_in, 5.5))
ax7.set_xlim(0, 10); ax7.set_ylim(0, 14); ax7.axis('off')
ax7.set_title('OPT4C 求解器算法流程', fontsize=9, pad=6)
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
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2), fontsize=6, ha='center', va='bottom', color='#555555')

draw_node(ax7, 5, 13.5, 3.0, 0.6, '输入: N个目标位姿 T*, K=16 Sobol种子', '#009E73', fs=7)
draw_node(ax7, 5, 12.3, 3.5, 0.6, 'CUDA Grid Launch: N blocks, 32 threads/block', CUDA_COLOR, fs=7)
arrow(ax7, 5, 12.0, 5, 11.5)
draw_node(ax7, 5, 11.0, 4.0, 0.7, 'Block i: 加载 T*_i 和 seed[0..15] 到寄存器', '#56B4E9', fs=7)
arrow(ax7, 5, 10.6, 5, 10.2)
draw_node(ax7, 5, 9.7, 4.5, 0.7, '16 lanes 并行: 每个 lane 执行 LM 迭代求解', CUDA_COLOR, fs=7, bold=True)
arrow(ax7, 5, 9.3, 5, 9.0)
draw_node(ax7, 2.2, 8.1, 3.5, 1.2,
    'LM迭代 (max 60):\n1. FK with frames → T,p,z\n2. 解析Jacobian: Jv=z×(p-p)\n3. H=JTJ+λI, g=JTe\n4. 6×6 Gauss Elim\n5. λ自适应缩放', CUDA_COLOR, fs=5.5)
draw_node(ax7, 7.8, 8.1, 3.5, 1.2,
    'Loss 评估:\n- Pose cost = |ep|²+|eR|²\n- Limit barrier penalty\n- Total = 0.5·pose+w·barrier\n- Trial accept: q←q+dq', CUDA_COLOR, fs=5.5)
arrow(ax7, 5, 9.4, 2.2, 8.7); arrow(ax7, 5, 9.4, 7.8, 8.7)
arrow(ax7, 2.2, 7.5, 5, 6.8); arrow(ax7, 7.8, 7.5, 5, 6.8)
draw_node(ax7, 5, 6.3, 4.5, 0.6, '收敛? (pos<5mm & rot<1°) 或 iter=max?', '#E69F00', fs=6.5, shape='diamond')
arrow(ax7, 8.0, 6.3, 9.0, 6.3, 'No')
draw_node(ax7, 9.5, 6.3, 1.0, 0.5, '继续', '#CCCCCC', fs=5)
arrow(ax7, 5, 5.9, 5, 5.0, 'Yes')
draw_node(ax7, 5, 4.5, 4.5, 0.6, '16个候选解写入 Shared Memory s_cand[]', '#56B4E9', fs=7)
arrow(ax7, 5, 4.2, 5, 3.8)
draw_node(ax7, 5, 3.2, 4.5, 0.8, 'Lane 0: 块内候选选择\nSuccess Rank→Near-Limit→Pose Cost', CUDA_COLOR, fs=7, bold=True)
arrow(ax7, 5, 2.8, 5, 2.4)
draw_node(ax7, 5, 1.8, 4.0, 0.6, 'best[i] 写入 Global Memory', '#56B4E9', fs=7)
arrow(ax7, 5, 1.5, 5, 1.0)
draw_node(ax7, 5, 0.4, 4.5, 0.6, '输出: N×18 best [q, pos, rot, cost, rank]', '#D55E00', fs=7)
ax7.text(9.8, 11.0, '每个Block处理\n1个Target的\n16个种子', fontsize=6, ha='center', bbox=dict(boxstyle='round', facecolor='#FFF3CD', alpha=0.8, edgecolor='#D4A017'))
ax7.text(9.8, 3.5, '单Kernel\n零调度开销\n无额外筛选\nKernel Launch', fontsize=6, ha='center', bbox=dict(boxstyle='round', facecolor='#D4EDDA', alpha=0.8, edgecolor='#28A745'))
save(fig7, 'fig7_algorithm_flowchart'); plt.close()

print(f"\nAll figures saved to: {OUTDIR}")
print("Files generated:")
for f in sorted(os.listdir(OUTDIR)):
    print(f"  {f}")
