#!/usr/bin/env python3
"""
批量IK实验数据可视化 —— 5张论文图表生成脚本。
生成中文期刊论文用图，用于替代 paper_complete.md 中的臃肿表格。
参考: docs/修改意见/figure1.md 的绘图清单。

图1: 不同批量规模下的吞吐量对比（折线图，对数坐标）
图2: CUDA B5 相对 cuRobo 的吞吐加速比（柱状图）
图3: 不同消融配置下的吞吐量对比（分组柱状图，3子图）
图4: 不同消融配置下的收敛率变化（折线图）
图5: 不同消融配置下的平均迭代次数（折线图）

数据源说明:
  - 图1/图2: 论文表6.1/表6.2 主对比数据（B5 vs cuRobo, repeat=30, zero_seed）。
    优先读取 results/ CSV（如存在且配置匹配），fallback 到论文权威数据。
  - 图3/图4/图5: 论文第6.3节消融表数据（Medium阈值: B0/B3/B5权威数据 + B1/B2/B4/B6旧30°阈值参考值）。

时间口径: host_api_total_time（跨求解器统一对比口径，含主机预处理与数据传输）
"""

import csv
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ============================================================
# 全局中文字体与样式设置（中文期刊风格）
# ============================================================
# 优先使用 Noto Sans CJK SC (思源黑体)，适合学术图表
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'AR PL UMing CN', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['mathtext.fontset'] = 'stix'   # 数学公式字体

# 全局样式 (中文学术期刊通常要求线条清晰、标注完整)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / 'results'
FIGURE_DIR = SCRIPT_DIR

# ============================================================
# 工具函数: CSV 读取
# ============================================================
def read_csv_metrics(csv_path):
    """读取 metric,value 格式的 CSV，返回 dict"""
    metrics = {}
    if not csv_path.exists():
        return metrics
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row['metric'].strip()
            try:
                val = float(row['value'].strip())
            except (ValueError, TypeError):
                val = row['value'].strip()
            metrics[key] = val
    return metrics


# ============================================================
# 论文主对比数据 (来源: 论文表6.1/表6.2, Medium阈值10mm/5°, repeat=30, zero_seed)
# 作为 CSV 不可用时的 fallback，确保图表与正文数字一致。
# 时间口径: gpu_end_to_end_time (GPU端到端)
# 数据更新日期: 2026-06-12
# ============================================================
# 论文 B5 (混合精度) 主对比数据: {N: throughput_targets_per_s}
PAPER_B5_THROUGHPUT = {
    100:  112414,
    500:  158251,
    1000: 148412,
    5000: 168683,
}

# 论文 cuRobo 主对比数据: {N: throughput_targets_per_s}
PAPER_CUROBO_THROUGHPUT = {
    100:  3118,
    500:  15844,
    1000: 31611,
    5000: 155059,
}

# 论文 B3 (FP64+自适应阻尼) 对照数据
PAPER_B3_THROUGHPUT = {
    100:  51361,
    500:  62384,
    5000: 66050,
}

# ============================================================
# 论文消融数据（来源：论文第6.3节消融表, Medium阈值10mm/5°, repeat=30, zero_seed）
# B1/B2/B4/B6 数据基于旧30°阈值（标注为参考值）, B0/B3/B5为Medium阈值实测
# 数据更新日期: 2026-06-12
# ============================================================
# 消融吞吐量数据: {N: {level: throughput_targets_per_s}}
ABLATION_THROUGHPUT = {
    100: {
        'B0': 7589,   'B3': 51361,  'B5': 113097,
        # B1/B2/B4/B6 为旧30°阈值参考值，不参与 Medium 主结论
        'B1': 138125, 'B2': 138562, 'B4': 43223,  'B6': 111238,
    },
    500: {
        'B0': 9896,   'B3': 62384,  'B5': 155071,
        # B4 为旧30°阈值参考值
        'B4': 50981,
    },
    5000: {
        'B0': 12507,  'B3': 66050,  'B5': 164207,
        # B4 为旧30°阈值参考值
        'B4': 56932,
    },
}

# 消融收敛率数据: {N: {level: convergence_rate}}
ABLATION_CONVRATE = {
    100:  {'B0': 0.830, 'B3': 1.000, 'B5': 1.000,
           # B1/B2/B4/B6 为旧30°阈值参考值
           'B1': 1.000, 'B2': 1.000, 'B4': 1.000, 'B6': 1.000},
    500:  {'B0': 0.522, 'B3': 1.000, 'B5': 0.998,
           'B4': 1.000},
    5000: {'B0': 0.564, 'B3': 1.000, 'B5': 0.9998,
           'B4': 1.000},
}

# 消融平均迭代次数: {N: {level: avg_iterations}}
ABLATION_ITERS = {
    100:  {'B0': 31.36, 'B3': 12.85, 'B5': 14.47,
           # B1/B2/B4/B6 为旧30°阈值参考值
           'B1': 4.31,  'B2': 4.31,  'B4': 13.66, 'B6': 13.95},
    500:  {'B0': 79.74, 'B3': 13.78, 'B5': 15.13,
           'B4': 14.88},
    5000: {'B0': 73.04, 'B3': 13.69, 'B5': 15.24,
           'B4': 14.63},
}

# 完整的消融级别列表（按演进顺序）
ALL_ABLATION_LEVELS = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6']


# ============================================================
# 配色方案（色盲友好 + 中文学术风格）
# ============================================================
COLOR_CUDA = '#2166AC'       # 深蓝 — CUDA DLS
COLOR_CUROBO = '#B2182B'     # 深红 — cuRobo
COLOR_SPEEDUP = '#F4A582'    # 暖橙 — 加速比
ABLATION_COLORS = {
    'B0': '#D1E5F0',  # 浅蓝灰
    'B1': '#92C5DE',
    'B2': '#4393C3',
    'B3': '#2166AC',  # 深蓝 — 自适应阻尼
    'B4': '#FDDBC7',  # 浅橙
    'B5': '#D6604D',  # 深红 — 混合精度 ★
    'B6': '#B2182B',  # 暗红
}

# 核心对比线的样式
MARKER_CUDA = 'o'
MARKER_CUROBO = 's'
MARKER_B0 = 's'
MARKER_B3 = '^'
MARKER_B5 = 'D'


# ============================================================
# 图1: 不同批量规模下的吞吐量对比
# 替代: 论文表6.1（主GPU对比表） + 表6.7（批量规模对比）
# ============================================================
def plot_figure1(b5_data, curobo_data):
    """折线图，对数Y轴，双线对比（B5 vs cuRobo）"""
    fig, ax = plt.subplots(figsize=(7, 5))

    N_values = [100, 500, 1000, 5000]
    b5_tp = [b5_data[N] for N in N_values]
    curobo_tp = [curobo_data[N] for N in N_values]

    # 折线
    ax.plot(N_values, b5_tp,
            color=COLOR_CUDA, marker=MARKER_CUDA, markersize=8,
            linewidth=2, label='CUDA DLS (B5 混合精度)',
            zorder=5)
    ax.plot(N_values, curobo_tp,
            color=COLOR_CUROBO, marker=MARKER_CUROBO, markersize=8,
            linewidth=2, label='cuRobo',
            zorder=4)

    # 数据标注（B5在上方，cuRobo在下方避免重叠）
    for N, tp in zip(N_values, b5_tp):
        ax.annotate(f'{tp:,.0f}',
                    (N, tp), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=8,
                    color=COLOR_CUDA, fontweight='bold')
    for N, tp in zip(N_values, curobo_tp):
        ax.annotate(f'{tp:,.0f}',
                    (N, tp), textcoords="offset points",
                    xytext=(0, -14), ha='center', fontsize=8,
                    color=COLOR_CUROBO, fontweight='bold')

    ax.set_xlabel('批量规模 $N$')
    ax.set_ylabel('吞吐量 / (targets·s$^{-1}$)')
    ax.set_title('图1  不同批量规模下的吞吐量对比')
    ax.set_yscale('log')
    ax.set_xticks(N_values)
    ax.set_xlim(50, 5500)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, which='major', alpha=0.3, linestyle='--')
    ax.grid(True, which='minor', alpha=0.1, linestyle=':')

    # Y轴对数格式化
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f'{y:,.0f}'))

    fig.tight_layout()
    outpath = FIGURE_DIR / 'figure1_throughput_comparison.png'
    fig.savefig(outpath)
    plt.close(fig)
    print(f'[OK] 图1 已保存: {outpath}')
    return outpath


# ============================================================
# 图2: CUDA B5 相对 cuRobo 的吞吐加速比
# 替代: 论文表6.2（加速比汇总表）
# ============================================================
def plot_figure2(b5_data, curobo_data):
    """柱状图，展示加速比 = B5吞吐 / cuRobo吞吐"""
    fig, ax = plt.subplots(figsize=(7, 5))

    N_values = [100, 500, 1000, 5000]
    speedups = [b5_data[N] / curobo_data[N] for N in N_values]

    x_pos = np.arange(len(N_values))
    # 渐变色调: 加速比越高颜色越深
    bar_colors = ['#D6604D', '#F4A582', '#FDDBC7', '#FDE0DD']
    bars = ax.bar(x_pos, speedups, width=0.55,
                  color=bar_colors,
                  edgecolor='#8B3A3A', linewidth=0.8)

    # 标注数值
    for i, (bar, sp) in enumerate(zip(bars, speedups)):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f'{sp:.1f}×',
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                color='#8B3A3A')

    ax.set_xlabel('批量规模 $N$')
    ax.set_ylabel('加速比 (B5 / cuRobo)')
    ax.set_title('图2  CUDA B5 相对 cuRobo 的吞吐加速比')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'$N={n}$' for n in N_values])
    ax.set_ylim(0, max(speedups) * 1.25)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # 水平参考线 y=1（性能持平）
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(len(N_values) - 0.6, 1.3, '性能持平 (1×)', fontsize=8,
            color='gray', alpha=0.8)

    fig.tight_layout()
    outpath = FIGURE_DIR / 'figure2_speedup_bars.png'
    fig.savefig(outpath)
    plt.close(fig)
    print(f'[OK] 图2 已保存: {outpath}')
    return outpath


# ============================================================
# 图3: 不同消融配置下的吞吐量对比（分组柱状图，3子图）
# ============================================================
def plot_figure3():
    """3个子图：N=100, N=500, N=5000 的消融吞吐量"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))

    subplot_configs = [
        (axes[0], 100, '(a) $N=100$'),
        (axes[1], 500, '(b) $N=500$'),
        (axes[2], 5000, '(c) $N=5000$'),
    ]

    for ax, N, title in subplot_configs:
        data = ABLATION_THROUGHPUT[N]
        levels_present = [l for l in ALL_ABLATION_LEVELS if l in data]
        values = [data[l] for l in levels_present]
        colors = [ABLATION_COLORS[l] for l in levels_present]

        x_pos = np.arange(len(levels_present))
        bars = ax.bar(x_pos, values, width=0.6, color=colors,
                      edgecolor='#333333', linewidth=0.5)

        # 数值标注
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{val/1000:.0f}k',
                    ha='center', va='bottom', fontsize=7.5,
                    fontweight='bold', rotation=0)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(levels_present, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('吞吐量 / (targets·s$^{-1}$)')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')

        # ★ 标注 B5 为最佳配置
        if 'B5' in data:
            idx_b5 = levels_present.index('B5')
            ax.annotate('★ 最佳',
                        (x_pos[idx_b5], data['B5']),
                        textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=8,
                        color='#B2182B', fontweight='bold')

    fig.suptitle('图3  不同消融配置下的吞吐量对比', fontsize=12, y=1.02)
    fig.tight_layout()
    outpath = FIGURE_DIR / 'figure3_ablation_throughput.png'
    fig.savefig(outpath)
    plt.close(fig)
    print(f'[OK] 图3 已保存: {outpath}')
    return outpath


# ============================================================
# 图4: 不同消融配置下的收敛率变化
# ============================================================
def plot_figure4():
    """折线图：B0, B3, B5 三条线的收敛率随N变化"""
    fig, ax = plt.subplots(figsize=(7, 5))

    N_values = [100, 500, 1000, 5000]
    levels = ['B0', 'B3', 'B5']
    markers = {'B0': 's', 'B3': '^', 'B5': 'D'}
    colors_line = {'B0': '#92C5DE', 'B3': '#2166AC', 'B5': '#D6604D'}
    labels = {'B0': 'B0 (FP64基线)', 'B3': 'B3 (+自适应阻尼)', 'B5': 'B5 (混合精度)'}

    for level in levels:
        conv_rates = []
        for N in N_values:
            if N in ABLATION_CONVRATE and level in ABLATION_CONVRATE[N]:
                conv_rates.append(ABLATION_CONVRATE[N][level] * 100)
            else:
                conv_rates.append(None)

        # 过滤None值
        valid_N = [N for N, cr in zip(N_values, conv_rates) if cr is not None]
        valid_cr = [cr for cr in conv_rates if cr is not None]

        if valid_N:
            ax.plot(valid_N, valid_cr,
                    color=colors_line[level],
                    marker=markers[level], markersize=8,
                    linewidth=2, label=labels[level])

            # 标注数值
            for N, cr in zip(valid_N, valid_cr):
                offset = 10 if level == 'B0' else -12
                ax.annotate(f'{cr:.1f}%' if cr < 99.9 else f'{cr:.2f}%',
                            (N, cr), textcoords="offset points",
                            xytext=(0, offset), ha='center', fontsize=8,
                            color=colors_line[level], fontweight='bold')

    # 100% 参考线
    ax.axhline(y=100.0, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(4000, 100.5, '100% 收敛', fontsize=8, color='green', alpha=0.7)

    ax.set_xlabel('批量规模 $N$')
    ax.set_ylabel('收敛率 / %')
    ax.set_title('图4  不同消融配置下的收敛率变化')
    ax.set_xticks(N_values)
    ax.set_xlim(50, 5500)
    ax.set_ylim(75, 105)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    fig.tight_layout()
    outpath = FIGURE_DIR / 'figure4_convergence_rate.png'
    fig.savefig(outpath)
    plt.close(fig)
    print(f'[OK] 图4 已保存: {outpath}')
    return outpath


# ============================================================
# 图5: 不同消融配置下的平均迭代次数
# ============================================================
def plot_figure5():
    """折线图：B0, B3, B5 三条线的平均迭代次数随N变化"""
    fig, ax = plt.subplots(figsize=(7, 5))

    N_values = [100, 500, 1000, 5000]
    levels = ['B0', 'B3', 'B5']
    markers = {'B0': 's', 'B3': '^', 'B5': 'D'}
    colors_line = {'B0': '#92C5DE', 'B3': '#2166AC', 'B5': '#D6604D'}
    labels = {'B0': 'B0 (FP64基线)', 'B3': 'B3 (+自适应阻尼)', 'B5': 'B5 (混合精度)'}

    # 补充 N=1000 的迭代数据（论文表6.1中有B5 N=1000的avg_iters=15.34）
    # 其他级别在N=1000的数据从论文表6.1推算
    extra_iters = {
        1000: {'B5': 15.34}  # 从表6.1 B5 N=1000
    }

    for level in levels:
        avg_iters = []
        for N in N_values:
            if N in ABLATION_ITERS and level in ABLATION_ITERS[N]:
                avg_iters.append(ABLATION_ITERS[N][level])
            elif N in extra_iters and level in extra_iters[N]:
                avg_iters.append(extra_iters[N][level])
            else:
                avg_iters.append(None)

        valid_N = [N for N, ai in zip(N_values, avg_iters) if ai is not None]
        valid_ai = [ai for ai in avg_iters if ai is not None]

        if valid_N:
            ax.plot(valid_N, valid_ai,
                    color=colors_line[level],
                    marker=markers[level], markersize=8,
                    linewidth=2, label=labels[level])

            for N, ai in zip(valid_N, valid_ai):
                offset = 10 if level != 'B5' else -12
                ax.annotate(f'{ai:.1f}',
                            (N, ai), textcoords="offset points",
                            xytext=(0, offset), ha='center', fontsize=8,
                            color=colors_line[level], fontweight='bold')

    ax.set_xlabel('批量规模 $N$')
    ax.set_ylabel('平均迭代次数')
    ax.set_title('图5  不同消融配置下的平均迭代次数')
    ax.set_xticks(N_values)
    ax.set_xlim(50, 5500)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    fig.tight_layout()
    outpath = FIGURE_DIR / 'figure5_avg_iterations.png'
    fig.savefig(outpath)
    plt.close(fig)
    print(f'[OK] 图5 已保存: {outpath}')
    return outpath


# ============================================================
# 主流程
# ============================================================
def main():
    print('=' * 60)
    print('批量IK实验数据可视化 —— 生成论文5张图表')
    print('=' * 60)

    # ---------- 数据加载策略 ----------
    # 优先尝试从 CSV 读取，若与论文B5配置一致则使用CSV；
    # 否则 fallback 到论文权威数据，确保图表与正文一致。
    print('\n[INFO] 检查 CSV 数据可用性...')

    # 尝试读取CSV（当前为B4-like配置）
    csv_cuda = {}
    csv_curobo = {}
    for N in [100, 500, 1000, 5000]:
        cuda_file = RESULTS_DIR / f'ur10_cuda_N{N}_seed42_repeat30_zero_seed.csv'
        curobo_file = RESULTS_DIR / f'ur10_curobo_N{N}_seed42_repeat30_zero_seed.csv'
        if cuda_file.exists():
            csv_cuda[N] = read_csv_metrics(cuda_file)
        if curobo_file.exists():
            csv_curobo[N] = read_csv_metrics(curobo_file)

    # 判断CSV数据是否匹配论文B5配置
    # 论文B5 N=100的avg_iterations为13.95，而CSV为13.66（更接近B4）
    use_csv_for_fig12 = False
    if csv_cuda and csv_curobo:
        csv_n100_iters = csv_cuda.get(100, {}).get('avg_iterations', 0)
        paper_b5_iters = ABLATION_ITERS[100].get('B5', 0)
        use_csv_for_fig12 = abs(csv_n100_iters - paper_b5_iters) < 0.1
        if use_csv_for_fig12:
            print(f'  CSV avg_iterations(N=100)={csv_n100_iters} ≈ 论文B5={paper_b5_iters}')
            print('  → 使用 CSV 数据生成图1/图2')
        else:
            print(f'  CSV avg_iterations(N=100)={csv_n100_iters} ≠ 论文B5={paper_b5_iters}')
            print('  → CSV数据为早期实验版本, fallback到论文B5权威数据')

    # ---------- 确定图1/图2使用的数据 ----------
    if use_csv_for_fig12:
        b5_data = {N: csv_cuda[N]['throughput_targets_per_s'] for N in [100, 500, 1000, 5000]}
        curobo_data = {N: csv_curobo[N]['throughput_targets_per_s'] for N in [100, 500, 1000, 5000]}
    else:
        b5_data = PAPER_B5_THROUGHPUT
        curobo_data = PAPER_CUROBO_THROUGHPUT

    # 打印数据摘要
    print('\n[INFO] 图1/图2 数据摘要:')
    for N in [100, 500, 1000, 5000]:
        b5_tp = b5_data[N]
        cb_tp = curobo_data[N]
        speedup = b5_tp / cb_tp
        print(f'  N={N:5d}: B5={b5_tp:>10,.0f} t/s  '
              f'cuRobo={cb_tp:>10,.0f} t/s  '
              f'加速比={speedup:.1f}×')

    print('\n[INFO] 图3/图4/图5 数据来源: 论文第6.2节消融表')

    # ---------- 生成图表 ----------
    print('\n[INFO] 生成图表...')

    paths = {}

    paths['图1'] = plot_figure1(b5_data, curobo_data)
    paths['图2'] = plot_figure2(b5_data, curobo_data)
    paths['图3'] = plot_figure3()
    paths['图4'] = plot_figure4()
    paths['图5'] = plot_figure5()

    print(f'\n{"=" * 60}')
    print('全部图表生成完毕！输出目录:', FIGURE_DIR)
    for name, p in paths.items():
        size_kb = os.path.getsize(p) / 1024
        print(f'  {name}: {p.name} ({size_kb:.0f} KB)')
    print('=' * 60)


if __name__ == '__main__':
    main()
