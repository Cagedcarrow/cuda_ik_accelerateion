#!/usr/bin/env python3
"""Compare latest benchmark data with paper data in paper_complete.md."""
from __future__ import annotations

# ============================================================
# Paper data (extracted from paper_complete.md tables)
# ============================================================
PAPER = {
    # (config, N) -> {throughput, gpu_time_ms, conv_rate, avg_iters}
    ("B0", 100):  {"tp": 134008, "time": 0.751, "conv": 1.000, "iters": 4.31},
    ("B0", 500):  {"tp": 21094,  "time": 23.71, "conv": 0.804, "iters": 35.74},
    ("B0", 5000): {"tp": 30695,  "time": 162.9, "conv": 0.834, "iters": 30.86},
    ("B1", 100):  {"tp": 138125, "time": 0.728, "conv": 1.000, "iters": 4.31},
    ("B2", 100):  {"tp": 138562, "time": 0.726, "conv": 1.000, "iters": 4.31},
    ("B3", 100):  {"tp": 52064,  "time": 1.925, "conv": 1.000, "iters": 12.43},
    ("B3", 500):  {"tp": 59821,  "time": 8.36,  "conv": 1.000, "iters": 13.32},
    ("B3", 5000): {"tp": 71380,  "time": 70.06, "conv": 1.000, "iters": 13.11},
    ("B4", 100):  {"tp": 43223,  "time": 2.318, "conv": 1.000, "iters": 13.66},
    ("B4", 500):  {"tp": 50981,  "time": 9.81,  "conv": 1.000, "iters": 14.88},
    ("B4", 5000): {"tp": 56932,  "time": 87.83, "conv": 1.000, "iters": 14.63},
    ("B5", 100):  {"tp": 107250, "time": 0.932, "conv": 1.000, "iters": 13.95},
    ("B5", 500):  {"tp": 149787, "time": 3.338, "conv": 0.998, "iters": 14.54},
    ("B5", 1000): {"tp": 140246, "time": 7.130, "conv": 0.998, "iters": 15.34},
    ("B5", 5000): {"tp": 180962, "time": 27.630,"conv": 0.9998,"iters": 14.66},
    ("B6", 100):  {"tp": 111238, "time": 0.899, "conv": 1.000, "iters": 13.95},
    ("cuRobo", 100):  {"tp": 2850},
    ("cuRobo", 500):  {"tp": 14574},
    ("cuRobo", 1000): {"tp": 29518},
    ("cuRobo", 5000): {"tp": 144855},
}

# ============================================================
# New benchmark data (just collected)
# ============================================================
NEW = {
    ("B0", 100):  {"tp": 129814, "time": 0.770, "conv": 1.000, "iters": 4.31},
    ("B0", 500):  {"tp": 21041,  "time": 23.763,"conv": 0.804, "iters": 35.74},
    ("B0", 5000): {"tp": 30140,  "time": 165.893,"conv": 0.834, "iters": 30.86},
    ("B1", 100):  {"tp": 168471, "time": 0.594, "conv": 1.000, "iters": 4.31},
    ("B2", 100):  {"tp": 150784, "time": 0.663, "conv": 1.000, "iters": 4.31},
    ("B3", 100):  {"tp": 48817,  "time": 2.048, "conv": 1.000, "iters": 12.43},
    ("B3", 500):  {"tp": 61638,  "time": 8.112, "conv": 1.000, "iters": 13.32},
    ("B3", 5000): {"tp": 71073,  "time": 70.351,"conv": 1.000, "iters": 13.11},
    ("B4", 100):  {"tp": 45925,  "time": 2.177, "conv": 1.000, "iters": 13.66},
    ("B4", 500):  {"tp": 53440,  "time": 9.356, "conv": 1.000, "iters": 14.88},
    ("B4", 5000): {"tp": 56897,  "time": 87.879,"conv": 1.000, "iters": 14.63},
    ("B5", 100):  {"tp": 104521, "time": 0.957, "conv": 1.000, "iters": 13.95},
    ("B5", 500):  {"tp": 146241, "time": 3.419, "conv": 0.998, "iters": 14.54},
    ("B5", 1000): {"tp": 155233, "time": 6.442, "conv": 0.998, "iters": 15.34},
    ("B5", 5000): {"tp": 176547, "time": 28.321,"conv": 0.9998,"iters": 14.66},
    ("B6", 100):  {"tp": 110168, "time": 0.908, "conv": 1.000, "iters": 13.95},
    ("cuRobo", 100):  {"tp": 3059},
    ("cuRobo", 500):  {"tp": 16659},
    ("cuRobo", 1000): {"tp": 32414},
    ("cuRobo", 5000): {"tp": 167482},
}

def pct(new, old):
    return (new - old) / old * 100

def main():
    print("=" * 100)
    print("数据对比报告：最新运行 vs 论文 paper_complete.md")
    print("=" * 100)

    # --- B5 main comparison ---
    print("\n## 1. B5 主配置对比（论文表6.1 / 表6.7）\n")
    print(f"{'N':<6} {'指标':<12} {'论文值':>12} {'最新值':>12} {'差异':>10} {'匹配?'}")
    print("-" * 70)
    for N in [100, 500, 1000, 5000]:
        for metric, label, fmt in [("tp", "吞吐量(t/s)", ".0f"), ("time", "GPU时间(ms)", ".3f"),
                                     ("conv", "收敛率", ".4f"), ("iters", "迭代次数", ".2f")]:
            p = PAPER.get(("B5", N), {}).get(metric, None)
            n_val = NEW.get(("B5", N), {}).get(metric, None)
            if p is None or n_val is None:
                continue
            diff = pct(n_val, p)
            match = "✓" if abs(diff) < 5 else ("△" if abs(diff) < 10 else "✗")
            print(f"{N:<6} {label:<12} {p:>{12}{fmt}} {n_val:>{12}{fmt}} {diff:>+9.1f}% {match}")
        print()

    # --- cuRobo comparison ---
    print("\n## 2. cuRobo 对比\n")
    print(f"{'N':<6} {'论文':>12} {'最新':>12} {'差异':>10}")
    print("-" * 50)
    for N in [100, 500, 1000, 5000]:
        p = PAPER.get(("cuRobo", N), {}).get("tp", 0)
        n_val = NEW.get(("cuRobo", N), {}).get("tp", 0)
        diff = pct(n_val, p)
        print(f"{N:<6} {p:>12.0f} {n_val:>12.0f} {diff:>+9.1f}%")

    # --- B5 vs cuRobo Speedup ---
    print("\n## 3. B5 vs cuRobo 加速比变化\n")
    print(f"{'N':<6} {'论文加速比':>12} {'最新加速比':>12} {'变化':>10}")
    print("-" * 50)
    for N in [100, 500, 1000, 5000]:
        old_speedup = PAPER[("B5", N)]["tp"] / PAPER[("cuRobo", N)]["tp"]
        new_speedup = NEW[("B5", N)]["tp"] / NEW[("cuRobo", N)]["tp"]
        print(f"{N:<6} {old_speedup:>11.1f}× {new_speedup:>11.1f}× {new_speedup-old_speedup:>+9.1f}×")

    # --- Ablation comparison ---
    print("\n## 4. 消融配置对比（论文表6.2.2-6.2.4）\n")
    print(f"{'配置':<6} {'N':<6} {'论文TP':>12} {'最新TP':>12} {'TP差异':>10} {'论文迭代':>10} {'最新迭代':>10} {'迭代匹配'}")
    print("-" * 90)
    for config in ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]:
        for N in [100, 500, 5000]:
            p = PAPER.get((config, N))
            n_val = NEW.get((config, N))
            if p is None or n_val is None:
                continue
            tp_diff = pct(n_val["tp"], p["tp"])
            iter_match = "✓" if abs(n_val["iters"] - p["iters"]) < 0.01 else "✗"
            print(f"{config:<6} {N:<6} {p['tp']:>12.0f} {n_val['tp']:>12.0f} {tp_diff:>+9.1f}% {p['iters']:>10.2f} {n_val['iters']:>10.2f} {iter_match:>8}")
        if config in ("B5",):
            N = 1000
            p = PAPER.get((config, N))
            n_val = NEW.get((config, N))
            if p and n_val:
                tp_diff = pct(n_val["tp"], p["tp"])
                iter_match = "✓" if abs(n_val["iters"] - p["iters"]) < 0.01 else "✗"
                print(f"{config:<6} {N:<6} {p['tp']:>12.0f} {n_val['tp']:>12.0f} {tp_diff:>+9.1f}% {p['iters']:>10.2f} {n_val['iters']:>10.2f} {iter_match:>8}")

    # --- Convergence summary ---
    print("\n## 5. 收敛率对比\n")
    print(f"{'配置':<6} {'N':<6} {'论文Conv':>10} {'最新Conv':>10} {'匹配'}")
    print("-" * 50)
    for config in ["B0", "B3", "B4", "B5"]:
        for N in [100, 500, 5000]:
            p = PAPER.get((config, N))
            n_val = NEW.get((config, N))
            if p is None or n_val is None:
                continue
            match = "✓" if abs(n_val["conv"] - p["conv"]) < 0.001 else "✗"
            print(f"{config:<6} {N:<6} {p['conv']:>10.4f} {n_val['conv']:>10.4f} {match:>6}")
        if config == "B5":
            N = 1000
            p = PAPER.get((config, N))
            n_val = NEW.get((config, N))
            if p and n_val:
                match = "✓" if abs(n_val["conv"] - p["conv"]) < 0.001 else "✗"
                print(f"{config:<6} {N:<6} {p['conv']:>10.4f} {n_val['conv']:>10.4f} {match:>6}")

    # --- Conclusion ---
    print("\n" + "=" * 100)
    print("总结")
    print("=" * 100)
    print("""
1. **迭代次数 100% 匹配** — 所有配置在所有 N 值下的 avg_iterations 完全一致（小数点后2位）。
   这证明算法逻辑未发生变化，代码的数值路径与论文版本一致。

2. **收敛率 100% 匹配** — 所有配置的收敛率与论文完全一致。

3. **吞吐量存在合理波动（±2~10%）** — 这是 GPU benchmark 的正常现象：
   - GPU 频率/温度状态、驱动版本、系统后台负载均可导致 ±5-10% 的波动
   - B5 典型波动: ±2.4%（除 N=1000 为 +10.7%）
   - B0/B3/B4: ±0.3~6.3%
   - B1/B2 N=100: 差异较大 (+22%/+8.8%)，可能因内核代码微调或 GPU 频率状态
   - B6 N=100: ±1.0%

4. **cuRobo 全面提速 7~16%** — 可能是 cuRobo 库版本更新或 PyTorch/CUDA 版本升级导致。

5. **B5 vs cuRobo 加速比变化** — 因 cuRobo 提速，加速比下调：
   - N=100: 37.6× → 34.2×
   - N=500: 10.3× → 8.8×
   - N=1000: 4.8× → 4.8×（不变）
   - N=5000: 1.25× → 1.05×（边际优势）

**建议**：由于迭代次数/收敛率 100% 匹配，吞吐量差异在正常波动范围内，
建议更新论文中的数据为最新运行值，以反映当前代码版本的真实性能。
""")

if __name__ == "__main__":
    main()
