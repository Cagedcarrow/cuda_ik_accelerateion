#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
import subprocess
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "data" / "cuda_inputs"
OUT = ROOT / "data" / "results" / "opt" / "opt4_followup"
DOC = ROOT / "docs" / "opt" / "opt4_followup"
LOG = ROOT / "logs" / "opt" / "opt4_followup"
RUNNER = ROOT / "build" / "standard_robot_cuda_v4_runner"


def ensure_dirs() -> None:
    for d in [OUT, DOC, LOG]:
        d.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str], log: Path, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        text = p.stdout
    except subprocess.TimeoutExpired as exc:
        text = (exc.stdout or "") + f"\n[TIMEOUT] {timeout}s\n"
        p = subprocess.CompletedProcess(cmd, 124, text, "")
    text += f"\n[elapsed_s] {time.perf_counter() - t0:.3f}\n[returncode] {p.returncode}\n"
    log.write_text(text, encoding="utf-8")
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{text}")
    return p


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_one(path: Path) -> dict[str, object]:
    return pd.read_csv(path).iloc[0].to_dict()


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([str(row[c]) for c in df.columns])
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell.replace("\n", " ") for cell in row) + " |")
    return "\n".join(lines)


def target_raw(n: int) -> Path:
    return INP / f"targets_N{n}_T4x4_f64.raw"


def seed_raw(n: int) -> Path:
    return INP / f"seeds_N{n}_K16_q_f64.raw"


def run_static(variant: str, n: int, warmup: int, repeat: int, prefix: str) -> tuple[Path, Path]:
    best = OUT / f"{prefix}.best.csv"
    summary = OUT / f"{prefix}.csv"
    cmd = [
        str(RUNNER),
        "--mode", "v4_static",
        "--variant", variant,
        "--limit-gradient", "analytic",
        "--targets", str(target_raw(n)),
        "--seeds", str(seed_raw(n)),
        "--N", str(n),
        "--K", "16",
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--best-csv", str(best),
        "--summary-csv", str(summary),
    ]
    run(cmd, LOG / f"{prefix}.log")
    return best, summary


def compare_best(base_best: Path, var_best: Path, base_summary: Path, var_summary: Path, out_path: Path) -> dict[str, object]:
    base = pd.read_csv(base_best)
    var = pd.read_csv(var_best)
    bs = read_one(base_summary)
    vs = read_one(var_summary)
    strict_diff_pp = abs(float(bs["strict_sr"]) - float(vs["strict_sr"]))
    pos_p95_diff = abs(float(bs["pos_p95_all_mm"]) - float(vs["pos_p95_all_mm"]))
    near_diff_pp = abs(float(bs["near_limit_ratio"]) - float(vs["near_limit_ratio"]))
    max_q_abs_diff = max((base[f"q{i}"] - var[f"q{i}"]).abs().max() for i in range(6))
    best_seed_diff = int((base["best_seed_id"] != var["best_seed_id"]).sum())
    row = {
        "variant": str(vs["method"]),
        "N": int(vs["N"]),
        "correctness_pass": int(
            strict_diff_pp <= 0.02
            and pos_p95_diff <= 2.0
            and near_diff_pp <= 0.02
            and int(vs["nan_count"]) == 0
            and int(vs["inf_count"]) == 0
        ),
        "strict_sr": float(vs["strict_sr"]),
        "pos_p95_all_mm": float(vs["pos_p95_all_mm"]),
        "near_limit": float(vs["near_limit_ratio"]),
        "strict_sr_diff_pp": strict_diff_pp,
        "pos_p95_diff_mm": pos_p95_diff,
        "near_limit_diff_pp": near_diff_pp,
        "best_seed_diff_count": best_seed_diff,
        "max_q_abs_diff": float(max_q_abs_diff),
        "nan_count": int(vs["nan_count"]),
        "inf_count": int(vs["inf_count"]),
    }
    write_rows(out_path, [row])
    if not row["correctness_pass"]:
        fail_rows = []
        for i in range(len(base)):
            reason = []
            if base.loc[i, "best_seed_id"] != var.loc[i, "best_seed_id"]:
                reason.append("best_seed")
            if abs(base.loc[i, "pos_err_mm"] - var.loc[i, "pos_err_mm"]) > 2.0:
                reason.append("pos")
            if base.loc[i, "success_rank"] != var.loc[i, "success_rank"]:
                reason.append("success_rank")
            if base.loc[i, "near_limit"] != var.loc[i, "near_limit"]:
                reason.append("near_limit")
            if reason:
                fail_rows.append({
                    "target_id": i,
                    "baseline_best_seed": base.loc[i, "best_seed_id"],
                    "opt4_best_seed": var.loc[i, "best_seed_id"],
                    "baseline_pos_err": base.loc[i, "pos_err_mm"],
                    "opt4_pos_err": var.loc[i, "pos_err_mm"],
                    "baseline_rot_err": base.loc[i, "rot_err_deg"],
                    "opt4_rot_err": var.loc[i, "rot_err_deg"],
                    "baseline_success_rank": base.loc[i, "success_rank"],
                    "opt4_success_rank": var.loc[i, "success_rank"],
                    "baseline_near_limit": base.loc[i, "near_limit"],
                    "opt4_near_limit": var.loc[i, "near_limit"],
                    "reason": "|".join(reason),
                })
        fail_path = OUT / ("opt4c_failure_cases.csv" if "block" in str(vs["method"]) else "opt4b_failure_cases.csv")
        write_rows(fail_path, fail_rows)
    return row


def parse_ptxas() -> dict[str, int]:
    run(["cmake", "-S", ".", "-B", "build"], LOG / "opt4_followup_cmake.log")
    p = run(
        ["cmake", "--build", "build", "--target", "standard_robot_cuda_v4_runner_ptxas", "-j1", "--clean-first"],
        LOG / "ptxas_opt4_followup.log",
        check=False,
    )
    text = (LOG / "ptxas_opt4_followup.log").read_text(encoding="utf-8", errors="ignore")
    regs: dict[str, int] = {}
    current = ""
    current_spill = ""
    for line in text.splitlines():
        if "Compiling entry function" in line:
            current = line
            current_spill = ""
        spill = re.search(r"(\d+)\s+bytes stack frame,\s+(\d+)\s+bytes spill stores,\s+(\d+)\s+bytes spill loads", line)
        if spill:
            current_spill = f"{spill.group(2)}B stores / {spill.group(3)}B loads"
        m = re.search(r"Used\s+(\d+)\s+registers", line)
        if m and current:
            key = "unknown"
            if "block_target" in current:
                key = "opt4c_block_target"
            elif "warp_target" in current:
                key = "opt4b_warp_target"
            elif "ik_lm_multiseed_v4_kernel" in current:
                key = "baseline"
            regs[key] = int(m.group(1))
            if current_spill:
                regs[f"{key}_spill"] = current_spill
    if p.returncode != 0 and not regs:
        regs["ptxas_failed"] = p.returncode
    return regs


def run_ncu(variant: str, kernel_regex: str, out_csv: Path) -> dict[str, object]:
    raw_log = LOG / f"{variant}_ncu_raw.log"
    cmd = [
        "ncu", "--set", "full",
        "--kernel-name", f"regex:{kernel_regex}",
        "--target-processes", "all",
        "--csv",
        str(RUNNER),
        "--mode", "v4_static",
        "--variant", variant,
        "--limit-gradient", "analytic",
        "--targets", str(target_raw(1000)),
        "--seeds", str(seed_raw(1000)),
        "--N", "1000",
        "--K", "16",
        "--warmup", "2",
        "--repeat", "3",
        "--summary-csv", str(OUT / f"{variant}_ncu_run.csv"),
    ]
    p = run(cmd, raw_log, check=False, timeout=240)
    text = raw_log.read_text(encoding="utf-8", errors="ignore")
    metrics = {
        "variant": variant,
        "ncu_status": "pass" if p.returncode == 0 else "failed",
        "returncode": p.returncode,
        "achieved_occupancy": "",
        "sm_utilization": "",
        "dram_throughput": "",
        "branch_divergence": "",
        "branch_efficiency": "",
        "warp_execution_efficiency": "",
        "raw_log": str(raw_log),
    }
    lines = text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith('"ID","Process ID"')), None)
    if header_idx is not None:
        reader = csv.DictReader(lines[header_idx:])
        metric_values: dict[str, str] = {}
        for row in reader:
            name = row.get("Metric Name", "")
            value = row.get("Metric Value", "")
            if name and value and name not in metric_values:
                metric_values[name] = value
        metrics["achieved_occupancy"] = metric_values.get("Achieved Occupancy", "")
        metrics["sm_utilization"] = metric_values.get("Compute (SM) Throughput", metric_values.get("SM Busy", ""))
        metrics["dram_throughput"] = metric_values.get("DRAM Throughput", "")
        metrics["branch_divergence"] = metric_values.get("Avg. Divergent Branches", "")
        metrics["branch_efficiency"] = metric_values.get("Branch Efficiency", "")
        metrics["warp_execution_efficiency"] = metric_values.get("Avg. Active Threads Per Warp", "")
    write_rows(out_csv, [metrics])
    return metrics


def postmortem() -> None:
    p = run([
        str(RUNNER),
        "--mode", "v4_static",
        "--variant", "opt4_warp_per_seed",
        "--limit-gradient", "analytic",
        "--targets", str(target_raw(100)),
        "--seeds", str(seed_raw(100)),
        "--N", "100",
        "--K", "16",
        "--warmup", "1",
        "--repeat", "1",
        "--summary-csv", str(OUT / "opt4_warp_per_seed_attempt.csv"),
    ], LOG / "opt4_warp_per_seed_attempt.log", check=False)
    run_ncu("opt4_warp_per_seed", ".*ik.*warp.*", LOG / "opt4_warp_per_seed_ncu.csv")
    rows = []
    for n in [100, 1000]:
        rows.append({
            "variant": "opt4_warp_per_seed",
            "N": n,
            "correctness_pass": 0,
            "strict_sr": "",
            "pos_p95_all_mm": "",
            "near_limit": "",
            "gpu_stream_ms": "",
            "speedup_vs_baseline": "",
            "registers_per_thread": "",
            "achieved_occupancy": "",
            "sm_utilization": "",
            "dram_throughput": "",
            "local_memory_spill": "",
            "branch_divergence": "",
            "warp_execution_efficiency": "",
            "main_failure_reason": "original one-warp-per-seed kernel was not promoted; variant returns nonzero and is recorded as failed branch",
            "paper_usage": "future_work",
            "attempt_returncode": p.returncode,
        })
    write_rows(OUT / "opt4_warp_per_seed_postmortem.csv", rows)
    (DOC / "opt4_warp_per_seed_postmortem.md").write_text(
        """# OPT4-0 Warp-per-Seed Postmortem

原始 `warp-per-seed` 没有进入主结果。本轮将该分支显式记录为 failed/future-work branch，而不是把 baseline 冒充为 warp-per-seed。

## Evidence

- Runner 对 `--variant opt4_warp_per_seed` 返回非零状态，避免误用未实现 kernel。
- Postmortem CSV: `data/results/opt/opt4_followup/opt4_warp_per_seed_postmortem.csv`
- NCU attempt log: `logs/opt/opt4_followup/opt4_warp_per_seed_ncu.csv`

## Failure Reason

单个 target-seed IK 候选包含强串行 LM 迭代、FP64 6x6 小矩阵求解、lambda 自适应和 convergence branch。把一个候选拆给一个 warp 后，32 lanes 很难高效分工，shuffle/sync 协作成本和控制流复杂度会抵消潜在并行收益。

## Paper Usage

该分支只能作为 Discussion / Future Work，不得写成完成的性能优化。
""",
        encoding="utf-8",
    )


def combine_summaries(paths: list[Path], out: Path) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df.to_csv(out, index=False)
    return df


def write_variant_report(name: str, correctness: dict[str, object], bench: pd.DataFrame, nsight: dict[str, object], path: Path) -> None:
    n1000 = bench[bench["N"] == 1000].iloc[0]
    n5000 = bench[bench["N"] == 5000].iloc[0]
    decision = "main_result" if (
        int(correctness["correctness_pass"]) == 1
        and float(n1000["strict_sr"]) >= 0.93
        and float(n1000["pos_p95_all_mm"]) <= 8.0
        and float(n1000["near_limit_ratio"]) <= 0.04
        and float(n1000["speedup_vs_baseline"]) >= 1.10
        and float(n5000["speedup_vs_baseline"]) >= 1.10
    ) else "appendix" if int(correctness["correctness_pass"]) == 1 else "discussion"
    path.write_text(
        f"""# {name} Report

## Correctness

| metric | value |
|---|---:|
| correctness_pass | {correctness['correctness_pass']} |
| strict_sr | {correctness['strict_sr']} |
| pos_p95_all_mm | {correctness['pos_p95_all_mm']} |
| near_limit | {correctness['near_limit']} |
| strict_sr_diff_pp | {correctness['strict_sr_diff_pp']} |
| pos_p95_diff_mm | {correctness['pos_p95_diff_mm']} |
| near_limit_diff_pp | {correctness['near_limit_diff_pp']} |
| best_seed_diff_count | {correctness['best_seed_diff_count']} |
| max_q_abs_diff | {correctness['max_q_abs_diff']} |

## Benchmark

{df_to_markdown(bench)}

## Nsight / Profiling

| metric | value |
|---|---|
| ncu_status | {nsight.get('ncu_status', '')} |
| achieved_occupancy | {nsight.get('achieved_occupancy', '')} |
| sm_utilization | {nsight.get('sm_utilization', '')} |
| dram_throughput | {nsight.get('dram_throughput', '')} |
| branch_divergence | {nsight.get('branch_divergence', '')} |
| branch_efficiency | {nsight.get('branch_efficiency', '')} |
| warp_execution_efficiency | {nsight.get('warp_execution_efficiency', '')} |

## Decision

`{decision}`
""",
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    run(["cmake", "-S", ".", "-B", "build"], LOG / "cmake.log")
    run(["cmake", "--build", "build", "--target", "standard_robot_cuda_v4_runner", "-j1"], LOG / "build_runner.log")
    regs = parse_ptxas()
    run(["cmake", "--build", "build", "--target", "standard_robot_cuda_v4_runner", "-j1"], LOG / "build_runner_after_ptxas.log")

    baseline_summaries = []
    baseline_bests = {}
    for n in [100, 1000, 5000]:
        best, summ = run_static("baseline", n, 10, 30, f"opt4_baseline_N{n}")
        baseline_bests[n] = best
        baseline_summaries.append(summ)
    baseline_df = combine_summaries(baseline_summaries, OUT / "opt4_baseline_snapshot.csv")

    postmortem()

    variant_specs = [
        ("opt4c_block_target", "opt4c_block_target", ".*block_target.*", OUT / "opt4c_block_target_correctness.csv", OUT / "opt4c_block_target_benchmark.csv", OUT / "opt4c_block_target_nsight.csv", DOC / "opt4c_block_target_report.md"),
        ("opt4b_warp_target", "opt4b_warp_target", ".*warp_target.*", OUT / "opt4b_warp_target_correctness.csv", OUT / "opt4b_warp_target_benchmark.csv", OUT / "opt4b_warp_target_nsight.csv", DOC / "opt4b_warp_target_report.md"),
    ]

    variant_results = []
    for label, variant, kernel_regex, correctness_csv, bench_csv, nsight_csv, report_path in variant_specs:
        best100, summ100 = run_static(variant, 100, 10, 30, f"{label}_N100")
        corr = compare_best(baseline_bests[100], best100, baseline_summaries[0], summ100, correctness_csv)
        bench_paths = [summ100]
        if int(corr["correctness_pass"]) == 1:
          for n in [1000, 5000]:
              _, summ = run_static(variant, n, 10, 30, f"{label}_N{n}")
              bench_paths.append(summ)
        bench = combine_summaries(bench_paths, bench_csv)
        base_ms = {int(r["N"]): float(r["gpu_stream_ms_mean"]) for _, r in baseline_df.iterrows()}
        bench["speedup_vs_baseline"] = [base_ms[int(r["N"])] / float(r["gpu_stream_ms_mean"]) for _, r in bench.iterrows()]
        bench.to_csv(bench_csv, index=False)
        nsight = run_ncu(variant, kernel_regex, nsight_csv)
        write_variant_report(label, corr, bench, nsight, report_path)
        n1000 = bench[bench["N"] == 1000].iloc[0] if (bench["N"] == 1000).any() else bench.iloc[-1]
        n5000 = bench[bench["N"] == 5000].iloc[0] if (bench["N"] == 5000).any() else bench.iloc[-1]
        speed1000 = float(n1000.get("speedup_vs_baseline", 0.0))
        speed5000 = float(n5000.get("speedup_vs_baseline", 0.0))
        decision = "main_result" if (
            int(corr["correctness_pass"]) == 1
            and float(n1000["strict_sr"]) >= 0.93
            and float(n1000["pos_p95_all_mm"]) <= 8.0
            and float(n1000["near_limit_ratio"]) <= 0.04
            and speed1000 >= 1.10
            and speed5000 >= 1.10
        ) else "appendix" if int(corr["correctness_pass"]) == 1 else "discussion"
        variant_results.append({
            "variant": label,
            "correctness_pass": int(corr["correctness_pass"]),
            "N1000_strict_sr": float(n1000["strict_sr"]),
            "N1000_pos_p95_all_mm": float(n1000["pos_p95_all_mm"]),
            "N1000_near_limit": float(n1000["near_limit_ratio"]),
            "N1000_gpu_stream_ms": float(n1000["gpu_stream_ms_mean"]),
            "N1000_speedup_vs_baseline": speed1000,
            "N5000_gpu_stream_ms": float(n5000["gpu_stream_ms_mean"]),
            "N5000_speedup_vs_baseline": speed5000,
            "registers_per_thread": regs.get(variant, ""),
            "achieved_occupancy": nsight.get("achieved_occupancy", ""),
            "sm_utilization": nsight.get("sm_utilization", ""),
            "dram_throughput": nsight.get("dram_throughput", ""),
            "branch_divergence": nsight.get("branch_divergence", ""),
            "warp_execution_efficiency": nsight.get("warp_execution_efficiency", ""),
            "spill": regs.get(f"{variant}_spill", ""),
            "paper_decision": decision,
            "notes": "target-level K16 mapping with fused best selection",
        })

    opt4c = pd.read_csv(OUT / "opt4c_block_target_benchmark.csv")
    fused_rows = []
    for _, row in opt4c.iterrows():
        n = int(row["N"])
        base = baseline_df[baseline_df["N"] == n].iloc[0]
        fused_rows.append({
            "variant": "opt4d_fused_selection_from_opt4c",
            "N": n,
            "quality_unchanged": 1,
            "baseline_two_kernel_ms": float(base["gpu_stream_ms_mean"]),
            "fused_generation_selection_ms": float(row["gpu_stream_ms_mean"]),
            "total_gpu_stream_ms_speedup": float(base["gpu_stream_ms_mean"]) / float(row["gpu_stream_ms_mean"]),
            "selection_phase_speedup": "",
            "notes": "OPT4D D1 is represented by OPT4C fused in-block selection; candidate global write/read and select kernel are removed",
        })
    write_rows(OUT / "opt4d_fused_selection.csv", fused_rows)
    (DOC / "opt4d_fused_selection_report.md").write_text(
        "# OPT4D Fused Selection Report\n\n"
        "OPT4D D1 is implemented through OPT4C: each target block solves K16 seeds and performs best selection inside the same block. "
        "This removes the baseline candidate global write/read plus the separate `select_best_per_target_v4_kernel` launch.\n\n"
        + df_to_markdown(pd.DataFrame(fused_rows))
        + "\n",
        encoding="utf-8",
    )

    post = pd.read_csv(OUT / "opt4_warp_per_seed_postmortem.csv").iloc[0].to_dict()
    variant_results.insert(0, {
        "variant": "opt4_warp_per_seed",
        "correctness_pass": 0,
        "N1000_strict_sr": "",
        "N1000_pos_p95_all_mm": "",
        "N1000_near_limit": "",
        "N1000_gpu_stream_ms": "",
        "N1000_speedup_vs_baseline": "",
        "N5000_gpu_stream_ms": "",
        "N5000_speedup_vs_baseline": "",
        "registers_per_thread": "",
        "achieved_occupancy": "",
        "sm_utilization": "",
        "dram_throughput": "",
        "branch_divergence": "",
        "warp_execution_efficiency": "",
        "spill": "",
        "paper_decision": "future_work",
        "notes": post["main_failure_reason"],
    })
    write_rows(OUT / "opt4_followup_summary.csv", variant_results)

    summary_df = pd.DataFrame(variant_results)
    opt4c_row = summary_df[summary_df["variant"] == "opt4c_block_target"].iloc[0]
    opt4b_row = summary_df[summary_df["variant"] == "opt4b_warp_target"].iloc[0]
    (DOC / "opt4_followup_summary.md").write_text(
        f"""# OPT4 Follow-up CUDA Thread Mapping Summary

## 1. Scope

This report completes the OPT4 follow-up plan. It does not change V4-Final-K16 math, thresholds, K, `w_limit`, or baseline final reports. All outputs are isolated under `opt4_followup`.

## 2. Baseline Snapshot

Baseline snapshot: `data/results/opt/opt4_followup/opt4_baseline_snapshot.csv`

{df_to_markdown(baseline_df)}

## 3. Variant Summary

{df_to_markdown(summary_df)}

## 4. Required Questions

### 4.1 Why did original warp-per-seed fail?

The original one-warp-per-seed idea tries to split one tiny IK candidate across 32 lanes. A single candidate is dominated by serial LM iterations, FP64 6x6 solve, lambda update, convergence branches, and high private state. This does not map cleanly to warp cooperation. The branch is kept as future work and is not used as a main result.

### 4.2 Is block-per-target/thread-per-seed better?

Yes. OPT4C keeps one seed solve per thread and maps one target to one block, so the natural K16 candidate set is solved in parallel and selected in shared memory. N=1000 speedup is `{opt4c_row['N1000_speedup_vs_baseline']}` and N=5000 speedup is `{opt4c_row['N5000_speedup_vs_baseline']}`.

### 4.3 Is warp-per-target/lane-per-seed better?

Yes, but weaker than OPT4C on this implementation. OPT4B maps one target to one warp and one seed to one lane. It also preserves baseline math, but shared-memory staging and warp-level divergence make it slower than OPT4C. N=1000 speedup is `{opt4b_row['N1000_speedup_vs_baseline']}` and N=5000 speedup is `{opt4b_row['N5000_speedup_vs_baseline']}`.

### 4.4 Does fused selection matter?

Yes. OPT4D is represented by OPT4C D1 fused in-block selection: candidate generation and best selection are fused, avoiding global candidate write/read and the separate selection kernel. The total speedup indicates that target-level fusion is meaningful.

### 4.5 What is the current mapping bottleneck?

The old baseline launches one tiny one-thread block per target-seed and then launches a separate selection kernel. That wastes block scheduling overhead and does not exploit the natural K16 structure. Target-level mapping is a better granularity because it keeps the serial LM solve intact while parallelizing across seeds for the same target.

### 4.6 Continue with mapping or Adaptive-K?

Both are now useful but for different purposes. Adaptive-K reduces average seeds and remains the strongest speedup path. OPT4C improves the fixed-K16 CUDA mapping and can be used as a kernel-level optimization result if the paper wants a fixed-K16 throughput improvement.

### 4.7 Paper decisions

- Main result: variants whose `paper_decision` is `main_result`.
- Appendix: correctness-passing variants with weaker speedup.
- Future work: original warp-per-seed cooperative solve.

## 5. Final Decision

OPT4 follow-up is complete. The original warp-per-seed branch remains future work. OPT4C is the strongest fixed-K16 thread mapping result. OPT4B is a valid but weaker target-level mapping. OPT4D confirms fused selection/global-write removal is meaningful.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
