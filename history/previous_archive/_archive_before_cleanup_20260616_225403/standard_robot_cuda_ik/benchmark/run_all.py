from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_cuda_6dof import run_cuda_6dof_benchmark
from bench_curobo import run_curobo_benchmark
from bench_kdl import run_kdl_benchmark
from bench_numeric_dls import run_numeric_dls_benchmark
from bench_pyroki import run_pyroki_benchmark
from common import BenchmarkResult, RESULTS_ROOT, format_markdown_table, save_error_log, save_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="ur10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--solver", choices=["all", "cuda", "curobo", "pyroki", "kdl", "numeric_dls"], default="all")
    parser.add_argument("--seed-strategy", default="zero_seed")
    parser.add_argument("--ablation-level", type=int, default=None, choices=[0,1,2,3,4,5,6,7],
                        help="CUDA ablation level (0-7). Uses standard binary when omitted.")
    parser.add_argument("--pos-tol", type=float, default=None,
                        help="Position tolerance in meters (default: Medium 0.01)")
    parser.add_argument("--rot-tol", type=float, default=None,
                        help="Rotation tolerance in radians (default: Medium 0.08727)")
    parser.add_argument("--publish-log", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = {
        "cuda": run_cuda_6dof_benchmark,
        "curobo": run_curobo_benchmark,
        "pyroki": run_pyroki_benchmark,
        "kdl": run_kdl_benchmark,
        "numeric_dls": run_numeric_dls_benchmark,
    }
    selected = list(mapping.keys()) if args.solver == "all" else [args.solver]
    results = []
    summary_paths = []
    error_paths = []
    for solver in selected:
        try:
            kwargs = dict(robot=args.robot, seed=args.seed, N=args.N, repeat=args.repeat, seed_strategy=args.seed_strategy)
            if solver in ("cuda", "curobo"):
                if args.pos_tol is not None:
                    kwargs["pos_tol"] = args.pos_tol
                if args.rot_tol is not None:
                    kwargs["rot_tol"] = args.rot_tol
            if solver == "cuda" and args.ablation_level is not None:
                kwargs["ablation_level"] = args.ablation_level
            result = mapping[solver](**kwargs)
            json_path, _ = save_summary(result, args.robot, args.seed, args.N)
            results.append(result)
            summary_paths.append(json_path)
        except Exception as exc:
            err_path = save_error_log(
                solver=solver,
                robot=args.robot,
                seed=args.seed,
                N=args.N,
                repeat=args.repeat,
                seed_strategy=args.seed_strategy,
                exc=exc,
            )
            error_paths.append(err_path)
            result = BenchmarkResult(
                solver_name=solver,
                robot_model=args.robot,
                num_targets=args.N,
                repeat_count=args.repeat,
                uses_gpu=solver in {"cuda", "curobo", "pyroki"},
                seed_strategy=args.seed_strategy,
            )
            result.notes.append(f"FAILED: {type(exc).__name__}: {exc}")
            results.append(result.finalize())

    competition_lines = []
    cuda_result = next((r for r in results if r.solver_name == "cuda"), None)
    if cuda_result is not None:
        best_throughput = max(results, key=lambda r: r.throughput_targets_per_s)
        best_conv = max(results, key=lambda r: r.convergence_rate)
        competition_lines.extend(["## Competitive Optimization Status", ""])
        competition_lines.append(
            f"- Current CUDA throughput: `{cuda_result.throughput_targets_per_s:.1f}` targets/s, "
            f"convergence: `{cuda_result.convergence_rate:.3f}`."
        )
        competition_lines.append(
            f"- Best throughput solver in this run: `{best_throughput.solver_name}` "
            f"({best_throughput.throughput_targets_per_s:.1f} targets/s)."
        )
        competition_lines.append(
            f"- Best convergence solver in this run: `{best_conv.solver_name}` "
            f"({best_conv.convergence_rate:.3f})."
        )
        if best_throughput.solver_name != "cuda" or best_conv.solver_name != "cuda":
            competition_lines.extend(
                [
                    "- CUDA is not yet dominant on all primary metrics. Next optimization loop must keep the same target/seed/tolerance setup and prioritize:",
                    "  block/warp remapping, shared-memory layout cleanup, register pressure reduction, seed/warm-start strategy, and tighter end-to-end copy/launch organization.",
                    "- Do not change test data, seed strategy, TCP, or convergence thresholds to make CUDA look better.",
                    "",
                ]
            )

    md = [
        "# Official UR10 Solver Benchmark",
        "",
        f"- Robot: `{args.robot}`",
        f"- Seed: `{args.seed}`",
        f"- N: `{args.N}`",
        f"- Repeat: `{args.repeat}`",
        f"- Seed strategy: `{args.seed_strategy}`",
        "",
        format_markdown_table(results),
        "",
        *competition_lines,
        "## Summary Files",
        "",
    ]
    for path in summary_paths:
        md.append(f"- `{path}`")
    if error_paths:
        md.extend(["", "## Error Logs", ""])
        for path in error_paths:
            md.append(f"- `{path}`")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    stem = f"{args.robot}_{args.solver}_N{args.N}_seed{args.seed}_repeat{args.repeat}_{args.seed_strategy}"
    out_path = RESULTS_ROOT / f"{stem}.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_path}")
    if args.publish_log:
        published = Path(__file__).resolve().parents[2] / "docs" / "logs" / "official_ur10_solver_benchmark.md"
        published.write_text("\n".join(md), encoding="utf-8")
        print(f"Wrote {published}")


if __name__ == "__main__":
    main()
