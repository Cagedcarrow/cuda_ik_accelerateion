#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

nsys profile \
  --trace=cuda,osrt \
  --sample=none \
  --force-overwrite=true \
  --output=data/experiments/补充实验/reports/nsys_opt4c_n1000 \
  ./build/standard_robot_cuda_v4_runner \
  --mode v4_static \
  --variant opt4c_block_target \
  --limit-gradient analytic \
  --graph-mode off \
  --precision-mode fp64 \
  --fallback-mode none \
  --targets data/experiments/inputs/targets_N1000_T4x4_f64.raw \
  --seeds data/experiments/inputs/seeds_N1000_K16_q_f64.raw \
  --N 1000 \
  --K 16 \
  --max-iter 60 \
  --repeat 30 \
  --warmup 10 \
  --best-csv data/experiments/补充实验/results/nsys_opt4c_best_N1000_K16.csv \
  --summary-csv data/experiments/补充实验/results/nsys_opt4c_summary_N1000_K16.csv \
  --timing-csv data/experiments/补充实验/results/nsys_opt4c_timing_N1000_K16.csv

nsys stats \
  --force-export=true \
  --format table \
  --output data/experiments/补充实验/reports/nsys_opt4c_n1000_stats \
  data/experiments/补充实验/reports/nsys_opt4c_n1000.nsys-rep

nsys export \
  --type sqlite \
  --force-overwrite=true \
  --output data/experiments/补充实验/reports/nsys_opt4c_n1000.sqlite \
  data/experiments/补充实验/reports/nsys_opt4c_n1000.nsys-rep

python3 data/experiments/补充实验/scripts/plot_nsys_timeline.py
