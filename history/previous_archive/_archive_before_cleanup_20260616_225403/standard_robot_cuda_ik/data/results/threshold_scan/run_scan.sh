#!/usr/bin/env bash
# 三档组合阈值全量扫描脚本
# 用法: cd standard_robot_cuda_ik && bash data/threshold_scan/run_scan.sh
# 输出: data/threshold_scan/results/combo/
set -euo pipefail

BIN=./build/standard_robot_cuda_runner_A7
OUT_DIR=data/threshold_scan/results/combo
mkdir -p "$OUT_DIR"

# 三档阈值定义 (按 docs/修改意见/8.md)
# Loose:  30mm / 10°
# Medium: 10mm /  5°
# Strict:  5mm /  1°
declare -A TIERS
TIERS[loose]="0.03:0.1745329252"
TIERS[medium]="0.01:0.0872664626"
TIERS[strict]="0.005:0.0174532925"

N_VALUES=(100 500 1000 5000)

echo "=== CUDA B5 三档组合阈值扫描 ==="
echo "Repeat=30 | zero_seed | max_iter=160 | weight_level=2"
echo ""

for TIER in loose medium strict; do
    IFS=':' read -r POS ROT <<< "${TIERS[$TIER]}"
    for N in "${N_VALUES[@]}"; do
        LOG="$OUT_DIR/cuda_b5_${TIER}_N${N}.log"
        echo "[$(date +%H:%M:%S)] CUDA B5 ${TIER} (pos=${POS}m rot=${ROT}rad) N=${N} ..."
        "$BIN" \
            --targets "data/targets/ur10_seed42_N${N}.bin" \
            --seeds "data/seeds/ur10_seed42_zero_seed_N${N}.bin" \
            --max-iter 160 --weight-level 2 --repeat 30 \
            --ablation-level 7 \
            --pos-tol "$POS" --rot-tol "$ROT" \
            > "$LOG" 2>&1
        TP=$(grep throughput_targets_per_s "$LOG" | cut -d= -f2)
        CONV=$(grep convergence_rate "$LOG" | cut -d= -f2)
        echo "  → TP=${TP} ConvRate=${CONV}"
    done
done

echo ""
echo "=== cuRobo 三档组合阈值扫描 ==="
echo "（使用 benchmark/run_curobo_combo.py）"
python3 benchmark/run_curobo_combo.py

echo ""
echo "=== 全量扫描完成 ==="
echo "结果目录: $OUT_DIR"
echo "CUDA B5: $(ls $OUT_DIR/cuda_b5_*.log | wc -l) 个日志文件"
echo "cuRobo:  $(ls $OUT_DIR/curobo_*.log | wc -l) 个日志文件"
