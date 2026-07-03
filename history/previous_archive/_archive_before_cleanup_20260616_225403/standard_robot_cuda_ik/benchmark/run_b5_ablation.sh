#!/bin/bash
# Batch run all ablation levels for paper comparison
# Usage: bash benchmark/run_b5_ablation.sh

set -e
cd /mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik
RESULTS_DIR="data/results"
mkdir -p "$RESULTS_DIR"

# Map: B-name -> A-level
# B0=A0 B1=A1 B2=A2 B3=A5 B4=A6 B5=A7 B6=A8
declare -A B_TO_A
B_TO_A=([0]=0 [1]=1 [2]=2 [3]=5 [4]=6 [5]=7 [6]=8)

echo "============================================"
echo "Starting ablation benchmark runs"
echo "Started at: $(date)"
echo "============================================"

for B_LEVEL in 0 1 2 3 4 5 6; do
    A_LEVEL=${B_TO_A[$B_LEVEL]}
    BINARY="build/standard_robot_cuda_runner_A${A_LEVEL}"

    if [ ! -f "$BINARY" ]; then
        echo "WARNING: Binary $BINARY not found, skipping B${B_LEVEL}"
        continue
    fi

    # Determine N values for this ablation level
    case $B_LEVEL in
        0|3|4|5) N_VALUES="100 500 5000" ;;  # Full sweep
        1|2) N_VALUES="100" ;;                 # N=100 only (B1/B2 limited)
        6) N_VALUES="100" ;;                   # B6 = CUDA Graph, N=100
    esac

    # B5 also needs N=1000
    if [ "$B_LEVEL" == "5" ]; then
        N_VALUES="100 500 1000 5000"
    fi

    for N in $N_VALUES; do
        echo ""
        echo "--- B${B_LEVEL} (A${A_LEVEL}) N=${N} ---"
        TARGETS="data/targets/ur10_seed42_N${N}.bin"
        SEEDS="data/seeds/ur10_seed42_zero_seed_N${N}.bin"

        if [ ! -f "$TARGETS" ]; then
            echo "ERROR: Targets file $TARGETS not found"
            continue
        fi
        if [ ! -f "$SEEDS" ]; then
            echo "ERROR: Seeds file $SEEDS not found"
            continue
        fi

        $BINARY \
            --targets "$TARGETS" \
            --seeds "$SEEDS" \
            --max-iter 160 \
            --weight-level 2 \
            --repeat 30 \
            --ablation-level "$A_LEVEL" \
            2>&1 | tee "${RESULTS_DIR}/ur10_cuda_B${B_LEVEL}_N${N}_seed42_repeat30_zero_seed.log"

        echo "Completed B${B_LEVEL} N=${N} at $(date)"
    done
done

echo ""
echo "============================================"
echo "All ablation benchmarks completed at: $(date)"
echo "============================================"
