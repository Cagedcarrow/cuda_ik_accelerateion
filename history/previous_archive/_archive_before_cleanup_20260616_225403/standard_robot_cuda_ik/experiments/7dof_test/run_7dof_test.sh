#!/usr/bin/env bash
# run_7dof_test.sh — One-click 7DOF Panda CUDA IK verification
#
# Steps:
#   1. Check prerequisites
#   2. Generate test targets/seeds (Python FK validation)
#   3. Run CPU DLS reference
#   4. Build CUDA test executable
#   5. Verify CUDA FK matches CPU FK
#   6. Run CUDA IK solver and compare convergence
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
PASS=0
FAIL=0

check() {
  local desc="$1"
  shift
  echo ""
  echo "=========================================="
  echo "  CHECK: $desc"
  echo "=========================================="
  if "$@"; then
    echo "  ✅ PASS: $desc"
    PASS=$((PASS + 1))
  else
    echo "  ❌ FAIL: $desc"
    FAIL=$((FAIL + 1))
  fi
}

# ---- Step 0: Environment ----
echo "=========================================="
echo "  7DOF Panda CUDA IK Test Suite"
echo "=========================================="

# Check CUDA toolkit
if ! command -v nvcc &>/dev/null; then
  echo "ERROR: nvcc not found. Set PATH to CUDA toolkit."
  exit 1
fi
echo "nvcc: $(nvcc --version | grep 'release' | head -1)"

# Check cmake
if ! command -v cmake &>/dev/null; then
  echo "ERROR: cmake not found."
  exit 1
fi
echo "cmake: $(cmake --version | head -1)"

# ---- Step 1: Generate test data (if not already present) ----
echo ""
echo "=========================================="
echo "  Step 1: Generate test data"
echo "=========================================="
if [ ! -f "${SCRIPT_DIR}/panda_test_seeds_N10.bin" ] || \
   [ ! -f "${SCRIPT_DIR}/panda_test_targets_N10.bin" ]; then
  echo "Generating test targets/seeds..."
  cd "${SCRIPT_DIR}"
  python3 test_fk.py
else
  echo "Test data already exists, skipping generation."
fi

# ---- Step 2: Run CPU DLS reference ----
echo ""
echo "=========================================="
echo "  Step 2: CPU DLS reference"
echo "=========================================="
cd "${SCRIPT_DIR}"
python3 panda_fk_reference.py

# ---- Step 3: Build CUDA test ----
echo ""
echo "=========================================="
echo "  Step 3: Build CUDA test executable"
echo "=========================================="
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake "${SCRIPT_DIR}" -DCMAKE_CUDA_ARCHITECTURES=89
make -j$(nproc)
echo "Build complete: ${BUILD_DIR}/panda_7dof_test"

# ---- Step 4: CUDA FK verification ----
echo ""
echo "=========================================="
echo "  Step 4: CUDA FK verification"
echo "=========================================="
cd "${BUILD_DIR}"
check "CUDA FK matches CPU FK" ./panda_7dof_test --verify-fk

# ---- Step 5: CUDA IK solver test ----
echo ""
echo "=========================================="
echo "  Step 5: CUDA IK solver (N=10)"
echo "=========================================="
cd "${BUILD_DIR}"
./panda_7dof_test \
  --targets "${SCRIPT_DIR}/panda_test_targets_N10.bin" \
  --seeds "${SCRIPT_DIR}/panda_test_seeds_N10.bin" \
  --max-iter 160 \
  --weight-level 0

# ---- Summary ----
echo ""
echo "=========================================="
echo "  Results: ${PASS} passed, ${FAIL} failed"
echo "=========================================="
if [ ${FAIL} -eq 0 ]; then
  echo "  ✅ All checks passed!"
else
  echo "  ❌ Some checks failed."
fi
exit ${FAIL}
