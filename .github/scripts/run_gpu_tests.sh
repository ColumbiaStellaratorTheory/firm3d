#!/bin/bash
#SBATCH --job-name=firm3d-gpu-ci
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH --gpus-per-task=1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --time=00:30:00

# Usage: sbatch run_gpu_tests.sh <work_dir> <firm3d_dir>
#   work_dir    - directory for output files (slurm log, coverage, exit code)
#   firm3d_dir  - path to the firm3d checkout ($GITHUB_WORKSPACE on the runner)
#
# Expects a conda environment named "firm3d-ci" to already exist.
# Create it once with tests/perlmutter/setup_ci_env.sh.

set -euo pipefail

WORK_DIR="${1:?Usage: $0 <work_dir> <firm3d_dir>}"
FIRM3D_DIR="${2:?Usage: $0 <work_dir> <firm3d_dir>}"
EXIT_CODE_FILE="$WORK_DIR/test_exit_code.txt"

echo "=== firm3d GPU CI on $(hostname) ==="
echo "Work dir  : $WORK_DIR"
echo "Firm3d dir: $FIRM3D_DIR"
echo "Commit    : $(git -C $FIRM3D_DIR rev-parse HEAD)"
echo "Started  : $(date)"

# ── Modules ───────────────────────────────────────────────────────────────────
module load cudatoolkit python cray-hdf5/1.14.3.7 cray-netcdf/4.9.2.1

# ── Conda environment ──────────────────────────────────────────────────────────
# Conda isn't available in non-interactive shells without sourcing its init.
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/.conda")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate firm3d-ci

# ── Build firm3d with CUDA ────────────────────────────────────────────────────
cd "$FIRM3D_DIR"
echo "--- Installing firm3d ---"
env CC=cc CXX=CC pip install -v --no-build-isolation -e ".[dev]" 2>&1 | tee "$WORK_DIR/build.log"

if ! grep -q "GPU bindings will be compiled with" "$WORK_DIR/build.log"; then
    echo "ERROR: GPU bindings were not compiled." >&2
    echo "1" > "$EXIT_CODE_FILE"
    exit 1
fi

# ── GPU sanity check ──────────────────────────────────────────────────────────
nvidia-smi
python -c "import firm3dpp; print('firm3dpp loaded OK')"

# ── Run tests ─────────────────────────────────────────────────────────────────
# Disable exit-on-error for the test command so we can capture its exit code
# and always write EXIT_CODE_FILE even when tests fail.
echo "--- Running GPU tests ---"
set +e
python -m coverage run -m unittest tests.field.test_gpu
TEST_EXIT=$?
set -e

python -m coverage xml -o "$WORK_DIR/coverage.xml" || true

echo "$TEST_EXIT" > "$EXIT_CODE_FILE"
echo "Finished: $(date)  (exit $TEST_EXIT)"
if  [ "$TEST_EXIT" -ne 0 ]; then
    exit $TEST_EXIT
fi

# if the correctness tests were successful, run the regression tests

set +e
(cd examples/gpu_boozer_tracing && python gpu_boozer_tracing.py)
BOOZER_EXIT=$?
cp examples/gpu_boozer_tracing/gpu_boozer_tracing_results.json "$WORK_DIR/gpu_boozer_tracing_results.json"

(cd examples/gpu_saw_tracing && python gpu_saw_tracing.py)
SAW_EXIT=$?
cp examples/gpu_saw_tracing/gpu_boozer_saw_tracing_results.json "$WORK_DIR/gpu_boozer_saw_tracing_results.json"

(cd examples/gpu_cartesian_tracing && python gpu_cartesian_tracing.py)
CARTESIAN_EXIT=$?
cp examples/gpu_cartesian_tracing/gpu_cartesian_tracing_results.json "$WORK_DIR/gpu_cartesian_tracing_results.json"
set -e

echo "boozer_tracing exit    : $BOOZER_EXIT"
echo "saw_tracing exit       : $SAW_EXIT"
echo "cartesian_tracing exit : $CARTESIAN_EXIT"

if [ "$BOOZER_EXIT" -ne 0 ] || [ "$SAW_EXIT" -ne 0 ] || [ "$CARTESIAN_EXIT" -ne 0 ]; then
  EXAMPLES_EXIT=1
else
  EXAMPLES_EXIT=0
fi

echo "$EXAMPLES_EXIT" > "$EXIT_CODE_FILE"
echo "Finished: $(date)  (exit $EXAMPLES_EXIT)"
exit $EXAMPLES_EXIT