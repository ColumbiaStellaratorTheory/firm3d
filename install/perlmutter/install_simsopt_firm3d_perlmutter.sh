#!/bin/bash
# Install firm3d (with CUDA) + simsopt into a single conda environment on Perlmutter.
# Required for GPU Cartesian tracing examples that call simsopt's BiotSavart /
# InterpolatedField alongside firm3d's trace_particles_cartesian_gpu.
#
# Run from the directory where firm3d has been cloned, e.g.:
#   cd /path/to/parent && bash firm3d/install/perlmutter/install_simsopt_firm3d_perlmutter.sh
#
# Usage notes:
#   - Requires cudatoolkit module (loaded below).
#   - firm3d itself is installed in editable mode so the repo stays live.
#   - simsopt pulls in jax; numpy is pinned to >=2.0,<2.4 to stay compatible
#     with both jax/simsopt (needs >=2.0) and numba (needs <2.4).

set -euo pipefail

check_success() {
    if [ $? -ne 0 ]; then
        echo "Error: $1. Exiting."
        exit 1
    fi
}

module load cudatoolkit python cray-hdf5/1.14.3.7 cray-netcdf/4.9.2.1

type conda >/dev/null 2>&1 || { echo "conda not found. Please load the python module first."; exit 1; }

echo "Enter the name for the new conda environment (e.g., firm3d-simsopt):"
read -r -p "Environment name: " env_name

echo "Creating conda environment '$env_name' from nersc-python..."
conda create -n "$env_name" --clone nersc-python -y
check_success "Failed to create conda environment"

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$env_name"

echo "Installing numpy (>=2.0,<2.4 for jax/simsopt + numba compatibility)..."
pip install "numpy>=2.0,<2.4"
check_success "Failed to install numpy"

echo "Installing Boost headers..."
conda install -y -c conda-forge libboost-headers
check_success "Failed to install Boost headers"

echo "Installing firm3d (with CUDA)..."
cd firm3d || { echo "Error: firm3d directory not found. Exiting."; exit 1; }
env CC=cc CXX=CC pip install -e ".[dev]"
check_success "Failed to install firm3d"
cd ..

echo "Installing simsopt..."
pip install simsopt
check_success "Failed to install simsopt"

echo ""
echo "Done. Environment '$env_name' is ready."
echo "Activate with: conda activate $env_name"
