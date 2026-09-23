#!/bin/bash

# Function to check last command's success
check_success() {
    if [ $? -ne 0 ]; then
        echo "Error: $1. Exiting."
        exit 1
    fi
}

# Ensure Conda is available
type conda >/dev/null 2>&1 || { echo "conda command not found. Please install Anaconda/Miniconda first."; exit 1; }

echo "Adding conda-forge channel..."
conda config --add channels conda-forge
check_success "Failed to add conda-forge channel"

echo "Enter the name or full path for the new conda environment (e.g., firm3d or /path/to/envs/firm3d):"
read -p "Your input: " env_name
# A path (contains a slash) is a conda prefix; anything else is an env name.
case "$env_name" in
    */*) env_flag=(--prefix "$env_name") ;;
    *)   env_flag=(--name "$env_name") ;;
esac

echo "Creating conda environment: $env_name"
# At 2025-12-02, NERSC Conda had Python 3.11.7:
conda create "${env_flag[@]}" python=3.11.7
check_success "Failed to create conda environment $env_name"

echo "Activating conda environment: $env_name"
source activate "$env_name" || conda activate "$env_name"
check_success "Failed to activate conda environment $env_name"

# Install Dependencies
echo "Installing FIRM3D dependencies..."
conda install -y compilers netcdf-fortran openmpi-mpicc openmpi-mpifort openblas scalapack gsl matplotlib "${env_flag[@]}"
pip install mpi4py
check_success "Failed to install FIRM3D dependencies"

# FIRM3D Installation
cd firm3d || { echo "Error: firm3d directory not found. Exiting."; exit 1; }
env CC=$CONDA_PREFIX/bin/mpicc CXX=$CONDA_PREFIX/bin/mpicxx pip install -e .
check_success "Failed to install FIRM3D"
cd ..

# BOOZ_XFORM Installation
cd booz_xform || { echo "Error: booz_xform directory not found. Exiting."; exit 1; }
env CC=$CONDA_PREFIX/bin/mpicc CXX=$CONDA_PREFIX/bin/mpicxx pip install -e .
check_success "Failed to install BOOZ_XFORM"
cd ..

echo "Successfully installed FIRM3D into the conda environment '$env_name'"
echo "To activate, run: conda activate $env_name"
