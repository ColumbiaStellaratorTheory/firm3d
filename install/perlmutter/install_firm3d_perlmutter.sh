#!/bin/bash

# load cuda module to build gpu code
module load cudatoolkit

# Function to check last command's success
check_success() {
    if [ $? -ne 0 ]; then
        echo "Error: $1. Exiting."
        exit 1
    fi
}

# Load modules
module load python cray-hdf5/1.14.3.7 cray-netcdf/4.9.2.1

# Ensure Conda is available
type conda >/dev/null 2>&1 || { echo "conda command not found. Please install Anaconda/Miniconda first."; exit 1; }

echo "Adding conda-forge channel..."
conda config --add channels conda-forge
check_success "Failed to add conda-forge channel"

echo "Enter the name for the new conda environment (e.g., firm3d):"
read -p "Your input: " env_name
# Add validation for env_name if needed

echo "Creating conda environment: $env_name"
conda create -n "$env_name" --clone nersc-python
check_success "Failed to create conda environment $env_name"

echo "Activating conda environment: $env_name"
source activate "$env_name" || conda activate "$env_name"
check_success "Failed to activate conda environment $env_name"

echo "Installing Boost headers..."
conda install -y -c conda-forge libboost-headers
check_success "Failed to install Boost headers"

# FIRM3D Installation
cd firm3d || { echo "Error: firm3d directory not found. Exiting."; exit 1; }
export CI=True
env CC=cc CXX=CC pip install -e .
check_success "Failed to install FIRM3D"
cd ..

pip install booz_xform
check_success "Failed to install BOOZ_XFORM"

echo "Successfully installed FIRM3D into the conda environment '$env_name'"
echo "To activate, run: conda activate $env_name"
