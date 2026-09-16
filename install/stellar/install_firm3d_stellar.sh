#!/bin/bash

# --- Helper function ---
check_success() {
    if [ $? -ne 0 ]; then
        echo "Error: $1. Exiting."
        exit 1
    fi
}

# --- Load system modules ---
module purge
module load anaconda3/2021.11
module load gcc/8
module load openmpi/gcc/4.1.0
module load netcdf/gcc/hdf5-1.10.6/openmpi-4.1.0/4.7.4
module load hdf5/gcc/openmpi-4.1.0/1.10.6
module load scalapack/openmpi-4.1.0/2.0.2
module load openblas/0.3.x 
module load gsl/2.6

# --- Show compilers for sanity ---
export CC=$(which mpicc)
export CXX=$(which mpicxx)
export FC=$(which mpifort)
echo "MPI C compiler: $CC"
echo "MPI C++ compiler: $CXX"
echo "MPI fortran compiler: $FC"

source /usr/licensed/anaconda3/2022.10/etc/profile.d/conda.sh

# --- Create environment ---
echo "Adding conda-forge channel..."
conda config --add channels conda-forge
check_success "Failed to add conda-forge channel"

echo "Enter the name for the new conda environment (e.g., firm3d):"
read -p "Your input; " env_name
conda create -n "$env_name" -y python=3.9 --no-default-packages

conda activate "$env_name"
check_success "Failed to activate conda environment $env_name"
export LD_LIBRARY_PATH=/usr/local/openmpi/4.1.0/gcc/lib64:$LD_LIBRARY_PATH

echo "Installing FIRM3D dependencies..."
#conda remove mpi mpi4py --force -y
check_success "Failed to install FIRM3D dependencies"


echo "Building mpi4py against system MPI..."
MPICC=/usr/local/openmpi/4.1.0/gcc/bin/mpicc pip install --no-cache-dir --no-binary=:all: --force-reinstall mpi4py
check_success "Failed to build mpi4py with system MPI"

# FIRM3D Installation
cd firm3d || { echo "Error: firm3d directory not found. Exiting."; exit 1; }
export CI=True
env CC=/usr/local/openmpi/4.1.0/gcc/bin/mpicc CXX=/usr/local/openmpi/4.1.0/gcc/bin/mpicxx pip install -e .
check_success "Failed to install FIRM3D"
cd ..

# BOOZ_XFORM Installation
cd booz_xform || { echo "Error: booz_xform directory not found. Exiting."; exit 1; }
export LDFLAGS="-L/usr/local/hdf5/gcc/openmpi-4.1.0/1.10.6/lib64 -lhdf5_hl -lhdf5"
env CC=/usr/local/openmpi/4.1.0/gcc/bin/mpicc CXX=/usr/local/openmpi/4.1.0/gcc/bin/mpicxx pip install -e .
check_success "Failed to install BOOZ_XFORM"
cd ..

echo "Successfully installed FIRM3D into the conda environment '$env_name'"
echo "To activate, run: conda activate $env_name"

conda activate $env_name
python -c "from mpi4py import MPI;"
python -c "from booz_xform import Booz_xform; print('Booz_xform imported successfully')"
