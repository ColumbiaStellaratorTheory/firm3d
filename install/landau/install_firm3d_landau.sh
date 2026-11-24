#!/bin/bash
# Install FIRM3D on Landau with system GCC and the NVIDIA SDK's CUDA/MPI.

# Function to check last command's success
check_success() {
    local status=$?
    if [ "$status" -ne 0 ]; then
        echo "Error: $1. Exiting." >&2
        exit 1
    fi
}

# Load the shared CUDA toolkit and MPI. Compiler choices are set below.
module load nvhpc/26.9
check_success "Failed to load nvhpc/26.9"

# Ensure Conda is available.
type conda >/dev/null 2>&1 || {
    cat >&2 <<'EOF'
conda command not found. Please install or initialize Conda first.

For a new Miniforge installation, run:

curl -fL \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
  -o "$HOME/Miniforge3-Linux-x86_64.sh" &&
bash "$HOME/Miniforge3-Linux-x86_64.sh" -b -p "$HOME/miniforge3" &&
source "$HOME/miniforge3/etc/profile.d/conda.sh" &&
conda --version &&
"$HOME/miniforge3/bin/conda" init bash

Then rerun this installation script from your FIRM3D checkout.
EOF
    exit 1
}

echo "Enter the name or full path for the new conda environment (e.g., firm3d-gcc or $HOME/envs/firm3d-gcc):"
read -r -p "Your input: " env_name
check_success "Failed to read the environment name"
# A path (contains a slash) is a conda prefix; anything else is an env name.
case "$env_name" in
    */*) env_flag=(--prefix "$env_name") ;;
    *)   env_flag=(--name "$env_name") ;;
esac

echo "Creating conda environment: $env_name"
conda create "${env_flag[@]}" --override-channels -c conda-forge \
    python=3.13 pip --no-default-packages
check_success "Failed to create conda environment $env_name"

echo "Activating conda environment: $env_name"
# Conda's activate script initializes the shell and activates the environment.
source "$(conda info --base)/bin/activate" "$env_name"
check_success "Failed to activate conda environment $env_name"

echo "Installing Boost headers and GSL..."
# Keep CUDA/MPI from the SDK; only these dependencies come from Conda.
conda install -y --override-channels -c conda-forge 'libboost-headers>=1.76' gsl
check_success "Failed to install Boost headers and GSL"

# Override the module's nvc/nvc++ defaults after Conda activation.
# Landau already has system GCC/G++ 13.3, so no compiler installation is needed.
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAToolkit_ROOT="$NVHPC_ROOT/cuda/13.3"
export CUDACXX="$CUDAToolkit_ROOT/bin/nvcc"
export CUDAHOSTCXX="$CXX"
export MPI_HOME="$NVHPC_ROOT/comm_libs/mpi"
export OMPI_CC="$CC"
export OMPI_CXX="$CXX"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"

echo "CPU C++ compiler: $CXX"
"$CXX" --version
check_success "Failed to run the system C++ compiler"
echo "CUDA compiler: $CUDACXX (host compiler: $CUDAHOSTCXX)"
"$CUDACXX" --version
check_success "Failed to run the CUDA compiler"

echo "Building mpi4py against the shared MPI..."
# CMake discovers MPI and records its library directory for imports.
MPI4PY_BUILD_BACKEND=scikit-build-core \
    CMAKE_ARGS="-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON" \
    pip install --no-cache-dir --no-binary=mpi4py mpi4py
check_success "Failed to build mpi4py with the shared MPI"

# Target Landau's A100 and require CUDA/MPI discovery.
export CMAKE_ARGS="-DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 \
-DCMAKE_REQUIRE_FIND_PACKAGE_CUDAToolkit=ON \
-DCMAKE_REQUIRE_FIND_PACKAGE_MPI=ON -DMPI_CXX_SKIP_MPICXX=ON"

# FIRM3D Installation
pip install -v -e .
check_success "Failed to install FIRM3D"

echo "Successfully installed FIRM3D into the conda environment '$env_name'"
echo "To activate, run: module load nvhpc/26.9 && conda activate $env_name"
