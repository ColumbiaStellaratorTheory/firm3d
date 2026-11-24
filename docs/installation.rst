Installation
============

FIRM3D can be installed on different platforms. The installation process involves setting up the required dependencies and building the package.

macOS
-----

All dependencies can be installed inside a conda environment using the provided install bash script located at ``install/macOS/install_firm3d_osx.sh``.

.. code-block:: bash

   # Clone the repository
   git clone <repository-url>
   cd firm3d

   # Run the installation script
   bash install/macOS/install_firm3d_osx.sh

Perlmutter (NERSC)
------------------

All dependencies can be installed inside a conda environment using the provided install bash script located at ``install/perlmutter/install_firm3d_perlmutter.sh``.

.. code-block:: bash

   # Clone the repository
   git clone <repository-url>
   cd firm3d

   # Run the installation script
   bash install/perlmutter/install_firm3d_perlmutter.sh

Landau (Columbia)
-----------------

Use ``install/landau/install_firm3d_landau.sh`` on the Landau workstation.
The script creates a new Conda environment
and builds FIRM3D using Landau's system GCC and the CUDA/MPI tools provided by
the ``nvhpc/26.9`` module.

.. code-block:: bash

   # Clone the repository
   git clone <repository-url>
   cd firm3d

   # Run the installation script
   bash install/landau/install_firm3d_landau.sh

After installation, activate it using:

.. code-block:: bash

   module load nvhpc/26.9
   conda activate <environment-name-or-path>

Manual Installation
-------------------

If you prefer to install manually or need to customize the installation:

1. **Install Dependencies**: Ensure you have the following dependencies installed:

   - Python 3.8+
   - CMake
   - C++ compiler (GCC, Clang, or MSVC)
   - NumPy
   - SciPy
   - Matplotlib
   - Boost libraries
   - Eigen
   - xtensor

   For detailed dependency versions, see the ``requirements.txt`` and ``pyproject.toml`` files in the repository root.

2. **Build the Package**:

   .. code-block:: bash

      python setup.py build_ext --inplace
      pip install -e .

   The GPU bindings are built automatically when a complete CUDA toolkit
   (``nvcc`` and ``cuda_runtime.h``) is found and usable by the C++ compiler.
   A distribution toolkit with headers in ``/usr/include`` is not visible to
   conda's compilers; build with the system compilers (``CC=gcc CXX=g++``) to
   enable it. To build without the GPU bindings even when CUDA is installed:

   .. code-block:: bash

      CMAKE_ARGS="-DUSE_CUDA=OFF" pip install -e .

3. **Verify Installation**:

   .. code-block:: python

      import firm3d
      print("FIRM3D installed successfully!")
