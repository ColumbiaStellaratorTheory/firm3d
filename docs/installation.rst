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

3. **Verify Installation**:

   .. code-block:: python

      import firm3d
      print("FIRM3D installed successfully!")
