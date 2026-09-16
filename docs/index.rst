.. FIRM3D documentation master file, created by
   sphinx-quickstart on Mon Jan 01 00:00:00 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FIRM3D's documentation!
==================================

FIRM3D (Fast Ion Reduced Models in 3D) is a software suite for modeling of energetic particle dynamics in 3D magnetic fields. The guiding center equations of motion are integrated in magnetic fields represented in Boozer coordinates, including VMEC equilibria and Alfvén eigenmodes from AE3D or FAR3D.

The core routines are based on `SIMSOPT <https://simsopt.readthedocs.io>`_, but have been extended to include additional physics and diagnostics that are not typically required in the optimization context. This standalone framework enables more modular development of FIRM3D with minimal dependencies.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   README
   installation
   magnetic_fields
   shear_alfven_waves
   guiding_center
   collisions
   poincare_maps
   stopping_criteria
   trajectory_saving
   magnetic_axis
   solvers
   api
   examples
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
