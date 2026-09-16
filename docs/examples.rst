Examples and Tutorials
======================

FIRM3D comes with a comprehensive set of examples demonstrating various features and use cases. These examples are located in the ``examples/`` directory and provide practical demonstrations of the library's capabilities.

Available Example Scripts
-------------------------

Fusion Distribution
~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/fusion_distribution/``

Traces 5000 particles in the Wistell-A configuration scaled to the size and field strength of ARIES-CS. Particles are initialized proportional to the fusion reactivity profile and traced until they reach the boundary (s=1) or the elapsed time is 1e-2 seconds.

.. code-block:: bash

   cd examples/fusion_distribution/
   python fusion_distribution.py

Fusion Distribution with Perturbations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/fusion_distribution_perturbed/``

Traces 5000 particles in the Landreman & Buller 2.5% beta QA configuration with an m = 1, n = 1 shear Alfvén wave. Particles are initialized proportional to the fusion reactivity profile and traced until they reach the boundary (s=1) or the elapsed time is 1e-2 seconds.

.. code-block:: bash

   cd examples/fusion_distribution_perturbed/
   python fusion_distribution_perturbed.py

Passing Frequencies
~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/passing_frequencies/``

Computes the passing particle frequencies in the Wistell-A configuration scaled to the size and field strength of ARIES-CS. Co-passing alpha particles (mu = 0, sign(v_parallel) = +1) with the alpha particle birth energy are assumed. This calculation informs the resonances and islands observed in the passing Poincaré map.

.. code-block:: bash

   cd examples/passing_frequencies/
   python passing_frequencies.py

Passing Map Analysis
~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/passing_map_perturbed_QA/``, ``examples/passing_map_perturbed_QH/``, and ``examples/passing_map_unperturbed/``

Computes the passing Poincaré map in various configurations. The perturbed examples use the Landreman & Buller 2.5% beta QA and QH configurations with shear Alfvén waves (m = 1, n = 1 for QA; m = 1, n = 2 for QH). The unperturbed example uses the Wistell-A configuration scaled to the size and field strength of ARIES-CS with co-passing alpha particles, with chaos_detection=True enabled to color the Poincaré map by the Weighted Birkhoff Average digit accuracy.

.. code-block:: bash

   cd examples/passing_map_perturbed_QA/
   python passing_map_perturbed.py

   cd examples/passing_map_perturbed_QH/
   python passing_map_perturbed.py

   cd examples/passing_map_unperturbed/
   python passing_map.py

Trapped Particle Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/trapped_frequencies/``, ``examples/trapped_map/``, and ``examples/trapped_map_QI/``

Computes trapped particle Poincaré maps and frequencies in various configurations. The trapped_frequencies example uses the Landreman & Buller 2.5% beta QA configuration with non-quasisymmetric modes artificially suppressed. The trapped_map examples use the Landreman & Buller QA configuration and the nfp = 3 vacuum QI configuration of A. Goodman. The trapped_map example enables chaos_detection=True, coloring the Poincaré map by Weighted Birkhoff Average digit accuracy.

.. code-block:: bash

   cd examples/trapped_frequencies/
   python trapped_frequencies.py

   cd examples/trapped_map/
   python trapped_map.py

   cd examples/trapped_map_QI/
   python trapped_map_QI.py

Chaos Detection with Weighted Birkhoff Averaging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/WBA_particles/``, ``examples/equilibrium_chaos_map/``, and ``examples/perturbed_chaos_map/``

Computes the Weighted Birkhoff Average (WBA) digit accuracy, a numerical measure of chaos, for guiding-center trajectories, following Duignan & Meiss (2023). WBA_particles traces particles in the Landreman & Buller 2.5% beta QA configuration and reports the fraction of phase space that is chaotic. equilibrium_chaos_map computes a chaos heatmap over the unperturbed QA phase space. perturbed_chaos_map computes the analogous heatmap in the perturbed 2.5% beta QH configuration with a shear Alfvén wave, together with the corresponding Poincaré map (phase_space_map.py) or the heatmap alone (heat_map_alone.py); this example takes about 40 minutes on 128 Perlmutter tasks.

.. code-block:: bash

   cd examples/WBA_particles/
   python WBA_tracing_boozer.py
   python WBA_tracing_perturbed.py

   cd examples/equilibrium_chaos_map/
   python phase_space_map.py

   cd examples/perturbed_chaos_map/
   python phase_space_map.py

Trajectory Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/plot_trajectory/``

Traces 1 trapped particle in the Wistell-A configuration scaled to the size and field strength of ARIES-CS. The particle is traced for 1e-4 seconds, with data saved at time intervals of 1e-7 s.

.. code-block:: bash

   cd examples/plot_trajectory/
   python plot_trajectory.py

Resolution Studies
~~~~~~~~~~~~~~~~~~

**Location**: ``examples/resolution_scan/``

Traces particles in the Landreman & Buller 2.5% beta QH configuration. A resolution scan is performed in the number of gridpoints in the field interpolant and the integration tolerance. The conservation of the canonical momentum, p_eta, is checked. With resolution = 64 and tolerance = 1e-10, the relative error converges to ~1e-8.

.. code-block:: bash

   cd examples/resolution_scan/
   python resolution_scan.py

Uniform Distributions
~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/uniform_surf_distribution/`` and ``examples/uniform_vol_distribution/``

Traces 5000 particles in the Wistell-A configuration scaled to the size and field strength of ARIES-CS. The surface distribution example initializes particles on the s=0.3 surface with positions chosen proportional to the volume element. The volume distribution example initializes particles with positions chosen proportional to the volume element throughout the volume. Particles are traced until they reach the boundary (s=1) or the elapsed time is 1e-2 seconds.

.. code-block:: bash

   cd examples/uniform_surf_distribution/
   python uniform_surf_distribution.py

   cd examples/uniform_vol_distribution/
   python uniform_vol_distribution.py

AE3D Integration
~~~~~~~~~~~~~~~~~

**Location**: ``examples/tracing_with_AE/``

Demonstrates the selection of an eigenmode from AE3D and postprocessing of the mode to perform guiding center tracing using the Wistell-A equilibrium. First, using the notebook ChooseAE.ipynb, the spectrum computed from AE3D and stellgap is visualized to identify global eigenmodes with respect to the continuum gap structure. Then, the script tracing_with_AE.py performs guiding center tracing using a fusion birth distribution function and plots the loss fraction as a function of time.

.. code-block:: bash

   cd examples/tracing_with_AE/
   python tracing_with_AE.py

Running All Examples
--------------------

A script is provided to run all examples:

.. code-block:: bash

   cd examples/
   bash run_all_examples.sh

This will execute all example scripts and generate output files for comparison.
