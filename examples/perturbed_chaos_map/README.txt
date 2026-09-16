This example computes the WBA in the Landreman & Buller 2.5% beta QH configuration. 

Landreman, Matt, Stefan Buller, and Michael Drevlak. "Optimization of quasi-symmetric stellarators with self-consistent bootstrap current and energetic particle confinement." Physics of Plasmas 29.8 (2022).

A single harmonic from one of the example SAWs is provided. The particles are instantiated at fixed E', and mapped as a function of the perturbed pitch angle (mu/E') and P_eta.

The file phase_space_map.py plots the phase space map with a poincare map and the perturbation strength. The file heat_map_alone.py only plots the phase space map.

Weighted Birkhoff Averaging is added as a setting to the Perturbed Passing map and applied to particle momentum to create a numerical
metric (Digit Accuracy) of chaos, implemented as described in:

N. Duignan and J. D. Meiss. "Distinguishing between regular and chaotic orbits of flows by the weighted birkhoff average." Physical Nonlinear Phenomena. (2023): 449:133749.

This script takes about 40 minutes on 128 Perlmutter tasks.