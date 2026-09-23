This example computes the passing Poincare map in the Landreman & Buller 2.5% beta QA configuration. 

Landreman, Matt, Stefan Buller, and Michael Drevlak. "Optimization of quasi-symmetric stellarators with self-consistent bootstrap current and energetic particle confinement." Physics of Plasmas 29.8 (2022).

An m = 1, n = 1 shear Alfven wave is included, with parameters descried in 

Paul, Elizabeth J., Harry E. Mynick, and Amitava Bhattacharjee. "Fast-ion transport in quasisymmetric equilibria in the presence of a resonant Alfvénic perturbation." Journal of Plasma Physics 89.5 (2023): 905890515.

On macOS (06.29.25), the wallclock time is about 3 minutes using the command "mpiexec -n 8 python passing_map_perturbed.py".

Weighted Birkhoff Averaging is added as a setting to the perturbed passing map with the flag chaos_detection=True. Averaging is applied to particle canonical momentum to create a numerical 
metric (Digit Accuracy) of chaos, implemented as described in:

N. Duignan and J. D. Meiss. "Distinguishing between regular and chaotic orbits of flows by the weighted birkhoff average." Physical Nonlinear Phenomena. (2023): 449:133749.
