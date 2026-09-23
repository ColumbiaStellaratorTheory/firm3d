This example computes the Weighted Birkhoff average of the canonical momentum in the beta = 2.5% QA configuration from Landreman & Buller. 

Landreman, Matt, Stefan Buller, and Michael Drevlak. "Optimization of quasi-symmetric stellarators with self-consistent bootstrap current and energetic particle confinement." Physics of Plasmas 29.8 (2022).

Particles are instantiated uniformly in phase space, they are traced for 1x10^{-2} s. Weighted Birkhoff Averaging is added to create a numerical metric (Digit Accuracy) of chaos, implemented as described in:

N. Duignan and J. D. Meiss. "Distinguishing between regular and chaotic orbits of flows by the weighted birkhoff average." Physical Nonlinear Phenomena. (2023): 449:133749.

The WBAParticles class can be used one of two ways: tracing the particles using trace_particles_boozer_perturbed or
trace_particles_boozer, and providing the trajectory output from this function, or by providing the initial
conditions for tracing, and the tracing occurring inside the WBAParticles class. The latter is less memory intensive, as the full trajectories are never shared accross all communicators, unless
they are saved with the flag save_gc_trajectories. The second method of using the class is performed in WBA_tracing_boozer.py. The same applies to the perturbed case, shown in WBA_tracing_perturbed.py.
