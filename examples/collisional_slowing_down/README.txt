This example traces 1000 alpha particles in the Wistell-A configuration, including Monte Carlo Coulomb collisions with a 50/50 DT background and its electrons. Particles are initialized proportional to the fusion reactivity profile and traced until they reach the boundary (s=1) or the elapsed time is 2e-1 seconds. This is the collisional counterpart of the fusion_distribution example, and uses the same equilibrium as the two GPU collisional examples.

Over 2e-1 seconds the confined population thermalizes to about 24% of the birth energy, while 0.2% of particles are lost, carrying 0.1% of the birth energy to the wall. This configuration confines alphas well, so the losses are small. Each lost particle takes to the wall only the energy it still had when it crossed the boundary, which collisions reduce below the birth energy, so the energy loss falls below the particle loss.

On perlmutter (08.13.26), the wallclock time is about 4 minutes using the attached slurm script.

This example and the gpu_boozer_collisional_tracing example trace the same
equilibrium field with the same equations and background, and agree: 0.2% of
particles lost and about 24% of the birth energy retained by the confined
population. The GPU Cartesian collisional example traces the field of a coil
set rather than the equilibrium field, so its loss fraction is not expected to
match.
