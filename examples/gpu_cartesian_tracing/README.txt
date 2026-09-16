This example traces 1000 particles in the Wistell-A configuration scaled to the size and field strength of ARIES-CS. Particles are launched uniformly over the s=0.3 flux surface with the pitch v_par/v drawn uniformly in [-1, 1], and traced until they reach the boundary (s=1) or the elapsed time is 1e-5 seconds.

coils.curves_22_7_21 holds the stellarator-symmetric half of a full-torus 40-coil set, so only stellsym is applied to it.

On perlmutter (04.20.26), the wallclock time is about 30 seconds using the attached slurm script.
