This example traces 25000 particles in the Wistell-A configuration scaled to
the size and field strength of ARIES-CS in the presence of a SAW. Particles
are initialized proportional to the fusion reactivity profile and traced
until they reach the boundary (s=1) or the elapsed time is 1e-3 seconds.

Because the waves do work on the particles, the magnetic moment is given per
particle rather than the kinetic energy, as it is for
trace_particles_boozer_perturbed on the CPU.

On perlmutter (04.20.25), the wallclock time was about 84 seconds using the
attached slurm script.
