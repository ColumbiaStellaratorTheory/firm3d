This example traces 1000 particles in the Wistell-A configuration scaled to
the size and field strength of ARIES-CS, in Cartesian coordinates, with the
field computed from coils rather than from an equilibrium. Particles are
initialized uniformly on the s=0.3 surface and traced until they cross the
plasma boundary, which a surface classifier detects, or the elapsed time is
1e-5 seconds. The initial and final state of every particle is written to
particle_data.csv.

On perlmutter (04.20.26), the wallclock time was about 30 seconds using the
attached slurm script.
