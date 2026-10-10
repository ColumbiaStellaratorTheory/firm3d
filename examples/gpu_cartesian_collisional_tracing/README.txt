This example traces 10000 alpha particles in the Wistell-A configuration, including Monte Carlo Coulomb collisions with a 50/50 DT background and its electrons. Particles are initialized proportional to the fusion reactivity profile and traced until they leave the plasma boundary or the elapsed time is 2e-1 seconds. Over that window the confined population thermalizes to about 24% of the birth energy, while 0.88% of particles are lost, carrying 0.50% of the birth energy to the wall.

The thermal profiles are functions of the normalized flux s, which a Cartesian state does not carry, so this entry point takes a flux_label callable mapping cylindrical points (r, phi, z) to s. Its values are interpolated alongside the magnetic field and evaluated at each particle after every accepted step. Here the label is built by mapping a dense Boozer grid of the matching equilibrium forward to cylindrical coordinates and answering queries with the nearest sample; any callable finite over the interpolation grid will do, and values above 1 outside the plasma are clamped by the profile lookup.

The coil field and the equilibrium supplying the label must describe the same device. The label here is built from wout_aten_rescaled.nc, the file the coil set and the boundary classifier come from. coils.curves_22_7_21 holds the stellarator-symmetric half of a full-torus 40-coil set, so only stellsym is applied to it.

This traces the field of a particular coil set, not the equilibrium field itself, so its loss fraction is not expected to match the two Boozer collisional examples.

On perlmutter (08.17.26), the wallclock time is about 10 minutes using the attached slurm script.
