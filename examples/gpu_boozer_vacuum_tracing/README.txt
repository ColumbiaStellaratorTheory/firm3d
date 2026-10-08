This example traces 30000 particles in the ARIES-CS configuration on a GPU,
in double and in single precision, and compares the two. Particles are
initialized proportional to the fusion reactivity profile and traced until
they reach the boundary (s=1) or the elapsed time is 1e-4 seconds. The loss
fraction is printed for each precision, and the initial and final state of
every particle is written to particle_data.csv.

Single precision reproduces the loss fraction but not individual orbits; see
the GPU tracing page of the documentation.
