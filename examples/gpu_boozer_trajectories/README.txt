This example traces 10000 particles in the Wistell-A configuration scaled to
the size and field strength of ARIES-CS, saving each trajectory rather than
only its final state. Particles are initialized proportional to the fusion
reactivity profile and traced until they reach the boundary (s=1) or the
elapsed time is 1e-3 seconds, with the state recorded every 1e-6 seconds.

Each trajectory is written to trajectories.h5 as its own dataset, of shape
(nsaved, 5), whose rows are (t, s, theta, zeta, vpar); a particle that was
lost has fewer rows. The example then traces the same particles in one
uninterrupted call and reports how far the two final states differ, which is
a measure of the integration error rather than of the saving.
