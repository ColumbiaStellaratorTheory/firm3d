This example traces 1000 particles in the Wistell-A configuration scaled to
the size and field strength of ARIES-CS, in Cartesian coordinates with the
field computed from coils, saving each trajectory rather than only its final
state. Particles are initialized uniformly on the s=0.3 surface and traced
until they cross the plasma boundary or the elapsed time is 1e-4 seconds,
with the state recorded every 1e-6 seconds.

Each trajectory is written to trajectories.h5 as its own dataset, of shape
(nsaved, 5), whose rows are (t, x, y, z, vpar); a particle that was lost has
fewer rows. The example then traces the same particles in one uninterrupted
call and reports how far the two final states differ.
