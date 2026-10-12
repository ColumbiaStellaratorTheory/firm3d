This example saves and classifies lost fusion-alpha GPU trajectories using
the same ARIES-CS equilibrium, fusion birth profile, seed, particle energy,
and solver settings as examples/orbit_classification.

The parameters near the top of save_trajectories.py control the run. Defaults
are 5000 particles, interpolation resolution 48, tolerance 1e-8, 10 ms of
tracing, and dt_save=1e-7 s. Smaller settings are used in GitHub Actions.
GPU batches of 128 bound history storage; max_hits sets the event capacity.

From this directory, on a GPU node, run:

python save_trajectories.py

For each particle lost at s=1, the script saves particle_i_traj.txt,
particle_i_hits.txt, and particle_i.npz in the current directory, matching
the CPU example. Trajectories use dense output, mirror hits use dense event
roots, and flux stops are checked at accepted endpoints. OrbitClassification
computes the trapping state and bounce diagnostics with helicity (M,N)=(1,0).
