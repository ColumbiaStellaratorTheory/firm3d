# GPU trajectory saving

Run the trajectory-saving example from the repository root on a GPU node:

```console
python examples/gpu_dense_output/save_trajectories.py
```

It mirrors `examples/orbit_classification/fusion_distribution_classification.py`:
5,000 fusion-born alpha particles in ARIES-CS, seed 0, cubic interpolation at
resolution 48, 10 ms integration, tolerance `1e-8`, and `dt_save=1e-7` s.
It uses the same density/temperature profiles, uniform parallel-velocity
sampling, no-K equations, loss boundary `s=1`, and `OrbitClassification` with
helicity `(M,N)=(1,0)`.

GPU batches of 128 bound the history buffer. Only lost particles are saved as
`particle_<i>_traj.txt`, `particle_<i>_hits.txt`, and `particle_<i>.npz`.
The NPZ contains the trajectory, hits, and numeric CPU classification diagnostics
(excluding optional `debug_data`). `summary.npz` records the whole ensemble's
launches, loss flags, end times, bounce counts, and configuration;
`trajectories.png` plots up to three lost orbits, or survivors if none are lost.
Results go to `examples/gpu_dense_output/output/`. Use `--nparticles`, `--tmax`,
`--dt-save`, `--resolution`, `--tol`, and `--batch-size` to adjust the run.

Mirror hits are interpolated from saved dense-output samples; converge
`--dt-save` separately from solver tolerance and keep angle changes below pi
between samples. GPU loss hits use the first accepted endpoint at or beyond
`s=1`; CPU hits use solver event roots.

Reload a numeric path without pickle:

```python
import numpy as np

directory = "examples/gpu_dense_output/output/"
with np.load(directory + "summary.npz", allow_pickle=False) as summary:
    i = np.flatnonzero(summary["lost"])[0]  # if the run has losses
with np.load(directory + f"particle_{i}.npz", allow_pickle=False) as saved:
    path = saved["trajectory"]  # t, s, theta, zeta, vpar
    trapping_states = saved["status"]  # banana=0, barely trapped=1, ripple=2
```

For a kinetic Poincaré plot, use the existing passing-map example with either
CPU or CATAPULT tracing:

```console
python examples/passing_map_unperturbed/passing_map.py --backend cpu
python examples/passing_map_unperturbed/passing_map.py --backend catapult
```

Both use the same physical parameters and `PassingPoincare` plotting helpers.
Both locate sections with dense event roots, independently of `--dt-save`.
The `--tmax` limit applies to each return on both backends; CATAPULT traces
continuously with `Nmaps * tmax` as its total upper bound. Converge `--dt-save`
separately from solver tolerance when computing WBA from saved history.
