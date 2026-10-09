# GPU trajectory saving and Poincaré plots

Run from the repository root in a CUDA-enabled firm3d environment:

```console
python examples/gpu_dense_output/save_trajectories.py
python examples/gpu_dense_output/plot_poincare.py
```

The first script traces 32 co-passing alpha particles in the bundled ATEN
equilibrium, in double precision. It saves every `1e-7` seconds for `1e-3`
seconds using `forget_exact_path=False` and the CPU solver's Dormand–Prince
continuous extension. Saving runs within one uninterrupted integration.
The defaults are a small demonstration; field-table resolution and integration
tolerance should also be converged for scientific use.

Files go to `examples/gpu_dense_output/output/`:

- `trajectories.npz`: numeric arrays named `particle_000000`, etc., with rows
  `(t, s, theta, zeta, vpar)`. Paths include the initial and terminal states;
  lost particles have shorter paths. The archive includes loss-hit arrays,
  initial conditions, coordinate names, equilibrium path, and solver settings.
- `trajectories.png`: radial positions versus time and projections of three
  saved orbits onto the pseudo-poloidal plane.
- `poincare.csv`: `(particle, t, s, theta, zeta, vpar)` at section crossings.
- `poincare.png`: the section in `(theta, s)` and in
  `(sqrt(s) cos(theta), sqrt(s) sin(theta))`.

Reload trajectories without pickle:

```python
import numpy as np

with np.load("examples/gpu_dense_output/output/trajectories.npz", allow_pickle=False) as saved:
    path = saved["particle_000000"]  # columns: t, s, theta, zeta, vpar
    print(path.shape, path[0], path[-1])
```

`plot_poincare.py` is CPU postprocessing and does not need a GPU. It unwraps
both Boozer angles and linearly interpolates crossings of `zeta = 0` modulo
`2*pi`, keeping positive toroidal crossings by default. These are approximate
sections from sampled trajectories, not the CPU tracer's event roots. Each
angle must advance by less than `pi` between saved samples; coarse sampling
can hide full turns or reversals. Initial points already on the plane are
excluded, and a saved point exactly on the plane is counted once.

Check section convergence by rerunning with a smaller `dt_save`, for example:

```console
python examples/gpu_dense_output/save_trajectories.py --dt-save 5e-8 --output-dir /tmp/gpu-dense-fine
python examples/gpu_dense_output/plot_poincare.py /tmp/gpu-dense-fine/trajectories.npz
```

The initial particles are deterministic, so compare crossings for the same
particle in `poincare.csv`. Changing `dt_save` changes section interpolation
accuracy without changing the adaptive GPU integration. Full histories use
memory proportional to particles times samples; the default GPU history
buffer is approximately 18 MB. Use `--nparticles`, `--tmax`, `--dt-save`,
`--resolution`, and `--tol` to adjust the trace.

The default example and the halved interval were validated on Perlmutter's
A100 80 GB GPU on 2026-10-09. Both returned 4,272 positive crossings from 32
surviving particles. Terminal states and all shared saved states were
identical. The maximum differences between the section samples were
`5.25e-10` seconds in time, `3.39e-4` in `s`, and `4.04e-4` radians in `theta`.
The largest angle advance per default save interval was `0.105` radians.
These differences compare save cadences; converge the field table and solver
tolerance separately. Analytic wrapping, direction, endpoint-counting, and
interpolation convergence checks can be run without a GPU:

```console
python -m unittest tests.test_gpu_poincare_example
```

Other section directions and angles are available:

```console
python examples/gpu_dense_output/plot_poincare.py --zeta-section 0.5 --direction both --output-dir /tmp/other-section
```
