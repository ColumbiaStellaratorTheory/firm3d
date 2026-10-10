# GPU trajectory saving and Poincaré plots

## Comparison with the existing CPU passing-map example

```console
python examples/gpu_dense_output/compare_poincare.py
```

This example follows the production parameters in
`examples/passing_map_unperturbed/passing_map.py`:

| Parameter | CPU and GPU |
| --- | --- |
| Equilibrium | `boozmn_aten_rescaled.nc` (full-resolution ATEN file) |
| Species and kinetic energy | Alpha particles at 3.5 MeV |
| Pitch and direction | `lambda = 0`, `sign_vpar = +1` |
| Launches | 120 radii, `s = j/121`, `j = 1,...,120`, `theta = zeta = 0` |
| Field interpolation | Cubic, 48 cells in each coordinate |
| Guiding-center equations | No-K, retaining `I(s)` and `G(s)` |
| Absolute and relative tolerance | `1e-8` |
| Section | Positive returns to `zeta = 0` modulo `2*pi` |
| Requested returns | 1,000 per particle |

The CPU reference runs the existing `PassingPoincare` class, including its
canonical-momentum and weighted-Birkhoff-average diagnostics. The comparison
figure colors both plots by initial radius so corresponding orbits can be
identified. It compares section coordinates, rather than the chaos diagnostic.

The GPU table samples the CPU field at the same cubic nodes. The full GPU
equations use zero `K` and zero angular derivatives of `K`, which reproduces
the CPU example's no-K equations. The script checks every table quantity at
1,000 deterministic random points, including symmetry reflections.
At the upper toroidal endpoint, it evaluates just inside the CPU table to
preserve that endpoint's derivative values instead of wrapping to zero.

The CPU class restarts at each return. The GPU integrates continuously to
the longest cumulative CPU return time plus two transits, then keeps the
first 1,000 returns of each particle. A continuous CPU trace with event roots
provides a second reference to separate restart effects from sampling errors.
CPU sampled sections are also compared with CPU event roots. GPU sections
are extracted at `dt_save = 1e-7` and `5e-8` seconds to check saving cadence.

The CPU example stops at `s = 0.99`; the GPU kernel stops at `s = 1`. GPU
postprocessing discards the path from its first saved sample at `s >= 0.99`,
so boundary comparisons are approximate to the saving cadence. Different
return counts are reported explicitly; errors only pair corresponding returns.

Outputs go to `examples/gpu_dense_output/output/comparison/`: a CPU/GPU
comparison figure, an error-versus-return figure, CSV section data, the fine
GPU trajectory archive, and `comparison.json` with coordinate errors, counts,
field checks, and timings.
Field setup, tracing, and section extraction are timed separately. CPU map
timing includes the momentum/WBA work; GPU timing includes trajectory saving
and host result assembly. These are different workloads, and their ratio
is not a pure integrator speedup. CPU tracing uses one MPI rank.

For a short check using the existing CPU example's CI parameters:

```console
python examples/gpu_dense_output/compare_poincare.py --ns-poinc 5 --nmaps 5 --resolution 10 --tol 1e-4 --output-dir /tmp/poincare-smoke
```

To audit selected orbits at tighter tolerance while retaining the same
120-radius launch grid:

```console
python examples/gpu_dense_output/compare_poincare.py --particle-ids 99 101 --tol 1e-10 --dt-save 1e-8 --output-dir /tmp/poincare-tight
```

## Small trajectory-saving demonstration

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
memory proportional to particles times samples. Use `--nparticles`, `--tmax`, `--dt-save`,
`--resolution`, and `--tol` to adjust the trace.

Analytic section-extraction checks can run without a GPU:

```console
python -m unittest tests.test_gpu_poincare_example
```

Other section directions and angles are available:

```console
python examples/gpu_dense_output/plot_poincare.py --zeta-section 0.5 --direction both --output-dir /tmp/other-section
```
