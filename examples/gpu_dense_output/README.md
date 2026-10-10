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

The full parameter set was run on Perlmutter with an A100 80 GB GPU and an
AMD EPYC 7763 CPU. All three methods returned 114,000 crossings: 114 particles
completed all 1,000 returns, and the other six reached the CPU example's
boundary before their first return. Return counts matched for every particle.
The continuous integration horizon was 5.862 ms. The GPU and CPU table
quantities agreed to a maximum scaled error of `1.25e-12`.

Single-call times, excluding field setup and file/figure writing:

| Workflow | Seconds |
| --- | ---: |
| Existing CPU map, including momentum and WBA | 213.98 |
| Continuous CPU, saved at `5e-8` s plus event roots | 208.78 |
| GPU trace, saved at `1e-7` s | 15.41 |
| GPU trace, saved at `5e-8` s | 16.12 |
| GPU section extraction, coarse / fine | 0.53 / 1.00 |

Across all paired returns, using circular differences for `theta`:

| Comparison | RMS difference in `s` | Maximum difference in `s` | Maximum difference in `theta` (rad) |
| --- | ---: | ---: | ---: |
| GPU fine vs existing CPU map | `7.73e-4` | `3.71e-2` | `0.369` |
| Continuous CPU roots vs existing CPU map | `1.22e-3` | `5.99e-2` | `0.633` |
| GPU coarse vs fine saving cadence | `1.72e-4` | `8.25e-4` | `3.01e-3` |
| CPU fine sampled sections vs its own event roots | `5.50e-5` | `2.14e-4` | `7.61e-4` |

The three plots show the same overall orbit structure, but corresponding
long orbits do not agree exactly at the CPU example's tolerance. For the first
ten returns, GPU fine versus CPU has maximum differences of `2.03e-4` in `s`
and `7.38e-4` radians in `theta`. Differences grow with return number; the
largest occurs for particle 101, launched at `s=102/121`. Halving the saving
interval reduces section-interpolation error but has little effect on that
long-orbit difference. The CPU continuous-versus-restarting control also
diverges, so sampling cadence alone does not establish orbit convergence.
Converge the integration tolerance separately for individual long orbits.

Measured values are saved in `perlmutter_comparison.json`; the script regenerates
the figures and a fresh `comparison.json`. The fine GPU history buffer in this
run is approximately 788 MB, in addition to the field table and host copies.

For a short check using the existing CPU example's CI parameters:

```console
python examples/gpu_dense_output/compare_poincare.py --ns-poinc 5 --nmaps 5 --resolution 10 --tol 1e-4 --output-dir /tmp/poincare-smoke
```

To audit selected orbits at tighter tolerance while retaining the same
120-radius launch grid:

```console
python examples/gpu_dense_output/compare_poincare.py --particle-ids 99 101 --tol 1e-10 --dt-save 1e-8 --output-dir /tmp/poincare-tight
```

This selected-orbit check was also run on Perlmutter. With `tol=1e-10` and
fine saving at `5e-9` s, the maximum GPU-versus-CPU-map differences fell to
`1.17e-3` in `s` and `8.49e-3` radians in `theta`. The two GPU save cadences
differed by at most `7.96e-6` in `s`. This supports converging integration
separately from section sampling; it does not establish full convergence.
For this two-particle workload, CPU map time was 13.26 s and GPU fine tracing
plus extraction was 25.61 s. The GPU performance advantage of the full
120-particle example does not extend to this small, tighter-tolerance workload.

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
