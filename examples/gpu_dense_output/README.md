# GPU trajectory saving

Run the trajectory-saving example from the repository root on a GPU node:

```console
python examples/gpu_dense_output/save_trajectories.py
```

It saves 32 co-passing alpha trajectories in the bundled ATEN equilibrium
using Dormand–Prince dense output, and writes `trajectories.npz` and
`trajectories.png` to `examples/gpu_dense_output/output/`.
Use `--nparticles`, `--tmax`, `--dt-save`, `--resolution`, and `--tol` to
adjust the trace. Histories use memory proportional to particles times samples.

Reload a numeric path without pickle:

```python
import numpy as np

with np.load("examples/gpu_dense_output/output/trajectories.npz", allow_pickle=False) as saved:
    path = saved["particle_000000"]  # t, s, theta, zeta, vpar
```

For a kinetic Poincaré plot, use the existing passing-map example with either
CPU or CATAPULT tracing:

```console
python examples/passing_map_unperturbed/passing_map.py --backend cpu
python examples/passing_map_unperturbed/passing_map.py --backend catapult
```

Both use the same physical parameters and `PassingPoincare` plotting helpers.
CATAPULT sections interpolate saved samples; converge `--dt-save` separately
from the solver tolerance. Its `--tmax` is the total continuous trace duration,
while CPU tracing limits each return separately.
