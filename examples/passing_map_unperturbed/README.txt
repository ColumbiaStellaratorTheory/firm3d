This example computes the kinetic passing Poincare map in the bundled ATEN
configuration, scaled to the size and field strength of ARIES-CS. Defaults
use 120 co-passing birth-energy alpha particles (mu = 0), 1000 returns,
48 interpolation cells per coordinate, and tolerance 1e-8.

From the repository root, run either backend with the same physical parameters:

python examples/passing_map_unperturbed/passing_map.py --backend cpu
python examples/passing_map_unperturbed/passing_map.py --backend catapult

Both use PassingPoincare and save numeric NPZ data and a PNG plot under output/.
CATAPULT requires a GPU. Its --tmax is the total continuous tracing duration;
CPU tracing limits each return separately. GPU sections interpolate saved
samples, so converge --dt-save independently of the integration tolerance.

The passing map is computed with the chaos_detection parameter enabled. Weighted Birkhoff Averaging is
added as a setting to the passing map and applied to a particle canonical momentum
to create a numerical metric (digit accuracy) of chaos, implemented as described in:

N. Duignan and J. D. Meiss. "Distinguishing between regular and chaotic orbits of flows by the weighted birkhoff average." Physical Nonlinear Phenomena. (2023): 449:133749.
