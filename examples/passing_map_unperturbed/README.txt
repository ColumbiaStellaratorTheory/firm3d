This example computes the kinetic passing Poincare map in the bundled ATEN
configuration, scaled to the size and field strength of ARIES-CS. Defaults
use 120 co-passing birth-energy alpha particles (mu = 0), 1000 returns,
48 interpolation cells per coordinate, and tolerance 1e-8.

From the repository root, run either backend with the same physical parameters:

python examples/passing_map_unperturbed/passing_map.py --backend cpu
python examples/passing_map_unperturbed/passing_map.py --backend catapult

Both use PassingPoincare and save numeric NPZ data and a PNG plot under output/.
CATAPULT requires a GPU. Both backends limit each return by --tmax; CATAPULT
uses Nmaps * tmax as the total upper bound for its continuous trace. GPU sections
use dense-output roots. Converge --dt-save separately for WBA, which uses saved
history. CATAPULT finds return times first, then saves history only through
completed returns.

The passing map is computed with the chaos_detection parameter enabled. Weighted Birkhoff Averaging is
added as a setting to the passing map and applied to a particle canonical momentum
to create a numerical metric (digit accuracy) of chaos, implemented as described in:

N. Duignan and J. D. Meiss. "Distinguishing between regular and chaotic orbits of flows by the weighted birkhoff average." Physical Nonlinear Phenomena. (2023): 449:133749.
