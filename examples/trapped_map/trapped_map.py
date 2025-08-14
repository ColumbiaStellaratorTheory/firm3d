import sys
import time

import numpy as np

from simsopt.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from simsopt.field.trajectory_helpers import TrappedPoincare
from simsopt.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from simsopt.util.functions import proc0_print
from simsopt.util.mpi import comm_size, comm_world, verbose

boozmn_filename = "../inputs/boozmn_beta2.5_QA.nc"

charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS
Ekin = FUSION_ALPHA_PARTICLE_ENERGY

resolution = 48  # Resolution for field interpolation
neta_poinc = 5  # Number of eta initial conditions for poincare
ns_poinc = 120  # Number of s initial conditions for poincare
Nmaps = 1000  # Number of Poincare return maps to compute
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation
order = 3  # order for interpolation
tol = 1e-8  # Tolerance for ODE solver
s_mirror = 0.5  # flux surface for mirroring
theta_mirror = np.pi / 2  # poloidal angle for mirroring
zeta_mirror = 0
helicity_M = 1  # helicity of field strength contours
helicity_N = 0
degree = 3  # Degree for Lagrange interpolation

sys.stdout = open(f"stdout_trapped_map_{resolution}_{comm_size}.txt", "a", buffering=1)

time1 = time.time()

bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm_world)

field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

poinc = TrappedPoincare(
    field,
    helicity_M,
    helicity_N,
    s_mirror,
    theta_mirror,
    zeta_mirror,
    mass,
    charge,
    Ekin,
    ns_poinc=ns_poinc,
    neta_poinc=neta_poinc,
    Nmaps=Nmaps,
    comm=comm_world,
    solver_options={"reltol": tol, "abstol": tol, "axis": 0},
    tmax=1e-4,
)

if verbose:
    poinc.plot_poincare()

time2 = time.time()

proc0_print("poincare time: ", time2 - time1)
