import time

import numpy as np

from firm3d.field.boozermagneticfield import (
    InterpolatedBoozerField,
)
from firm3d.trajectory_helpers import TrappedPoincare
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import in_github_actions, proc0_print, setup_logging
from firm3d.util.mpi import comm_size, comm_world, verbose

boozmn_filename = "../inputs/boozmn_beta2.5_QA.nc"

charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS
Ekin = FUSION_ALPHA_PARTICLE_ENERGY

resolution = 10 if in_github_actions else 48  # Resolution for field interpolation
neta_poinc = 5  # Number of eta initial conditions for poincare
ns_poinc = 5 if in_github_actions else 120  # Number of s initial conditions
Nmaps = 5 if in_github_actions else 1500  # Number of Poincare return maps to compute
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation
order = 3  # order for interpolation
tol = 1e-4 if in_github_actions else 1e-8  # Tolerance for ODE solver
s_mirror = 0.5  # flux surface for mirroring
theta_mirror = np.pi / 2  # poloidal angle for mirroring
zeta_mirror = 0
helicity_M = 1  # helicity of field strength contours
helicity_N = 0
degree = 3  # Degree for Lagrange interpolation

# Setup logging to redirect output to file
setup_logging(f"stdout_trapped_map_{resolution}_{comm_size}.txt")

time1 = time.time()

field = InterpolatedBoozerField.from_booz_xform(
    boozmn_filename,
    degree=order,
    ns=ns_interp,
    ntheta=ntheta_interp,
    nzeta=nzeta_interp,
    comm=comm_world,
)

poinc = TrappedPoincare(
    field,
    helicity_M,
    helicity_N,
    mass,
    charge,
    Ekin,
    s_mirror,
    theta_mirror,
    zeta_mirror,
    ns_poinc=ns_poinc,
    neta_poinc=neta_poinc,
    chaos_detection=True,
    Nmaps=Nmaps,
    comm=comm_world,
    solver_options={"reltol": tol, "abstol": tol, "axis": 0},
    tmax=1e-4,
)

if verbose and not in_github_actions:
    poinc.plot_poincare()

time2 = time.time()

proc0_print("poincare time: ", time2 - time1)
