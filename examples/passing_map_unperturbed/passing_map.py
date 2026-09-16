import time

from firm3d.field.boozermagneticfield import (
    InterpolatedBoozerField,
)
from firm3d.trajectory_helpers import PassingPoincare
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import in_github_actions, proc0_print, setup_logging
from firm3d.util.mpi import comm_size, comm_world, verbose

boozmn_filename = "../inputs/boozmn_aten_rescaled.nc"

charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS
Ekin = FUSION_ALPHA_PARTICLE_ENERGY

resolution = 10 if in_github_actions else 48  # Resolution for field interpolation
sign_vpar = 1.0  # sign(vpar). should be +/- 1.
lam = 0.0  # lambda = v_perp^2/(v^2 B) = const. along trajectory
ntheta_poinc = 1  # Number of zeta initial conditions for poincare
ns_poinc = 5 if in_github_actions else 120  # Number of s initial conditions
Nmaps = 5 if in_github_actions else 1000  # Number of Poincare return maps to compute
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation
order = 3  # order for interpolation
tol = 1e-4 if in_github_actions else 1e-8  # Tolerance for ODE solver
degree = 3  # Degree for Lagrange interpolation

# Setup logging to redirect output to file
setup_logging(f"stdout_passing_map_{resolution}_{comm_size}.txt")

time1 = time.time()

field = InterpolatedBoozerField.from_booz_xform(
    boozmn_filename,
    degree=order,
    ns=ns_interp,
    ntheta=ntheta_interp,
    nzeta=nzeta_interp,
    comm=comm_world,
)

poinc = PassingPoincare(
    field,
    lam,
    sign_vpar,
    mass,
    charge,
    Ekin,
    ns_poinc=ns_poinc,
    ntheta_poinc=ntheta_poinc,
    Nmaps=Nmaps,
    comm=comm_world,
    helicity_N=1 * field.nfp,
    helicity_M=1,
    solver_options={"reltol": tol, "abstol": tol},
    chaos_detection=True,
)

if verbose and not in_github_actions:
    poinc.plot_poincare()

time2 = time.time()

proc0_print("poincare time: ", time2 - time1)
