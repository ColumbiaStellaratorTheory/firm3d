import time

from firm3d.field.boozermagneticfield import InterpolatedBoozerField
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

use_catapult = False  # Set True to trace on the GPU
resolution = 10 if in_github_actions else 48  # Resolution for field interpolation
sign_vpar = 1.0  # sign(vpar). should be +/- 1.
lam = 0.0  # lambda = v_perp^2/(v^2 B) = const. along trajectory
ntheta_poinc = 1  # Number of theta initial conditions
ns_poinc = 5 if in_github_actions else 120  # Number of s initial conditions
Nmaps = 5 if in_github_actions else 1000  # Number of Poincare return maps to compute
ns_interp = resolution  # Number of radial grid points for interpolation
ntheta_interp = resolution  # Number of poloidal grid points for interpolation
nzeta_interp = resolution  # Number of toroidal grid points for interpolation
order = 3  # Order for interpolation
tol = 1e-4 if in_github_actions else 1e-8  # Tolerance for ODE solver

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
if use_catapult:
    from firm3d.catapult.field import CatapultBoozerField

    field = CatapultBoozerField(field, ns_interp, ntheta_interp, nzeta_interp)

poinc = PassingPoincare(
    field,
    lam=lam,
    sign_vpar=sign_vpar,
    mass=mass,
    charge=charge,
    Ekin=Ekin,
    ns_poinc=ns_poinc,
    ntheta_poinc=ntheta_poinc,
    Nmaps=Nmaps,
    comm=comm_world,
    solver_options={"reltol": tol, "abstol": tol},
)
if verbose and not in_github_actions:
    poinc.plot_poincare()

time2 = time.time()

proc0_print("poincare time: ", time2 - time1)
