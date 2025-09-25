import time

import numpy as np

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
    ShearAlfvenHarmonic,
)
from firm3d.field.trajectory_helpers import PassingPerturbedPoincare
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import proc0_print, setup_logging
from firm3d.util.mpi import comm_size, comm_world, verbose

boozmn_filename = "../inputs/boozmn_beta2.5_QH.nc"

charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS
Ekin = FUSION_ALPHA_PARTICLE_ENERGY

resolution = 48  # Resolution for field interpolation
sign_vpar = 1.0  # sign(vpar). should be +/- 1.
lam = 0.1  # lambda = v_perp^2/(v^2 B) = const. along trajectory
nchi_poinc = 1  # Number of chi initial conditions for poincare
ns_poinc = 120  # Number of s initial conditions for poincare
Nmaps = 1000  # Number of Poincare return maps to compute
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation
order = 3  # order for interpolation
tol = 1e-8  # Tolerance for ODE solver
degree = 3  # Degree for Lagrange interpolation
helicity_M = 1  # field strength helicity (QH)
helicity_N = -4  # field strength helicity (QH)

# SAW parameters
Phihat = -3.63941e2
Phim = 1
Phin = 2
omega = 133425
phase = 0

# Setup logging to redirect output to file
setup_logging(f"stdout_passing_map_{resolution}_{comm_size}.txt")

time1 = time.time()

bri = BoozerRadialInterpolant(
    boozmn_filename,
    order,
    no_K=True,
    comm=comm_world,
    helicity_M=helicity_M,
    helicity_N=helicity_N,
)

field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

saw = ShearAlfvenHarmonic(Phihat, Phim, Phin, omega, phase, field)

# Point for evaluation of Eprime
p0 = np.zeros((1, 3))
p0[0, 0] = 0.5  # s

poinc = PassingPerturbedPoincare(
    saw,
    sign_vpar,
    mass,
    charge,
    helicity_M,
    helicity_N,
    Ekin=Ekin,
    p0=p0,
    lam=lam,
    ns_poinc=ns_poinc,
    nchi_poinc=nchi_poinc,
    Nmaps=Nmaps,
    comm=comm_world,
    solver_options={"reltol": tol, "abstol": tol},
)

if verbose:
    poinc.plot_poincare()

time2 = time.time()

proc0_print("poincare time: ", time2 - time1)
