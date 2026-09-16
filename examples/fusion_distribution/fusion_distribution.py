import time

import numpy as np

# Ensure mpi4py is imported and initialized before firm3d modules
# This ensures the mpi4py C API is available for C++ bindings
from mpi4py import MPI  # noqa: F401

from firm3d.field.boozermagneticfield import (
    InterpolatedBoozerField,
)
from firm3d.field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    trace_particles_boozer,
)
from firm3d.field.tracing_helpers import (
    initialize_position_profile,
    initialize_velocity_uniform,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import in_github_actions, proc0_print, setup_logging, sigmav
from firm3d.util.mpi import comm_size, comm_world, verbose

time1 = time.time()

resolution = 10 if in_github_actions else 48  # Resolution for field interpolation
nParticles = 50 if in_github_actions else 5000  # Number of particles to trace
reltol = 1e-4 if in_github_actions else 1e-8  # Relative tolerance for the ODE solver
abstol = 1e-4 if in_github_actions else 1e-8  # Absolute tolerance for the ODE solver
order = 3  # Order for radial interpolation
degree = 3  # Degree for 3d interpolation
order = 3
boozmn_filename = "../inputs/boozmn_aten_rescaled.nc"
tmax = 1e-4 if in_github_actions else 1e-2  # Time for integration
ns_interp = resolution
ntheta_interp = resolution
nzeta_interp = resolution

# Setup logging to redirect output to file
setup_logging(f"stdout_{nParticles}_{resolution}_{comm_size}.txt")

## Setup field interpolation
field = InterpolatedBoozerField.from_booz_xform(
    boozmn_filename,
    degree=order,
    ns=ns_interp,
    ntheta=ntheta_interp,
    nzeta=nzeta_interp,
    comm=comm_world,
)

# Define fusion birth distribution
# Bader, A., et al. "Modeling of energetic particle transport in optimized
# stellarators." Nuclear Fusion 61.11 (2021): 116060.
nD = lambda s: 1 - s**5  # Normalized density
nT = nD
T = lambda s: 11.5 * (1 - s)  # Temperature in keV


# D-T cross-section
# Reactivity profile
reactivity = lambda s: nD(s) * nT(s) * sigmav(T(s))

points = initialize_position_profile(field, nParticles, reactivity, comm=comm_world)

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
# Initialize uniformly distributed parallel velocities
vpar0 = np.sqrt(2 * Ekin / mass)
vpar_init = initialize_velocity_uniform(vpar0, nParticles, comm=comm_world)

time1 = time.time()
## Trace alpha particles in Boozer coordinates until they hit the s = 1 surface
res_tys, res_zeta_hits = trace_particles_boozer(
    field,
    points,
    vpar_init,
    tmax=tmax,
    mass=mass,
    charge=charge,
    comm=comm_world,
    Ekin=Ekin,
    stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
    forget_exact_path=True,
    abstol=abstol,
    reltol=reltol,
)

time2 = time.time()
proc0_print("Elapsed time for tracing = ", time2 - time1)

## Post-process results to obtain lost particles
if verbose and not in_github_actions:
    from firm3d.trajectory_helpers import compute_loss_fraction

    times, loss_frac = compute_loss_fraction(res_tys, tmin=1e-5, tmax=1e-2)
    import matplotlib

    matplotlib.use("Agg")  # Don't use interactive backend
    import matplotlib.pyplot as plt

    plt.figure()
    plt.loglog(times, loss_frac)
    plt.xlim([1e-5, 1e-2])
    plt.ylim([1e-3, 1])
    plt.xlabel("Time [s]")
    plt.ylabel("Fraction of lost particles")
    plt.savefig("loss_fraction.png")
