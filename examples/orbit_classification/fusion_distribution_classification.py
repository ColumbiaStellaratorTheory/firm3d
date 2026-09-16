import sys
import time

import numpy as np

from firm3d._core.util import parallel_loop_bounds
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
from firm3d.plotting.orbit_classification import OrbitClassification
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import in_github_actions, proc0_print, sigmav

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    verbose = comm.rank == 0
    comm_size = comm.size
except ImportError:
    comm = None
    verbose = True
    comm_size = 1

time1 = time.time()

resolution = 10 if in_github_actions else 48  # Resolution for field interpolation
nParticles = 50 if in_github_actions else 5000  # Number of particles to trace
reltol = 1e-4 if in_github_actions else 1e-8  # Relative tolerance for the ODE solver
abstol = 1e-4 if in_github_actions else 1e-8  # Absolute tolerance for the ODE solver
order = 3  # Order for radial interpolation
degree = 3  # Degree for 3d interpolation
boozmn_filename = "../inputs/boozmn_ariescs.nc"
tmax = 1e-4 if in_github_actions else 1e-2  # Time for integration
ns_interp = resolution
ntheta_interp = resolution
nzeta_interp = resolution
# Helicity of the field strength for classifying ripple and barely-trapped
helicity_M = 1
helicity_N = 0
dt_save = 1e-7  # Time interval for saving trajectory points

# Redirect stdout to file for entire script duration
stdout_file = open(  # noqa: SIM115
    f"stdout_{nParticles}_{resolution}_{comm_size}.txt", "a", buffering=1
)
sys.stdout = stdout_file

## Setup field interpolation
field = InterpolatedBoozerField.from_booz_xform(
    boozmn_filename,
    degree=order,
    ns=ns_interp,
    ntheta=ntheta_interp,
    nzeta=nzeta_interp,
    comm=comm,
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

points_init = initialize_position_profile(
    field, nParticles, reactivity, comm=comm, seed=0
)

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
# Initialize uniformly distributed parallel velocities
vpar0 = np.sqrt(2 * Ekin / mass)
vpar_init = initialize_velocity_uniform(vpar0, nParticles, comm=comm, seed=0)

first, last = parallel_loop_bounds(comm, nParticles)
for iParticle in range(first, last):
    point = np.zeros((1, 3))
    point[0, :] = points_init[iParticle, :]
    ## Trace alpha particles in Boozer coordinates until they hit the s = 1 surface
    res_tys, res_hits = trace_particles_boozer(
        field,
        point,
        [vpar_init[iParticle]],
        tmax=tmax,
        mass=mass,
        charge=charge,
        Ekin=Ekin,
        vpars=[0],
        vpars_stop=False,
        stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        forget_exact_path=False,
        abstol=abstol,
        reltol=reltol,
        dt_save=dt_save,
    )

    res_hit = res_hits[0]
    res_ty = res_tys[0]
    bounce_times = []
    if len(res_hit) > 0:
        if np.any(res_hit[:, 1] == -1):  # Particle was lost to the wall
            np.savetxt("particle_" + str(iParticle) + "_traj.txt", res_ty)
            np.savetxt("particle_" + str(iParticle) + "_hits.txt", res_hit)
        else:
            continue  # Particle was not lost to the wall, skip

        oc = OrbitClassification(field, Ekin, mass, charge, helicity_M, helicity_N)
        particle_dict = oc.classify_orbit(res_ty, res_hit)
        np.savez(f"particle_{iParticle}.npz", **particle_dict)

proc0_print(
    f"Total time for tracing and classifying particles: "
    f"{time.time() - time1:.2f} seconds"
)
