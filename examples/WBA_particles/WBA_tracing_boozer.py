import time

import numpy as np

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.tracing_helpers import (
    initialize_position_uniform_vol,
    initialize_velocity_uniform,
)
from firm3d.trajectory_helpers import WBAParticles, return_chaotic_percentage
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import in_github_actions, proc0_print
from firm3d.util.mpi import comm_world

time1 = time.time()

resolution = 10 if in_github_actions else 48  # Resolution for field interpolation
nParticles = 10 if in_github_actions else 5000  # Number of particles to trace
reltol = 1e-10  # Relative tolerance for the ODE solver
abstol = 1e-10  # Absolute tolerance for the ODE solver
order = 3  # Order for radial interpolation
degree = 3  # Degree for 3d interpolation
boozmn_filename = "../inputs/boozmn_beta2.5_QA.nc"
tmax = 1e-4 if in_github_actions else 1e-2  # Maximum integration time
ns_interp = resolution
ntheta_interp = resolution
nzeta_interp = resolution
# Helicity of the field strength for classifying ripple and barely-trapped
helicity_M = 1
helicity_N = 0
dt_save = 1e-7  # Time interval for saving trajectory points


## Setup radial interpolation
bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm_world)

## Setup 3d interpolation
field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

tracing_points = initialize_position_uniform_vol(
    field,
    nParticles,
    comm=comm_world,
    seed=0,
)

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
# Initialize uniformly distributed parallel velocities
vpar0 = np.sqrt(2 * Ekin / mass)
vpar_init = initialize_velocity_uniform(vpar0, nParticles, comm=comm_world, seed=0)

### Alternatively: one could provide the traced trajectories to the WBAParticles class,
# which will then compute the DA for those trajectories.
# pass gc_tys to the WBAParticles class, and provide
object_WBA = WBAParticles(
    field,
    mass,
    charge,
    Ekin,
    helicity_N,
    helicity_M,
    points=tracing_points,
    v_pars=vpar_init,
    tmax=tmax,
    min_timestep=1e-7,
    savedata=not in_github_actions,
    comm=comm_world,
    tol=abstol,
    convergence_points=10,
)

DAs_all = np.array(object_WBA.DAs)
convergence_DAs = object_WBA.convergence_DAs
convergence_times = object_WBA.convergence_times

chaotic_percentage = return_chaotic_percentage(object_WBA.DA_at_tfinal)

proc0_print("Percent of space that is chaotic: ", chaotic_percentage, "%")
