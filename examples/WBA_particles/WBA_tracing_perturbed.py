import time

import numpy as np

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
    ShearAlfvenHarmonic,
)
from firm3d.field.tracing_helpers import (
    initialize_position_uniform_vol,
    initialize_velocity_uniform,
)
from firm3d.trajectory_helpers import (
    WBAPerturbedParticles,
    return_chaotic_percentage,
)
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
tmax = 1e-4 if in_github_actions else 1e-2  # Time for integration
ns_interp = resolution
ntheta_interp = resolution
nzeta_interp = resolution
# Helicity of the field strength for classifying ripple and barely-trapped
helicity_M = 1
helicity_N = 0
dt_save = 1e-7  # Time interval for saving trajectory points

Phin = 1  # Toroidal mode number of the perturbation
Phim = 1  # Poloidal mode number of the perturbation
omega = 136000  # Frequency of the perturbation in rad/s

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

saw = ShearAlfvenHarmonic(100, Phim=1, Phin=1, omega=136000, B0=field, phase=0)

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

field.set_points(tracing_points)
modB = field.modB()[:, 0]
mus_per_mass = ((Ekin / mass) - 0.5 * vpar_init**2) / (modB)

object_WBA = WBAPerturbedParticles(
    saw,
    mass,
    charge,
    Ekin,
    Phin,
    Phim,
    omega,
    helicity_N,
    helicity_M,
    points=tracing_points,
    v_pars=vpar_init,
    mu_per_mass=mus_per_mass,
    tmax=tmax,
    min_timestep=1e-7,
    savedata=not in_github_actions,
    comm=comm_world,
    tol=abstol,
    nconvergence_points=10,
)

DAs_all = np.array(object_WBA.DAs)
convergence_DAs = object_WBA.convergence_DAs
convergence_times = object_WBA.convergence_times


chaotic_percentage = return_chaotic_percentage(object_WBA.DA_at_tfinal)

proc0_print("Percent of space that is chaotic: ", chaotic_percentage, "%")
