import time

import numpy as np

from firm3d.field.boozermagneticfield import (
    InterpolatedBoozerField,
)
from firm3d.field.collisions import (
    ThermalBackground,
    trace_particles_boozer_with_collisions,
)
from firm3d.field.tracing import (
    MaxToroidalFluxStoppingCriterion,
)
from firm3d.field.tracing_helpers import (
    initialize_position_profile,
    initialize_velocity_uniform,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    FUSION_ALPHA_PARTICLE_ENERGY,
    PROTON_MASS,
)
from firm3d.util.functions import in_github_actions, proc0_print, setup_logging
from firm3d.util.mpi import comm_size, comm_world

time1 = time.time()

resolution = 10 if in_github_actions else 48  # Resolution for field interpolation
nParticles = 50 if in_github_actions else 1000  # Number of particles to trace
reltol = 1e-4 if in_github_actions else 1e-8  # Relative tolerance for the ODE solver
abstol = 1e-4 if in_github_actions else 1e-8  # Absolute tolerance for the ODE solver
order = 3  # Order for radial interpolation
degree = 3  # Degree for 3d interpolation
wout_filename = "../inputs/wout_aten_rescaled.nc"
tmax = 1e-4 if in_github_actions else 2e-1  # Time for integration
ns_interp = resolution
ntheta_interp = resolution
nzeta_interp = resolution

# Setup logging to redirect output to file
setup_logging(f"stdout_{nParticles}_{resolution}_{comm_size}.txt")

## Setup field interpolation
field = InterpolatedBoozerField.from_booz_xform(
    wout_filename,
    degree=order,
    ns=ns_interp,
    ntheta=ntheta_interp,
    nzeta=nzeta_interp,
    comm=comm_world,
    write_boozmn=False,
    # Vacuum guiding-center equations, matching the GPU collisional examples.
    # from_booz_xform otherwise defaults to the no-K equations, which the GPU
    # collisional entry points do not accept and which give different orbits.
    enforce_vacuum=True,
)

# Define fusion birth distribution
# Bader, A., et al. "Modeling of energetic particle transport in optimized
# stellarators." Nuclear Fusion 61.11 (2021): 116060.
nD = lambda s: 1 - s**5  # Normalized density
nT = nD
T = lambda s: 11.5 * (1 - s)  # Temperature in keV


# D-T cross-section
def sigmav(T):
    if T > 0:
        return T ** (-2 / 3) * np.exp(-19.94 * T ** (-1 / 3))
    else:
        return 0


# Reactivity profile
reactivity = lambda s: nD(s) * nT(s) * sigmav(T(s))

points = initialize_position_profile(field, nParticles, reactivity, comm=comm_world)

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
# Isotropic pitch angle: v_par/v drawn uniformly in [-1, 1] at fixed birth energy
v0 = np.sqrt(2 * Ekin / mass)
vpar_init = initialize_velocity_uniform(v0, nParticles, comm=comm_world)

n_ref = 1e20  # m^-3
ne = lambda s: n_ref * nD(s)
Te = lambda s: 1e3 * T(s)  # eV

deuterium = ThermalBackground(
    n_profile=lambda s: 0.5 * ne(s),
    T_profile=Te,
    mass=2 * PROTON_MASS,
    charge=ELEMENTARY_CHARGE,
)
tritium = ThermalBackground(
    n_profile=lambda s: 0.5 * ne(s),
    T_profile=Te,
    mass=3 * PROTON_MASS,
    charge=ELEMENTARY_CHARGE,
)
electrons = ThermalBackground(
    n_profile=ne,
    T_profile=Te,
    mass=ELECTRON_MASS,
    charge=-ELEMENTARY_CHARGE,
)

time1 = time.time()
## Trace alpha particles in Boozer coordinates, with collisions, until they
## hit the s = 1 surface.
res_tys, res_zeta_hits = trace_particles_boozer_with_collisions(
    field,
    points,
    vpar_init,
    backgrounds=[deuterium, tritium, electrons],
    tmax=tmax,
    mass=mass,
    charge=charge,
    comm=comm_world,
    Ekin=Ekin,
    stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
    abstol=abstol,
    reltol=reltol,
    DP_hmin=1e-10,
    rng_seed=0,
)

time2 = time.time()
proc0_print("Elapsed time for tracing = ", time2 - time1)

t_end = np.array([traj[-1, 0] for traj in res_tys])
v_end = np.array([traj[-1, 5] for traj in res_tys])
lost = np.array([len(hits) > 0 for hits in res_zeta_hits])

particle_loss = lost.sum() / nParticles
energy_loss = np.sum((v_end[lost] / v0) ** 2) / nParticles

proc0_print(f"Number of particles = {nParticles}")
proc0_print(f"Particle loss fraction: {particle_loss:.3f}")
proc0_print(f"Energy loss fraction: {energy_loss:.3f}")
proc0_print(
    f"Mean energy fraction of confined alphas: {np.mean((v_end[~lost] / v0) ** 2):.3f}"
)
