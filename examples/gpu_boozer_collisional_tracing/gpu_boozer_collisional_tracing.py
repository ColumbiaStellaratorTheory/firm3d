import numpy as np
import pandas as pd

from firm3d.catapult.tracing import trace_particles_boozer_with_collisions_gpu
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.collisions import ThermalBackground
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
from firm3d.util.functions import in_github_actions
from firm3d.util.mpi import comm_world

resolution = 5 if in_github_actions else 15  # Resolution for field interpolation
nparticles = 100 if in_github_actions else 1000  # Number of particles to trace
tol = 1e-4 if in_github_actions else 1e-8  # Tolerance for ODE solver
tmax = 2e-1

wout_filename = "../inputs/wout_aten_rescaled.nc"
bri = BoozerRadialInterpolant(
    wout_filename, 3, comm=comm_world, enforce_vacuum=True, write_boozmn=False
)

field = InterpolatedBoozerField(
    bri,
    3,
    ns_interp=resolution,
    ntheta_interp=resolution,
    nzeta_interp=resolution,
)
# set seed for consistency
np.random.seed(8)

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
stz_inits = initialize_position_profile(field, nparticles, reactivity, comm=comm_world)

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
# Isotropic pitch angle: v_par/v drawn uniformly in [-1, 1] at fixed birth energy
v0 = np.sqrt(2 * Ekin / mass)
vpar_inits = initialize_velocity_uniform(v0, nparticles)

# Background plasma the alphas collide with: a 50/50 DT fuel mix and the
# electrons that neutralize it, on the same profiles that set the birth
# distribution above.  Temperature is in eV at this interface, while T(s)
# above is in keV.
n_ref = 1e20  # m^-3
ne = lambda s: n_ref * nD(s)
Te = lambda s: 1e3 * T(s)

backgrounds = [
    ThermalBackground(
        n_profile=lambda s: 0.5 * ne(s),
        T_profile=Te,
        mass=2 * PROTON_MASS,
        charge=ELEMENTARY_CHARGE,
    ),
    ThermalBackground(
        n_profile=lambda s: 0.5 * ne(s),
        T_profile=Te,
        mass=3 * PROTON_MASS,
        charge=ELEMENTARY_CHARGE,
    ),
    ThermalBackground(
        n_profile=ne,
        T_profile=Te,
        mass=ELECTRON_MASS,
        charge=-ELEMENTARY_CHARGE,
    ),
]

last_time = trace_particles_boozer_with_collisions_gpu(
    field,
    stz_inits,
    vpar_inits,
    backgrounds=backgrounds,
    tmax=tmax,
    mass=mass,
    charge=charge,
    vtotal=v0,
    tol=tol,
    ns=resolution,
    ntheta=resolution,
    nzeta=resolution,
    rng_seed=0,
)
# The collisional output has seven columns rather than six: the total speed
# is reported before the final step size, because collisions change it and
# it is no longer recoverable from the launch energy.
particle_data = pd.DataFrame(
    {
        "s_start": stz_inits[:, 0],
        "t_start": stz_inits[:, 1],
        "z_start": stz_inits[:, 2],
        "vpar_start": vpar_inits,
        "s_end": last_time[:, 1],
        "t_end": last_time[:, 2],
        "z_end": last_time[:, 3],
        "vpar_end": last_time[:, 4],
        "v_end": last_time[:, 5],
        "last_time": last_time[:, 0],
        "dt_end": last_time[:, 6],
    }
)
particle_data.to_csv("./particle_data.csv")

t_end = last_time[:, 0]
v_end = last_time[:, 5]
lost = t_end < tmax

particle_loss = lost.sum() / nparticles
energy_loss = np.sum((v_end[lost] / v0) ** 2) / nparticles

print(f"Number of particles= {nparticles}")
print(f"Particle loss fraction: {particle_loss:.3f}")
print(f"Energy loss fraction: {energy_loss:.3f}")
print(f"Mean energy fraction of confined: {np.mean((v_end[~lost] / v0) ** 2):.4f}")
