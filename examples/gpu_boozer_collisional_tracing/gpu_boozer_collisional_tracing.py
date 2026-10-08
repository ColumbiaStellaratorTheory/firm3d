import numpy as np
import pandas as pd

from firm3d.catapult.field import CatapultBoozerField
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
from firm3d.util.functions import in_github_actions, in_gpu_benchmark, sigmav

import json
import time

if in_gpu_benchmark:
    resolution, nparticles, tol, tmax = 15, 100000, 1e-6, 2e-1
elif in_github_actions:
    resolution, nparticles, tol, tmax = 5, 100, 1e-4, 1e-1
else:
    resolution, nparticles, tol, tmax = 15, 30000, 1e-6, 1e-3

wout_filename = "../inputs/wout_aten_rescaled.nc"
start_bri = time.perf_counter()
bri = BoozerRadialInterpolant(
    wout_filename, 3, enforce_vacuum=True, write_boozmn=False
)
bri_time = time.perf_counter() - start_bri

start_ibf = time.perf_counter()
field = InterpolatedBoozerField(
    bri,
    3,
    ns_interp=resolution,
    ntheta_interp=resolution,
    nzeta_interp=resolution,
)
ibf_time = time.perf_counter() - start_ibf

# set seed for consistency
np.random.seed(8)

# Define fusion birth distribution
# Bader, A., et al. "Modeling of energetic particle transport in optimized
# stellarators." Nuclear Fusion 61.11 (2021): 116060.
nD = lambda s: 1 - s**5  # Normalized density
nT = nD
T = lambda s: 11.5 * (1 - s)  # Temperature in keV

# Reactivity profile
reactivity = lambda s: nD(s) * nT(s) * sigmav(T(s))
stz_inits = initialize_position_profile(field, nparticles, reactivity)

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

# The field is tabulated for the GPU once; collisions are traced in double
# precision, so the field is built that way.
start_setup = time.perf_counter()
field_gpu = CatapultBoozerField(field, resolution, resolution, resolution)
setup_time_dbl = time.perf_counter() - start_setup

start_dbl = time.perf_counter()
last_time = trace_particles_boozer_with_collisions_gpu(
    field_gpu,
    stz_inits,
    vpar_inits,
    backgrounds=backgrounds,
    tmax=tmax,
    mass=mass,
    charge=charge,
    Ekin=Ekin,
    tol=tol,
    rng_seed=0,
)
dbl_time = time.perf_counter() - start_dbl

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

loss_fraction_dbl = lost.sum() / nparticles
energy_loss = np.sum((v_end[lost] / v0) ** 2) / nparticles

print(f"Number of particles= {nparticles}")
print(f"Particle loss fraction: {loss_fraction_dbl:.3f}")
print(f"Energy loss fraction: {energy_loss:.3f}")
print(f"Mean energy fraction of confined: {np.mean((v_end[~lost] / v0) ** 2):.4f}")

### record for regression testing
timing_result = {
    "nparticles": nparticles,
    "tolerance": tol,
    "resolution": resolution,
    "loss_fraction_dbl": loss_fraction_dbl,
    # "loss_fraction_flt": loss_fraction_flt,
    "tmax": tmax,
    "times": {
        "bri_setup": bri_time,
        "field_interpolation": ibf_time,
        "catapult_setup": setup_time_dbl,
        "tracing_dbl": dbl_time,
        # "tracing_flt": flt_time,
    },
}
with open("gpu_boozer_collisional_tracing_results.json", "w") as f:
    json.dump(timing_result, f, indent=2)
