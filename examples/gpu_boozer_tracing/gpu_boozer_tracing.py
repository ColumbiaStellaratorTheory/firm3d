#!/usr/bin/env python

import numpy as np
import pandas as pd

from firm3d.catapult.field import CatapultBoozerField
from firm3d.catapult.tracing import trace_particles_boozer_gpu
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
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
from firm3d.util.functions import in_github_actions, sigmav

resolution = 5 if in_github_actions else 15  # Resolution for field interpolation
nparticles = 100 if in_github_actions else 30000  # Number of particles to trace
tol = 1e-4 if in_github_actions else 1e-6  # Tolerance for ODE solver
tmax = 1e-4

### CREATE A FIELD FOR TRACING
boozmn_filename = "../inputs/boozmn_ariescs_low_res.nc"
bri = BoozerRadialInterpolant(boozmn_filename, 3, enforce_vacuum=True)

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
# Reactivity profile
reactivity = lambda s: nD(s) * nT(s) * sigmav(T(s))
stz_inits = initialize_position_profile(field, nparticles, reactivity, seed=1)

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
# Initialize uniformly distributed parallel velocities
vpar0 = np.sqrt(2 * Ekin / mass)
vpar_inits = initialize_velocity_uniform(vpar0, nparticles, seed=1)

# The field is tabulated for the GPU once, at the resolution and precision to
# trace in; the tracing calls then need neither.
field_dbl = CatapultBoozerField(bri, resolution, resolution, resolution)
field_flt = CatapultBoozerField(
    bri, resolution, resolution, resolution, precision="single"
)

# Trace in double precision. As for the CPU tracer, res_tys holds each
# particle's (t, s, theta, zeta, vpar) rows and res_hits its boundary crossing,
# so the same post-processing serves both.
res_tys_dbl, res_hits_dbl = trace_particles_boozer_gpu(
    field_dbl,
    stz_inits,
    vpar_inits,
    tmax=tmax,
    mass=mass,
    charge=charge,
    Ekin=Ekin,
    tol=tol,
    forget_exact_path=True,
)

# trace in single precision: the inputs are cast to the field's precision
res_tys_flt, res_hits_flt = trace_particles_boozer_gpu(
    field_flt,
    stz_inits,
    vpar_inits,
    tmax=tmax,
    mass=mass,
    charge=charge,
    Ekin=Ekin,
    tol=tol,
    forget_exact_path=True,
)

final_dbl = np.array([traj[-1] for traj in res_tys_dbl])
final_flt = np.array([traj[-1] for traj in res_tys_flt])
particle_data = pd.DataFrame(
    {
        "s_start": stz_inits[:, 0],
        "t_start": stz_inits[:, 1],
        "z_start": stz_inits[:, 2],
        "vpar_start": vpar_inits,
        "last_time_dbl": final_dbl[:, 0],
        "s_end_dbl": final_dbl[:, 1],
        "t_end_dbl": final_dbl[:, 2],
        "z_end_dbl": final_dbl[:, 3],
        "vpar_end_dbl": final_dbl[:, 4],
        "last_time_flt": final_flt[:, 0],
        "s_end_flt": final_flt[:, 1],
        "t_end_flt": final_flt[:, 2],
        "z_end_flt": final_flt[:, 3],
        "vpar_end_flt": final_flt[:, 4],
    }
)

particle_data.to_csv("./particle_data.csv")
print(f"tmax= {tmax}")
print(f"Number of particles= {nparticles}")
print(f"Flt. Loss fraction: {np.mean([len(hits) > 0 for hits in res_hits_flt]):.3f}")
print(f"Dbl. Loss fraction: {np.mean([len(hits) > 0 for hits in res_hits_dbl]):.3f}")
