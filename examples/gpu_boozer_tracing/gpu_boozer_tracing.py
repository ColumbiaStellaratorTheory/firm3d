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
from firm3d.util.functions import in_github_actions, in_gpu_benchmark, sigmav
import json
import time
import firm3dpp

if in_gpu_benchmark:
    resolution, nparticles, tol, tmax = 15, 100000, 1e-6, 1e-2
elif in_github_actions:
    resolution, nparticles, tol, tmax = 5, 100, 1e-4, 1e-4
else:
    resolution, nparticles, tol, tmax = 15, 30000, 1e-6, 1e-4


def _to_pseudo_cartesian(stz_inits, dtype):
    """
    A copy of (s, theta, zeta) initial conditions as (s cos theta, s sin theta,
    zeta), the coordinates CATAPULT integrates in, in the given dtype.
    """
    x_inits = np.array(stz_inits, dtype=dtype, order="C")
    s = x_inits[:, 0].copy()
    theta = x_inits[:, 1].copy()
    x_inits[:, 0] = s * np.cos(theta)
    x_inits[:, 1] = s * np.sin(theta)
    return x_inits

### CREATE A FIELD FOR TRACING
boozmn_filename = "../inputs/boozmn_ariescs_low_res.nc"
start_bri = time.perf_counter()
bri = BoozerRadialInterpolant(boozmn_filename, 3, enforce_vacuum=True)
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

# D-T cross-section
# Reactivity profile
reactivity = lambda s: nD(s) * nT(s) * sigmav(T(s))
stz_inits = initialize_position_profile(field, nparticles, reactivity, seed=1)

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
# Isotropic pitch angle: v_par/v drawn uniformly in [-1, 1] at fixed birth energy
v0 = np.sqrt(2 * Ekin / mass)
vpar_inits = initialize_velocity_uniform(v0, nparticles, seed=1)

# The field is tabulated for the GPU once, at the resolution and precision to
# trace in; the tracing calls then need neither.
start_setup = time.perf_counter()
field_dbl = CatapultBoozerField(bri, resolution, resolution, resolution)
setup_time_dbl = time.perf_counter() - start_setup


# Trace in double precision. As for the CPU tracer, res_tys holds each
# particle's (t, s, theta, zeta, vpar) rows and res_hits its boundary crossing,
# so the same post-processing serves both.
start_dbl = time.perf_counter()

quad_info = field_dbl.quad_info

# save however you want
np.save("quad_info.npy", quad_info)

# now read it
read_quad_info = np.load("quad_info.npy")
CUDA_VISIBLE_DEVICES = "2"
kwargs = {
    "quad_pts": read_quad_info,
    "srange": field_dbl.srange,
    "trange": field_dbl.trange,
    "zrange": field_dbl.zrange,
    "stz_init": _to_pseudo_cartesian(stz_inits, dtype=np.float64),
    "m": mass,
    "q": charge,
    "vtotal": v0,
    "vtang": vpar_inits,
    "tmax": tmax,
    "tol": tol,
    "dt_in": -np.ones(nparticles),
    "mu_in": -np.ones(nparticles),
    "psi0": field_dbl.psi0,
    "nparticles": nparticles,
}

output = np.asarray(firm3dpp.boozer_gpu_tracing(**kwargs), dtype=np.float64).reshape(nparticles, 7)

print(output)
loss_times = output[:, 0]
