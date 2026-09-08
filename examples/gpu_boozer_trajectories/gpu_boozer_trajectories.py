#!/usr/bin/env python
import numpy as np
import firm3dpp

from firm3d.catapult.utils import (
    boozer_interpolant,
)

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
from firm3d.util.functions import in_github_actions

resolution = 5 if in_github_actions else 15  # Resolution for field interpolation
nparticles = 100 if in_github_actions else 10000  # Number of particles to trace
tol = 1e-4 if in_github_actions else 1e-8  # Tolerance for ODE solver

### CREATE A FIELD FOR TRACING
boozmn_filename = "../inputs/boozmn_aten_rescaled.nc"
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
def sigmav(T):
    if T > 0:
        return T ** (-2 / 3) * np.exp(-19.94 * T ** (-1 / 3))
    else:
        return 0


# Reactivity profile
reactivity = lambda s: nD(s) * nT(s) * sigmav(T(s))
stz_inits = initialize_position_profile(field, nparticles, reactivity)

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
# Initialize uniformly distributed parallel velocities
vpar0 = np.sqrt(2 * Ekin / mass)
vpar_inits = initialize_velocity_uniform(vpar0, nparticles)

stz_inits = np.ascontiguousarray(stz_inits)
vpar_inits = np.ascontiguousarray(vpar_inits)


def pseudocart_to_boozer(pt):
    x1 = pt[1]
    x2 = pt[2]
    s = np.sqrt(x1**2 + x2**2)
    theta = np.arctan2(x2, x1)
    pt[1] = s
    pt[2] = theta
    return pt


### This example shows how to save trajectories with CATAPULT
### Ideally this would be implemented across field types
def save_trajectories_boozer_gpu(
    field,
    stz_inits,
    parallel_speeds,
    tmax,
    mass,
    charge,
    vtotal,
    tol,
    ns,
    ntheta,
    nzeta,
    dt=None,
    mu=None,
    dt_save=None,
):
    n_particles = stz_inits.shape[0]
    current_time = np.zeros(n_particles)
    trajectories = [[] for _ in range(n_particles)]
    dt = -np.ones(n_particles)
    mu = -np.ones(n_particles)

    # create interpolant data
    srange, trange, zrange, quad_info, maxJ = boozer_interpolant(
        field, field.nfp, ns, ntheta, nzeta, vacuum=True
    )
    psi0 = field.psi0

    # convert Boozer to pseudo-Cartesian coordinates
    s = stz_inits[:, 0]
    theta = stz_inits[:, 1]
    x1 = s * np.cos(theta)
    x2 = s * np.sin(theta)

    stz_inits[:, 0] = x1
    stz_inits[:, 1] = x2
    stz_inits = np.ascontiguousarray(stz_inits)

    # when we filter particles out for leaving
    # we need to remember their original index
    ids = np.arange(n_particles, dtype=int)

    n_steps = int(tmax / dt_save)
    for step in range(n_steps):
        # keep track of the tmax we will reach at the end of the loop
        # each particle needs to advance to step_end_time
        local_tmax = np.maximum((step + 1) * dt_save - current_time, 0.0)

        # advance particles to step_end_time
        dt = np.ascontiguousarray(dt)
        local_tmax = np.ascontiguousarray(local_tmax)
        mu = np.ascontiguousarray(mu)

        step_data = firm3dpp.boozer_gpu_tracing(
            quad_pts=quad_info,
            srange=srange,
            trange=trange,
            zrange=zrange,
            stz_init=stz_inits.copy(),
            m=mass,
            q=charge,
            vtotal=vtotal,
            vtang=parallel_speeds.copy(),
            tmax=local_tmax,
            tol=tol,
            dt_in=dt,
            mu_in=mu,
            psi0=psi0,
            nparticles=n_particles,
            vacuum=True,
        )
        step_data = np.reshape(step_data, (n_particles, 7))

        dt = step_data[:, 5].copy()
        mu = step_data[:, 6].copy()

        # compute new current time for each particle
        step_data[:, 0] += current_time
        current_time = step_data[:, 0]

        # store data using stored indices
        for i, idx in enumerate(ids):
            if local_tmax[i] > 0.0:
                trajectories[idx].append(step_data[i, :])

        # find lost particles
        s_end = np.sqrt(step_data[:, 1] ** 2 + step_data[:, 2] ** 2)
        idx_keep = (current_time < tmax) & (s_end < 1.0)

        # remove lost particles
        stz_inits = step_data[idx_keep, 1:4].copy()
        parallel_speeds = step_data[idx_keep, 4].copy()
        ids = ids[idx_keep]
        current_time = current_time[idx_keep]
        dt = dt[idx_keep].copy()
        mu = mu[idx_keep].copy()

        n_particles = stz_inits.shape[0]

        if n_particles == 0:
            print("All particles finished")
            break

    return trajectories


tmax = 1e-3
trajectories = save_trajectories_boozer_gpu(
    bri,
    stz_inits.copy(),
    vpar_inits,
    tmax=tmax,
    mass=mass,
    charge=charge,
    vtotal=vpar0,
    tol=tol,
    ns=resolution,
    ntheta=resolution,
    nzeta=resolution,
    dt_save=1e-6,
)


### compare final positions to unsaved trajectories
srange, trange, zrange, quad_info, maxJ = boozer_interpolant(
    bri, bri.nfp, resolution, resolution, resolution, vacuum=True
)

s = stz_inits[:, 0]
theta = stz_inits[:, 1]
x1 = s * np.cos(theta)
x2 = s * np.sin(theta)
stz_inits[:, 0] = x1
stz_inits[:, 1] = x2
full_time_data = firm3dpp.boozer_gpu_tracing(
    quad_pts=quad_info,
    srange=srange,
    trange=trange,
    zrange=zrange,
    stz_init=stz_inits.copy(),
    m=mass,
    q=charge,
    vtotal=vpar0,
    vtang=vpar_inits.copy(),
    tmax=[tmax] * stz_inits.shape[0],
    tol=tol,
    psi0=bri.psi0,
    dt_in=-np.ones(stz_inits.shape[0]),
    mu_in=-np.ones(stz_inits.shape[0]),
    nparticles=stz_inits.shape[0],
    vacuum=True,
)
full_time_data = np.reshape(full_time_data, (stz_inits.shape[0], 7))

final_pos_traj = np.array([trajectory[-1] for trajectory in trajectories])

rel_diff = np.abs(full_time_data - final_pos_traj) / np.abs(full_time_data)
print("relative difference: ", rel_diff)
print("max. rel. diff: ", rel_diff.max())

print("worst starting point")
row_idx = np.argmax(np.max(rel_diff, axis=1))
print(stz_inits[row_idx])
print(vpar_inits[row_idx])
print("relative errors: ", rel_diff[row_idx])
print(final_pos_traj[row_idx])
print(full_time_data[row_idx])
