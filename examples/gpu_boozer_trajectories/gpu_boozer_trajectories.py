#!/usr/bin/env python
import h5py
import numpy as np

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
nparticles = 100 if in_github_actions else 10000  # Number of particles to trace
tol = 1e-4 if in_github_actions else 1e-8  # Tolerance for ODE solver
tmax = 1e-4 if in_github_actions else 1e-3  # Tracing time
dt_save = 1e-6  # Interval at which to save the trajectory

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

# Reactivity profile
reactivity = lambda s: nD(s) * nT(s) * sigmav(T(s))
stz_inits = initialize_position_profile(field, nparticles, reactivity)

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
# Initialize uniformly distributed parallel velocities
vpar0 = np.sqrt(2 * Ekin / mass)
vpar_inits = initialize_velocity_uniform(vpar0, nparticles)

### SAVE TRAJECTORIES
# As for trace_particles_boozer on the CPU: with forget_exact_path=False,
# res_tys holds each particle's (t, s, theta, zeta, vpar) rows every dt_save,
# from the initial state to the state where tracing stopped, and res_hits is
# non-empty for a particle that was lost.
# tabulate the field for the GPU once; the tracing and the check share it
field_gpu = CatapultBoozerField(bri, resolution, resolution, resolution)

res_tys, res_hits = trace_particles_boozer_gpu(
    field_gpu,
    stz_inits,
    vpar_inits,
    tmax=tmax,
    Ekin=Ekin,
    mass=mass,
    charge=charge,
    tol=tol,
    dt_save=dt_save,
)

with h5py.File("trajectories.h5", "w") as f:
    f.attrs["dt_save"] = dt_save
    f.attrs["tmax"] = tmax
    f.attrs["boozmn"] = boozmn_filename
    f.attrs["n_particles"] = len(res_tys)
    for i, traj in enumerate(res_tys):
        f.create_dataset(f"particle_{i:06d}", data=traj)

### CHECK AGAINST A SINGLE UNINTERRUPTED TRACE
# Feeding dt and mu back between chunks continues the same adaptive step
# sequence as a single trace, so while steps are error-limited the final
# states agree to roundoff. When the tolerance is loose enough that steps
# are capped by the maximum step size, which the kernel sets from the field
# at the start of each call, the two runs take different steps and differ
# at the level of the integration error.
res_tys_single, res_hits_single = trace_particles_boozer_gpu(
    field_gpu,
    stz_inits,
    vpar_inits,
    tmax=tmax,
    Ekin=Ekin,
    mass=mass,
    charge=charge,
    tol=tol,
    forget_exact_path=True,
)
final_saved = np.array([traj[-1] for traj in res_tys])
final_single = np.array([traj[-1] for traj in res_tys_single])
lost = np.array([len(hits) > 0 for hits in res_hits])
lost_single = np.array([len(hits) > 0 for hits in res_hits_single])
print(
    f"Number of particles = {nparticles}, lost = {lost.sum()} "
    f"(single trace: {lost_single.sum()})"
)
ds = np.abs(final_saved[:, 1] - final_single[:, 1])
print(
    f"|s_final(saved) - s_final(single trace)|: median {np.median(ds):.3e}, "
    f"max {ds.max():.3e}"
)
