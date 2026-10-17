"""Save and classify lost fusion-alpha GPU orbits, matching the CPU example."""

import time

import numpy as np

from firm3d._core.util import parallel_loop_bounds
from firm3d.catapult.field import CatapultBoozerField
from firm3d.catapult.tracing import trace_particles_boozer_gpu
from firm3d.field.boozermagneticfield import InterpolatedBoozerField
from firm3d.field.tracing import MaxToroidalFluxStoppingCriterion
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
from firm3d.util.functions import in_github_actions, proc0_print, setup_logging, sigmav
from firm3d.util.mpi import comm_size, comm_world

boozmn_filename = "../inputs/boozmn_ariescs.nc"
resolution = 10 if in_github_actions else 48  # Resolution for field interpolation
nParticles = 50 if in_github_actions else 5000  # Number of particles to trace
tol = 1e-4 if in_github_actions else 1e-8  # Absolute and relative solver tolerance
order = 3  # Order for field interpolation
tmax = 1e-4 if in_github_actions else 1e-2  # Time for integration
dt_save = 1e-7  # Time interval for saving trajectory points
batch_size = 128  # Bound the number of histories held in GPU memory
max_hits = 10000  # Per-particle capacity for mirror and loss hits
helicity_M = 1  # Field-strength helicity for orbit classification
helicity_N = 0
Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE

setup_logging(f"stdout_{nParticles}_{resolution}_{comm_size}.txt")
time1 = time.time()

# Setup field interpolation
field = InterpolatedBoozerField.from_booz_xform(
    boozmn_filename,
    degree=order,
    ns=resolution,
    ntheta=resolution,
    nzeta=resolution,
    comm=comm_world,
)
gpu_field = CatapultBoozerField(field, resolution, resolution, resolution)

# Same fusion birth distribution and seed as the CPU loss-classification example.
nD = lambda s: 1 - s**5  # Normalized density
nT = nD
T = lambda s: 11.5 * (1 - s)  # Temperature in keV
reactivity = lambda s: nD(s) * nT(s) * sigmav(T(s))
points_init = initialize_position_profile(
    field, nParticles, reactivity, comm=comm_world, seed=0
)
vpar0 = np.sqrt(2 * Ekin / mass)
vpar_init = initialize_velocity_uniform(vpar0, nParticles, comm=comm_world, seed=0)
oc = OrbitClassification(field, Ekin, mass, charge, helicity_M, helicity_N)

# Trace in bounded batches until the particles hit s=1 or reach tmax.
first, last = parallel_loop_bounds(comm_world, nParticles)
for start in range(first, last, batch_size):
    end = min(start + batch_size, last)
    res_tys, res_hits = trace_particles_boozer_gpu(
        gpu_field,
        points_init[start:end],
        vpar_init[start:end],
        tmax=tmax,
        mass=mass,
        charge=charge,
        Ekin=Ekin,
        vpars=[0],
        vpars_stop=False,
        stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        forget_exact_path=False,
        tol=tol,
        dt_save=dt_save,
        max_hits=max_hits,
    )
    for iParticle, (res_ty, res_hit) in enumerate(zip(res_tys, res_hits), start=start):
        if not len(res_hit) or not np.any(res_hit[:, 1] == -1):
            continue  # Particle was not lost to the wall, skip
        np.savetxt(f"particle_{iParticle}_traj.txt", res_ty)
        np.savetxt(f"particle_{iParticle}_hits.txt", res_hit)
        particle_dict = oc.classify_orbit(res_ty, res_hit)
        np.savez(f"particle_{iParticle}.npz", **particle_dict)
    del res_tys, res_hits

proc0_print(
    f"Total time for tracing and classifying particles: "
    f"{time.time() - time1:.2f} seconds"
)
