import numpy as np

from firm3d.catapult.field import CatapultPerturbedBoozerField
from firm3d.catapult.tracing import trace_particles_boozer_perturbed_gpu
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
    ShearAlfvenWavesSuperposition,
)
from firm3d.field.tracing_helpers import initialize_position_profile

# for SAW wave
from firm3d.saw.ae3d import AE3DEigenvector
from firm3d.util.constants import ALPHA_PARTICLE_CHARGE as CHARGE
from firm3d.util.constants import ALPHA_PARTICLE_MASS as MASS
from firm3d.util.constants import FUSION_ALPHA_PARTICLE_ENERGY as ENERGY
from firm3d.util.functions import in_github_actions, sigmav

import pandas as pd
import json
import time

np.random.seed(1800)

### tracing parameters
nparticles = 100000  # Number of particles to trace
tmax = 1e-2  # Time for integration
tol = 1e-6

### CREATE A FIELD FOR TRACING
boozmn_filename = "../inputs/boozmn_aten_rescaled.nc"

start_bri = time.perf_counter()
bri = BoozerRadialInterpolant(boozmn_filename, 3, enforce_vacuum=True)
bri_time = time.perf_counter() - start_bri

nfp = bri.nfp
degree = 3
n_metagrid_pts = 15  # Resolution for field interpolation
srange = (0, 1, n_metagrid_pts)
thetarange = (0, np.pi, n_metagrid_pts)
zetarange = (0, 2 * np.pi / nfp, n_metagrid_pts)

start_ibf = time.perf_counter()
field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=n_metagrid_pts,
    ntheta_interp=n_metagrid_pts,
    nzeta_interp=n_metagrid_pts,
)
ibf_time = time.perf_counter() - start_ibf

### SET UP A PERTURBED B FIELD
saw_filename = "../tracing_with_AE/ae.npy"

# generate saw object
saw = ShearAlfvenWavesSuperposition.from_ae3d(
    eigenvector=AE3DEigenvector.load_from_numpy(
        filename=saw_filename,
    ),
    B0=field,
    max_dB_normal_by_B0=5e-3,
    minor_radius_meters=1.7,
)


# Define fusion birth distribution
# Bader, A., et al. "Modeling of energetic particle transport in optimized
# stellarators." Nuclear Fusion 61.11 (2021): 116060.
nD = lambda s: 1 - s**5  # Normalized density
nT = nD
T = lambda s: 11.5 * (1 - s)  # Temperature in keV

# Reactivity profile
reactivity = lambda s: nD(s) * nT(s) * sigmav(T(s))

stz_inits = initialize_position_profile(field, nparticles, reactivity)

# tabulate the perturbed field for the GPU once
start_setup = time.perf_counter()
field_gpu_dbl = CatapultPerturbedBoozerField(
    saw, n_metagrid_pts, n_metagrid_pts, n_metagrid_pts
)
setup_time_dbl = time.perf_counter() - start_setup
field_gpu_flt = CatapultPerturbedBoozerField(
    saw, n_metagrid_pts, n_metagrid_pts, n_metagrid_pts
)
VELOCITY = np.sqrt(2 * ENERGY / MASS)
vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (nparticles,))

# The waves do work on the particles, so, as for trace_particles_boozer_perturbed,
# the magnetic moment is given per particle rather than the energy.
field.set_points(stz_inits)
mu_init = (VELOCITY**2 - vpar_init**2) / (2 * field.modB()[:, 0])

start_dbl = time.perf_counter()
res_tys_dbl, res_hits_dbl = trace_particles_boozer_perturbed_gpu(
    field_gpu_dbl,
    stz_inits,
    vpar_init,
    mu_init,
    tmax=tmax,
    mass=MASS,
    charge=CHARGE,
    Ekin=ENERGY,
    tol=tol,
)
dbl_time = time.perf_counter() - start_dbl

start_flt = time.perf_counter()
res_tys_flt, res_hits_flt = trace_particles_boozer_perturbed_gpu(
    field_gpu_flt,
    stz_inits,
    vpar_init,
    mu_init,
    tmax=tmax,
    mass=MASS,
    charge=CHARGE,
    Ekin=ENERGY,
    tol=tol,
)
flt_time = time.perf_counter() - start_flt


final_dbl = np.array([traj[-1] for traj in res_tys_dbl])
final_flt = np.array([traj[-1] for traj in res_tys_flt])
particle_data = pd.DataFrame(
    {
        "s_start": stz_inits[:, 0],
        "t_start": stz_inits[:, 1],
        "z_start": stz_inits[:, 2],
        "vpar_start": vpar_init,
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
loss_fraction_flt = float(np.mean([len(hits) > 0 for hits in res_hits_flt]))
loss_fraction_dbl = float(np.mean([len(hits) > 0 for hits in res_hits_dbl]))

print(f"tmax= {tmax}")
print(f"Number of particles= {nparticles}")
print(f"Flt. Loss fraction: {loss_fraction_flt:.3f}")
print(f"Dbl. Loss fraction: {loss_fraction_dbl:.3f}")

### record for regression testing
timing_result = {
    "nparticles": nparticles,
    "tolerance": tol,
    "resolution": n_metagrid_pts,
    "loss_fraction_dbl": loss_fraction_dbl,
    "loss_fraction_flt": loss_fraction_flt,
    "tmax": tmax,
    "times": {
        "bri_setup": bri_time,
        "field_interpolation": ibf_time,
        "catapult_setup": setup_time_dbl,
        "tracing_dbl": dbl_time,
        "tracing_flt": flt_time,
    },
}
with open("gpu_boozer_saw_tracing_results.json", "w") as f:
    json.dump(timing_result, f, indent=2)
