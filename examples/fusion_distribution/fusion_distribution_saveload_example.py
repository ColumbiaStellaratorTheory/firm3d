"""
Example: trace with original field, save field to JSON, trace with loaded field,
then compare one saved value (first trajectory) between the two runs.

Demonstrates that save/load preserves the field so that tracing gives the same
result. Saves the first trajectory from each run with np.savetxt and prints the
absolute difference (should be at numerical noise level).
"""

import time

import numpy as np

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    trace_particles_boozer,
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
from firm3d.util.functions import proc0_print, setup_logging, sigmav
from firm3d.util.mpi import comm_size, comm_world

time1 = time.time()

# Same setup as fusion_distribution.py (can reduce resolution/nParticles for quick run)
resolution = 48
nParticles = 1
reltol = 1e-8
abstol = 1e-8
order = 3
degree = 3
boozmn_filename = "../inputs/boozmn_aten_rescaled.nc"
tmax = 1e-2
ns_interp = resolution
ntheta_interp = resolution
nzeta_interp = resolution

setup_logging(f"stdout_saveload_{nParticles}_{resolution}_{comm_size}.txt")

bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm_world)
field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

nD = lambda s: 1 - s**5
nT = nD
T = lambda s: 11.5 * (1 - s)


reactivity = lambda s: nD(s) * nT(s) * sigmav(T(s))

points = initialize_position_profile(field, nParticles, reactivity, comm=comm_world)
Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
vpar0 = np.sqrt(2 * Ekin / mass)
vpar_init = initialize_velocity_uniform(vpar0, nParticles, comm=comm_world)

# --- Trace with original field ---
res_tys_orig, _ = trace_particles_boozer(
    field,
    points,
    vpar_init,
    tmax=tmax,
    mass=mass,
    charge=charge,
    comm=comm_world,
    Ekin=Ekin,
    stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
    forget_exact_path=True,
    abstol=abstol,
    reltol=reltol,
)

# Save first trajectory from original run (one value we will compare)
trajectory_original_txt = "trajectory_original.txt"
if len(res_tys_orig) > 0:
    np.savetxt(trajectory_original_txt, res_tys_orig[0], fmt="%.18e")

time2 = time.time()
proc0_print("Elapsed time for original tracing = ", time2 - time1)

# --- Save field to JSON and load it ---
field_json_path = "field.json"
field.to_json(field_json_path)
loaded_field = InterpolatedBoozerField.from_json(field_json_path)

time3 = time.time()
proc0_print("Elapsed time for save/load = ", time3 - time2)

# --- Trace with loaded field (same points and vpar) ---
res_tys_loaded, _ = trace_particles_boozer(
    loaded_field,
    points,
    vpar_init,
    tmax=tmax,
    mass=mass,
    charge=charge,
    comm=comm_world,
    Ekin=Ekin,
    stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
    forget_exact_path=True,
    abstol=abstol,
    reltol=reltol,
)

# Save first trajectory from loaded-field run
trajectory_loaded_txt = "trajectory_loaded.txt"
if len(res_tys_loaded) > 0:
    np.savetxt(trajectory_loaded_txt, res_tys_loaded[0], fmt="%.18e")

time4 = time.time()
proc0_print("Elapsed time for loaded-field tracing = ", time4 - time3)

# --- Load both saved trajectories and print absolute difference ---
if len(res_tys_orig) > 0 and len(res_tys_loaded) > 0:
    orig = np.loadtxt(trajectory_original_txt)
    loaded = np.loadtxt(trajectory_loaded_txt)
    # Align shapes: trajectories can differ in length if one hit boundary earlier
    n = min(orig.shape[0], loaded.shape[0])
    diff = np.abs(orig[:n] - loaded[:n])
    max_abs_diff = np.max(diff)
    mean_abs_diff = np.mean(diff)
    proc0_print("Absolute difference (first trajectory, first n rows):")
    proc0_print("  max  = ", max_abs_diff)
    proc0_print("  mean = ", mean_abs_diff)
else:
    proc0_print("No trajectories to compare (empty results).")

proc0_print("Total elapsed = ", time4 - time1)
