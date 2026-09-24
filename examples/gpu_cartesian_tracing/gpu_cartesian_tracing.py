import numpy as np
import pandas as pd
from simsopt.field import (
    BiotSavart,
    InterpolatedField,
    SurfaceClassifier,
    coils_via_symmetries,
    load_coils_from_makegrid_file,
)
from simsopt.field.sampling import draw_uniform_on_surface
from simsopt.geo import SurfaceRZFourier
from simsopt.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)

from firm3d.catapult.field import CatapultCartesianField
from firm3d.catapult.tracing import trace_particles_cartesian_gpu
from firm3d.field.tracing_helpers import (
    initialize_velocity_uniform,
)
import json
import time

degree = 3  # degree of interpolant
resolution = 16  # resolution of interpolant
order = 12  # order of coil curves
nparticles = 100000
tmax = 1e-2
tol=1e-6

filename = "../inputs/coils.curves_22_7_21"
wout_filename = "../inputs/wout_aten_rescaled.nc"

surf = SurfaceRZFourier.from_wout(wout_filename)

coils = load_coils_from_makegrid_file(filename, order, ppp=20, group_names=None)

curves = []
currents = []
for _i, coil in enumerate(coils):
    curves.append(coil.curve)
    currents.append(coil.current)

coils_full = coils_via_symmetries(curves, currents, surf.nfp, True)

start_field = time.perf_counter()
bs = BiotSavart(coils_full)
field_time = time.perf_counter() - start_field 

surf_launch = SurfaceRZFourier.from_wout(wout_filename, s=0.3)

sc_particle = SurfaceClassifier(surf, h=0.1, p=2)
rs = np.linalg.norm(surf.gamma()[:, :, 0:2], axis=2)
zs = surf.gamma()[:, :, 2]

rrange = (np.min(rs), np.max(rs), resolution)
phirange = (0, 2 * np.pi / surf.nfp, resolution * 2)
# exploit stellarator symmetry and only consider positive z values:
zrange = (0, np.max(zs), resolution // 2)

start_if = time.perf_counter()
bsh = InterpolatedField(
    bs, degree, rrange, phirange, zrange, True, nfp=surf.nfp, stellsym=True
)
if_time = time.perf_counter() - start_if

# sample particles from surface
xyz, _ = draw_uniform_on_surface(surf_launch, nparticles, safetyfactor=10)

vpar0 = np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)
vpar_inits = initialize_velocity_uniform(vpar0, nparticles)

# tabulate the field and the boundary distance for the GPU once
field_dbl = CatapultCartesianField(bsh, sc_particle, precision="double")
start_dbl = time.perf_counter()
res_tys_dbl, res_hits_dbl = trace_particles_cartesian_gpu(
    field_dbl,
    xyz,
    vpar_inits,
    tmax=tmax,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
    tol=tol,
    forget_exact_path=True,
)
dbl_time = time.perf_counter() - start_dbl


field_flt = CatapultCartesianField(bsh, sc_particle, precision="single")
start_flt = time.perf_counter()
res_tys_flt, res_hits_flt = trace_particles_cartesian_gpu(
    field_flt,
    xyz,
    vpar_inits,
    tmax=tmax,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
    tol=tol,
    forget_exact_path=True,
)
flt_time = time.perf_counter() - start_flt

loss_fraction_flt = float(np.mean([len(hits) > 0 for hits in res_hits_flt]))
loss_fraction_dbl = float(np.mean([len(hits) > 0 for hits in res_hits_dbl]))

### record for regression testing
timing_result = {
    "nparticles": nparticles,
    "tolerance": tol,
    "resolution": resolution,
    "loss_fraction_dbl": loss_fraction_dbl,
    "loss_fraction_flt": loss_fraction_flt,
    "tmax": tmax,
    "times": {
        "field_setup": field_time,
        "field_interpolation": if_time,
        "tracing_dbl": dbl_time,
        "tracing_flt": flt_time,
    },
}
with open("gpu_cartesian_tracing_results.json", "w") as f:
    json.dump(timing_result, f, indent=2)