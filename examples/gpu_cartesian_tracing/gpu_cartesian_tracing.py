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

from firm3d.catapult.tracing import trace_particles_cartesian_gpu
from firm3d.field.tracing_helpers import (
    initialize_velocity_uniform,
)

degree = 3  # degree of interpolant
n = 16  # resolution of interpolant
order = 12  # order of coil curves

filename = "../inputs/coils.curves_22_7_21"
wout_filename = "../inputs/wout_aten_rescaled.nc"

surf = SurfaceRZFourier.from_wout(wout_filename)

coils = load_coils_from_makegrid_file(filename, order, ppp=20, group_names=None)

curves = []
currents = []
for _i, coil in enumerate(coils):
    curves.append(coil.curve)
    currents.append(coil.current)

# coils.curves_22_7_21 holds the stellarator-symmetric half of a full-torus
# 40-coil set, so only stellsym is applied here.
coils_full = coils_via_symmetries(curves, currents, 1, True)
bs = BiotSavart(coils_full)

surf_launch = SurfaceRZFourier.from_wout(wout_filename, s=0.3)

sc_particle = SurfaceClassifier(surf, h=0.1, p=2)
rs = np.linalg.norm(surf.gamma()[:, :, 0:2], axis=2)
zs = surf.gamma()[:, :, 2]

rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2 * np.pi / surf.nfp, n * 2)
# exploit stellarator symmetry and only consider positive z values:
zrange = (0, np.max(zs), n // 2)
bsh = InterpolatedField(
    bs, degree, rrange, phirange, zrange, True, nfp=surf.nfp, stellsym=True
)

# sample particles from surface
nparticles = 1000
xyz, _ = draw_uniform_on_surface(surf_launch, nparticles, safetyfactor=10)

v0 = np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)
vpar_inits = initialize_velocity_uniform(v0, nparticles)

tmax = 1e-5
last_time = trace_particles_cartesian_gpu(
    bsh,
    sc_particle,
    xyz,
    vpar_inits,
    tmax=tmax,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    vtotal=v0,
    tol=1e-8,
)
particle_data = pd.DataFrame(
    {
        "x_start": xyz[:, 0],
        "y_start": xyz[:, 1],
        "z_start": xyz[:, 2],
        "vpar_start": vpar_inits,
        "x_end": last_time[:, 1],
        "y_end": last_time[:, 2],
        "z_end": last_time[:, 3],
        "vpar_end": last_time[:, 4],
        "last_time": last_time[:, 0],
        "dt_end": last_time[:, 5],
    }
)
particle_data.to_csv("./particle_data.csv")


did_leave = [t < tmax for t in particle_data["last_time"]]
loss_frac = sum(did_leave) / len(did_leave)
print(f"Number of particles= {nparticles}")
print(f"Loss fraction: {loss_frac:.3f}")
