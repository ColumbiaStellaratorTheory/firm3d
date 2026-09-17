import h5py
import numpy as np
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

from firm3d.catapult.tracing import (
    save_trajectories_cartesian_gpu,
    trace_particles_cartesian_gpu,
)
from firm3d.field.tracing_helpers import (
    initialize_velocity_uniform,
)
from firm3d.util.functions import in_github_actions

degree = 3  # degree of interpolant
n = 30  # resolution of interpolant
order = 12  # order of coil curves
tol = 1e-8
nparticles = 100 if in_github_actions else 1000  # Number of particles to trace
tmax = 1e-4  # Tracing time
# Interval at which to save the trajectory. Steps at this tolerance are
# 0.5-4e-7 s, and a save time inside a step gets no row, so keep dt_save above
# the step size.
dt_save = 1e-6

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

xyz_inits, _ = draw_uniform_on_surface(surf_launch, nparticles, safetyfactor=10)

vpar0 = np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)
vpar_inits = initialize_velocity_uniform(vpar0, nparticles)

### SAVE TRAJECTORIES
# Each entry is an (nsaved, 7) array of (t, x, y, z, vpar, dt, mu), one row per
# multiple of dt_save reached, at the first step boundary after it; lost
# particles have fewer rows.
trajectories = save_trajectories_cartesian_gpu(
    bsh,
    sc_particle,
    xyz_inits,
    vpar_inits,
    tmax=tmax,
    dt_save=dt_save,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    vtotal=vpar0,
    tol=tol,
)

with h5py.File("trajectories.h5", "w") as f:
    f.attrs["dt_save"] = dt_save
    f.attrs["tmax"] = tmax
    f.attrs["wout"] = wout_filename
    f.attrs["n_particles"] = len(trajectories)
    for i, traj in enumerate(trajectories):
        f.create_dataset(f"particle_{i:06d}", data=traj)

### CHECK AGAINST A SINGLE UNINTERRUPTED TRACE
# Feeding dt and mu back between chunks continues the same adaptive step
# sequence as a single trace, so while steps are error-limited the final
# states agree to roundoff. When the tolerance is loose enough that steps
# are capped by the maximum step size, which the kernel sets from the field
# at the start of each call, the two runs take different steps and differ
# at the level of the integration error.
last_time = trace_particles_cartesian_gpu(
    bsh,
    sc_particle,
    xyz_inits,
    vpar_inits,
    tmax=tmax,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    vtotal=vpar0,
    tol=tol,
)
final_saved = np.array([traj[-1] for traj in trajectories])
lost = last_time[:, 0] < 0.999 * tmax
print(f"Number of particles = {nparticles}, lost = {lost.sum()}")
dx = np.linalg.norm(final_saved[:, 1:4] - last_time[:, 1:4], axis=1)
print(
    f"|x_final(saved) - x_final(single trace)|: median {np.median(dx):.3e} m, "
    f"max {dx.max():.3e} m"
)
