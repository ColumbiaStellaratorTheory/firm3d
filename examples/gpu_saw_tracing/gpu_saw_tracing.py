import numpy as np

from firm3d.catapult.tracing import trace_particles_boozer_gpu
from firm3d.catapult.utils import boozer_saw_interpolant
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
    ShearAlfvenWavesSuperposition,
)

# for SAW wave
from firm3d.saw.ae3d import AE3DEigenvector
from firm3d.util.constants import ALPHA_PARTICLE_CHARGE as CHARGE
from firm3d.util.constants import ALPHA_PARTICLE_MASS as MASS
from firm3d.util.constants import FUSION_ALPHA_PARTICLE_ENERGY as ENERGY
from firm3d.util.functions import in_github_actions
from firm3d.util.sampling import sample_stz

np.random.seed(1800)

### tracing parameters
nparticles = 100 if in_github_actions else 25000  # Number of particles to trace
tmax = 1e-4 if in_github_actions else 1e-3  # Time for integration


### CREATE A FIELD FOR TRACING
boozmn_filename = "../inputs/boozmn_aten_rescaled.nc"
bri = BoozerRadialInterpolant(boozmn_filename, 3, enforce_vacuum=True)

nfp = bri.nfp
degree = 3
n_metagrid_pts = 5 if in_github_actions else 15  # Resolution for field interpolation
srange = (0, 1, n_metagrid_pts)
thetarange = (0, np.pi, n_metagrid_pts)
zetarange = (0, 2 * np.pi / nfp, n_metagrid_pts)
field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=n_metagrid_pts,
    ntheta_interp=n_metagrid_pts,
    nzeta_interp=n_metagrid_pts,
)

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


# set up interpolant data
srange, trange, zrange, quad_info, maxJ = boozer_saw_interpolant(
    field, nfp, n_metagrid_pts, n_metagrid_pts, n_metagrid_pts
)

stz_inits = np.vstack([sample_stz(field, maxJ) for i in range(nparticles)])
VELOCITY = np.sqrt(2 * ENERGY / MASS)
vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (nparticles,))
stz = np.ascontiguousarray(stz_inits)

tol = 1e-4 if in_github_actions else 1e-9  # Tolerance for ODE solver
last_time = trace_particles_boozer_gpu(
    saw,
    stz,
    vpar_init,
    [tmax for _ in range(nparticles)],
    MASS,
    CHARGE,
    np.sqrt(2 * ENERGY / MASS),
    tol,
    n_metagrid_pts,
    n_metagrid_pts,
    n_metagrid_pts,
)
loss_times = last_time[:, 0]
print("loss times: ", loss_times)
print("loss frac: ", np.mean(loss_times < tmax))
