import numpy as np

from firm3d.catapult.tracing import trace_particles_boozer_gpu
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


# Define fusion birth distribution
# Bader, A., et al. "Modeling of energetic particle transport in optimized
# stellarators." Nuclear Fusion 61.11 (2021): 116060.
nD = lambda s: 1 - s**5  # Normalized density
nT = nD
T = lambda s: 11.5 * (1 - s)  # Temperature in keV

# Reactivity profile
reactivity = lambda s: nD(s) * nT(s) * sigmav(T(s))

stz = initialize_position_profile(field, nparticles, reactivity)
VELOCITY = np.sqrt(2 * ENERGY / MASS)
vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (nparticles,))

tol = 1e-4 if in_github_actions else 1e-9  # Tolerance for ODE solver
last_time = trace_particles_boozer_gpu(
    saw,
    stz,
    vpar_init,
    tmax,
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
