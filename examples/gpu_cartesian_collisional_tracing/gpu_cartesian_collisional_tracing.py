import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from simsopt.field import (
    BiotSavart,
    InterpolatedField,
    SurfaceClassifier,
    coils_via_symmetries,
    load_coils_from_makegrid_file,
)
from simsopt.geo import SurfaceRZFourier

from firm3d.catapult.tracing import trace_particles_cartesian_with_collisions_gpu
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.collisions import ThermalBackground
from firm3d.field.coordinates import boozer_to_cylindrical
from firm3d.field.tracing_helpers import (
    initialize_position_profile,
    initialize_velocity_uniform,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    FUSION_ALPHA_PARTICLE_ENERGY,
    PROTON_MASS,
)

degree = 3  # degree of interpolant
n = 16  # resolution of interpolant
order = 12  # order of coil curves
# 10000 rather than 1000 so the loss fraction is not Poisson-limited.
nparticles = 10000
tmax = 2e-1
tol = 1e-8

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

# Build the normalized toroidal flux s(r, phi, z) for evaluation of profiles.
bri = BoozerRadialInterpolant(wout_filename, 3, enforce_vacuum=True)
bfield = InterpolatedBoozerField(bri, 3, ns_interp=n, ntheta_interp=n, nzeta_interp=n)

n_s, n_ang = 48, 48
s_grid = np.linspace(0.02, 1.0, n_s)
theta_grid = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
zeta_grid = np.linspace(0, 2 * np.pi, bri.nfp * n_ang, endpoint=False)
stz_samples = (
    np.array(np.meshgrid(s_grid, theta_grid, zeta_grid, indexing="ij")).reshape(3, -1).T
)
cyl_samples = boozer_to_cylindrical(bfield, stz_samples)
tree = cKDTree(
    np.column_stack(
        [
            cyl_samples[:, 0] * np.cos(cyl_samples[:, 1]),
            cyl_samples[:, 0] * np.sin(cyl_samples[:, 1]),
            cyl_samples[:, 2],
        ]
    )
)
s_samples = stz_samples[:, 0]


def flux_label(points_rphiz):
    """Normalized toroidal flux at cylindrical points (r, phi, z)."""
    p = np.asarray(points_rphiz)
    _, idx = tree.query(
        np.column_stack([p[:, 0] * np.cos(p[:, 1]), p[:, 0] * np.sin(p[:, 1]), p[:, 2]])
    )
    return s_samples[idx]


# Fusion-reactivity birth distribution, sampled in Boozer coordinates and
# mapped through the equilibrium.
nD = lambda s: 1 - s**5  # Normalized density
T_keV = lambda s: 11.5 * (1 - s)


def sigmav(T):
    if T > 0:
        return T ** (-2 / 3) * np.exp(-19.94 * T ** (-1 / 3))
    else:
        return 0


reactivity = lambda s: nD(s) * nD(s) * sigmav(T_keV(s))

# Background plasma the alphas collide with: a 50/50 DT fuel mix and the
# electrons that neutralize it.  Temperature is in eV at this interface.
n_ref = 1e20  # m^-3
ne = lambda s: n_ref * nD(s)
Te = lambda s: 1e3 * T_keV(s)

backgrounds = [
    ThermalBackground(
        n_profile=lambda s: 0.5 * ne(s),
        T_profile=Te,
        mass=2 * PROTON_MASS,
        charge=ELEMENTARY_CHARGE,
    ),
    ThermalBackground(
        n_profile=lambda s: 0.5 * ne(s),
        T_profile=Te,
        mass=3 * PROTON_MASS,
        charge=ELEMENTARY_CHARGE,
    ),
    ThermalBackground(
        n_profile=ne,
        T_profile=Te,
        mass=ELECTRON_MASS,
        charge=-ELEMENTARY_CHARGE,
    ),
]

np.random.seed(8)
stz_inits = initialize_position_profile(bfield, nparticles, reactivity)
cyl_inits = boozer_to_cylindrical(bfield, stz_inits)
xyz = np.column_stack(
    [
        cyl_inits[:, 0] * np.cos(cyl_inits[:, 1]),
        cyl_inits[:, 0] * np.sin(cyl_inits[:, 1]),
        cyl_inits[:, 2],
    ]
)

v0 = np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)
vpar_inits = initialize_velocity_uniform(v0, nparticles)

last_time = trace_particles_cartesian_with_collisions_gpu(
    bsh,
    sc_particle,
    flux_label,
    xyz,
    vpar_inits,
    backgrounds=backgrounds,
    tmax=tmax,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    vtotal=v0,
    tol=tol,
    rng_seed=0,
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
        "v_end": last_time[:, 5],
        "last_time": last_time[:, 0],
        "dt_end": last_time[:, 6],
    }
)
particle_data.to_csv("./particle_data.csv")

t_end = last_time[:, 0]
v_end = last_time[:, 5]
lost = t_end < tmax

particle_loss = lost.sum() / nparticles
energy_loss = np.sum((v_end[lost] / v0) ** 2) / nparticles

print(f"Number of particles= {nparticles}")
print(f"Particle loss fraction: {particle_loss:.3f}")
print(f"Energy loss fraction: {energy_loss:.3f}")
print(f"Mean energy fraction of confined: {np.mean((v_end[~lost] / v0) ** 2):.4f}")
