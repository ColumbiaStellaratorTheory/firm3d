# import time
import unittest
import numpy as np
import firm3dpp
import time
try:
    from simsopt.field.tracing import (
        IterationStoppingCriterion as SimsoptIterationStoppingCriterion,
        LevelsetStoppingCriterion as SimsoptLevelsetStoppingCriterion,
    )
    from simsopt.geo import SurfaceRZFourier

    from simsopt.field import (
        BiotSavart,
        InterpolatedField,
        SurfaceClassifier,
        coils_via_symmetries,
        load_coils_from_makegrid_file,
        trace_particles,
    )
    import simsoptpp as sopp

    HAS_SIMSOPT = True
except Exception:
    HAS_SIMSOPT = False
    InterpolatedField = type(None)
from firm3d.catapult.utils import (
    boozer_interpolant,
    boozer_saw_interpolant,
    cartesian_interpolant,
)
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
    ShearAlfvenWavesSuperposition,
)
from firm3d.field.tracing import (
    IterationStoppingCriterion,
    trace_particles_boozer,
    trace_particles_boozer_perturbed,
)
from firm3d.saw.ae3d import AE3DEigenvector
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE as CHARGE,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_MASS as MASS,
)
from firm3d.util.constants import (
    FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
)
from firm3d.catapult.field import (
    CatapultBoozerField,
    CatapultCartesianField,
    CatapultPerturbedBoozerField,
)
from firm3d.catapult.tracing import (
    save_trajectories_boozer_gpu,
    trace_particles_boozer_gpu,
    trace_particles_boozer_perturbed_gpu,
    trace_particles_cartesian_gpu,
)
from firm3d.trajectory_helpers import compute_loss_fraction
import firm3dpp

HAS_CUDA = hasattr(firm3dpp, "test_gpu_interpolation")
n_test_pts = 1000

degree = 3  # degree of interpolant
n_metagrid_pts = 32  # resolution of interpolant
order = 12  # order of coil curves

filename = "/global/homes/m/mczek/firm3d_dev/firm3d/examples/inputs/coils.curves_22_7_21"
wout_filename = "/global/homes/m/mczek/firm3d_dev/firm3d/examples/inputs/wout_aten_rescaled.nc"

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

range0 = (np.min(rs), np.max(rs), n_metagrid_pts)
range1 = (0, 2 * np.pi / surf.nfp, n_metagrid_pts * 2)
# exploit stellarator symmetry and only consider positive z values:
range2 = (0, np.max(zs), n_metagrid_pts // 2)
bsh = InterpolatedField(
    bs,
    degree,
    range0,
    range1,
    range2,
    True,
    nfp=surf.nfp,
    stellsym=True,
)

### rejection sample points inside the loss surface
rphiz = np.empty((n_test_pts, 3))
for i in range(n_test_pts):
    pt = np.random.uniform(low=0, high=1, size=(1, 3))
    pt[0, 0] = pt[0, 0] * (range0[1] - range0[0]) + range0[0]
    pt[0, 1] *= 2 * np.pi
    pt[0, 2] = (pt[0, 2] - 0.5) * 2 * range2[1]

    # particle is outside the surface or too close to the surface
    max_iters = 1000
    for _ in range(max_iters):
        if sc_particle.evaluate_rphiz(pt) > 0.2:
            break
        pt = np.random.uniform(low=0, high=1, size=(1, 3))
        pt[0, 0] = pt[0, 0] * (range0[1] - range0[0]) + range0[0]
        pt[0, 1] *= 2 * np.pi
        pt[0, 2] = (pt[0, 2] - 0.5) * 2 * range2[1]
    else:
        raise RuntimeError("Could not sample a valid point inside the surface")
    rphiz[i, :] = pt

range0, range1, range2, quad_info = cartesian_interpolant(
    bsh, sc_particle, tracing_model="fo"
)
r = rphiz[:, 0].reshape(-1, 1)
phi = rphiz[:, 1].reshape(-1, 1)
z = rphiz[:, 2].reshape(-1, 1)
x = r * np.cos(phi)
y = r * np.sin(phi)
xyz = np.hstack((x, y, z))

VELOCITY = np.sqrt(2 * ENERGY / MASS)
vpar_init = np.random.normal(size=(n_test_pts, 3))
vpar_init /= np.linalg.norm(vpar_init, axis=1)[:, np.newaxis]
vpar_init *= VELOCITY

tmax=2e-4
tol=1e-6
### cpu tracing
tys = []
res_phi_hits = []
tracing_start = time.perf_counter()
for i in range(xyz.shape[0]):
    print(f"Tracing particle {i+1}/{xyz.shape[0]} on CPU...")
    res_ty, res_phi_hit = sopp.particle_fullorbit_tracing(
                            bsh, xyz[i, :], vpar_init[i, :],
                            MASS, CHARGE, tmax=tmax, tol=tol, stopping_criteria=[SimsoptLevelsetStoppingCriterion(sc_particle.dist)])
    tys.append(res_ty)
tracing_end = time.perf_counter()
print(f"CPU tracing took {tracing_end - tracing_start:.6f} seconds")
cpu_loss_times = np.array([ty[-1][0] for ty in tys])

### gpu_tracing
tracing_start = time.perf_counter()
gpu_output = firm3dpp.fo_cartesian_gpu_tracing(
    quad_pts=quad_info,
    rrange=range0,
    phirange=range1,
    zrange=range2,
    xyz_init=xyz,
    m=MASS,
    q=CHARGE,
    vtotal=VELOCITY,
    vtang=vpar_init,
    tmax=np.array([tmax] * n_test_pts),
    tol=tol,
    dt_in=np.array([-1.0] * n_test_pts),
    mu_in=np.array([-1.0] * n_test_pts),
    nparticles=n_test_pts,
)
tracing_end = time.perf_counter()
print(f"GPU tracing took {tracing_end - tracing_start:.6f} seconds")
print(f"Number of test points: {n_test_pts}, tmax: {tmax}, tol: {tol}")
gpu_output = np.reshape(gpu_output, (n_test_pts, -1))
gpu_loss_times = gpu_output[:, 0]
# print(gpu_loss_times)

cpu_loss_frac = np.mean(cpu_loss_times < tmax)
gpu_loss_frac = np.mean(gpu_loss_times < tmax)
print("CPU loss fraction:", cpu_loss_frac)
print("GPU loss fraction:", gpu_loss_frac)