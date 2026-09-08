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

from firm3d.field.tracing_helpers import (
    initialize_velocity_uniform,
)
from firm3d.catapult.utils import cartesian_interpolant
import firm3dpp

degree = 3  # degree of interpolant
n = 30  # resolution of interpolant
order = 12  # order of coil curves
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

nparticles = 1000
xyz_inits, _ = draw_uniform_on_surface(surf_launch, nparticles, safetyfactor=10)

vpar0 = np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)
vpar_inits = initialize_velocity_uniform(vpar0, nparticles)


### This example shows how to save trajectories with CATAPULT
### Ideally this would be implemented across field types
def save_trajectories_cartesian_gpu(
    field,
    surface_classifier,
    xyz_inits,
    parallel_speeds,
    tmax,
    mass,
    charge,
    vtotal,
    tol,
    dt=None,
    mu=None,
    dt_save=None,
):
    n_particles = xyz_inits.shape[0]
    current_time = np.zeros(n_particles)
    trajectories = [[] for _ in range(n_particles)]
    dt = -np.ones(n_particles).astype(xyz_inits.dtype)
    mu = -np.ones(n_particles).astype(xyz_inits.dtype)

    # create interpolant data
    r_range, phi_range, z_range, quad_info = cartesian_interpolant(
        field, surface_classifier, dtype=xyz_inits.dtype
    )

    # when we filter particles out for leaving
    # we need to remember their original index
    ids = np.arange(n_particles, dtype=int)

    n_steps = int(tmax / dt_save)
    for step in range(n_steps):
        # keep track of the tmax we will reach at the end of the loop
        # each particle needs to advance to step_end_time
        local_tmax = np.maximum((step + 1) * dt_save - current_time, 0.0)

        # advance particles to step_end_time
        dt = np.ascontiguousarray(dt)
        local_tmax = np.ascontiguousarray(local_tmax)
        mu = np.ascontiguousarray(mu)

        step_data = firm3dpp.cartesian_gpu_tracing(
            quad_pts=quad_info,
            rrange=r_range,
            phirange=phi_range,
            zrange=z_range,
            xyz_init=xyz_inits,
            m=mass,
            q=charge,
            vtotal=vtotal,
            vtang=parallel_speeds,
            tmax=local_tmax,
            tol=tol,
            dt_in=dt,
            mu_in=mu,
            nparticles=n_particles,
        )
        step_data = np.reshape(step_data, (n_particles, 7))

        dt = step_data[:, 5].copy()
        mu = step_data[:, 6].copy()

        # find lost particles
        idx_keep = step_data[:, 0] >= local_tmax

        # compute new current time for each particle
        step_data[:, 0] += current_time
        current_time = step_data[:, 0]

        # store data using stored indices
        for i, idx in enumerate(ids):
            if local_tmax[i] > 0.0:
                trajectories[idx].append(step_data[i, :])

        # remove lost particles
        xyz_inits = step_data[idx_keep, 1:4].copy()
        parallel_speeds = step_data[idx_keep, 4].copy()
        ids = ids[idx_keep]
        current_time = current_time[idx_keep]
        dt = dt[idx_keep].copy()
        mu = mu[idx_keep].copy()

        n_particles = xyz_inits.shape[0]

        if n_particles == 0:
            print("All particles finished")
            break

    return trajectories


tmax = 1e-5
trajectories = save_trajectories_cartesian_gpu(
    bsh,
    sc_particle,
    xyz_inits.copy(),
    vpar_inits,
    tmax=tmax,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    vtotal=vpar0,
    tol=tol,
    dt_save=1e-7,
)


### save trajectories
import h5py

with h5py.File("trajectories.h5", "w") as f:
    f.attrs["dt_save"] = 1e-6
    f.attrs["tmax"] = tmax
    f.attrs["wout"] = wout_filename
    f.attrs["n_particles"] = len(trajectories)
    for i, traj in enumerate(trajectories):
        f.create_dataset(f"particle_{i:06d}", data=np.asarray(traj, dtype=np.float32))

# trajectory_data = pd.concat([df_from_trajectory(trajectory, i) for i, trajectory in enumerate(trajectories)], ignore_index=True)

### compare final positions to unsaved trajectories
r_range, phi_range, z_range, quad_info = cartesian_interpolant(
    bsh, sc_particle, dtype=xyz_inits.dtype
)
dt = -np.ones(nparticles).astype(xyz_inits.dtype)
mu = -np.ones(nparticles).astype(xyz_inits.dtype)
full_time_data = firm3dpp.cartesian_gpu_tracing(
    quad_pts=quad_info,
    rrange=r_range,
    phirange=phi_range,
    zrange=z_range,
    xyz_init=xyz_inits,
    m=ALPHA_PARTICLE_MASS,
    q=ALPHA_PARTICLE_CHARGE,
    vtotal=vpar0,
    vtang=vpar_inits,
    tmax=[tmax for _ in range(nparticles)],
    tol=tol,
    dt_in=dt,
    mu_in=mu,
    nparticles=nparticles,
)
full_time_data = np.reshape(full_time_data, (xyz_inits.shape[0], 7))

final_pos_traj = np.array([trajectory[-1] for trajectory in trajectories])
print(full_time_data.dtype)
print(final_pos_traj.dtype)

print(np.isnan(full_time_data).sum())
print(np.isnan(final_pos_traj).sum())

print(final_pos_traj[np.isnan(final_pos_traj).any(axis=1)])

rel_diff = np.abs(full_time_data - final_pos_traj) / np.abs(full_time_data)
print("relative difference: ", rel_diff)
print("max. rel. diff: ", rel_diff.max())

print("worst starting point")
row_idx = np.argmax(np.max(rel_diff, axis=1))
print(xyz_inits[row_idx])
print(vpar_inits[row_idx])
print("relative errors: ", rel_diff[row_idx])
print(final_pos_traj[row_idx])
print(full_time_data[row_idx])
