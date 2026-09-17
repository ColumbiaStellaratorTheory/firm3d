__all__ = ["trace_particles_boozer_gpu", "trace_particles_cartesian_gpu"]
import numpy as np

import firm3dpp
from firm3d.catapult.utils import (
    boozer_interpolant,
    boozer_saw_interpolant,
    cartesian_interpolant,
)
from firm3d.field.boozermagneticfield import ShearAlfvenWavesSuperposition


def trace_particles_boozer_gpu(
    field,
    stz_inits,
    parallel_speeds,
    tmax,
    mass,
    charge,
    vtotal,
    tol,
    ns,
    ntheta,
    nzeta,
    dt=None,
    mu=None,
    in_boozer=True,  # if in Boozer coordinates, else in pseudo-Cartesian coordinates
):
    """
    Trace particles in Boozer coordinates using CATAPULT

    Tracing runs in single or double precision according to the dtype of
    stz_inits (float32 or float64); parallel_speeds, dt, and mu must share
    that dtype.

    field: a magnetic field object representing the field in Boozer coordinates
    stz_inits: initial conditions for particles, shape (nparticles, 3), in
        (s, theta, zeta) coordinates if in_boozer is True, else in the
        pseudo-Cartesian coordinates (x1, x2, zeta) that CATAPULT integrates
        in, with x1 = s cos(theta) and x2 = s sin(theta)
    parallel_speeds: initial parallel speeds of the particles
    tmax: maximum time to trace particles, either a scalar applied to every
        particle or a per-particle array of shape (nparticles,)
    mass: mass of each particle
    charge: charge of each particle
    vtotal: total velocity of each particle
    tol: tolerance for the ODE solver
    dt: the initial time step size for the solver (optional; chosen from the
        maximum stable step size if not given)
    mu: the magnetic moment of each particle (optional; computed from the
        initial conditions if not given)
    in_boozer: if True, the initial conditions are in Boozer coordinates, else
        in pseudo-Cartesian coordinates; the result is returned in the same
        coordinates

    Returns:
        An array of shape (nparticles, 7) whose columns are
        (t, s, theta, zeta, vpar, dt, mu) if in_boozer is True, else
        (t, x1, x2, zeta, vpar, dt, mu). t is the time at which tracing
        stopped, so a particle is lost if t < tmax. zeta is returned wrapped
        to [0, 2 pi), unlike the CPU tracer, which returns it unwrapped.
        Columns 1-4 can be used as the initial conditions of a follow-on
        call, and the dt and mu columns fed back in through the dt and mu
        arguments, to continue tracing.
    """
    nparticles = stz_inits.shape[0]

    if in_boozer:
        stz_inits = stz_inits.copy()

        s = stz_inits[:, 0]
        theta = stz_inits[:, 1]
        x1 = s * np.cos(theta)
        x2 = s * np.sin(theta)
        stz_inits[:, 0] = x1
        stz_inits[:, 1] = x2

    # if only one tmax value is provided, use it for all particles
    if np.ndim(tmax) == 0:
        tmax = np.full(nparticles, tmax, dtype=np.float64)

    if isinstance(field, ShearAlfvenWavesSuperposition):
        B0 = field.B0
        srange, trange, zrange, quad_info, maxJ = boozer_saw_interpolant(
            B0, B0.nfp, ns, ntheta, nzeta, dtype=stz_inits.dtype
        )
        saw_nharmonics = len(field)
        saw_omega = field.get_wave(0).omega
        saw_s = field.get_wave(0).phihat.get_s_basis()
        saw_srange = (saw_s[0], saw_s[-1], len(saw_s))
        saw_m = [field.get_wave(i).Phim for i in range(saw_nharmonics)]
        saw_n = [field.get_wave(i).Phin for i in range(saw_nharmonics)]
        saw_phihats = np.ascontiguousarray(
            np.column_stack(
                [
                    np.array([field.get_wave(i).phihat(s_val) for s_val in saw_s])
                    for i in range(saw_nharmonics)
                ]
            )
        )

        if B0.field_type == "vac":
            last_time = firm3dpp.boozer_saw_gpu_tracing(
                quad_pts=quad_info,
                srange=srange,
                trange=trange,
                zrange=zrange,
                saw_omega=saw_omega,
                saw_srange=saw_srange,
                saw_m=saw_m,
                saw_n=saw_n,
                saw_phihats=saw_phihats,
                saw_nharmonics=saw_nharmonics,
                stz_init=stz_inits,
                m=mass,
                q=charge,
                vtotal=vtotal,
                vtang=parallel_speeds,
                tmax=tmax,
                tol=tol,
                dt_in=dt
                if dt is not None
                else -np.ones(nparticles).astype(stz_inits.dtype),
                mu_in=mu
                if mu is not None
                else -np.ones(nparticles).astype(stz_inits.dtype),
                psi0=B0.psi0,
                nparticles=nparticles,
            )
        elif B0.field_type == "nok":
            last_time = firm3dpp.boozer_saw_nok_gpu_tracing(
                quad_pts=quad_info,
                srange=srange,
                trange=trange,
                zrange=zrange,
                saw_omega=saw_omega,
                saw_srange=saw_srange,
                saw_m=saw_m,
                saw_n=saw_n,
                saw_phihats=saw_phihats,
                saw_nharmonics=saw_nharmonics,
                stz_init=stz_inits,
                m=mass,
                q=charge,
                vtotal=vtotal,
                vtang=parallel_speeds,
                tmax=tmax,
                tol=tol,
                dt_in=dt
                if dt is not None
                else -np.ones(nparticles).astype(stz_inits.dtype),
                mu_in=mu
                if mu is not None
                else -np.ones(nparticles).astype(stz_inits.dtype),
                psi0=B0.psi0,
                nparticles=nparticles,
            )
        else:
            raise ValueError(f"Unsupported field type {B0.field_type} for SAW tracing")
    else:
        if field.field_type not in ["vac", ""]:
            raise ValueError(
                f"Unsupported field type {field.field_type} for Boozer tracing, \
                     expected 'vac' or ''"
            )
        vacuum = field.field_type == "vac"  # true if vacuum, false if finite beta
        srange, trange, zrange, quad_info, maxJ = boozer_interpolant(
            field, field.nfp, ns, ntheta, nzeta, vacuum=vacuum
        )
        psi0 = field.psi0
        last_time = firm3dpp.boozer_gpu_tracing(
            quad_pts=quad_info.astype(stz_inits.dtype),
            srange=srange,
            trange=trange,
            zrange=zrange,
            stz_init=stz_inits.copy(),
            m=mass,
            q=charge,
            vtotal=vtotal,
            vtang=parallel_speeds.copy(),
            tmax=tmax,
            tol=tol,
            dt_in=dt
            if dt is not None
            else -np.ones(nparticles).astype(stz_inits.dtype),
            mu_in=mu
            if mu is not None
            else -np.ones(nparticles).astype(stz_inits.dtype),
            psi0=psi0,
            nparticles=nparticles,
            vacuum=vacuum,
        )

    last_time = np.reshape(last_time, (nparticles, 7))

    if in_boozer:
        x1 = last_time[:, 1]
        x2 = last_time[:, 2]
        s = np.sqrt(x1**2 + x2**2)
        theta = np.arctan2(x2, x1)
        last_time[:, 1] = s
        last_time[:, 2] = theta

    return last_time


def trace_particles_cartesian_gpu(
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
):
    """
    Trace particles in Cartesian coordinates using CATAPULT

    field: a magnetic field object representing the field in Cartesian coordinates
    surface_classifier: a simsopt surface classifier object for detecting a surface
    xyz_inits: initial conditions for particles, shape (nparticles, 3), in
        (x, y, z) coordinates; must be float64
    parallel_speeds: initial parallel speeds of the particles
    tmax: maximum time to trace particles, either a scalar applied to every
        particle or a per-particle array of shape (nparticles,)
    mass: mass of each particle
    charge: charge of each particle
    vtotal: total velocity of each particle
    tol: tolerance for the ODE solver
    dt: the initial time step size for the solver (optional; chosen from the
        maximum stable step size if not given)
    mu: the magnetic moment of each particle (optional; computed from the
        initial conditions if not given)

    Returns:
        An array of shape (nparticles, 7) whose columns are
        (t, x, y, z, vpar, dt, mu). t is the time at which tracing stopped,
        so a particle is lost if t < tmax. Columns 1-4 can be used as the
        initial conditions of a follow-on call, and the dt and mu columns fed
        back in through the dt and mu arguments, to continue tracing.
    """

    nparticles = xyz_inits.shape[0]

    # if only one tmax value is provided, use it for all particles
    if np.ndim(tmax) == 0:
        tmax = np.full(nparticles, tmax, dtype=np.float64)

    r_range, phi_range, z_range, quad_info = cartesian_interpolant(
        field, surface_classifier, dtype=xyz_inits.dtype
    )
    last_time = firm3dpp.cartesian_gpu_tracing(
        quad_pts=quad_info,
        rrange=r_range,
        phirange=phi_range,
        zrange=z_range,
        xyz_init=xyz_inits,
        m=mass,
        q=charge,
        vtotal=vtotal,
        vtang=parallel_speeds,
        tmax=tmax,
        tol=tol,
        dt_in=dt if dt is not None else -np.ones(nparticles).astype(xyz_inits.dtype),
        mu_in=mu if mu is not None else -np.ones(nparticles).astype(xyz_inits.dtype),
        nparticles=nparticles,
    )
    last_time = np.reshape(last_time, (nparticles, 7))
    return last_time


def trace_particles_boozer_gpu_trajectories(
    field,
    stz_inits,
    parallel_speeds,
    tmax,
    mass,
    charge,
    vtotal,
    tol,
    ns,
    ntheta,
    nzeta,
    dt_save,
):

    n_particles = stz_inits.shape[0]
    current_time = np.zeros(n_particles)
    trajectories = [[] for _ in range(n_particles)]
    dt = -np.ones(n_particles)
    mu = -np.ones(n_particles)

    # create interpolant data
    srange, trange, zrange, quad_info, maxJ = boozer_interpolant(
        field, field.nfp, ns, ntheta, nzeta, vacuum=True
    )
    psi0 = field.psi0

    # convert Boozer to pseudo-Cartesian coordinates
    s = stz_inits[:, 0]
    theta = stz_inits[:, 1]
    x1 = s * np.cos(theta)
    x2 = s * np.sin(theta)

    stz_inits[:, 0] = x1
    stz_inits[:, 1] = x2
    stz_inits = np.ascontiguousarray(stz_inits)

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

        step_data = firm3dpp.boozer_gpu_tracing(
            quad_pts=quad_info,
            srange=srange,
            trange=trange,
            zrange=zrange,
            stz_init=stz_inits.copy(),
            m=mass,
            q=charge,
            vtotal=vtotal,
            vtang=parallel_speeds.copy(),
            tmax=local_tmax,
            tol=tol,
            dt_in=dt,
            mu_in=mu,
            psi0=psi0,
            nparticles=n_particles,
            vacuum=True,
        )
        step_data = np.reshape(step_data, (n_particles, 7))

        dt = step_data[:, 5].copy()
        mu = step_data[:, 6].copy()

        # compute new current time for each particle
        step_data[:, 0] += current_time
        current_time = step_data[:, 0]

        # store data using stored indices
        for i, idx in enumerate(ids):
            if local_tmax[i] > 0.0:
                trajectories[idx].append(step_data[i, :])

        # find lost particles
        s_end = np.sqrt(step_data[:, 1] ** 2 + step_data[:, 2] ** 2)

        idx_keep = (current_time < tmax) & (s_end < 1.0)

        # remove lost particles
        stz_inits = step_data[idx_keep, 1:4].copy()
        parallel_speeds = step_data[idx_keep, 4].copy()
        ids = ids[idx_keep]
        current_time = current_time[idx_keep]
        dt = dt[idx_keep].copy()
        mu = mu[idx_keep].copy()

        n_particles = stz_inits.shape[0]

        if n_particles == 0:
            break

    return trajectories
