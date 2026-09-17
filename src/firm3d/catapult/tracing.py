__all__ = [
    "trace_particles_boozer_gpu",
    "trace_particles_cartesian_gpu",
    "save_trajectories_boozer_gpu",
    "save_trajectories_cartesian_gpu",
]

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


def _save_trajectories(trace_chunk, inits, parallel_speeds, tmax, dt_save):
    """
    Trace particles in chunks of dt_save, recording the state at the end of
    each chunk. This is the field-type-independent trajectory-saving loop
    shared by save_trajectories_boozer_gpu and save_trajectories_cartesian_gpu.

    trace_chunk(inits, parallel_speeds, tmax, dt, mu) must trace the given
    particles from t=0 for up to tmax (per particle) and return an
    (nparticles, 7) array (t, x1, x2, x3, vpar, dt, mu); the wrappers in this
    module do that once their interpolant is prebuilt.

    Between chunks the returned dt and mu are fed back in, so the chunked
    integration continues the same adaptive step sequence as a single
    uninterrupted trace. The kernel does not shorten a step to land exactly
    on a save time: it stops at the first step boundary at or after it. Each
    saved row is therefore the state at that boundary, its time up to one
    step past the multiple of dt_save it stands for, and a save interval
    that falls entirely inside one step produces no row. Lost particles are
    dropped from later chunks, with their original index remembered so the
    returned list lines up with the input.
    """
    nparticles = inits.shape[0]
    dtype = inits.dtype
    trajectories = [[] for _ in range(nparticles)]
    ids = np.arange(nparticles)
    current_time = np.zeros(nparticles)
    dt = np.full(nparticles, -1.0, dtype=dtype)
    mu = np.full(nparticles, -1.0, dtype=dtype)

    # ceil of the ratio, but tolerant of float rounding in an exact multiple
    # (1e-3 / 1e-6 evaluates to 1000.0000000000001, which must give 1000 chunks)
    nsteps = max(int(np.ceil((tmax / dt_save) * (1 - 1e-12))), 1)
    for step in range(nsteps):
        # each particle advances to the end of this chunk; the tracer starts
        # every call at t=0, so pass the remaining time for this chunk
        chunk_end = min((step + 1) * dt_save, tmax)
        local_tmax = np.maximum(chunk_end - current_time, 0.0)

        step_data = trace_chunk(inits, parallel_speeds, local_tmax, dt, mu)
        step_data[:, 0] += current_time
        current_time = step_data[:, 0]

        for i, idx in enumerate(ids):
            # a particle that overshot this save time in an earlier chunk was
            # traced for zero time and has nothing new to record
            if local_tmax[i] > 0.0:
                trajectories[idx].append(step_data[i, :])

        # a particle whose chunk ended early was lost
        keep = current_time >= 0.999 * chunk_end
        inits = np.ascontiguousarray(step_data[keep, 1:4], dtype=dtype)
        parallel_speeds = np.ascontiguousarray(step_data[keep, 4], dtype=dtype)
        dt = np.ascontiguousarray(step_data[keep, 5], dtype=dtype)
        mu = np.ascontiguousarray(step_data[keep, 6], dtype=dtype)
        ids = ids[keep]
        current_time = current_time[keep]
        if ids.size == 0:
            break

    return [np.array(traj) for traj in trajectories]


def save_trajectories_boozer_gpu(
    field,
    stz_inits,
    parallel_speeds,
    tmax,
    dt_save,
    mass,
    charge,
    vtotal,
    tol,
    ns,
    ntheta,
    nzeta,
):
    """
    Trace particles in Boozer coordinates using CATAPULT, saving the
    trajectory of each particle every dt_save.

    Arguments are as for trace_particles_boozer_gpu, plus dt_save, the
    interval at which to record the state. The interpolant is built once and
    reused for every chunk. Precision follows the dtype of stz_inits.

    Returns:
        A list with one entry per particle: an array of shape (nsaved, 7)
        whose rows are (t, s, theta, zeta, vpar, dt, mu), one for each
        multiple of dt_save the particle reached. The kernel does not shorten
        a step to land on a save time, so t is that of the first step
        boundary at or after the multiple of dt_save, up to one step late,
        and no row is written for a save time that a single step jumped
        over; choose dt_save above the step size (at most the quarter
        transit time (G/|B|) pi/2 / v) for a regular cadence. A lost particle
        has fewer rows. zeta is wrapped to [0, 2 pi).
    """
    if isinstance(field, ShearAlfvenWavesSuperposition):
        raise ValueError(
            "save_trajectories_boozer_gpu supports equilibrium fields only"
        )
    if field.field_type not in ["vac", ""]:
        raise ValueError(
            f"Unsupported field type {field.field_type} for Boozer tracing, "
            "expected 'vac' or ''"
        )
    dtype = stz_inits.dtype
    vacuum = field.field_type == "vac"
    srange, trange, zrange, quad_info, _ = boozer_interpolant(
        field, field.nfp, ns, ntheta, nzeta, vacuum=vacuum
    )
    quad_info = quad_info.astype(dtype)
    psi0 = field.psi0
    # the loop works in the pseudo-Cartesian coordinates CATAPULT integrates
    # in, so a chunk's output feeds the next chunk's input directly
    s = stz_inits[:, 0]
    theta = stz_inits[:, 1]
    inits = np.ascontiguousarray(
        np.column_stack((s * np.cos(theta), s * np.sin(theta), stz_inits[:, 2])),
        dtype=dtype,
    )
    parallel_speeds = np.ascontiguousarray(parallel_speeds, dtype=dtype)

    def trace_chunk(inits, parallel_speeds, local_tmax, dt, mu):
        n = inits.shape[0]
        out = firm3dpp.boozer_gpu_tracing(
            quad_pts=quad_info,
            srange=srange,
            trange=trange,
            zrange=zrange,
            stz_init=inits,
            m=mass,
            q=charge,
            vtotal=vtotal,
            vtang=parallel_speeds,
            tmax=np.asarray(local_tmax, dtype=np.float64),
            tol=tol,
            dt_in=dt,
            mu_in=mu,
            psi0=psi0,
            nparticles=n,
            vacuum=vacuum,
        )
        return np.asarray(out, dtype=dtype).reshape(n, 7)

    trajectories = _save_trajectories(
        trace_chunk, inits, parallel_speeds, tmax, dt_save
    )
    for traj in trajectories:
        x1 = traj[:, 1].copy()
        x2 = traj[:, 2].copy()
        traj[:, 1] = np.hypot(x1, x2)
        traj[:, 2] = np.arctan2(x2, x1)
    return trajectories


def save_trajectories_cartesian_gpu(
    field,
    surface_classifier,
    xyz_inits,
    parallel_speeds,
    tmax,
    dt_save,
    mass,
    charge,
    vtotal,
    tol,
):
    """
    Trace particles in Cartesian coordinates using CATAPULT, saving the
    trajectory of each particle every dt_save.

    Arguments are as for trace_particles_cartesian_gpu, plus dt_save, the
    interval at which to record the state. The interpolant is built once and
    reused for every chunk.

    Returns:
        A list with one entry per particle: an array of shape (nsaved, 7)
        whose rows are (t, x, y, z, vpar, dt, mu), one for each multiple of
        dt_save the particle reached. As for save_trajectories_boozer_gpu, t
        is that of the first step boundary at or after the multiple of
        dt_save (up to one step late, at most the quarter transit time
        r pi/2 / v), and a save time that a single step jumped over gets no
        row. A lost particle has fewer rows.
    """
    dtype = xyz_inits.dtype
    r_range, phi_range, z_range, quad_info = cartesian_interpolant(
        field, surface_classifier, dtype=dtype
    )
    inits = np.ascontiguousarray(xyz_inits, dtype=dtype)
    parallel_speeds = np.ascontiguousarray(parallel_speeds, dtype=dtype)

    def trace_chunk(inits, parallel_speeds, local_tmax, dt, mu):
        n = inits.shape[0]
        out = firm3dpp.cartesian_gpu_tracing(
            quad_pts=quad_info,
            rrange=r_range,
            phirange=phi_range,
            zrange=z_range,
            xyz_init=inits,
            m=mass,
            q=charge,
            vtotal=vtotal,
            vtang=parallel_speeds,
            tmax=np.asarray(local_tmax, dtype=np.float64),
            tol=tol,
            dt_in=dt,
            mu_in=mu,
            nparticles=n,
        )
        return np.asarray(out, dtype=dtype).reshape(n, 7)

    return _save_trajectories(trace_chunk, inits, parallel_speeds, tmax, dt_save)
