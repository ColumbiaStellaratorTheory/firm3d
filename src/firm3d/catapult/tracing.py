__all__ = [
    "trace_particles_boozer_gpu",
    "trace_particles_boozer_perturbed_gpu",
    "trace_particles_cartesian_gpu",
    "save_trajectories_boozer_gpu",
    "save_trajectories_cartesian_gpu",
]

import numpy as np

import firm3dpp
from firm3d.catapult.field import (
    CatapultBoozerField,
    CatapultCartesianField,
    CatapultPerturbedBoozerField,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)


def _check_per_particle(nparticles, **arrays):
    """
    Refuse a per-particle array of the wrong length, which the kernel would
    read past its end, or holding a NaN or infinity, which breaks its grid
    indexing and ends in an illegal memory access instead of an exception.
    """
    for name, value in arrays.items():
        if value.shape != (nparticles,):
            raise ValueError(
                f"{name} must have one entry per particle, shape ({nparticles},), "
                f"got {value.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains NaN or infinite values")


def _check_finite_scalar(name, value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value}")


def _launch_boozer(
    cfield, x_inits, parallel_speeds, tmax, dt, mu, mass, charge, vtotal, tol
):
    """
    One CATAPULT launch in a CatapultBoozerField or a
    CatapultPerturbedBoozerField, from pseudo-Cartesian initial conditions
    x_inits of shape (nparticles, 3). Every T-typed array is cast to the
    field's dtype, which the bindings require, and the arguments are checked
    before a binding is looked up, so that a malformed call fails the same way
    with or without the GPU bindings. Returns the (nparticles, 7) array
    (t, x1, x2, zeta, vpar, dt, mu) in that dtype.
    """
    dtype = cfield.dtype
    x_inits = np.ascontiguousarray(x_inits, dtype=dtype)
    if x_inits.ndim != 2 or x_inits.shape[1] != 3:
        raise ValueError(
            f"initial positions must have shape (nparticles, 3), got {x_inits.shape}"
        )
    if not np.all(np.isfinite(x_inits)):
        raise ValueError("initial positions contain NaN or infinite values")
    _check_finite_scalar("vtotal", vtotal)
    nparticles = x_inits.shape[0]
    parallel_speeds = np.ascontiguousarray(parallel_speeds, dtype=dtype)
    tmax = np.ascontiguousarray(tmax, dtype=np.float64)
    dt = np.ascontiguousarray(dt, dtype=dtype)
    mu = np.ascontiguousarray(mu, dtype=dtype)
    _check_per_particle(
        nparticles, parallel_speeds=parallel_speeds, tmax=tmax, dt=dt, mu=mu
    )
    kwargs = {
        "quad_pts": cfield.quad_info,
        "srange": cfield.srange,
        "trange": cfield.trange,
        "zrange": cfield.zrange,
        "stz_init": x_inits,
        "m": mass,
        "q": charge,
        "vtotal": vtotal,
        "vtang": parallel_speeds,
        "tmax": tmax,
        "tol": tol,
        "dt_in": dt,
        "mu_in": mu,
        "psi0": cfield.psi0,
        "nparticles": nparticles,
    }
    # the waves are extra arguments to their own kernels; an equilibrium field
    # instead tells the one kernel whether to assume a vacuum
    if isinstance(cfield, CatapultPerturbedBoozerField):
        kwargs.update(
            saw_omega=cfield.saw_omega,
            saw_srange=cfield.saw_srange,
            saw_m=cfield.saw_m,
            saw_n=cfield.saw_n,
            saw_phihats=cfield.saw_phihats,
            saw_nharmonics=cfield.saw_nharmonics,
        )
        trace = (
            firm3dpp.boozer_saw_gpu_tracing
            if cfield.vacuum
            else firm3dpp.boozer_saw_nok_gpu_tracing
        )
    else:
        kwargs["vacuum"] = cfield.vacuum
        trace = firm3dpp.boozer_gpu_tracing
    return np.asarray(trace(**kwargs), dtype=dtype).reshape(nparticles, 7)


def _launch_cartesian(
    cfield, xyz_inits, parallel_speeds, tmax, dt, mu, mass, charge, vtotal, tol
):
    """
    One CATAPULT launch in a CatapultCartesianField. Returns the
    (nparticles, 7) array (t, x, y, z, vpar, dt, mu) in the field's dtype.
    """
    dtype = cfield.dtype
    xyz_inits = np.ascontiguousarray(xyz_inits, dtype=dtype)
    if xyz_inits.ndim != 2 or xyz_inits.shape[1] != 3:
        raise ValueError(
            f"initial positions must have shape (nparticles, 3), got {xyz_inits.shape}"
        )
    if not np.all(np.isfinite(xyz_inits)):
        raise ValueError("initial positions contain NaN or infinite values")
    _check_finite_scalar("vtotal", vtotal)
    nparticles = xyz_inits.shape[0]
    parallel_speeds = np.ascontiguousarray(parallel_speeds, dtype=dtype)
    tmax = np.ascontiguousarray(tmax, dtype=np.float64)
    dt = np.ascontiguousarray(dt, dtype=dtype)
    mu = np.ascontiguousarray(mu, dtype=dtype)
    _check_per_particle(
        nparticles, parallel_speeds=parallel_speeds, tmax=tmax, dt=dt, mu=mu
    )
    out = firm3dpp.cartesian_gpu_tracing(
        quad_pts=cfield.quad_info,
        rrange=cfield.rrange,
        phirange=cfield.phirange,
        zrange=cfield.zrange,
        xyz_init=xyz_inits,
        m=mass,
        q=charge,
        vtotal=vtotal,
        vtang=parallel_speeds,
        tmax=tmax,
        tol=tol,
        dt_in=dt,
        mu_in=mu,
        nparticles=nparticles,
    )
    return np.asarray(out, dtype=dtype).reshape(nparticles, 7)


def _per_particle(value, nparticles, dtype, default):
    """A per-particle array from a scalar, an array, or None (the default)."""
    if value is None:
        return np.full(nparticles, default, dtype=dtype)
    if np.ndim(value) == 0:
        return np.full(nparticles, value, dtype=dtype)
    return np.ascontiguousarray(value, dtype=dtype)


def _to_pseudo_cartesian(stz_inits, dtype):
    """
    A copy of (s, theta, zeta) initial conditions as (s cos theta, s sin theta,
    zeta), the coordinates CATAPULT integrates in, in the given dtype.
    """
    x_inits = np.array(stz_inits, dtype=dtype, order="C")
    s = x_inits[:, 0].copy()
    theta = x_inits[:, 1].copy()
    x_inits[:, 0] = s * np.cos(theta)
    x_inits[:, 1] = s * np.sin(theta)
    return x_inits


def _to_boozer(result):
    """Turn columns 1 and 2 of a kernel result from (x1, x2) into (s, theta)."""
    x1 = result[:, 1].copy()
    x2 = result[:, 2].copy()
    result[:, 1] = np.hypot(x1, x2)
    result[:, 2] = np.arctan2(x2, x1)
    return result


def _save_trajectories(
    trace_chunk, inits, parallel_speeds, tmax, dt_save, dt=None, mu=None
):
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
    dt = _per_particle(dt, nparticles, dtype, -1.0)
    mu = _per_particle(mu, nparticles, dtype, -1.0)

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
    dt=None,
    mu=None,
):
    """
    Trace particles in Boozer coordinates using CATAPULT, saving the
    trajectory of each particle every dt_save.

    Arguments are as for trace_particles_boozer_gpu, plus dt_save, the
    interval at which to record the state. Equilibrium fields only: the
    kernel restarts time at each chunk, which a wave's phase cannot follow, so
    a CatapultPerturbedBoozerField is refused.

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
    if not isinstance(field, CatapultBoozerField):
        raise TypeError(
            f"field must be a CatapultBoozerField, got {type(field).__name__}; "
            "a field with waves is traced by trace_particles_boozer_perturbed_gpu"
        )
    dtype = field.dtype

    # the loop works in the pseudo-Cartesian coordinates CATAPULT integrates
    # in, so a chunk's output feeds the next chunk's input directly
    inits = _to_pseudo_cartesian(stz_inits, dtype)
    parallel_speeds = np.ascontiguousarray(parallel_speeds, dtype=dtype)

    def trace_chunk(inits, parallel_speeds, local_tmax, dt, mu):
        return _launch_boozer(
            field,
            inits,
            parallel_speeds,
            local_tmax,
            dt,
            mu,
            mass,
            charge,
            vtotal,
            tol,
        )

    trajectories = _save_trajectories(
        trace_chunk, inits, parallel_speeds, tmax, dt_save, dt, mu
    )
    return [_to_boozer(traj) for traj in trajectories]


def save_trajectories_cartesian_gpu(
    field,
    xyz_inits,
    parallel_speeds,
    tmax,
    dt_save,
    mass,
    charge,
    vtotal,
    tol,
    dt=None,
    mu=None,
):
    """
    Trace particles in Cartesian coordinates using CATAPULT, saving the
    trajectory of each particle every dt_save.

    Arguments are as for trace_particles_cartesian_gpu, plus dt_save, the
    interval at which to record the state.

    Returns:
        A list with one entry per particle: an array of shape (nsaved, 7)
        whose rows are (t, x, y, z, vpar, dt, mu), one for each multiple of
        dt_save the particle reached. As for save_trajectories_boozer_gpu, t
        is that of the first step boundary at or after the multiple of
        dt_save (up to one step late, at most the quarter transit time
        r pi/2 / v), and a save time that a single step jumped over gets no
        row. A lost particle has fewer rows.
    """
    if not isinstance(field, CatapultCartesianField):
        raise TypeError(
            f"field must be a CatapultCartesianField, got {type(field).__name__}"
        )
    dtype = field.dtype
    inits = np.ascontiguousarray(xyz_inits, dtype=dtype)
    parallel_speeds = np.ascontiguousarray(parallel_speeds, dtype=dtype)

    def trace_chunk(inits, parallel_speeds, local_tmax, dt, mu):
        return _launch_cartesian(
            field,
            inits,
            parallel_speeds,
            local_tmax,
            dt,
            mu,
            mass,
            charge,
            vtotal,
            tol,
        )

    return _save_trajectories(
        trace_chunk, inits, parallel_speeds, tmax, dt_save, dt, mu
    )


def _one_tmax(tmax):
    """The single tmax trajectory saving needs; per-particle values are refused."""
    if np.ptp(tmax) != 0:
        raise NotImplementedError(
            "trajectories are saved to one tmax for all particles; pass a scalar "
            "tmax, or forget_exact_path=True for per-particle values"
        )
    return float(tmax[0])


def _vtotal(Ekin, mass):
    """The speed for a kinetic energy, which CATAPULT takes once for all particles."""
    if np.ndim(Ekin) != 0:
        raise ValueError(
            "CATAPULT takes one kinetic energy for all particles; pass a scalar Ekin"
        )
    return float(np.sqrt(2 * Ekin / mass))


def _cpu_format(inits, parallel_speeds, bodies, tmax):
    """
    Assemble the CPU tracers' (res_tys, res_hits) from GPU results.

    inits, parallel_speeds: the initial conditions, for the t = 0 row.
    bodies: per particle, the rows (t, x1, x2, x3, vpar, ...) after t = 0:
        the saved trajectory, or just the final state.
    tmax: per particle, to decide who was lost.

    res_tys[i] has rows (t, x1, x2, x3, vpar): the initial state, then the
    rows of bodies[i]. res_hits[i] is a single row (t, -1, x1, x2, x3, vpar)
    at the final state of a lost particle, as the CPU tracer records a hit on
    its first stopping criterion, and an empty array otherwise. Everything is
    returned in float64.
    """
    nparticles = inits.shape[0]
    first = np.column_stack(
        (np.zeros(nparticles), np.asarray(inits, dtype=np.float64), parallel_speeds)
    )
    res_tys = []
    res_hits = []
    for i in range(nparticles):
        body = np.asarray(bodies[i], dtype=np.float64)[:, :5]
        res_tys.append(np.vstack((first[i], body)))
        # the kernel stops at t >= tmax in the field's precision, so allow
        # for tmax's own rounding to float32 before calling a particle lost
        if body[-1, 0] < tmax[i] * (1 - 1e-6):
            res_hits.append(np.array([[body[-1, 0], -1.0, *body[-1, 1:5]]]))
        else:
            res_hits.append(np.asarray([]))
    return res_tys, res_hits


def trace_particles_boozer_gpu(
    field,
    stz_inits,
    parallel_speeds,
    tmax=1e-4,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
    tol=1e-9,
    dt_save=1e-6,
    forget_exact_path=False,
    dt=None,
    mu=None,
):
    """
    Trace particles in an equilibrium field in Boozer coordinates using
    CATAPULT. The arguments and the result follow trace_particles_boozer,
    where CATAPULT has the same option.

    field: a CatapultBoozerField, holding the field tabulated at the
        resolution and precision to trace in
    stz_inits: initial positions, shape (nparticles, 3), in (s, theta, zeta)
    parallel_speeds: initial parallel speeds of the particles
    tmax: maximum time to trace particles, either a scalar applied to every
        particle or a per-particle array of shape (nparticles,)
    mass: mass of each particle
    charge: charge of each particle
    Ekin: kinetic energy in Joule, one value for all particles
    tol: tolerance for the ODE solver, used as both the absolute and the
        relative tolerance
    dt_save: interval at which to record the trajectory when
        forget_exact_path is False
    forget_exact_path: if True, keep only the initial and final state of each
        particle, in a single launch; if False, save the trajectory every
        dt_save (see save_trajectories_boozer_gpu for how the save times
        relate to the kernel's steps)
    dt: the initial time step size for the solver (optional; chosen from the
        maximum stable step size if not given)
    mu: the magnetic moment of each particle (optional; computed from the
        initial conditions if not given, which is what a particle starting
        from rest in the perpendicular direction requires)

    Returns: 2 element tuple containing
        - res_tys: a list with one (ntimesteps, 5) array per particle of rows
          (t, s, theta, zeta, vpar); the first row is the initial state at
          t = 0 and the last the state where tracing stopped. Unlike the CPU
          tracer, the last row of a lost particle is the state at or just
          past the s = 1 crossing rather than the last state inside it, a
          survivor's final time can exceed tmax by up to one step, theta is
          in (-pi, pi], and zeta is wrapped to [0, 2 pi).
        - res_hits: a list with one array per particle: a single row
          (t, -1, s, theta, zeta, vpar) at the final state of a lost
          particle, or an empty array. The kernel stops particles at s = 1
          and nowhere else, which is the CPU tracer's
          MaxToroidalFluxStoppingCriterion(1.0), so the row's index is -1 as
          for a hit on the CPU's first stopping criterion; there is no
          stopping_criteria argument.
    """
    if not isinstance(field, CatapultBoozerField):
        raise TypeError(
            f"field must be a CatapultBoozerField, got {type(field).__name__}; "
            "a field with waves is traced by trace_particles_boozer_perturbed_gpu"
        )
    dtype = field.dtype
    nparticles = stz_inits.shape[0]
    tmax = _per_particle(tmax, nparticles, np.float64, None)
    kwargs = {
        "mass": mass,
        "charge": charge,
        "vtotal": _vtotal(Ekin, mass),
        "tol": tol,
        "dt": dt,
        "mu": mu,
    }
    if forget_exact_path:
        # one launch, in the pseudo-Cartesian coordinates the kernel
        # integrates in, and back
        final = _launch_boozer(
            field,
            _to_pseudo_cartesian(stz_inits, dtype),
            parallel_speeds,
            tmax,
            _per_particle(dt, nparticles, dtype, -1.0),
            _per_particle(mu, nparticles, dtype, -1.0),
            mass,
            charge,
            kwargs["vtotal"],
            tol,
        )
        bodies = _to_boozer(final)[:, None, :]
    else:
        bodies = save_trajectories_boozer_gpu(
            field, stz_inits, parallel_speeds, _one_tmax(tmax), dt_save, **kwargs
        )
    return _cpu_format(stz_inits, parallel_speeds, bodies, tmax)


def trace_particles_boozer_perturbed_gpu(
    perturbed_field,
    stz_inits,
    parallel_speeds,
    mus,
    tmax=1e-4,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    Ekin=None,
    tol=1e-9,
    forget_exact_path=True,
    dt=None,
):
    """
    Trace particles in a field with shear Alfven waves in Boozer coordinates
    using CATAPULT, returning what trace_particles_boozer_perturbed returns.
    The arguments follow it: the waves do work on the particles, so the
    magnetic moment, which they conserve, is given per particle and the
    energy is not.

    perturbed_field: a CatapultPerturbedBoozerField, holding the equilibrium
        tabulated at the resolution to trace in, and the waves
    stz_inits: initial positions, shape (nparticles, 3), in (s, theta, zeta)
    parallel_speeds: initial parallel speeds of the particles
    mus: magnetic moment of each particle, shape (nparticles,)
    tmax: maximum time to trace particles, either a scalar applied to every
        particle or a per-particle array of shape (nparticles,)
    mass: mass of each particle
    charge: charge of each particle
    Ekin: kinetic energy in Joule setting the speed the kernel scales its
        maximum step and tolerances by; if None, the initial energy of the
        first particle is used, as in trace_particles_boozer_perturbed
    tol: tolerance for the ODE solver
    forget_exact_path: must be True. The kernel starts every launch at t = 0
        of the waves' phase, so trajectories cannot yet be saved in chunks
        as they are for equilibrium fields; the default differs from the CPU
        tracer's for that reason.

    Returns:
        (res_tys, res_hits) as for trace_particles_boozer_gpu, each res_tys
        entry holding the initial and final state.
    """
    if not forget_exact_path:
        raise NotImplementedError(
            "trajectories in a perturbed field cannot be saved in chunks, since "
            "the kernel restarts the waves' phase at each launch; pass "
            "forget_exact_path=True"
        )
    if not isinstance(perturbed_field, CatapultPerturbedBoozerField):
        raise TypeError(
            "perturbed_field must be a CatapultPerturbedBoozerField, got "
            f"{type(perturbed_field).__name__}; an equilibrium field is traced "
            "by trace_particles_boozer_gpu"
        )
    dtype = perturbed_field.dtype
    nparticles = stz_inits.shape[0]
    tmax = _per_particle(tmax, nparticles, np.float64, None)
    mus = np.ascontiguousarray(mus, dtype=dtype)
    if mus.shape != (nparticles,):
        raise ValueError(f"mus must have shape ({nparticles},), got {mus.shape}")

    if Ekin is None:
        # the speed the kernel normalizes by, from the first particle's
        # energy at its birth point in the equilibrium
        stz0 = np.asarray(stz_inits[:1], dtype=np.float64)
        perturbed_field.B0.set_points(stz0)
        modB = perturbed_field.B0.modB()[0, 0]
        v2 = parallel_speeds[0] ** 2 + 2 * mus[0] * modB
        if not np.isfinite(v2) or v2 <= 0:
            raise ValueError(
                "could not derive a speed from the first particle's vpar, mu and "
                f"|B| = {modB}; pass Ekin"
            )
        vtotal = float(np.sqrt(v2))
    else:
        vtotal = float(np.sqrt(2 * Ekin / mass))

    final = _launch_boozer(
        perturbed_field,
        _to_pseudo_cartesian(stz_inits, dtype),
        parallel_speeds,
        tmax,
        _per_particle(dt, nparticles, dtype, -1.0),
        mus,
        mass,
        charge,
        vtotal,
        tol,
    )
    return _cpu_format(stz_inits, parallel_speeds, _to_boozer(final)[:, None, :], tmax)


def trace_particles_cartesian_gpu(
    field,
    xyz_inits,
    parallel_speeds,
    tmax=1e-4,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
    tol=1e-9,
    dt_save=1e-6,
    forget_exact_path=False,
    dt=None,
    mu=None,
):
    """
    Trace particles in Cartesian coordinates using CATAPULT. The arguments
    and the result follow simsopt's trace_particles, where CATAPULT has the
    same option.

    field: a CatapultCartesianField, holding the field and the surface
        classifier tabulated at the precision to trace in; particles are
        stopped when they cross that surface
    xyz_inits: initial positions, shape (nparticles, 3), in (x, y, z)
    parallel_speeds: initial parallel speeds of the particles
    tmax: maximum time to trace particles, either a scalar applied to every
        particle or a per-particle array of shape (nparticles,)
    mass: mass of each particle
    charge: charge of each particle
    Ekin: kinetic energy in Joule, one value for all particles
    tol: tolerance for the ODE solver, used as both the absolute and the
        relative tolerance
    dt_save, forget_exact_path, dt, mu: as for trace_particles_boozer_gpu

    Returns: 2 element tuple containing
        - res_tys: a list with one (ntimesteps, 5) array per particle of rows
          (t, x, y, z, vpar), from the initial state at t = 0 to the state
          where tracing stopped
        - res_hits: a list with one array per particle: a single row
          (t, -1, x, y, z, vpar) at the final state of a particle that
          crossed the surface, or an empty array
    """
    if not isinstance(field, CatapultCartesianField):
        raise TypeError(
            f"field must be a CatapultCartesianField, got {type(field).__name__}"
        )
    dtype = field.dtype
    nparticles = xyz_inits.shape[0]
    tmax = _per_particle(tmax, nparticles, np.float64, None)
    kwargs = {
        "mass": mass,
        "charge": charge,
        "vtotal": _vtotal(Ekin, mass),
        "tol": tol,
        "dt": dt,
        "mu": mu,
    }
    if forget_exact_path:
        bodies = _launch_cartesian(
            field,
            xyz_inits,
            parallel_speeds,
            tmax,
            _per_particle(dt, nparticles, dtype, -1.0),
            _per_particle(mu, nparticles, dtype, -1.0),
            mass,
            charge,
            kwargs["vtotal"],
            tol,
        )[:, None, :]
    else:
        bodies = save_trajectories_cartesian_gpu(
            field, xyz_inits, parallel_speeds, _one_tmax(tmax), dt_save, **kwargs
        )
    return _cpu_format(xyz_inits, parallel_speeds, bodies, tmax)
