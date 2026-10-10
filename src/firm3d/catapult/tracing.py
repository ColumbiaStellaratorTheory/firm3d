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


def _event_options(
    inits,
    *,
    zetas=None,
    phases=None,
    n_zetas=None,
    m_thetas=None,
    omegas=None,
    vpars=None,
    stopping_criteria=None,
    phases_stop=False,
    vpars_stop=False,
    max_hits=1024,
    max_phase_hits=0,
    max_phase_interval=None,
    boozer=True,
):
    """Validate CPU-style event requests before launching any GPU work."""
    if (
        all(
            value is None
            for value in (
                zetas,
                phases,
                n_zetas,
                m_thetas,
                omegas,
                vpars,
                stopping_criteria,
            )
        )
        and not phases_stop
        and not vpars_stop
        and not max_phase_hits
        and max_phase_interval is None
    ):
        return None
    if zetas is not None:
        if any(x is not None for x in (phases, n_zetas, m_thetas, omegas)):
            raise ValueError("zetas cannot be combined with phase-plane arguments")
        phases = zetas

    def array(name, values):
        result = np.asarray([] if values is None else values, dtype=np.float64)
        if result.ndim != 1 or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite one-dimensional array")
        return result

    phases = array("phases", phases)
    vpars = array("vpars", vpars)
    criteria = [] if stopping_criteria is None else list(stopping_criteria)
    count = len(phases)
    modes = [
        np.full(count, default) if values is None else array(name, values)
        for name, values, default in (
            ("n_zetas", n_zetas, 1),
            ("m_thetas", m_thetas, 0),
            ("omegas", omegas, 0),
        )
    ]
    if any(len(mode) != count for mode in modes):
        raise ValueError(
            "phases, n_zetas, m_thetas, and omegas must have equal lengths"
        )
    if phases_stop and not count:
        raise ValueError("phases_stop requires phase planes")
    if vpars_stop and not len(vpars):
        raise ValueError("vpars_stop requires vpars")
    for name, value, minimum in (
        ("max_hits", max_hits, 1),
        ("max_phase_hits", max_phase_hits, 0),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
            or value > np.iinfo(np.int32).max
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if max_phase_hits and not count:
        raise ValueError("max_phase_hits requires phase planes")
    if max_phase_interval is not None:
        if np.ndim(max_phase_interval) != 0:
            raise ValueError("max_phase_interval must be a scalar")
        _check_finite_scalar("max_phase_interval", max_phase_interval)
        if not count:
            raise ValueError("max_phase_interval requires phase planes")
    if not boozer and count:
        raise ValueError("phase planes require Boozer coordinates")
    if not (count or len(vpars) or criteria):
        return None
    inits = np.asarray(inits, dtype=np.float64)
    return {
        "planes": np.column_stack((phases, *modes)).ravel(),
        "vpars": vpars,
        "stopping_criteria": criteria,
        "phases_stop": phases_stop,
        "vpars_stop": vpars_stop,
        "max_hits": max_hits if count or len(vpars) else 1,
        "max_phase_hits": max_phase_hits,
        "max_phase_interval": 0 if max_phase_interval is None else max_phase_interval,
        "theta_offsets": (
            inits[:, 1] - np.arctan2(np.sin(inits[:, 1]), np.cos(inits[:, 1]))
            if boozer
            else np.zeros(len(inits))
        ),
    }


def _unpack_events(output, nparticles, dtype, save_times, options):
    """Keep event times in double precision, including the terminal path row."""
    samples, hits, end_times = output
    bodies = _unpack_output(samples, nparticles, dtype, save_times)
    if isinstance(bodies, np.ndarray):
        bodies = bodies.astype(np.float64)
        bodies[:, 0] = end_times
    else:
        bodies = [body.astype(np.float64) for body in bodies]
        for body, end in zip(bodies, end_times):
            body[:-1, 0] = save_times[: len(body) - 1]
            body[-1, 0] = end
    rows = np.asarray(hits, dtype=np.float64).reshape(
        nparticles, options["max_hits"], 6
    )
    events = [particle[np.isfinite(particle[:, 0])] for particle in rows]
    nplanes = len(options["planes"]) // 4
    for body, particle, end in zip(bodies, events, end_times):
        if len(particle) and particle[-1, 0] == end and particle[-1, 1] >= nplanes:
            # A stopping velocity level is represented exactly in public rows.
            terminal = body if body.ndim == 1 else body[-1]
            terminal[4] = particle[-1, 5]
    return bodies, events


def _launch_boozer(
    cfield,
    x_inits,
    parallel_speeds,
    tmax,
    dt,
    mu,
    mass,
    charge,
    vtotal,
    tol,
    save_times=None,
    event_options=None,
):
    """
    One uninterrupted CATAPULT trace in a CatapultBoozerField or a
    CatapultPerturbedBoozerField, from pseudo-Cartesian initial conditions
    x_inits of shape (nparticles, 3). Every T-typed array is cast to the
    field's dtype, which the bindings require, and the arguments are checked
    before a binding is looked up, so that a malformed call fails the same way
    with or without the GPU bindings. Returns the (nparticles, 7) array
    (t, x1, x2, zeta, vpar, dt, mu) in that dtype. When save_times is given,
    return a list of trajectory arrays with the unused buffer rows removed.
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
    if np.any(tmax < 0):
        raise ValueError("tmax must be nonnegative")
    if nparticles == 0:
        empty = [] if save_times is not None else np.empty((0, 7), dtype=dtype)
        return (empty, []) if event_options is not None else empty
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
    if save_times is not None:
        kwargs["save_times"] = save_times
    if event_options is not None:
        kwargs["event_options"] = event_options
        return _unpack_events(
            trace(**kwargs), nparticles, dtype, save_times, event_options
        )
    return _unpack_output(trace(**kwargs), nparticles, dtype, save_times)


def _launch_cartesian(
    cfield,
    xyz_inits,
    parallel_speeds,
    tmax,
    dt,
    mu,
    mass,
    charge,
    vtotal,
    tol,
    save_times=None,
    event_options=None,
):
    """
    One uninterrupted CATAPULT trace in a CatapultCartesianField. Returns the
    (nparticles, 7) array (t, x, y, z, vpar, dt, mu) in the field's dtype,
    or a list of trajectory arrays when save_times is given.
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
    if np.any(tmax < 0):
        raise ValueError("tmax must be nonnegative")
    if nparticles == 0:
        empty = [] if save_times is not None else np.empty((0, 7), dtype=dtype)
        return (empty, []) if event_options is not None else empty
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
        **({"save_times": save_times} if save_times is not None else {}),
        **({"event_options": event_options} if event_options is not None else {}),
    )
    if event_options is not None:
        return _unpack_events(out, nparticles, dtype, save_times, event_options)
    return _unpack_output(out, nparticles, dtype, save_times)


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
    if x_inits.ndim != 2 or x_inits.shape[1] != 3:
        raise ValueError("initial positions must have shape (nparticles, 3)")
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
    result[:, 3] %= 2 * np.pi
    return result


def _unpack_output(output, nparticles, dtype, save_times):
    """Remove the unused NaN rows from each particle's GPU output buffer."""
    if save_times is None:
        return np.asarray(output, dtype=dtype).reshape(nparticles, 7)
    rows = np.asarray(output, dtype=dtype).reshape(nparticles, len(save_times) + 1, 7)
    return [particle[np.isfinite(particle[:, 0])] for particle in rows]


def _save_times(tmax, dt_save):
    """The common save grid; the kernel also interpolates each particle's tmax."""
    _check_finite_scalar("dt_save", dt_save)
    tmax = np.asarray(tmax, dtype=np.float64)
    if not np.all(np.isfinite(tmax)) or np.any(tmax < 0):
        raise ValueError("tmax must be finite and nonnegative")
    end = float(np.max(tmax, initial=0.0))
    ratio = end / dt_save
    if not np.isfinite(ratio):
        raise ValueError("tmax / dt_save is too large")
    # Include the upper candidate in case division rounds the ratio down to
    # an integer. Keep every represented grid time strictly before the endpoint.
    times = np.arange(1, np.ceil(ratio) + 1, dtype=np.float64) * dt_save
    return times[times < end]


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
    Trace in one integration launch, sampling the Dormand-Prince dense output
    every dt_save and at each particle's tmax. tmax may be scalar or per particle.

    field is a CatapultBoozerField or CatapultPerturbedBoozerField. Waves
    require explicit mu; their absolute phase is preserved throughout the
    launch. dt and mu optionally specify the initial step and magnetic moment.

    Returns a list of (nsaved, 7) arrays with rows
    (t, s, theta, zeta, vpar, dt, mu), excluding the initial state at t=0.
    Every requested time is sampled, including multiple times inside one
    accepted step. dt is the enclosing accepted step's size; an interpolated
    row is not an adaptive integrator checkpoint. Lost particles end at the
    kernel's first accepted endpoint beyond the boundary. zeta is wrapped
    to [0, 2 pi). Sampling does not change the adaptive step sequence.
    """
    if not isinstance(field, (CatapultBoozerField, CatapultPerturbedBoozerField)):
        raise TypeError(
            "field must be a CatapultBoozerField or CatapultPerturbedBoozerField, "
            f"got {type(field).__name__}"
        )
    if isinstance(field, CatapultPerturbedBoozerField) and mu is None:
        raise ValueError("a perturbed field requires explicit magnetic moments (mu)")
    dtype = field.dtype
    nparticles = stz_inits.shape[0]
    tmax = _per_particle(tmax, nparticles, np.float64, None)
    trajectories = _launch_boozer(
        field,
        _to_pseudo_cartesian(stz_inits, dtype),
        parallel_speeds,
        tmax,
        _per_particle(dt, nparticles, dtype, -1.0),
        _per_particle(mu, nparticles, dtype, -1.0),
        mass,
        charge,
        vtotal,
        tol,
        save_times=_save_times(tmax, dt_save),
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
    Trace in one integration launch, sampling the Dormand-Prince dense output
    every dt_save and at each particle's tmax. tmax may be scalar or per particle.

    Returns a list of (nsaved, 7) arrays with rows (t, x, y, z, vpar, dt, mu),
    excluding the initial state at t=0. Every requested time is sampled;
    dt is the enclosing accepted step's size. Lost particles end at the first
    accepted endpoint beyond the classifier's surface. The adaptive step
    sequence is independent of dt_save.
    """
    if not isinstance(field, CatapultCartesianField):
        raise TypeError(
            f"field must be a CatapultCartesianField, got {type(field).__name__}"
        )
    dtype = field.dtype
    nparticles = xyz_inits.shape[0]
    tmax = _per_particle(tmax, nparticles, np.float64, None)
    return _launch_cartesian(
        field,
        xyz_inits,
        parallel_speeds,
        tmax,
        _per_particle(dt, nparticles, dtype, -1.0),
        _per_particle(mu, nparticles, dtype, -1.0),
        mass,
        charge,
        vtotal,
        tol,
        save_times=_save_times(tmax, dt_save),
    )


def _vtotal(Ekin, mass):
    """The speed for a kinetic energy, which CATAPULT takes once for all particles."""
    if np.ndim(Ekin) != 0:
        raise ValueError(
            "CATAPULT takes one kinetic energy for all particles; pass a scalar Ekin"
        )
    return float(np.sqrt(2 * Ekin / mass))


def _cpu_format(inits, parallel_speeds, bodies, tmax, infer_losses=True):
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
    if isinstance(bodies, np.ndarray):
        # Endpoint-only tracing returns a dense (nparticles, 1, 7) array.
        # Assemble all two-row paths together instead of allocating and
        # filtering an array for each particle. The views have disjoint rows.
        paths = np.empty((nparticles, 2, 5), dtype=np.float64)
        paths[:, 0] = first
        paths[:, 1] = bodies[:, 0, :5]
        res_tys = [path if path[1, 0] > 0 else path[:1] for path in paths]
        lost = (
            bodies[:, 0, 0] < tmax * (1 - 1e-6)
            if infer_losses
            else np.zeros(nparticles, dtype=bool)
        )
        lost_indices = np.flatnonzero(lost)
        hit_rows = np.column_stack(
            (paths[lost, 1, 0], -np.ones(len(lost_indices)), paths[lost, 1, 1:])
        )
        res_hits = [np.empty(0) for _ in range(nparticles)]
        for i, row in zip(lost_indices, hit_rows):
            res_hits[i] = row[None, :]
        return res_tys, res_hits
    res_tys = []
    res_hits = []
    for i in range(nparticles):
        body = np.asarray(bodies[i], dtype=np.float64)[:, :5]
        res_tys.append(np.vstack((first[i], body[body[:, 0] > 0])))
        # the kernel stops at t >= tmax in the field's precision, so allow
        # for tmax's own rounding to float32 before calling a particle lost
        if infer_losses and body[-1, 0] < tmax[i] * (1 - 1e-6):
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
    *,
    zetas=None,
    phases=None,
    n_zetas=None,
    m_thetas=None,
    omegas=None,
    vpars=None,
    stopping_criteria=None,
    phases_stop=False,
    vpars_stop=False,
    max_hits=1024,
    max_phase_hits=0,
    max_phase_interval=None,
    dt=None,
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
        particle, in one uninterrupted trace; if False, save the trajectory
        every dt_save using dense output within accepted steps

    zetas: section angles modulo 2*pi; shorthand for phases with n_zetas=1,
        m_thetas=0 and omegas=0. Alternatively, phases, n_zetas, m_thetas and
        omegas specify n*zeta + m*theta - omega*t = phase modulo 2*pi.
    vpars: parallel-velocity levels to record, including zero for mirror hits.
    phases_stop, vpars_stop: stop at the earliest requested crossing.
    stopping_criteria: CPU Max/MinToroidalFlux, Iteration, ToroidalTransit or
        StepSize criteria, checked at accepted endpoints. The field boundary
        remains enforced. Custom CPU callbacks cannot run on the GPU.
    max_hits: per-particle event capacity (default 1024); overflow raises.
    max_phase_hits: stop after this many phase hits; zero disables the limit.
    max_phase_interval: optional maximum time from launch or the last phase
        hit to the next hit. Stop at this deadline without recording a hit.
    dt: optional initial step, scalar or per particle.

    Returns (res_tys, res_hits). Paths have rows (t, s, theta, zeta, vpar),
    including the launch and terminal state. Sampled path angles are wrapped.
    Hits have CPU-format rows (t, index, s, theta, zeta, vpar): phase indices
    precede velocity indices; stopping criteria use -1-i. Event theta retains
    winding and zeta is wrapped to [0, 2*pi). Dense event roots are independent
    of dt_save and work with forget_exact_path=True. A launch on a plane is
    excluded; a crossing at the right endpoint is included once. Without
    explicit criteria, the enforced field-boundary hit has index -1; otherwise
    it follows the requested criteria with index -1-len(stopping_criteria).
    All public rows use float64; field precision controls state accuracy.
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
    }
    events = _event_options(
        stz_inits,
        zetas=zetas,
        phases=phases,
        n_zetas=n_zetas,
        m_thetas=m_thetas,
        omegas=omegas,
        vpars=vpars,
        stopping_criteria=stopping_criteria,
        phases_stop=phases_stop,
        vpars_stop=vpars_stop,
        max_hits=max_hits,
        max_phase_hits=max_phase_hits,
        max_phase_interval=max_phase_interval,
    )
    if events is not None:
        output, hits = _launch_boozer(
            field,
            _to_pseudo_cartesian(stz_inits, dtype),
            parallel_speeds,
            tmax,
            _per_particle(dt, nparticles, dtype, -1.0),
            _per_particle(None, nparticles, dtype, -1.0),
            mass,
            charge,
            kwargs["vtotal"],
            tol,
            save_times=None if forget_exact_path else _save_times(tmax, dt_save),
            event_options=events,
        )
        bodies = (
            _to_boozer(output)[:, None, :]
            if forget_exact_path
            else [_to_boozer(traj) for traj in output]
        )
        paths, _ = _cpu_format(
            stz_inits, parallel_speeds, bodies, tmax, infer_losses=False
        )
        return paths, hits
    if forget_exact_path:
        # one uninterrupted trace, in the pseudo-Cartesian coordinates the kernel
        # integrates in, and back. The step and the magnetic moment are the
        # kernel's to choose: mu follows from Ekin and the parallel speed,
        # as it does for trace_particles_boozer.
        final = _launch_boozer(
            field,
            _to_pseudo_cartesian(stz_inits, dtype),
            parallel_speeds,
            tmax,
            _per_particle(dt, nparticles, dtype, -1.0),
            _per_particle(None, nparticles, dtype, -1.0),
            mass,
            charge,
            kwargs["vtotal"],
            tol,
        )
        bodies = _to_boozer(final)[:, None, :]
    else:
        bodies = save_trajectories_boozer_gpu(
            field, stz_inits, parallel_speeds, tmax, dt_save, dt=dt, **kwargs
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
    dt_save=1e-6,
    *,
    zetas=None,
    phases=None,
    n_zetas=None,
    m_thetas=None,
    omegas=None,
    vpars=None,
    stopping_criteria=None,
    phases_stop=False,
    vpars_stop=False,
    max_hits=1024,
    max_phase_hits=0,
    max_phase_interval=None,
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
    forget_exact_path: if True (the default), save only the initial and
        final state; if False, sample dense output every dt_save. Both paths
        use one uninterrupted integration and preserve the waves' absolute phase.
    dt_save: trajectory save interval when forget_exact_path is False.

    Returns:
        (res_tys, res_hits) as for trace_particles_boozer_gpu, each res_tys
        entry holding the initial and final state and, when requested,
        the dense-output samples between them.
    """
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

    if nparticles == 0:
        return [], []

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

    events = _event_options(
        stz_inits,
        zetas=zetas,
        phases=phases,
        n_zetas=n_zetas,
        m_thetas=m_thetas,
        omegas=omegas,
        vpars=vpars,
        stopping_criteria=stopping_criteria,
        phases_stop=phases_stop,
        vpars_stop=vpars_stop,
        max_hits=max_hits,
        max_phase_hits=max_phase_hits,
        max_phase_interval=max_phase_interval,
    )
    output = _launch_boozer(
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
        save_times=None if forget_exact_path else _save_times(tmax, dt_save),
        event_options=events,
    )
    if events is not None:
        output, hits = output
    if forget_exact_path:
        bodies = _to_boozer(output)[:, None, :]
    else:
        bodies = [_to_boozer(traj) for traj in output]
    paths, losses = _cpu_format(
        stz_inits, parallel_speeds, bodies, tmax, infer_losses=events is None
    )
    return paths, hits if events is not None else losses


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
    *,
    vpars=None,
    stopping_criteria=None,
    vpars_stop=False,
    max_hits=1024,
    dt=None,
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
    dt_save, forget_exact_path: as for trace_particles_boozer_gpu

    vpars, vpars_stop, max_hits and dt have the same meaning as in
    trace_particles_boozer_gpu. Cartesian tracing supports Iteration and
    StepSize stopping criteria; Boozer flux/transit criteria require a Boozer
    field. The surface classifier remains enforced.

    Returns (res_tys, res_hits) in CPU format, with velocity hit indices
    starting at zero and criterion indices -1-i.
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
    }
    events = _event_options(
        xyz_inits,
        vpars=vpars,
        stopping_criteria=stopping_criteria,
        vpars_stop=vpars_stop,
        max_hits=max_hits,
        boozer=False,
    )
    if events is not None:
        output, hits = _launch_cartesian(
            field,
            xyz_inits,
            parallel_speeds,
            tmax,
            _per_particle(dt, nparticles, dtype, -1.0),
            _per_particle(None, nparticles, dtype, -1.0),
            mass,
            charge,
            kwargs["vtotal"],
            tol,
            save_times=None if forget_exact_path else _save_times(tmax, dt_save),
            event_options=events,
        )
        bodies = output[:, None, :] if forget_exact_path else output
        paths, _ = _cpu_format(
            xyz_inits, parallel_speeds, bodies, tmax, infer_losses=False
        )
        return paths, hits
    if forget_exact_path:
        bodies = _launch_cartesian(
            field,
            xyz_inits,
            parallel_speeds,
            tmax,
            _per_particle(dt, nparticles, dtype, -1.0),
            _per_particle(None, nparticles, dtype, -1.0),
            mass,
            charge,
            kwargs["vtotal"],
            tol,
        )[:, None, :]
    else:
        bodies = save_trajectories_cartesian_gpu(
            field, xyz_inits, parallel_speeds, tmax, dt_save, dt=dt, **kwargs
        )
    return _cpu_format(xyz_inits, parallel_speeds, bodies, tmax)
