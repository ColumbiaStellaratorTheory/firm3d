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
        return [] if save_times is not None else np.empty((0, 7), dtype=dtype)
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
        return [] if save_times is not None else np.empty((0, 7), dtype=dtype)
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
    )
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
    if isinstance(bodies, np.ndarray):
        # Endpoint-only tracing returns a dense (nparticles, 1, 7) array.
        # Assemble all two-row paths together instead of allocating and
        # filtering an array for each particle. The views have disjoint rows.
        paths = np.empty((nparticles, 2, 5), dtype=np.float64)
        paths[:, 0] = first
        paths[:, 1] = bodies[:, 0, :5]
        res_tys = [path if path[1, 0] > 0 else path[:1] for path in paths]
        lost = bodies[:, 0, 0] < tmax * (1 - 1e-6)
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

    Returns: 2 element tuple containing
        - res_tys: a list with one (ntimesteps, 5) array per particle of rows
          (t, s, theta, zeta, vpar); the first row is the initial state at
          t = 0 and the last the state where tracing stopped. Unlike the CPU
          tracer, the last row of a lost particle is the state at or just
          past the s = 1 crossing rather than the last state inside it, a
          survivor's final state is interpolated at tmax, theta is
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
    }
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
            _per_particle(None, nparticles, dtype, -1.0),
            _per_particle(None, nparticles, dtype, -1.0),
            mass,
            charge,
            kwargs["vtotal"],
            tol,
        )
        bodies = _to_boozer(final)[:, None, :]
    else:
        bodies = save_trajectories_boozer_gpu(
            field, stz_inits, parallel_speeds, tmax, dt_save, **kwargs
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

    output = _launch_boozer(
        perturbed_field,
        _to_pseudo_cartesian(stz_inits, dtype),
        parallel_speeds,
        tmax,
        _per_particle(None, nparticles, dtype, -1.0),
        mus,
        mass,
        charge,
        vtotal,
        tol,
        save_times=None if forget_exact_path else _save_times(tmax, dt_save),
    )
    if forget_exact_path:
        bodies = _to_boozer(output)[:, None, :]
    else:
        bodies = [_to_boozer(traj) for traj in output]
    return _cpu_format(stz_inits, parallel_speeds, bodies, tmax)


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
    }
    if forget_exact_path:
        bodies = _launch_cartesian(
            field,
            xyz_inits,
            parallel_speeds,
            tmax,
            _per_particle(None, nparticles, dtype, -1.0),
            _per_particle(None, nparticles, dtype, -1.0),
            mass,
            charge,
            kwargs["vtotal"],
            tol,
        )[:, None, :]
    else:
        bodies = save_trajectories_cartesian_gpu(
            field, xyz_inits, parallel_speeds, tmax, dt_save, **kwargs
        )
    return _cpu_format(xyz_inits, parallel_speeds, bodies, tmax)
