__all__ = [
    "trace_particles_boozer_gpu",
    "trace_particles_boozer_with_collisions_gpu",
    "trace_particles_cartesian_gpu",
    "trace_particles_cartesian_with_collisions_gpu",
]
import numpy as np

import firm3dpp
from firm3d.catapult.utils import (
    boozer_interpolant,
    boozer_saw_interpolant,
    cartesian_interpolant,
)
from firm3d.field.boozermagneticfield import ShearAlfvenWavesSuperposition
from firm3d.field.tracing_helpers import _validate_parallel_speeds


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
):
    """
    Trace particles in Boozer coordinates using CATAPULT
    field: a magnetic field object representing the field in Boozer coordinates
    stz_inits: initial conditions for particles in (s, theta, zeta) coordinates
    parallel_speeds: initial parallel speeds of the particles
    tmax: maximum time to trace particles
    mass: mass of each particle
    charge: charge of each particle
    vtotal: total velocity of each particle
    tol: tolerance for the ODE solver
    dt: the initial time step size for the solver (optional)
    """
    stz_inits = np.ascontiguousarray(stz_inits, dtype=float).copy()
    parallel_speeds = np.ascontiguousarray(parallel_speeds, dtype=float)
    if dt is not None:
        dt = np.ascontiguousarray(dt, dtype=float)
    nparticles = stz_inits.shape[0]
    _validate_parallel_speeds(parallel_speeds, vtotal)

    if isinstance(field, ShearAlfvenWavesSuperposition):
        B0 = field.B0
        srange, trange, zrange, quad_info, maxJ = boozer_saw_interpolant(
            B0, B0.nfp, ns, ntheta, nzeta
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
                dt_in=dt if dt is not None else -np.ones(nparticles),
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
                dt_in=dt if dt is not None else -np.ones(nparticles),
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
            quad_pts=quad_info,
            srange=srange,
            trange=trange,
            zrange=zrange,
            stz_init=stz_inits,
            m=mass,
            q=charge,
            vtotal=vtotal,
            vtang=parallel_speeds,
            tmax=tmax,
            tol=tol,
            dt_in=-np.ones(nparticles),
            psi0=psi0,
            nparticles=nparticles,
            vacuum=vacuum,
        )

    last_time = np.reshape(last_time, (nparticles, 6))
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
):
    """
    Trace particles in Cartesian coordinates using CATAPULT
    field: a magnetic field object representing the field in Cartesian coordinates
    surface_classifier: a simsopt surface classifier object for detecting a surface
    xyz_inits: initial conditions for particles in (x, y, z) coordinates
    parallel_speeds: initial parallel speeds of the particles
    tmax: maximum time to trace particles
    mass: mass of each particle
    charge: charge of each particle
    vtotal: total velocity of each particle
    tol: tolerance for the ODE solver
    dt: the initial time step size for the solver (optional)
    """
    xyz_inits = np.ascontiguousarray(xyz_inits, dtype=float)
    parallel_speeds = np.ascontiguousarray(parallel_speeds, dtype=float)
    if dt is not None:
        dt = np.ascontiguousarray(dt, dtype=float)
    nparticles = xyz_inits.shape[0]
    _validate_parallel_speeds(parallel_speeds, vtotal)
    r_range, phi_range, z_range, quad_info = cartesian_interpolant(
        field, surface_classifier
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
        dt_in=dt if dt is not None else -np.ones(nparticles),
        nparticles=nparticles,
    )
    last_time = np.reshape(last_time, (nparticles, 6))
    return last_time


def trace_particles_cartesian_with_collisions_gpu(
    field,
    surface_classifier,
    flux_label,
    xyz_inits,
    parallel_speeds,
    backgrounds,
    tmax,
    mass,
    charge,
    vtotal,
    tol,
    dt=None,
    rng_seed=0,
):
    """
    Trace particles in Cartesian coordinates on the GPU, including Monte
    Carlo Coulomb collisions.

    The collision operator matches
    :func:`trace_particles_boozer_with_collisions_gpu`.  The thermal
    profiles remain functions of the flux label ``s``; since the Cartesian
    state does not carry ``s``, ``flux_label`` supplies it as a scalar field
    that is interpolated alongside the magnetic field and evaluated at the
    particle position after each accepted orbit step.

    The output has seven columns: ``[t, x, y, z, v_par, v, dt]``; see
    :func:`trace_particles_boozer_with_collisions_gpu` for why the total
    speed ``v`` is reported and for the other differences from CPU tracing.

    Args:
        field: InterpolatedField object (simsopt) defined in cylindrical
            coordinates, as in :func:`trace_particles_cartesian_gpu`.
        surface_classifier: SurfaceClassifier object for detecting the
            plasma boundary.
        flux_label: Callable mapping an ``(N, 3)`` array of cylindrical
            points ``(r, phi, z)`` to the flux label ``s`` the background
            profiles are parametrized by, e.g. built from
            :class:`~firm3d.field.coordinates.BoozerCoordinateTransformer`.
            It must return finite values on the whole interpolation grid;
            values above 1 outside the plasma are clamped by the profile
            lookup.
        xyz_inits: ``(nparticles, 3)`` array of initial ``(x, y, z)``.
        parallel_speeds: ``(nparticles,)`` array of initial ``v_par`` in m/s.
        backgrounds: A :class:`~firm3d.field.collisions.ThermalBackground` or
            a list of them; coefficients are summed over species.
        tmax: Integration time in seconds.
        mass: EP mass in kg.
        charge: EP charge in C.
        vtotal: Initial speed of every particle, in m/s.
        tol: Tolerance for the adaptive step control.
        dt: Initial step size; ``None`` lets the tracer choose.
        rng_seed: Base seed.  Each particle uses a Philox stream keyed on its
            global index, so results do not depend on the block layout.

    Returns:
        ``(nparticles, 7)`` array of final states,
        ``[t, x, y, z, v_par, v, dt]``.

    Raises:
        ValueError: If ``backgrounds`` is empty or holds more than
            ``firm3dpp.COLL_MAX_SPECIES`` species; if ``parallel_speeds``
            does not match ``xyz_inits`` in length or exceeds ``vtotal`` in
            magnitude; if ``flux_label`` returns non-finite values on the
            grid; or if the profiles give an unphysical Coulomb logarithm.
    """
    from firm3d.field.collisions import (
        ThermalBackground,
        _validate_coulomb_log,
        _validate_species_count,
    )

    xyz_inits = np.ascontiguousarray(xyz_inits, dtype=float)
    parallel_speeds = np.ascontiguousarray(parallel_speeds, dtype=float)
    nparticles = xyz_inits.shape[0]

    if not callable(flux_label):
        raise ValueError(
            "flux_label must be callable; the collisional kernel reads the "
            "flux label as an interpolant column, so tracing without one is "
            "not possible -- use trace_particles_cartesian_gpu for a "
            "collisionless trace"
        )

    if len(parallel_speeds) != nparticles:
        raise ValueError(
            f"parallel_speeds has length {len(parallel_speeds)} but xyz_inits "
            f"has {nparticles} rows; the C++ layer indexes both by particle "
            f"and would read past the shorter one"
        )

    _validate_parallel_speeds(parallel_speeds, vtotal)

    if isinstance(backgrounds, ThermalBackground):
        backgrounds = [backgrounds]
    if len(backgrounds) == 0:
        raise ValueError(
            "backgrounds is empty; use trace_particles_cartesian_gpu for a "
            "collisionless trace"
        )
    _validate_species_count(backgrounds)
    cpp_backgrounds = [b._to_cpp() for b in backgrounds]
    _validate_coulomb_log(cpp_backgrounds, float(mass), float(charge))

    r_range, phi_range, z_range, quad_info = cartesian_interpolant(
        field, surface_classifier, flux_label=flux_label
    )
    last_time = firm3dpp.cartesian_collision_gpu_tracing(
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
        dt_in=-np.ones(nparticles) if dt is None else dt * np.ones(nparticles),
        nparticles=nparticles,
        backgrounds=cpp_backgrounds,
        rng_seed=int(rng_seed),
    )
    return np.reshape(last_time, (nparticles, 7))


def trace_particles_boozer_with_collisions_gpu(
    field,
    stz_inits,
    parallel_speeds,
    backgrounds,
    tmax,
    mass,
    charge,
    vtotal,
    tol,
    ns,
    ntheta,
    nzeta,
    dt=None,
    rng_seed=0,
):
    """
    Trace particles in Boozer coordinates on the GPU, including Monte Carlo
    Coulomb collisions.

    The collision operator is the one
    :func:`~firm3d.field.collisions.trace_particles_boozer_with_collisions`
    uses.  It is applied as a kick after each
    accepted orbit step, sub-cycled when the collision rates are fast relative
    to that step.

    The output has seven columns: ``[t, s, theta, zeta, v_par, v, dt]``.
    Collisions change the total speed ``v``, so it is reported: it carries
    the kinetic energy :math:`E = \\tfrac{1}{2} m v^2`, and with :math:`|B|`
    at the returned position it gives back
    :math:`\\mu = (v^2 - v_\\parallel^2)/(2|B|)`.  The first six columns
    match the CPU entry point, which appends ``v`` for the same reason; the
    final step size ``dt`` sits in the last column, as in
    :func:`trace_particles_boozer_gpu`.

    Differences from the CPU entry point:

    * ``vtotal`` is the initial speed of every particle, not merely a
      normalization: ``mu`` is derived from it as
      ``(vtotal**2 - v_par**2) / (2 |B|)``.  Passing ``|v_par| > vtotal``
      would therefore give a negative ``mu``, so it is rejected.
    * only the final state is returned, not the trajectory;
    * stopping criteria are not available.

    Results are not comparable element-by-element with the CPU tracer even at
    a fixed seed: the two draw from different generators (Philox against
    mt19937_64) and take different step sequences, so agreement is
    statistical.

    Args:
        field: A :class:`~firm3d.field.boozermagneticfield.BoozerMagneticField`
            with ``field_type`` ``"vac"`` or ``""``.
        stz_inits: ``(nparticles, 3)`` array of initial ``(s, theta, zeta)``.
        parallel_speeds: ``(nparticles,)`` array of initial ``v_par`` in m/s.
        backgrounds: A :class:`~firm3d.field.collisions.ThermalBackground` or a
            list of them; coefficients are summed over species.
        tmax: Integration time in seconds.
        mass: EP mass in kg.
        charge: EP charge in C.
        vtotal: Initial speed of every particle, in m/s.  ``mu`` is derived
            from this and ``parallel_speeds``; see above.
        tol: Tolerance for the adaptive step control.
        ns, ntheta, nzeta: Interpolant resolution.
        dt: Initial step size; ``None`` lets the tracer choose.
        rng_seed: Base seed.  Each particle uses a Philox stream keyed on its
            global index, so results do not depend on the block layout.

    Returns:
        ``(nparticles, 7)`` array of final states,
        ``[t, s, theta, zeta, v_par, v, dt]``.

    Raises:
        ValueError: If ``field.field_type`` is not ``"vac"`` or ``""``; if
            ``backgrounds`` is empty or holds more than
            ``firm3dpp.COLL_MAX_SPECIES`` species; if ``parallel_speeds``
            does not match ``stz_inits`` in length or exceeds ``vtotal`` in
            magnitude; or if the profiles give an unphysical Coulomb
            logarithm.
    """
    from firm3d.field.collisions import (
        ThermalBackground,
        _validate_coulomb_log,
        _validate_species_count,
    )

    stz_inits = np.ascontiguousarray(stz_inits, dtype=float).copy()
    parallel_speeds = np.ascontiguousarray(parallel_speeds, dtype=float)
    nparticles = stz_inits.shape[0]
    if field.field_type not in ["vac", ""]:
        raise ValueError(
            f"Unsupported field type {field.field_type} for Boozer tracing, "
            f"expected 'vac' or ''"
        )
    vacuum = field.field_type == "vac"

    if len(parallel_speeds) != nparticles:
        raise ValueError(
            f"parallel_speeds has length {len(parallel_speeds)} but stz_inits "
            f"has {nparticles} rows; the C++ layer indexes both by particle "
            f"and would read past the shorter one"
        )

    _validate_parallel_speeds(parallel_speeds, vtotal)

    if isinstance(backgrounds, ThermalBackground):
        backgrounds = [backgrounds]
    if len(backgrounds) == 0:
        raise ValueError(
            "backgrounds is empty; use trace_particles_boozer_gpu for a "
            "collisionless trace"
        )
    _validate_species_count(backgrounds)
    cpp_backgrounds = [b._to_cpp() for b in backgrounds]
    _validate_coulomb_log(cpp_backgrounds, float(mass), float(charge))

    srange, trange, zrange, quad_info, maxJ = boozer_interpolant(
        field, field.nfp, ns, ntheta, nzeta, vacuum=vacuum
    )
    last_time = firm3dpp.boozer_collision_gpu_tracing(
        quad_pts=quad_info,
        srange=srange,
        trange=trange,
        zrange=zrange,
        stz_init=stz_inits,
        m=mass,
        q=charge,
        vtotal=vtotal,
        vtang=parallel_speeds,
        tmax=tmax,
        tol=tol,
        dt_in=-np.ones(nparticles) if dt is None else dt * np.ones(nparticles),
        psi0=field.psi0,
        nparticles=nparticles,
        backgrounds=cpp_backgrounds,
        vacuum=vacuum,
        rng_seed=int(rng_seed),
    )
    return np.reshape(last_time, (nparticles, 7))
