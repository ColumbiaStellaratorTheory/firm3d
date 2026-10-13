import numbers
import warnings

import numpy as np

import firm3dpp as sopp

from .._core.types import RealArray
from .._core.util import parallel_loop_bounds
from ..field.boozermagneticfield import BoozerMagneticField
from ..field.tracing_helpers import _validate_parallel_speeds
from ..util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)

__all__ = [
    "ThermalBackground",
    "trace_particles_boozer_perturbed_with_collisions",
    "trace_particles_boozer_with_collisions",
]


def _eval_profile_on_grid(profile, s_grid, name):
    r"""
    Evaluate ``profile`` at every node of ``s_grid``, checking what it returns.

    The C++ layer stores one number per node, so each call must return
    something convertible to a real scalar; an array-valued or non-numeric
    profile is refused here rather than failing confusingly further down.
    Values must be finite and non-negative.

    Zero is allowed.  A species with :math:`n = 0` or :math:`T = 0` is simply
    inactive at that ``s`` -- the C++ collision operator skips species with
    ``n <= 0`` or ``T <= 0`` -- which is how the edge of a profile such as
    ``1 - s**5`` is meant to behave.  Only negative values are unphysical.
    """
    vals = np.empty(s_grid.size, dtype=float)
    for i, s in enumerate(s_grid):
        raw = profile(float(s))
        # numbers.Real admits int, float and the numpy scalar types, and
        # rejects arrays, None, complex and numeric strings such as "1e20".
        if not isinstance(raw, numbers.Real):
            raise ValueError(
                f"{name} must return a real scalar, but returned {raw!r} "
                f"at s = {float(s):.4f}."
            )
        vals[i] = float(raw)

    for label, bad in (
        ("finite", ~np.isfinite(vals)),
        ("non-negative", vals < 0.0),
    ):
        if bad.any():
            i = int(np.argmax(bad))
            raise ValueError(
                f"{name} must be {label} on the grid, but returned "
                f"{float(vals[i])!r} at s = {float(s_grid[i]):.4f}."
            )

    return vals


class ThermalBackground:
    r"""
    Maxwellian background species for Monte Carlo Coulomb collision calculations.

    Density and temperature profiles are specified as callables of the
    normalized toroidal flux ``s``, following the same convention used by
    :func:`~firm3d.field.tracing_helpers.initialize_position_profile`.

    The Coulomb logarithm is computed locally from the profiles as the ratio
    of the Debye length to the minimum impact parameter:

    .. math::

        \ln\Lambda_{ab} = \ln\!\left(\frac{\lambda_D}{b_\mathrm{min}}\right),
        \qquad
        b_\mathrm{min} = \max(b_\mathrm{cl},\, b_\mathrm{qm}),

    .. math::

        b_\mathrm{cl} = \frac{|q_a q_b|}
                             {4\pi\varepsilon_0\,m_r\,v_\mathrm{eff}^2},
        \qquad
        b_\mathrm{qm} = \frac{\hbar}{2\,m_r\,v_\mathrm{eff}},
        \qquad
        v_\mathrm{eff}^2 = v^2 + v_{\mathrm{th},b}^2

    where :math:`\lambda_D` is the total Debye length from all background
    species and :math:`m_r = m_a m_b/(m_a+m_b)` is the reduced mass.
    :math:`b_\mathrm{cl}` is the classical 90-degree deflection radius and
    :math:`b_\mathrm{qm}` the de Broglie wavelength; taking the larger covers
    the quantum regime, which fast EPs reach against electrons.

    Args:
        n_profile: Callable ``n(s)`` returning number density in m\ :sup:`-3`.
            Must return a finite, non-negative real scalar at every grid
            point; zero marks the species inactive there.
        T_profile: Callable ``T(s)`` returning temperature in eV, under the
            same requirements as ``n_profile``.
        mass: Species mass in kg.
        charge: Species charge in C (signed).
        n_grid_points: Number of uniformly-spaced points in ``s`` on which
            the profiles are pre-evaluated before passing to C++.  The C++
            layer uses linear interpolation between these points, so at least
            two are required; fewer raises :class:`ValueError`.

    Example::

        from firm3d.util.constants import PROTON_MASS, ELEMENTARY_CHARGE
        n_ref = 1e20  # m^-3
        T_ref = 10e3  # 10 keV, in eV

        background = ThermalBackground(
            n_profile=lambda s: n_ref * (1 - s**5),
            T_profile=lambda s: T_ref * (1 - s),
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )
    """

    def __init__(
        self,
        n_profile,
        T_profile,
        mass,
        charge,
        n_grid_points=512,
    ):
        if not callable(n_profile):
            raise ValueError("n_profile must be callable.")
        if not callable(T_profile):
            raise ValueError("T_profile must be callable.")
        n_grid_points = int(n_grid_points)
        if n_grid_points < 2:
            raise ValueError(f"n_grid_points must be at least 2, got {n_grid_points}")

        self.n_profile = n_profile
        self.T_profile = T_profile
        self.mass = float(mass)
        self.charge = float(charge)
        self._n_grid_points = n_grid_points

        # Evaluate both profiles now so that a profile returning the wrong
        # kind of value, or a negative one, is reported here rather than
        # part-way into a trace.
        s_grid = self._s_grid()
        _eval_profile_on_grid(self.n_profile, s_grid, "n_profile")
        _eval_profile_on_grid(self.T_profile, s_grid, "T_profile")

    def _s_grid(self):
        return np.linspace(0.0, 1.0, self._n_grid_points)

    def _to_cpp(self):
        """Return a sopp.ThermalBackground struct with pre-evaluated grids."""
        s_grid = self._s_grid()
        n_vals = _eval_profile_on_grid(self.n_profile, s_grid, "n_profile")
        T_vals = _eval_profile_on_grid(self.T_profile, s_grid, "T_profile")

        bg = sopp.ThermalBackground()
        bg.s_grid = s_grid.tolist()
        bg.n_grid = n_vals.tolist()
        bg.T_grid = T_vals.tolist()
        bg.mass = self.mass
        bg.charge = self.charge
        return bg


def _validate_species_count(backgrounds):
    """
    Refuse more background species than the C++ layer can hold.
    """
    if len(backgrounds) > sopp.COLL_MAX_SPECIES:
        raise ValueError(
            f"at most {sopp.COLL_MAX_SPECIES} background species are "
            f"supported, but {len(backgrounds)} were given"
        )


def _validate_coulomb_log(cpp_backgrounds, mass, charge):
    r"""
    Check the Coulomb logarithm implied by the background profiles.

    Raises :class:`ValueError` if :math:`\ln\Lambda \le 0` anywhere, and warns
    once if :math:`0 < \ln\Lambda < 2`, where the binary-collision
    approximation is marginal.
    """
    if not cpp_backgrounds:
        return

    # Scan the finest of the per-species s-grids for the smallest ln Lambda.
    s_grids = [np.asarray(bg.s_grid, dtype=float) for bg in cpp_backgrounds]
    s_scan = max(s_grids, key=len)

    ln_min = np.inf
    s_min = 0.0
    worst_species = -1
    for s in s_scan:
        lnL, i_species = sopp.min_coulomb_log(
            0.0, float(s), mass, charge, cpp_backgrounds
        )
        if i_species >= 0 and lnL < ln_min:
            ln_min = lnL
            s_min = float(s)
            worst_species = i_species

    if worst_species < 0:
        return  # no active species anywhere; C++ skips them all

    where = f"as v -> 0 at s = {s_min:.4f}, for background species {worst_species}"

    if ln_min <= 0.0:
        raise ValueError(
            f"background profiles give ln_Lambda = {ln_min:.3f} <= 0 {where}.  "
            f"The binary-collision model is undefined there.  Raise T wherever "
            f"n > 0, or restrict the profiles to the region you actually trace."
        )

    if ln_min < 2.0:
        warnings.warn(
            f"background profiles give ln_Lambda = {ln_min:.3f} < 2 {where}.  "
            f"The binary-collision approximation is marginal there, so the "
            f"collision coefficients should not be trusted quantitatively.",
            RuntimeWarning,
            stacklevel=3,
        )


def trace_particles_boozer_with_collisions(
    field: BoozerMagneticField,
    stz_inits: RealArray,
    parallel_speeds: RealArray,
    backgrounds,
    tmax=1e-2,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
    tol=1e-9,
    abstol=None,
    reltol=None,
    comm=None,
    stopping_criteria=None,
    axis=2,
    ode_solver="dormand_prince",
    DP_hmin=0.0,
    rng_seed=0,
    mode=None,
):
    r"""
    Trace guiding-center particles including Monte Carlo Coulomb collisions
    with one or more Maxwellian background species.

    The orbit is advanced by adaptive Dormand-Prince in
    :math:`(s, \theta, \zeta, v_\parallel)` at fixed :math:`\mu`, and the
    collision operator is applied as a kick to :math:`(v, \xi)` at each
    accepted step -- drift by explicit Euler plus the Milstein noise term,
    following ASCOT5's ``mccc_gc_milstein.c``.  The kick is sub-cycled when
    the collision rates are fast compared with the orbit step. Because :math:`\mu` is
    constant within a step it is a parameter of the orbit equations rather
    than a state variable, so the vacuum, ``noK`` and full guiding-center
    equations are all supported; which one is used follows
    ``field.field_type``, as in
    :func:`~firm3d.field.tracing.trace_particles_boozer`.

    See Hirvijoki et al., *Phys. Plasmas* **20**, 092505 (2013) and Boozer &
    Kuo-Petravic, *Phys. Fluids* **24**, 851 (1981) for the theoretical basis.

    Args:
        field: The :class:`~firm3d.field.boozermagneticfield.BoozerMagneticField`
            instance.
        stz_inits: ``(nparticles, 3)`` array of initial positions
            :math:`(s, \theta, \zeta)` in Boozer coordinates.
        parallel_speeds: ``(nparticles,)`` array of initial :math:`v_\parallel`
            in m/s.
        backgrounds: A :class:`ThermalBackground` or a list thereof.  When
            multiple species are provided their collision coefficients are
            summed.
        tmax: Integration time in seconds.
        mass: EP mass in kg.
        charge: EP charge in C.
        Ekin: Initial kinetic energy in J.  Either a scalar applied to all
            particles or a ``(nparticles,)`` array.
        tol: Default tolerance when solver-specific tolerances are not set.
        abstol: Absolute tolerance for the DP adaptive step control.
        reltol: Relative tolerance for the DP adaptive step control.
        comm: MPI communicator; particles are distributed across ranks.
        stopping_criteria: List of stopping criteria (same as
            :func:`~firm3d.field.tracing.trace_particles_boozer`).
        axis: Coordinate singularity handling (0, 1, or 2; default 2).
        ode_solver: ``"dormand_prince"`` (recommended) or ``"boost"``.
        DP_hmin: Minimum step size for the Dormand-Prince solver, in
            seconds.  When the adaptive step falls below this value the
            step is accepted anyway.  Prevents the solver from grinding
            when a particle diffuses deep below the background thermal
            speed, where the pitch-scattering rate diverges as 1/v^3.
        rng_seed: Seed for the per-particle Wiener process.  Each particle
            uses ``rng_seed + particle_index`` so that MPI runs are
            reproducible.
        mode: Which guiding-center equations to use: ``"gc"``, ``"gc_vac"``
            or ``"gc_nok"``.  Defaults to ``"gc_" + field.field_type``;
            passing a value inconsistent with the field warns and proceeds
            with the value given.

    Returns:
        Tuple ``(res_tys, res_hits)`` where each element of ``res_tys`` is a
        numpy array of shape ``(2, 6)`` holding the initial and final state
        of the particle, with columns ``[t, s, θ, ζ, v_par, v]``.  The extra
        column ``v`` (total speed) allows the kinetic energy
        :math:`E = \tfrac{1}{2} m v^2` and magnetic moment
        :math:`\mu = (v^2 - v_\parallel^2) / (2B)` to be reconstructed.
    """
    if stopping_criteria is None:
        stopping_criteria = []
    if abstol is None:
        abstol = tol
    if reltol is None:
        reltol = tol
    if ode_solver not in ("boost", "dormand_prince"):
        raise ValueError(
            f"collision tracing supports ode_solver 'boost' or "
            f"'dormand_prince', got {ode_solver!r}; 'symplectic' is "
            f"collisionless-only"
        )

    # Accept a single background or a list
    if isinstance(backgrounds, ThermalBackground):
        backgrounds = [backgrounds]
    _validate_species_count(backgrounds)
    cpp_backgrounds = [b._to_cpp() for b in backgrounds]
    _validate_coulomb_log(cpp_backgrounds, float(mass), float(charge))

    if mode is not None:
        mode = mode.lower()
        assert mode in ["gc", "gc_vac", "gc_nok"]
        if "gc_" + field.field_type != mode:
            warnings.warn(
                f"Prescribed mode is inconsistent with field_type. "
                f"Proceeding with mode={mode}.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        mode = "gc_" + field.field_type

    nparticles = stz_inits.shape[0]
    assert len(parallel_speeds) == nparticles

    if np.isscalar(Ekin):
        Ekin = Ekin * np.ones(nparticles)
    assert len(Ekin) == nparticles

    vtotal = np.sqrt(2.0 * np.asarray(Ekin) / mass)
    _validate_parallel_speeds(parallel_speeds, vtotal)

    res_tys = []
    res_hits = []
    first, last = parallel_loop_bounds(comm, nparticles)
    failure = None
    failure_exc = None
    for i in range(first, last):
        seed_i = int(rng_seed) + i
        try:
            res_ty, res_hit = sopp.particle_guiding_center_boozer_collision_tracing(
                field,
                stz_inits[i, :].tolist(),
                float(mass),
                float(charge),
                float(vtotal[i]),
                float(parallel_speeds[i]),
                float(tmax),
                cpp_backgrounds,
                vacuum=(mode == "gc_vac"),
                noK=(mode == "gc_nok"),
                stopping_criteria=stopping_criteria,
                axis=int(axis),
                abstol=float(abstol),
                reltol=float(reltol),
                ode_solver=str(ode_solver),
                DP_hmin=float(DP_hmin),
                rng_seed=seed_i,
            )
        except Exception as exc:
            # Recorded rather than raised, so that every rank reaches the
            # allgather below; re-raised with `from` on all ranks afterwards.
            failure = f"particle {i}: {type(exc).__name__}: {exc}"
            failure_exc = exc
            break
        res_tys.append(np.asarray(res_ty))
        res_hits.append(np.asarray(res_hit))

    if comm is not None:
        failures = comm.allgather(failure)
        failed = [(r, m) for r, m in enumerate(failures) if m is not None]
        if failed:
            rank, msg = failed[0]
            raise RuntimeError(
                f"collision tracing failed on {len(failed)} of {comm.size} "
                f"rank(s); first failure on rank {rank} -- {msg}"
            )
        res_tys = [i for o in comm.allgather(res_tys) for i in o]
        res_hits = [i for o in comm.allgather(res_hits) for i in o]
    elif failure is not None:
        raise RuntimeError(f"collision tracing failed on {failure}") from failure_exc

    return res_tys, res_hits


def trace_particles_boozer_perturbed_with_collisions(
    perturbed_field,
    stz_inits: RealArray,
    parallel_speeds: RealArray,
    mus: RealArray,
    backgrounds,
    tmax=1e-2,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    tol=1e-9,
    abstol=None,
    reltol=None,
    comm=None,
    stopping_criteria=None,
    axis=2,
    ode_solver="dormand_prince",
    DP_hmin=0.0,
    rng_seed=0,
    mode=None,
):
    r"""
    Trace guiding-center particles through a shear Alfven wave including
    Monte Carlo Coulomb collisions.

    The collisional counterpart of
    :func:`~firm3d.field.tracing.trace_particles_boozer_perturbed`, and it
    takes the same ``(parallel_speeds, mus)`` description of the initial
    velocity rather than an energy, since the wave does work on the particle
    and the speed is not a constant of the motion.

    The scheme is the one described in
    :func:`trace_particles_boozer_with_collisions`: the orbit is advanced at
    fixed :math:`\mu` and the collision operator is applied as a sub-cycled
    kick between steps.  That splitting still holds here because
    :math:`\mu` remains an adiabatic invariant at shear-Alfven frequencies
    (:math:`\omega \ll \Omega_c`) even though the energy is not conserved.
    The kick reconstructs the speed from :math:`v^2 = v_\parallel^2 + 2\mu|B_0|`.

    Args:
        perturbed_field: The
            :class:`~firm3d.field.boozermagneticfield.ShearAlfvenWave`.
        stz_inits: ``(nparticles, 3)`` array of initial positions.
        parallel_speeds: ``(nparticles,)`` array of initial
            :math:`v_\parallel` in m/s.
        mus: ``(nparticles,)`` array of magnetic moments.
        backgrounds: A :class:`ThermalBackground` or list thereof.
        mode: ``"gc_vac"`` or ``"gc_nok"``; defaults to
            ``"gc_" + perturbed_field.B0.field_type``.  There is no full-K
            perturbed right-hand side, so a field with ``field_type == ""``
            must be traced with an explicit mode.

    Returns:
        Tuple ``(res_tys, res_hits)`` with the same layout as
        :func:`trace_particles_boozer_with_collisions`.
    """
    if stopping_criteria is None:
        stopping_criteria = []
    if abstol is None:
        abstol = tol
    if reltol is None:
        reltol = tol
    if ode_solver not in ("boost", "dormand_prince"):
        raise ValueError(
            f"collision tracing supports ode_solver 'boost' or "
            f"'dormand_prince', got {ode_solver!r}; 'symplectic' is "
            f"collisionless-only"
        )

    if isinstance(backgrounds, ThermalBackground):
        backgrounds = [backgrounds]
    _validate_species_count(backgrounds)
    cpp_backgrounds = [b._to_cpp() for b in backgrounds]
    _validate_coulomb_log(cpp_backgrounds, float(mass), float(charge))

    field_type = perturbed_field.B0.field_type
    if mode is not None:
        mode = mode.lower()
        assert mode in ["gc_vac", "gc_nok"]
        if "gc_" + field_type != mode:
            warnings.warn(
                f"Prescribed mode is inconsistent with field_type. "
                f"Proceeding with mode={mode}.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        mode = "gc_" + field_type
        if mode not in ("gc_vac", "gc_nok"):
            raise ValueError(
                f"perturbed collision tracing supports vacuum_saw and nok_saw "
                f"only, but field_type is {field_type!r}; there is no full-K "
                f"perturbed right-hand side.  Pass mode explicitly to override."
            )

    nparticles = stz_inits.shape[0]
    assert len(parallel_speeds) == nparticles
    assert len(mus) == nparticles

    res_tys = []
    res_hits = []
    first, last = parallel_loop_bounds(comm, nparticles)
    failure = None
    failure_exc = None
    for i in range(first, last):
        try:
            res_ty, res_hit = (
                sopp.particle_guiding_center_boozer_perturbed_collision_tracing(
                    perturbed_field,
                    stz_inits[i, :].tolist(),
                    float(mass),
                    float(charge),
                    float(parallel_speeds[i]),
                    float(mus[i]),
                    float(tmax),
                    cpp_backgrounds,
                    vacuum=(mode == "gc_vac"),
                    noK=(mode == "gc_nok"),
                    stopping_criteria=stopping_criteria,
                    axis=int(axis),
                    abstol=float(abstol),
                    reltol=float(reltol),
                    ode_solver=str(ode_solver),
                    DP_hmin=float(DP_hmin),
                    rng_seed=int(rng_seed) + i,
                )
            )
        except Exception as exc:
            failure = f"particle {i}: {type(exc).__name__}: {exc}"
            failure_exc = exc
            break
        res_tys.append(np.asarray(res_ty))
        res_hits.append(np.asarray(res_hit))

    if comm is not None:
        failures = comm.allgather(failure)
        failed = [(r, m) for r, m in enumerate(failures) if m is not None]
        if failed:
            rank, msg = failed[0]
            raise RuntimeError(
                f"perturbed collision tracing failed on {len(failed)} of "
                f"{comm.size} rank(s); first failure on rank {rank} -- {msg}"
            )
        res_tys = [i for o in comm.allgather(res_tys) for i in o]
        res_hits = [i for o in comm.allgather(res_hits) for i in o]
    elif failure is not None:
        raise RuntimeError(
            f"perturbed collision tracing failed on {failure}"
        ) from failure_exc

    return res_tys, res_hits
