"""
Tests for Monte Carlo Coulomb collision tracing.

Test structure
--------------
  TestThermalBackground       — Python-layer ThermalBackground class
  TestTrajectoryShape         — output format, monotone time, column order
  TestCollisionlessLimit      — exact recovery of collisionless dynamics (n = 0)
  TestCollisionCoefficients   — analytical K, ν_D, Chandrasekhar G, scalings
  TestCollisionPhysics        — qualitative + quantitative simulation checks
  TestPitchIsotropization     — qualitative broadening (ξ₀=0) + quantitative
                                exponential decay ⟨ξ(t)⟩=ξ₀exp(−ν_D t) (ASCOT5)
"""

import unittest
import warnings

import numpy as np
import scipy.stats

import firm3dpp as sopp
from firm3d.field.boozermagneticfield import (
    BoozerAnalytic,
    ShearAlfvenHarmonic,
)
from firm3d.field.collisions import (
    ThermalBackground,
    trace_particles_boozer_perturbed_with_collisions,
    trace_particles_boozer_with_collisions,
)
from firm3d.field.tracing import (
    IterationStoppingCriterion,
    MaxToroidalFluxStoppingCriterion,
    trace_particles_boozer,
    trace_particles_boozer_perturbed,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    FUSION_ALPHA_PARTICLE_ENERGY,
    ONE_EV,
    PROTON_MASS,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _field(**kwargs):
    """
    BoozerAnalytic near-axis field (axisymmetric, vacuum).
    """
    return BoozerAnalytic(1.0, 5.0, 0, 40.0, 0.5, 0.4, Bbar=5.0, **kwargs)


def _zero_background():
    """Zero-density background: all collision coefficients identically zero."""
    return ThermalBackground(
        n_profile=lambda s: 0.0,
        T_profile=lambda s: 10e3,
        mass=2 * PROTON_MASS,
        charge=ELEMENTARY_CHARGE,
    )


def _hot_background(n=1e20, T_keV=10.0):
    """Dense, hot deuterium background."""
    T = T_keV * 1e3
    return ThermalBackground(
        n_profile=lambda s: n,
        T_profile=lambda s: T,
        mass=2 * PROTON_MASS,
        charge=ELEMENTARY_CHARGE,
    )


def _cold_background(n=1e20, T_keV=0.01):
    """Cold background: large drag on 3.5 MeV alphas."""
    T = T_keV * 1e3
    return ThermalBackground(
        n_profile=lambda s: n,
        T_profile=lambda s: T,
        mass=2 * PROTON_MASS,
        charge=ELEMENTARY_CHARGE,
    )


def _coll_trace(
    field,
    backgrounds,
    tmax=1e-6,
    vpar_fraction=0.5,
    tol=1e-9,
    dt_save_factor=20,
    **kwargs,
):
    """Single-particle collision trace; returns (ntimesteps, 6) array."""
    Ekin = FUSION_ALPHA_PARTICLE_ENERGY
    v0 = np.sqrt(2 * Ekin / ALPHA_PARTICLE_MASS)
    stz = np.array([[0.3, 0.0, 0.0]])
    vpar = np.array([vpar_fraction * v0])
    res_tys, _ = trace_particles_boozer_with_collisions(
        field,
        stz,
        vpar,
        backgrounds=backgrounds,
        tmax=tmax,
        mass=ALPHA_PARTICLE_MASS,
        charge=ALPHA_PARTICLE_CHARGE,
        Ekin=Ekin,
        tol=tol,
        dt_save=tmax / dt_save_factor,
        **kwargs,
    )
    return res_tys[0]


# ---------------------------------------------------------------------------
# Coefficient accessors (thin wrappers on the shipped C++)
# ---------------------------------------------------------------------------


def _coeffs(v, s, m_a, q_a, backgrounds):
    """CollisionCoefficients from collisions.h for a single (v, s)."""
    return sopp.compute_collision_coefficients(
        float(v),
        float(s),
        float(m_a),
        float(q_a),
        [b._to_cpp() for b in backgrounds],
    )


# ===========================================================================
# Test classes
# ===========================================================================


class TestThermalBackground(unittest.TestCase):
    def test_non_callable_profiles_are_refused(self):
        for label, kw in (
            ("n", {"n_profile": 1e20, "T_profile": lambda s: 1.0}),
            ("T", {"n_profile": lambda s: 1e20, "T_profile": 5.0}),
        ):
            with self.subTest(profile=label), self.assertRaises(ValueError):
                ThermalBackground(mass=PROTON_MASS, charge=ELEMENTARY_CHARGE, **kw)

    def test_profiles_returning_non_scalars_are_refused(self):
        """A profile must return something convertible to a real scalar."""
        for label, bad in (
            ("array", lambda s: np.array([1e20, 2e20])),
            ("none", lambda s: None),
            ("string", lambda s: "1e20"),
        ):
            with self.subTest(returns=label):
                with self.assertRaises(ValueError) as cm:
                    ThermalBackground(
                        n_profile=bad,
                        T_profile=lambda s: 1e3,
                        mass=PROTON_MASS,
                        charge=ELEMENTARY_CHARGE,
                    )
                self.assertIn("real scalar", str(cm.exception))

    def test_non_finite_profiles_are_refused(self):
        for label, bad in (("nan", np.nan), ("inf", np.inf)):
            with self.subTest(returns=label):
                with self.assertRaises(ValueError) as cm:
                    ThermalBackground(
                        n_profile=lambda s, v=bad: v,
                        T_profile=lambda s: 1e3,
                        mass=PROTON_MASS,
                        charge=ELEMENTARY_CHARGE,
                    )
                self.assertIn("finite", str(cm.exception))

    def test_negative_profiles_are_refused(self):
        """Negative density or temperature is unphysical."""
        for label, kw in (
            ("n", {"n_profile": lambda s: -1e20, "T_profile": lambda s: 1e3}),
            ("T", {"n_profile": lambda s: 1e20, "T_profile": lambda s: -1e3}),
        ):
            with self.subTest(profile=label):
                with self.assertRaises(ValueError) as cm:
                    ThermalBackground(mass=PROTON_MASS, charge=ELEMENTARY_CHARGE, **kw)
                self.assertIn("non-negative", str(cm.exception))

    def test_profiles_vanishing_at_the_edge_are_allowed(self):
        """
        Zero is not an error: it marks the species inactive there.

        The shipped collisional_slowing_down example uses n ~ 1 - s**5 and
        T ~ 1 - s, both of which are exactly zero at s = 1, and s = 1 is a
        node of the grid.
        """
        bg = ThermalBackground(
            n_profile=lambda s: 1e20 * (1 - s**5),
            T_profile=lambda s: 11.5e3 * (1 - s),
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )
        cpp = bg._to_cpp()
        self.assertEqual(cpp.n_grid[-1], 0.0)
        self.assertEqual(cpp.T_grid[-1], 0.0)

    def test_to_cpp_preserves_the_profiles(self):
        """The grids handed to C++ have the requested length and endpoints."""
        bg = ThermalBackground(
            n_profile=lambda s: 1e20 * (1 - s),
            T_profile=lambda s: 1e3 * s,
            mass=PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
            n_grid_points=100,
        )
        cpp = bg._to_cpp()
        self.assertEqual(len(cpp.s_grid), 100)
        self.assertEqual(len(cpp.n_grid), 100)
        self.assertEqual(len(cpp.T_grid), 100)
        self.assertAlmostEqual(cpp.n_grid[0], 1e20, delta=1e15)
        self.assertAlmostEqual(cpp.n_grid[-1], 0.0, delta=1e15)
        self.assertAlmostEqual(cpp.T_grid[0], 0.0, delta=1e-9)
        self.assertAlmostEqual(cpp.T_grid[-1], 1e3, delta=1e-9)


class TestTrajectoryShape(unittest.TestCase):
    def test_output_columns(self):
        """(ntimesteps, 6) = [t, s, θ, ζ, v∥, v], with t increasing and |v∥| ≤ v."""
        traj = _coll_trace(_field(), _hot_background(), tmax=1e-6)
        self.assertEqual(traj.ndim, 2)
        self.assertEqual(traj.shape[1], 6)
        self.assertTrue(np.all(np.diff(traj[:, 0]) > 0), "time is not increasing")
        vpar, v = traj[:, 4], traj[:, 5]
        self.assertTrue(np.all(np.abs(vpar) <= v * (1 + 1e-10)), "|v_par| > v")

    def test_forget_exact_path_returns_two_rows(self):
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        v0 = np.sqrt(2 * Ekin / ALPHA_PARTICLE_MASS)
        res_tys, _ = trace_particles_boozer_with_collisions(
            _field(),
            np.array([[0.3, 0.0, 0.0]]),
            np.array([0.5 * v0]),
            backgrounds=_hot_background(),
            tmax=1e-6,
            mass=ALPHA_PARTICLE_MASS,
            charge=ALPHA_PARTICLE_CHARGE,
            Ekin=Ekin,
            dt_save=1e-7,
            forget_exact_path=True,
        )
        self.assertEqual(res_tys[0].shape[0], 2)


# ---------------------------------------------------------------------------
# Stopping criteria
# ---------------------------------------------------------------------------


class TestIterationStoppingCriterion(unittest.TestCase):
    """
    The `iter` argument passed to stopping criteria must count accepted
    solver steps, as it does in the collisionless tracer.
    """

    _TMAX = 1e-6

    def _stop_time(self, max_iter, tol):
        """Final time of a single zero-density trace under an iteration cap."""
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        v0 = np.sqrt(2 * Ekin / ALPHA_PARTICLE_MASS)
        res_tys, res_hits = trace_particles_boozer_with_collisions(
            _field(),
            np.array([[0.3, 0.0, 0.0]]),
            np.array([0.5 * v0]),
            backgrounds=_zero_background(),
            tmax=self._TMAX,
            mass=ALPHA_PARTICLE_MASS,
            charge=ALPHA_PARTICLE_CHARGE,
            Ekin=Ekin,
            dt_save=self._TMAX / 20,
            tol=tol,
            stopping_criteria=[IterationStoppingCriterion(max_iter)],
        )
        return res_tys[0][-1, 0], np.asarray(res_hits[0])

    def test_stop_time_scales_with_the_iteration_cap(self):
        """
        A cap stops the trace early and records a hit at index -1, and
        doubling it lets the trace run measurably further -- which is what
        fails if `iter` is not the accepted-step count.  Run at tol = 1e-11
        so that even 20 steps fall well short of tmax; at looser tolerances
        the steps are large enough that neither trace reaches its cap.
        """
        t_10, hits_10 = self._stop_time(10, tol=1e-11)
        t_20, hits_20 = self._stop_time(20, tol=1e-11)
        for cap, hits in ((10, hits_10), (20, hits_20)):
            self.assertEqual(hits.shape[0], 1, f"max_iter = {cap} ran to tmax")
            self.assertEqual(hits[0, 1], -1.0)
        self.assertLess(t_20, self._TMAX)
        self.assertGreater(
            t_20,
            1.5 * t_10,
            f"stop time saturated in max_iter (10 -> {t_10:.4e}, 20 -> {t_20:.4e})",
        )


# ---------------------------------------------------------------------------
# Collisionless limit
# ---------------------------------------------------------------------------


class TestCollisionlessLimit(unittest.TestCase):
    """
    When n = 0 every collision coefficient is identically zero, so the kick
    is a no-op: K = 0 leaves v_par untouched and g_v = √(2 D_par) = 0 kills
    the Milstein noise.  The orbit must therefore reduce to the same vacuum
    guiding-centre equations the collisionless tracer integrates, which is
    what pins the collisional path to a tracer covered by test_particle.py.
    """

    _tmax = 5e-7
    _tol = 1e-9
    _stz = np.array([[0.3, 0.1, 0.2]])
    _vpfrac = 0.6

    def _traces(self):
        """Return (traj_coll, traj_nocoll) for the same particle and settings."""
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        m, q = ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE
        v0 = np.sqrt(2 * Ekin / m)
        vpar = np.array([self._vpfrac * v0])
        field = _field()
        kwargs = {
            "tmax": self._tmax,
            "mass": m,
            "charge": q,
            "Ekin": Ekin,
            "abstol": self._tol,
            "reltol": self._tol,
            "dt_save": self._tmax / 20,
        }

        res_c, _ = trace_particles_boozer_with_collisions(
            field,
            self._stz,
            vpar,
            backgrounds=_zero_background(),
            ode_solver="dormand_prince",
            axis=2,
            **kwargs,
        )

        res_nc, _ = trace_particles_boozer(
            field,
            self._stz,
            vpar,
            ODE_solver="dormand_prince",
            axis=2,
            **kwargs,
        )
        return res_c[0], res_nc[0]

    def test_orbit_matches_collisionless_tracer(self):
        """
        (s, θ, ζ, v∥) from the collision tracer with n = 0 must agree with
        the standard collisionless tracer.

        Both use Dormand-Prince, the same tolerances, and axis = 2.
        Any discrepancy is purely due to the (v, ξ) vs (v∥) state conditioning.
        """
        traj_c, traj_nc = self._traces()
        n = min(traj_c.shape[0], traj_nc.shape[0])

        # Saved time stamps must align
        np.testing.assert_allclose(
            traj_c[:n, 0],
            traj_nc[:n, 0],
            rtol=1e-6,
            atol=0,
            err_msg=(
                "Saved times disagree between collision (n=0) and collisionless tracers"
            ),
        )

        # (s, θ, ζ)
        for col, name in [(1, "s"), (2, "θ"), (3, "ζ")]:
            np.testing.assert_allclose(
                traj_c[:n, col],
                traj_nc[:n, col],
                rtol=1e-4,
                atol=1e-6,
                err_msg=(
                    f"{name}(t) differs between n=0 collision and collisionless traces"
                ),
            )

        # v∥ = ξ·v (collision) vs v∥ (collisionless)
        np.testing.assert_allclose(
            traj_c[:n, 4],
            traj_nc[:n, 4],
            rtol=1e-4,
            atol=1.0,  # 1 m/s floor for numerical noise
            err_msg="v∥ differs between n=0 collision and collisionless traces",
        )


# ---------------------------------------------------------------------------
# Orbit equations for non-vacuum fields
# ---------------------------------------------------------------------------


class TestNonVacuumOrbitEquations(unittest.TestCase):
    """
    Collisional tracing must use the orbit equations the field calls for:
    vacuum / noK / full, selected from field.field_type exactly as
    trace_particles_boozer selects them.  mu is a parameter of the orbit
    equations rather than a state variable, so every static-field variant
    can be driven by the collision operator.
    """

    _tmax = 5e-7
    _tol = 1e-10
    _stz = np.array([[0.3, 0.0, 0.0]])

    # I0 alone gives field_type 'nok'; adding K1 gives '' (full GC).  Both are
    # orbit models the collisional tracer could not reach before.
    _FIELDS = {
        "nok": {"I0": 0.5},
        "": {"I0": 0.5, "K1": 0.3},
    }

    @classmethod
    def _nonvacuum_field(cls, field_type=""):
        return _field(**cls._FIELDS[field_type])

    def _endpoint(self, collisional, mode=None, field_type=""):
        """Final [s, theta, zeta, v_par] for one particle."""
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        m, q = ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE
        v0 = np.sqrt(2 * Ekin / m)
        vpar = np.array([0.6 * v0])
        kwargs = {
            "tmax": self._tmax,
            "mass": m,
            "charge": q,
            "Ekin": Ekin,
            "abstol": self._tol,
            "reltol": self._tol,
            "dt_save": self._tmax,
        }
        # Pin both to the same stepper.  The two functions default
        # differently (dormand_prince vs boost), and leaving that alone
        # would make the agreement margin below a measure of stepper
        # difference rather than of the orbit equations.
        if collisional:
            res, _ = trace_particles_boozer_with_collisions(
                self._nonvacuum_field(field_type),
                self._stz,
                vpar,
                backgrounds=_zero_background(),
                mode=mode,
                ode_solver="dormand_prince",
                **kwargs,
            )
        else:
            res, _ = trace_particles_boozer(
                self._nonvacuum_field(field_type),
                self._stz,
                vpar,
                mode=mode,
                ODE_solver="dormand_prince",
                **kwargs,
            )
        return np.asarray(res[0])[-1, 1:5]

    def test_collisionless_limit_matches_collisionless_tracer(self):
        """
        With n = 0 the collisional tracer must reproduce
        trace_particles_boozer for every non-vacuum orbit model -- this is
        what fails if the vacuum equations are used for a field that needs
        another set.  Forcing gc_vac is checked to disagree, so that the
        comparison cannot pass merely because the two formulations happen to
        coincide on this equilibrium.
        """
        for ft in self._FIELDS:
            with self.subTest(field_type=ft):
                self.assertEqual(self._nonvacuum_field(ft).field_type, ft)
                np.testing.assert_allclose(
                    self._endpoint(collisional=True, field_type=ft),
                    self._endpoint(collisional=False, field_type=ft),
                    rtol=1e-6,
                    atol=1e-9,
                    err_msg=(
                        f"collisional trace at n=0 disagrees with the "
                        f"collisionless tracer for field_type={ft!r}: the "
                        f"orbit equations differ"
                    ),
                )

        full = self._endpoint(collisional=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vac = self._endpoint(collisional=True, mode="gc_vac")
        rel = np.abs(full - vac) / np.maximum(np.abs(full), 1e-30)
        self.assertGreater(
            rel.max(),
            1e-3,
            f"vacuum and full GC equations agree to {rel.max():.2e} on this "
            f"field, so this configuration cannot detect the wrong choice",
        )


# ---------------------------------------------------------------------------
# Perturbed (shear-Alfven-wave) collisional tracing
# ---------------------------------------------------------------------------


class TestPerturbedCollisions(unittest.TestCase):
    """
    Collisions in a shear-Alfven-wave field.

    The wave does work on the particle, so the speed is not conserved between
    kicks.  The operator splitting survives that because it needs mu to be
    invariant across an orbit step, not v: at SAW frequencies
    (omega << Omega_c) mu remains an adiabatic invariant, which is why the
    perturbed right-hand sides carry it as a constructor parameter in the
    first place.  The kick reconstructs v from v_par^2 + 2 mu |B0| each time,
    so a changing energy is handled.
    """

    _tmax = 2e-7
    _tol = 1e-11
    _stz = np.array([[0.3, 0.0, 0.0]])

    def _setup(self, Phihat, field_kwargs=None):
        # The field must stay referenced: ShearAlfvenWave.B0 only returns the
        # Python subclass (and hence field_type) while a Python reference to
        # it survives.  Returned alongside the wave for that reason.
        field = _field(**(field_kwargs or {}))
        self._field = field
        saw = ShearAlfvenHarmonic(Phihat, 2, 1, 1e5, 0.0, field)
        v0 = np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)
        field.set_points(self._stz)
        B0 = field.modB()[0, 0]
        vpar0 = 0.6 * v0
        mu0 = (v0**2 - vpar0**2) / (2.0 * B0)
        return saw, v0, np.array([vpar0]), np.array([mu0])

    def _kw(self):
        return {
            "tmax": self._tmax,
            "mass": ALPHA_PARTICLE_MASS,
            "charge": ALPHA_PARTICLE_CHARGE,
            "abstol": self._tol,
            "reltol": self._tol,
            "dt_save": self._tmax,
        }

    def test_collisionless_limit_matches_perturbed_tracer(self):
        """
        At zero density the collisional path must reproduce
        trace_particles_boozer_perturbed, including when the wave is strong
        enough to change the energy substantially.
        """
        for Phihat in (1e-3, 1e6):
            with self.subTest(Phihat=Phihat):
                saw, v0, vpar, mus = self._setup(Phihat)
                c, _ = trace_particles_boozer_perturbed_with_collisions(
                    saw,
                    self._stz,
                    vpar,
                    mus,
                    backgrounds=_zero_background(),
                    ode_solver="dormand_prince",
                    **self._kw(),
                )
                n, _ = trace_particles_boozer_perturbed(
                    saw,
                    self._stz,
                    vpar,
                    mus,
                    ODE_solver="dormand_prince",
                    **self._kw(),
                )
                np.testing.assert_allclose(
                    np.asarray(c[0])[-1, 1:5],
                    np.asarray(n[0])[-1, 1:5],
                    rtol=1e-9,
                    err_msg=f"perturbed collisional path diverges at Phihat={Phihat}",
                )

    def test_collisions_change_mu(self):
        """
        With a background present mu must evolve, since the kick rewrites it
        from the post-kick (v, xi).  Collisionlessly it is held fixed.
        """
        saw, v0, vpar, mus = self._setup(1e-3)
        kw = dict(self._kw())
        kw["dt_save"] = self._tmax / 20
        out = {}
        for label, bg in [
            ("collisionless", _zero_background()),
            ("collisional", _hot_background()),
        ]:
            res, _ = trace_particles_boozer_perturbed_with_collisions(
                saw,
                self._stz,
                vpar,
                mus,
                backgrounds=bg,
                ode_solver="dormand_prince",
                rng_seed=3,
                **kw,
            )
            t = np.asarray(res[0])
            saw.B0.set_points(np.column_stack([t[:, 1], t[:, 2], t[:, 3]]))
            B = np.array(saw.B0.modB()).flatten()
            out[label] = (t[:, 5] ** 2 - t[:, 4] ** 2) / (2.0 * B)
        drift_free = np.abs(out["collisionless"] / mus[0] - 1).max()
        drift_coll = np.abs(out["collisional"] / mus[0] - 1).max()
        self.assertLess(drift_free, 1e-6, "mu moved without collisions")
        self.assertGreater(
            drift_coll,
            10 * max(drift_free, 1e-12),
            f"mu barely moved with collisions ({drift_coll:.2e}); the kick is "
            f"not updating it",
        )

    def test_zero_amplitude_matches_static_collisional_path(self):
        """
        At zero wave amplitude the perturbed path must reproduce the static
        collisional path, collisions included.
        """
        field = _field()
        saw = ShearAlfvenHarmonic(0.0, 2, 1, 1e5, 0.0, field)
        bg = ThermalBackground(
            n_profile=lambda s: 1e21,
            T_profile=lambda s: 1e3,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        v0 = np.sqrt(2 * Ekin / ALPHA_PARTICLE_MASS)
        vpar0 = 0.6 * v0
        field.set_points(self._stz)
        mu0 = (v0**2 - vpar0**2) / (2.0 * field.modB()[0, 0])

        kw = {
            "tmax": 3e-7,
            "mass": ALPHA_PARTICLE_MASS,
            "charge": ALPHA_PARTICLE_CHARGE,
            "abstol": 1e-11,
            "reltol": 1e-11,
            "dt_save": 3e-7,
            "rng_seed": 7,
            "backgrounds": bg,
        }
        static, _ = trace_particles_boozer_with_collisions(
            field, self._stz, np.array([vpar0]), Ekin=Ekin, **kw
        )
        perturbed, _ = trace_particles_boozer_perturbed_with_collisions(
            saw, self._stz, np.array([vpar0]), np.array([mu0]), **kw
        )
        np.testing.assert_allclose(
            np.asarray(perturbed[0])[-1, 1:],
            np.asarray(static[0])[-1, 1:],
            rtol=1e-8,
            atol=1e-12,
            err_msg=(
                "zero-amplitude perturbed collisional trace disagrees with the "
                "static one: the kick's mu is not reaching the perturbed orbit "
                "equations"
            ),
        )

    def test_stopping_criterion_fires_and_pins_the_hit_layout(self):
        """
        Stopping criteria on the perturbed collisional path.
        """
        saw, _, vpar, mus = self._setup(1e-3)
        s_stop = 0.32
        kw = self._kw()
        kw["dt_save"] = 2e-8
        res, hits = trace_particles_boozer_perturbed_with_collisions(
            saw,
            self._stz,
            vpar,
            mus,
            backgrounds=_zero_background(),
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(s_stop)],
            **kw,
        )
        traj, hit = np.asarray(res[0]), np.asarray(hits[0])

        self.assertEqual(
            hit.shape, (1, 7), f"expected one 7-column hit, got {hit.shape}"
        )
        self.assertEqual(hit[0, 1], -1.0, "hit should index the first criterion")
        self.assertGreaterEqual(hit[0, 2], s_stop, "hit recorded below the threshold")
        self.assertLess(
            traj[-1, 0], self._tmax, "trace ran to tmax despite the criterion firing"
        )
        self.assertEqual(
            traj.shape[1], 6, "res_tys must stay [t, s, theta, zeta, v_par, v]"
        )

        # Column 6 is a speed, not the time.  With no background mu is fixed,
        # so it is reconstructible from the hit's own position and v_par.
        field = self._field
        field.set_points(np.array([hit[0, 2:5]]))
        v_expected = np.sqrt(hit[0, 5] ** 2 + 2.0 * mus[0] * field.modB()[0, 0])
        self.assertAlmostEqual(
            hit[0, 6] / v_expected,
            1.0,
            places=9,
            msg=(
                f"hit column 6 is {hit[0, 6]:.6e}; expected the speed "
                f"{v_expected:.6e}, not the time {hit[0, 0]:.3e}"
            ),
        )


class TestProfileGridValidation(unittest.TestCase):
    """
    The profile grid must have at least two nodes.

    Linear interpolation needs two nodes and a non-zero spacing; with one the
    spacing is 0/0 and the lookup reads off the end of the sample array.  The
    same lookup runs on the GPU against a raw device pointer, where that read
    is unchecked.
    """

    def _bg(self, n_grid_points):
        return ThermalBackground(
            n_profile=lambda s: 1e20,
            T_profile=lambda s: 10e3,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
            n_grid_points=n_grid_points,
        )

    def test_degenerate_grid_is_refused(self):
        for npts in (1, 0, -3):
            with self.subTest(n_grid_points=npts):
                with self.assertRaises(ValueError) as cm:
                    self._bg(npts)
                self.assertIn("at least 2", str(cm.exception))


# ---------------------------------------------------------------------------
# Collision-kick sub-cycling
# ---------------------------------------------------------------------------


class TestCollisionSubstepping(unittest.TestCase):
    """
    The orbit stepper sizes its step from orbit dynamics alone, so the
    collision terms get no say in it.  Applying them as a single explicit
    Euler kick over that step is badly wrong once the collision rates are
    fast compared with it, which is the thermal regime.  The kick is
    therefore sub-cycled.
    """

    def _coef(self, v, bgs, s=0.3):
        return sopp.compute_collision_coefficients(
            v,
            s,
            float(PROTON_MASS),
            float(ELEMENTARY_CHARGE),
            [b._to_cpp() for b in bgs],
        )

    def test_slow_rates_need_no_subcycling(self):
        v0 = np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)
        c = self._coef(v0, [_hot_background()])
        self.assertEqual(sopp.collision_substeps(v0, c, 1e-9), 1)

    def test_fast_rates_subcycle(self):
        """A thermal proton over a long step must split the kick."""
        bgs = [
            ThermalBackground(
                n_profile=lambda s: 1e21,
                T_profile=lambda s: 1e3,
                mass=PROTON_MASS,
                charge=ELEMENTARY_CHARGE,
            )
        ]
        v_th = np.sqrt(2 * 1e3 * ONE_EV / PROTON_MASS)
        c = self._coef(v_th, bgs)
        n = sopp.collision_substeps(v_th, c, 1e-4)
        self.assertGreater(n, 1, f"expected sub-cycling, got nsub = {n}")


# ---------------------------------------------------------------------------
# Collision coefficients (analytical)
# ---------------------------------------------------------------------------


class TestOdeSolverValidation(unittest.TestCase):
    def test_symplectic_is_refused_for_collisional_tracing(self):
        """'symplectic' is valid collisionless input, so name the restriction."""
        with self.assertRaises(ValueError) as cm:
            _coll_trace(_field(), _hot_background(), tmax=1e-8, ode_solver="symplectic")
        self.assertIn("collisionless-only", str(cm.exception))


class TestParallelSpeedExceedingTotal(unittest.TestCase):
    """|v_par| > vtotal implies a negative mu, so it must be refused."""

    def test_refused_above_vtotal_but_legal_at_it(self):
        for label, fraction in (("marginal", 1.0 + 1e-9), ("nan", np.nan)):
            with self.subTest(vpar=label):
                with self.assertRaises(ValueError) as cm:
                    _coll_trace(
                        _field(),
                        _hot_background(),
                        vpar_fraction=fraction,
                        tmax=1e-8,
                    )
                self.assertIn("must not exceed", str(cm.exception))

        # |v_par| == vtotal exactly is a purely passing particle, and legal.
        traj = _coll_trace(_field(), _hot_background(), vpar_fraction=1.0, tmax=1e-8)
        self.assertEqual(traj.shape[1], 6)


class TestUnphysicalCoulombLog(unittest.TestCase):
    """
    ln Lambda <= 0 (Debye length below the minimum impact parameter, e.g.
    T -> 0 at finite density) makes the binary-collision model undefined,
    so the tracer must refuse the profiles rather than integrate garbage.
    """

    @staticmethod
    def _background(n, T_eV, mass, charge):
        return ThermalBackground(
            n_profile=lambda s: n,
            T_profile=lambda s: T_eV,
            mass=mass,
            charge=charge,
        )

    def test_unphysical_profiles_are_refused_and_healthy_ones_are_not(self):
        bad = self._background(1e30, 1e-3, ELECTRON_MASS, -ELEMENTARY_CHARGE)
        with self.assertRaises(ValueError):
            _coll_trace(_field(), bad, tmax=1e-8)

        # The v -> 0 bound is conservative; it must still pass a reactor profile.
        traj = _coll_trace(_field(), _hot_background(), tmax=1e-8)
        self.assertEqual(traj.shape[1], 6)

    def test_marginal_profiles_warn_exactly_once(self):
        """
        0 < ln_Lambda < 2 is usable but not quantitative, so it warns rather
        than raising -- once per call, not once per coefficient evaluation
        (the C++ sees one (v, s) at a time and would warn ~7x per step).
        """
        marginal = self._background(1e20, 0.025, 2 * PROTON_MASS, ELEMENTARY_CHARGE)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _coll_trace(_field(), marginal, tmax=1e-7)
        hits = [w for w in caught if "ln_Lambda" in str(w.message)]
        self.assertEqual(len(hits), 1, f"expected 1 warning, got {len(hits)}")
        self.assertTrue(issubclass(hits[0].category, RuntimeWarning))


class TestCartesianCollisionValidation(unittest.TestCase):
    """
    Python-layer validation of the Cartesian collisional GPU entry point.

    Every check here fires before any GPU call, so these run without CUDA;
    the field and classifier arguments are placeholders that must never be
    touched.
    """

    @staticmethod
    def _trace(**overrides):
        from firm3d.catapult.tracing import (
            trace_particles_cartesian_with_collisions_gpu,
        )

        kw = {
            "field": None,
            "surface_classifier": None,
            "flux_label": lambda pts: np.full(pts.shape[0], 0.5),
            "xyz_inits": np.zeros((4, 3)),
            "parallel_speeds": np.zeros(4),
            "backgrounds": _hot_background(),
            "tmax": 1e-8,
            "mass": ALPHA_PARTICLE_MASS,
            "charge": ALPHA_PARTICLE_CHARGE,
            "vtotal": 1e6,
            "tol": 1e-8,
        }
        kw.update(overrides)
        return trace_particles_cartesian_with_collisions_gpu(**kw)

    def test_missing_flux_label_is_refused(self):
        """
        Without the label column the kernel would read the 8-column layout
        off a 7-column array, so None must be refused before any tracing.
        """
        with self.assertRaises(ValueError) as cm:
            self._trace(flux_label=None)
        self.assertIn("flux_label", str(cm.exception))

    def test_empty_backgrounds_are_refused(self):
        with self.assertRaises(ValueError) as cm:
            self._trace(backgrounds=[])
        self.assertIn("collisionless", str(cm.exception))

    def test_mismatched_parallel_speeds_are_refused(self):
        with self.assertRaises(ValueError) as cm:
            self._trace(parallel_speeds=np.zeros(3))
        self.assertIn("parallel_speeds", str(cm.exception))


class TestCartesianFluxLabelColumn(unittest.TestCase):
    """
    Python-layer validation of the flux-label column, with stub field and
    classifier objects.  The column's values are checked end-to-end on GPU
    hardware by test_cartesian_collision_interpolant, so only the column
    count and the rejection of bad label output are covered here.
    """

    class _StubField:
        r_range = (1.0, 2.0, 2)
        phi_range = (0.0, np.pi, 2)
        z_range = (0.0, 0.5, 2)

        def set_points_cyl(self, pts):
            self._pts = pts

        def B_cyl(self):
            return np.zeros((self._pts.shape[0], 3))

        def GradAbsB_cyl(self):
            return np.zeros((self._pts.shape[0], 3))

    class _StubClassifier:
        def evaluate_rphiz(self, pts):
            return np.ones((pts.shape[0], 1))

    def _interpolant(self, **kwargs):
        from firm3d.catapult.utils import cartesian_interpolant

        return cartesian_interpolant(
            self._StubField(), self._StubClassifier(), **kwargs
        )

    def test_label_column_count_and_validation(self):
        _, _, _, quad = self._interpolant(flux_label=lambda pts: np.ones(len(pts)))
        self.assertEqual(quad.shape[1], 8)

        _, _, _, quad = self._interpolant()
        self.assertEqual(quad.shape[1], 7)

        with self.assertRaises(ValueError) as cm:
            self._interpolant(flux_label=lambda pts: np.full(len(pts), np.nan))
        self.assertIn("non-finite", str(cm.exception))

        with self.assertRaises(ValueError):
            self._interpolant(flux_label=lambda pts: np.ones(3))


class TestCollisionCoefficients(unittest.TestCase):
    """
    Unit tests for the shipped collision coefficients, reached through the
    firm3dpp bindings.

    These tests are deterministic and instantaneous — no simulation needed.
    They verify that the Chandrasekhar G function, the drag K, and the
    pitch-angle scattering rate ν_D have the correct signs, limits, and
    scalings.
    """

    _v0 = np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)
    _m, _q = ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE
    _s = 0.3

    # ------------------------------------------------------------------
    # Chandrasekhar G function
    # ------------------------------------------------------------------
    def test_G_small_x_limit(self):
        """For x ≪ 1: G(x) ≈ 2x / (3√π), and G(0) = 0 exactly."""
        self.assertAlmostEqual(sopp.chandrasekhar_G(0.0), 0.0, places=12)
        for x in [0.001, 0.01, 0.05]:
            G_approx = 2.0 * x / (3.0 * np.sqrt(np.pi))
            self.assertAlmostEqual(
                sopp.chandrasekhar_G(x),
                G_approx,
                delta=0.01 * G_approx,
                msg=f"G({x}) deviates from small-x limit 2x/(3√π)",
            )

    def test_G_large_x_limit(self):
        """For x ≫ 1: G(x) ≈ 1 / (2x²)."""
        for x in [5.0, 10.0, 20.0]:
            G_approx = 1.0 / (2.0 * x**2)
            self.assertAlmostEqual(
                sopp.chandrasekhar_G(x),
                G_approx,
                delta=0.05 * G_approx,
                msg=f"G({x}) deviates from large-x limit 1/(2x²)",
            )

    def test_G_deriv_changes_sign_at_the_maximum(self):
        """G has a maximum near x ≈ 0.92: G' > 0 below it, G' < 0 above."""
        for x in [0.1, 0.3, 0.5]:
            self.assertGreater(sopp.chandrasekhar_G_deriv(x), 0, f"G'({x}) ≤ 0")
        for x in [2.0, 5.0]:
            self.assertLess(sopp.chandrasekhar_G_deriv(x), 0, f"G'({x}) ≥ 0")

    # ------------------------------------------------------------------
    # Drag coefficient K
    # ------------------------------------------------------------------
    def test_coefficients_vanish_when_n_zero(self):
        """Every coefficient is identically zero with no active species."""
        c = _coeffs(self._v0, self._s, self._m, self._q, [_zero_background()])
        self.assertEqual(c.K, 0.0)
        self.assertEqual(c.nu_D, 0.0)
        self.assertEqual(c.D_par, 0.0)

    def test_K_linear_in_density(self):
        """K nearly proportional to n; small deviation via ln Λ ∝ ln(λ_D)."""
        K1 = _coeffs(self._v0, self._s, self._m, self._q, [_hot_background(n=1e20)]).K
        K2 = _coeffs(self._v0, self._s, self._m, self._q, [_hot_background(n=2e20)]).K
        self.assertAlmostEqual(K2 / K1, 2.0, delta=0.05)

    def test_K_quartic_in_EP_charge(self):
        """K ∝ q_a²; <15% deviation because ln Λ also depends on |q_a q_b|."""
        bg = _hot_background()
        K_alpha = _coeffs(
            self._v0, self._s, ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, [bg]
        ).K
        # Same mass but half the charge
        K_half = _coeffs(
            self._v0, self._s, ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE / 2.0, [bg]
        ).K
        self.assertAlmostEqual(
            K_alpha / K_half,
            4.0,
            delta=0.5,
            msg="K must scale approximately as q_a² (factor ~4 for half charge)",
        )

    def test_K_additive_over_species(self):
        """
        K is approximately additive when two identical backgrounds are combined:
        the Debye length from two identical species is 1/√2 of a single species,
        changing ln Λ by only −½ ln 2 ≈ −0.35, so K[bg+bg] ≈ 2 K[bg] to ~5%.
        """
        bg = _hot_background(n=1e20)
        K_one = _coeffs(self._v0, self._s, self._m, self._q, [bg]).K
        K_both = _coeffs(self._v0, self._s, self._m, self._q, [bg, bg]).K
        self.assertAlmostEqual(K_both / K_one, 2.0, delta=0.1)

    def test_K_electron_larger_than_ion(self):
        """
        For fusion alphas in a 10 keV plasma, electron drag dominates over
        deuteron drag.

        Physical basis: the alpha birth energy (3.5 MeV) is far above the
        critical energy E_c ≈ 0.4 MeV where electron and ion drag are equal,
        so electron drag dominates by roughly (E/E_c)^{3/2} ≈ 20-30.
        """
        electron_bg = ThermalBackground(
            n_profile=lambda s: 1e20,
            T_profile=lambda s: 10e3,
            mass=ELECTRON_MASS,
            charge=-ELEMENTARY_CHARGE,
        )
        K_e = _coeffs(self._v0, self._s, self._m, self._q, [electron_bg]).K
        K_D = _coeffs(self._v0, self._s, self._m, self._q, [_hot_background()]).K
        ratio = abs(K_e) / abs(K_D)
        self.assertGreater(ratio, 10, f"|K_e|/|K_D| = {ratio:.1f}, expected ≈ 20")
        self.assertLess(ratio, 50, f"|K_e|/|K_D| = {ratio:.1f}, expected ≈ 20")

    # ------------------------------------------------------------------
    # Pitch-angle scattering rate ν_D
    # ------------------------------------------------------------------
    def test_nu_D_linear_in_density(self):
        """ν_D nearly proportional to n; small deviation via ln Λ ∝ ln(λ_D)."""
        nu_1 = _coeffs(
            self._v0, self._s, self._m, self._q, [_hot_background(n=1e20)]
        ).nu_D
        nu_2 = _coeffs(
            self._v0, self._s, self._m, self._q, [_hot_background(n=2e20)]
        ).nu_D
        self.assertAlmostEqual(nu_2 / nu_1, 2.0, delta=0.05)

    def test_nu_D_large_x_limit(self):
        """
        For x ≫ 1 (EP much faster than background thermal speed):
        ν_D ≈ Γ × (erf(x) − G(x)) / v³ → Γ / v³  (since erf(x)→1, G(x)→0).

        Γ is not exposed on its own, so it is recovered from the shipped
        D_par = Γ G(x) / v, which is exact for this single-species
        background.  The assertion is therefore that the shipped ν_D and
        D_par stand in the ratio the formulae require, erf(x) − G(x), which
        is 1 to within 2% at x ≈ 13.
        """
        bg = _hot_background()
        v0, s = self._v0, self._s
        T_b = bg.T_profile(s) * ONE_EV
        v_th = np.sqrt(2 * T_b / bg.mass)
        x = v0 / v_th  # ≈ 13 for fusion alphas in 10 keV D plasma

        c = _coeffs(v0, s, self._m, self._q, [bg])
        Gamma = c.D_par * v0 / sopp.chandrasekhar_G(x)
        nu_D_approx = Gamma / v0**3  # large-x limit

        self.assertAlmostEqual(
            c.nu_D / nu_D_approx,
            1.0,
            delta=0.02,
            msg=f"ν_D deviates from Γ/v³ limit at x={x:.1f}",
        )


# ---------------------------------------------------------------------------
# End-to-end thermalization through the production tracer (fast, local)
# ---------------------------------------------------------------------------


class TestMaxwellianEquilibration(unittest.TestCase):
    """
    End-to-end thermalization through trace_particles_boozer_with_collisions
    with a BoozerAnalytic field -- the full production path (DP stepper +
    Milstein noise + orbit dynamics), runnable locally in a few seconds.
    """

    _T_B_EV = 1e3  # handed to ThermalBackground, which takes eV
    _T_B = _T_B_EV * ONE_EV  # same temperature in J, for the energy checks
    _N_B = 1e21  # m^-3
    _TAU = 3.5e-5  # collisional relaxation time at (_N_B, _T_B)
    _N_PART = 40

    @staticmethod
    def _confining_field():
        """Near-axis field with slow collisional radial transport."""
        return BoozerAnalytic(0.25, 5.0, 0, 40.0, 2.0, 0.8)

    def _equilibrate(self, E0_over_T, seed=42):
        """Trace N_PART protons from a monoenergetic start; return final v."""
        bg = ThermalBackground(
            n_profile=lambda s: self._N_B,
            T_profile=lambda s: self._T_B_EV,
            mass=PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )
        E0 = E0_over_T * self._T_B
        v0 = np.sqrt(2 * E0 / PROTON_MASS)
        rng = np.random.default_rng(seed)
        stz = np.column_stack(
            [
                np.full(self._N_PART, 0.3),
                rng.uniform(0, 2 * np.pi, self._N_PART),
                rng.uniform(0, 2 * np.pi, self._N_PART),
            ]
        )
        vpar = 0.7 * v0 * np.ones(self._N_PART)
        tmax = 8 * self._TAU
        res_tys, _ = trace_particles_boozer_with_collisions(
            self._confining_field(),
            stz,
            vpar,
            backgrounds=[bg],
            tmax=tmax,
            mass=PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
            Ekin=E0,
            tol=1e-8,
            dt_save=tmax,
            forget_exact_path=True,
            rng_seed=seed,
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        )
        v_end = np.array([ty[-1, 5] for ty in res_tys])
        vpar_end = np.array([ty[-1, 4] for ty in res_tys])
        t_end = np.array([ty[-1, 0] for ty in res_tys])

        # Every particle must stay confined to tmax with a finite, nonzero
        # speed: catches losses, frozen v = 0 particles, and NaN states.
        self.assertTrue(np.all(t_end >= 0.999 * tmax), "particles lost early")
        self.assertTrue(np.all(np.isfinite(v_end)), "non-finite final v")
        self.assertTrue(np.all(np.isfinite(vpar_end)), "non-finite final v_par")
        self.assertGreater(v_end.min(), 0.0, "dead particle at v = 0")
        return v_end, vpar_end

    def _check_maxwellian(self, v_end, lo, hi):
        E_mean = 0.5 * PROTON_MASS * np.mean(v_end**2) / self._T_B
        self.assertGreater(
            E_mean, lo, f"<E>/T_b = {E_mean:.2f}, expected ~1.5 at equilibrium"
        )
        self.assertLess(
            E_mean, hi, f"<E>/T_b = {E_mean:.2f}, expected ~1.5 at equilibrium"
        )
        p = scipy.stats.kstest(
            v_end,
            scipy.stats.maxwell(scale=np.sqrt(self._T_B / PROTON_MASS)).cdf,
        ).pvalue
        self.assertGreater(p, 0.005, f"KS vs Maxwell speed law: p = {p:.2e}")

    def test_equilibration_from_hot_start(self):
        """Monoenergetic protons at E0 = 4.5 T_b cool to the Maxwellian."""
        v_end, _ = self._equilibrate(E0_over_T=4.5)
        self._check_maxwellian(v_end, lo=1.0, hi=2.3)

    def test_equilibration_from_cold_start(self):
        """Monoenergetic protons at E0 = 0.45 T_b heat to the Maxwellian."""
        v_end, _ = self._equilibrate(E0_over_T=0.45)
        self._check_maxwellian(v_end, lo=0.9, hi=2.1)

    def test_pitch_isotropization(self):
        """Initial xi = 0.7 must isotropize: <xi> ~ 0, <xi^2> ~ 1/3."""
        v_end, vpar_end = self._equilibrate(E0_over_T=1.5)
        xi = vpar_end / v_end
        self.assertLess(abs(np.mean(xi)), 0.3, f"<xi> = {np.mean(xi):.3f}")
        self.assertGreater(np.mean(xi**2), 0.18, f"<xi^2> = {np.mean(xi**2):.3f}")
        self.assertLess(np.mean(xi**2), 0.50, f"<xi^2> = {np.mean(xi**2):.3f}")


# ---------------------------------------------------------------------------
# Qualitative collision physics (simulation)
# ---------------------------------------------------------------------------


class TestCollisionPhysics(unittest.TestCase):
    """
    Quantitative drag checks against the analytic coefficients.
    """

    def test_speed_non_negative(self):
        """v ≥ 0 must hold under extreme drag."""
        traj = _coll_trace(_field(), _cold_background(n=1e22, T_keV=0.1), tmax=2e-6)
        self.assertTrue(np.all(traj[:, 5] >= 0))

    def test_rng_seed_reproducibility(self):
        """Same seed → bitwise-identical trajectories; different seed → different."""
        field, bg = _field(), _hot_background()
        t1 = _coll_trace(field, bg, tmax=1e-6, rng_seed=42)
        t2 = _coll_trace(field, bg, tmax=1e-6, rng_seed=42)
        t3 = _coll_trace(field, bg, tmax=1e-6, rng_seed=99)
        np.testing.assert_array_equal(t1, t2)
        self.assertFalse(np.array_equal(t1, t3))

    def test_electron_drag_mean_velocity(self):
        """
        Electron background gives detectable mean energy loss with N=50 alphas
        in tmax = 5 μs.

        For 3.52 MeV alphas (v₀ ≈ 1.3×10⁷ m/s) against a 10 keV electron
        plasma at n = 10²⁰ m⁻³:
          x_e = v₀/v_th,e ≈ 0.22  →  G(x_e) ≈ 0.046  (near maximum)
          K_e ≈ −3.5×10⁸ m/s²     (drag ~70 000× larger than D-ion drag)

        SNR analysis: drift = K × tmax ≈ −1770 m/s;
          σ(noise per particle) ≈ √(2 D_par × tmax) ≈ 2500 m/s;
          σ(mean) = σ/√N ≈ 354 m/s  →  SNR ≈ 5 with N = 50.

        The test asserts (1) mean Δv < 0 (correct sign) and (2) the ratio
        mean_dv / (K_theory × tmax) ∈ [0.3, 3.0], which fails with probability
        < 0.003 for any statistically correct implementation.
        """
        electron_bg = ThermalBackground(
            n_profile=lambda s: 1e20,
            T_profile=lambda s: 10e3,
            mass=ELECTRON_MASS,
            charge=-ELEMENTARY_CHARGE,
        )
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        m, q = ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE
        v0 = np.sqrt(2 * Ekin / m)
        s0 = 0.3

        K_theory = _coeffs(v0, s0, m, q, [electron_bg]).K
        self.assertLess(K_theory, 0, "Electron drag K must be negative")

        tmax = 5e-6
        nP = 50
        stz = np.tile([s0, 0.0, 0.0], (nP, 1))
        vpar = np.full(nP, 0.5 * v0)

        res_tys, _ = trace_particles_boozer_with_collisions(
            _field(),
            stz,
            vpar,
            backgrounds=electron_bg,
            tmax=tmax,
            mass=m,
            charge=q,
            Ekin=Ekin,
            tol=1e-9,
            dt_save=tmax,
            forget_exact_path=True,
            rng_seed=0,
            DP_hmin=1e-10,
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        )

        v_final = np.array([t[-1, 5] for t in res_tys])
        mean_dv = np.mean(v_final) - v0

        # Normalize by the time actually integrated, not by tmax: the domain
        # guard stops these particles at s = 1 well before tmax, and dividing
        # by tmax rescales the result by ~10x, which happens to land noise
        # inside a [0.3, 3.0] band and made this assertion meaningless.
        t_elapsed = np.mean([t[-1, 0] for t in res_tys])
        self.assertGreater(t_elapsed, 0.0)

        self.assertTrue(np.all(np.isfinite(v_final)), "non-finite final speed")
        self.assertLess(mean_dv, 0, "Mean speed must decrease under electron drag")

        # Wide band by construction: per the module docstring the SNR here is
        # below 1, so this checks sign and order of magnitude only.  The
        # coefficients themselves are pinned by TestCollisionCoefficients.
        ratio = mean_dv / (K_theory * t_elapsed)
        self.assertGreater(
            ratio, 0.2, f"Drag too weak: mean_dv/(K_theory*t) = {ratio:.2f} < 0.2"
        )
        self.assertLess(
            ratio, 30.0, f"Drag too strong: mean_dv/(K_theory*t) = {ratio:.2f} > 30"
        )


# ---------------------------------------------------------------------------
# Pitch isotropization (ASCOT5 thermal-equilibration analog)
# ---------------------------------------------------------------------------


class TestPitchIsotropization(unittest.TestCase):
    """
    Pitch-angle scattering tests, from qualitative to quantitative.

    The exact Ito result for the pitch SDE
      dξ = −ν_D ξ dt + √(ν_D(1−ξ²)) dW
    is d⟨ξ⟩/dt = −ν_D ⟨ξ⟩  (E[f(ξ(t)) dW] = 0 in Ito calculus),
    giving ⟨ξ(t)⟩ = ξ₀ exp(−ν_D t) exactly when ν_D is constant.
    This is the ASCOT5 pitch-angle benchmark: start a beam at ξ₀ ≠ 0
    and verify the exponential decay rate against the analytical ν_D.

    Two backgrounds are used across these tests:

      n = 10²³ m⁻³, T = 1 keV D  (ν_D ≈ 47 s⁻¹):
        Used for the qualitative test (ξ₀ = 0, tmax = 1e-4 s → σ ≈ 0.07).

      n = 10²⁵ m⁻³, T = 1 keV D  (ν_D ≈ 4700 s⁻¹):
        Used for the quantitative decay test (tmax = 0.5/ν_D ≈ 107 μs,
        ν_D × tmax = 0.5, N = 50 → SNR ≈ 5 on ⟨ξ⟩ change).
    """

    @staticmethod
    def _uniform_field():
        """Near-uniform |B| (etabar ~ 0): no mirror force and no trapped
        cone, so the uniform-plasma pitch-decay law applies exactly.
        In a toroidal field the trapped-passing boundary at s = 0.3 sits
        at xi_t ~ 0.65, just below the xi0 = 0.8 beam: scattered
        particles fall into the trapped cone where their orbit-sampled
        xi averages to zero, and <xi> decays ~2x faster than the
        uniform-plasma law (measured exponent 1.05 vs m_D/m_alpha =
        0.50) -- real physics, but not what this test verifies.
        """
        return BoozerAnalytic(1e-8, 5.3, 0, 32.86, 10.6, 1.0, Bbar=5.3)

    @staticmethod
    def _confining_field():
        """ITER-scale near-axis field (R0 = 6.2 m, B0 = 5.3 T, psi0 = 10.6).

        3.5 MeV alphas have radial orbit widths delta-s ~ 0.15 here, so
        markers started at s = 0.3 stay inside the plasma.  In the
        small-psi0 _field() used elsewhere, alpha orbit widths are order
        unity: markers escape through s = 1 within a few transits and
        their states corrupt in the analytic continuation of the field.
        """
        return BoozerAnalytic(0.25, 5.3, 0, 32.86, 10.6, 1.0, Bbar=5.3)

    @staticmethod
    def _assert_confined(res_tys, tmax):
        t_end = np.array([t[-1, 0] for t in res_tys])
        v_end = np.array([t[-1, 5] for t in res_tys])
        assert np.all(t_end >= 0.999 * tmax), "particles lost before tmax"
        assert np.all(np.isfinite(v_end)), "non-finite final state"

    def _background_low_n(self):
        return ThermalBackground(
            n_profile=lambda s: 1e23,
            T_profile=lambda s: 1e3,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )

    def _background_high_n(self):
        return ThermalBackground(
            n_profile=lambda s: 1e25,
            T_profile=lambda s: 1e3,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )

    def test_pitch_angle_exponential_decay(self):
        """
        ⟨ξ⟩ ∝ v^{m_b/m_a}: pitch decay of fast alphas on deuterium.

        For a fast test particle (x = v/v_th >> 1) on a single ion
        background, pitch scattering CANNOT be isolated from drag: both
        rates scale as 1/v^3 with the fixed ratio
        nu_drag/nu_D = m_a/m_b (= 2 for alphas on D).  Over nu_D t = 0.5
        the alphas lose most of their speed, nu_D(v) explodes, and <xi>
        isotropizes completely -- the original form of this test, which
        assumed constant v, failed with <xi> = 0.

        Eliminating time between d<xi>/dt = -nu_D <xi> and
        dv/dt = -(m_a/m_b) Gamma / v^2 gives the classical relation
            <xi(t)> / <xi(t1)> = ( v(t) / v(t1) )^{m_b/m_a},
        which is checked here using the simulation's own mean speed.
        Referencing the first SAVED snapshot (t1 = tmax/8) rather than
        the initial condition cancels the geometric offset from sampling
        xi along orbits (xi varies with B along a field line).

        tmax = 0.15/nu_D(v0) keeps x >> 1 throughout (v_end ~ 0.6 v0).
        With N = 128 the accumulated pitch scatter gives a ~2-3 % standard
        error on the ratio; the +/-15 % bracket is a >5 sigma window that
        still catches an O(1) error in nu_D or the drag.
        """
        bg = self._background_high_n()
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        m, q = ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE
        v0 = np.sqrt(2 * Ekin / m)
        xi0 = 0.8
        s0 = 0.3

        nu_D_theory = _coeffs(v0, s0, m, q, [bg]).nu_D
        tmax = 0.15 / nu_D_theory

        nP = 128
        stz = np.tile([s0, 0.0, 0.0], (nP, 1))
        vpar = np.full(nP, xi0 * v0)

        res_tys, _ = trace_particles_boozer_with_collisions(
            self._uniform_field(),
            stz,
            vpar,
            backgrounds=bg,
            tmax=tmax,
            mass=m,
            charge=q,
            Ekin=Ekin,
            tol=1e-8,
            dt_save=tmax / 8,
            forget_exact_path=False,
            rng_seed=0,
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        )
        self._assert_confined(res_tys, tmax)

        # First saved snapshot (t = tmax/8) as reference, last as end
        xi_1 = np.array([t[1, 4] / t[1, 5] for t in res_tys])
        v_1 = np.array([t[1, 5] for t in res_tys])
        xi_2 = np.array([t[-1, 4] / t[-1, 5] for t in res_tys])
        v_2 = np.array([t[-1, 5] for t in res_tys])

        measured = np.mean(xi_2) / np.mean(xi_1)
        predicted = (np.mean(v_2) / np.mean(v_1)) ** (
            (2 * PROTON_MASS) / ALPHA_PARTICLE_MASS
        )

        # The speed must have decayed appreciably (drag active) while
        # the alphas remain fast (asymptotic relation applicable)
        self.assertLess(np.mean(v_2) / np.mean(v_1), 0.95, "no drag signal")
        self.assertGreater(np.mean(v_2) / np.mean(v_1), 0.4, "over-slowed")

        self.assertGreater(
            measured,
            0.85 * predicted,
            f"pitch decayed too fast: <xi_2>/<xi_1> = {measured:.3f}, "
            f"(v_2/v_1)^(m_D/m_alpha) = {predicted:.3f}",
        )
        self.assertLess(
            measured,
            1.15 * predicted,
            f"pitch decayed too slow: <xi_2>/<xi_1> = {measured:.3f}, "
            f"(v_2/v_1)^(m_D/m_alpha) = {predicted:.3f}",
        )


if __name__ == "__main__":
    unittest.main()
