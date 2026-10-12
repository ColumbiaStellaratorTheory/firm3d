"""Continuous CPU tracing limits at dense roots and per-particle horizons."""

import unittest
from unittest.mock import patch

import firm3dpp
import numpy as np

from firm3d.field.tracing import (
    IterationStoppingCriterion,
    MaxToroidalFluxStoppingCriterion,
    trace_particles_boozer,
)
from tests.field.test_catapult_poincare import constant_field, SPEED, PERIOD
from firm3d.field.boozermagneticfield import BoozerAnalytic


class TestCPUPhaseLimitValidation(unittest.TestCase):
    def test_invalid_limits_and_horizons(self):
        invalid = [
            {"max_phase_hits": -1},
            {"max_phase_hits": True},
            {"max_phase_hits": 1.5},
            {"max_phase_hits": 2**31},
            {"max_phase_interval": 0},
            {"max_phase_interval": np.inf},
            {"max_phase_interval": [1]},
            {"tmax": -1},
            {"tmax": np.nan},
            {"tmax": [1, 2]},
            {"max_phase_hits": 1, "phases": []},
            {"max_phase_hits": 1, "ODE_solver": "symplectic"},
        ]
        for change in invalid:
            options = {"phases": [0], "n_zetas": [1], "m_thetas": [0], "omegas": [0]}
            options.update(change)
            with (
                self.subTest(change=change),
                patch(
                    "firm3d.field.tracing.sopp.particle_guiding_center_boozer_tracing"
                ) as native,
            ):
                with self.assertRaises(ValueError):
                    trace_particles_boozer(
                        constant_field(), np.array([[0.3, 0.2, 0]]), [SPEED], **options
                    )
                native.assert_not_called()


@unittest.skipUnless(
    "max_phase_hits" in firm3dpp.particle_guiding_center_boozer_tracing.__doc__,
    "requires continuous CPU phase-limit bindings",
)
class TestCPUPhaseLimits(unittest.TestCase):
    def trace(self, field=None, initial=None, speeds=None, **kwargs):
        options = {
            "tmax": 1e-4,
            "phases": [0],
            "n_zetas": [1],
            "m_thetas": [0],
            "omegas": [0],
            "max_phase_hits": 4,
            "max_phase_interval": 1e-4,
            "vpars": [0],
            "vpars_stop": True,
            "axis": 0,
            "tol": 1e-11,
            "dt_save": 1e-7,
            "forget_exact_path": False,
        }
        options.update(kwargs)
        return trace_particles_boozer(
            field or constant_field(),
            np.array([[0.3, 0.2, 0]]) if initial is None else initial,
            [SPEED] if speeds is None else speeds,
            **options,
        )

    def test_count_and_history_end_at_root_for_both_directions_and_solvers(self):
        field = constant_field()
        field.set_points(np.array([[0.3, 0.2, 0]]))
        period = PERIOD * (25 + 0.4 * 0.2) / (5 * SPEED)
        for sign in [1, -1]:
            for solver in ["boost", "dormand_prince"]:
                for forget in [False, True]:
                    with self.subTest(sign=sign, solver=solver, forget=forget):
                        paths, hits = self.trace(
                            speeds=[sign * SPEED],
                            ODE_solver=solver,
                            forget_exact_path=forget,
                        )
                        self.assertEqual(hits[0].shape, (4, 6))
                        np.testing.assert_allclose(
                            hits[0][:, 0], period * np.arange(1, 5), rtol=2e-10
                        )
                        np.testing.assert_allclose(
                            hits[0][:, 4], sign * PERIOD * np.arange(1, 5), atol=1e-12
                        )
                        np.testing.assert_allclose(
                            paths[0][-1],
                            hits[0][-1, [0, 2, 3, 4, 5]],
                            rtol=2e-15,
                            atol=1e-14,
                        )
                        self.assertTrue(np.all(np.diff(paths[0][:, 0]) > 0))
                        if not forget:
                            np.testing.assert_allclose(
                                paths[0][1:-1, 0],
                                1e-7 * np.arange(1, len(paths[0]) - 1),
                                atol=1e-20,
                            )

    def test_deadline_and_global_horizon_discard_later_roots(self):
        paths, hits = self.trace(max_phase_interval=1e-9)
        self.assertEqual(np.asarray(hits[0]).size, 0)
        self.assertAlmostEqual(paths[0][-1, 0], 1e-9, delta=1e-24)
        _, full = self.trace()
        horizon = (full[0][0, 0] + full[0][1, 0]) / 2
        paths, hits = self.trace(tmax=horizon)
        self.assertEqual(len(hits[0]), 1)
        self.assertAlmostEqual(paths[0][-1, 0], horizon, delta=1e-20)

    def test_crossing_exactly_at_deadline_and_horizon(self):
        for sign in [1, -1]:
            with self.subTest(sign=sign):
                _, reference = self.trace(speeds=[sign * SPEED], max_phase_hits=1)
                time = reference[0][0, 0]
                for limits in [{"max_phase_interval": time}, {"tmax": time}]:
                    paths, hits = self.trace(
                        speeds=[sign * SPEED], max_phase_hits=1, **limits
                    )
                    self.assertEqual(len(hits[0]), 1)
                    self.assertAlmostEqual(paths[0][-1, 0], time, delta=1e-20)

    def test_deadline_resets_after_each_crossing(self):
        # Two phase planes give alternating short and long intervals. The
        # first completes; the second times out after a fresh interval.
        _, reference = self.trace(
            phases=[0, 1],
            n_zetas=[1, 1],
            m_thetas=[0, 0],
            omegas=[0, 0],
            max_phase_hits=1,
        )
        first = reference[0][0, 0]
        interval = 1.5 * first
        paths, hits = self.trace(
            phases=[0, 1],
            n_zetas=[1, 1],
            m_thetas=[0, 0],
            omegas=[0, 0],
            max_phase_interval=interval,
        )
        self.assertEqual(len(hits[0]), 1)
        self.assertEqual(hits[0][0, 1], 1)
        self.assertAlmostEqual(paths[0][-1, 0], first + interval, delta=1e-20)

    def test_per_particle_horizons_including_zero(self):
        ends = np.array([0, 1e-9, 1e-7])
        paths, hits = self.trace(
            initial=np.tile([0.3, 0.2, 0], (3, 1)), speeds=[SPEED] * 3, tmax=ends
        )
        np.testing.assert_allclose([path[-1, 0] for path in paths], ends, atol=1e-22)
        self.assertEqual(len(paths[0]), 1)
        self.assertTrue(all(np.asarray(particle).size == 0 for particle in hits))

    def test_velocity_root_stops_before_return(self):
        field = BoozerAnalytic(etabar=0.03, B0=5, N=0, G0=25, psi0=100, iota0=0.4)
        paths, hits = self.trace(
            field, initial=np.array([[0.3, np.pi + 0.2, 0]]), speeds=[0.15 * SPEED]
        )
        self.assertEqual(hits[0][-1, 1], 1)
        self.assertAlmostEqual(hits[0][-1, 5], 0, delta=1e-7)
        self.assertTrue(np.all(np.diff(hits[0][:, 0]) >= 0))
        self.assertEqual(paths[0][-1, 0], hits[0][-1, 0])

    def test_public_custom_solver_minimum_step(self):
        # DP_hmin is forwarded in the solver's normalized time coordinates.
        # An iteration stop exposes the first accepted physical step time.
        tnorm = PERIOD * 25 / (5 * SPEED)
        paths, hits = self.trace(
            ODE_solver="dormand_prince",
            DP_hmin=0.08,
            axis=2,
            tol=1e-14,
            stopping_criteria=[IterationStoppingCriterion(0)],
        )
        self.assertEqual(hits[0][0, 1], -1)
        self.assertAlmostEqual(hits[0][0, 0], 0.08 * tnorm, delta=1e-21)
        self.assertEqual(paths[0][-1, 0], hits[0][0, 0])

    def test_flux_stop_retains_criterion_index(self):
        paths, hits = self.trace(
            initial=np.array([[0.995, 0.2, 0]]),
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(0.99)],
        )
        self.assertEqual(len(hits[0]), 1)
        self.assertEqual(hits[0][0, 1], -1)
        self.assertEqual(paths[0][-1, 0], hits[0][0, 0])


if __name__ == "__main__":
    unittest.main()
