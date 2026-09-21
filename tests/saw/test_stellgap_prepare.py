"""Preparation reports, bounded pilot work, cache reuse, and timing projections."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from scipy.interpolate import make_interp_spline

from firm3d.field.boozermagneticfield import BoozerRadialInterpolant
from firm3d.saw.stellgap import Continuum


EQUILIBRIUM_FILE = (
    Path(__file__).parents[1] / "test_files" / "boozmn_n3are_R7.75B5.7.nc"
)


class TestContinuumPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.field = BoozerRadialInterpolant(str(EQUILIBRIUM_FILE), order=3, no_K=True)

    def make_continuum(
        self, surfaces=(0.8, 0.1, 0.6, 0.3, 0.6, 0.4, 0.8), density=None
    ):
        s = np.linspace(0, 1, 7)
        radius = np.column_stack((np.full_like(s, 3), s))
        height = np.column_stack((np.zeros_like(s), s))
        with patch.multiple(
            self.field, xm_b=np.array([0, 1]), xn_b=np.array([0, 3]),
            rmnc_splines=make_interp_spline(s, radius),
            zmns_splines=make_interp_spline(s, height),
            numns_splines=make_interp_spline(s, np.zeros_like(radius)),
            iota_spline=make_interp_spline(s, np.full_like(s, 0.5)),
            psi0=-0.8,
        ):
            return Continuum(
                self.field, surfaces, [[0, 0], [1, 0], [2, 3]], density,
                reference_field=2.0,
            )

    def test_default_checks_one_surface_and_reports_memory_without_solving(self):
        continuum = self.make_continuum()
        with patch.object(continuum, "_solve_surface") as solver:
            report = continuum.prepare(max_shape=(128, 128), keep_eigenvectors=True)
            solver.assert_not_called()
        self.assertEqual(report["matrix_shape"], (3, 3))
        self.assertEqual(report["surface_count"], 7)
        self.assertEqual([entry["surface"] for entry in report["pilots"]], [0.4])
        self.assertFalse(report["pilots"][0]["solved"])
        self.assertIsNone(report["timing"]["estimates"])
        self.assertGreater(report["timing"]["preparation_seconds"], 0)
        self.assertEqual(report["equilibrium_support"]["n_range"], (0, 3))
        self.assertEqual(report["reference_field"]["source"], "explicit")
        self.assertEqual(report["convergence"]["unchecked_surfaces"], [.1, .3, .6, .8])
        self.assertEqual(report["convergence"]["perturbation_basis"], "unverified")
        memory = report["memory"]["at_pilot_grids"]
        components = memory["components"]
        self.assertEqual(components["basis_tables"], 34 * 3**2 + 8 * 3)
        self.assertEqual(components["two_matrices"], 16 * 3**2)
        self.assertEqual(components["surface_eigenvectors"], 8 * 3**2)
        self.assertEqual(components["retained_output_eigenvectors"], 8 * 7 * 3**2)
        self.assertEqual(components["pilot_cache"], 16 * 3**2)
        self.assertGreater(memory["peak_bytes"], components["two_matrices"])
        limit = report["memory"]["at_grid_limit"]
        self.assertEqual(limit["components"]["geometry_and_fft_buffers"], 576 * 128**2)
        self.assertGreaterEqual(limit["peak_bytes"], memory["peak_bytes"])

    def test_repeated_prepare_reuses_basis_and_quadrature_and_returns_copies(self):
        continuum = self.make_continuum()
        first = continuum.prepare()
        with patch.object(continuum, "_plan_basis") as basis:
            with patch.object(continuum, "_converge_surface") as quadrature:
                second = continuum.prepare()
                basis.assert_not_called()
                quadrature.assert_not_called()
        first["normalization"]["frequency_factors"][0] = -1
        first["pilots"][0]["quadrature"]["converged"] = False
        self.assertGreater(second["normalization"]["frequency_factors"][0], 0)
        self.assertTrue(second["pilots"][0]["quadrature"]["converged"])
        self.assertTrue(continuum.prepare()["pilots"][0]["quadrature"]["converged"])

    def test_benchmark_upgrades_cached_matrices_and_bounds_pilot_union(self):
        continuum = self.make_continuum(surfaces=(.1, .3, .6, .8))
        continuum.prepare()
        with patch.object(
            continuum, "_converge_surface", wraps=continuum._converge_surface
        ) as quadrature:
            report = continuum.prepare(benchmark=True, pilot_count=2)
            self.assertEqual(quadrature.call_count, 2)
            continuum.prepare(benchmark=True, pilot_count=3)
            self.assertEqual(quadrature.call_count, 2)
        self.assertEqual(len(continuum._preparation["pilots"]), 3)
        report = continuum.prepare(benchmark=True)
        self.assertTrue(all(pilot["solved"] for pilot in report["pilots"]))
        with patch.object(continuum, "_solve_surface") as solver:
            continuum.prepare(benchmark=True)
            solver.assert_not_called()
        for entry in continuum._preparation["pilots"].values():
            self.assertIsNone(entry["matrices"])
            self.assertIsNone(entry["solution"]["eigenvectors"])

    def test_benchmarked_run_solves_only_remaining_surfaces_and_matches_fresh_run(self):
        continuum = self.make_continuum()
        continuum.prepare(benchmark=True, keep_eigenvectors=True)
        with patch.object(
            continuum, "_solve_surface", wraps=continuum._solve_surface
        ) as solver:
            result = continuum.run(keep_eigenvectors=True)
        self.assertEqual([call.args[0] for call in solver.call_args_list], [.6, .3, .6])
        fresh = self.make_continuum().run(keep_eigenvectors=True)
        for name in ("surfaces", "modes", "eigenvalues", "normalized_frequencies",
                     "dominant_modes"):
            np.testing.assert_array_equal(getattr(result, name), getattr(fresh, name))
        np.testing.assert_allclose(result.eigenvectors, fresh.eigenvectors, atol=1e-13)
        result.diagnostics[0]["solver"]["scaled_residuals"][:] = -1
        result.eigenvectors[0] = 0
        again = continuum.run(keep_eigenvectors=True)
        self.assertTrue(np.all(again.diagnostics[0]["solver"]["scaled_residuals"] >= 0))
        np.testing.assert_allclose(again.eigenvectors, fresh.eigenvectors, atol=1e-13)

    def test_runtime_projection_uses_measured_costs_and_cyclic_workload(self):
        continuum = self.make_continuum()
        report = continuum.prepare(benchmark=True, mpi_ranks=2)
        estimates = report["timing"]["estimates"]
        mean = estimates["surface_seconds_min_mean_max"][1]
        self.assertAlmostEqual(estimates["serial_remaining_seconds"], 3 * mean)
        # Remaining indices 2, 3, 4 give two tasks on rank 0 and one on rank 1.
        self.assertAlmostEqual(estimates["projected_mpi_remaining_seconds"], 2 * mean)
        self.assertEqual(estimates["remaining_surface_count"], 3)
        self.assertTrue(estimates["projection_only"])
        for pilot in report["pilots"]:
            seconds = pilot["seconds"]
            self.assertGreater(seconds["geometry"], 0)
            self.assertGreater(seconds["fourier_moments"], 0)
            self.assertGreater(seconds["assembly"], 0)
            self.assertGreater(seconds["solve_and_checks"], 0)
            subtotal = sum(seconds[name] for name in (
                "geometry", "fourier_moments", "assembly", "planning_and_comparison"
            ))
            self.assertAlmostEqual(subtotal, seconds["quadrature_total"])
        many = continuum.prepare(benchmark=True, mpi_ranks=10)["timing"]["estimates"]
        self.assertEqual(many["projected_mpi_active_ranks"], 3)
        self.assertAlmostEqual(many["projected_mpi_remaining_seconds"], mean)

    def test_one_unique_surface_can_be_fully_reused(self):
        continuum = self.make_continuum(surfaces=(.4, .4, .4))
        report = continuum.prepare(benchmark=True)
        self.assertEqual(len(report["pilots"]), 1)
        self.assertEqual(report["timing"]["estimates"]["serial_remaining_seconds"], 0)
        self.assertEqual(report["convergence"]["unchecked_surfaces"], [])
        with patch.object(continuum, "_converge_surface") as quadrature:
            with patch.object(continuum, "_solve_surface") as solver:
                result = continuum.run()
                quadrature.assert_not_called()
                solver.assert_not_called()
        np.testing.assert_array_equal(result.eigenvalues[0], result.eigenvalues[2])

    def test_changed_settings_and_mode_order_invalidate_pilot_work(self):
        continuum = self.make_continuum()
        continuum.prepare(benchmark=True)
        with patch.object(
            continuum, "_converge_surface", wraps=continuum._converge_surface
        ) as quadrature:
            continuum.prepare(rtol=1e-7)
            self.assertEqual(quadrature.call_count, 1)
            continuum.prepare(rtol=1e-7, keep_eigenvectors=True)
            self.assertEqual(quadrature.call_count, 2)
            continuum.modes[:] = continuum.modes[::-1]
            continuum.prepare(rtol=1e-7, keep_eigenvectors=True)
            self.assertEqual(quadrature.call_count, 3)

    def test_density_is_rechecked_without_discarding_density_independent_pilots(self):
        density = Mock(return_value=1e-7)
        continuum = self.make_continuum(density=density)
        continuum.prepare(benchmark=True)
        density.reset_mock()
        density.return_value = 4e-7
        with patch.object(continuum, "_converge_surface") as quadrature:
            continuum.prepare(benchmark=True)
            quadrature.assert_not_called()
        self.assertEqual(density.call_count, len(continuum.surfaces))
        density.return_value = -1
        with self.assertRaisesRegex(ValueError, "density.*s=0.8"):
            continuum.run()

    def test_fixed_grids_and_refinement_failure_keep_existing_checks(self):
        continuum = self.make_continuum()
        first = continuum.prepare()
        shape = first["pilots"][0]["quadrature"]["shape"]
        fixed = continuum.prepare(shape=shape, max_shape=(128, 128))
        quadrature = fixed["pilots"][0]["quadrature"]
        self.assertTrue(quadrature["fixed_grid"])
        self.assertEqual(quadrature["shape"], shape)
        with self.assertRaisesRegex(RuntimeError, "s=0.4"):
            continuum.prepare(shape=shape, max_shape=shape)
        self.assertEqual(continuum._preparation["pilots"], {})

    def test_invalid_preparation_controls(self):
        continuum = self.make_continuum()
        for settings in (
            {"benchmark": 1}, {"keep_eigenvectors": "yes"},
            {"pilot_count": 0}, {"pilot_count": 4}, {"pilot_count": 1.5},
            {"mpi_ranks": False}, {"mpi_ranks": 0}, {"mpi_ranks": 2.5},
            {"rtol": -1}, {"rtol": 0, "atol": 0}, {"max_shape": None},
        ):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    continuum.prepare(**settings)

    def test_pilot_does_not_skip_bad_geometry_on_an_unchecked_surface(self):
        continuum = self.make_continuum()
        continuum.prepare(benchmark=True)
        original = continuum._sample_geometry

        def fail_on_unchecked(surface, theta, zeta, expected_orientation=None):
            if surface == .3:
                raise ValueError("injected singular geometry")
            return original(surface, theta, zeta, expected_orientation)

        with patch.object(continuum, "_sample_geometry", side_effect=fail_on_unchecked):
            with self.assertRaisesRegex(ValueError, "s=0.3.*singular"):
                continuum.run()


if __name__ == "__main__":
    unittest.main()
