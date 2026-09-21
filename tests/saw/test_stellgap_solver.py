"""Generalized eigensolve accuracy, null modes, failures, and the serial run."""

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


class TestContinuumSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.field = BoozerRadialInterpolant(str(EQUILIBRIUM_FILE), order=3, no_K=True)

    def make_continuum(self, modes=None, surfaces=None, density=None):
        if modes is None:
            modes = [[0, 0], [6, 3], [2, 0], [1, 3]]
        if surfaces is None:
            surfaces = [0.8, 0.2, 0.8]
        s = np.linspace(0, 1, 7)
        radius = np.column_stack((np.full_like(s, 3), s))
        height = np.column_stack((np.zeros_like(s), s))
        nu = np.column_stack((np.zeros_like(s), 0.1 * s**2))
        with patch.multiple(
            self.field,
            xm_b=np.array([0, 1]),
            xn_b=np.array([0, 3]),
            rmnc_splines=make_interp_spline(s, radius),
            zmns_splines=make_interp_spline(s, height),
            numns_splines=make_interp_spline(s, nu),
            iota_spline=make_interp_spline(s, np.full_like(s, 0.5)),
            psi0=-0.8,
        ):
            return Continuum(self.field, surfaces, modes, density=density)

    def assert_eigenpairs(self, K, M0, solution):
        values = solution["eigenvalues"]
        vectors = solution["eigenvectors"]
        np.testing.assert_allclose(
            K @ vectors, (M0 @ vectors) * values, atol=1e-12, rtol=1e-12
        )
        np.testing.assert_allclose(
            vectors.T @ M0 @ vectors, np.eye(len(values)), atol=1e-13
        )
        checks = solution["diagnostics"]
        self.assertLessEqual(np.max(checks["scaled_residuals"]), checks["tolerance"])
        self.assertLessEqual(checks["mass_orthogonality_error"], checks["tolerance"])

    def test_constant_weight_dispersion_preserves_two_exact_null_modes(self):
        continuum = self.make_continuum()
        theta = 2 * np.pi * np.arange(64) / 64
        zeta = 2 * np.pi * np.arange(48) / 48
        geometry = {
            "A": np.full((64, 48), 3.0), "W0": np.full((64, 48), 2.0), "iota": 0.5
        }
        K, M0 = continuum._assemble_quadrature(theta, zeta, geometry)
        original_K, original_M0 = K.copy(), M0.copy()
        solution = continuum._solve_surface(0.5, K, M0)
        parallel = 0.5 * continuum.modes[:, 0] - continuum.modes[:, 1]
        np.testing.assert_allclose(
            solution["eigenvalues"], np.sort(1.5 * parallel**2), atol=1e-13
        )
        np.testing.assert_allclose(solution["eigenvalues"][:2], 0, atol=1e-13)
        np.testing.assert_array_equal(K, original_K)
        np.testing.assert_array_equal(M0, original_M0)
        self.assert_eigenpairs(K, M0, solution)

    def test_coupled_mass_and_degenerate_subspace(self):
        continuum = self.make_continuum(modes=[[0, 0], [1, 0], [2, 0]])
        M0 = np.array([[2, 0.2, 0.1], [0.2, 3, 0.3], [0.1, 0.3, 4]])
        factor = np.linalg.cholesky(M0)
        rotation, _ = np.linalg.qr(np.random.default_rng(39).normal(size=(3, 3)))
        K = factor @ rotation @ np.diag([0.0, 4.0, 4.0]) @ rotation.T @ factor.T
        solution = continuum._solve_surface(0.5, K, M0)
        np.testing.assert_allclose(solution["eigenvalues"], [0, 4, 4], atol=1e-13)
        expected_vectors = np.linalg.solve(factor.T, rotation[:, 1:])
        actual_vectors = solution["eigenvectors"][:, 1:]
        np.testing.assert_allclose(
            actual_vectors @ actual_vectors.T @ M0,
            expected_vectors @ expected_vectors.T @ M0,
            atol=1e-13,
        )
        self.assert_eigenpairs(K, M0, solution)

    def test_zero_stiffness_and_small_positive_eigenvalues(self):
        continuum = self.make_continuum(modes=[[0, 0], [1, 0]])
        zero = continuum._solve_surface(0.5, np.zeros((2, 2)), np.diag([2, 3]))
        np.testing.assert_array_equal(zero["eigenvalues"], [0, 0])
        self.assert_eigenpairs(np.zeros((2, 2)), np.diag([2, 3]), zero)
        tiny = np.finfo(float).eps**2
        positive = continuum._solve_surface(0.5, np.diag([tiny, 1.0]), np.eye(2))
        self.assertGreater(positive["eigenvalues"][0], 0)
        self.assertAlmostEqual(positive["eigenvalues"][0] / tiny, 1)
        self.assertEqual(positive["diagnostics"]["roundoff_zero_count"], 0)

    def test_roundoff_negative_is_zeroed_and_raw_value_retained(self):
        continuum = self.make_continuum(modes=[[0, 0], [1, 0]])
        raw_values = np.array([-np.finfo(float).eps, 2.0])
        vectors = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
        with patch("firm3d.saw.stellgap.eigh", return_value=(raw_values, vectors)):
            solution = continuum._solve_surface(0.5, np.ones((2, 2)), np.eye(2))
        np.testing.assert_array_equal(solution["eigenvalues"], [0, 2])
        np.testing.assert_array_equal(
            solution["diagnostics"]["raw_eigenvalues"], raw_values
        )
        self.assertEqual(solution["diagnostics"]["roundoff_zero_count"], 1)
        self.assertGreater(
            solution["diagnostics"]["negative_tolerances"][0], abs(raw_values[0])
        )
        self.assert_eigenpairs(np.ones((2, 2)), np.eye(2), solution)

    def test_appreciable_negative_is_not_hidden_by_an_unrelated_large_mode(self):
        continuum = self.make_continuum(modes=[[0, 0], [1, 0]])
        for diagonal in ([-0.1, 1], [-1, 1e20]):
            with self.subTest(diagonal=diagonal):
                with self.assertRaisesRegex(
                    RuntimeError, "Negative eigenvalue at s=0.5"
                ):
                    continuum._solve_surface(0.5, np.diag(diagonal), np.eye(2))

    def test_indefinite_and_singular_mass_are_reported_with_surface(self):
        continuum = self.make_continuum(modes=[[0, 0], [1, 0]])
        for mass in (np.diag([1, -1]), np.ones((2, 2)), np.zeros((2, 2))):
            with self.subTest(mass=mass):
                with self.assertRaisesRegex(RuntimeError, "s=0.5.*positive definite"):
                    continuum._solve_surface(0.5, np.eye(2), mass)

    def test_asymmetry_is_detected_before_lapack_reads_a_triangle(self):
        continuum = self.make_continuum(modes=[[0, 0], [1, 0]])
        bad = np.array([[1.0, 0.01], [0, 1]])
        for name, stiffness, mass in (("K", bad, np.eye(2)), ("M0", np.eye(2), bad)):
            with self.subTest(name=name), patch("firm3d.saw.stellgap.eigh") as solver:
                with self.assertRaisesRegex(ValueError, name + " symmetry.*s=0.5"):
                    continuum._solve_surface(0.5, stiffness, mass)
                solver.assert_not_called()

    def test_invalid_matrices_are_rejected_before_solving(self):
        continuum = self.make_continuum(modes=[[0, 0], [1, 0]])
        invalid = [np.ones((2, 1)), np.eye(2, dtype=complex), np.full((2, 2), "one")]
        for value in (np.nan, np.inf):
            array = np.eye(2)
            array[0, 0] = value
            invalid.append(array)
        for bad in invalid:
            for stiffness, mass in ((bad, np.eye(2)), (np.eye(2), bad)):
                with self.subTest(dtype=bad.dtype, shape=bad.shape):
                    with self.assertRaisesRegex(
                        ValueError, "finite real matrix.*s=0.5"
                    ):
                        continuum._solve_surface(0.5, stiffness, mass)
        huge = np.array([[1e308, -1e308], [1e308, 1e308]])
        with self.assertRaisesRegex(ValueError, "K matrix scaling.*s=0.5"):
            continuum._solve_surface(0.5, huge, np.eye(2))

    def test_bad_residuals_and_mass_orthogonality_are_rejected(self):
        continuum = self.make_continuum(modes=[[0, 0], [1, 0]])
        bad_solutions = [
            (np.array([1.0, 3.0]), np.eye(2)),
            (np.array([1.0, 2.0]), 2 * np.eye(2)),
        ]
        for values, vectors in bad_solutions:
            with patch("firm3d.saw.stellgap.eigh", return_value=(values, vectors)):
                with self.assertRaisesRegex(
                    RuntimeError, "Eigenpair checks failed at s=0.5"
                ):
                    continuum._solve_surface(0.5, np.diag([1, 2]), np.eye(2))

    def test_nonfinite_solver_output_is_rejected(self):
        continuum = self.make_continuum(modes=[[0, 0], [1, 0]])
        for values, vectors in (
            (np.array([1.0, np.nan]), np.eye(2)),
            (np.array([1.0, 2.0]), np.full((2, 2), np.inf)),
        ):
            with patch("firm3d.saw.stellgap.eigh", return_value=(values, vectors)):
                with self.assertRaisesRegex(RuntimeError, "nonfinite values at s=0.5"):
                    continuum._solve_surface(0.5, np.diag([1, 2]), np.eye(2))

    def test_run_preserves_surface_and_mode_order_and_optional_vectors(self):
        continuum = self.make_continuum()
        result = continuum.run(keep_eigenvectors=True)
        np.testing.assert_array_equal(result["surfaces"], [0.8, 0.2, 0.8])
        np.testing.assert_array_equal(result["modes"], continuum.modes)
        self.assertEqual(result["eigenvalues"].shape, (3, 4))
        self.assertEqual(result["eigenvectors"].shape, (3, 4, 4))
        self.assertEqual(result["eigenvalue_units"], "T^2/m^2")
        self.assertIsNone(result["density"])
        self.assertTrue(np.all(np.diff(result["eigenvalues"], axis=1) >= 0))
        np.testing.assert_allclose(result["eigenvalues"][:, :2], 0, atol=1e-12)
        np.testing.assert_array_equal(
            result["eigenvalues"][0], result["eigenvalues"][2]
        )
        for record in result["diagnostics"]:
            self.assertTrue(record["quadrature"]["converged"])
            self.assertEqual(record["frequency_grid_convergence"], "unverified")
            self.assertLessEqual(
                max(record["solver"]["scaled_residuals"]), record["solver"]["tolerance"]
            )
        result["surfaces"][0] = 0.1
        result["modes"][0] = [99, 99]
        self.assertEqual(continuum.surfaces[0], 0.8)
        np.testing.assert_array_equal(continuum.modes[0], [0, 0])
        compact = continuum.run()
        self.assertIsNone(compact["eigenvectors"])
        np.testing.assert_array_equal(compact["eigenvalues"], result["eigenvalues"])

    def test_density_does_not_change_eigenvalues(self):
        continuum = self.make_continuum(surfaces=[0.4])
        reference = continuum.run()["eigenvalues"]
        for density in (1e-7, lambda s: 2e-7 * (1 + s)):
            result = self.make_continuum(surfaces=[0.4], density=density).run()
            np.testing.assert_array_equal(result["eigenvalues"], reference)
            expected = density(0.4) if callable(density) else density
            np.testing.assert_allclose(result["density"], [expected])

    def test_density_on_every_surface_is_checked_before_any_solve(self):
        def density(surface):
            return 1e-7 if surface < 0.5 else -1

        continuum = self.make_continuum(surfaces=[0.2, 0.8], density=density)
        with patch.object(continuum, "_converge_surface") as quadrature:
            with self.assertRaisesRegex(ValueError, "density.*s=0.8"):
                continuum.run()
            quadrature.assert_not_called()
        broken = self.make_continuum(density=Mock(side_effect=ZeroDivisionError))
        with self.assertRaisesRegex(ValueError, "density evaluation failed at s=0.8"):
            broken.run()

    def test_failed_quadrature_prevents_solving_and_keeps_surface_context(self):
        continuum = self.make_continuum(surfaces=[0.4])
        with patch.object(continuum, "_solve_surface") as solver:
            with self.assertRaisesRegex(RuntimeError, "s=0.4"):
                continuum.run(shape=(25, 5), max_shape=(25, 5))
            solver.assert_not_called()

    def test_run_keeps_field_points_and_communicator_untouched(self):
        continuum = self.make_continuum(surfaces=[0.4])
        points = np.array([[0.4, 0.1, 0.2]])
        self.field.set_points(points)
        communicator = Mock()
        with patch.object(self.field, "comm", communicator):
            continuum.run()
            self.assertEqual(communicator.mock_calls, [])
        np.testing.assert_array_equal(self.field.get_points_ref(), points)

    def test_run_rejects_nonboolean_eigenvector_setting(self):
        continuum = self.make_continuum()
        for value in (1, "yes", None):
            with self.assertRaisesRegex(ValueError, "keep_eigenvectors"):
                continuum.run(keep_eigenvectors=value)


if __name__ == "__main__":
    unittest.main()
