"""Direct full-torus Gram assembly as a reference for the FFT implementation."""

import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh

from firm3d.field.boozermagneticfield import BoozerRadialInterpolant
from firm3d.saw.stellgap import Continuum


class TestContinuumQuadrature(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filename = (
            Path(__file__).parents[1] / "test_files" / "boozmn_n3are_R7.75B5.7.nc"
        )
        cls.field = BoozerRadialInterpolant(str(filename), order=3, no_K=True)

    def setUp(self):
        self.theta = 2 * np.pi * np.arange(32) / 32
        self.zeta = 2 * np.pi * np.arange(24) / 24

    def constant_geometry(self, iota=0.5, A=3.0, W0=2.0):
        shape = (len(self.theta), len(self.zeta))
        return {"iota": iota, "A": np.full(shape, A), "W0": np.full(shape, W0)}

    def test_constant_weights_give_diagonal_dispersion_and_exact_null_modes(self):
        modes = np.array([[0, 0], [6, 3], [2, 0], [1, 3], [-3, -3]])
        continuum = Continuum(self.field, [0.5], modes)
        K, M0 = continuum._assemble_quadrature(
            self.theta, self.zeta, self.constant_geometry()
        )
        parallel = 0.5 * modes[:, 0] - modes[:, 1]
        np.testing.assert_allclose(M0, 2 * np.eye(len(modes)), atol=1e-13)
        np.testing.assert_allclose(K, np.diag(3 * parallel**2), atol=1e-13)
        np.testing.assert_array_equal(K[:2], np.zeros((2, len(modes))))
        np.testing.assert_allclose(
            eigvalsh(K, M0), np.sort(1.5 * parallel**2), atol=1e-13
        )
        np.linalg.cholesky(M0)

    def test_variable_weights_have_expected_sum_and_difference_couplings(self):
        modes = np.array([[1, 1], [2, -2], [0, -1]])
        continuum = Continuum(self.field, [0.5], modes)
        theta, zeta = np.meshgrid(self.theta, self.zeta, indexing="ij")
        geometry = {
            "iota": 0.37,
            "A": 3 + 0.6 * np.cos(theta + 3 * zeta) + 0.4 * np.cos(theta),
            "W0": 2 + 0.4 * np.cos(theta + 3 * zeta) + 0.2 * np.cos(theta),
        }
        K, M0 = continuum._assemble_quadrature(self.theta, self.zeta, geometry)
        expected_mass = np.array([[2, 0.2, 0.1], [0.2, 2, 0], [0.1, 0, 2]])
        derivative_moments = np.array([[3, 0.3, -0.2], [0.3, 3, 0], [-0.2, 0, 3]])
        parallel = 0.37 * modes[:, 0] - modes[:, 1]
        expected_stiffness = derivative_moments * np.outer(parallel, parallel)
        np.testing.assert_allclose(M0, expected_mass, atol=1e-13)
        np.testing.assert_allclose(K, expected_stiffness, atol=1e-13)
        np.testing.assert_allclose(M0, M0.T, rtol=0, atol=1e-14)
        np.testing.assert_allclose(K, K.T, rtol=0, atol=1e-14)
        np.linalg.cholesky(M0)
        self.assertGreater(np.linalg.eigvalsh(K).min(), 0)

    def test_undersampling_loses_mass_rank_without_negative_energy(self):
        continuum = Continuum(self.field, [0.5], [[1, 0], [3, 0]])
        self.theta = 2 * np.pi * np.arange(4) / 4
        self.zeta = 2 * np.pi * np.arange(8) / 8
        K, M0 = continuum._assemble_quadrature(
            self.theta, self.zeta, self.constant_geometry()
        )
        # cos(theta) and cos(3*theta) coincide on this four-point grid.
        np.testing.assert_allclose(M0, 2 * np.ones((2, 2)), atol=1e-13)
        self.assertEqual(np.linalg.matrix_rank(M0), 1)
        null_vector = np.array([1, -1])
        self.assertAlmostEqual(null_vector @ M0 @ null_vector, 0)
        self.assertGreaterEqual(np.linalg.eigvalsh(M0).min(), -1e-13)
        self.assertGreaterEqual(np.linalg.eigvalsh(K).min(), -1e-13)

        self.theta = 2 * np.pi * np.arange(16) / 16
        _, resolved_mass = continuum._assemble_quadrature(
            self.theta, self.zeta, self.constant_geometry()
        )
        np.testing.assert_allclose(resolved_mass, 2 * np.eye(2), atol=1e-13)
        self.assertEqual(np.linalg.matrix_rank(resolved_mass), 2)

    def test_reversing_a_cosine_mode_does_not_change_either_matrix(self):
        modes = np.array([[1, 1], [2, -2], [0, -1]])
        theta, zeta = np.meshgrid(self.theta, self.zeta, indexing="ij")
        geometry = {
            "iota": 0.37,
            "A": 3 + 0.6 * np.cos(theta + 3 * zeta),
            "W0": 2 + 0.2 * np.cos(theta),
        }
        original = Continuum(self.field, [0.5], modes)._assemble_quadrature(
            self.theta, self.zeta, geometry
        )
        modes[0] *= -1
        reversed_mode = Continuum(self.field, [0.5], modes)._assemble_quadrature(
            self.theta, self.zeta, geometry
        )
        for original_matrix, changed_matrix in zip(original, reversed_mode):
            np.testing.assert_allclose(changed_matrix, original_matrix, atol=1e-13)

    def test_rejects_one_period_grids_and_duplicated_endpoints(self):
        continuum = Continuum(self.field, [0.5], [[1, 1]])
        geometry = self.constant_geometry()
        for name in ("theta", "zeta"):
            valid = self.theta if name == "theta" else self.zeta
            cases = [
                valid / 3,
                np.linspace(0, 2 * np.pi, len(valid)),
                valid + 0.1,
                [],
                [[0]],
                [np.nan],
                [1j],
            ]
            for angles in cases:
                with self.subTest(name=name, angles=angles):
                    theta = angles if name == "theta" else self.theta
                    zeta = angles if name == "zeta" else self.zeta
                    with self.assertRaisesRegex(ValueError, name):
                        continuum._assemble_quadrature(theta, zeta, geometry)

    def test_rejects_invalid_weight_arrays(self):
        continuum = Continuum(self.field, [0.5], [[1, 1]])
        shape = (len(self.theta), len(self.zeta))
        cases = [np.ones((2, 2)), 1, np.zeros(shape), -np.ones(shape)]
        cases.extend(
            [
                np.full(shape, np.nan),
                np.full(shape, np.inf),
                np.full(shape, 1j),
                np.full(shape, "1"),
            ]
        )
        for name in ("A", "W0"):
            for invalid in cases:
                with self.subTest(name=name, value=invalid):
                    geometry = self.constant_geometry()
                    geometry[name] = invalid
                    with self.assertRaisesRegex(
                        ValueError, name + ".*finite.*positive"
                    ):
                        continuum._assemble_quadrature(self.theta, self.zeta, geometry)

    def test_rejects_invalid_iota_and_reports_arithmetic_failure(self):
        continuum = Continuum(self.field, [0.5], [[1, 1]])
        for iota in (np.nan, np.inf, 1j, [0.5]):
            with self.subTest(iota=iota):
                with self.assertRaisesRegex(ValueError, "iota"):
                    continuum._assemble_quadrature(
                        self.theta, self.zeta, self.constant_geometry(iota=iota)
                    )
        with self.assertRaisesRegex(ValueError, "Direct quadrature"):
            continuum._assemble_quadrature(
                self.theta, self.zeta, self.constant_geometry(W0=1e308)
            )

    def test_positive_mass_and_nonnegative_stiffness_on_real_geometry(self):
        continuum = Continuum(self.field, [0.5], [[1, 1], [2, 4], [0, -1]])
        period_grid = continuum._plan_angular_grid(continuum._plan_basis())
        ntheta, nzeta_period = period_grid["shape"]
        nzeta = self.field.nfp * nzeta_period
        theta = 2 * np.pi * np.arange(ntheta) / ntheta
        zeta = 2 * np.pi * np.arange(nzeta) / nzeta
        geometry = continuum._sample_geometry(0.5, theta, zeta)
        K, M0 = continuum._assemble_quadrature(theta, zeta, geometry)
        for matrix in (K, M0):
            self.assertTrue(np.all(np.isfinite(matrix)))
            symmetry_error = np.linalg.norm(matrix - matrix.T) / np.linalg.norm(matrix)
            self.assertLess(symmetry_error, 1e-14)
        np.linalg.cholesky(M0)
        self.assertGreaterEqual(np.linalg.eigvalsh(K).min(), -1e-13 * np.linalg.norm(K))


if __name__ == "__main__":
    unittest.main()
