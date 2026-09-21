"""Basis normalization, full-torus selection rules, and FFT grid planning."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.interpolate import make_interp_spline

from firm3d.field.boozermagneticfield import BoozerRadialInterpolant
from firm3d.saw.stellgap import Continuum


EQUILIBRIUM_FILE = (
    Path(__file__).parents[1] / "test_files" / "boozmn_n3are_R7.75B5.7.nc"
)


class TestContinuumGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.field = BoozerRadialInterpolant(str(EQUILIBRIUM_FILE), order=3, no_K=True)

    def setUp(self):
        # Use a small Fourier table with known bounds; geometry is not sampled here.
        spline = make_interp_spline(np.linspace(0, 1, 4), np.zeros((4, 3)))
        patcher = patch.multiple(
            self.field,
            nfp=4,
            xm_b=np.array([0, 2, 1]),
            xn_b=np.array([0, 4, -8]),
            rmnc_splines=spline,
            zmns_splines=spline,
            numns_splines=spline,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_hand_computed_moments_for_an_ordinary_family(self):
        modes = [[1, 1], [2, 3], [0, -1]]
        continuum = Continuum(self.field, [0.5], modes)
        basis = continuum._plan_basis()
        np.testing.assert_array_equal(continuum.modes, modes)
        np.testing.assert_array_equal(
            basis["sum_modes"],
            [
                [[2, 2], [3, 4], [1, 0]],
                [[3, 4], [4, 6], [2, 2]],
                [[1, 0], [2, 2], [0, -2]],
            ],
        )
        np.testing.assert_array_equal(
            basis["difference_modes"],
            [
                [[0, 0], [-1, -2], [1, 2]],
                [[1, 2], [0, 0], [2, 4]],
                [[-1, -2], [-2, -4], [0, 0]],
            ],
        )
        np.testing.assert_array_equal(
            basis["sum_allowed"],
            [[False, True, True], [True, False, False], [True, False, False]],
        )
        np.testing.assert_array_equal(
            basis["difference_allowed"],
            [[True, False, False], [False, True, True], [False, True, True]],
        )

    def test_self_conjugate_families_allow_both_moment_types(self):
        cases = [
            [[0, 0], [1, 0], [2, 4], [-1, 4]],
            [[0, 2], [1, 6], [-2, -2]],
        ]
        for modes in cases:
            with self.subTest(modes=modes):
                basis = Continuum(self.field, [0.5], modes)._plan_basis()
                self.assertTrue(basis["difference_allowed"].all())
                self.assertTrue(basis["sum_allowed"].all())

    def test_normalized_cosines_are_orthonormal_on_the_full_torus(self):
        modes = np.array([[1, 0], [0, 0], [-1, 4], [2, -4]])
        continuum = Continuum(self.field, [0.5], modes)
        basis = continuum._plan_basis()
        np.testing.assert_array_equal(
            basis["normalization"], [np.sqrt(2), 1, np.sqrt(2), np.sqrt(2)]
        )
        angles = 2 * np.pi * np.arange(32) / 32
        theta, zeta = np.meshgrid(angles, angles, indexing="ij")
        phase = theta.ravel()[:, None] * modes[:, 0]
        phase -= zeta.ravel()[:, None] * modes[:, 1]
        functions = np.cos(phase) * basis["normalization"]
        gram = functions.T @ functions / theta.size
        np.testing.assert_allclose(gram, np.eye(len(modes)), atol=1e-14)

    def test_initial_grid_and_signed_fft_indices(self):
        continuum = Continuum(self.field, [0.5], [[1, 1], [2, 3], [0, -1]])
        basis = continuum._plan_basis()
        grid = continuum._plan_angular_grid(basis)
        self.assertEqual(grid["shape"], (7, 5))
        self.assertEqual(grid["minimum_shape"], (7, 5))
        np.testing.assert_allclose(grid["theta"], 2 * np.pi * np.arange(7) / 7)
        np.testing.assert_allclose(grid["zeta"], 2 * np.pi * np.arange(5) / 20)
        self.assertLess(grid["theta"][-1], 2 * np.pi)
        self.assertLess(grid["zeta"][-1], 2 * np.pi / 4)
        np.testing.assert_array_equal(grid["sum_indices"][0], [3, 1, 3, 1])
        np.testing.assert_array_equal(grid["sum_indices"][1], [4, 0, 4, 0])
        np.testing.assert_array_equal(grid["difference_indices"][0], [0, 0, 2, 5, 0])
        np.testing.assert_array_equal(grid["difference_indices"][1], [0, 0, 4, 1, 0])

    def test_required_moments_can_exceed_the_equilibrium_cutoff(self):
        continuum = Continuum(self.field, [0.5], [[0, 0], [-9, 12]])
        basis = continuum._plan_basis()
        grid = continuum._plan_angular_grid(basis)
        self.assertEqual(grid["minimum_shape"], (37, 13))
        for shape in ((36, 13), (17, 13), (37, 12), (37, 5)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "below Nyquist"):
                    continuum._plan_angular_grid(basis, shape)
        finer = continuum._plan_angular_grid(basis, (40, 16))
        self.assertEqual(finer["shape"], (40, 16))
        self.assertEqual(finer["minimum_shape"], (37, 13))
        np.testing.assert_array_equal(finer["sum_indices"][0], [0, 31, 31, 22])
        np.testing.assert_array_equal(finer["sum_indices"][1], [0, 13, 13, 10])

    def test_forbidden_moments_do_not_require_fft_bins(self):
        continuum = Continuum(self.field, [0.5], [[100, 1]])
        basis = continuum._plan_basis()
        grid = continuum._plan_angular_grid(basis)
        self.assertFalse(basis["sum_allowed"][0, 0])
        self.assertEqual(grid["sum_indices"][0].size, 0)
        self.assertEqual(grid["sum_indices"][1].size, 0)
        self.assertEqual(grid["shape"], (5, 5))

    def test_one_period_fft_moments_match_full_torus_integrals(self):
        def weight(theta, zeta):
            return (
                2
                + 0.4 * np.cos(3 * theta - 4 * zeta)
                + 0.2 * np.cos(2 * theta - 4 * zeta)
                + 0.3 * np.cos(theta)
            )

        angles = 2 * np.pi * np.arange(64) / 64
        full_theta, full_zeta = np.meshgrid(angles, angles, indexing="ij")
        full_weight = weight(full_theta, full_zeta)
        cases = [
            [[1, 1], [2, 3], [0, -1]],
            [[0, 0], [1, 0], [2, 4], [-1, 4]],
            [[1, 2], [2, 6], [-2, -2]],
        ]
        for modes in cases:
            with self.subTest(modes=modes):
                continuum = Continuum(self.field, [0.5], modes)
                basis = continuum._plan_basis()
                grid = continuum._plan_angular_grid(basis, (24, 20))
                samples = weight(grid["theta"][:, None], grid["zeta"][None, :])
                coefficients = np.fft.fft2(samples) / samples.size
                for kind in ("difference", "sum"):
                    allowed = basis[kind + "_allowed"]
                    actual = np.zeros(allowed.shape)
                    actual[allowed] = coefficients[grid[kind + "_indices"]].real
                    expected = np.zeros(allowed.shape)
                    for i in range(len(modes)):
                        for j in range(len(modes)):
                            m, n = basis[kind + "_modes"][i, j]
                            phase = m * full_theta - n * full_zeta
                            expected[i, j] = np.mean(full_weight * np.cos(phase))
                    np.testing.assert_allclose(actual, expected, atol=1e-13)

    def test_planning_uses_the_copied_field_period_count(self):
        continuum = Continuum(self.field, [0.5], [[1, 1], [2, 3]])
        with patch.object(self.field, "nfp", 5):
            basis = continuum._plan_basis()
            grid = continuum._plan_angular_grid(basis)
        self.assertTrue(basis["sum_allowed"][0, 1])
        self.assertAlmostEqual(grid["zeta"][1], 2 * np.pi / (4 * grid["shape"][1]))

    def test_rejects_invalid_grid_shapes(self):
        continuum = Continuum(self.field, [0.5], [[0, 0]])
        basis = continuum._plan_basis()
        shapes = [0, [], [5], [5, 5, 5], [[5, 5]], [0, 5], [-1, 5]]
        shapes.extend([[5.0, 5.0], [np.inf, 5], [True, False], ["5", "5"]])
        for shape in shapes:
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "grid shape"):
                    continuum._plan_angular_grid(basis, shape)

    def test_rejects_equilibrium_indices_incompatible_with_one_period(self):
        for indices in (np.array([0, 1, -8]), np.array([0.0, 4.0, -8.0])):
            with self.subTest(indices=indices):
                with patch.object(self.field, "xn_b", indices):
                    continuum = Continuum(self.field, [0.5], [[0, 0]])
                with self.assertRaisesRegex(ValueError, "Equilibrium.*indices"):
                    continuum._plan_angular_grid(continuum._plan_basis())

    def test_rejects_mode_arithmetic_overflow_before_forming_pairs(self):
        for m in (2**62, -(2**63)):
            with self.subTest(m=m):
                continuum = Continuum(self.field, [0.5], [[m, 0]])
                with self.assertRaisesRegex(ValueError, "64-bit sums and differences"):
                    continuum._plan_basis()


class TestEquilibriumGrid(unittest.TestCase):
    def test_existing_equilibrium_support_fits_below_nyquist(self):
        field = BoozerRadialInterpolant(str(EQUILIBRIUM_FILE), order=3, no_K=True)
        continuum = Continuum(field, [0.5], [[0, 0], [1, 3]])
        basis = continuum._plan_basis()
        grid = continuum._plan_angular_grid(basis)
        self.assertGreater(grid["shape"][0], 2 * np.abs(field.xm_b).max())
        self.assertGreater(grid["shape"][1], 2 * np.abs(field.xn_b).max() // field.nfp)
        self.assertIsNone(continuum._theta_grid)
        self.assertIsNone(continuum._zeta_grid)


if __name__ == "__main__":
    unittest.main()
