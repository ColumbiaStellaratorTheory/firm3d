"""FFT moments and cosine assembly checked against the full-torus Gram reference."""

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


class TestContinuumFFT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.field = BoozerRadialInterpolant(str(EQUILIBRIUM_FILE), order=3, no_K=True)

    def make_continuum(self, modes, nfp=3):
        # Synthetic weights need only a small, compatible coordinate spectrum.
        spline = make_interp_spline(np.linspace(0, 1, 4), np.zeros((4, 3)))
        with patch.multiple(
            self.field,
            nfp=nfp,
            xm_b=np.array([0, 2, 1]),
            xn_b=np.array([0, nfp, -2 * nfp]),
            rmnc_splines=spline,
            zmns_splines=spline,
            numns_splines=spline,
        ):
            return Continuum(self.field, [0.5], modes)

    def check_reference(self, continuum, grid, geometry, K, M0):
        # Repeat exactly the same weight samples around the full torus.
        nzeta = grid["shape"][1] * continuum._nfp
        zeta = 2 * np.pi * np.arange(nzeta) / nzeta
        full_geometry = {
            "iota": geometry["iota"],
            "A": np.tile(geometry["A"], (1, continuum._nfp)),
            "W0": np.tile(geometry["W0"], (1, continuum._nfp)),
        }
        reference = continuum._assemble_quadrature(grid["theta"], zeta, full_geometry)
        for actual, expected in zip((K, M0), reference):
            scale = np.linalg.norm(expected)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13 * scale)
            np.testing.assert_allclose(actual, actual.T, rtol=0, atol=1e-14 * scale)
            self.assertTrue(np.all(np.isfinite(actual)))
            self.assertGreaterEqual(np.linalg.eigvalsh(actual).min(), -1e-13 * scale)
        np.linalg.cholesky(M0)

    def test_constant_weights_and_exact_null_modes(self):
        modes = np.array([[0, 0], [6, 3], [2, 0], [1, 3], [-3, -3]])
        continuum = self.make_continuum(modes)
        basis = continuum._plan_basis()
        grid = continuum._plan_angular_grid(basis)
        geometry = {
            "iota": 0.5,
            "A": np.full(grid["shape"], 3.0),
            "W0": np.full(grid["shape"], 2.0),
        }
        moments = continuum._fourier_moments(basis, grid, geometry)
        K, M0 = continuum._assemble_moments(basis, moments, geometry["iota"])
        parallel = 0.5 * modes[:, 0] - modes[:, 1]
        np.testing.assert_allclose(M0, 2 * np.eye(len(modes)), atol=1e-14)
        np.testing.assert_allclose(K, np.diag(3 * parallel**2), atol=1e-13)
        np.testing.assert_array_equal(K[:2], np.zeros((2, len(modes))))
        self.check_reference(continuum, grid, geometry, K, M0)

    def test_analytic_sum_and_difference_couplings(self):
        modes = np.array([[1, 1], [2, -2], [0, -1]])
        continuum = self.make_continuum(modes)
        basis = continuum._plan_basis()
        grid = continuum._plan_angular_grid(basis)
        theta, zeta = grid["theta"][:, None], grid["zeta"][None, :]
        geometry = {
            "iota": 0.37,
            "A": 3 + 0.6 * np.cos(theta + 3 * zeta) + 0.4 * np.cos(theta),
            "W0": 2 + 0.4 * np.cos(theta + 3 * zeta) + 0.2 * np.cos(theta),
        }
        moments = continuum._fourier_moments(basis, grid, geometry)
        K, M0 = continuum._assemble_moments(basis, moments, geometry["iota"])
        expected_mass = np.array([[2, 0.2, 0.1], [0.2, 2, 0], [0.1, 0, 2]])
        expected_derivatives = np.array([[3, 0.3, -0.2], [0.3, 3, 0], [-0.2, 0, 3]])
        parallel = 0.37 * modes[:, 0] - modes[:, 1]
        np.testing.assert_allclose(M0, expected_mass, atol=1e-14)
        np.testing.assert_allclose(
            K, expected_derivatives * np.outer(parallel, parallel), atol=1e-13
        )
        for name in ("A", "W0"):
            for kind in ("difference", "sum"):
                forbidden = ~basis[kind + "_allowed"]
                np.testing.assert_array_equal(moments[name + "_" + kind][forbidden], 0)
        self.check_reference(continuum, grid, geometry, K, M0)

    def test_full_torus_equivalence_for_different_mode_families(self):
        cases = [
            (1, [[0, 0], [1, 1], [-2, 3], [2, -1]]),
            (3, [[0, 0], [1, 0], [2, 3], [-1, 3]]),
            (4, [[1, 1], [2, 3], [0, -1]]),
            (4, [[0, 2], [1, 6], [-2, -2]]),
            (5, [[1, 1], [2, 4], [0, -1], [-3, -4]]),
        ]
        for nfp, modes in cases:
            for shape in ((25, 21), (26, 22)):
                with self.subTest(nfp=nfp, modes=modes, shape=shape):
                    continuum = self.make_continuum(modes, nfp)
                    basis = continuum._plan_basis()
                    grid = continuum._plan_angular_grid(basis, shape)
                    theta, zeta = grid["theta"][:, None], grid["zeta"][None, :]
                    phase = theta + nfp * zeta
                    geometry = {
                        "iota": 0.43,
                        "A": np.exp(0.4 * np.cos(phase) + 0.2 * np.cos(theta)),
                        "W0": 1 / (1 - 0.3 * np.cos(2 * theta - nfp * zeta)),
                    }
                    moments = continuum._fourier_moments(basis, grid, geometry)
                    K, M0 = continuum._assemble_moments(
                        basis, moments, geometry["iota"]
                    )
                    self.check_reference(continuum, grid, geometry, K, M0)

    def test_nonlinear_moments_beyond_coordinate_spectrum_are_retained(self):
        continuum = self.make_continuum([[0, 0], [3, 12]], nfp=4)
        basis = continuum._plan_basis()
        grid = continuum._plan_angular_grid(basis, (64, 64))
        phase = grid["theta"][:, None] - 4 * grid["zeta"][None, :]
        weight = 1 / (1 - 0.6 * np.cos(phase))
        geometry = {"iota": 0.43, "A": weight, "W0": weight}
        moments = continuum._fourier_moments(basis, grid, geometry)
        # <cos(l*phase)/(1-a*cos(phase))> = q**l / sqrt(1-a**2), q=1/3 here.
        self.assertAlmostEqual(moments["W0_sum"][0, 1], (1 / 3)**3 / 0.8, places=14)
        self.assertAlmostEqual(moments["W0_sum"][1, 1], (1 / 3)**6 / 0.8, places=14)
        K, M0 = continuum._assemble_moments(basis, moments, geometry["iota"])
        self.check_reference(continuum, grid, geometry, K, M0)

    def test_arbitrary_positive_samples_preserve_gram_properties(self):
        continuum = self.make_continuum([[1, 1], [2, 4], [0, -1], [-3, -4]], nfp=5)
        basis = continuum._plan_basis()
        grid = continuum._plan_angular_grid(basis, (24, 20))
        rng = np.random.default_rng(451)
        geometry = {
            "iota": 0.43,
            "A": np.exp(rng.normal(size=grid["shape"])),
            "W0": np.exp(rng.normal(size=grid["shape"])),
        }
        moments = continuum._fourier_moments(basis, grid, geometry)
        K, M0 = continuum._assemble_moments(basis, moments, geometry["iota"])
        self.check_reference(continuum, grid, geometry, K, M0)

    def test_forbidden_moments_do_not_wrap_to_zero_frequency(self):
        continuum = self.make_continuum([[100, 1]], nfp=4)
        basis = continuum._plan_basis()
        grid = continuum._plan_angular_grid(basis)
        self.assertEqual(grid["shape"], (5, 5))
        geometry = {"A": np.full((5, 5), 3.0), "W0": np.full((5, 5), 2.0)}
        moments = continuum._fourier_moments(basis, grid, geometry)
        self.assertEqual(moments["A_sum"][0, 0], 0)
        self.assertEqual(moments["W0_sum"][0, 0], 0)
        K, M0 = continuum._assemble_moments(basis, moments, iota=0.5)
        np.testing.assert_allclose(K, [[3 * 49**2]])
        np.testing.assert_allclose(M0, [[2]])

    def test_mode_sign_reversal_and_inputs_are_preserved(self):
        matrices = []
        for modes in ([[1, 1], [2, -2]], [[-1, -1], [2, -2]]):
            continuum = self.make_continuum(modes)
            basis = continuum._plan_basis()
            grid = continuum._plan_angular_grid(basis, (24, 20))
            theta, zeta = grid["theta"][:, None], grid["zeta"][None, :]
            weight = 2 + 0.3 * np.cos(theta + 3 * zeta)
            geometry = {"A": weight, "W0": weight}
            original = weight.copy()
            moments = continuum._fourier_moments(basis, grid, geometry)
            copies = {name: values.copy() for name, values in moments.items()}
            matrices.append(continuum._assemble_moments(basis, moments, 0.43))
            np.testing.assert_array_equal(weight, original)
            for name in moments:
                np.testing.assert_array_equal(moments[name], copies[name])
        np.testing.assert_allclose(matrices[0], matrices[1], atol=1e-14)

    def test_rejects_invalid_sampled_weights(self):
        continuum = self.make_continuum([[0, 0], [1, 3]])
        basis = continuum._plan_basis()
        grid = continuum._plan_angular_grid(basis)
        valid = np.ones(grid["shape"])
        invalid = [valid[:, :-1], valid.astype(complex), valid.astype(str)]
        for value in (0.0, -1.0, np.nan, np.inf):
            bad = valid.copy()
            bad[0, 0] = value
            invalid.append(bad)
        for name in ("A", "W0"):
            for bad in invalid:
                with self.subTest(name=name, dtype=bad.dtype, shape=bad.shape):
                    geometry = {"A": valid, "W0": valid, name: bad}
                    with self.assertRaisesRegex(ValueError, name):
                        continuum._fourier_moments(basis, grid, geometry)

    def test_rejects_invalid_moments_iota_and_overflow(self):
        continuum = self.make_continuum([[0, 0], [2, 3]])
        basis = continuum._plan_basis()
        grid = continuum._plan_angular_grid(basis)
        geometry = {"A": np.ones(grid["shape"]), "W0": np.ones(grid["shape"])}
        moments = continuum._fourier_moments(basis, grid, geometry)
        invalid = (
            np.ones((1, 2)),
            np.ones((2, 2), complex),
            np.full((2, 2), np.nan),
        )
        for name in moments:
            for bad in invalid:
                with self.subTest(name=name, shape=bad.shape):
                    with self.assertRaisesRegex(ValueError, name):
                        continuum._assemble_moments(basis, {**moments, name: bad}, 0.43)
        for iota in (np.nan, np.inf, 1j, "0.4", [0.4]):
            with self.subTest(iota=iota):
                with self.assertRaisesRegex(ValueError, "iota"):
                    continuum._assemble_moments(basis, moments, iota)
        with self.assertRaisesRegex(ValueError, "assembly failed"):
            continuum._assemble_moments(basis, moments, np.finfo(float).max)

    def test_real_equilibrium_matches_independent_full_torus_sampling(self):
        continuum = Continuum(self.field, [0.2, 0.5, 0.8], [[1, 1], [2, -2], [0, -1]])
        basis = continuum._plan_basis()
        grid = continuum._plan_angular_grid(basis)
        nzeta = grid["shape"][1] * continuum._nfp
        full_zeta = 2 * np.pi * np.arange(nzeta) / nzeta
        for surface in continuum.surfaces:
            with self.subTest(surface=surface):
                geometry = continuum._sample_geometry(
                    surface, grid["theta"], grid["zeta"]
                )
                moments = continuum._fourier_moments(basis, grid, geometry)
                K, M0 = continuum._assemble_moments(basis, moments, geometry["iota"])
                full = continuum._sample_geometry(surface, grid["theta"], full_zeta)
                reference = continuum._assemble_quadrature(
                    grid["theta"], full_zeta, full
                )
                for actual, expected in zip((K, M0), reference):
                    np.testing.assert_allclose(
                        actual, expected, rtol=1e-12,
                        atol=1e-13 * np.linalg.norm(expected)
                    )
                self.check_reference(continuum, grid, geometry, K, M0)


if __name__ == "__main__":
    unittest.main()
