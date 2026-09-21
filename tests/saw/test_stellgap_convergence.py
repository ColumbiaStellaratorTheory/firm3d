"""Per-surface moment convergence, fixed-grid verification, and bounded failures."""

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


class TestContinuumConvergence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.field = BoozerRadialInterpolant(str(EQUILIBRIUM_FILE), order=3, no_K=True)

    def make_continuum(self, modes=None):
        if modes is None:
            modes = [[0, 0], [3, 9]]
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
            return Continuum(self.field, [0.4, 0.8], modes)

    def reciprocal_weights(self, surface, theta, zeta, expected_orientation=None):
        if expected_orientation is not None:
            self.assertEqual(expected_orientation, -1)
        weight = 1 / (1 - 0.95 * np.cos(theta[:, None] - 3 * zeta[None, :]))
        return {
            "A": weight,
            "W0": weight,
            "iota": 0.4,
            "orientation": -1,
            "min_jacobian_quality": 0.9,
        }

    def test_smooth_geometry_converges_with_null_modes_and_surface_metadata(self):
        continuum = self.make_continuum([[0, 0], [6, 3], [3, 3]])
        basis = continuum._plan_basis()
        records = []
        for surface in continuum.surfaces:
            result = continuum._converge_surface(surface, basis, max_shape=(512, 512))
            report = result["quadrature"]
            self.assertTrue(report["converged"])
            self.assertFalse(report["fixed_grid"])
            self.assertEqual(report["surface"], surface)
            self.assertEqual(report["shape"], report["history"][-1]["fine_shape"])
            self.assertLessEqual(report["history"][-1]["max_tolerance_ratio"], 1)
            self.assertEqual(report["basis_convergence"], "unverified")
            self.assertEqual(report["equilibrium_convergence"], "unverified")
            self.assertEqual(result["orientation"], -1)
            self.assertGreater(result["min_jacobian_quality"], 0)
            np.testing.assert_allclose(result["K"][:2], 0, atol=1e-28)
            np.linalg.cholesky(result["M0"])
            records.append(report)
        self.assertIsNot(records[0]["history"], records[1]["history"])
        self.assertEqual(records[0]["surface"], 0.4)

    def test_nonlinear_tail_refines_and_recovers_analytic_matrices(self):
        continuum = self.make_continuum()
        basis = continuum._plan_basis()
        with patch.object(continuum, "_sample_geometry", self.reciprocal_weights):
            result = continuum._converge_surface(
                0.4, basis, rtol=1e-10, atol=1e-12, max_shape=(512, 512)
            )
        history = result["quadrature"]["history"]
        self.assertGreater(len(history), 1)
        self.assertFalse(history[0]["converged"])
        self.assertTrue(history[-1]["converged"])
        root = np.sqrt(1 - 0.95**2)
        q = (1 - root) / 0.95
        zero, third, sixth = 1 / root, q**3 / root, q**6 / root
        mass = [[zero, np.sqrt(2) * third], [np.sqrt(2) * third, zero + sixth]]
        stiffness = [[0, 0], [0, (3 * 0.4 - 9)**2 * (zero - sixth)]]
        np.testing.assert_allclose(result["M0"], mass, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(result["K"], stiffness, rtol=1e-11, atol=1e-12)

    def test_grid_limit_reports_surface_unmet_tolerance_and_next_grid(self):
        continuum = self.make_continuum()
        basis = continuum._plan_basis()
        with patch.object(
            continuum, "_sample_geometry", side_effect=self.reciprocal_weights
        ) as sampler, patch.object(continuum, "_assemble_moments") as assembler:
            with self.assertRaises(RuntimeError) as caught:
                continuum._converge_surface(0.4, basis, max_shape=(26, 26))
        message = str(caught.exception)
        for text in ("s=0.4", "(26, 26)", "(52, 52)", "max_shape", "rtol", "ratio"):
            self.assertIn(text, message)
        self.assertEqual(sampler.call_count, 2)
        assembler.assert_not_called()

    def test_fixed_grid_is_verified_and_returned_without_replacement(self):
        continuum = self.make_continuum()
        basis = continuum._plan_basis()
        with patch.object(
            continuum, "_sample_geometry", side_effect=self.reciprocal_weights
        ) as sampler:
            result = continuum._converge_surface(
                0.4, basis, shape=(128, 128), max_shape=(256, 256)
            )
        self.assertEqual(sampler.call_count, 2)
        report = result["quadrature"]
        self.assertTrue(report["fixed_grid"])
        self.assertEqual(report["shape"], (128, 128))
        self.assertEqual(report["verification_shape"], (256, 256))
        grid = continuum._plan_angular_grid(basis, (128, 128))
        geometry = self.reciprocal_weights(0.4, grid["theta"], grid["zeta"])
        moments = continuum._fourier_moments(basis, grid, geometry)
        K, M0 = continuum._assemble_moments(basis, moments, geometry["iota"])
        np.testing.assert_array_equal(result["K"], K)
        np.testing.assert_array_equal(result["M0"], M0)

    def test_underresolved_fixed_grid_fails_instead_of_refining_silently(self):
        continuum = self.make_continuum()
        basis = continuum._plan_basis()
        with patch.object(
            continuum, "_sample_geometry", side_effect=self.reciprocal_weights
        ) as sampler:
            with self.assertRaisesRegex(RuntimeError, "Fixed-grid.*s=0.4.*ratio"):
                continuum._converge_surface(0.4, basis, shape=(13, 13))
        self.assertEqual(sampler.call_count, 2)

    def test_limit_must_allow_a_finer_verification_grid(self):
        continuum = self.make_continuum()
        basis = continuum._plan_basis()
        with self.assertRaisesRegex(RuntimeError, "No coarse/fine comparison"):
            continuum._converge_surface(
                0.4, basis, shape=(13, 13), max_shape=(13, 13)
            )

    def test_oversized_grid_is_rejected_before_grid_allocation(self):
        continuum = self.make_continuum()
        basis = continuum._plan_basis()
        with patch("firm3d.saw.stellgap.np.arange") as arange:
            with self.assertRaisesRegex(ValueError, "s=0.4.*exceeds max_shape"):
                continuum._converge_surface(
                    0.4, basis, shape=(10**9, 10**9), max_shape=(128, 128)
                )
            arange.assert_not_called()

    def test_comparison_scales_each_weight_and_handles_zero_moments(self):
        continuum = self.make_continuum()
        coarse = {}
        fine = {}
        for name, scale in (("A", 1e150), ("W0", 1e-150)):
            coarse[name + "_difference"] = np.eye(2) * scale
            coarse[name + "_sum"] = np.zeros((2, 2))
            fine[name + "_difference"] = np.eye(2) * scale
            fine[name + "_sum"] = np.array([[0, 1e-12], [1e-12, 0]]) * scale
        check = continuum._compare_moments(coarse, fine, rtol=1e-8, atol=1e-10)
        self.assertTrue(check["converged"])
        self.assertAlmostEqual(check["max_tolerance_ratio"], 0.01)
        self.assertFalse(
            continuum._compare_moments(coarse, fine, rtol=1e-8, atol=0)["converged"]
        )
        with np.errstate(all="raise"):
            unchanged = continuum._compare_moments(coarse, coarse, rtol=1e-8, atol=0)
        self.assertTrue(unchanged["converged"])
        self.assertEqual(unchanged["max_tolerance_ratio"], 0)
        # A zero mean after numerical underflow cannot certify convergence.
        coarse["A_difference"][:] = 0
        with self.assertRaisesRegex(ValueError, "A zero-moment scale"):
            continuum._compare_moments(coarse, coarse, rtol=1e-8, atol=1e-10)

    def test_every_required_moment_is_checked(self):
        continuum = self.make_continuum()
        coarse = {
            "A_difference": np.eye(2), "A_sum": np.zeros((2, 2)),
            "W0_difference": np.eye(2), "W0_sum": np.zeros((2, 2)),
        }
        for name in coarse:
            with self.subTest(name=name):
                fine = {key: values.copy() for key, values in coarse.items()}
                fine[name][0, 1] += 0.1
                check = continuum._compare_moments(coarse, fine, rtol=1e-8, atol=1e-10)
                self.assertFalse(check["converged"])
                self.assertEqual(check["worst_moment"], name)
                self.assertEqual(check["worst_pair"], (0, 1))

    def test_invalid_settings_are_rejected_before_sampling(self):
        continuum = self.make_continuum()
        basis = continuum._plan_basis()
        settings = [{"rtol": 0, "atol": 0}]
        for key in ("rtol", "atol"):
            for value in (-1, np.nan, np.inf, True, 1j, "small", [1e-8]):
                settings.append({key: value})
        for value in (None, 0, [1], [10, 0], [10.0, 10.0], [True, True]):
            settings.append({"max_shape": value})
        with patch.object(continuum, "_sample_geometry") as sampler:
            for keywords in settings:
                with self.subTest(settings=keywords):
                    with self.assertRaises(ValueError):
                        continuum._converge_surface(0.4, basis, **keywords)
            sampler.assert_not_called()

    def test_finer_grid_geometry_errors_include_surface_and_resolution(self):
        continuum = self.make_continuum()
        basis = continuum._plan_basis()

        def sample(surface, theta, zeta, expected_orientation):
            if len(theta) > 13:
                raise ValueError("Jg changes sign on the grid")
            return self.reciprocal_weights(surface, theta, zeta, expected_orientation)

        with patch.object(continuum, "_sample_geometry", side_effect=sample):
            with self.assertRaisesRegex(
                ValueError, r"s=0.4.*\(26, 26\).*Jg changes sign"
            ):
                continuum._converge_surface(0.4, basis)

    def test_real_equilibrium_result_agrees_with_another_finer_grid(self):
        continuum = Continuum(self.field, [0.5], [[1, 1], [2, -2], [0, -1]])
        basis = continuum._plan_basis()
        result = continuum._converge_surface(0.5, basis)
        shape = tuple(2 * count for count in result["quadrature"]["shape"])
        grid = continuum._plan_angular_grid(basis, shape)
        geometry = continuum._sample_geometry(0.5, grid["theta"], grid["zeta"])
        moments = continuum._fourier_moments(basis, grid, geometry)
        reference = continuum._assemble_moments(basis, moments, geometry["iota"])
        for actual, expected in zip((result["K"], result["M0"]), reference):
            np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
