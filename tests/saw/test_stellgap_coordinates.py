"""Check local coordinate sampling against analytic and real equilibria."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

from firm3d.field.boozermagneticfield import BoozerRadialInterpolant
from firm3d.saw.stellgap import Continuum


EQUILIBRIUM_FILE = (
    Path(__file__).parents[1] / "test_files" / "boozmn_n3are_R7.75B5.7.nc"
)


class TestAnalyticCoordinates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.field = BoozerRadialInterpolant(str(EQUILIBRIUM_FILE), order=3, no_K=True)

    def setUp(self):
        s = np.linspace(0, 1, 7)
        zeros = np.zeros_like(s)
        radius = np.column_stack((10 + 0.1 * s, s**2, 0.25 * s**3))
        height = np.column_stack((zeros, 0.7 * s + 0.2 * s**2, -0.4 * s**3))
        nu = np.column_stack((zeros, 0.03 * s**2, 0.02 * s**3))
        patcher = patch.multiple(
            self.field,
            xm_b=np.array([0, 2, 1]),
            xn_b=np.array([0, 3, -6]),
            rmnc_splines=make_interp_spline(s, radius),
            zmns_splines=make_interp_spline(s, height),
            numns_splines=make_interp_spline(s, nu),
            iota_spline=make_interp_spline(s, 0.6 + 0.1 * s),
            G_spline=make_interp_spline(s, 5 + 0.2 * s),
            I_spline=make_interp_spline(s, 0.1 - 0.01 * s),
            psi0=-0.8,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.theta = np.array([0.0, 0.13, 0.9, 2.7])
        self.zeta = np.array([-0.2, 0.05, 0.7])

    def test_values_and_all_derivatives(self):
        continuum = Continuum(self.field, [0.3, 0.6], [[1, 1]])
        for s in continuum.surfaces:
            with self.subTest(surface=s):
                values = continuum._sample_coordinates(s, self.theta, self.zeta)
                alpha = 2 * self.theta[:, None] - 3 * self.zeta[None, :]
                beta = self.theta[:, None] + 6 * self.zeta[None, :]
                ca, sa = np.cos(alpha), np.sin(alpha)
                cb, sb = np.cos(beta), np.sin(beta)
                expected = {
                    "R": 10 + 0.1 * s + s**2 * ca + 0.25 * s**3 * cb,
                    "R_s": 0.1 + 2 * s * ca + 0.75 * s**2 * cb,
                    "R_theta": -2 * s**2 * sa - 0.25 * s**3 * sb,
                    "R_zeta": 3 * s**2 * sa - 1.5 * s**3 * sb,
                    "Z": (0.7 * s + 0.2 * s**2) * sa - 0.4 * s**3 * sb,
                    "Z_s": (0.7 + 0.4 * s) * sa - 1.2 * s**2 * sb,
                    "Z_theta": (1.4 * s + 0.4 * s**2) * ca - 0.4 * s**3 * cb,
                    "Z_zeta": -(2.1 * s + 0.6 * s**2) * ca - 2.4 * s**3 * cb,
                    "nu": 0.03 * s**2 * sa + 0.02 * s**3 * sb,
                    "nu_s": 0.06 * s * sa + 0.06 * s**2 * sb,
                    "nu_theta": 0.06 * s**2 * ca + 0.02 * s**3 * cb,
                    "nu_zeta": -0.09 * s**2 * ca + 0.12 * s**3 * cb,
                }
                expected["phi"] = self.zeta[None, :] - expected["nu"]
                expected["phi_s"] = -expected["nu_s"]
                expected["phi_theta"] = -expected["nu_theta"]
                expected["phi_zeta"] = 1 - expected["nu_zeta"]
                for name, array in expected.items():
                    np.testing.assert_allclose(
                        values[name], array, rtol=1e-12, atol=1e-12, err_msg=name
                    )
                self.assertAlmostEqual(values["iota"], 0.6 + 0.1 * s)
                self.assertAlmostEqual(values["G"], 5 + 0.2 * s)
                self.assertAlmostEqual(values["I"], 0.1 - 0.01 * s)
                self.assertEqual(values["psi0"], -0.8)

    def test_sampling_is_local_and_leaves_field_unchanged(self):
        points = np.array([[0.4, 0.1, 0.2]])
        self.field.set_points(points)
        communicator = Mock()
        with patch.object(self.field, "comm", communicator):
            continuum = Continuum(self.field, [0.3], [[1, 1]])
            continuum._sample_coordinates(0.3, self.theta, self.zeta)
            self.assertEqual(communicator.mock_calls, [])
            self.assertIs(self.field.comm, communicator)
        np.testing.assert_array_equal(self.field.get_points_ref(), points)
        self.assertTrue(self.field.rmnc_splines.extrapolate)

    def test_snapshot_is_independent_of_later_field_changes(self):
        continuum = Continuum(self.field, [0.3], [[1, 1]])
        before = continuum._sample_coordinates(0.3, self.theta, self.zeta)
        for name in (
            "rmnc_splines",
            "zmns_splines",
            "numns_splines",
            "iota_spline",
            "G_spline",
            "I_spline",
        ):
            getattr(self.field, name).c[:] = 100
        self.field.xm_b[:] = 7
        self.field.xn_b[:] = 12
        self.field.psi0 = 2
        after = continuum._sample_coordinates(0.3, self.theta, self.zeta)
        for name in before:
            np.testing.assert_array_equal(after[name], before[name], err_msg=name)

    def test_grid_cache_reuses_factors_and_handles_changed_angles(self):
        continuum = Continuum(self.field, [0.3, 0.6], [[1, 1]])
        before = continuum._sample_coordinates(0.3, self.theta, self.zeta)
        cached_cosine = continuum._cos_theta
        continuum._sample_coordinates(0.6, self.theta.copy(), self.zeta.copy())
        self.assertIs(continuum._cos_theta, cached_cosine)

        self.theta[0] += 0.2
        after = continuum._sample_coordinates(0.3, self.theta, self.zeta)
        self.assertIsNot(continuum._cos_theta, cached_cosine)
        np.testing.assert_allclose(after["R"][1:], before["R"][1:])
        self.assertFalse(np.allclose(after["R"][0], before["R"][0]))

        single = continuum._sample_coordinates(0.3, [self.theta[1]], [self.zeta[1]])
        self.assertEqual(single["R"].shape, (1, 1))
        self.assertAlmostEqual(single["R"][0, 0], before["R"][1, 1])

    def test_rejects_discontinuous_first_derivatives(self):
        linear = make_interp_spline([0, 1], np.zeros((2, 3)), k=1)
        repeated_knot = BSpline(
            [0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1], np.zeros((7, 3)), k=3
        )
        for spline in (linear, repeated_knot):
            with self.subTest(degree=spline.k):
                with patch.object(self.field, "rmnc_splines", spline):
                    with self.assertRaisesRegex(ValueError, "continuous first"):
                        Continuum(self.field, [0.3], [[1, 1]])

    def test_respects_common_spline_domain(self):
        s = np.linspace(0.2, 0.8, 7)
        with patch.object(self.field, "G_spline", make_interp_spline(s, 5 + s)):
            for surface in (0.1, 0.9):
                with self.subTest(surface=surface):
                    with self.assertRaisesRegex(ValueError, "common.*interval"):
                        Continuum(self.field, [surface], [[1, 1]])
            continuum = Continuum(self.field, [0.2, 0.8], [[1, 1]])
            for surface in (0.1, 0.9):
                with self.assertRaisesRegex(ValueError, "surface must lie"):
                    continuum._sample_coordinates(surface, self.theta, self.zeta)

    def test_rejects_invalid_sampling_coordinates(self):
        continuum = Continuum(self.field, [0.3], [[1, 1]])
        for surface in (0, 1.1, np.nan, np.inf, [0.3], 0.3j):
            with self.subTest(surface=surface):
                with self.assertRaisesRegex(ValueError, "surface"):
                    continuum._sample_coordinates(surface, self.theta, self.zeta)
        for angles in ([], [[0.1]], [np.nan], [np.inf], [0.1j], ["0.1"]):
            with self.subTest(angles=angles):
                with self.assertRaisesRegex(ValueError, "theta"):
                    continuum._sample_coordinates(0.3, angles, self.zeta)
                with self.assertRaisesRegex(ValueError, "zeta"):
                    continuum._sample_coordinates(0.3, self.theta, angles)


class TestEquilibriumCoordinates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.field = BoozerRadialInterpolant(str(EQUILIBRIUM_FILE), order=3, no_K=True)

    def test_matches_field_coordinates_and_radial_finite_differences(self):
        continuum = Continuum(self.field, [0.31, 0.67], [[1, 1]])
        theta = np.array([0.1, 1.2, 2.3])
        zeta = np.array([0.2, 0.4, 0.8, 1.1])
        theta_mesh, zeta_mesh = np.meshgrid(theta, zeta, indexing="ij")
        step = 1e-5
        for surface in continuum.surfaces:
            with self.subTest(surface=surface):
                values = continuum._sample_coordinates(surface, theta, zeta)
                points = np.column_stack(
                    (
                        np.full(theta_mesh.size, surface),
                        theta_mesh.ravel(),
                        zeta_mesh.ravel(),
                    )
                )
                self.field.set_points(points)
                for name in ("R", "Z", "nu"):
                    expected = getattr(self.field, name)().reshape(theta_mesh.shape)
                    np.testing.assert_allclose(
                        values[name], expected, rtol=1e-12, atol=1e-12
                    )

                lower = continuum._sample_coordinates(surface - step, theta, zeta)
                upper = continuum._sample_coordinates(surface + step, theta, zeta)
                for name in ("R", "Z", "nu", "phi"):
                    difference = (upper[name] - lower[name]) / (2 * step)
                    np.testing.assert_allclose(
                        values[name + "_s"],
                        difference,
                        rtol=1e-6,
                        atol=1e-8,
                        err_msg=name,
                    )


if __name__ == "__main__":
    unittest.main()
