"""Geometric continuum weights, orientation, and singularity diagnostics."""

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


class TestAnalyticGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.field = BoozerRadialInterpolant(str(EQUILIBRIUM_FILE), order=3, no_K=True)

    def setUp(self):
        self.theta = np.linspace(0, 2 * np.pi, 31, endpoint=False)
        self.zeta = np.linspace(0, 2 * np.pi / 3, 17, endpoint=False)

    def make_continuum(
        self, vertical_sign=1, nu_amplitude=0.2, psi0=-0.8, density=None
    ):
        """A circular torus in sheared angles, with chi = theta - 3*zeta.

        R = 3 + s*cos(chi), Z = vertical_sign*s*sin(chi),
        nu = nu_amplitude*s**2*sin(chi). Cubic splines reproduce these exactly.
        """
        s = np.linspace(0, 1, 7)
        radius = np.column_stack((np.full_like(s, 3), s))
        height = np.column_stack((np.zeros_like(s), vertical_sign * s))
        nu = np.column_stack((np.zeros_like(s), nu_amplitude * s**2))
        with patch.multiple(
            self.field,
            xm_b=np.array([0, 1]),
            xn_b=np.array([0, 3]),
            rmnc_splines=make_interp_spline(s, radius),
            zmns_splines=make_interp_spline(s, height),
            numns_splines=make_interp_spline(s, nu),
            iota_spline=make_interp_spline(s, 0.7 + 0.1 * s),
            psi0=psi0,
        ):
            return Continuum(self.field, [0.4, 0.8], [[1, 1]], density=density)

    def test_analytic_weights_and_quality_for_both_orientations(self):
        chi = self.theta[:, None] - 3 * self.zeta[None, :]
        for vertical_sign in (-1, 1):
            continuum = self.make_continuum(vertical_sign=vertical_sign)
            for s in continuum.surfaces:
                with self.subTest(vertical_sign=vertical_sign, surface=s):
                    geometry = continuum._sample_geometry(s, self.theta, self.zeta)
                    radius = 3 + s * np.cos(chi)
                    shear = 0.2 * s**2
                    shear_derivative = 0.4 * s
                    iota = 0.7 + 0.1 * s
                    Jg = -vertical_sign * s * radius
                    H = s**2 * radius**2
                    S = (iota - 3)**2 * s**2 + radius**2 * (
                        1 - (iota - 3) * shear * np.cos(chi)
                    )**2
                    radial_length = np.sqrt(
                        1 + (radius * shear_derivative * np.sin(chi))**2
                    )
                    poloidal_length = np.sqrt(s**2 + (radius * shear * np.cos(chi))**2)
                    toroidal_length = np.sqrt(
                        9 * s**2 + radius**2 * (1 + 3 * shear * np.cos(chi))**2
                    )
                    quality = s * radius / (
                        radial_length * poloidal_length * toroidal_length
                    )
                    expected = {
                        "Jg": Jg,
                        "H": H,
                        "S": S,
                        "A": H / (np.abs(Jg) * S),
                        "W0": H * np.abs(Jg) / (0.8**2 * S),
                        "jacobian_quality": quality,
                    }
                    for name, values in expected.items():
                        np.testing.assert_allclose(
                            geometry[name],
                            values,
                            rtol=1e-12,
                            atol=1e-12,
                            err_msg=name,
                        )
                    self.assertEqual(geometry["orientation"], -vertical_sign)
                    self.assertAlmostEqual(
                        geometry["min_jacobian_quality"], quality.min()
                    )
                    self.assertAlmostEqual(geometry["iota"], iota)

    def test_reports_small_quality_for_strongly_sheared_coordinates(self):
        continuum = self.make_continuum(nu_amplitude=100)
        geometry = continuum._sample_geometry(0.8, self.theta, self.zeta)
        self.assertGreater(geometry["min_jacobian_quality"], 0)
        self.assertLess(geometry["min_jacobian_quality"], 1e-4)
        self.assertTrue(np.all(geometry["A"] > 0))
        self.assertTrue(np.all(geometry["W0"] > 0))

    def test_checks_expected_orientation_without_remembering_call_order(self):
        continuum = self.make_continuum()
        outer = continuum._sample_geometry(0.8, self.theta, self.zeta)
        inner = continuum._sample_geometry(
            0.4, self.theta, self.zeta, expected_orientation=outer["orientation"]
        )
        self.assertEqual(inner["orientation"], -1)
        with self.assertRaisesRegex(ValueError, "orientation -1 at s=0.4.*expected"):
            continuum._sample_geometry(
                0.4, self.theta, self.zeta, expected_orientation=1
            )
        for orientation in (0, 2, True, 1.0, "-1"):
            with self.subTest(orientation=orientation):
                with self.assertRaisesRegex(ValueError, "expected_orientation"):
                    continuum._sample_geometry(0.4, self.theta, self.zeta, orientation)

    def test_does_not_repair_zero_or_sign_changing_jacobians(self):
        continuum = self.make_continuum()
        for case in ("zero", "sign change"):
            with self.subTest(case=case):
                values = continuum._sample_coordinates(0.4, self.theta, self.zeta)
                for name in ("R_s", "Z_s", "nu_s"):
                    if case == "zero":
                        values[name][0, :] = 0
                    else:
                        values[name][0, :] *= -1
                message = "Jg must be nonzero" if case == "zero" else "Jg changes sign"
                with patch.object(
                    continuum, "_sample_coordinates", return_value=values
                ):
                    with self.assertRaisesRegex(ValueError, message + ".*s=0.4"):
                        continuum._sample_geometry(0.4, self.theta, self.zeta)

    def test_rejects_degenerate_surface_tangents(self):
        continuum = self.make_continuum()
        values = continuum._sample_coordinates(0.4, self.theta, self.zeta)
        for name in ("R_theta", "Z_theta", "nu_theta"):
            values[name][:] = 0
        with patch.object(continuum, "_sample_coordinates", return_value=values):
            with self.assertRaisesRegex(ValueError, "H must be positive at s=0.4"):
                continuum._sample_geometry(0.4, self.theta, self.zeta)

    def test_rejects_nonfinite_coordinates_and_iota(self):
        continuum = self.make_continuum()
        for name, invalid in (("Z", np.nan), ("R_s", np.inf), ("nu_zeta", np.nan)):
            with self.subTest(name=name):
                values = continuum._sample_coordinates(0.4, self.theta, self.zeta)
                values[name][0, 0] = invalid
                with patch.object(
                    continuum, "_sample_coordinates", return_value=values
                ):
                    with self.assertRaisesRegex(ValueError, name + ".*finite at s=0.4"):
                        continuum._sample_geometry(0.4, self.theta, self.zeta)
        values = continuum._sample_coordinates(0.4, self.theta, self.zeta)
        values["iota"] = np.nan
        with patch.object(continuum, "_sample_coordinates", return_value=values):
            with self.assertRaisesRegex(ValueError, "iota must be finite at s=0.4"):
                continuum._sample_geometry(0.4, self.theta, self.zeta)

    def test_reports_floating_point_failure_with_surface(self):
        continuum = self.make_continuum()
        values = continuum._sample_coordinates(0.4, self.theta, self.zeta)
        values["R"][:] = 1e200
        values["nu_s"][:] = 1e200
        with patch.object(continuum, "_sample_coordinates", return_value=values):
            with self.assertRaisesRegex(
                ValueError, "Geometry calculation failed at s=0.4"
            ):
                continuum._sample_geometry(0.4, self.theta, self.zeta)

    def test_weights_are_independent_of_density_and_magnetic_components(self):
        def density(surface):
            raise AssertionError("Geometry must not evaluate density.")

        continuum = self.make_continuum(density=density)
        original = continuum._sample_geometry(0.4, self.theta, self.zeta)
        continuum._flux_splines["G"].c[:] = -123
        continuum._flux_splines["I"].c[:] = 456
        changed = continuum._sample_geometry(0.4, self.theta, self.zeta)
        for name in ("A", "W0", "Jg"):
            np.testing.assert_array_equal(changed[name], original[name])

    def test_only_mass_weight_depends_on_flux_magnitude_not_its_sign(self):
        original = self.make_continuum()._sample_geometry(0.4, self.theta, self.zeta)
        for flux in (0.8, -1.6):
            with self.subTest(psi0=flux):
                continuum = self.make_continuum(psi0=flux)
                changed = continuum._sample_geometry(0.4, self.theta, self.zeta)
                np.testing.assert_array_equal(changed["A"], original["A"])
                np.testing.assert_allclose(
                    changed["W0"], original["W0"] * (0.8 / flux)**2
                )
                self.assertEqual(changed["orientation"], original["orientation"])


class TestEquilibriumGeometry(unittest.TestCase):
    def test_matches_cartesian_metric_identities_on_a_real_equilibrium(self):
        field = BoozerRadialInterpolant(str(EQUILIBRIUM_FILE), order=3, no_K=True)
        continuum = Continuum(field, [0.3, 0.8], [[1, 1], [2, 4]])
        grid = continuum._plan_angular_grid(continuum._plan_basis())
        orientation = None
        for surface in continuum.surfaces:
            with self.subTest(surface=surface):
                values = continuum._sample_coordinates(
                    surface, grid["theta"], grid["zeta"]
                )
                geometry = continuum._sample_geometry(
                    surface, grid["theta"], grid["zeta"], orientation
                )
                orientation = geometry["orientation"]
                cosine = np.cos(values["phi"])
                sine = np.sin(values["phi"])
                tangents = []
                for suffix in ("_s", "_theta", "_zeta"):
                    tangents.append(
                        np.stack(
                            (
                                values["R" + suffix] * cosine
                                - values["R"] * values["phi" + suffix] * sine,
                                values["R" + suffix] * sine
                                + values["R"] * values["phi" + suffix] * cosine,
                                values["Z" + suffix],
                            ),
                            axis=-1,
                        )
                    )
                basis = np.stack(tangents, axis=-2)
                metric = basis @ np.swapaxes(basis, -1, -2)
                H = metric[:, :, 1, 1] * metric[:, :, 2, 2] - metric[:, :, 1, 2]**2
                iota = values["iota"]
                S = (
                    metric[:, :, 2, 2]
                    + 2 * iota * metric[:, :, 1, 2]
                    + iota**2 * metric[:, :, 1, 1]
                )
                np.testing.assert_allclose(
                    geometry["Jg"], np.linalg.det(basis), rtol=1e-11
                )
                np.testing.assert_allclose(
                    geometry["Jg"]**2, np.linalg.det(metric), rtol=1e-11
                )
                np.testing.assert_allclose(geometry["H"], H, rtol=1e-11)
                np.testing.assert_allclose(geometry["S"], S, rtol=1e-11)
                np.testing.assert_allclose(
                    geometry["H"] / geometry["Jg"]**2,
                    np.linalg.inv(metric)[:, :, 0, 0],
                    rtol=1e-11,
                )
                self.assertGreater(geometry["min_jacobian_quality"], 0)
                self.assertLessEqual(geometry["jacobian_quality"].max(), 1 + 1e-14)


if __name__ == "__main__":
    unittest.main()
