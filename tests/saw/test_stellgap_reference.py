"""Geometric volume normalization, independent quadrature, and provenance."""

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


class TestReferenceField(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.field = BoozerRadialInterpolant(str(EQUILIBRIUM_FILE), order=3, no_K=True)

    def make_continuum(
        self, *, sign=1, iota=3.0, surfaces=(0.4,), modes=((1, 1),),
        density=None, reference_field=None, data_grid=None,
    ):
        """R=3+s*cos(chi), Z=sign*s*sin(chi), nu=0, chi=theta-3*zeta.

        For iota=3, Bg=abs(psi0)/s and |Jg|=s*R. Their volume average is
        2*abs(psi0), despite the integrable field singularity at the axis.
        """
        s = np.linspace(0, 1, 7)
        radius = np.column_stack((np.full_like(s, 3), s))
        height = np.column_stack((np.zeros_like(s), sign * s))
        if data_grid is None:
            data_grid = np.r_[0, (np.arange(6) + 0.5) / 6, 1]
        with patch.multiple(
            self.field,
            xm_b=np.array([0, 1]), xn_b=np.array([0, 3]),
            rmnc_splines=make_interp_spline(s, radius),
            zmns_splines=make_interp_spline(s, height),
            numns_splines=make_interp_spline(s, np.zeros_like(radius)),
            iota_spline=make_interp_spline(s, np.full_like(s, iota)),
            psi0=-0.8, s_half_ext=np.asarray(data_grid),
        ):
            return Continuum(
                self.field, surfaces, modes, density, reference_field=reference_field
            )

    def test_analytic_volume_and_reference_for_both_orientations(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                continuum = self.make_continuum(sign=sign)
                with patch.object(
                    continuum, "_sample_geometry", wraps=continuum._sample_geometry
                ) as sampler:
                    reference = continuum.get_reference_field(rtol=1e-12)
                self.assertAlmostEqual(reference["value"], 1.6)
                self.assertAlmostEqual(reference["volume"], 6 * np.pi**2)
                self.assertAlmostEqual(reference["field_integral"], 9.6 * np.pi**2)
                self.assertEqual(reference["orientation"], -sign)
                self.assertEqual(reference["source"], "geometric_volume_average")
                self.assertEqual(reference["domain"], (0, 1))
                self.assertTrue(reference["converged"])
                self.assertEqual(reference["equilibrium_accuracy"], "unverified")
                self.assertIn("half-grid", reference["endpoint_policy"])
                self.assertGreater(reference["min_jacobian_quality"], 0)
                for call in sampler.call_args_list:
                    self.assertGreater(call.args[0], 0)
                    self.assertLess(call.args[0], 1)

    def test_reference_does_not_depend_on_surfaces_modes_or_density(self):
        def unused_density(surface):
            raise AssertionError("Reference integration must not evaluate density.")

        first = self.make_continuum().get_reference_field()
        second = self.make_continuum(
            surfaces=(0.8, 0.1, 0.8), modes=((100, 1), (200, 4)),
            density=unused_density,
        ).get_reference_field()
        self.assertEqual(first, second)

    def test_explicit_reference_bypasses_geometry_and_full_volume_requirement(self):
        continuum = self.make_continuum(
            reference_field=5.7, data_grid=[0.2, 0.3, 0.6, 0.8]
        )
        with patch.object(continuum, "_sample_geometry") as sampler:
            reference = continuum.get_reference_field()
            sampler.assert_not_called()
        self.assertEqual(reference, {
            "value": 5.7, "units": "T", "source": "explicit",
            "domain": None, "converged": None,
        })

    def test_rejects_invalid_explicit_fields(self):
        for value in (0, -1, np.nan, np.inf, True, 1j, "5", [5]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "reference_field"):
                    self.make_continuum(reference_field=value)

    def test_rejects_restricted_native_data_even_with_synthetic_endpoints(self):
        for data in ([0.2, 0.3, 0.6, 0.8], [0, 0.2, 0.3, 0.7, 0.8, 1]):
            continuum = self.make_continuum(data_grid=data)
            with patch.object(continuum, "_sample_geometry") as sampler:
                with self.assertRaisesRegex(ValueError, "coverage.*reference_field"):
                    continuum.get_reference_field()
                sampler.assert_not_called()

    def test_rejects_restricted_spline_support(self):
        s = np.linspace(0.2, 0.8, 5)
        with patch.object(self.field, "iota_spline", make_interp_spline(s, s + 0.5)):
            continuum = Continuum(self.field, [0.4], [[1, 1]])
        with self.assertRaisesRegex(ValueError, "coverage.*reference_field"):
            continuum.get_reference_field()

    def test_cache_reuses_only_matching_settings_and_returns_copies(self):
        continuum = self.make_continuum()
        with patch.object(
            continuum, "_integrate_reference_field",
            wraps=continuum._integrate_reference_field,
        ) as integrate:
            first = continuum.get_reference_field()
            count = integrate.call_count
            first["value"] = -1
            first["history"][0]["relative_errors"]["radial"]["volume"] = -1
            second = continuum.get_reference_field()
            self.assertEqual(integrate.call_count, count)
            self.assertAlmostEqual(second["value"], 1.6)
            self.assertGreaterEqual(
                second["history"][0]["relative_errors"]["radial"]["volume"], 0
            )
            continuum.get_reference_field(rtol=1e-10)
            self.assertGreater(integrate.call_count, count)

    def test_snapshot_keeps_reference_independent_of_source_field_mutations(self):
        continuum = self.make_continuum()
        expected = continuum.get_reference_field()
        with patch.multiple(self.field, s_half_ext=np.array([0.2, 0.8]), psi0=7):
            # Different settings force a fresh integration of the copied data.
            actual = continuum.get_reference_field(rtol=1e-10)
        self.assertEqual(actual["value"], expected["value"])
        self.assertEqual(actual["native_radial_range"], expected["native_radial_range"])

    def test_ratio_cancellation_cannot_hide_radial_or_angular_integral_errors(self):
        for direction in ("radial", "angular"):
            continuum = self.make_continuum()

            def constant_field_geometry(surface, theta, zeta, orientation=None):
                jacobian = np.ones((len(theta), len(zeta)))
                if direction == "radial":
                    jacobian *= 1 + surface**4
                else:
                    jacobian *= 1 + 0.5 * np.cos(3 * theta[:, None])
                return {
                    "Jg": jacobian, "S": (3 * jacobian / 0.8)**2,
                    "orientation": 1, "min_jacobian_quality": 1.0,
                }

            with self.subTest(direction=direction), patch.object(
                continuum, "_sample_geometry", side_effect=constant_field_geometry
            ):
                reference = continuum.get_reference_field(radial_order=1, rtol=1e-10)
            errors = reference["history"][0]["relative_errors"][direction]
            self.assertLess(errors["value"], 1e-12)
            self.assertGreater(errors["volume"], 1e-10)
            self.assertGreater(errors["field_integral"], 1e-10)
            self.assertGreater(len(reference["history"]), 1)
            self.assertAlmostEqual(reference["value"], 3.0)
            expected_volume = (1.2 if direction == "radial" else 1) * (2 * np.pi)**2
            self.assertAlmostEqual(reference["volume"], expected_volume)

    def test_nonlinear_geometry_requires_angular_refinement(self):
        continuum = self.make_continuum(iota=0.5)
        reference = continuum.get_reference_field(rtol=1e-10)
        self.assertGreater(reference["shape"][0], 6)
        finer = continuum.get_reference_field(
            rtol=1e-12, radial_order=8, shape=reference["shape"]
        )
        self.assertLess(abs(reference["value"] / finer["value"] - 1), 1e-10)

    def test_limits_fail_before_sampling_and_failure_is_not_cached(self):
        continuum = self.make_continuum()
        for settings in ({"max_radial_order": 4}, {"max_shape": (3, 3)}):
            with patch.object(continuum, "_sample_geometry") as sampler:
                with self.assertRaisesRegex(RuntimeError, "verification needs"):
                    continuum.get_reference_field(**settings)
                sampler.assert_not_called()
            self.assertIsNone(continuum._reference_cache)
        self.assertTrue(continuum.get_reference_field()["converged"])

    def test_failed_refinement_reports_errors_without_claiming_convergence(self):
        continuum = self.make_continuum(iota=0.5)
        with self.assertRaisesRegex(RuntimeError, "Last comparison:.*relative_errors"):
            continuum.get_reference_field(rtol=1e-12, max_shape=(6, 6))
        self.assertIsNone(continuum._reference_cache)

    def test_invalid_settings(self):
        continuum = self.make_continuum()
        settings = [
            {"rtol": 0}, {"rtol": np.nan}, {"rtol": True},
            {"radial_order": 0}, {"radial_order": 1.5}, {"radial_order": True},
            {"max_radial_order": -1}, {"max_radial_order": False},
            {"shape": (2, 2)}, {"max_shape": (0, 2)}, {"max_shape": None},
        ]
        for setting in settings:
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                continuum.get_reference_field(**setting)

    def test_bad_geometry_keeps_reference_and_surface_context(self):
        continuum = self.make_continuum()
        with patch.object(
            continuum, "_sample_geometry", side_effect=ValueError("Jg changes sign")
        ):
            with self.assertRaisesRegex(ValueError, "Reference-field.*s=.*Jg"):
                continuum.get_reference_field()

    def test_orientation_is_checked_across_radial_nodes(self):
        continuum = self.make_continuum()
        sample = continuum._sample_coordinates

        def folded_coordinates(surface, theta, zeta):
            values = sample(surface, theta, zeta)
            if surface > 0.5:
                for name in ("R_s", "Z_s", "nu_s"):
                    values[name] *= -1
            return values

        with patch.object(
            continuum, "_sample_coordinates", side_effect=folded_coordinates
        ):
            with self.assertRaisesRegex(ValueError, "Reference-field.*orientation"):
                continuum.get_reference_field()

    def test_run_reuses_reference_and_explicit_value_does_not_change_eigenvalues(self):
        continuum = self.make_continuum()
        reference = continuum.get_reference_field(rtol=1e-10)
        with patch.object(continuum, "_integrate_reference_field") as integrate:
            result = continuum.run()
            integrate.assert_not_called()
        self.assertEqual(result["reference_field"], reference)
        explicit = self.make_continuum(reference_field=5.7).run()
        np.testing.assert_array_equal(explicit["eigenvalues"], result["eigenvalues"])
        self.assertEqual(explicit["reference_field"]["value"], 5.7)
        result["reference_field"]["value"] = -1
        self.assertEqual(continuum.get_reference_field(rtol=1e-10), reference)


if __name__ == "__main__":
    unittest.main()
