"""Frequency conventions, density conversion, and dominant cosine harmonics."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from scipy.interpolate import make_interp_spline

from firm3d.field.boozermagneticfield import BoozerRadialInterpolant
from firm3d.saw.stellgap import Continuum, ContinuumResult
from firm3d.util.constants import VACUUM_PERMEABILITY


EQUILIBRIUM_FILE = (
    Path(__file__).parents[1] / "test_files" / "boozmn_n3are_R7.75B5.7.nc"
)


class TestContinuumResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.field = BoozerRadialInterpolant(str(EQUILIBRIUM_FILE), order=3, no_K=True)

    def make_continuum(
        self, *, surfaces=(0.8, 0.2, 0.8), modes=((0, 0), (1, 0), (2, 0)),
        density=None, reference_field=2.0,
    ):
        s = np.linspace(0, 1, 7)
        with patch.multiple(
            self.field,
            G_spline=make_interp_spline(s, -3 - s),
            I_spline=make_interp_spline(s, np.full_like(s, 0.25)),
            iota_spline=make_interp_spline(s, np.full_like(s, 0.5)),
        ):
            return Continuum(
                self.field, surfaces, modes, density, reference_field=reference_field
            )

    def solve(self, continuum, *, K=None, M0=None, keep_eigenvectors=False):
        """Use known matrices to isolate result handling from geometric quadrature."""
        if M0 is None:
            M0 = np.diag([2.0, 5.0, 3.0])
        if K is None:
            K = M0 @ np.diag([0.0, 9.0, 4.0])
        matrices = {
            "K": K, "M0": M0, "iota": 0.5, "orientation": -1,
            "min_jacobian_quality": 1.0,
            "quadrature": {"converged": True, "verification_shape": (158, 162)},
        }
        with patch.object(continuum, "_converge_surface", return_value=matrices):
            return continuum.run(keep_eigenvectors=keep_eigenvectors)

    def test_known_frequencies_and_dominant_harmonics_follow_sorted_eigenpairs(self):
        continuum = self.make_continuum(density=2e-7)
        result = self.solve(continuum, keep_eigenvectors=True)
        self.assertIsInstance(result, ContinuumResult)
        expected_values = np.tile([0.0, 4.0, 9.0], (3, 1))
        np.testing.assert_allclose(result.eigenvalues, expected_values, atol=1e-14)
        combination = -3 - continuum.surfaces + 0.5 * 0.25
        expected_normalized = abs(combination[:, None]) * np.sqrt(expected_values) / 4
        np.testing.assert_allclose(result.normalized_frequencies, expected_normalized)
        expected_khz = np.sqrt(expected_values / (VACUUM_PERMEABILITY * 2e-7))
        expected_khz /= 2 * np.pi * 1000
        np.testing.assert_allclose(result.frequencies_khz, expected_khz)
        expected_modes = np.tile([[0, 0], [2, 0], [1, 0]], (3, 1, 1))
        np.testing.assert_array_equal(result.dominant_modes, expected_modes)
        np.testing.assert_array_equal(result.normalized_frequencies[:, 0], 0)
        np.testing.assert_array_equal(result.frequencies_khz[:, 0], 0)
        np.testing.assert_array_equal(result.surfaces, [0.8, 0.2, 0.8])
        np.testing.assert_allclose(result.normalization["G_plus_iota_I"], combination)
        omega_A = 4 / (abs(combination) * np.sqrt(VACUUM_PERMEABILITY * 2e-7))
        np.testing.assert_allclose(
            result.normalized_frequencies * omega_A[:, None],
            2 * np.pi * 1000 * result.frequencies_khz,
        )

    def test_fourfold_density_halves_khz_without_geometry_or_solving(self):
        continuum = self.make_continuum()
        result = self.solve(continuum, keep_eigenvectors=True)
        with patch.object(continuum, "_sample_geometry") as geometry:
            with patch.object(continuum, "_solve_surface") as solver:
                first = result.with_density(1e-7)
                second = first.with_density(4e-7)
                geometry.assert_not_called()
                solver.assert_not_called()
        np.testing.assert_allclose(second.frequencies_khz, first.frequencies_khz / 2)
        for name in ("eigenvalues", "normalized_frequencies", "eigenvectors",
                     "dominant_modes"):
            self.assertIs(getattr(first, name), getattr(result, name))
            self.assertIs(getattr(second, name), getattr(result, name))
        self.assertIsNone(result.density)
        self.assertIsNone(result.frequencies_khz)
        np.testing.assert_array_equal(first.density, [1e-7] * 3)
        np.testing.assert_array_equal(second.density, [4e-7] * 3)
        self.assertFalse(
            np.shares_memory(first.frequencies_khz, second.frequencies_khz)
        )

    def test_initializer_and_later_density_use_the_same_conversion(self):
        density = Mock(side_effect=lambda s: 1e-7 * (1 + s))
        initialized = self.solve(self.make_continuum(density=density))
        self.assertEqual(
            [call.args[0] for call in density.call_args_list], [0.8, 0.2, 0.8]
        )
        base = self.solve(self.make_continuum())
        density.reset_mock()
        converted = base.with_density(density)
        self.assertEqual(
            [call.args[0] for call in density.call_args_list], [0.8, 0.2, 0.8]
        )
        np.testing.assert_array_equal(converted.density, initialized.density)
        np.testing.assert_array_equal(
            converted.frequencies_khz, initialized.frequencies_khz
        )
        np.testing.assert_array_equal(
            converted.normalized_frequencies, initialized.normalized_frequencies
        )

    def test_reference_scaling_changes_only_normalized_frequencies(self):
        first = self.solve(self.make_continuum(reference_field=2, density=1e-7))
        second = self.solve(self.make_continuum(reference_field=4, density=1e-7))
        np.testing.assert_allclose(
            second.normalized_frequencies, first.normalized_frequencies / 4
        )
        np.testing.assert_array_equal(second.eigenvalues, first.eigenvalues)
        np.testing.assert_array_equal(second.frequencies_khz, first.frequencies_khz)
        np.testing.assert_array_equal(second.dominant_modes, first.dominant_modes)

    def test_magnetic_component_sign_does_not_change_normalized_frequencies(self):
        original = self.make_continuum()
        reversed_sign = self.make_continuum()
        for name in ("G", "I"):
            reversed_sign._flux_splines[name].c *= -1
        first = self.solve(original)
        second = self.solve(reversed_sign)
        np.testing.assert_array_equal(
            second.normalized_frequencies, first.normalized_frequencies
        )
        np.testing.assert_array_equal(
            second.normalization["G_plus_iota_I"], -first.normalization["G_plus_iota_I"]
        )

    def test_labels_are_available_without_retaining_eigenvectors(self):
        continuum = self.make_continuum()
        compact = self.solve(continuum)
        retained = self.solve(continuum, keep_eigenvectors=True)
        self.assertIsNone(compact.eigenvectors)
        np.testing.assert_array_equal(compact.dominant_modes, retained.dominant_modes)
        self.assertEqual(compact.mode_convention["nfp"], 3)
        self.assertEqual(compact.mode_convention["mode_family"], 0)
        self.assertIn("sqrt(2)", compact.mode_convention["basis"])
        self.assertIn("no branch tracking", compact.mode_convention["ordering"])

    def test_labels_use_normalized_basis_coefficients_and_preserve_input_signs(self):
        continuum = self.make_continuum(modes=((0, 0), (-1, 0)), surfaces=(0.4,))
        vectors = np.array([[0.8, -0.6], [0.6, 0.8]])
        K = vectors @ np.diag([1.0, 3.0]) @ vectors.T
        result = self.solve(continuum, K=K, M0=np.eye(2), keep_eigenvectors=True)
        # The first mode has |c0| > |c1|, but |c0| < sqrt(2)*|c1|.
        np.testing.assert_array_equal(result.dominant_modes[0], [[0, 0], [-1, 0]])

    def test_exact_coefficient_tie_uses_input_order(self):
        continuum = self.make_continuum(modes=((3, 0), (1, 0)), surfaces=(0.4,))
        vectors = np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2)
        K = vectors @ np.diag([1.0, 3.0]) @ vectors.T
        with patch(
            "firm3d.saw.stellgap.eigh", return_value=(np.array([1.0, 3.0]), vectors)
        ):
            result = self.solve(continuum, K=K, M0=np.eye(2))
        np.testing.assert_array_equal(result.dominant_modes[0], [[3, 0], [3, 0]])

    def test_degenerate_vectors_preserve_subspace_without_stable_labels(self):
        continuum = self.make_continuum(surfaces=(0.4,))
        K, M0 = np.diag([0.0, 4.0, 4.0]), np.eye(3)
        first = self.solve(continuum, K=K, M0=M0, keep_eigenvectors=True)
        angle = np.pi / 3
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ])
        with patch(
            "firm3d.saw.stellgap.eigh",
            return_value=(np.array([0.0, 4.0, 4.0]), vectors),
        ):
            second = self.solve(
                self.make_continuum(surfaces=(0.4,)), K=K, M0=M0,
                keep_eigenvectors=True,
            )
        np.testing.assert_array_equal(first.eigenvalues, second.eigenvalues)
        first_space = first.eigenvectors[0, :, 1:]
        second_space = second.eigenvectors[0, :, 1:]
        np.testing.assert_allclose(
            first_space @ first_space.T, second_space @ second_space.T, atol=1e-15
        )
        self.assertFalse(np.array_equal(first.dominant_modes, second.dominant_modes))

    def test_invalid_density_profiles_report_surface_and_leave_result_unchanged(self):
        result = self.solve(self.make_continuum(density=1e-7))
        before = result.frequencies_khz.copy()
        for value in (0, -1, True, np.nan, np.inf, "1e-7", 1j, [1e-7]):
            for density in (value, lambda s: value):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, "density.*s=0.8"):
                        result.with_density(density)
        with self.assertRaisesRegex(ValueError, "density.*s=0.2"):
            result.with_density(lambda s: 1e-7 if s > 0.5 else -1)
        with self.assertRaisesRegex(ValueError, "density evaluation failed at s=0.8"):
            result.with_density(Mock(side_effect=ZeroDivisionError))
        np.testing.assert_array_equal(result.frequencies_khz, before)
        np.testing.assert_array_equal(result.density, [1e-7] * 3)

    def test_none_density_removes_only_dimensional_results(self):
        result = self.solve(self.make_continuum(density=1e-7))
        removed = result.with_density(None)
        self.assertIsNone(removed.density)
        self.assertIsNone(removed.frequencies_khz)
        self.assertIs(removed.normalized_frequencies, result.normalized_frequencies)
        self.assertIsNotNone(result.frequencies_khz)

    def test_invalid_normalization_fails_before_quadrature_or_solving(self):
        s = np.linspace(0, 1, 7)
        for name, value in (("G", np.nan), ("I", np.inf), ("iota", np.nan)):
            continuum = self.make_continuum()
            continuum._flux_splines[name].c[:] = value
            with self.subTest(name=name), patch.object(continuum, "_converge_surface"):
                with self.assertRaisesRegex(ValueError, name + ".*s=0.8"):
                    continuum.run()
        for G, I in ((-0.125, 0.25), (1e308, 1e308)):
            continuum = self.make_continuum()
            continuum._flux_splines["G"] = make_interp_spline(s, np.full_like(s, G))
            continuum._flux_splines["I"] = make_interp_spline(s, np.full_like(s, I))
            if G > 0:
                continuum._flux_splines["iota"].c[:] = 4
            with patch.object(continuum, "_converge_surface") as quadrature:
                with self.assertRaisesRegex(ValueError, "G\\+iota\\*I.*s=0.8"):
                    continuum.run()
                quadrature.assert_not_called()
        for field in (1e-300, 1e300):
            continuum = self.make_continuum(reference_field=field)
            with self.assertRaisesRegex(ValueError, "normalization factor.*s=0.8"):
                continuum.run()

    def test_conversion_keeps_representable_values_when_squared_formula_overflows(self):
        result = self.solve(self.make_continuum())
        result.eigenvalues[:] = 1e300
        converted = result.with_density(1e-300)
        expected = 1e300 / (2 * np.pi * 1e3 * np.sqrt(VACUUM_PERMEABILITY))
        np.testing.assert_allclose(converted.frequencies_khz, expected)
        with self.assertRaisesRegex(ValueError, "Frequency conversion.*s=0.8"):
            result.with_density(np.nextafter(0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
