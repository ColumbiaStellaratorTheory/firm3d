"""Input contracts for the continuum solver, using a real radial interpolant."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from firm3d.field.boozermagneticfield import BoozerRadialInterpolant
from firm3d.saw.stellgap import Continuum


class TestContinuumInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filename = (
            Path(__file__).parents[1] / "test_files" / "boozmn_n3are_R7.75B5.7.nc"
        )
        cls.field = BoozerRadialInterpolant(str(filename), order=3, no_K=True)

    def test_copies_arrays_without_reordering_or_changing_field(self):
        surfaces = np.array([0.8, 0.2, 0.8, 1.0])
        modes = np.array([[2, 1], [3, 4], [0, -1]])
        points = np.array([[0.4, 0.1, 0.2]])
        self.field.set_points(points)
        continuum = Continuum(self.field, surfaces, modes)

        self.assertIs(continuum.field, self.field)
        self.assertIsNone(continuum.density)
        self.assertEqual(continuum.mode_family, 1)
        np.testing.assert_array_equal(continuum.surfaces, surfaces)
        np.testing.assert_array_equal(continuum.modes, modes)
        np.testing.assert_array_equal(self.field.get_points_ref(), points)
        self.assertIsNone(self.field.comm)

        surfaces[0] = 0.1
        modes[0, 0] = 99
        self.assertEqual(continuum.surfaces[0], 0.8)
        self.assertEqual(continuum.modes[0, 0], 2)

    def test_cosine_families_include_both_toroidal_residues(self):
        cases = [
            (4, [[1, 1], [2, 3], [3, 5], [4, 7]], 1),
            (4, [[1, 2], [2, 6], [3, 10]], 2),
            (5, [[1, 1], [2, 4], [3, -1]], 1),
            (4, [[0, 0], [1, 4], [2, -4]], 0),
            (1, [[1, 0], [2, 5], [3, -2]], 0),
        ]
        for nfp, modes, expected_family in cases:
            with self.subTest(nfp=nfp, modes=modes):
                with patch.object(self.field, "nfp", nfp):
                    continuum = Continuum(self.field, [0.5], modes)
                self.assertEqual(continuum.mode_family, expected_family)

    def test_rejects_incompatible_family(self):
        with patch.object(self.field, "nfp", 4):
            with self.assertRaisesRegex(ValueError, "family 2; expected family 1"):
                Continuum(self.field, [0.5], [[1, 1], [2, 2]])

    def test_rejects_duplicate_cosines(self):
        for modes in (
            [[1, 1], [1, 1]],
            [[1, 1], [-1, -1]],
            [[0, 1], [0, -1]],
            [[0, 0], [0, 0]],
        ):
            with self.subTest(modes=modes):
                with self.assertRaisesRegex(ValueError, "duplicates a cosine"):
                    Continuum(self.field, [0.5], modes)

    def test_rejects_invalid_mode_tables(self):
        cases = [[], [1, 1], [[1]], [[1, 1, 1]], [[1.5, 1]], [[1.0, 1.0]]]
        cases.extend([[[True, False]], [[1, np.nan]], [[1, 1j]], [["1", "1"]]])
        cases.append(np.array([[1, 2**63]], dtype=np.uint64))
        for modes in cases:
            with self.subTest(modes=modes):
                with self.assertRaisesRegex(ValueError, "mode"):
                    Continuum(self.field, [0.5], modes)

    def test_rejects_invalid_surfaces(self):
        cases = [[], 0.5, [[0.5]], [0.0], [-0.1], [1.01], [np.nan], [np.inf]]
        cases.extend([[0.5 + 1j], [True], ["0.5"]])
        for surfaces in cases:
            with self.subTest(surfaces=surfaces):
                with self.assertRaisesRegex(ValueError, "surfaces"):
                    Continuum(self.field, surfaces, [[1, 1]])

    def test_respects_the_fields_radial_interval(self):
        with patch.object(self.field, "s_half_ext", np.array([0.2, 0.8])):
            Continuum(self.field, [0.8, 0.2], [[1, 1]])
            for surface in (0.1, 0.9):
                with self.subTest(surface=surface):
                    with self.assertRaisesRegex(ValueError, "surfaces"):
                        Continuum(self.field, [surface], [[1, 1]])

    def test_rejects_invalid_fields(self):
        with self.assertRaisesRegex(ValueError, "BoozerRadialInterpolant"):
            Continuum(object(), [0.5], [[1, 1]])
        cases = [
            ("stellsym", False),
            ("nfp", 0),
            ("nfp", -1),
            ("nfp", 2.5),
            ("nfp", True),
            ("psi0", 0.0),
            ("psi0", np.nan),
            ("psi0", np.inf),
        ]
        for name, value in cases:
            with self.subTest(name=name, value=value):
                with patch.object(self.field, name, value):
                    with self.assertRaisesRegex(ValueError, "field"):
                        Continuum(self.field, [0.5], [[1, 1]])

    def test_scalar_density(self):
        continuum = Continuum(self.field, [0.5], [[1, 1]], density=2.0e-7)
        self.assertEqual(continuum.density, 2.0e-7)
        for density in (0, -1, np.nan, np.inf, True, 1j, "1", [1]):
            with self.subTest(density=density):
                with self.assertRaisesRegex(ValueError, "density"):
                    Continuum(self.field, [0.5], [[1, 1]], density=density)

    def test_stores_density_callable_without_evaluation(self):
        def density(surface):
            raise AssertionError("Density must not be sampled during initialization.")

        continuum = Continuum(self.field, [0.2, 0.8], [[1, 1]], density=density)
        self.assertIs(continuum.density, density)


if __name__ == "__main__":
    unittest.main()
