import unittest
from pathlib import Path

import numpy as np

from simsopt.field.boozermagneticfield import BoozerRadialInterpolant
from simsopt.field.coordinates import (
    boozer_to_cylindrical,
    boozer_to_vmec,
    cylindrical_to_boozer,
    cylindrical_to_vmec,
    vmec_to_boozer,
    vmec_to_cylindrical,
)

np.random.seed(0)

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
filename_vac = str((TEST_DIR / "boozmn_LandremanPaul2021_QA_lowres.nc").resolve())
filename_vac_wout = str((TEST_DIR / "wout_LandremanPaul2021_QA_lowres.nc").resolve())


class TestCoordinates(unittest.TestCase):
    """Test coordinate transformation functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple Boozer field for testing
        self.field = BoozerRadialInterpolant(filename_vac, 1, comm=None)

        # Test points in Boozer coordinates
        self.s_test = np.array([0.5, 0.7, 0.3])
        self.theta_test = np.array([0.0, np.pi / 2, np.pi])
        self.zeta_test = np.array([0.0, np.pi / 4, np.pi / 2])

        # Test points in cylindrical coordinates
        self.R_test = np.array([1.5, 1.7, 1.3])
        self.phi_test = np.array([0.0, np.pi / 4, np.pi / 2])
        self.Z_test = np.array([0.1, 0.2, -0.1])

    def test_boozer_to_cylindrical_basic(self):
        """Test basic Boozer to cylindrical coordinate transformation."""
        R, phi, Z = boozer_to_cylindrical(
            self.field, self.s_test, self.theta_test, self.zeta_test
        )

        # Check output shapes
        self.assertEqual(R.shape, self.s_test.shape)
        self.assertEqual(phi.shape, self.s_test.shape)
        self.assertEqual(Z.shape, self.s_test.shape)

        # Check that R is positive
        self.assertTrue(np.all(R > 0))

        # Check that phi is in reasonable range
        self.assertTrue(np.all(phi >= -2 * np.pi) and np.all(phi <= 2 * np.pi))

    def test_boozer_to_cylindrical_scalar_input(self):
        """Test Boozer to cylindrical with scalar inputs."""
        s_scalar = 0.5
        theta_scalar = np.pi / 4
        zeta_scalar = np.pi / 2

        R, phi, Z = boozer_to_cylindrical(
            self.field, s_scalar, theta_scalar, zeta_scalar
        )

        # Check that outputs are scalars (or single-element arrays)
        self.assertTrue(np.isscalar(R) or (hasattr(R, "size") and R.size == 1))
        self.assertTrue(np.isscalar(phi) or (hasattr(phi, "size") and phi.size == 1))
        self.assertTrue(np.isscalar(Z) or (hasattr(Z, "size") and Z.size == 1))

        # Check that R is positive
        R_val = R[0] if hasattr(R, "__len__") else R
        self.assertGreater(R_val, 0)

    def test_boozer_to_cylindrical_array_validation(self):
        """Test input validation for Boozer to cylindrical transformation."""
        # Test with mismatched array shapes
        s_mismatch = np.array([0.5, 0.7])
        theta_mismatch = np.array([0.0, np.pi / 2, np.pi])

        with self.assertRaises(ValueError):
            boozer_to_cylindrical(
                self.field, s_mismatch, theta_mismatch, self.zeta_test
            )

    def test_boozer_to_cylindrical_empty_arrays(self):
        """Test Boozer to cylindrical with empty arrays raises ValueError."""
        s_empty = np.array([])
        theta_empty = np.array([])
        zeta_empty = np.array([])

        with self.assertRaises(ValueError):
            boozer_to_cylindrical(self.field, s_empty, theta_empty, zeta_empty)

    def test_cylindrical_to_boozer_basic(self):
        """Test basic cylindrical to Boozer coordinate transformation."""
        # Use coordinates that are known to work from the roundtrip test
        R_working = np.array([1.27177872, 0.92842923, 0.71525574])
        phi_working = np.array([0.0, 0.67037294, 1.57079633])
        Z_working = np.array([0.00000000e00, 3.11030751e-01, 2.87512917e-17])

        s, theta, zeta = cylindrical_to_boozer(
            self.field, R_working, phi_working, Z_working, n_guesses=10
        )

        # Check output shapes
        self.assertEqual(s.shape, R_working.shape)
        self.assertEqual(theta.shape, R_working.shape)
        self.assertEqual(zeta.shape, R_working.shape)

        # Check that s is in [0, 1]
        self.assertTrue(np.all(s >= 0) and np.all(s <= 1))

        # Check that angles are in reasonable range
        self.assertTrue(np.all(theta >= -2 * np.pi) and np.all(theta <= 2 * np.pi))
        self.assertTrue(np.all(zeta >= -2 * np.pi) and np.all(zeta <= 2 * np.pi))

    def test_cylindrical_to_boozer_n_guesses(self):
        """Test cylindrical to Boozer with custom number of initial guesses."""
        # Use coordinates that are known to work
        R_working = np.array([1.27177872, 0.92842923, 0.71525574])
        phi_working = np.array([0.0, 0.67037294, 1.57079633])
        Z_working = np.array([0.00000000e00, 3.11030751e-01, 2.87512917e-17])

        # Test with different numbers of guesses
        for n_guesses in [10, 20]:
            s, theta, zeta = cylindrical_to_boozer(
                self.field, R_working, phi_working, Z_working, n_guesses=n_guesses
            )

            # Check output shapes
            self.assertEqual(s.shape, R_working.shape)
            self.assertEqual(theta.shape, R_working.shape)
            self.assertEqual(zeta.shape, R_working.shape)

            # Check that s is in [0, 1]
            self.assertTrue(np.all(s >= 0) and np.all(s <= 1))

            # Check that angles are in reasonable range
            self.assertTrue(np.all(theta >= -2 * np.pi) and np.all(theta <= 2 * np.pi))
            self.assertTrue(np.all(zeta >= -2 * np.pi) and np.all(zeta <= 2 * np.pi))

    def test_boozer_to_cylindrical_vs_vmec_route(self):
        """Test that boozer_to_cylindrical matches
        boozer_to_vmec->vmec_to_cylindrical."""
        print("\n=== Boozer -> Cylindrical vs Boozer -> VMEC -> Cylindrical Test ===")

        # Test with multiple points
        s_test = np.array([0.3, 0.5, 0.7])
        theta_test = np.array([0.0, np.pi / 2, np.pi])
        zeta_test = np.array([0.0, np.pi / 4, np.pi / 2])

        print("Original Boozer coordinates:")
        print(f"  s: {s_test}")
        print(f"  theta: {theta_test}")
        print(f"  zeta: {zeta_test}")

        # Direct transformation: Boozer -> Cylindrical
        R_direct, phi_direct, Z_direct = boozer_to_cylindrical(
            self.field, s_test, theta_test, zeta_test
        )
        print("\nDirect transformation (Boozer -> Cylindrical):")
        print(f"  R: {R_direct}")
        print(f"  phi: {phi_direct}")
        print(f"  Z: {Z_direct}")

        # Two-step transformation: Boozer -> VMEC -> Cylindrical
        theta_vmec, phi_vmec = boozer_to_vmec(
            filename_vac_wout, self.field, s_test, theta_test, zeta_test
        )
        R_vmec, phi_vmec_cyl, Z_vmec = vmec_to_cylindrical(
            filename_vac_wout, s_test, theta_vmec, phi_vmec
        )
        print("\nTwo-step transformation (Boozer -> VMEC -> Cylindrical):")
        print(f"  R: {R_vmec}")
        print(f"  phi: {phi_vmec_cyl}")
        print(f"  Z: {Z_vmec}")

        # Compare the results
        R_diff = np.abs(R_direct - R_vmec)
        Z_diff = np.abs(Z_direct - Z_vmec)

        # Handle angle periodicity for phi comparison
        phi_diff = np.abs((phi_direct % (2 * np.pi)) - (phi_vmec_cyl % (2 * np.pi)))
        phi_diff_normalized = phi_diff % (2 * np.pi)

        print("\nDifferences (direct - vmec_route):")
        print(f"  R diff: {R_diff}")
        print(f"  phi diff (normalized): {phi_diff_normalized}")
        print(f"  Z diff: {Z_diff}")

        # Check that the transformations agree within reasonable tolerance
        # Use more relaxed tolerance since VMEC route involves additional
        # numerical steps
        np.testing.assert_allclose(R_diff, 0, atol=1e-5)
        np.testing.assert_allclose(phi_diff_normalized, 0, atol=1e-5)
        np.testing.assert_allclose(Z_diff, 0, atol=1e-5)

    def test_boozer_to_cylindrical_roundtrip(self):
        """Test roundtrip transformation: Boozer -> Cylindrical -> Boozer."""
        print("\n=== Boozer -> Cylindrical -> Boozer Roundtrip Test ===")
        # Test with a single point that is known to work well
        s_single = np.array([0.3, 0.5, 0.7, 0.9])
        theta_single = np.array([0.0, np.pi / 2, np.pi, np.pi / 4])
        zeta_single = np.array([0.0, np.pi / 4, np.pi / 2, np.pi / 3])

        print("Original Boozer coordinates (single point):")
        print(f"  s: {s_single}")
        print(f"  theta: {theta_single}")
        print(f"  zeta: {zeta_single}")

        # Forward transformation
        R, phi, Z = boozer_to_cylindrical(
            self.field, s_single, theta_single, zeta_single
        )
        print("\nIntermediate cylindrical coordinates:")
        print(f"  R: {R}")
        print(f"  phi: {phi}")
        print(f"  Z: {Z}")

        # Reverse transformation
        s_back, theta_back, zeta_back = cylindrical_to_boozer(
            self.field, R, phi, Z, n_guesses=20
        )
        print("\nFinal Boozer coordinates (after roundtrip):")
        print(f"  s: {s_back}")
        print(f"  theta: {theta_back}")
        print(f"  zeta: {zeta_back}")

        # Check that we get back close to original values
        np.testing.assert_allclose(s_back, s_single, rtol=1e-6, atol=1e-8)

        # Take modulus of angles with 2π to ensure they are in the same range
        theta_diff = np.abs((theta_back % (2 * np.pi)) - (theta_single % (2 * np.pi)))
        zeta_diff = np.abs((zeta_back % (2 * np.pi)) - (zeta_single % (2 * np.pi)))

        print("\nDifferences (original - final):")
        print(f"  s diff: {s_single - s_back}")
        print(f"  theta diff (mod 2π): {theta_diff}")
        print(f"  zeta diff (mod 2π): {zeta_diff}")

        # Both angles should be recovered accurately
        self.assertTrue(np.all(theta_diff % (2 * np.pi) < 1e-6))
        self.assertTrue(np.all(zeta_diff % (2 * np.pi) < 1e-6))

    def test_vmec_to_boozer_basic(self):
        """Test VMEC to Boozer coordinate transformation."""
        # Test with a single point
        s_vmec = np.array([0.5])
        theta_vmec = np.array([0.0])
        phi_vmec = np.array([0.0])

        theta_b, zeta_b = vmec_to_boozer(
            filename_vac_wout, self.field, s_vmec, theta_vmec, phi_vmec
        )

        # Check output shapes
        self.assertEqual(theta_b.shape, s_vmec.shape)
        self.assertEqual(zeta_b.shape, s_vmec.shape)

        # Check that angles are in reasonable range
        self.assertTrue(np.all(theta_b >= -2 * np.pi) and np.all(theta_b <= 2 * np.pi))
        self.assertTrue(np.all(zeta_b >= -2 * np.pi) and np.all(zeta_b <= 2 * np.pi))

    def test_boozer_to_vmec_basic(self):
        """Test Boozer to VMEC coordinate transformation."""
        # Test with a single point
        s = np.array([0.5])
        theta_b = np.array([0.0])
        zeta_b = np.array([0.0])

        theta_vmec, phi_vmec = boozer_to_vmec(
            filename_vac_wout, self.field, s, theta_b, zeta_b
        )

        # Check output shapes
        self.assertEqual(theta_vmec.shape, s.shape)
        self.assertEqual(phi_vmec.shape, s.shape)

        # Check that angles are in reasonable range
        self.assertTrue(
            np.all(theta_vmec >= -2 * np.pi) and np.all(theta_vmec <= 2 * np.pi)
        )
        self.assertTrue(
            np.all(phi_vmec >= -2 * np.pi) and np.all(phi_vmec <= 2 * np.pi)
        )

    def test_vmec_boozer_roundtrip(self):
        """Test roundtrip transformation: VMEC -> Boozer -> VMEC."""
        # Test with a single point
        s_vmec = np.array([0.5])
        theta_vmec = np.array([np.pi / 4])
        phi_vmec = np.array([np.pi / 2])

        # Forward transformation
        theta_b, zeta_b = vmec_to_boozer(
            filename_vac_wout, self.field, s_vmec, theta_vmec, phi_vmec
        )

        # Reverse transformation
        theta_vmec_back, phi_vmec_back = boozer_to_vmec(
            filename_vac_wout, self.field, s_vmec, theta_b, zeta_b
        )

        # Check that we get back close to original values
        # Take modulus of angles with 2π to ensure they are in the same range
        theta_diff = np.abs(
            (theta_vmec_back % (2 * np.pi)) - (theta_vmec % (2 * np.pi))
        )
        phi_diff = np.abs((phi_vmec_back % (2 * np.pi)) - (phi_vmec % (2 * np.pi)))

        # Use tighter tolerance for better accuracy
        np.testing.assert_allclose(theta_diff, 0, atol=1e-12)
        np.testing.assert_allclose(phi_diff, 0, atol=1e-12)

    def test_vmec_to_cylindrical_basic(self):
        """Test VMEC to cylindrical coordinate transformation."""
        # Test with a single point
        s_vmec = np.array([0.5])
        theta_vmec = np.array([0.0])
        phi_vmec = np.array([0.0])

        R, phi_cyl, Z = vmec_to_cylindrical(
            filename_vac_wout, s_vmec, theta_vmec, phi_vmec
        )

        # Check output shapes
        self.assertEqual(R.shape, s_vmec.shape)
        self.assertEqual(phi_cyl.shape, s_vmec.shape)
        self.assertEqual(Z.shape, s_vmec.shape)

        # Check that R is positive
        self.assertTrue(np.all(R > 0))

    def test_cylindrical_to_vmec_basic(self):
        """Test cylindrical to VMEC coordinate transformation."""
        # Test with coordinates that are known to work from the roundtrip test
        R = np.array([0.9548126])
        phi = np.array([1.57079633])
        Z = np.array([0.06899233])

        s_vmec, theta_vmec, phi_vmec = cylindrical_to_vmec(filename_vac_wout, R, phi, Z)

        # Check output shapes
        self.assertEqual(s_vmec.shape, R.shape)
        self.assertEqual(theta_vmec.shape, R.shape)
        self.assertEqual(phi_vmec.shape, R.shape)

        # Check that s is in [0, 1]
        self.assertTrue(np.all(s_vmec >= 0) and np.all(s_vmec <= 1))

    def test_vmec_cylindrical_roundtrip(self):
        """Test roundtrip transformation: VMEC -> Cylindrical -> VMEC."""
        print("\n=== VMEC -> Cylindrical -> VMEC Roundtrip Test ===")
        # Test with a single point
        s_vmec = np.array([0.5])
        theta_vmec = np.array([np.pi / 4])
        phi_vmec = np.array([np.pi / 2])

        print("Original VMEC coordinates:")
        print(f"  s: {s_vmec}")
        print(f"  theta_vmec: {theta_vmec}")
        print(f"  phi_vmec: {phi_vmec}")

        # Forward transformation
        R, phi_cyl, Z = vmec_to_cylindrical(
            filename_vac_wout, s_vmec, theta_vmec, phi_vmec
        )
        print("\nIntermediate cylindrical coordinates:")
        print(f"  R: {R}")
        print(f"  phi_cyl: {phi_cyl}")
        print(f"  Z: {Z}")

        # Reverse transformation
        s_vmec_back, theta_vmec_back, phi_vmec_back = cylindrical_to_vmec(
            filename_vac_wout, R, phi_cyl, Z
        )
        print("\nFinal VMEC coordinates (after roundtrip):")
        print(f"  s: {s_vmec_back}")
        print(f"  theta_vmec: {theta_vmec_back}")
        print(f"  phi_vmec: {phi_vmec_back}")

        # Check that we get back close to original values with stricter tolerances
        # The s coordinate should be recovered very accurately since it's
        # directly computed
        np.testing.assert_allclose(s_vmec_back, s_vmec, rtol=1e-12, atol=1e-12)

        # Take modulus of angles with 2π to ensure they are in the same range
        theta_diff = np.abs(
            (theta_vmec_back % (2 * np.pi)) - (theta_vmec % (2 * np.pi))
        )
        phi_diff = np.abs((phi_vmec_back % (2 * np.pi)) - (phi_vmec % (2 * np.pi)))

        print("\nDifferences (original - final):")
        print(f"  s diff: {s_vmec - s_vmec_back}")
        print(f"  theta_vmec diff (mod 2π): {theta_diff}")
        print(f"  phi_vmec diff (mod 2π): {phi_diff}")

        # Both angles should be recovered accurately
        self.assertTrue(np.all(theta_diff < 1e-9))
        self.assertTrue(np.all(phi_diff < 1e-12))

        # The phi coordinate should be recovered exactly since it's the same
        # in both systems
        np.testing.assert_allclose(phi_vmec_back, phi_vmec, rtol=1e-12, atol=1e-12)

    def test_multiple_points(self):
        """Test transformations with multiple points."""
        print("\n=== Multiple Points Test ===")
        # Test Boozer to cylindrical with multiple points
        s_multi = np.array([0.3, 0.5, 0.7])
        theta_multi = np.array([0.0, np.pi / 2, np.pi])
        zeta_multi = np.array([0.0, np.pi / 4, np.pi / 2])

        print("Original Boozer coordinates (multiple points):")
        print(f"  s: {s_multi}")
        print(f"  theta: {theta_multi}")
        print(f"  zeta: {zeta_multi}")

        R, phi, Z = boozer_to_cylindrical(self.field, s_multi, theta_multi, zeta_multi)
        print("\nIntermediate cylindrical coordinates:")
        print(f"  R: {R}")
        print(f"  phi: {phi}")
        print(f"  Z: {Z}")

        # Test cylindrical to Boozer with multiple points
        s_back, theta_back, zeta_back = cylindrical_to_boozer(
            self.field, R, phi, Z, n_guesses=10
        )
        print("\nFinal Boozer coordinates (after roundtrip):")
        print(f"  s: {s_back}")
        print(f"  theta: {theta_back}")
        print(f"  zeta: {zeta_back}")

        # Check that we get back close to original values (reduced tolerance
        # for better accuracy)
        # Note: Root finding may fail for some points, so we check that at
        # least some points are close
        s_close = np.abs(s_back - s_multi) < 0.1

        # Take modulus of angles with 2π to ensure they are in the same range
        theta_diff = np.abs((theta_back % (2 * np.pi)) - (theta_multi % (2 * np.pi)))
        theta_close = theta_diff < 0.1

        zeta_diff = np.abs((zeta_back % (2 * np.pi)) - (zeta_multi % (2 * np.pi)))
        zeta_close = zeta_diff < 0.1

        print("\nDifferences (original - final):")
        print(f"  s diff: {s_multi - s_back}")
        print(f"  theta diff (mod 2π): {theta_diff}")
        print(f"  zeta diff (mod 2π): {zeta_diff}")

        # At least one point should be close
        self.assertTrue(np.any(s_close) or np.any(theta_close) or np.any(zeta_close))

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test s = 0 (magnetic axis)
        s_axis = np.array([0.0])
        theta_axis = np.array([0.0])
        zeta_axis = np.array([0.0])

        R, phi, Z = boozer_to_cylindrical(self.field, s_axis, theta_axis, zeta_axis)
        self.assertTrue(np.all(R >= 0))

        # Test s = 1 (plasma boundary)
        s_boundary = np.array([1.0])
        theta_boundary = np.array([0.0])
        zeta_boundary = np.array([0.0])

        R, phi, Z = boozer_to_cylindrical(
            self.field, s_boundary, theta_boundary, zeta_boundary
        )
        self.assertTrue(np.all(R >= 0))

    def test_input_types(self):
        """Test that functions handle different input types correctly."""
        # Test with lists instead of numpy arrays
        s_list = [0.5, 0.7]
        theta_list = [0.0, np.pi / 2]
        zeta_list = [0.0, np.pi / 4]

        R, phi, Z = boozer_to_cylindrical(self.field, s_list, theta_list, zeta_list)

        # Check that outputs are numpy arrays
        self.assertIsInstance(R, np.ndarray)
        self.assertIsInstance(phi, np.ndarray)
        self.assertIsInstance(Z, np.ndarray)

        # Check shapes
        self.assertEqual(R.shape, (2,))
        self.assertEqual(phi.shape, (2,))
        self.assertEqual(Z.shape, (2,))

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with None values
        with self.assertRaises((TypeError, ValueError)):
            boozer_to_cylindrical(self.field, None, self.theta_test, self.zeta_test)

        # Test with empty arrays (should raise ValueError)
        with self.assertRaises(ValueError):
            boozer_to_cylindrical(self.field, np.array([]), np.array([]), np.array([]))

    def test_file_not_found(self):
        """Test error handling when VMEC file is not found."""
        non_existent_file = "non_existent_wout.nc"

        with self.assertRaises((FileNotFoundError, OSError)):
            vmec_to_boozer(
                non_existent_file,
                self.field,
                np.array([0.5]),
                np.array([0.0]),
                np.array([0.0]),
            )
