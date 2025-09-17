import unittest
import warnings
from unittest.mock import Mock, patch

import numpy as np

from simsopt.field.boozermagneticfield import BoozerMagneticField, ShearAlfvenWave
from simsopt.field.tracing import (
    IterationStoppingCriterion,
    MaxToroidalFluxStoppingCriterion,
    MinToroidalFluxStoppingCriterion,
    StepSizeStoppingCriterion,
    ToroidalTransitStoppingCriterion,
    compute_poloidal_transits,
    compute_resonances,
    compute_toroidal_transits,
    trace_particles_boozer,
    trace_particles_boozer_perturbed,
)
from simsopt.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
)


class TestTracingFunctions(unittest.TestCase):
    """Test cases for tracing.py functions to improve coverage."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock field objects for testing
        self.mock_field = Mock(spec=BoozerMagneticField)
        self.mock_field.field_type = "vac"

        self.mock_perturbed_field = Mock(spec=ShearAlfvenWave)
        self.mock_perturbed_field.B0 = self.mock_field

        # Test data
        self.stz_inits = np.array([[0.5, 0.0, 0.0], [0.7, np.pi, np.pi]])
        self.parallel_speeds = np.array([1e6, 2e6])
        self.mus = np.array([1e-10, 2e-10])
        self.tmax = 1e-6

    def test_trace_particles_boozer_perturbed_basic(self):
        """Test basic functionality of trace_particles_boozer_perturbed."""
        # Mock the C++ function call
        with patch(
            "simsoptpp.particle_guiding_center_boozer_perturbed_tracing"
        ) as mock_cpp:
            mock_cpp.return_value = (
                [[0, 0.5, 0, 0, 1e6], [1e-6, 0.5, 0.1, 0.1, 1e6]],
                [[1e-6, 0, 0.5, 0, 0, 1e6]],
            )

            # Mock the modB method to return a proper array
            mock_modB = Mock()
            mock_modB.return_value = np.array([[1.0]])
            self.mock_perturbed_field.B0.modB = mock_modB

            result = trace_particles_boozer_perturbed(
                self.mock_perturbed_field,
                self.stz_inits,
                self.parallel_speeds,
                self.mus,
                tmax=self.tmax,
            )

            self.assertEqual(len(result), 2)
            self.assertEqual(len(result[0]), 2)  # res_tys
            self.assertEqual(len(result[1]), 2)  # res_hits

    def test_trace_particles_boozer_perturbed_with_parameters(self):
        """Test trace_particles_boozer_perturbed with various parameters."""
        with patch(
            "simsoptpp.particle_guiding_center_boozer_perturbed_tracing"
        ) as mock_cpp:
            mock_cpp.return_value = ([[0, 0.5, 0, 0, 1e6]], [[1e-6, 0, 0.5, 0, 0, 1e6]])

            # Mock the modB method
            mock_modB = Mock()
            mock_modB.return_value = np.array([[1.0]])
            self.mock_perturbed_field.B0.modB = mock_modB

            # Test with various parameter combinations
            result = trace_particles_boozer_perturbed(
                self.mock_perturbed_field,
                self.stz_inits[:1],
                self.parallel_speeds[:1],
                self.mus[:1],
                tmax=self.tmax,
                mass=2 * ALPHA_PARTICLE_MASS,
                charge=2 * ALPHA_PARTICLE_CHARGE,
                Ekin=1e-13,
                thetas=[0, np.pi / 2],
                zetas=[0, np.pi],
                omega_thetas=[1e6, 2e6],
                omega_zetas=[1e6],
                vpars=[5e5, 1.5e6],
                dt_save=1e-7,
                mode="gc_vac",
                forget_exact_path=True,
                thetas_stop=True,
                zetas_stop=True,
                vpars_stop=True,
                axis=1,
            )

            self.assertEqual(len(result), 2)

    def test_trace_particles_boozer_perturbed_mode_validation(self):
        """Test mode validation in trace_particles_boozer_perturbed."""
        with patch(
            "simsoptpp.particle_guiding_center_boozer_perturbed_tracing"
        ) as mock_cpp:
            mock_cpp.return_value = ([[0, 0.5, 0, 0, 1e6]], [[1e-6, 0, 0.5, 0, 0, 1e6]])

            # Mock the modB method
            mock_modB = Mock()
            mock_modB.return_value = np.array([[1.0]])
            self.mock_perturbed_field.B0.modB = mock_modB

            # Test invalid mode
            with self.assertRaises(AssertionError):
                trace_particles_boozer_perturbed(
                    self.mock_perturbed_field,
                    self.stz_inits[:1],
                    self.parallel_speeds[:1],
                    self.mus[:1],
                    tmax=self.tmax,
                    mode="invalid_mode",
                )

    def test_trace_particles_boozer_perturbed_warnings(self):
        """Test warnings in trace_particles_boozer_perturbed."""
        with patch(
            "simsoptpp.particle_guiding_center_boozer_perturbed_tracing"
        ) as mock_cpp:
            mock_cpp.return_value = ([[0, 0.5, 0, 0, 1e6]], [[1e-6, 0, 0.5, 0, 0, 1e6]])

            # Test mode inconsistency warning
            self.mock_perturbed_field.B0.field_type = "mhd"

            # Mock the modB method
            mock_modB = Mock()
            mock_modB.return_value = np.array([[1.0]])
            self.mock_perturbed_field.B0.modB = mock_modB

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                trace_particles_boozer_perturbed(
                    self.mock_perturbed_field,
                    self.stz_inits[:1],
                    self.parallel_speeds[:1],
                    self.mus[:1],
                    tmax=self.tmax,
                    mode="gc_vac",
                )
                self.assertTrue(
                    any(
                        "Prescribed mode is inconsistent" in str(warning.message)
                        for warning in w
                    )
                )

    def test_trace_particles_boozer_perturbed_mpi(self):
        """Test trace_particles_boozer_perturbed with MPI communicator."""
        mock_comm = Mock()
        mock_comm.size = 2
        mock_comm.rank = 0
        mock_comm.allgather.return_value = [
            [[[0, 0.5, 0, 0, 1e6]]],
            [[[1e-6, 0.5, 0.1, 0.1, 1e6]]],
        ]

        with patch(
            "simsoptpp.particle_guiding_center_boozer_perturbed_tracing"
        ) as mock_cpp:
            mock_cpp.return_value = ([[0, 0.5, 0, 0, 1e6]], [[1e-6, 0, 0.5, 0, 0, 1e6]])

            # Mock the modB method
            mock_modB = Mock()
            mock_modB.return_value = np.array([[1.0]])
            self.mock_perturbed_field.B0.modB = mock_modB

            result = trace_particles_boozer_perturbed(
                self.mock_perturbed_field,
                self.stz_inits,
                self.parallel_speeds,
                self.mus,
                tmax=self.tmax,
                comm=mock_comm,
            )

            self.assertEqual(len(result), 2)

    def test_trace_particles_boozer_edge_cases(self):
        """Test edge cases in trace_particles_boozer."""
        # Test with empty stopping criteria
        with patch("simsoptpp.particle_guiding_center_boozer_tracing") as mock_cpp:
            mock_cpp.return_value = ([[0, 0.5, 0, 0, 1e6]], [[1e-6, 0, 0.5, 0, 0, 1e6]])

            result = trace_particles_boozer(
                self.mock_field,
                self.stz_inits,
                self.parallel_speeds,
                tmax=self.tmax,
                stopping_criteria=[],
                thetas=[],
                zetas=[],
                omega_thetas=[],
                omega_zetas=[],
                vpars=[],
            )

            self.assertEqual(len(result), 2)

    def test_trace_particles_boozer_validation_errors(self):
        """Test validation errors in trace_particles_boozer."""
        # Test zetas_stop without zetas
        with self.assertRaises(ValueError):
            trace_particles_boozer(
                self.mock_field,
                self.stz_inits,
                self.parallel_speeds,
                tmax=self.tmax,
                zetas_stop=True,
                zetas=[],
                omega_zetas=[],
            )

        # Test thetas_stop without thetas
        with self.assertRaises(ValueError):
            trace_particles_boozer(
                self.mock_field,
                self.stz_inits,
                self.parallel_speeds,
                tmax=self.tmax,
                thetas_stop=True,
                thetas=[],
                omega_thetas=[],
            )

        # Test vpars_stop without vpars
        with self.assertRaises(ValueError):
            trace_particles_boozer(
                self.mock_field,
                self.stz_inits,
                self.parallel_speeds,
                tmax=self.tmax,
                vpars_stop=True,
                vpars=[],
            )

    def test_trace_particles_boozer_symplectic_warnings(self):
        """Test warnings in trace_particles_boozer for symplectic solver."""
        with patch("simsoptpp.particle_guiding_center_boozer_tracing") as mock_cpp:
            mock_cpp.return_value = ([[0, 0.5, 0, 0, 1e6]], [[1e-6, 0, 0.5, 0, 0, 1e6]])

            # Test symplectic solver warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                trace_particles_boozer(
                    self.mock_field,
                    self.stz_inits,
                    self.parallel_speeds,
                    tmax=self.tmax,
                    solveSympl=True,
                    abstol=1e-9,  # Should trigger warning
                    axis=2,  # Should trigger warning and be reset to 0
                )
                self.assertTrue(
                    any(
                        "Symplectic solver does not use absolute or relative tolerance"
                        in str(warning.message)
                        for warning in w
                    )
                )
                self.assertTrue(
                    any(
                        "Symplectic solver must be run with axis = 0"
                        in str(warning.message)
                        for warning in w
                    )
                )

    def test_trace_particles_boozer_rk45_warnings(self):
        """Test warnings in trace_particles_boozer for RK45 solver."""
        with patch("simsoptpp.particle_guiding_center_boozer_tracing") as mock_cpp:
            mock_cpp.return_value = ([[0, 0.5, 0, 0, 1e6]], [[1e-6, 0, 0.5, 0, 0, 1e6]])

            # Test RK45 solver warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                trace_particles_boozer(
                    self.mock_field,
                    self.stz_inits,
                    self.parallel_speeds,
                    tmax=self.tmax,
                    solveSympl=False,
                    dt=1e-7,  # Should trigger warning
                    roottol=1e-9,  # Should trigger warning
                    predictor_step=True,  # Should trigger warning
                )
                self.assertTrue(
                    any(
                        "RK45 solver does not use dt, roottol, or predictor_step"
                        in str(warning.message)
                        for warning in w
                    )
                )

    def test_trace_particles_boozer_mode_validation(self):
        """Test mode validation in trace_particles_boozer."""
        with patch("simsoptpp.particle_guiding_center_boozer_tracing") as mock_cpp:
            mock_cpp.return_value = ([[0, 0.5, 0, 0, 1e6]], [[1e-6, 0, 0.5, 0, 0, 1e6]])

            # Test invalid mode
            with self.assertRaises(AssertionError):
                trace_particles_boozer(
                    self.mock_field,
                    self.stz_inits,
                    self.parallel_speeds,
                    tmax=self.tmax,
                    mode="invalid_mode",
                )

    def test_trace_particles_boozer_mode_warnings(self):
        """Test mode inconsistency warnings in trace_particles_boozer."""
        with patch("simsoptpp.particle_guiding_center_boozer_tracing") as mock_cpp:
            mock_cpp.return_value = ([[0, 0.5, 0, 0, 1e6]], [[1e-6, 0, 0.5, 0, 0, 1e6]])

            # Test mode inconsistency warning
            self.mock_field.field_type = "mhd"
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                trace_particles_boozer(
                    self.mock_field,
                    self.stz_inits,
                    self.parallel_speeds,
                    tmax=self.tmax,
                    mode="gc_vac",
                )
                self.assertTrue(
                    any(
                        "Prescribed mode is inconsistent" in str(warning.message)
                        for warning in w
                    )
                )

    def test_trace_particles_boozer_mpi(self):
        """Test trace_particles_boozer with MPI communicator."""
        mock_comm = Mock()
        mock_comm.size = 2
        mock_comm.rank = 0
        mock_comm.allgather.return_value = [
            [[[0, 0.5, 0, 0, 1e6]]],
            [[[1e-6, 0.5, 0.1, 0.1, 1e6]]],
        ]

        with patch("simsoptpp.particle_guiding_center_boozer_tracing") as mock_cpp:
            mock_cpp.return_value = ([[0, 0.5, 0, 0, 1e6]], [[1e-6, 0, 0.5, 0, 0, 1e6]])

            result = trace_particles_boozer(
                self.mock_field,
                self.stz_inits,
                self.parallel_speeds,
                tmax=self.tmax,
                comm=mock_comm,
            )

            self.assertEqual(len(result), 2)

    def test_trace_particles_boozer_forget_exact_path(self):
        """Test forget_exact_path functionality in trace_particles_boozer."""
        with patch("simsoptpp.particle_guiding_center_boozer_tracing") as mock_cpp:
            mock_cpp.return_value = (
                [[0, 0.5, 0, 0, 1e6], [1e-6, 0.5, 0.1, 0.1, 1e6]],
                [[1e-6, 0, 0.5, 0, 0, 1e6]],
            )

            result = trace_particles_boozer(
                self.mock_field,
                self.stz_inits,
                self.parallel_speeds,
                tmax=self.tmax,
                forget_exact_path=True,
            )

            # Should only return first and last positions
            self.assertEqual(len(result[0][0]), 2)

    def test_compute_resonances_edge_cases(self):
        """Test edge cases in compute_resonances."""
        # Test with empty results
        res_tys = []
        res_hits = []
        result = compute_resonances(res_tys, res_hits)
        self.assertEqual(result, [])

        # Test with single particle, no hits
        res_tys = [np.array([[0, 0.5, 0, 0, 1e6]])]
        res_hits = [np.array([])]
        result = compute_resonances(res_tys, res_hits)
        self.assertEqual(result, [])

    def test_compute_poloidal_transits_edge_cases(self):
        """Test edge cases in compute_poloidal_transits."""
        # Test with empty results
        res_tys = []
        result = compute_poloidal_transits(res_tys)
        self.assertEqual(len(result), 0)

        # Test with single trajectory point
        res_tys = [np.array([[0, 0.5, 0, 0, 1e6]])]
        result = compute_poloidal_transits(res_tys)
        self.assertEqual(result[0], 0)

    def test_compute_toroidal_transits_edge_cases(self):
        """Test edge cases in compute_toroidal_transits."""
        # Test with empty results
        res_tys = []
        result = compute_toroidal_transits(res_tys)
        self.assertEqual(len(result), 0)

        # Test with single trajectory point
        res_tys = [np.array([[0, 0.5, 0, 0, 1e6]])]
        result = compute_toroidal_transits(res_tys)
        self.assertEqual(result[0], 0)

    def test_stopping_criteria_instantiation(self):
        """Test that stopping criteria classes can be instantiated."""
        # Test MinToroidalFluxStoppingCriterion
        min_criterion = MinToroidalFluxStoppingCriterion(0.1)
        self.assertIsNotNone(min_criterion)

        # Test MaxToroidalFluxStoppingCriterion
        max_criterion = MaxToroidalFluxStoppingCriterion(0.9)
        self.assertIsNotNone(max_criterion)

        # Test ToroidalTransitStoppingCriterion - fix constructor arguments
        transit_criterion = ToroidalTransitStoppingCriterion(10)
        self.assertIsNotNone(transit_criterion)

        # Test IterationStoppingCriterion
        iteration_criterion = IterationStoppingCriterion(1000)
        self.assertIsNotNone(iteration_criterion)

        # Test StepSizeStoppingCriterion
        step_criterion = StepSizeStoppingCriterion(1e-10)
        self.assertIsNotNone(step_criterion)

    def test_trace_particles_boozer_perturbed_energy_calculation(self):
        """Test energy calculation in trace_particles_boozer_perturbed."""
        with patch(
            "simsoptpp.particle_guiding_center_boozer_perturbed_tracing"
        ) as mock_cpp:
            mock_cpp.return_value = ([[0, 0.5, 0, 0, 1e6]], [[1e-6, 0, 0.5, 0, 0, 1e6]])

            # Test with Ekin provided
            result = trace_particles_boozer_perturbed(
                self.mock_perturbed_field,
                self.stz_inits[:1],
                self.parallel_speeds[:1],
                self.mus[:1],
                tmax=self.tmax,
                Ekin=1e-13,
            )

            self.assertEqual(len(result), 2)

            # Test without Ekin (should calculate from vpar and mu)
            self.mock_perturbed_field.set_points = Mock()
            self.mock_perturbed_field.B0.modB.return_value = np.array([[1.0]])

            result = trace_particles_boozer_perturbed(
                self.mock_perturbed_field,
                self.stz_inits[:1],
                self.parallel_speeds[:1],
                self.mus[:1],
                tmax=self.tmax,
            )

            self.assertEqual(len(result), 2)

    def test_trace_particles_boozer_energy_array(self):
        """Test trace_particles_boozer with array Ekin."""
        with patch("simsoptpp.particle_guiding_center_boozer_tracing") as mock_cpp:
            mock_cpp.return_value = ([[0, 0.5, 0, 0, 1e6]], [[1e-6, 0, 0.5, 0, 0, 1e6]])

            # Test with array Ekin
            Ekin_array = np.array([1e-13, 2e-13])
            result = trace_particles_boozer(
                self.mock_field,
                self.stz_inits,
                self.parallel_speeds,
                tmax=self.tmax,
                Ekin=Ekin_array,
            )

            self.assertEqual(len(result), 2)

    def test_trace_particles_boozer_parameter_defaults(self):
        """Test parameter default values in trace_particles_boozer."""
        with patch("simsoptpp.particle_guiding_center_boozer_tracing") as mock_cpp:
            mock_cpp.return_value = ([[0, 0.5, 0, 0, 1e6]], [[1e-6, 0, 0.5, 0, 0, 1e6]])

            # Test with minimal parameters to check defaults
            result = trace_particles_boozer(
                self.mock_field, self.stz_inits, self.parallel_speeds, tmax=self.tmax
            )

            self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
