"""Run directly with mpiexec -n 2 (or 4) python tests/saw/test_stellgap_mpi.py."""

import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from scipy.interpolate import make_interp_spline

from firm3d.field.boozermagneticfield import BoozerRadialInterpolant
from firm3d.saw.stellgap import Continuum

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


EQUILIBRIUM_FILE = (
    Path(__file__).parents[1] / "test_files" / "boozmn_n3are_R7.75B5.7.nc"
)


@unittest.skipIf(MPI is None, "mpi4py is optional")
class TestContinuumMPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comm = MPI.COMM_WORLD
        cls.rank = cls.comm.Get_rank()
        cls.field = BoozerRadialInterpolant(
            str(EQUILIBRIUM_FILE), order=3, no_K=True, comm=cls.comm
        )

    def make_continuum(
        self, surfaces=(.8, .2, .4, .5, .6, .8, .3), density=None,
        reference_field=2.0, analytic=True,
    ):
        modes = [[0, 0], [1, 0], [2, 3]]
        if not analytic:
            return Continuum(
                self.field, surfaces, modes, density, reference_field=reference_field
            )
        s = np.linspace(0, 1, 7)
        radius = np.column_stack((np.full_like(s, 3), s))
        height = np.column_stack((np.zeros_like(s), s))
        with patch.multiple(
            self.field, xm_b=np.array([0, 1]), xn_b=np.array([0, 3]),
            rmnc_splines=make_interp_spline(s, radius),
            zmns_splines=make_interp_spline(s, height),
            numns_splines=make_interp_spline(s, np.zeros_like(radius)),
            iota_spline=make_interp_spline(s, np.full_like(s, 0.5)),
            psi0=-0.8,
        ):
            return Continuum(
                self.field, surfaces, modes, density, reference_field=reference_field
            )

    @contextmanager
    def collective_assertions(self):
        """Propagate assertion failures so every rank can reach the next test."""
        error = None
        try:
            yield
        except Exception as exception:
            error = f"rank {self.rank}: {type(exception).__name__}: {exception}"
        errors = self.comm.allgather(error)
        failures = [message for message in errors if message is not None]
        if failures:
            self.fail("; ".join(failures))

    def assert_same_result(self, actual, expected):
        for name in ("surfaces", "modes", "dominant_modes", "density"):
            np.testing.assert_array_equal(
                getattr(actual, name), getattr(expected, name)
            )
        for name in ("eigenvalues", "normalized_frequencies", "frequencies_khz"):
            np.testing.assert_allclose(
                getattr(actual, name), getattr(expected, name), rtol=1e-12, atol=1e-14
            )
        if actual.eigenvectors is not None:
            # These fixtures have isolated eigenvalues; vector signs may differ.
            np.testing.assert_allclose(
                abs(actual.eigenvectors), abs(expected.eigenvectors),
                rtol=1e-12, atol=1e-12,
            )
        for actual_surface, expected_surface in zip(
            actual.diagnostics, expected.diagnostics
        ):
            self.assertEqual(actual_surface["surface"], expected_surface["surface"])
            self.assertEqual(
                actual_surface["quadrature"], expected_surface["quadrature"]
            )
            np.testing.assert_allclose(
                actual_surface["solver"]["scaled_residuals"],
                expected_surface["solver"]["scaled_residuals"], atol=1e-13,
            )

    def test_matches_serial_with_uneven_counts_duplicates_and_real_geometry(self):
        for analytic in (True, False):
            for keep in (False, True):
                with self.subTest(analytic=analytic, keep=keep):
                    continuum = self.make_continuum(density=1e-7, analytic=analytic)
                    result = continuum.run(self.comm, keep_eigenvectors=keep)
                    with self.collective_assertions():
                        if self.rank == 0:
                            fresh = self.make_continuum(density=1e-7, analytic=analytic)
                            expected = fresh.run(keep_eigenvectors=keep)
                            self.assert_same_result(result, expected)
                            self.assertEqual(result.eigenvectors is None, not keep)
                        else:
                            self.assertIsNone(result)

    def test_more_ranks_than_surfaces(self):
        continuum = self.make_continuum(surfaces=(.8, .2))
        with patch.object(
            continuum, "_plan_basis", wraps=continuum._plan_basis
        ) as plan:
            result = continuum.run(self.comm)
        with self.collective_assertions():
            if self.rank == 0:
                np.testing.assert_array_equal(result.surfaces, [.8, .2])
                self.assertEqual(result.eigenvalues.shape, (2, 3))
            else:
                self.assertIsNone(result)
            if self.rank >= 2:
                plan.assert_not_called()

    def test_root_only_density_reference_and_run_options(self):
        density = Mock(side_effect=lambda s: 1e-7 * (1 + s))
        continuum = self.make_continuum(density=density, reference_field=None)
        reference = continuum.get_reference_field
        if self.rank != 0:
            reference = Mock(side_effect=AssertionError("Worker computed reference"))
        with patch.object(
            continuum, "get_reference_field", wraps=reference
        ) as get_field:
            # Non-root options are deliberately invalid: root controls the run.
            result = continuum.run(
                self.comm, rtol=1e-8 if self.rank == 0 else -1,
                keep_eigenvectors=False if self.rank == 0 else "ignored",
            )
        with self.collective_assertions():
            if self.rank == 0:
                self.assertEqual(density.call_count, len(continuum.surfaces))
                self.assertEqual(get_field.call_count, 1)
                self.assertTrue(result.reference_field["converged"])
                np.testing.assert_allclose(result.density, 1e-7 * (1 + result.surfaces))
            else:
                density.assert_not_called()
                get_field.assert_not_called()

    def test_reuses_root_pilots_and_assigns_remaining_indices_cyclically(self):
        continuum = self.make_continuum()
        with self.collective_assertions():
            if self.rank == 0:
                continuum.prepare(benchmark=True)
        with patch.object(
            continuum, "_solve_surface", wraps=continuum._solve_surface
        ) as solver:
            continuum.run(self.comm)
        with self.collective_assertions():
            # The prepared unique pilots are .2, .5, and .8.
            expected = [s for i, s in enumerate(continuum.surfaces)
                        if i % self.comm.size == self.rank and s not in (.2, .5, .8)]
            self.assertEqual([call.args[0] for call in solver.call_args_list], expected)

    def test_rejects_mismatched_inputs_on_all_ranks(self):
        if self.comm.size < 2:
            self.skipTest("Requires at least two MPI ranks")
        for mismatch in ("geometry", "surfaces", "modes"):
            with self.subTest(mismatch=mismatch):
                continuum = self.make_continuum()
                if self.rank == 1:
                    if mismatch == "geometry":
                        continuum._coordinate_splines["R"].c[0, 0] += 0.1
                    elif mismatch == "surfaces":
                        continuum.surfaces[0] = .7
                    else:
                        continuum.modes[1, 0] = 3
                with self.collective_assertions():
                    with self.assertRaisesRegex(RuntimeError, "rank 1 setup.*differ"):
                        continuum.run(self.comm)

    def test_worker_setup_failure_reaches_all_ranks(self):
        if self.comm.size < 2:
            self.skipTest("Requires at least two MPI ranks")
        continuum = self.make_continuum()
        planner = continuum._plan_basis
        if self.rank == 1:
            planner = Mock(side_effect=ValueError("injected basis failure"))
        with patch.object(continuum, "_plan_basis", wraps=planner):
            with self.collective_assertions():
                with self.assertRaisesRegex(
                    RuntimeError, "rank 1 setup.*basis failure"
                ):
                    continuum.run(self.comm)

    def test_preparation_worker_and_result_failures_reach_all_ranks(self):
        for stage in ("preparation", "surfaces", "results"):
            with self.subTest(stage=stage):
                continuum = self.make_continuum()
                if stage == "preparation":
                    method = "prepare"
                    failing_rank = 0
                elif stage == "surfaces":
                    method = "_solve_surface"
                    failing_rank = min(1, self.comm.size - 1)
                else:
                    method = "_build_result"
                    failing_rank = 0
                original = getattr(continuum, method)

                def injected(*args, **kwargs):
                    # Keep the root pilot (.5) successful to reach surface work.
                    is_pilot = stage == "surfaces" and args[0] == .5
                    if self.rank == failing_rank and not is_pilot:
                        raise ValueError("injected recoverable failure")
                    return original(*args, **kwargs)

                with patch.object(continuum, method, side_effect=injected):
                    with self.collective_assertions():
                        with self.assertRaisesRegex(
                            RuntimeError, f"rank {failing_rank} {stage}.*recoverable"
                        ):
                            continuum.run(self.comm)
                # A new collective proves that peers exited the failed run together.
                self.assertEqual(len(self.comm.allgather("recovered")), self.comm.size)

    def test_orientation_is_checked_across_ranks(self):
        if self.comm.size < 2:
            self.skipTest("Requires at least two MPI ranks")
        continuum = self.make_continuum()
        prepare_surface = continuum._prepare_surface

        def reversed_orientation(*args, **kwargs):
            entry = prepare_surface(*args, **kwargs)
            if self.rank == 1:
                entry["metadata"]["orientation"] *= -1
            return entry

        with patch.object(
            continuum, "_prepare_surface", side_effect=reversed_orientation
        ):
            with self.collective_assertions():
                with self.assertRaisesRegex(
                    RuntimeError, "rank 1 surfaces.*orientation"
                ):
                    continuum.run(self.comm)

    def test_degenerate_eigenspace_projector(self):
        continuum = self.make_continuum(surfaces=(.8, .2, .5))
        original = continuum._converge_surface

        def constant_matrices(*args, **kwargs):
            matrices = original(*args, **kwargs)
            matrices["K"] = np.diag([0., 4., 4.])
            matrices["M0"] = np.eye(3)
            return matrices

        solve_surface = continuum._solve_surface

        def rotated_eigenvectors(*args):
            solution = solve_surface(*args)
            angle = self.rank * .3
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
            vectors = solution["eigenvectors"]
            vectors[:, 1:] = vectors[:, 1:] @ rotation
            return solution

        with patch.object(
            continuum, "_converge_surface", side_effect=constant_matrices
        ):
            with patch.object(
                continuum, "_solve_surface", side_effect=rotated_eigenvectors
            ):
                result = continuum.run(self.comm, keep_eigenvectors=True)
        with self.collective_assertions():
            if self.rank == 0:
                for values, vectors in zip(result.eigenvalues, result.eigenvectors):
                    np.testing.assert_allclose(values, [0., 4., 4.])
                    block = vectors[:, 1:]
                    np.testing.assert_allclose(
                        block @ block.T, np.diag([0., 1., 1.]), atol=1e-14
                    )

    def test_subcommunicator_uses_its_own_root(self):
        subcomm = self.comm.Split(color=self.rank % 2, key=-self.rank)
        try:
            result = self.make_continuum().run(subcomm)
            with self.collective_assertions():
                self.assertEqual(result is None, subcomm.rank != 0)
                if result is not None:
                    self.assertEqual(result.eigenvalues.shape, (7, 3))
        finally:
            subcomm.Free()

    def test_invalid_communicator(self):
        continuum = self.make_continuum()
        for comm in (MPI.COMM_NULL, object()):
            with self.assertRaisesRegex(ValueError, "non-null.*intracommunicator"):
                continuum.run(comm)


if __name__ == "__main__":
    unittest.main(verbosity=2)
