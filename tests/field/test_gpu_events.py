"""Dense GPU event roots and CPU stopping-criterion contracts."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import firm3dpp
import numpy as np

from firm3d.catapult.field import CatapultBoozerField, CatapultPerturbedBoozerField
from firm3d.catapult.tracing import (
    _event_options,
    trace_particles_boozer_gpu,
    trace_particles_boozer_perturbed_gpu,
)
from firm3d.field.boozermagneticfield import BoozerAnalytic
from firm3d.field.tracing import (
    IterationStoppingCriterion,
    MaxToroidalFluxStoppingCriterion,
    MinToroidalFluxStoppingCriterion,
    StepSizeStoppingCriterion,
    ToroidalTransitStoppingCriterion,
    trace_particles_boozer,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE as CHARGE,
    ALPHA_PARTICLE_MASS as MASS,
    FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
)

ROOT = Path(__file__).resolve().parents[2]
PERIOD = 2 * np.pi
SPEED = np.sqrt(2 * ENERGY / MASS)
HAS_CUDA = hasattr(firm3dpp, "boozer_gpu_tracing")


def constant_field(iota=0.4):
    return BoozerAnalytic(etabar=0, B0=1, N=0, G0=1, psi0=1, iota0=iota)


class TestGPUEventsHost(unittest.TestCase):
    def test_polynomial_roots_and_winding(self):
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not shutil.which(compiler[0]):
            self.skipTest("C++ compiler not available")
        with tempfile.TemporaryDirectory() as folder:
            exe = str(Path(folder) / "events")
            subprocess.run(
                compiler
                + [
                    "-std=c++17",
                    "-O2",
                    "-I" + str(ROOT / "src/firm3dpp"),
                    str(ROOT / "tests/dopri5_events.cpp"),
                    "-o",
                    exe,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = subprocess.run([exe], check=True, capture_output=True, text=True)
            self.assertIn("phase checks passed", result.stdout)

    def test_configuration(self):
        points = np.array([[0.3, 7, 0]])
        self.assertIsNone(_event_options(points))
        options = _event_options(points, zetas=[0, np.pi], vpars=[0])
        np.testing.assert_array_equal(options["planes"], [0, 1, 0, 0, np.pi, 1, 0, 0])
        np.testing.assert_allclose(options["theta_offsets"], [PERIOD])
        for kwargs in [
            {"zetas": [0], "phases": [0]},
            {"phases_stop": True},
            {"vpars_stop": True},
            {"max_phase_hits": 1},
            {"zetas": [0], "max_hits": 0},
            {"zetas": [0], "max_hits": 2.5},
            {"zetas": [np.nan]},
            {"vpars": [[0]]},
            {"phases": [0], "n_zetas": [1, 2]},
            {"zetas": [0], "boozer": False},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                _event_options(points, **kwargs)

    def test_native_event_format_and_intentional_stop(self):
        field = CatapultBoozerField(constant_field(), 2, 2, 2, precision="single")
        samples = np.array([0.25, 0.3, 0, 0, 0, 0.2, 0.1], dtype=np.float32)
        end = 0.250000000000001
        hits = np.full((1, 4, 6), np.nan)
        hits[0, 0] = [end, 1, 0.3, 7, 0, 0]
        with patch.object(
            firm3dpp,
            "boozer_gpu_tracing",
            create=True,
            return_value=(samples, hits.ravel(), np.array([end])),
        ) as trace:
            paths, events = trace_particles_boozer_gpu(
                field,
                np.array([[0.3, 7, 0]]),
                [0.5 * SPEED],
                tmax=1,
                zetas=[0],
                vpars=[0],
                vpars_stop=True,
                max_hits=4,
                forget_exact_path=True,
            )
        self.assertEqual(paths[0][-1, 0], end)
        np.testing.assert_array_equal(events[0], hits[0, :1])
        self.assertEqual(trace.call_args.kwargs["event_options"]["max_hits"], 4)


@unittest.skipUnless(HAS_CUDA, "requires CUDA bindings and a GPU")
class TestGPUEventsDevice(unittest.TestCase):
    def test_sections_both_directions_and_cadence_independence(self):
        points = np.array([[0.3, 7, 0.2], [0.4, -7, 6.1], [0.2, 0.1, 0]])
        speeds = np.array([0.8, -0.7, 0.5]) * SPEED
        for precision, tol in [("double", 1e-10), ("single", 1e-6)]:
            field = CatapultBoozerField(constant_field(), 2, 2, 2, precision=precision)
            tmax = np.array([2.2e-6, 2.2e-6, 0])
            saved = []
            for cadence, forget in [(1e-8, False), (1e-5, False), (1e-8, True)]:
                paths, hits = trace_particles_boozer_gpu(
                    field,
                    points,
                    speeds,
                    tmax=tmax,
                    dt_save=cadence,
                    tol=tol,
                    zetas=[0, np.pi],
                    max_hits=32,
                    forget_exact_path=forget,
                )
                saved.append(hits)
                for i in range(2):
                    self.assertTrue(np.all(np.diff(hits[i][:, 0]) > 0))
                    for row in hits[i]:
                        n = round(
                            (points[i, 2] + speeds[i] * row[0] - row[1] * np.pi)
                            / PERIOD
                        )
                        expected = (
                            row[1] * np.pi + n * PERIOD - points[i, 2]
                        ) / speeds[i]
                        np.testing.assert_allclose(
                            row[0], expected, rtol=5e-5, atol=1e-12
                        )
                        np.testing.assert_allclose(
                            row[3], points[i, 1] + 0.4 * speeds[i] * row[0], atol=2e-4
                        )
                    self.assertEqual(paths[i][-1, 0], tmax[i])
                self.assertEqual(len(hits[2]), 0)
                self.assertEqual(len(paths[2]), 1)
            for a, b in zip(saved[0], saved[1]):
                np.testing.assert_array_equal(a, b)
            for a, b in zip(saved[0], saved[2]):
                np.testing.assert_array_equal(a, b)

    def test_helical_moving_plane_preserves_angle_winding(self):
        field = CatapultBoozerField(constant_field(), 2, 2, 2)
        point = np.array([[0.3, 7, 0.2]])
        _, hits = trace_particles_boozer_gpu(
            field,
            point,
            [SPEED],
            tmax=2e-6,
            phases=[0.5],
            n_zetas=[1],
            m_thetas=[2],
            omegas=[0.1 * SPEED],
            tol=1e-10,
            forget_exact_path=True,
        )
        initial_phase = 0.2 + 2 * 7
        turns = np.arange(np.floor((initial_phase - 0.5) / PERIOD) + 1, 20)
        times = (0.5 + PERIOD * turns - initial_phase) / (1.7 * SPEED)
        times = times[times <= 2e-6]
        np.testing.assert_allclose(hits[0][:, 0], times, rtol=1e-8, atol=1e-15)
        np.testing.assert_allclose(hits[0][:, 3], 7 + 0.4 * SPEED * times, atol=2e-7)

    def test_multiple_and_simultaneous_planes_in_one_step(self):
        field = CatapultBoozerField(constant_field(0), 2, 2, 2)
        points = np.array([[0.3, 0, 0]])
        paths, hits = trace_particles_boozer_gpu(
            field,
            points,
            [SPEED],
            tmax=1e-7,
            dt=1e-7,
            tol=1e-10,
            phases=[0, 0],
            n_zetas=[100, 100],
            max_hits=64,
            forget_exact_path=True,
        )
        times = np.arange(1, 100 * SPEED * 1e-7 / PERIOD + 1) * PERIOD / (100 * SPEED)
        times = times[times <= 1e-7]
        np.testing.assert_allclose(hits[0][:, 0], np.repeat(times, 2), rtol=1e-12)
        np.testing.assert_array_equal(hits[0][:, 1], np.tile([0, 1], len(times)))
        self.assertEqual(paths[0][-1, 0], 1e-7)

    def test_launch_exclusion_and_exact_endpoint(self):
        field = CatapultBoozerField(constant_field(0), 2, 2, 2)
        horizon = 1e-7
        paths, hits = trace_particles_boozer_gpu(
            field,
            np.array([[0.3, 0, 0]]),
            [SPEED],
            tmax=horizon,
            dt=horizon,
            phases=[0],
            n_zetas=[0],
            omegas=[-PERIOD / horizon],
            phases_stop=True,
            forget_exact_path=True,
        )
        self.assertEqual(len(hits[0]), 1)
        np.testing.assert_allclose(hits[0][0, 0], horizon, rtol=1e-14)
        np.testing.assert_allclose(paths[0][-1, 0], horizon, rtol=1e-14)

    def test_section_on_accepted_step_boundary_is_saved_once(self):
        field = CatapultBoozerField(constant_field(0), 2, 2, 2)
        paths, hits = trace_particles_boozer_gpu(
            field,
            np.array([[0.3, 0, 0]]),
            [SPEED],
            tmax=1e-8,
            dt=1e-9,
            zetas=[SPEED * 1e-9],
            forget_exact_path=True,
        )
        self.assertEqual(len(hits[0]), 1)
        np.testing.assert_allclose(hits[0][0, 0], 1e-9, rtol=1e-14)
        self.assertEqual(paths[0][-1, 0], 1e-8)

    def test_event_counters_survive_work_stealing(self):
        field = CatapultBoozerField(constant_field(0), 2, 2, 2)
        n = 50000
        points = np.tile([0.3, 0, 0], (n, 1))
        zero = np.arange(n) % 3 == 0
        paths, hits = trace_particles_boozer_gpu(
            field,
            points,
            np.full(n, SPEED),
            tmax=np.where(zero, 0, 1e-7),
            dt=1e-9,
            stopping_criteria=[IterationStoppingCriterion(0)],
            forget_exact_path=True,
        )
        np.testing.assert_array_equal([len(hit) for hit in hits], ~zero)
        np.testing.assert_allclose(
            [path[-1, 0] for path in paths], np.where(zero, 0, 1e-9), rtol=1e-14
        )

    def test_phase_stop_count_and_time_dependent_plane(self):
        field = CatapultBoozerField(constant_field(), 2, 2, 2)
        points = np.array([[0.3, 0, 0]])
        for options, count in [({"phases_stop": True}, 1), ({"max_phase_hits": 3}, 3)]:
            paths, hits = trace_particles_boozer_gpu(
                field,
                points,
                [SPEED],
                tmax=1e-5,
                dt_save=1e-9,
                phases=[0],
                n_zetas=[0],
                m_thetas=[0],
                omegas=[-PERIOD / 1e-7],
                **options,
            )
            self.assertEqual(len(hits[0]), count)
            np.testing.assert_allclose(
                hits[0][:, 0], np.arange(1, count + 1) * 1e-7, rtol=1e-13
            )
            self.assertEqual(paths[0][-1, 0], hits[0][-1, 0])
            self.assertTrue(np.all(np.diff(paths[0][:, 0]) > 0))

    def test_standard_criterion_indices_and_endpoint_checks(self):
        field = CatapultBoozerField(constant_field(0), 2, 2, 2)
        cases = [
            ([MaxToroidalFluxStoppingCriterion(0.2)], -1),
            (
                [
                    MaxToroidalFluxStoppingCriterion(0.8),
                    MinToroidalFluxStoppingCriterion(0.4),
                ],
                -2,
            ),
            ([IterationStoppingCriterion(0)], -1),
            ([StepSizeStoppingCriterion(1e-8)], -1),
            ([None, IterationStoppingCriterion(0)], -2),
        ]
        for criteria, index in cases:
            paths, hits = trace_particles_boozer_gpu(
                field,
                np.array([[0.3, 0, 0]]),
                [SPEED],
                tmax=1e-6,
                dt=1e-9,
                dt_save=1e-10,
                stopping_criteria=criteria,
            )
            self.assertEqual(hits[0][0, 1], index)
            self.assertEqual(len(hits[0]), 1)
            np.testing.assert_allclose(paths[0][-1, 0], 1e-9, rtol=1e-14)
            np.testing.assert_allclose(
                paths[0][-1], hits[0][0, [0, 2, 3, 4, 5]], rtol=1e-13, atol=1e-15
            )
        for sign in [1, -1]:
            paths, hits = trace_particles_boozer_gpu(
                field,
                np.array([[0.3, 0, 0.2]]),
                [sign * SPEED],
                tmax=2e-6,
                stopping_criteria=[ToroidalTransitStoppingCriterion(1)],
                forget_exact_path=True,
            )
            self.assertEqual(hits[0][-1, 1], -1)
            self.assertLess(paths[0][-1, 0], 2e-6)

    def test_velocity_hits_and_mirror_stop_match_cpu(self):
        source = BoozerAnalytic(etabar=0.03, B0=5, N=0, G0=25, psi0=100, iota0=0.4)
        points = np.array([[0.3, np.pi + 0.2, 0]])
        speeds = np.array([0.15 * SPEED])
        options = {
            "tmax": 1e-4,
            "vpars": [0.05 * SPEED, 0],
            "vpars_stop": True,
            "mass": MASS,
            "charge": CHARGE,
            "Ekin": ENERGY,
            "forget_exact_path": True,
        }
        _, reference = trace_particles_boozer(
            source, points, speeds, tol=1e-11, **options
        )
        self.assertEqual(len(reference[0]), 1)
        for precision, tol in [("double", 1e-10), ("single", 1e-6)]:
            field = CatapultBoozerField(source, 12, 32, 4, precision=precision)
            paths, hits = trace_particles_boozer_gpu(
                field, points, speeds, tol=tol, **options
            )
            self.assertEqual(len(hits[0]), 1)
            self.assertEqual(hits[0][0, 1], 0)
            self.assertEqual(hits[0][0, 5], 0.05 * SPEED)
            np.testing.assert_allclose(
                hits[0][:, [0, 2, 4]],
                reference[0][:, [0, 2, 4]],
                rtol=2e-3,
                atol=2e-4,
            )
            angle_error = (hits[0][:, 3] - reference[0][:, 3] + np.pi) % PERIOD - np.pi
            np.testing.assert_allclose(angle_error, 0, atol=2e-4)
            self.assertEqual(paths[0][-1, 0], hits[0][0, 0])
        _, zero_reference = trace_particles_boozer(
            source,
            points,
            speeds,
            tmax=1e-4,
            vpars=[0],
            vpars_stop=True,
            phases=[0],
            n_zetas=[1],
            m_thetas=[0],
            omegas=[0],
            tol=1e-11,
            forget_exact_path=True,
        )
        path, zero_hits = trace_particles_boozer_gpu(
            field,
            points,
            speeds,
            tmax=1e-4,
            zetas=[0],
            vpars=[0],
            vpars_stop=True,
            tol=1e-6,
            forget_exact_path=True,
        )
        self.assertEqual(zero_hits[0][-1, 1], 1)
        self.assertEqual(zero_hits[0][-1, 5], 0)
        np.testing.assert_allclose(
            zero_hits[0][-1, 0], zero_reference[0][-1, 0], rtol=2e-3
        )
        self.assertEqual(path[0][-1, 0], zero_hits[0][-1, 0])
        _, recorded = trace_particles_boozer_gpu(
            field,
            points,
            speeds,
            tmax=8e-5,
            vpars=[0],
            max_hits=16,
            tol=1e-6,
            forget_exact_path=True,
        )
        self.assertGreater(len(recorded[0]), 1)
        np.testing.assert_array_equal(recorded[0][:, 5], 0)

    def test_perturbed_kernel_events_keep_absolute_time(self):
        from tests.field.test_catapult_field import get_saw

        source = constant_field()
        for equilibrium in [
            source,
            BoozerAnalytic(etabar=0, B0=1, N=0, G0=1, I0=0.1, psi0=1, iota0=0.4),
        ]:
            field = CatapultPerturbedBoozerField(get_saw(equilibrium), 3, 3, 3)
            points = np.array([[0.3, 0, 0]])
            mus = np.array([0.0])
            outputs = []
            for cadence in [1e-8, 1e-6]:
                paths, hits = trace_particles_boozer_perturbed_gpu(
                    field,
                    points,
                    [SPEED],
                    mus,
                    Ekin=ENERGY,
                    tmax=1e-6,
                    phases=[0],
                    n_zetas=[0],
                    omegas=[-PERIOD / 1e-7],
                    max_phase_hits=3,
                    forget_exact_path=False,
                    dt_save=cadence,
                )
                outputs.append(hits[0])
                np.testing.assert_allclose(
                    hits[0][:, 0], [1e-7, 2e-7, 3e-7], rtol=1e-13
                )
                self.assertEqual(paths[0][-1, 0], hits[0][-1, 0])
            np.testing.assert_array_equal(*outputs)

    def test_cartesian_iteration_stop(self):
        from firm3d.catapult.field import CatapultCartesianField
        from firm3d.catapult.tracing import trace_particles_cartesian_gpu

        field = CatapultCartesianField.__new__(CatapultCartesianField)
        field.dtype = np.dtype(np.float64)
        field.rrange = (0.5, 1.5, 4)
        field.phirange = (0, np.pi, 4)
        field.zrange = (-0.1, 0.1, 4)
        field.quad_info = np.zeros((1, 7, 64))
        field.quad_info[0, 2, :] = 1
        field.quad_info[0, 6, :] = 1
        paths, hits = trace_particles_cartesian_gpu(
            field,
            np.array([[1, 0, 0]]),
            [SPEED],
            tmax=1e-7,
            dt=1e-9,
            stopping_criteria=[IterationStoppingCriterion(0)],
            vpars=[0],
            forget_exact_path=True,
        )
        self.assertEqual(hits[0][0, 1], -1)
        np.testing.assert_allclose(paths[0][-1, 0], 1e-9, rtol=1e-14)
        with self.assertRaisesRegex(ValueError, "Boozer"):
            trace_particles_cartesian_gpu(
                field,
                np.array([[1, 0, 0]]),
                [SPEED],
                stopping_criteria=[MaxToroidalFluxStoppingCriterion(1)],
            )

    def test_overflow_fails_and_next_trace_succeeds(self):
        field = CatapultBoozerField(constant_field(), 2, 2, 2)
        options = {"tmax": 2e-6, "zetas": [0], "forget_exact_path": True}
        with self.assertRaisesRegex((ValueError, OverflowError), "max_hits"):
            trace_particles_boozer_gpu(
                field, np.array([[0.3, 0, 0]]), [SPEED], max_hits=1, **options
            )
        _, hits = trace_particles_boozer_gpu(
            field, np.array([[0.3, 0, 0]]), [SPEED], max_hits=8, **options
        )
        self.assertEqual(len(hits[0]), int(SPEED * 2e-6 / PERIOD))


if __name__ == "__main__":
    unittest.main()
