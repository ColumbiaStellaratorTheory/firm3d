"""CATAPULT passing maps: host contracts and independent CPU/analytic checks."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import firm3dpp

from firm3d.catapult.field import CatapultBoozerField
from firm3d.field.boozermagneticfield import BoozerAnalytic, InterpolatedBoozerField
from firm3d.trajectory_helpers import PassingPoincare, compute_rotational_profile
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE as CHARGE,
    ALPHA_PARTICLE_MASS as MASS,
    FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
)

PERIOD = 2 * np.pi
SPEED = np.sqrt(2 * ENERGY / MASS)
GPU_TRACE = "firm3d.catapult.tracing.trace_particles_boozer_gpu"
HAS_CUDA = hasattr(firm3dpp, "boozer_gpu_tracing")


def constant_field():
    return BoozerAnalytic(etabar=0, B0=5, N=0, G0=25, I0=0.2, iota0=0.4, psi0=1)


def map_options(**kwargs):
    return dict(
        lam=0,
        sign_vpar=1,
        mass=MASS,
        charge=CHARGE,
        Ekin=ENERGY,
        s_init=[0.3],
        thetas_init=[0.2],
        Nmaps=2,
        **kwargs,
    )


def sampled_path(rate=PERIOD / 7):
    times = np.arange(0, 25, 0.13)
    return np.column_stack(
        (
            times,
            0.3 + 0.005 * times,
            (0.2 + 0.03 * times + np.pi) % PERIOD - np.pi,
            rate * times % PERIOD,
            np.full(len(times), SPEED),
        )
    )


def native_hits(transit=7, count=2):
    times = transit * np.arange(1, count + 1)
    return np.column_stack(
        (
            times,
            np.zeros(count),
            0.3 + 0.005 * times,
            0.2 + 0.03 * times,
            np.zeros(count),
            np.full(count, SPEED),
        )
    )


class TestCatapultPoincareHost(unittest.TestCase):
    def setUp(self):
        self.source = constant_field()
        self.field = CatapultBoozerField(self.source, 2, 2, 2)

    def test_nok_table_preserves_covariant_components(self):
        self.assertEqual(self.field.field_type, "nok")
        self.assertFalse(self.field.vacuum)
        table = self.field.quad_info
        np.testing.assert_allclose(table[:, 4], 25)
        np.testing.assert_allclose(table[:, 6], 0.2)
        np.testing.assert_allclose(table[:, 8], 0.4)
        np.testing.assert_array_equal(table[:, 9:], 0)
        # No-K input may not implement K at all.
        with patch.object(self.source, "K", side_effect=AssertionError("K requested")):
            CatapultBoozerField(self.source, 2, 2, 2)

    def test_batched_trace_and_cpu_transit_time_format(self):
        with patch(
            GPU_TRACE, return_value=([sampled_path()], [native_hits()])
        ) as trace:
            poinc = PassingPoincare(
                self.field,
                **map_options(),
                tmax=25,
                dt_save=0.13,
                solver_options={"abstol": 1e-8, "reltol": 1e-8},
            )
        trace.assert_called_once()
        self.assertTrue(trace.call_args.kwargs["forget_exact_path"])
        self.assertEqual(trace.call_args.kwargs["zetas"], [0])
        self.assertEqual(trace.call_args.kwargs["vpars"], [0])
        self.assertTrue(trace.call_args.kwargs["vpars_stop"])
        self.assertEqual(trace.call_args.kwargs["max_phase_hits"], 2)
        self.assertEqual(trace.call_args.kwargs["tol"], 1e-8)
        self.assertIs(poinc.field, self.source)
        self.assertEqual(poinc.backend, "catapult")
        np.testing.assert_allclose(poinc.t_all, [[0, 7, 7]])
        np.testing.assert_allclose(poinc.s_all, [[0.3, 0.335, 0.37]])
        np.testing.assert_allclose(poinc.thetas_all, [[0.2, 0.41, 0.62]])
        self.assertEqual(poinc.peta_all, [])
        self.assertEqual(poinc.DA_all, [[]])

    def test_section_points_come_from_native_hits(self):
        hits = native_hits()
        hits[:, 2] = [0.42, 0.43]
        with patch(GPU_TRACE, return_value=([sampled_path()[[0, -1]]], [hits])):
            poinc = PassingPoincare(self.field, **map_options(), tmax=25, dt_save=25)
        np.testing.assert_allclose(poinc.s_all, [[0.3, 0.42, 0.43]])
        np.testing.assert_allclose(poinc.t_all, [[0, 7, 7]])

    def test_loss_and_velocity_reversal_discard_later_returns(self):
        for column, value in [(1, 0.995), (4, -SPEED)]:
            with self.subTest(column=column):
                path = sampled_path()
                path[path[:, 0] >= 10, column] = value
                stop = np.array([[10, -1 if column == 1 else 1, 0.99, 0.5, 2.7, 0]])
                hits = np.vstack((native_hits(count=1), stop))
                with patch(GPU_TRACE, return_value=([path[path[:, 0] <= 10]], [hits])):
                    poinc = PassingPoincare(self.field, **map_options(), tmax=25)
                np.testing.assert_allclose(poinc.t_all, [[0, 7]])

    def test_short_horizon_warns_and_keeps_completed_returns(self):
        path = sampled_path()
        path = path[path[:, 0] < 10]
        with (
            patch(GPU_TRACE, return_value=([path], [native_hits(count=1)])),
            self.assertWarnsRegex(UserWarning, "reached tmax before Nmaps"),
        ):
            poinc = PassingPoincare(self.field, **map_options(), tmax=10)
        np.testing.assert_allclose(poinc.t_all, [[0, 7]])

    def test_momentum_and_wba_stop_at_requested_return(self):
        options = map_options()
        options["Nmaps"] = 4
        with (
            patch(
                GPU_TRACE,
                return_value=([sampled_path(PERIOD / 4)], [native_hits(4, 4)]),
            ),
            patch(
                "firm3d.trajectory_helpers.poincare.return_DA",
                side_effect=lambda values: (values[-1, 0], 9),
            ) as wba,
        ):
            poinc = PassingPoincare(
                self.field,
                **options,
                tmax=25,
                helicity_M=1,
                helicity_N=0,
                chaos_detection=True,
                nconvergence_points=2,
            )
        self.assertEqual(len(poinc.peta_all[0]), 5)
        self.assertTrue(np.all(np.isfinite(poinc.peta_all[0])))
        self.assertEqual(poinc.DA_times, [[2, 3]])
        self.assertEqual(poinc.DA_all, [[9, 9]])
        np.testing.assert_allclose(
            [call.args[0][-1, 0] for call in wba.call_args_list], [12, 16]
        )
        for call in wba.call_args_list:
            self.assertTrue(np.all(np.diff(call.args[0][:, 0]) > 0))

    def test_empty_launches_and_zero_maps_do_not_launch_gpu(self):
        for changes, expected in [({"lam": 1}, []), ({"Nmaps": 0}, [[0.3]])]:
            options = map_options()
            options.update(changes)
            with patch(GPU_TRACE) as trace:
                poinc = PassingPoincare(self.field, **options)
            trace.assert_not_called()
            self.assertEqual(poinc.s_all, expected)

    def test_unsupported_solver_settings_and_invalid_cadence(self):
        for options in [
            {"abstol": 1e-8, "reltol": 1e-9},
            {"axis": 1},
            {"ODE_solver": "rk4"},
            {"roottol": 1e-12},
            {"tol": 0},
        ]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                PassingPoincare(self.field, **map_options(), solver_options=options)
        for cadence in [0, -1, np.nan, np.inf]:
            with self.subTest(cadence=cadence), self.assertRaises(ValueError):
                PassingPoincare(self.field, **map_options(), dt_save=cadence)


@unittest.skipUnless(HAS_CUDA, "requires CUDA bindings and a GPU")
class TestCatapultPoincareGPU(unittest.TestCase):
    def test_rotational_profile_uses_catapult_maps(self):
        field = CatapultBoozerField(constant_field(), 2, 2, 2)
        profile = compute_rotational_profile(
            field,
            0,
            1,
            MASS,
            CHARGE,
            ENERGY,
            1,
            0,
            0,
            1,
            None,
            ns_poinc=2,
            Nmaps=4,
            s_profile=True,
            tmax=1.5e-5,
            dt_save=1e-8,
            solver_options={"tol": 1e-10},
        )
        np.testing.assert_allclose(profile[:, 0], [1 / 3, 2 / 3], atol=1e-8)
        np.testing.assert_allclose(profile[:, 3], 0.4, atol=1e-6)

    def test_constant_nok_field_matches_analytic_returns_and_cpu(self):
        source = constant_field()
        for precision in ["double", "single"]:
            field = CatapultBoozerField(source, 2, 2, 2, precision=precision)
            for sign in [1, -1]:
                with self.subTest(precision=precision, sign=sign):
                    options = map_options(
                        tmax=1.5e-5,
                        solver_options={
                            "tol": 1e-10 if precision == "double" else 1e-8
                        },
                    )
                    options.update(sign_vpar=sign, Nmaps=4)
                    cpu = PassingPoincare(source, **options)
                    gpu = PassingPoincare(field, **options, dt_save=1e-8)
                    period = PERIOD * (25 + 0.4 * 0.2) / (SPEED * 5)
                    tolerance = 1e-6 if precision == "double" else 3e-4
                    np.testing.assert_allclose(gpu.s_all, cpu.s_all, atol=tolerance)
                    # CPU event maps wrap each return; a continuous GPU trace
                    # retains the angle's accumulated turns.
                    np.testing.assert_allclose(
                        gpu.thetas_all,
                        np.unwrap(cpu.thetas_all, axis=-1),
                        atol=tolerance,
                    )
                    np.testing.assert_allclose(
                        gpu.thetas_all[0],
                        0.2 + sign * 0.4 * PERIOD * np.arange(5),
                        atol=tolerance,
                    )
                    np.testing.assert_allclose(
                        gpu.t_all[0][1:], period, rtol=tolerance, atol=2e-12
                    )
                    np.testing.assert_allclose(
                        gpu.t_all, cpu.t_all, rtol=tolerance, atol=2e-12
                    )

    def test_cpu_nok_interpolant_agreement_and_physical_sections(self):
        filename = (
            Path(__file__).resolve().parents[2]
            / "examples/inputs/boozmn_aten_rescaled.nc"
        )
        source = InterpolatedBoozerField.from_booz_xform(
            str(filename), degree=3, ns=6, ntheta=6, nzeta=6
        )
        field = CatapultBoozerField(source, 6, 6, 6)
        points = np.random.default_rng(42).uniform(
            [0.02, -PERIOD, -PERIOD], [0.95, PERIOD, PERIOD], (100, 3)
        )
        source.set_points(points)
        expected = np.hstack(
            (
                source.modB(),
                source.modB_derivs(),
                source.G(),
                source.dGds(),
                source.I(),
                source.dIds(),
                source.iota(),
                np.zeros((len(points), 3)),
            )
        )
        actual = firm3dpp.test_gpu_interpolation(
            field.quad_info,
            field.srange,
            field.trange,
            field.zrange,
            points.copy(),
            "boozer",
            len(points),
        ).reshape(expected.shape)
        self.assertLess(
            np.max(np.abs(actual - expected) / (1 + np.abs(expected))), 1e-10
        )
        options = map_options(tmax=1e-4, solver_options={"tol": 1e-9})
        options.update(s_init=[0.2, 0.4], thetas_init=[0, 0], Nmaps=5)
        cpu = PassingPoincare(source, **options)
        gpu = PassingPoincare(field, **options, dt_save=1e-9)
        self.assertEqual([len(p) for p in cpu.s_all], [6, 6])
        self.assertEqual([len(p) for p in gpu.s_all], [6, 6])
        np.testing.assert_allclose(gpu.s_all, cpu.s_all, atol=2e-4)
        error = (np.asarray(gpu.thetas_all) - cpu.thetas_all + np.pi) % PERIOD - np.pi
        self.assertLess(np.max(np.abs(error)), 3e-4)


if __name__ == "__main__":
    unittest.main()
