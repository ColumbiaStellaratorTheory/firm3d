"""GPU dense sampling: numerical reference, host routing, and device checks."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import firm3dpp

from firm3d.catapult.field import (
    CatapultBoozerField,
    CatapultCartesianField,
    CatapultPerturbedBoozerField,
)
from firm3d.catapult.tracing import (
    _save_times,
    save_trajectories_boozer_gpu,
    save_trajectories_cartesian_gpu,
    trace_particles_boozer_gpu,
    trace_particles_boozer_perturbed_gpu,
    trace_particles_cartesian_gpu,
)
from firm3d.field.boozermagneticfield import BoozerAnalytic
from firm3d.field.tracing import trace_particles_boozer
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE as CHARGE,
    ALPHA_PARTICLE_MASS as MASS,
    FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
)

ROOT = Path(__file__).resolve().parents[2]
HAS_CUDA = hasattr(firm3dpp, "boozer_gpu_tracing")
SPEED = np.sqrt(2 * ENERGY / MASS)


class TestDenseOutputHost(unittest.TestCase):
    def test_continuous_extension(self):
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not shutil.which(compiler[0]):
            self.skipTest("C++ compiler not available")
        with tempfile.TemporaryDirectory() as folder:
            exe = str(Path(folder) / "dense_output")
            command = compiler + [
                "-std=c++17",
                "-O2",
                "-I" + str(ROOT / "src/firm3dpp"),
            ]
            # MacPorts installs the CPU Boost headers outside the default path.
            if Path("/opt/local/include/boost").exists():
                command.append("-I/opt/local/include")
            subprocess.run(
                command + [str(ROOT / "tests/dopri5_dense_output.cpp"), "-o", exe],
                check=True,
                capture_output=True,
                text=True,
            )
            result = subprocess.run([exe], check=True, capture_output=True, text=True)
            self.assertIn("Float and double polynomial checks passed", result.stdout)

    def test_save_grid(self):
        np.testing.assert_allclose(_save_times([2.5, 4.0], 1), [1, 2, 3])
        self.assertEqual(len(_save_times([1e-3], 1e-6)), 999)
        self.assertEqual(len(_save_times([0.2], 1)), 0)
        self.assertEqual(len(_save_times([0], 1)), 0)
        self.assertEqual(len(_save_times([], 1)), 0)
        for dt_save in [0, -1, np.nan, np.inf]:
            with self.subTest(dt_save=dt_save), self.assertRaises(ValueError):
                _save_times([1], dt_save)
        for tmax in [-1, np.nan, np.inf]:
            with self.subTest(tmax=tmax), self.assertRaises(ValueError):
                _save_times([tmax], 1)

    def test_one_launch_and_particle_order(self):
        # Mock only the device launch. Exercise the public API's precision,
        # save grid, trimming, per-particle tmax, and CPU result assembly.
        for dtype in [np.float64, np.float32]:
            field = CatapultBoozerField.__new__(CatapultBoozerField)
            field.dtype = np.dtype(dtype)
            field.vacuum = True
            field.psi0 = 1.0
            field.srange = field.trange = field.zrange = (0, 1, 4)
            field.quad_info = np.zeros(64 * 6, dtype=dtype)
            stz = np.array([[0.3, 0.0, 0.1], [0.4, 0.0, 0.2], [0.5, 0, 0.3]])
            vpar = np.array([1e6, 2e6, 3e6])
            terminal = np.array([2.5, 1.2, 0])

            def launch(dtype=dtype, stz=stz, vpar=vpar, **kwargs):
                self.assertEqual(kwargs["stz_init"].dtype, dtype)
                np.testing.assert_array_equal(kwargs["save_times"], [1, 2])
                out = np.full((3, 3, 7), np.nan, dtype=dtype)
                for i, times in enumerate(([1, 2, 2.5], [1, 1.2], [0])):
                    for j, time in enumerate(times):
                        out[i, j] = [time, stz[i, 0], 0, stz[i, 2], vpar[i], 0.2, 0.1]
                return out.ravel()

            with patch.object(firm3dpp, "boozer_gpu_tracing", launch, create=True):
                tys, hits = trace_particles_boozer_gpu(
                    field,
                    stz,
                    vpar,
                    tmax=terminal,
                    dt_save=1,
                )
            self.assertEqual([len(traj) for traj in tys], [4, 3, 1])
            for i, traj in enumerate(tys):
                np.testing.assert_array_equal(traj[0], [0, *stz[i], vpar[i]])
                self.assertAlmostEqual(traj[-1, 0], terminal[i], places=6)
                self.assertEqual(len(hits[i]), 0)

    def test_perturbed_sampling_uses_one_absolute_time_launch(self):
        field = CatapultPerturbedBoozerField.__new__(CatapultPerturbedBoozerField)
        field.dtype = np.dtype(np.float64)
        field.vacuum = True
        field.psi0 = 1
        field.srange = field.trange = field.zrange = (0, 1, 4)
        field.quad_info = np.zeros(640)
        field.saw_omega = 123
        field.saw_srange = (0, 1, 4)
        field.saw_m, field.saw_n = [1], [2]
        field.saw_nharmonics = 1
        field.saw_phihats = np.ones((4, 1))
        result = np.array(
            [[1, 0.3, 0, 0, 1e6, 0.1, 0.2], [2, 0.3, 0, 0, 1e6, 0.1, 0.2]]
        )
        with patch.object(
            firm3dpp,
            "boozer_saw_gpu_tracing",
            return_value=result.ravel(),
            create=True,
        ) as launch:
            tys, _ = trace_particles_boozer_perturbed_gpu(
                field,
                np.array([[0.3, 0, 0]]),
                np.array([1e6]),
                np.array([0.2]),
                tmax=2,
                dt_save=1,
                Ekin=ENERGY,
                forget_exact_path=False,
            )
        self.assertEqual(launch.call_count, 1)
        np.testing.assert_array_equal(launch.call_args.kwargs["save_times"], [1])
        np.testing.assert_array_equal(tys[0][:, 0], [0, 1, 2])


@unittest.skipUnless(HAS_CUDA, "CUDA support not available")
class TestDenseOutputGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Constant |B| gives exact s and vpar, and linear theta and zeta.
        cls.field = BoozerAnalytic(etabar=0, B0=1, N=0, G0=1, psi0=1, iota0=0.4)
        cls.stz = np.array([[0.3, 0.1, 0.2], [0.6, -0.2, 6.28]])
        cls.vpar = np.array([0.7, -0.5]) * SPEED

    def test_multiple_samples_in_one_step_and_off_grid_tmax(self):
        tmax, dt_save = 5.55e-9, 1e-10
        expected_times = np.r_[np.arange(1, 56) * dt_save, tmax]
        for precision, tol in [("double", 1e-9), ("single", 1e-6)]:
            with self.subTest(precision=precision):
                gpu = CatapultBoozerField(self.field, 3, 3, 3, precision=precision)
                trajectories = save_trajectories_boozer_gpu(
                    gpu,
                    self.stz,
                    self.vpar,
                    tmax,
                    dt_save,
                    MASS,
                    CHARGE,
                    SPEED,
                    tol,
                    dt=1e-9,
                )
                final, _ = trace_particles_boozer_gpu(
                    gpu,
                    self.stz,
                    self.vpar,
                    tmax=tmax,
                    tol=tol,
                    forget_exact_path=True,
                )
                for i, traj in enumerate(trajectories):
                    np.testing.assert_allclose(
                        traj[:, 0], expected_times, rtol=1e-7, atol=0
                    )
                    self.assertTrue(np.any(traj[:, 5] > 5 * dt_save))
                    exact = np.column_stack(
                        (
                            np.full(len(traj), self.stz[i, 0]),
                            self.stz[i, 1] + 0.4 * self.vpar[i] * expected_times,
                            (self.stz[i, 2] + self.vpar[i] * expected_times)
                            % (2 * np.pi),
                            np.full(len(traj), self.vpar[i]),
                        )
                    )
                    np.testing.assert_allclose(
                        traj[:, 1:5], exact, rtol=3e-6, atol=3e-6
                    )
                    np.testing.assert_allclose(
                        traj[-1, :5], final[i][-1], rtol=3e-6, atol=3e-6
                    )

    def test_cpu_dense_output_at_same_times(self):
        gpu = CatapultBoozerField(self.field, 3, 3, 3)
        tmax, dt_save = 2.37e-7, 1e-8
        kwargs = {
            "tmax": tmax,
            "dt_save": dt_save,
            "mass": MASS,
            "charge": CHARGE,
            "Ekin": ENERGY,
            "tol": 1e-10,
        }
        gpu_tys, _ = trace_particles_boozer_gpu(gpu, self.stz, self.vpar, **kwargs)
        cpu_tys, _ = trace_particles_boozer(
            self.field,
            self.stz,
            self.vpar,
            axis=2,
            **kwargs,
        )
        for actual, expected in zip(gpu_tys, cpu_tys):
            np.testing.assert_allclose(
                actual[:, 0], expected[:, 0], rtol=1e-14, atol=1e-20
            )
            # Compare angles through sine/cosine to allow their branch cuts.
            np.testing.assert_allclose(
                actual[:, [1, 4]], expected[:, [1, 4]], rtol=1e-9, atol=1e-9
            )
            for column in [2, 3]:
                np.testing.assert_allclose(
                    np.sin(actual[:, column]), np.sin(expected[:, column]), atol=1e-9
                )
                np.testing.assert_allclose(
                    np.cos(actual[:, column]), np.cos(expected[:, column]), atol=1e-9
                )

    def test_per_particle_tmax_and_zero_time(self):
        gpu = CatapultBoozerField(self.field, 3, 3, 3)
        stz = np.tile(self.stz[0], (3, 1))
        vpar = np.full(3, self.vpar[0])
        tys, _ = trace_particles_boozer_gpu(
            gpu,
            stz,
            vpar,
            tmax=np.array([0, 1e-9, 3.5e-9]),
            dt_save=1e-9,
        )
        for traj, times in zip(tys, ([0], [0, 1e-9], [0, 1e-9, 2e-9, 3e-9, 3.5e-9])):
            np.testing.assert_allclose(traj[:, 0], times, rtol=1e-14, atol=0)

    def test_save_cadence_does_not_change_adaptive_trace(self):
        gpu = CatapultBoozerField(self.field, 3, 3, 3)
        coarse, _ = trace_particles_boozer_gpu(
            gpu,
            self.stz,
            self.vpar,
            tmax=1e-7,
            dt_save=1e-8,
        )
        fine, _ = trace_particles_boozer_gpu(
            gpu,
            self.stz,
            self.vpar,
            tmax=1e-7,
            dt_save=1e-9,
        )
        for a, b in zip(coarse, fine):
            np.testing.assert_array_equal(a[-1], b[-1])

    def test_cartesian_losses_zero_times_and_work_stealing(self):
        # An exact constant Cartesian field and linear signed distance table
        # avoid a simsopt dependency. More particles than the A100's maximum
        # resident blocks can hold forces the persistent kernel to refill slots.
        gpu = CatapultCartesianField.__new__(CatapultCartesianField)
        gpu.dtype = np.dtype(np.float64)
        gpu.rrange = (0.5, 1.5, 4)
        gpu.phirange = (0, np.pi, 4)
        gpu.zrange = (-0.1, 0.1, 4)
        gpu.quad_info = np.zeros((1, 7, 64))
        gpu.quad_info[0, 2, :] = 1
        gpu.quad_info[0, 6, :] = np.tile(0.02 - np.linspace(-0.1, 0.1, 4), 16)
        nparticles = 50000
        ids = np.arange(nparticles)
        stz = np.zeros((nparticles, 3))
        stz[:, 0], stz[:, 1] = np.cos(0.2), np.sin(0.2)
        lost = ids % 3 == 1
        zero = ids % 3 == 0
        stz[lost, 2] = 0.019
        vpar = SPEED * (0.5 + 0.1 * ids / nparticles)
        tmax = np.where(zero, 0, 1e-9)
        tys, hits = trace_particles_cartesian_gpu(
            gpu,
            stz,
            vpar,
            tmax=tmax,
            dt_save=2e-10,
        )
        self.assertEqual(len(tys), nparticles)
        for i, traj in enumerate(tys):
            np.testing.assert_array_equal(traj[0], [0, *stz[i], vpar[i]])
            np.testing.assert_allclose(
                traj[:, 3], stz[i, 2] + vpar[i] * traj[:, 0], atol=1e-12
            )
            np.testing.assert_allclose(traj[:, 4], vpar[i], rtol=1e-14)
            self.assertEqual(bool(len(hits[i])), bool(lost[i]))
            if zero[i]:
                self.assertEqual(len(traj), 1)
            elif lost[i]:
                self.assertLess(traj[-1, 0], tmax[i])
                self.assertGreaterEqual(traj[-1, 3], 0.02)
            else:
                np.testing.assert_allclose(traj[:, 0], np.arange(6) * 2e-10, atol=1e-24)

        # The accepted loss step is h=1e-9; the next proposed step is larger.
        # Seven-column output reports the enclosing step, including at a loss.
        loss = save_trajectories_cartesian_gpu(
            gpu,
            stz[1:2],
            vpar[1:2],
            1e-8,
            1e-10,
            MASS,
            CHARGE,
            SPEED,
            1e-9,
            dt=1e-9,
        )[0]
        self.assertAlmostEqual(loss[-1, 0], 1e-9, delta=1e-23)
        self.assertAlmostEqual(loss[-1, 5], 1e-9, delta=1e-23)

    def test_perturbed_and_finite_beta_save_cadence(self):
        from tests.field.test_catapult_field import get_saw

        finite_beta = BoozerAnalytic(
            etabar=0.1,
            B0=1,
            N=0,
            G0=1,
            psi0=1,
            iota0=0.4,
            K1=0.1,
            I0=0.05,
        )
        equilibrium = CatapultBoozerField(finite_beta, 4, 4, 4)
        waves = CatapultPerturbedBoozerField(get_saw(self.field), 4, 4, 4)
        # The finite-beta wave kernel uses the supported no-K equilibrium.
        no_k = BoozerAnalytic(etabar=0.1, B0=1, N=0, G0=1, psi0=1, iota0=0.4, I0=0.05)
        self.assertEqual(no_k.field_type, "nok")
        finite_beta_waves = CatapultPerturbedBoozerField(get_saw(no_k), 4, 4, 4)
        self.field.set_points(self.stz)
        mus = (SPEED**2 - self.vpar**2) / (2 * self.field.modB().ravel())
        for field in [equilibrium, waves, finite_beta_waves]:

            def run(dt_save, field=field):
                kwargs = {
                    "tmax": 2.35e-7,
                    "dt_save": dt_save,
                    "forget_exact_path": False,
                    "Ekin": ENERGY,
                }
                if isinstance(field, CatapultPerturbedBoozerField):
                    return trace_particles_boozer_perturbed_gpu(
                        field,
                        self.stz,
                        self.vpar,
                        mus,
                        **kwargs,
                    )[0]
                return trace_particles_boozer_gpu(field, self.stz, self.vpar, **kwargs)[
                    0
                ]

            coarse = run(1e-8)
            fine = run(1e-9)
            for a, b in zip(coarse, fine):
                np.testing.assert_array_equal(a[-1], b[-1])
                self.assertAlmostEqual(a[-1, 0], 2.35e-7, places=15)


if __name__ == "__main__":
    unittest.main()
