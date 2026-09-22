import unittest
from pathlib import Path

import numpy as np

from firm3d.catapult.field import CatapultBoozerField, CatapultPerturbedBoozerField
from firm3d.catapult.tracing import (
    advance_particles_boozer_gpu,
    advance_particles_boozer_perturbed_gpu,
    save_trajectories_boozer_gpu,
    trace_particles_boozer_gpu,
    trace_particles_boozer_perturbed_gpu,
)
from firm3d.catapult.utils import boozer_interpolant, boozer_saw_interpolant
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    ShearAlfvenWavesSuperposition,
)
from firm3d.field.tracing import trace_particles_boozer
from firm3d.saw.ae3d import AE3DEigenvector

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
EXAMPLES_DIR = (Path(__file__).parent / ".." / ".." / "examples").resolve()
filename_vac = str(TEST_DIR / "boozmn_LandremanPaul2021_QA_lowres.nc")
filename_mhd = str(TEST_DIR / "boozmn_n3are_R7.75B5.7.nc")
filename_ae = str(EXAMPLES_DIR / "tracing_with_AE" / "ae.npy")

RESOLUTION = (2, 3, 4)


def get_saw(field):
    return ShearAlfvenWavesSuperposition.from_ae3d(
        eigenvector=AE3DEigenvector.load_from_numpy(filename=filename_ae),
        B0=field,
        max_dB_normal_by_B0=5e-3,
        minor_radius_meters=1.7,
    )


class TestCatapultBoozerField(unittest.TestCase):
    """
    Building the field objects needs no GPU, so these run in the CPU suite.
    """

    def setUp(self):
        self.field = BoozerRadialInterpolant(filename_vac, 3, enforce_vacuum=True)

    def test_matches_interpolant(self):
        cfield = CatapultBoozerField(self.field, *RESOLUTION)
        srange, trange, zrange, quad_info, maxJ = boozer_interpolant(
            self.field, self.field.nfp, *RESOLUTION, vacuum=True
        )
        self.assertEqual(cfield.srange, srange)
        self.assertEqual(cfield.trange, trange)
        self.assertEqual(cfield.zrange, zrange)
        np.testing.assert_array_equal(cfield.quad_info, quad_info)
        self.assertEqual(cfield.maxJ, maxJ)
        self.assertEqual(cfield.dtype, np.float64)
        self.assertEqual(cfield.precision, "double")
        self.assertTrue(cfield.vacuum)
        self.assertEqual(cfield.field_type, "vac")
        self.assertEqual(cfield.nfp, self.field.nfp)
        self.assertEqual(cfield.psi0, self.field.psi0)

    def test_single_precision(self):
        double = CatapultBoozerField(self.field, *RESOLUTION)
        single = CatapultBoozerField(self.field, *RESOLUTION, precision="single")
        self.assertEqual(single.dtype, np.float32)
        self.assertEqual(single.precision, "single")
        np.testing.assert_array_equal(
            single.quad_info, double.quad_info.astype(np.float32)
        )
        self.assertEqual(
            CatapultBoozerField(self.field, *RESOLUTION, precision=np.float32).dtype,
            np.float32,
        )

    def test_finite_beta(self):
        field = BoozerRadialInterpolant(filename_mhd, 3)
        cfield = CatapultBoozerField(field, *RESOLUTION)
        self.assertFalse(cfield.vacuum)
        self.assertEqual(cfield.field_type, "")

    def test_rejects(self):
        with self.assertRaises(ValueError):
            CatapultBoozerField(self.field, *RESOLUTION, precision="half")
        # a perturbed field belongs to CatapultPerturbedBoozerField
        with self.assertRaises(TypeError):
            CatapultBoozerField(get_saw(self.field), *RESOLUTION)
        cfield = CatapultBoozerField(self.field, *RESOLUTION)
        stz = np.array([[0.5, 0.0, 0.0]])
        vpar = np.array([1e6])
        # the tracers take a tabulated field, not the field it was built from
        with self.assertRaises(TypeError):
            trace_particles_boozer_gpu(self.field, stz, vpar)
        # CATAPULT takes one kinetic energy for all particles
        with self.assertRaises(ValueError):
            trace_particles_boozer_gpu(cfield, stz, vpar, Ekin=np.array([1e6]))


class TestCatapultPerturbedBoozerField(unittest.TestCase):
    def setUp(self):
        self.field = BoozerRadialInterpolant(filename_vac, 3, enforce_vacuum=True)
        self.saw = get_saw(self.field)

    def test_matches_interpolant(self):
        cfield = CatapultPerturbedBoozerField(self.saw, *RESOLUTION)
        srange, trange, zrange, quad_info, maxJ = boozer_saw_interpolant(
            self.field, self.field.nfp, *RESOLUTION
        )
        self.assertEqual(
            (cfield.srange, cfield.trange, cfield.zrange), (srange, trange, zrange)
        )
        np.testing.assert_array_equal(cfield.quad_info, quad_info)
        self.assertEqual(cfield.maxJ, maxJ)
        self.assertIs(cfield.B0, self.field)
        self.assertTrue(cfield.vacuum)
        self.assertEqual(cfield.psi0, self.field.psi0)
        self.assertEqual(cfield.saw_nharmonics, len(self.saw))
        self.assertEqual(cfield.saw_phihats.shape[1], len(self.saw))
        self.assertEqual(cfield.saw_omega, self.saw.get_wave(0).omega)

    def test_rejects(self):
        # an equilibrium belongs to CatapultBoozerField
        with self.assertRaises(TypeError):
            CatapultPerturbedBoozerField(self.field, *RESOLUTION)

    def test_single_precision(self):
        double = CatapultPerturbedBoozerField(self.saw, *RESOLUTION)
        single = CatapultPerturbedBoozerField(self.saw, *RESOLUTION, precision="single")
        self.assertEqual(single.dtype, np.float32)
        self.assertEqual(single.saw_phihats.dtype, np.float32)
        np.testing.assert_array_equal(
            single.quad_info, double.quad_info.astype(np.float32)
        )
        np.testing.assert_array_equal(
            single.saw_phihats, double.saw_phihats.astype(np.float32)
        )

    def test_refused_before_launch(self):
        # what the kernels cannot take is refused in Python, before any launch
        equilibrium = CatapultBoozerField(self.field, *RESOLUTION)
        perturbed = CatapultPerturbedBoozerField(self.saw, *RESOLUTION)
        stz = np.array([[0.5, 0.0, 0.0], [0.4, 1.0, 2.0]])
        vpar = np.array([1e6, -1e6])
        mus = np.array([1e6, 1e6])
        args = (1e-6, 1.0, 1.0, 1e6, 1e-8)

        # each tracer and saver keeps to its kind of field, on both backends
        with self.assertRaises(TypeError):
            trace_particles_boozer_gpu(perturbed, stz, vpar)
        with self.assertRaises(TypeError):
            save_trajectories_boozer_gpu(perturbed, stz, vpar, 1e-6, 1e-7, *args[1:])
        with self.assertRaises(TypeError):
            trace_particles_boozer_perturbed_gpu(equilibrium, stz, vpar, mus)
        with self.assertRaises(TypeError):
            trace_particles_boozer(self.saw, stz, vpar)

        # a per-particle array of the wrong length would be read past its
        # end by the kernel: one case per path that checks it
        with self.assertRaises(ValueError):
            advance_particles_boozer_gpu(equilibrium, stz, vpar[:1], *args)
        with self.assertRaises(ValueError):
            advance_particles_boozer_gpu(equilibrium, stz[:, :2], vpar, *args)
        with self.assertRaises(ValueError):
            advance_particles_boozer_perturbed_gpu(perturbed, stz, vpar, mus[:1])
        with self.assertRaises(ValueError):
            trace_particles_boozer_gpu(
                equilibrium, stz, vpar[:1], forget_exact_path=True
            )

        # and anything non-finite, which would crash the kernel: one case per
        # check (a per-particle array, the positions, the speed)
        with self.assertRaises(ValueError):
            advance_particles_boozer_gpu(
                equilibrium, stz, np.array([1e6, np.nan]), *args
            )
        with self.assertRaises(ValueError):
            advance_particles_boozer_gpu(equilibrium, stz * np.nan, vpar, *args)
        with self.assertRaises(ValueError):
            advance_particles_boozer_gpu(
                equilibrium, stz, vpar, 1e-6, 1.0, 1.0, np.nan, 1e-8
            )

        # trajectories need one tmax for all particles, and cannot yet be
        # saved in a perturbed field
        with self.assertRaises(NotImplementedError):
            trace_particles_boozer_gpu(
                equilibrium, stz, vpar, tmax=np.array([1e-6, 2e-6])
            )
        with self.assertRaises(NotImplementedError):
            trace_particles_boozer_perturbed_gpu(
                perturbed, stz, vpar, mus, forget_exact_path=False
            )


if __name__ == "__main__":
    unittest.main()
