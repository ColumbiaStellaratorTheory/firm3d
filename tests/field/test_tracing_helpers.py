"""
Tests for the position initialization helpers in tracing_helpers.

The seed/comm tests cover: seed=None must leave the caller's global numpy RNG
alone, and a seeded sample must be identical on every rank. The rank
assertions only have force under more than one rank, e.g.
mpirun -n 2 python -m pytest tests/field/test_tracing_helpers.py

The s-marginal tests check that sampling with weight J * profile(s) gives an
s distribution proportional to profile(s) dV/ds, with dV/ds = 4 pi^2 vp(s)
taken from the VMEC wout file matching the boozmn file.
"""

import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.io import netcdf_file
from scipy.stats import kstest

from firm3d.field.boozermagneticfield import BoozerAnalytic, BoozerRadialInterpolant
from firm3d.field.tracing_helpers import (
    initialize_position_profile,
    initialize_position_uniform_surf,
    initialize_position_uniform_vol,
)
from firm3d.util.functions import sigmav

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
filename_mhd = str(TEST_DIR / "boozmn_n3are_R7.75B5.7.nc")
filename_mhd_wout = str(TEST_DIR / "wout_n3are_R7.75B5.7.nc")

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

NPARTICLES = 32

# An analytic field keeps these tests about the sampling and communication
# structure: no equilibrium file to read and no interpolant to build.
FIELD = BoozerAnalytic(etabar=1.2, B0=1.0, N=0, G0=1.1, psi0=1.0, iota0=0.4)


def sample_profile(seed):
    return initialize_position_profile(
        FIELD,
        NPARTICLES,
        lambda s: (1 - s**5) ** 2,
        ns_max=5,
        ntheta_max=5,
        nzeta_max=5,
        comm=comm,
        seed=seed,
    )


def sample_surf(seed):
    return initialize_position_uniform_surf(
        FIELD, NPARTICLES, 0.3, ntheta_max=5, nzeta_max=5, comm=comm, seed=seed
    )


# initialize_position_uniform_vol delegates to initialize_position_profile, so
# these two cover both copies of the sampling loop.
SAMPLERS = (sample_profile, sample_surf)


class TracingHelpersTests(unittest.TestCase):
    def test_seed_is_reproducible(self):
        for sample in SAMPLERS:
            with self.subTest(sample.__name__):
                points = sample(0)
                self.assertEqual(points.shape, (NPARTICLES, 3))
                np.testing.assert_array_equal(points, sample(0))

    def test_none_seed_preserves_rng_state(self):
        """seed=None must not reseed the caller's global numpy RNG."""
        for sample in SAMPLERS:
            with self.subTest(sample.__name__):
                np.random.seed(12345)
                sample(None)
                expected = np.random.uniform(0, 1, 5)

                np.random.seed(12345)
                sample(None)
                np.testing.assert_array_equal(expected, np.random.uniform(0, 1, 5))

    @unittest.skipIf(comm is None, "mpi4py not available")
    def test_all_ranks_get_the_same_distinct_sample(self):
        """
        The sample is drawn once on rank 0 and broadcast, so every rank must
        hold the same array of nparticles distinct positions. A sample that
        repeats positions across ranks still has the expected shape, so the
        distinctness is asserted on directly.
        """
        for sample in SAMPLERS:
            with self.subTest(sample.__name__):
                points = sample(0)
                self.assertEqual(len(np.unique(points, axis=0)), NPARTICLES)
                for other in comm.allgather(points):
                    np.testing.assert_array_equal(points, other)


def reactivity(s):
    """D-T reactivity profile of the fusion_distribution examples, T0 = 11.5 keV."""
    return (1 - s**5) ** 2 * sigmav(11.5 * (1 - s))


class SMarginalTests(unittest.TestCase):
    """
    The s-marginal of a sample drawn with weight J * profile(s) is
    profile(s) dV/ds, since integrating J over the angles gives dV/ds. On the
    VMEC half grid dV/ds = 4 pi^2 vp(s), so vp from the wout file is the
    reference. The n3are (ARIES-CS) equilibrium has dV/ds varying by about
    10% over s, which is what the test resolves.
    """

    NPARTICLES = 20000

    @classmethod
    def setUpClass(cls):
        # K is not needed for J = (G + iota I)/B^2 and dominates the build time.
        cls.field = BoozerRadialInterpolant(filename_mhd, 3, no_K=True, comm=comm)
        f = netcdf_file(filename_mhd_wout, mmap=False)
        vp = f.variables["vp"][()][1:]
        f.close()
        cls.vp = InterpolatedUnivariateSpline(cls.field.s_half_ext[1:-1], vp)

    def ks_pvalue(self, s, weight):
        """KS p-value of the sampled s against the marginal density weight(s)."""
        s_grid = np.linspace(0, 1, 2001)
        cdf = cumulative_trapezoid([weight(si) for si in s_grid], s_grid, initial=0)
        cdf /= cdf[-1]
        return kstest(s, lambda x: np.interp(x, s_grid, cdf)).pvalue

    def test_uniform_vol_s_marginal_is_dVds(self):
        s = initialize_position_uniform_vol(
            self.field, self.NPARTICLES, comm=comm, seed=0
        )[:, 0]
        self.assertGreater(self.ks_pvalue(s, self.vp), 0.01)
        # Without the Jacobian weight the marginal would be uniform in s.
        self.assertLess(self.ks_pvalue(s, lambda si: 1.0), 1e-3)

    def test_profile_s_marginal_is_profile_times_dVds(self):
        s = initialize_position_profile(
            self.field, self.NPARTICLES, reactivity, comm=comm, seed=0
        )[:, 0]
        self.assertGreater(
            self.ks_pvalue(s, lambda si: reactivity(si) * self.vp(si)), 0.01
        )


if __name__ == "__main__":
    unittest.main()
