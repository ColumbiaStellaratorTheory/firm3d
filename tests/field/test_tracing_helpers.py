"""
Tests for the position initialization helpers in tracing_helpers, covering the
two behaviors changed by the seed/comm fix: seed=None must leave the caller's
global numpy RNG alone, and a seeded sample must be identical on every rank.
The rank assertions only have force under more than one rank, e.g.
mpirun -n 2 python -m pytest tests/field/test_tracing_helpers.py
"""

import unittest

import numpy as np

from firm3d.field.boozermagneticfield import BoozerAnalytic
from firm3d.field.tracing_helpers import (
    initialize_position_profile,
    initialize_position_uniform_surf,
)

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


if __name__ == "__main__":
    unittest.main()
