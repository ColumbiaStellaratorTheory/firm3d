import unittest

import numpy as np

from firm3d.trajectory_helpers import compute_loss_fraction


def trajectory(final_time):
    """A minimal (t, s, theta, zeta, vpar) trajectory ending at final_time."""
    return np.array(
        [
            [0.0, 0.5, 0.0, 0.0, 1e6],
            [final_time, 0.6, 1.0, 2.0, 1e6],
        ]
    )


class ComputeLossFractionTests(unittest.TestCase):
    def test_float32_survivor_is_not_lost(self):
        # a single-precision GPU run records a survivor's final time as
        # float32(tmax), one float32 ulp below the float64 tmax; the
        # classification margin must tolerate that rounding
        tmax = 1e-2
        res_tys = [trajectory(np.float64(np.float32(tmax)))]
        self.assertLess(res_tys[0][-1, 0], tmax)
        _, loss_frac = compute_loss_fraction(res_tys, tmax=tmax)
        self.assertEqual(loss_frac[-1], 0.0)

    def test_float64_survivor_is_not_lost(self):
        tmax = 1e-2
        _, loss_frac = compute_loss_fraction([trajectory(tmax)], tmax=tmax)
        self.assertEqual(loss_frac[-1], 0.0)

    def test_lost_particle_still_counts(self):
        tmax = 1e-2
        time_lost = 3.3e-4
        res_tys = [trajectory(time_lost), trajectory(tmax)]
        times, loss_frac = compute_loss_fraction(res_tys, tmax=tmax)
        self.assertEqual(loss_frac[-1], 0.5)
        np.testing.assert_array_equal(loss_frac[times < time_lost], 0.0)
        np.testing.assert_array_equal(loss_frac[times > 3.4e-4], 0.5)

    def test_loss_just_under_a_sample_time_counts_at_that_sample(self):
        # the float32 tolerance applies to the lost/survived decision at
        # tmax only, never to when along the curve a loss is counted
        tmax = 1e-2
        times, _ = compute_loss_fraction([trajectory(tmax)], tmax=tmax)
        sample = times[500]
        res_tys = [trajectory(sample * (1 - 1e-7)), trajectory(tmax)]
        _, loss_frac = compute_loss_fraction(res_tys, tmax=tmax)
        self.assertEqual(loss_frac[500], 0.5)

    def test_loss_just_before_tmax_counts(self):
        tmax = 1e-2
        res_tys = [trajectory(tmax * (1 - 1e-5)), trajectory(tmax)]
        _, loss_frac = compute_loss_fraction(res_tys, tmax=tmax)
        self.assertEqual(loss_frac[-1], 0.5)
