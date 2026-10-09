"""Independent analytic checks for the saved-trajectory Poincare example."""

from pathlib import Path
import runpy
import unittest

import numpy as np


crossings = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "examples/gpu_dense_output/plot_poincare.py"
    )
)["section_crossings"]
PERIOD = 2 * np.pi


def linear_path(times, rate):
    theta = 3.05 + 0.03 * times
    zeta = 0.25 + rate * times
    return np.column_stack(
        (
            times,
            0.3 + 0.005 * times,
            (theta + np.pi) % PERIOD - np.pi,
            zeta % PERIOD,
            1e6 + 100 * times,
        )
    )


class PoincareExampleTests(unittest.TestCase):
    def test_wrapped_angles_and_directions(self):
        times = np.arange(0, 25, 0.13)
        for rate in [PERIOD / 7, -PERIOD / 7]:
            for plane in [0, 0.5, PERIOD + 0.5]:
                with self.subTest(rate=rate, plane=plane):
                    path = linear_path(times, rate)
                    original = path.copy()
                    direction = "positive" if rate > 0 else "negative"
                    actual = crossings(path, plane, direction)
                    expected_times = np.sort(
                        ((plane % PERIOD + PERIOD * np.arange(-10, 11)) - 0.25) / rate
                    )
                    expected_times = expected_times[
                        (expected_times > 0) & (expected_times <= times[-1])
                    ]
                    np.testing.assert_allclose(actual[:, 0], expected_times, atol=1e-13)
                    np.testing.assert_allclose(
                        actual[:, 1], 0.3 + 0.005 * expected_times
                    )
                    np.testing.assert_allclose(
                        actual[:, 2], 3.05 + 0.03 * expected_times
                    )
                    np.testing.assert_allclose(actual[:, 3], plane % PERIOD, atol=1e-13)
                    np.testing.assert_allclose(actual[:, 4], 1e6 + 100 * expected_times)
                    np.testing.assert_allclose(actual, crossings(path, plane, "both"))
                    other = "negative" if rate > 0 else "positive"
                    self.assertEqual(crossings(path, plane, other).shape, (0, 5))
                    np.testing.assert_array_equal(path, original)

    def test_section_samples_counted_once_without_launch(self):
        times = np.arange(25, dtype=float)
        path = np.column_stack(
            (
                times,
                np.full(25, 0.5),
                np.zeros(25),
                times * PERIOD / 8 % PERIOD,
                np.ones(25),
            )
        )
        np.testing.assert_allclose(crossings(path)[:, 0], [8, 16, 24])
        path[:, 3] = (-times * PERIOD / 8) % PERIOD
        np.testing.assert_allclose(
            crossings(path, direction="negative")[:, 0], [8, 16, 24]
        )

    def test_quadratic_section_converges_under_halved_cadence(self):
        exact_time = PERIOD - 6.1
        errors = []
        for step in [0.2, 0.1, 0.05]:
            times = np.arange(0, 1, step)
            path = np.column_stack(
                (
                    times,
                    0.3 + 0.1 * times**2,
                    np.zeros(len(times)),
                    (6.1 + times) % PERIOD,
                    np.ones(len(times)),
                )
            )
            hit = crossings(path)
            self.assertEqual(hit.shape, (1, 5))
            self.assertAlmostEqual(hit[0, 0], exact_time)
            error = abs(hit[0, 1] - (0.3 + 0.1 * exact_time**2))
            self.assertLessEqual(error, 0.1 * step**2 / 4 + 1e-14)
            errors.append(error)
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])

    def test_short_paths_and_outside_boundary(self):
        self.assertEqual(crossings(np.empty((0, 5))).shape, (0, 5))
        self.assertEqual(crossings(np.array([[0, 0.5, 0, 0.1, 1e6]])).shape, (0, 5))
        path = linear_path(np.arange(0, 25, 0.13), PERIOD / 7)
        path[:, 1] = 1.1
        self.assertEqual(crossings(path).shape, (0, 5))


if __name__ == "__main__":
    unittest.main()
