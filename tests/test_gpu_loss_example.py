"""Analytic checks for the GPU loss example's classifier inputs."""

from pathlib import Path
import runpy
import unittest

import numpy as np


classification_inputs = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "examples/gpu_dense_output/save_trajectories.py"
    )
)["classification_inputs"]


class LossExampleTests(unittest.TestCase):
    def test_mirrors_unwrap_both_angles_and_preserve_loss_endpoint(self):
        path = np.array(
            [[0, 0.5, 3.1, 6.2, 2], [1, 0.7, -3.1, 0.1, -2], [2, 1.01, -3, 0.2, 2]]
        )
        original = path.copy()
        loss = np.array([[2, -1, *path[-1, 1:]]])
        unwrapped, hits = classification_inputs(path, loss)
        np.testing.assert_array_equal(path, original)
        np.testing.assert_allclose(unwrapped[:, 2:4], np.unwrap(path[:, 2:4], axis=0))
        np.testing.assert_allclose(hits[:, :2], [[0.5, 0], [1.5, 0], [2, -1]])
        np.testing.assert_allclose(
            hits[:2, 2:], (unwrapped[:-1, 1:] + unwrapped[1:, 1:]) / 2
        )
        np.testing.assert_array_equal(hits[-1, 2:], unwrapped[-1, 1:])

    def test_exact_sample_mirror_counted_once_without_launch(self):
        path = np.column_stack(
            (np.arange(5), np.full(5, 0.5), np.zeros((5, 2)), [0, 1, 0, -1, 0])
        )
        _, hits = classification_inputs(path, [])
        np.testing.assert_array_equal(hits[:, 0], [2, 4])
        np.testing.assert_array_equal(hits[:, 1], 0)
        np.testing.assert_array_equal(hits[:, -1], 0)

    def test_interpolated_mirror_converges_with_cadence(self):
        exact = np.sqrt(0.3)
        errors = []
        for step in [0.2, 0.1, 0.05]:
            t = np.arange(0, 1, step)
            path = np.column_stack(
                (t, np.full(len(t), 0.5), np.zeros((len(t), 2)), t**2 - 0.3)
            )
            _, hits = classification_inputs(path, [])
            self.assertEqual(hits.shape, (1, 6))
            errors.append(abs(hits[0, 0] - exact))
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])

    def test_passing_and_short_paths(self):
        path = np.array([[0, 0.5, 0, 0, 1], [1, 0.6, 0.1, 0.2, 1]])
        for short in [path, path[:1]]:
            _, hits = classification_inputs(short, [])
            self.assertEqual(hits.shape, (0, 6))
        for invalid in [path[::-1], path * np.nan, path[:, :4], np.empty((0, 5))]:
            with self.assertRaises(ValueError):
                classification_inputs(invalid, [])


if __name__ == "__main__":
    unittest.main()
