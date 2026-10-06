import unittest

import numpy as np

from windows import Standardizer, chronological_masks, make_windows


class WindowTests(unittest.TestCase):
    def test_history_targets_and_horizon_order(self):
        y = np.arange(30, dtype=float)
        batch = make_windows(y[:, None], y, 4, [3, 1])
        np.testing.assert_array_equal(batch.X[0, :, 0], [0, 1, 2, 3])
        np.testing.assert_array_equal(batch.y[0], [6, 4])
        np.testing.assert_array_equal(batch.target_indices[:, 0], batch.origin_indices + 3)
        np.testing.assert_array_equal(batch.anchor, y[batch.origin_indices])
        for index, origin in enumerate(batch.origin_indices):
            self.assertEqual(batch.X[index, -1, 0], origin)
            self.assertTrue((batch.target_indices[index] > origin).all())

    def test_split_drops_cross_boundary_labels(self):
        y = np.arange(40, dtype=float)
        batch = make_windows(y[:, None], y, 4, [1, 3])
        tr, va, te = chronological_masks(batch.target_indices, 20, 30)
        self.assertTrue((batch.target_indices[tr] < 20).all())
        self.assertTrue(((batch.target_indices[va] >= 20) & (batch.target_indices[va] < 30)).all())
        self.assertTrue((batch.target_indices[te] >= 30).all())
        self.assertTrue((~(tr | va | te)).any())
        self.assertFalse((tr & va | va & te | tr & te).any())

    def test_invalid_windows(self):
        y = np.arange(20, dtype=float)
        for lookback, horizons in [(0, [1]), (True, [1]), (2, [0]), (2, [1, 1]), (2, []), (20, [1])]:
            with self.subTest(lookback=lookback, horizons=horizons):
                with self.assertRaises(ValueError):
                    make_windows(y[:, None], y, lookback, horizons)
        y[3] = np.nan
        with self.assertRaises(ValueError):
            make_windows(y[:, None], y, 3, [1])

    def test_invalid_split(self):
        for indices, tr, va in [(np.array([[1.5]]), 5, 9), (np.array([[1, 2]]), 5, 5), (np.array([[1, 2]]), 5, 9)]:
            with self.assertRaises(ValueError):
                chronological_masks(indices, tr, va)

    def test_standardizer_training_only_and_constant_column(self):
        train = np.array([[1., 7.], [3., 7.]])
        scaler = Standardizer().fit(train)
        np.testing.assert_array_equal(scaler.mean_, [2., 7.])
        np.testing.assert_array_equal(scaler.scale_, [1., 1.])
        future = np.array([[101., 7.]])
        np.testing.assert_array_equal(scaler.transform(future), [[99., 0.]])
        np.testing.assert_array_equal(scaler.mean_, [2., 7.])
        np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(future)), future)
        with self.assertRaises(RuntimeError):
            Standardizer().transform(train)
        with self.assertRaises(ValueError):
            scaler.transform(np.ones((2, 3)))


if __name__ == "__main__":
    unittest.main()
