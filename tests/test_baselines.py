"""Tests use only small in-memory synthetic arrays; no files are loaded."""

import unittest

import numpy as np

from baselines import RidgeDeltaRegressor, persistence, regression_metrics


class RidgeDeltaTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-2.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [2.0, 1.0]])
        self.delta = np.column_stack((2 * self.X[:, 0] + 0.5, -self.X[:, 0] + self.X[:, 1] - 0.25))
        self.anchor = np.linspace(0.0, 1.0, len(self.X))

    def test_multioutput_fit_predict(self):
        model = RidgeDeltaRegressor(alpha=1e-8)
        self.assertIs(model.fit(self.X, self.delta), model)
        prediction = model.predict(self.X, self.anchor)
        self.assertEqual(prediction.shape, (5, 2))
        np.testing.assert_allclose(prediction, self.delta + self.anchor[:, None], atol=1e-6)

    def test_single_output_stays_two_dimensional(self):
        model = RidgeDeltaRegressor().fit(self.X, self.delta[:, :1])
        self.assertEqual(model.predict(self.X, self.anchor[:, None]).shape, (5, 1))

    def test_constant_and_near_constant_feature_columns(self):
        X = np.column_stack((self.X[:, 0], np.ones(5), 1 + np.arange(5) * 1e-14))
        delta = (2 * X[:, 0])[:, None]
        model = RidgeDeltaRegressor(alpha=1e-8).fit(X, delta)
        prediction = model.predict(X, np.zeros(5))
        self.assertTrue(np.isfinite(prediction).all())
        np.testing.assert_allclose(prediction, delta, atol=1e-6)

    def test_unpenalized_output_intercept(self):
        delta = np.tile([2.0, -3.0], (5, 1))
        model = RidgeDeltaRegressor(alpha=2.0).fit(self.X, delta)
        np.testing.assert_allclose(model.predict(self.X, np.zeros(5)), delta)

    def test_one_training_sample(self):
        model = RidgeDeltaRegressor().fit([[0.0, 2.0]], [[1.0, -1.0]])
        np.testing.assert_allclose(model.predict([[5.0, 8.0]], [2.0]), [[3.0, 1.0]])

    def test_predict_uses_training_scaling_without_refitting(self):
        model = RidgeDeltaRegressor(alpha=0.1).fit(self.X, self.delta)
        sample = np.array([[8.0, -2.0]])
        alone = model.predict(sample, [1.0])
        in_batch = model.predict(np.vstack((sample, [[100.0, 200.0]])), [1.0, 0.0])
        np.testing.assert_allclose(alone[0], in_batch[0])

    def test_fit_does_not_mutate_inputs(self):
        X, delta = self.X.copy(), self.delta.copy()
        RidgeDeltaRegressor().fit(self.X, self.delta)
        np.testing.assert_array_equal(self.X, X)
        np.testing.assert_array_equal(self.delta, delta)

    def test_positive_finite_alpha_required(self):
        for alpha in (0, -1, np.nan, np.inf, True, "1", 1j, None):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                RidgeDeltaRegressor(alpha)

    def test_predict_before_fit(self):
        with self.assertRaises(RuntimeError):
            RidgeDeltaRegressor().predict(self.X, self.anchor)

    def test_fit_rejects_bad_shapes(self):
        pairs = ((self.X[0], self.delta), (self.X, self.delta[:, 0]), (self.X, self.delta[:-1]), (np.empty((0, 2)), np.empty((0, 1))), (np.empty((5, 0)), self.delta))
        for X, delta in pairs:
            with self.subTest(X_shape=np.shape(X), delta_shape=np.shape(delta)), self.assertRaises(ValueError):
                RidgeDeltaRegressor().fit(X, delta)

    def test_fit_rejects_nonfinite_and_complex_values(self):
        for bad in (np.nan, np.inf, -np.inf):
            X, delta = self.X.copy(), self.delta.copy()
            X[0, 0] = bad
            delta[0, 0] = bad
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    RidgeDeltaRegressor().fit(X, self.delta)
                with self.assertRaises(ValueError):
                    RidgeDeltaRegressor().fit(self.X, delta)
        with self.assertRaises(ValueError):
            RidgeDeltaRegressor().fit(self.X.astype(complex) + 1j, self.delta)

    def test_predict_rejects_invalid_shapes(self):
        model = RidgeDeltaRegressor().fit(self.X, self.delta)
        pairs = ((self.X[:, :1], self.anchor), (self.X, self.anchor[:-1]), (self.X, np.ones((5, 2))), (self.X[0], [1.0]), (np.empty((0, 2)), []))
        for X, anchor in pairs:
            with self.subTest(X_shape=np.shape(X), anchor_shape=np.shape(anchor)), self.assertRaises(ValueError):
                model.predict(X, anchor)

    def test_predict_rejects_nonfinite(self):
        model = RidgeDeltaRegressor().fit(self.X, self.delta)
        X = self.X.copy()
        X[0, 0] = np.nan
        with self.assertRaises(ValueError):
            model.predict(X, self.anchor)
        with self.assertRaises(ValueError):
            model.predict(self.X, [np.inf] * 5)


class PersistenceTests(unittest.TestCase):
    def test_multiple_horizons(self):
        np.testing.assert_array_equal(persistence([1.0, 2.0], 3), [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    def test_column_anchor(self):
        np.testing.assert_array_equal(persistence([[1.0], [2.0]], 1), [[1.0], [2.0]])

    def test_invalid_output_counts(self):
        for count in (0, -1, 1.5, True, np.inf, "2"):
            with self.subTest(count=count), self.assertRaises(ValueError):
                persistence([1.0], count)

    def test_invalid_anchor(self):
        for anchor in ([], 1.0, [[1.0, 2.0]], [np.nan], [np.inf], [1j]):
            with self.subTest(anchor=anchor), self.assertRaises(ValueError):
                persistence(anchor, 2)


class MetricsTests(unittest.TestCase):
    def test_known_metrics_and_mse_skill(self):
        result = regression_metrics([1.0, 2.0, 3.0], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0])
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["n_outputs"], 1)
        self.assertAlmostEqual(result["MAE"], 2 / 3)
        self.assertAlmostEqual(result["RMSE"], np.sqrt(2 / 3))
        self.assertAlmostEqual(result["R2"], 0.0)
        self.assertAlmostEqual(result["Skill"], 6 / 7)

    def test_multioutput_r2_centers_each_output(self):
        y = np.array([[0.0, 100.0], [1.0, 101.0]])
        result = regression_metrics(y, y + 1)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["n_outputs"], 2)
        self.assertAlmostEqual(result["R2"], -3.0)
        self.assertIsNone(result["Skill"])

    def test_constant_targets_and_perfect_baseline_return_none(self):
        result = regression_metrics([[1.0, 4.0], [1.0, 4.0]], [[2.0, 3.0], [2.0, 3.0]], [[1.0, 4.0], [1.0, 4.0]])
        self.assertIsNone(result["R2"])
        self.assertIsNone(result["Skill"])

    def test_perfect_prediction(self):
        result = regression_metrics([1.0, 2.0], [1.0, 2.0], [0.0, 0.0])
        self.assertEqual(result["MAE"], 0.0)
        self.assertEqual(result["RMSE"], 0.0)
        self.assertEqual(result["R2"], 1.0)
        self.assertEqual(result["Skill"], 1.0)

    def test_invalid_shapes(self):
        inputs = (([], [], None), (1.0, 1.0, None), ([1.0], [[1.0]], None), ([1.0, 2.0], [1.0], None), ([1.0, 2.0], [1.0, 2.0], [[1.0, 2.0]]), (np.ones((1, 1, 1)), np.ones((1, 1, 1)), None))
        for y, p, baseline in inputs:
            with self.subTest(y_shape=np.shape(y), p_shape=np.shape(p)), self.assertRaises(ValueError):
                regression_metrics(y, p, baseline)

    def test_nonfinite_inputs_rejected_instead_of_dropped(self):
        for y, p, baseline in (([np.nan], [1.0], None), ([1.0], [np.inf], None), ([1.0], [1.0], [np.nan]), ([1j], [1.0], None)):
            with self.subTest(y=y, p=p), self.assertRaises(ValueError):
                regression_metrics(y, p, baseline)


if __name__ == "__main__":
    unittest.main()
