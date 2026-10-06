import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
import numpy as np

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    if exc.name != "tensorflow":
        raise
    tf = None


@unittest.skipIf(tf is None, "install requirements-neural.txt for neural tests")
class NeuralTests(unittest.TestCase):
    def setUp(self):
        from models import TemporalAttention, build_model
        self.build = build_model
        self.attention = TemporalAttention
        rng = np.random.default_rng(17)
        self.x = rng.normal(size=(6, 5, 2)).astype(np.float32)
        self.y = rng.normal(size=(6, 2)).astype(np.float32)

    def tearDown(self):
        tf.keras.backend.clear_session()

    def test_all_baselines_forward_and_train(self):
        for kind in ("gru", "bigru", "attention"):
            with self.subTest(kind=kind):
                model = self.build(kind, 5, 2, 2, units=4, dropout=0)
                self.assertEqual(model(self.x).shape, (6, 2))
                loss = np.asarray(model.train_on_batch(self.x, self.y))
                self.assertTrue(np.isfinite(loss).all())

    def test_stacked_baselines(self):
        for kind in ("gru", "bigru", "attention"):
            model = self.build(kind, 5, 2, 2, units=[4, 3], dropout=0)
            self.assertEqual(model(self.x).shape, (6, 2))

    def test_attention_normalizes_time_and_has_gradients(self):
        layer = self.attention()
        x = tf.constant(self.x)
        with tf.GradientTape() as tape:
            weighted = layer(x)
            loss = tf.reduce_sum(weighted * tf.reverse(x, axis=[1]))
        weights = layer.weights_for(x).numpy()
        self.assertEqual(weights.shape, (6, 5, 1))
        np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-6)
        gradients = tape.gradient(loss, layer.trainable_weights)
        self.assertTrue(all(g is not None for g in gradients))
        self.assertTrue(all(np.isfinite(g.numpy()).all() for g in gradients))
        self.assertGreater(float(tf.linalg.global_norm(gradients)), 0.)

    def test_attention_save_load_round_trip(self):
        model = self.build("attention", 5, 2, 2, units=4, dropout=0)
        model.train_on_batch(self.x, self.y)
        before = model(self.x, training=False).numpy()
        with tempfile.TemporaryDirectory(prefix="forecast-baseline-test-") as folder:
            path = Path(folder) / "synthetic-test.keras"
            model.save(path)
            restored = tf.keras.models.load_model(path)
            np.testing.assert_allclose(restored(self.x).numpy(), before, atol=1e-6)

    def test_invalid_configuration(self):
        cases = [{"kind": "unknown"}, {"lookback": 0}, {"units": []},
                 {"units": [4, 0]}, {"units": True}, {"n_outputs": 1.5},
                 {"dropout": float("nan")}, {"l2": -1},
                 {"learning_rate": float("inf")}, {"seed": -1}]
        for patch in cases:
            config = {"kind": "gru", "lookback": 5, "n_features": 2, "n_outputs": 2, "units": 4}
            config.update(patch)
            with self.subTest(patch=patch):
                with self.assertRaises(ValueError):
                    self.build(**config)


if __name__ == "__main__":
    unittest.main()
