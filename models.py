"""Sanitized generic baselines extracted from a reconstructed thesis module.

Not recovered paper source, not an exact reproduction of published experiments.
Only standard GRU, BiGRU and single-head temporal attention are included.
No data loading, fitted statistics, search protocol or model updating is included.
"""

import math
from numbers import Integral

import tensorflow as tf
from tensorflow.keras import layers, regularizers


@tf.keras.utils.register_keras_serializable(package="ForecastBaselines")
class TemporalAttention(layers.Layer):
    """Softmax scores over the time axis, followed by sequence reweighting.

    Input/output: (batch, lookback, channels); weights: (batch, lookback, 1).
    This is reweighting plus flattening in the model, not a weighted-sum head.
    """

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("TemporalAttention expects rank-3 input")
        self.score = layers.Dense(1, activation="tanh", name="time_score")
        self.score.build(input_shape)
        super().build(input_shape)

    def weights_for(self, inputs):
        return tf.nn.softmax(self.score(inputs), axis=1)

    def call(self, inputs):
        return inputs * self.weights_for(inputs)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def build_model(kind, lookback, n_features, n_outputs=1, *, units=None,
                dropout=0.2, l2=0.0, learning_rate=0.001,
                activation="tanh", seed=42):
    """Build and compile a direct multi-output regression baseline.

    n_outputs is an output count, not a lead time. The caller defines targets
    and their horizon order. BiGRU runs in both directions ONLY within the
    historical input window; no observation after its origin is necessary.
    units may be a positive integer or a nonempty list of positive integers.
    """
    if kind not in {"gru", "bigru", "attention"}:
        raise ValueError("kind must be gru, bigru or attention")
    lookback = _positive_integer(lookback, "lookback")
    n_features = _positive_integer(n_features, "n_features")
    n_outputs = _positive_integer(n_outputs, "n_outputs")
    if units is None:
        units = [128] if kind == "gru" else [64]
    elif isinstance(units, Integral):
        units = [units]
    try:
        units = [_positive_integer(value, "units") for value in units]
    except TypeError as exc:
        raise ValueError("units must be an integer or a sequence") from exc
    if not units:
        raise ValueError("units must not be empty")
    dropout, l2, learning_rate = map(float, (dropout, l2, learning_rate))
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("dropout must be finite and in [0, 1)")
    if not math.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be finite and nonnegative")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if activation not in {"tanh", "selu"}:
        raise ValueError("activation must be tanh or selu")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    tf.keras.utils.set_random_seed(int(seed))
    reg = regularizers.l2(l2) if l2 else None
    inp = layers.Input((lookback, n_features), name="historical_features")
    x = inp
    for index, count in enumerate(units):
        sequence = kind == "attention" or index < len(units) - 1
        recurrent = layers.GRU(
            count, return_sequences=sequence, activation=activation,
            recurrent_activation="sigmoid", recurrent_dropout=0.0,
            dropout=0.0, reset_after=True, kernel_regularizer=reg,
            recurrent_regularizer=reg,
        )
        if kind != "gru":
            recurrent = layers.Bidirectional(
                recurrent, name=f"bidirectional_{index}"
            )
        x = recurrent(x)
    if kind == "attention":
        x = TemporalAttention(name="temporal_attention")(x)
        x = layers.Flatten(name="weighted_sequence_flatten")(x)
    x = layers.Dropout(dropout, name="backbone_dropout")(x)
    if kind != "gru" and len(units) > 1:
        x = layers.Dense(128, activation="relu", kernel_regularizer=reg,
                         name="dense_hidden")(x)
    out = layers.Dense(n_outputs, activation="linear", name="forecast")(x)
    model = tf.keras.Model(inp, out, name=f"baseline_{kind}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate, clipnorm=1.0
        ), loss="mse", metrics=[tf.keras.metrics.MeanSquaredError(name="mse")]
    )
    return model
