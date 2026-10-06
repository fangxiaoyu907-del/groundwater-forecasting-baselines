"""Data-free, general-purpose forecasting baselines.

The centered ridge objective and metric formulas were extracted and sanitized
from an existing baseline experiment, then generalized to multiple outputs.
This module is not an implementation or reproduction of a paper's proposed
model. It includes no dataset, dataset schema, paper configuration, pretrained
weights, fitted preprocessing values, online adaptation, or staged models.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np


def _finite_array(value, name: str) -> np.ndarray:
    """Convert an array-like value without silently discarding imaginary parts."""
    try:
        array = np.asarray(value)
        if np.iscomplexobj(array):
            raise ValueError(f"{name} must contain real numbers")
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return array


def _matrix(value, name: str) -> np.ndarray:
    array = _finite_array(value, name)
    if array.ndim != 2 or min(array.shape) == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    return array


def _anchor(value) -> np.ndarray:
    array = _finite_array(value, "anchor")
    if array.ndim == 1:
        return array[:, None]
    if array.ndim == 2 and array.shape[1] == 1:
        return array
    raise ValueError("anchor must have shape (n_samples,) or (n_samples, 1)")


class RidgeDeltaRegressor:
    """Predict future level changes, then add the last observed level.

    Fit only on training ``X`` and training ``target_delta``, both two-dimensional.
    Every output column is a separate forecast horizon. Feature means, feature
    scales, and the output intercept are estimated only during ``fit``. The
    intercept is unpenalized; the ridge objective is mean squared error plus
    ``alpha * sum(coefficient**2)`` for each output. No state-export method is
    provided: fitted state may contain private training-derived information.
    """

    def __init__(self, alpha: float = 1.0):
        if (
            isinstance(alpha, (bool, np.bool_))
            or not isinstance(alpha, Real)
            or not np.isfinite(alpha)
            or alpha <= 0
        ):
            raise ValueError("alpha must be a finite positive real number")
        self.alpha = float(alpha)
        self._state = None

    def fit(self, X, target_delta) -> "RidgeDeltaRegressor":
        """Fit on training samples only, without validation or test input."""
        features = _matrix(X, "X")
        targets = _matrix(target_delta, "target_delta")
        if len(features) != len(targets):
            raise ValueError("X and target_delta must have equal sample counts")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                feature_mean = features.mean(axis=0)
                feature_scale = features.std(axis=0)
                feature_scale = np.where(feature_scale < 1e-12, 1.0, feature_scale)
                standardized = (features - feature_mean) / feature_scale
                target_mean = targets.mean(axis=0)
                sample_count = len(features)
                gram = standardized.T @ standardized / sample_count
                gram += self.alpha * np.eye(features.shape[1])
                cross = standardized.T @ (targets - target_mean) / sample_count
                coefficient = np.linalg.solve(gram, cross)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise ValueError("training values are not numerically solvable") from exc
        if not all(
            np.isfinite(array).all()
            for array in (feature_mean, feature_scale, target_mean, coefficient)
        ):
            raise ValueError("training produced nonfinite fitted values")
        self._state = (feature_mean, feature_scale, coefficient, target_mean)
        return self

    def predict(self, X, anchor) -> np.ndarray:
        """Return levels with shape ``(n_samples, n_outputs)``.

        ``anchor`` is the last observed level available at the forecast origin,
        not a future target. Prediction never fits or updates preprocessing.
        """
        if self._state is None:
            raise RuntimeError("fit must be called before predict")
        features = _matrix(X, "X")
        origins = _anchor(anchor)
        mean, scale, coefficient, intercept = self._state
        if features.shape[1] != len(mean):
            raise ValueError("X has a different feature count from training")
        if len(features) != len(origins):
            raise ValueError("X and anchor must have equal sample counts")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                predictions = ((features - mean) / scale) @ coefficient
                predictions = predictions + intercept + origins
        except FloatingPointError as exc:
            raise ValueError("prediction values exceed the numeric range") from exc
        if not np.isfinite(predictions).all():
            raise ValueError("prediction produced nonfinite values")
        return predictions


def persistence(anchor, n_outputs: int) -> np.ndarray:
    """Repeat the last observed level at every requested forecast horizon."""
    if (
        isinstance(n_outputs, (bool, np.bool_))
        or not isinstance(n_outputs, Integral)
        or n_outputs <= 0
    ):
        raise ValueError("n_outputs must be a positive integer")
    return np.repeat(_anchor(anchor), int(n_outputs), axis=1)


def regression_metrics(y, p, baseline=None) -> dict:
    """Compute pooled MAE/RMSE, variance-weighted R2, and MSE skill.

    Arrays must have matching nonempty one-dimensional or two-dimensional
    shapes. For multiple outputs, each output has its own mean for R2. R2 is
    ``None`` when all targets are constant. Skill is ``1 - MSE / baseline_MSE``
    and is ``None`` if no baseline was supplied or its MSE is zero. ``n`` counts
    samples, not scalar target cells. No metrics are silently computed after
    dropping invalid samples.
    """
    observed = _finite_array(y, "y")
    predicted = _finite_array(p, "p")
    if observed.ndim not in (1, 2) or predicted.shape != observed.shape:
        raise ValueError("y and p must have matching one-dimensional or two-dimensional shapes")
    reference = None
    if baseline is not None:
        reference = _finite_array(baseline, "baseline")
        if reference.shape != observed.shape:
            raise ValueError("baseline must have the same shape as y")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            error = observed - predicted
            mse = float(np.mean(error**2))
            variance = float(np.mean((observed - observed.mean(axis=0)) ** 2))
            r2 = float(1 - mse / variance) if variance > 0 else None
            skill = None
            if reference is not None:
                base_mse = float(np.mean((observed - reference) ** 2))
                if base_mse > 0:
                    skill = float(1 - mse / base_mse)
            metrics = {
                "n": len(observed),
                "n_outputs": observed.shape[1] if observed.ndim == 2 else 1,
                "MAE": float(np.mean(np.abs(error))),
                "RMSE": float(np.sqrt(mse)),
                "R2": r2,
                "Skill": skill,
            }
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("metric values exceed the numeric range") from exc
    if any(value is not None and not np.isfinite(value) for value in metrics.values()):
        raise ValueError("metrics produced nonfinite values")
    return metrics
