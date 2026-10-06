"""New generic, data-independent helpers for the public baseline demo."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class WindowBatch:
    X: np.ndarray
    y: np.ndarray
    anchor: np.ndarray
    origin_indices: np.ndarray
    target_indices: np.ndarray


def make_windows(features, target, lookback, horizons):
    """Use history ending at origin t to predict target[t + h], h > 0.

    Assumes chronological, equally spaced observations, supplied by the caller.
    Does not sort, read files, interpolate, impute or normalize any observations.
    """
    x, y = np.asarray(features, dtype=float), np.asarray(target, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or x.shape[1] < 1:
        raise ValueError("features must be (time, features), target (time,)")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("observations must be finite")
    lookback = _integer(lookback, "lookback")
    horizons = tuple(_integer(h, "horizon") for h in horizons)
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be nonempty and unique")
    if len(y) < lookback + max(horizons):
        raise ValueError("not enough observations for a complete window")
    origins = np.arange(lookback - 1, len(y) - max(horizons))
    indices = origins[:, None] + np.asarray(horizons)[None, :]
    history = origins[:, None] - np.arange(lookback - 1, -1, -1)[None, :]
    return WindowBatch(x[history], y[indices], y[origins], origins, indices)


def chronological_masks(target_indices, train_stop, validation_stop):
    """Split on target indices, not randomly on overlapping windows.

    A row is retained only if ALL its future labels belong to the same split.
    Rows crossing a split boundary are discarded rather than leaking labels.
    Boundaries are exclusive: train < train_stop, validation < validation_stop.
    """
    indices = np.asarray(target_indices)
    if indices.ndim != 2 or not indices.size or indices.shape[1] < 1:
        raise ValueError("target_indices must be a nonempty 2-D array")
    if not np.issubdtype(indices.dtype, np.integer) or (indices < 0).any():
        raise ValueError("target_indices must be nonnegative integers")
    train_stop = _integer(train_stop, "train_stop")
    validation_stop = _integer(validation_stop, "validation_stop")
    if validation_stop <= train_stop:
        raise ValueError("validation_stop must be after train_stop")
    train = np.all(indices < train_stop, axis=1)
    validation = np.all((indices >= train_stop) & (indices < validation_stop), axis=1)
    test = np.all(indices >= validation_stop, axis=1)
    if not all(mask.any() for mask in (train, validation, test)):
        raise ValueError("each chronological split must contain complete targets")
    return train, validation, test


class Standardizer:
    """Fit feature-wise means/scales on caller-supplied training values only."""

    def fit(self, values):
        x = self._values(values)
        self.mean_ = x.mean(axis=tuple(range(x.ndim - 1)))
        self.scale_ = x.std(axis=tuple(range(x.ndim - 1)))
        self.scale_ = np.where(self.scale_ < 1e-12, 1.0, self.scale_)
        return self

    @staticmethod
    def _values(values):
        x = np.asarray(values, dtype=float)
        if x.ndim not in (2, 3) or not x.size or not np.isfinite(x).all():
            raise ValueError("values must be a nonempty finite 2-D or 3-D array")
        return x

    def transform(self, values):
        x = self._values(values)
        if not hasattr(self, "mean_"):
            raise RuntimeError("call fit with training values first")
        if x.shape[-1] != len(self.mean_):
            raise ValueError("feature count differs from training values")
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, values):
        x = self._values(values)
        if not hasattr(self, "mean_"):
            raise RuntimeError("call fit with training values first")
        if x.shape[-1] != len(self.mean_):
            raise ValueError("feature count differs from training values")
        return x * self.scale_ + self.mean_
