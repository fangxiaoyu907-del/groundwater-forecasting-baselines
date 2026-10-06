"""Synthetic-only demo; writes no datasets, weights or fitted parameters."""

import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")

import numpy as np

from baselines import RidgeDeltaRegressor, persistence, regression_metrics
from gwo import grey_wolf_search
from windows import Standardizer, chronological_masks, make_windows


def synthetic_series(seed=42):
    """Illustrative unitless signal, unrelated to any real monitoring site."""
    rng = np.random.default_rng(seed)
    t = np.arange(360)
    driver = np.sin(t / 15.)
    level = 0.6 * np.sin(t / 21.) + 0.3 * driver + rng.normal(0, 0.03, len(t))
    return np.column_stack([level, driver]), level


def run_demo(neural=False, epochs=3):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    features, target = synthetic_series()
    horizons = [1, 3]
    batch = make_windows(features, target, lookback=8, horizons=horizons)
    train, val, test = chronological_masks(batch.target_indices, 220, 290)
    flat = batch.X.reshape(len(batch.X), -1)
    reference = persistence(batch.anchor[test], len(horizons))
    delta = batch.y - batch.anchor[:, None]
    candidates = []
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        fitted = RidgeDeltaRegressor(alpha).fit(flat[train], delta[train])
        pred = fitted.predict(flat[val], batch.anchor[val])
        candidates.append((float(np.mean((pred - batch.y[val]) ** 2)), alpha, fitted))
    _, selected_alpha, ridge = min(candidates, key=lambda item: item[0])
    predictions = {
        "Persistence": reference,
        "RidgeDelta": ridge.predict(flat[test], batch.anchor[test]),
    }
    if neural:
        from models import build_model
        # Neither validation nor test observations enter preprocessing fits.
        xs = Standardizer().fit(batch.X[train])
        ys = Standardizer().fit(batch.y[train])
        x = xs.transform(batch.X).astype(np.float32)
        y = ys.transform(batch.y).astype(np.float32)
        for kind in ["gru", "bigru", "attention"]:
            model = build_model(kind, 8, 2, len(horizons), units=8, dropout=0, seed=42)
            best_score, best_weights = float("inf"), None
            for _ in range(epochs):
                # Small CPU demonstration without background data-loader threads.
                xt, yt = x[train], y[train]
                for start in range(0, len(xt), 32):
                    model.train_on_batch(xt[start:start + 32], yt[start:start + 32])
                pred = model(x[val], training=False).numpy()
                score = float(np.mean((pred - y[val]) ** 2))
                if not np.isfinite(score):
                    raise RuntimeError("nonfinite neural validation score")
                if score < best_score:
                    best_score = score
                    best_weights = model.get_weights()
            model.set_weights(best_weights)
            predictions[kind] = ys.inverse_transform(model(x[test], training=False).numpy())
    # GWO is exercised on a mathematical objective, not a private model search.
    search = grey_wolf_search(lambda v: float(v @ v), [-2., -2.], [2., 2.],
                             population=5, iterations=4, seed=42)
    return {
        "data": "in-memory synthetic unitless signal only; not research evidence",
        "horizons_in_steps": horizons,
        "split_counts": {"train": int(train.sum()), "validation": int(val.sum()),
                         "test": int(test.sum()), "boundary_rows_dropped": int((~(train | val | test)).sum())},
        "ridge_alpha_selected_on_validation": selected_alpha,
        "gwo_synthetic_objective": {"evaluations": len(search["history"]), "best_score": search["best_score"]},
        "test_metrics_by_horizon": {
            name: {
                str(h): regression_metrics(batch.y[test, j], p[:, j], reference[:, j])
                for j, h in enumerate(horizons)
            } for name, p in predictions.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", action="store_true", help="also run TensorFlow baselines")
    parser.add_argument("--epochs", type=int, default=3, help="small synthetic training budget")
    args = parser.parse_args()
    print(json.dumps(run_demo(args.neural, args.epochs), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
