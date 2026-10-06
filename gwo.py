"""A NumPy-only grey-wolf search with historical best-so-far leaders.

The generic initialization, archive ordering, three-leader update and clipping
mathematics were extracted from reconstructed experiment code. This is not
recovered source code from the first paper, or a claim to implement the canonical
original-paper GWO exactly. Leaders are the best three evaluated entries from
the whole historical archive, rather than just the current population.

No real data, experimental configuration, training protocol or unpublished
model is included. The function performs no file or network operations.
"""

from __future__ import annotations

import numpy as np


def grey_wolf_search(
    objective, lower, upper, population=5, iterations=8, seed=42
):
    """Minimize a finite real scalar objective inside one-dimensional bounds.

    ``lower`` and ``upper`` must be equally shaped, nonempty one-dimensional
    finite numeric arrays, with each lower bound strictly below its upper bound.
    ``population`` and ``iterations`` must be non-boolean integers, at least
    three and one respectively.

    The objective receives a fresh position copy on every call. Each of the
    ``population * iterations`` evaluations is recorded in ``history`` as a
    dictionary with one-based ``iteration`` and ``wolf`` identifiers, a copied
    NumPy ``position``, and a floating-point ``score``. The return dictionary
    contains ``best_position``, ``best_score`` and ``history``. Its best position
    is an evaluated historical position, not an unevaluated final proposal.
    """
    if not callable(objective):
        raise ValueError("objective must be callable")
    for name, value, minimum in (
        ("population", population, 3),
        ("iterations", iterations, 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be a non-boolean integer >= {minimum}")
    population, iterations = int(population), int(iterations)

    try:
        lo = np.array(lower, dtype=float, copy=True)
        hi = np.array(upper, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bounds must be finite numeric arrays") from exc
    if (
        lo.ndim != 1
        or hi.ndim != 1
        or lo.shape != hi.shape
        or lo.size == 0
        or not np.isfinite(lo).all()
        or not np.isfinite(hi).all()
        or not (lo < hi).all()
    ):
        raise ValueError(
            "bounds must be nonempty finite 1-D arrays of equal shape, "
            "with lower < upper elementwise"
        )

    rng = np.random.default_rng(seed)
    positions = rng.uniform(lo, hi, size=(population, lo.size))
    history = []

    for iteration in range(iterations):
        for wolf, position in enumerate(positions):
            value = np.asarray(objective(position.copy()))
            if (
                value.ndim != 0
                or not np.issubdtype(value.dtype, np.number)
                or np.issubdtype(value.dtype, np.complexfloating)
            ):
                raise ValueError("objective must return a finite real numeric scalar")
            score = float(value)
            if not np.isfinite(score):
                raise ValueError("objective must return a finite real numeric scalar")
            history.append(
                {
                    "iteration": iteration + 1,
                    "wolf": wolf + 1,
                    "position": position.copy(),
                    "score": score,
                }
            )

        # The final population has already been evaluated. Do not manufacture
        # unevaluated final proposals or return them as an optimization result.
        if iteration == iterations - 1:
            break
        leaders = sorted(history, key=lambda entry: entry["score"])[:3]
        a = 2.0 * (1.0 - iteration / iterations)
        next_positions = []
        for current in positions:
            estimates = []
            for leader in leaders:
                r1, r2 = rng.random(lo.size), rng.random(lo.size)
                leader_position = leader["position"]
                estimates.append(
                    leader_position
                    - (2.0 * a * r1 - a)
                    * np.abs(2.0 * r2 * leader_position - current)
                )
            next_positions.append(np.clip(np.mean(estimates, axis=0), lo, hi))
        positions = np.asarray(next_positions)

    best = min(history, key=lambda entry: entry["score"])
    return {
        "best_position": best["position"].copy(),
        "best_score": best["score"],
        "history": history,
    }
