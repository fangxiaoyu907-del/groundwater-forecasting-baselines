"""Synthetic sphere-objective checks; no research data or model training."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gwo import grey_wolf_search


def sphere(position):
    return float(np.dot(position, position))


class GreyWolfSearchTests(unittest.TestCase):
    def test_same_seed_reproduces_all_evaluations(self):
        options = {"population": 5, "iterations": 8, "seed": 42}
        first = grey_wolf_search(sphere, [-5, -3], [5, 3], **options)
        second = grey_wolf_search(sphere, [-5, -3], [5, 3], **options)
        np.testing.assert_array_equal(first["best_position"], second["best_position"])
        self.assertEqual(first["best_score"], second["best_score"])
        self.assertEqual(len(first["history"]), len(second["history"]))
        for left, right in zip(first["history"], second["history"]):
            np.testing.assert_array_equal(left["position"], right["position"])
            self.assertEqual(left["score"], right["score"])

    def test_evaluation_count_bounds_and_history_identifiers(self):
        seen = []

        def counted_sphere(position):
            seen.append(position.copy())
            return sphere(position)

        lo, hi = np.array([-4.0, -2.0]), np.array([4.0, 2.0])
        result = grey_wolf_search(
            counted_sphere, lo, hi, population=4, iterations=5, seed=17
        )
        self.assertEqual(len(seen), 20)
        self.assertEqual(len(result["history"]), 20)
        for index, (entry, received) in enumerate(zip(result["history"], seen)):
            self.assertEqual(entry["iteration"], index // 4 + 1)
            self.assertEqual(entry["wolf"], index % 4 + 1)
            np.testing.assert_array_equal(entry["position"], received)
            self.assertTrue(np.isfinite(received).all())
            self.assertTrue((received >= lo).all())
            self.assertTrue((received <= hi).all())
            self.assertEqual(entry["score"], sphere(received))

    def test_result_is_historical_best_and_not_worse_than_initial_population(self):
        result = grey_wolf_search(sphere, [-5, -5], [5, 5])
        scores = [entry["score"] for entry in result["history"]]
        self.assertEqual(result["best_score"], min(scores))
        self.assertLessEqual(result["best_score"], min(scores[:5]))
        self.assertEqual(result["best_score"], sphere(result["best_position"]))
        best = min(result["history"], key=lambda entry: entry["score"])
        np.testing.assert_array_equal(result["best_position"], best["position"])
        self.assertFalse(np.shares_memory(result["best_position"], best["position"]))

    def test_single_iteration_returns_only_evaluated_initial_positions(self):
        result = grey_wolf_search(sphere, [-2], [2], population=3, iterations=1)
        self.assertEqual(len(result["history"]), 3)
        self.assertEqual(
            result["best_score"], min(entry["score"] for entry in result["history"])
        )

    def test_objective_mutation_cannot_change_internal_positions(self):
        received = []

        def mutating_sphere(position):
            original = position.copy()
            received.append(position)
            score = sphere(original)
            position[:] = 9999.0
            return score

        expected = grey_wolf_search(sphere, [-3, -3], [3, 3], seed=9)
        actual = grey_wolf_search(mutating_sphere, [-3, -3], [3, 3], seed=9)
        self.assertEqual(actual["best_score"], expected["best_score"])
        np.testing.assert_array_equal(actual["best_position"], expected["best_position"])
        for entry, reference, mutated in zip(
            actual["history"], expected["history"], received
        ):
            np.testing.assert_array_equal(entry["position"], reference["position"])
            self.assertEqual(entry["score"], reference["score"])
            self.assertFalse(np.shares_memory(entry["position"], mutated))

    def test_invalid_bounds_are_rejected(self):
        invalid = [
            ([], []),
            (-1, 1),
            ([[-1]], [[1]]),
            ([-1, -2], [1]),
            ([0], [0]),
            ([1], [0]),
            ([np.nan], [1]),
            ([-1], [np.inf]),
            ([-np.inf], [1]),
            (["not-a-number"], [1]),
            ([1j], [2]),
        ]
        for lower, upper in invalid:
            with self.subTest(lower=lower, upper=upper), self.assertRaises(ValueError):
                grey_wolf_search(sphere, lower, upper)

    def test_population_and_iterations_require_non_boolean_integers(self):
        for name, invalid_values in (
            ("population", [True, np.bool_(True), 2, 0, -1, 3.0, "3", None]),
            ("iterations", [True, np.bool_(True), 0, -1, 1.0, "1", None]),
        ):
            for value in invalid_values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    grey_wolf_search(sphere, [-1], [1], **{name: value})
        accepted = grey_wolf_search(
            sphere, [-1], [1], population=np.int64(3), iterations=np.int64(1)
        )
        self.assertEqual(len(accepted["history"]), 3)

    def test_objective_requires_a_finite_real_numeric_scalar(self):
        invalid_values = [
            np.nan, np.inf, -np.inf, [1.0], np.array([1.0]), 1 + 0j,
            "1.0", True, None,
        ]
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                grey_wolf_search(lambda _: value, [-1], [1])
        with self.assertRaises(ValueError):
            grey_wolf_search(None, [-1], [1])
        accepted = grey_wolf_search(
            lambda x: np.float64(sphere(x)), [-1], [1], iterations=1
        )
        self.assertTrue(np.isfinite(accepted["best_score"]))


if __name__ == "__main__":
    unittest.main()
