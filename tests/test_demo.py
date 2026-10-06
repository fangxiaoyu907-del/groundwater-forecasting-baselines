import json
import unittest

from run_demo import run_demo


class DemoTests(unittest.TestCase):
    def test_numpy_demo_runs_and_is_json_serializable(self):
        report = run_demo()
        self.assertEqual(report["horizons_in_steps"], [1, 3])
        self.assertEqual(report["gwo_synthetic_objective"]["evaluations"], 20)
        self.assertEqual(set(report["test_metrics_by_horizon"]), {"Persistence", "RidgeDelta"})
        self.assertGreater(report["split_counts"]["boundary_rows_dropped"], 0)
        json.dumps(report, allow_nan=False)

    def test_invalid_epoch_budget(self):
        with self.assertRaises(ValueError):
            run_demo(epochs=0)


if __name__ == "__main__":
    unittest.main()
