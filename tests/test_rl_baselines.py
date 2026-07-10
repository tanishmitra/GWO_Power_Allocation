from __future__ import annotations

import unittest

from scripts.evaluate_rl_baselines import build_config, evaluate_baselines, parse_args


class RLBaselineEvaluationTests(unittest.TestCase):
    def test_evaluate_baselines_returns_expected_summaries(self) -> None:
        args = parse_args(
            [
                "--subcarriers",
                "4",
                "--horizon",
                "3",
                "--gwo-population",
                "6",
                "--gwo-iterations",
                "3",
            ]
        )
        config = build_config(args)

        summary = evaluate_baselines(config, include_pso=False, seed=11)

        self.assertEqual(summary["num_subcarriers"], 4)
        self.assertEqual(summary["num_time_steps"], 3)
        self.assertIn("random_policy", summary["summaries"])
        self.assertIn("equal_power", summary["summaries"])
        self.assertIn("gwo_per_snapshot", summary["summaries"])
        for method_summary in summary["summaries"].values():
            self.assertIn("mean_weighted_objective", method_summary)
            self.assertIn("mean_runtime_ms_per_step", method_summary)


if __name__ == "__main__":
    unittest.main()
