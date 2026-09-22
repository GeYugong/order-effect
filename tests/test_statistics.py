import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import contrast, holm


class StatisticsTests(unittest.TestCase):
    def test_identical_distributions_have_zero_contrast(self):
        values = [.3, .4, .5, .6, .7]
        out = contrast({"a": values, "b": values}, {"a": 1, "b": -1})
        self.assertAlmostEqual(out["estimate"], 0)
        self.assertAlmostEqual(out["p_two_sided"], 1)
        self.assertLess(out["ci95"][0], 0)
        self.assertGreater(out["ci95"][1], 0)

    def test_contrast_uses_runs_not_task_counts(self):
        out = contrast({"a": [.8], "b": [.1]}, {"a": 1, "b": -1})
        self.assertFalse(out["available"])

    def test_zero_sample_variance_not_reported_as_certainty(self):
        out = contrast({"a": [.8]*5, "b": [.1]*5}, {"a": 1, "b": -1})
        self.assertFalse(out["available"])

    def test_holm_adjustment(self):
        tests = {str(i): {"available": True, "p_two_sided": p} for i, p in enumerate([.01, .04, .2])}
        holm(tests)
        self.assertAlmostEqual(tests["0"]["p_holm"], .03)
        self.assertAlmostEqual(tests["1"]["p_holm"], .08)
        self.assertAlmostEqual(tests["2"]["p_holm"], .2)


if __name__ == "__main__":
    unittest.main()
