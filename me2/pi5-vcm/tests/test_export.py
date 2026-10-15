from __future__ import annotations

import unittest

import numpy as np

from pi5_vcm.export import _tune_threshold


class ExportTests(unittest.TestCase):
    def test_tune_threshold_honors_recall_constraint_and_oos_false_accepts(self) -> None:
        validation = {
            "intent_labels": np.array([0, 19]),
            "intent_probs": np.array(
                [
                    [0.96] + [0.0] * 18 + [0.01],
                    [0.0] * 19 + [1.0],
                ],
                dtype=np.float32,
            ),
        }
        tune_oos = {
            "intent_labels": np.array([19]),
            "intent_probs": np.array(
                [[0.75] + [0.0] * 18 + [0.10]], dtype=np.float32
            ),
        }

        result = _tune_threshold(validation, tune_oos)

        self.assertEqual(result["tau"], 0.76)
        self.assertTrue(result["constraint_met"])
        self.assertEqual(result["tune_oos_false_accepts"], 0)
        self.assertEqual(result["in_scope_recall"], 1.0)

    def test_tune_threshold_rejects_non_oos_tuning_examples(self) -> None:
        validation = {
            "intent_labels": np.array([0]),
            "intent_probs": np.array([[1.0] + [0.0] * 19], dtype=np.float32),
        }
        tune_oos = {
            "intent_labels": np.array([0]),
            "intent_probs": np.array([[1.0] + [0.0] * 19], dtype=np.float32),
        }

        with self.assertRaisesRegex(ValueError, "only OUT_OF_SCOPE"):
            _tune_threshold(validation, tune_oos)


if __name__ == "__main__":
    unittest.main()
