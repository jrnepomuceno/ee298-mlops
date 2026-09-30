"""Tests for safe diagnostics with intent-only ONNX models."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from inference.ort_infer import run_utterance
from rpi5.harness import HarnessConfig, PiHarness


class _Output:
    name = "intent_logits"
    shape = [1, 3]


class _Session:
    def get_outputs(self):
        return [_Output()]

    def run(self, _names, _inputs):
        return [np.asarray([[0.1, 2.0, 0.2]], dtype=np.float32)]


class IntentOnlyInferenceTests(unittest.TestCase):
    def test_single_output_returns_raw_label_without_slots(self):
        with mock.patch("inference.ort_infer.kaldi_fbank",
                        return_value=np.zeros((4, 80), dtype=np.float32)):
            result = run_utterance(
                _Session(), np.zeros(1600, dtype=np.float32), 400,
                ["PLAY_MUSIC", "NEXT", "STOP"], [],
            )
        self.assertEqual(result["intent"], "NEXT")
        self.assertEqual(result["slots"], {})
        self.assertEqual(result["transcript"], "")
        self.assertEqual(result["backend"], "intent_only")

    def test_single_output_requires_matching_label_count(self):
        with mock.patch("inference.ort_infer.kaldi_fbank",
                        return_value=np.zeros((4, 80), dtype=np.float32)):
            with self.assertRaisesRegex(ValueError, "does not match its labels"):
                run_utterance(
                    _Session(), np.zeros(1600, dtype=np.float32), 400,
                    ["PLAY_MUSIC"], [],
                )

    def test_intent_only_harness_event_is_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as directory:
            labels_path = Path(directory) / "labels.json"
            labels_path.write_text(json.dumps({
                "task": "intent_classification_only",
                "labels": ["PLAY_MUSIC", "NEXT", "STOP"],
            }), encoding="utf-8")
            checkpoint = Path(directory) / "model.onnx"
            checkpoint.write_bytes(b"test")
            with mock.patch("rpi5.harness.load_session",
                            return_value=(_Session(), "mels", [], [])):
                harness = PiHarness(HarnessConfig(
                    checkpoint=str(checkpoint),
                    intent_labels_path=str(labels_path),
                    warmup=0,
                ))
            with mock.patch("rpi5.harness.run_utterance",
                            return_value={
                                "intent": "NEXT",
                                "intent_confidence": 0.91,
                                "transcript": "",
                                "slots": {},
                                "frames": 4,
                                "latency_ms": 1.0,
                            }):
                event = harness.recognize_audio(np.zeros(1600, dtype=np.float32))
            self.assertEqual(event["action"]["status"], "diagnostic_only")
            self.assertFalse(event["action"]["side_effects"])
            self.assertFalse(event["reply"]["speak"])
            with self.assertRaisesRegex(RuntimeError, "actions are disabled"):
                harness.recognize_and_act(np.zeros(1600, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()