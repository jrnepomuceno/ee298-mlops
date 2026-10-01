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


class _Input:
    def __init__(self, name):
        self.name = name


class _Session:
    def get_inputs(self):
        return [_Input("mels")]

    def get_outputs(self):
        return [_Output()]

    def run(self, _names, _inputs):
        return [np.asarray([[0.1, 2.0, 0.2]], dtype=np.float32)]


class _ScoreSession(_Session):
    def __init__(self, logits):
        self.logits = logits

    def run(self, _names, _inputs):
        return [np.asarray([self.logits], dtype=np.float32)]


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

    def test_volume_thresholds_can_select_down_over_raw_up_winner(self):
        with mock.patch("inference.ort_infer.kaldi_fbank",
                        return_value=np.zeros((4, 80), dtype=np.float32)):
            result = run_utterance(
                _ScoreSession([2.0, 1.0, 0.0]),
                np.zeros(1600, dtype=np.float32), 400,
                ["volume_up", "volume_down", "oov"], [],
                intent_thresholds={"volume_up": 0.80, "volume_down": 0.20},
            )

        self.assertEqual(result["intent"], "volume_down")
        self.assertEqual(result["pre_threshold_intent"], "volume_up")
        self.assertAlmostEqual(result["intent_confidence"], 0.2447, places=3)
        self.assertAlmostEqual(
            result["volume_intent_scores"]["volume_up"], 0.6652, places=3)
        self.assertAlmostEqual(
            result["volume_intent_scores"]["volume_down"], 0.2447, places=3)

    def test_volume_thresholds_accept_uppercase_v6_labels(self):
        with mock.patch("inference.ort_infer.kaldi_fbank",
                        return_value=np.zeros((4, 80), dtype=np.float32)):
            result = run_utterance(
                _ScoreSession([2.0, 1.0, 0.0]),
                np.zeros(1600, dtype=np.float32), 400,
                ["VOLUME_UP", "VOLUME_DOWN", "OOV"], [],
                intent_thresholds={"volume_up": 0.80, "volume_down": 0.20},
            )

        self.assertEqual(result["intent"], "VOLUME_DOWN")
        self.assertEqual(result["pre_threshold_intent"], "VOLUME_UP")

    def test_volume_thresholds_reject_pair_when_neither_clears_cutoff(self):
        with mock.patch("inference.ort_infer.kaldi_fbank",
                        return_value=np.zeros((4, 80), dtype=np.float32)):
            result = run_utterance(
                _ScoreSession([2.0, 1.0, 0.0]),
                np.zeros(1600, dtype=np.float32), 400,
                ["volume_up", "volume_down", "oov"], [],
                intent_thresholds={"volume_up": 0.80, "volume_down": 0.30},
            )

        self.assertEqual(result["intent"], "oov")
        self.assertEqual(result["pre_threshold_intent"], "volume_up")

    def test_default_volume_thresholds_keep_down_at_060(self):
        with mock.patch("inference.ort_infer.kaldi_fbank",
                        return_value=np.zeros((4, 80), dtype=np.float32)):
            result = run_utterance(
                _ScoreSession([0.0, 2.0, 0.0]),
                np.zeros(1600, dtype=np.float32), 400,
                ["volume_up", "volume_down", "oov"], [],
            )

        self.assertEqual(result["intent"], "volume_down")

    def test_bounded_slot_diagnostic_decodes_owned_slot_values(self):
        class SlotSession:
            def get_inputs(self):
                return [_Input("mels"), _Input("lengths")]

            def run(self, _names, inputs):
                self.asserted_lengths = inputs["lengths"].tolist()
                intent = np.zeros((1, 3), dtype=np.float32)
                intent[0, 1] = 10.0
                minute = np.zeros((1, 61), dtype=np.float32)
                minute[0, 2] = 10.0
                second = np.zeros((1, 60), dtype=np.float32)
                second[0, 5] = 10.0
                alarm_hour = np.zeros((1, 12), dtype=np.float32)
                alarm_minute = np.zeros((1, 4), dtype=np.float32)
                alarm_meridiem = np.zeros((1, 2), dtype=np.float32)
                degrees = np.zeros((1, 25), dtype=np.float32)
                percent = np.zeros((1, 101), dtype=np.float32)
                task = np.zeros((1, 2), dtype=np.float32)
                return [intent, minute, second, alarm_hour, alarm_minute,
                        alarm_meridiem, degrees, percent, task]

        contract = {
            "task": "intent_bounded_numeric_slots",
            "slot_labels": ["timer_minute", "timer_second", "alarm_hour",
                            "alarm_minute", "alarm_meridiem", "degrees",
                            "percent", "task"],
            "slot_owners": {"timer_minute": "TIMER", "timer_second": "TIMER"},
            "slot_values": {"timer_minute": list(range(61)),
                            "timer_second": list(range(60))},
            "feature_config": {"snip_edges": False, "max_frames": None},
        }
        session = SlotSession()
        with mock.patch("inference.ort_infer.kaldi_fbank",
                        return_value=np.zeros((4, 80), dtype=np.float32)) as fbank:
            result = run_utterance(
                session, np.zeros(1600, dtype=np.float32), 400,
                ["PLAY_MUSIC", "TIMER", "STOP"], [], contract)

        self.assertEqual(session.asserted_lengths, [4])
        self.assertFalse(fbank.call_args.kwargs["snip_edges"])
        self.assertEqual(result["intent"], "TIMER")
        self.assertEqual(result["slots"]["timer_minute"]["value"], 2)
        self.assertEqual(result["slots"]["timer_second"]["value"], 5)
        self.assertEqual(result["backend"], "intent_bounded_numeric_slots")

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