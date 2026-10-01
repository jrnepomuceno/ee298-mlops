"""Tests for mapping v1 intent-only predictions into runtime intents."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rpi5.v1_adapter import (
    DEFAULT_V1_SLOTS,
    V1_ACTION_LABELS,
    V1_TO_RUNTIME_INTENT,
    adapt_v1_action_result,
    adapt_v1_result,
)
from rpi5.harness import FacadePipeline, HarnessConfig, PiHarness


class V1AdapterTests(unittest.TestCase):
    def _result(self, intent):
        return {
            "intent": intent,
            "intent_confidence": 0.99,
            "slots": {},
            "backend": "intent_only",
        }

    def test_all_17_mapped_labels_defined(self):
        self.assertEqual(len(V1_TO_RUNTIME_INTENT), 17)
        self.assertNotIn("MESSAGE", V1_TO_RUNTIME_INTENT)

    def test_maps_supported_slotless_labels(self):
        self.assertEqual(adapt_v1_result(self._result("PLAY_MUSIC"))["intent"], "play_music")
        self.assertEqual(adapt_v1_result(self._result("PAUSE"))["intent"], "pause_music")
        self.assertEqual(adapt_v1_result(self._result("STOP"))["intent"], "stop_music")
        self.assertEqual(adapt_v1_result(self._result("TIME"))["intent"], "what_time")
        self.assertEqual(adapt_v1_result(self._result("WEATHER"))["intent"], "what_weather")
        self.assertEqual(adapt_v1_result(self._result("LIGHT_ON"))["intent"], "turn_on_lights")
        self.assertEqual(adapt_v1_result(self._result("LIGHT_OFF"))["intent"], "turn_off_lights")
        self.assertEqual(adapt_v1_result(self._result("LIST_REMINDERS"))["intent"], "what_reminders")
        self.assertEqual(adapt_v1_result(self._result("VOLUME_UP"))["intent"], "volume_up")
        self.assertEqual(adapt_v1_result(self._result("VOLUME_DOWN"))["intent"], "volume_down")

    def test_unmapped_message_rejects_to_oov(self):
        result = adapt_v1_result(self._result("MESSAGE"))
        self.assertEqual(result["intent"], "oov")
        self.assertEqual(result["model_intent"], "MESSAGE")

    def test_action_allowlist_matches_project_requirements(self):
        self.assertEqual(V1_ACTION_LABELS, frozenset({
            "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF",
            "PAUSE", "STOP", "VOLUME_UP", "VOLUME_DOWN", "LIST_REMINDERS",
            "TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "CREATE_REMINDER",
        }))

        for deferred_label in ("CALL", "MESSAGE", "NEXT"):
            with self.subTest(label=deferred_label):
                res = adapt_v1_action_result(self._result(deferred_label))
                self.assertEqual(res["intent"], "oov")

    def test_default_slots_applied_for_slot_bearing_intents(self):
        timer_res = adapt_v1_action_result(self._result("TIMER"), default_slots=True)
        self.assertEqual(timer_res["intent"], "set_timer")
        self.assertEqual(timer_res["slots"], DEFAULT_V1_SLOTS["set_timer"])

        alarm_res = adapt_v1_action_result(self._result("ALARM"), default_slots=True)
        self.assertEqual(alarm_res["intent"], "set_alarm")
        self.assertEqual(alarm_res["slots"], DEFAULT_V1_SLOTS["set_alarm"])

        temp_res = adapt_v1_action_result(self._result("TEMPERATURE"), default_slots=True)
        self.assertEqual(temp_res["intent"], "set_temperature")
        self.assertEqual(temp_res["slots"], DEFAULT_V1_SLOTS["set_temperature"])

        dim_res = adapt_v1_action_result(self._result("BRIGHTNESS"), default_slots=True)
        self.assertEqual(dim_res["intent"], "dim_lights")
        self.assertEqual(dim_res["slots"], DEFAULT_V1_SLOTS["dim_lights"])

        remind_res = adapt_v1_action_result(self._result("CREATE_REMINDER"), default_slots=True)
        self.assertEqual(remind_res["intent"], "remind")
        self.assertEqual(remind_res["slots"], DEFAULT_V1_SLOTS["remind"])

    def test_default_slots_disabled_leaves_slots_empty(self):
        timer_res = adapt_v1_action_result(self._result("TIMER"), default_slots=False)
        self.assertEqual(timer_res["intent"], "set_timer")
        self.assertEqual(timer_res["slots"], {})

    def test_volume_threshold_adjusted_intent_preserved(self):
        res = self._result("VOLUME_DOWN")
        res["pre_threshold_intent"] = "VOLUME_UP"
        adapted = adapt_v1_action_result(res)
        self.assertEqual(adapted["intent"], "volume_down")
        self.assertEqual(adapted["model_intent"], "VOLUME_UP")

    def test_all_allowed_actions_pass_facade_pipeline(self):
        pipeline = FacadePipeline(dry_run=True)
        for label in V1_ACTION_LABELS:
            with self.subTest(label=label):
                adapted = adapt_v1_action_result(self._result(label), default_slots=True)
                outcome = pipeline.process(adapted, source="v1-test")
                self.assertTrue(outcome.handled)
                self.assertNotEqual(outcome.request.action_code, "")


class _Output:
    name = "intent_logits"
    shape = [1, 18]


class _Input:
    def __init__(self, name):
        self.name = name


class _Session:
    def get_inputs(self):
        return [_Input("mels")]

    def get_outputs(self):
        return [_Output()]

    def run(self, _names, _inputs):
        logits = [0.0] * 18
        logits[2] = 5.0  # TIME
        return [np.asarray([logits], dtype=np.float32)]


class V1HarnessIntegrationTests(unittest.TestCase):
    def setUp(self):
        import json
        import tempfile
        self.temp_dir = tempfile.TemporaryDirectory()
        self.contract_path = Path(self.temp_dir.name) / "intent_v1_labels.json"
        labels = [
            "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF",
            "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN",
            "CALL", "MESSAGE", "LIST_REMINDERS", "TIMER", "ALARM",
            "TEMPERATURE", "BRIGHTNESS", "CREATE_REMINDER",
        ]
        self.contract_path.write_text(json.dumps({
            "task": "intent_classification_only",
            "labels": labels,
        }), encoding="utf-8")
        self.ckpt_path = Path(self.temp_dir.name) / "model_int8.onnx"
        self.ckpt_path.write_bytes(b"dummy_onnx")

    def tearDown(self):
        self.temp_dir.cleanup()

    @mock.patch("rpi5.harness.load_session")
    def test_harness_with_enable_v1_actions_enables_actions(self, mock_load):
        mock_load.return_value = (_Session(), "mels", [], [])
        config = HarnessConfig(
            checkpoint=str(self.ckpt_path),
            intent_labels_path=str(self.contract_path),
            enable_v1_actions=True,
            warmup=0,
        )
        harness = PiHarness(config)
        self.assertFalse(harness.diagnostic_only)
        self.assertTrue(harness.v1_actions_enabled)

        with mock.patch("inference.ort_infer.kaldi_fbank",
                        return_value=np.zeros((4, 80), dtype=np.float32)):
            event = harness.recognize_audio(np.zeros(1600, dtype=np.float32))

        self.assertEqual(event["result"]["intent"], "what_time")
        self.assertNotEqual(event["action"]["status"], "diagnostic_only")

    @mock.patch("rpi5.harness.load_session")
    def test_harness_rejects_v1_with_bounded_slot_contract(self, mock_load):
        class _SlotOutput:
            name = "intent_logits"
            shape = [1, 18]

        class _SlotSession:
            def get_inputs(self):
                return [_Input("mels"), _Input("lengths")]

            def get_outputs(self):
                return [_SlotOutput(), _SlotOutput()]

            def run(self, _names, _inputs):
                return []

        mock_load.return_value = (_SlotSession(), "mels", [], [])
        slot_contract = Path(self.temp_dir.name) / "v6_contract.json"
        slot_contract.write_text(json.dumps({
            "task": "intent_bounded_numeric_slots",
            "labels": [f"INTENT_{i}" for i in range(18)],
            "slot_labels": ["percent"],
        }), encoding="utf-8")
        config = HarnessConfig(
            checkpoint=str(self.ckpt_path),
            intent_labels_path=str(slot_contract),
            enable_v1_actions=True,
            warmup=0,
        )
        with self.assertRaisesRegex(ValueError, "v1 actions require an intent-only model contract"):
            PiHarness(config)

    @mock.patch("rpi5.harness.load_session")
    def test_harness_rejects_both_v1_and_v6_enabled(self, mock_load):
        mock_load.return_value = (_Session(), "mels", [], [])
        config = HarnessConfig(
            checkpoint=str(self.ckpt_path),
            intent_labels_path=str(self.contract_path),
            enable_v1_actions=True,
            enable_v6_actions=True,
            warmup=0,
        )
        with self.assertRaisesRegex(ValueError, "cannot enable both v1 and v6"):
            PiHarness(config)


if __name__ == "__main__":
    unittest.main()
