"""Model-free scenario tests for every supported intent and the OOV gate."""
from __future__ import annotations

import unittest
from unittest import mock

from rpi5.facade import INTENT_SPECS, RejectResult
from rpi5.harness import FacadePipeline
from rpi5.mock_inference import MOCK_SLOTS, MockInference
from rpi5.mock_run import main as mock_run_main
from rpi5.orchestrator import default_orchestrator


class RecordingRgb:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def processing(self) -> None:
        self.calls.append("processing")

    def speaking(self) -> None:
        self.calls.append("speaking")

    def idle(self) -> None:
        self.calls.append("idle")


class MockIntentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rgb = RecordingRgb()
        self.spoken: list[str] = []
        self.events: list[dict] = []
        self.pipeline = FacadePipeline(
            orchestrator=default_orchestrator(
                dry_run=True,
                rgb=self.rgb,
                speak=self.spoken.append,
                on_event=self.events.append,
            ),
            threshold=0.75,
            dry_run=True,
        )
        self.inference = MockInference()

    def test_catalog_covers_every_facade_intent(self) -> None:
        self.assertEqual(set(MOCK_SLOTS) - {"oov"}, set(INTENT_SPECS))
        self.assertIn("oov", MOCK_SLOTS)
        self.assertNotIn("mute", MOCK_SLOTS)

    def test_temperature_scenario_is_audio_only(self) -> None:
        outcome = self.pipeline.process(
            self.inference.recognize("set_temperature", slot_overrides={"temperature": 25}),
            source="mock",
        )
        self.assertTrue(outcome.handled)
        self.assertEqual(outcome.reply_text, "Temperature set to 25 degrees.")
        self.assertEqual(outcome.request.slots["temperature"], 25)
        self.assertFalse(outcome.execution.side_effects)

    def test_mock_temperature_is_range_checked(self) -> None:
        with mock.patch("sys.argv", ["mock_run", "--intent", "set_temperature",
                                     "--temperature", "36"]):
            with self.assertRaisesRegex(SystemExit, "between 10 and 35 Celsius"):
                mock_run_main()

    def test_every_intent_runs_through_dry_run_tasks(self) -> None:
        for intent in INTENT_SPECS:
            with self.subTest(intent=intent):
                outcome = self.pipeline.process(
                    self.inference.recognize(intent), source="mock")
                self.assertTrue(outcome.handled)
                self.assertEqual(outcome.request.intent, intent)
                self.assertEqual(outcome.execution.action_code,
                                 INTENT_SPECS[intent]["code"])
                self.assertFalse(outcome.execution.side_effects)
                self.assertTrue(outcome.reply_text)

        self.assertEqual(len(self.events), len(INTENT_SPECS))
        self.assertEqual(len(self.spoken), len(INTENT_SPECS))
        self.assertEqual(self.rgb.calls.count("processing"), len(INTENT_SPECS))
        self.assertEqual(self.rgb.calls.count("idle"), len(INTENT_SPECS))

    def test_oov_scenario_is_rejected_without_side_effects(self) -> None:
        outcome = self.pipeline.process(self.inference.recognize("oov"), source="mock")
        self.assertFalse(outcome.handled)
        self.assertIsInstance(outcome.reject, RejectResult)
        self.assertEqual(outcome.reject.reason, "oov")
        self.assertFalse(outcome.execution)

    def test_low_confidence_scenario_is_rejected(self) -> None:
        outcome = self.pipeline.process(
            self.inference.recognize("turn_on_lights", confidence=0.2),
            source="mock",
        )
        self.assertFalse(outcome.handled)
        self.assertEqual(outcome.reject.reason, "low_confidence")

    def test_volume_intents_use_independent_confidence_thresholds(self) -> None:
        pipeline = FacadePipeline(
            threshold=0.75,
            intent_thresholds={"volume_up": 0.90, "volume_down": 0.60},
            dry_run=True,
        )
        down = pipeline.process({
            "intent": "volume_down", "intent_confidence": 0.65, "slots": {},
        }, source="threshold-test")
        up = pipeline.process({
            "intent": "volume_up", "intent_confidence": 0.85, "slots": {},
        }, source="threshold-test")

        self.assertTrue(down.handled)
        self.assertFalse(up.handled)
        self.assertEqual(up.reject.reason, "low_confidence")

    def test_invalid_mock_intent_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "no mock scenario"):
            self.inference.recognize("not_an_intent")

    def test_live_device_mode_requires_a_single_intent(self) -> None:
        with mock.patch("sys.argv", ["mock_run", "--intent", "all", "--live-lights"]):
            with self.assertRaisesRegex(SystemExit, "one explicit --intent"):
                mock_run_main()

    def test_live_volume_requires_volume_intent(self) -> None:
        with mock.patch("sys.argv", ["mock_run", "--intent", "all", "--live-volume"]):
            with self.assertRaisesRegex(SystemExit, "one explicit --intent"):
                mock_run_main()
        with mock.patch("sys.argv", ["mock_run", "--intent", "set_temperature", "--live-volume"]):
            with self.assertRaisesRegex(SystemExit, "volume_up or volume_down"):
                mock_run_main()

    def test_live_timer_requires_timer_intent_and_speaker(self) -> None:
        with mock.patch("sys.argv", ["mock_run", "--intent", "what_time", "--live-timer"]):
            with self.assertRaisesRegex(SystemExit, "set_timer or stop_timer"):
                mock_run_main()
        with mock.patch("sys.argv", ["mock_run", "--intent", "set_timer", "--live-timer"]):
            with self.assertRaisesRegex(SystemExit, "requires --live-speaker"):
                mock_run_main()

    def test_live_timer_duration_is_bounded(self) -> None:
        with mock.patch("sys.argv", ["mock_run", "--intent", "set_timer", "--live-timer",
                                     "--live-speaker", "--timer-seconds", "31"]):
            with self.assertRaisesRegex(SystemExit, "timer-seconds must be between 1 and 30"):
                mock_run_main()

    def test_live_rgb_requires_live_timer(self) -> None:
        with mock.patch("sys.argv", ["mock_run", "--intent", "set_timer", "--live-rgb",
                                     "--rgb-executable", "/opt/bin/quadcastrgb"]):
            with self.assertRaisesRegex(SystemExit, "requires --live-timer"):
                mock_run_main()

    def test_live_call_requires_explicit_confirmation(self) -> None:
        with mock.patch("sys.argv", ["mock_run", "--intent", "call", "--live-call",
                                     "--mock-contact", "+15551234567"]):
            with self.assertRaisesRegex(SystemExit, "requires --confirm-call"):
                mock_run_main()

    def test_live_call_duration_is_bounded(self) -> None:
        with mock.patch("sys.argv", ["mock_run", "--intent", "call", "--live-call",
                                     "--confirm-call", "--mock-contact",
                                     "+15551234567", "--call-seconds", "0"]):
            with self.assertRaisesRegex(SystemExit, "between 1 and 120"):
                mock_run_main()

    def test_live_weather_requires_weather_intent(self) -> None:
        with mock.patch("sys.argv", ["mock_run", "--intent", "what_time",
                                     "--live-weather"]):
            with self.assertRaisesRegex(SystemExit, "requires --intent what_weather"):
                mock_run_main()


if __name__ == "__main__":
    unittest.main()
