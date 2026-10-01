"""Tests for mapping bounded v6 predictions into runtime intents."""
from __future__ import annotations

import unittest

from rpi5.v6_adapter import (V6_ACTION_LABELS, adapt_v6_action_result,
                            adapt_v6_result)
from rpi5.harness import FacadePipeline


def _slot(value, confidence=0.99):
    return {"value": value, "confidence": confidence}


class V6AdapterTests(unittest.TestCase):
    def _result(self, intent, slots=None):
        return {"intent": intent, "intent_confidence": 0.99,
                "slots": slots or {}, "backend": "intent_bounded_numeric_slots"}

    def test_maps_supported_music_labels(self):
        self.assertEqual(adapt_v6_result(self._result("PLAY_MUSIC"), {})["intent"],
                         "play_music")
        self.assertEqual(adapt_v6_result(self._result("NEXT"), {})["intent"],
                         "next_music")
        self.assertEqual(adapt_v6_result(self._result("STOP"), {})["intent"],
                         "stop_music")

    def test_action_allowlist_matches_supported_project_intents(self):
        self.assertEqual(V6_ACTION_LABELS, frozenset({
            "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF",
            "PAUSE", "STOP", "VOLUME_UP", "VOLUME_DOWN", "LIST_REMINDERS",
            "TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "CREATE_REMINDER",
        }))

        for label in ("CALL", "MESSAGE", "NEXT"):
            with self.subTest(label=label):
                result = adapt_v6_action_result(self._result(label), {})
                self.assertEqual(result["intent"], "oov")

    def test_action_mode_allows_timer_with_valid_slots(self):
        play_result = adapt_v6_action_result(self._result("PLAY_MUSIC"), {})
        self.assertEqual(play_result["intent"], "play_music")

        timer_result = adapt_v6_action_result(self._result("TIMER", {
            "timer_minute": _slot(5),
            "timer_second": _slot(0),
        }), {})
        self.assertEqual(timer_result["intent"], "set_timer")
        self.assertEqual(timer_result["slots"], {
            "duration": 5, "duration_unit": "minute",
        })

    def test_action_mode_allows_volume_labels(self):
        self.assertEqual(
            adapt_v6_action_result(self._result("VOLUME_UP"), {})["intent"],
            "volume_up",
        )
        self.assertEqual(
            adapt_v6_action_result(self._result("VOLUME_DOWN"), {})["intent"],
            "volume_down",
        )

        adjusted = self._result("VOLUME_DOWN")
        adjusted["pre_threshold_intent"] = "VOLUME_UP"
        result = adapt_v6_action_result(adjusted, {})
        self.assertEqual(result["intent"], "volume_down")
        self.assertEqual(result["model_intent"], "VOLUME_UP")

    def test_action_mode_allows_time_query(self):
        result = adapt_v6_action_result(self._result("TIME"), {})
        self.assertEqual(result["intent"], "what_time")
        outcome = FacadePipeline(dry_run=True).process(result, source="test")
        self.assertTrue(outcome.handled)
        self.assertEqual(outcome.request.action_code, "query.time")

    def test_project_supported_labels_reach_facade(self):
        cases = (
            ("WEATHER", {}, "what_weather"),
            ("LIST_REMINDERS", {}, "what_reminders"),
            ("LIGHT_ON", {}, "turn_on_lights"),
            ("LIGHT_OFF", {}, "turn_off_lights"),
            ("BRIGHTNESS", {"percent": _slot(40)}, "dim_lights"),
            ("TEMPERATURE", {"degrees": _slot(25)}, "set_temperature"),
            ("TIMER", {"timer_minute": _slot(5),
                       "timer_second": _slot(0)}, "set_timer"),
            ("ALARM", {"alarm_hour": _slot(7),
                       "alarm_minute": _slot(30),
                       "alarm_meridiem": _slot("PM")}, "set_alarm"),
            ("CREATE_REMINDER", {"task": _slot("buy eggs")}, "remind"),
        )
        pipeline = FacadePipeline(dry_run=True)
        for label, slots, expected_intent in cases:
            with self.subTest(label=label):
                adapted = adapt_v6_action_result(
                    self._result(label, slots), {})
                self.assertEqual(adapted["intent"], expected_intent)
                outcome = pipeline.process(adapted, source="test")
                self.assertTrue(outcome.handled)

    def test_unmapped_message_rejects_instead_of_dispatching(self):
        result = adapt_v6_result(self._result("MESSAGE"), {})
        self.assertEqual(result["intent"], "oov")
        self.assertEqual(result["slots"], {})
        self.assertEqual(result["model_intent"], "MESSAGE")

    def test_maps_timer_components_to_seconds(self):
        result = adapt_v6_result(self._result("TIMER", {
            "timer_minute": _slot(2),
            "timer_second": _slot(5),
        }), {})
        self.assertEqual(result["intent"], "set_timer")
        self.assertEqual(result["slots"], {"duration": 125,
                                            "duration_unit": "second"})

    def test_maps_alarm_components(self):
        result = adapt_v6_result(self._result("ALARM", {
            "alarm_hour": _slot(7),
            "alarm_minute": _slot(30),
            "alarm_meridiem": _slot("PM"),
        }), {})
        self.assertEqual(result["slots"], {"time": "7:30 PM"})

    def test_low_confidence_slot_is_left_missing(self):
        result = adapt_v6_result(self._result("BRIGHTNESS", {
            "percent": _slot(40, 0.4),
        }), {}, slot_threshold=0.75)
        self.assertEqual(result["intent"], "dim_lights")
        self.assertEqual(result["slots"], {})


if __name__ == "__main__":
    unittest.main()