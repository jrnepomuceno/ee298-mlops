"""Tests for mapping bounded v6 predictions into runtime intents."""
from __future__ import annotations

import unittest

from rpi5.v6_adapter import adapt_v6_action_result, adapt_v6_result
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

    def test_action_mode_allows_reviewed_music_and_rejects_other_labels(self):
        play_result = adapt_v6_action_result(self._result("PLAY_MUSIC"), {})
        self.assertEqual(play_result["intent"], "play_music")

        next_result = adapt_v6_action_result(self._result("NEXT"), {})
        self.assertEqual(next_result["intent"], "oov")

        timer_result = adapt_v6_action_result(self._result("TIMER", {
            "timer_minute": _slot(5),
            "timer_second": _slot(0),
        }), {})
        self.assertEqual(timer_result["intent"], "oov")
        self.assertEqual(timer_result["slots"], {})

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