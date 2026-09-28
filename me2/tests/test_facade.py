"""Unit tests for the facade (pure decode) and orchestrator (routing)."""
from __future__ import annotations

import unittest

from rpi5.facade import (
    ActionRequest, RejectResult, decode, INTENT_SPECS, KNOWN_INTENTS,
    DEFAULT_CONFIDENCE_THRESHOLD,
)
from rpi5.orchestrator import Orchestrator, default_orchestrator
from rpi5.executors.base import ExecutionResult


class DecodeGateTests(unittest.TestCase):
    def test_oov_intent_rejected(self):
        r = decode("oov", {}, 0.99)
        self.assertIsInstance(r, RejectResult)
        self.assertEqual(r.reason, "oov")

    def test_unknown_intent_rejected_as_oov(self):
        r = decode("dance_party", {}, 0.99)
        self.assertIsInstance(r, RejectResult)
        self.assertEqual(r.reason, "oov")

    def test_low_confidence_rejected(self):
        r = decode("turn_on_lights", {}, 0.5)
        self.assertIsInstance(r, RejectResult)
        self.assertEqual(r.reason, "low_confidence")

    def test_confidence_above_threshold_passes(self):
        r = decode("turn_on_lights", {}, DEFAULT_CONFIDENCE_THRESHOLD + 0.01)
        self.assertIsInstance(r, ActionRequest)

    def test_none_confidence_rejected(self):
        r = decode("turn_on_lights", {}, None)
        self.assertIsInstance(r, RejectResult)
        self.assertEqual(r.reason, "low_confidence")


class DecodeSlotTests(unittest.TestCase):
    def test_missing_required_slot(self):
        r = decode("dim_lights", {}, 0.9)
        self.assertIsInstance(r, RejectResult)
        self.assertEqual(r.reason, "missing_slot")

    def test_out_of_range_slot(self):
        r = decode("dim_lights", {"percent": 150}, 0.9)
        self.assertIsInstance(r, RejectResult)
        self.assertEqual(r.reason, "invalid_slot")

    def test_non_numeric_slot(self):
        r = decode("set_temperature", {"temperature": "hot"}, 0.9)
        self.assertIsInstance(r, RejectResult)
        self.assertEqual(r.reason, "invalid_slot")

    def test_word_number_coercion(self):
        r = decode("dim_lights", {"percent": "forty"}, 0.9)
        self.assertIsInstance(r, ActionRequest)
        self.assertEqual(r.slots["percent"], 40)

    def test_string_digit_coercion(self):
        r = decode("set_timer", {"duration": "5", "duration_unit": "minute"}, 0.9)
        self.assertIsInstance(r, ActionRequest)
        self.assertEqual(r.slots["duration"], 5)


class DecodeReplyTests(unittest.TestCase):
    def test_all_known_intents_have_specs(self):
        for intent in KNOWN_INTENTS:
            self.assertIn(intent, INTENT_SPECS)
            self.assertIn("category", INTENT_SPECS[intent])
            self.assertIn("code", INTENT_SPECS[intent])
            self.assertIn("reply", INTENT_SPECS[intent])

    def test_reply_contains_slot_value(self):
        r = decode("dim_lights", {"percent": 40}, 0.9)
        self.assertIsInstance(r, ActionRequest)
        self.assertIn("40", r.reply_text)

    def test_timer_reply_with_unit(self):
        r = decode("set_timer", {"duration": 5, "duration_unit": "minute"}, 0.9)
        self.assertIsInstance(r, ActionRequest)
        self.assertIn("5", r.reply_text)
        self.assertIn("minute", r.reply_text)

    def test_what_time_reply_has_clock(self):
        r = decode("what_time", {}, 0.9)
        self.assertIsInstance(r, ActionRequest)
        # Should contain a time-like string (digits + AM/PM)
        self.assertTrue(any(ch.isdigit() for ch in r.reply_text))


class OrchestratorRoutingTests(unittest.TestCase):
    def test_routes_to_correct_category(self):
        events = []
        orch = default_orchestrator(on_event=lambda e: events.append(e))
        req = decode("dim_lights", {"percent": 40}, 0.9)
        self.assertIsInstance(req, ActionRequest)
        result = orch.run(req)
        self.assertTrue(result.handled)
        self.assertEqual(result.execution.category, "lights")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["intent"], "dim_lights")

    def test_reject_produces_event(self):
        events = []
        orch = default_orchestrator(on_event=lambda e: events.append(e))
        rej = decode("oov", {}, 0.9)
        self.assertIsInstance(rej, RejectResult)
        result = orch.run(rej)
        self.assertFalse(result.handled)
        self.assertIsNotNone(result.reject)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "rejected")

    def test_executor_error_fails_soft(self):
        """A raising executor should not crash the orchestrator."""
        from rpi5.executors.base import Executor

        class BadExecutor(Executor):
            category = "lights"
            def run(self, req):
                raise RuntimeError("boom")

        orch = Orchestrator({"lights": BadExecutor()})
        req = decode("turn_on_lights", {}, 0.9)
        result = orch.run(req)
        self.assertFalse(result.handled)
        self.assertIn("boom", result.execution.detail)
        self.assertEqual(result.reply_text, "Sorry, that didn't work.")

    def test_no_executor_for_category(self):
        orch = Orchestrator({})  # empty
        req = decode("turn_on_lights", {}, 0.9)
        result = orch.run(req)
        self.assertFalse(result.handled)
        self.assertIn("no executor", result.execution.detail)


class DryRunSideEffectTests(unittest.TestCase):
    def test_dry_run_has_no_side_effects(self):
        orch = default_orchestrator(dry_run=True)
        req = decode("turn_on_lights", {}, 0.9)
        result = orch.run(req)
        self.assertTrue(result.handled)
        self.assertFalse(result.execution.side_effects)

    def test_info_query_returns_answer_even_in_dry_run(self):
        orch = default_orchestrator(dry_run=True)
        req = decode("what_time", {}, 0.9)
        result = orch.run(req)
        self.assertTrue(result.handled)
        self.assertIn("answer", result.execution.payload)

    def test_weather_live_answer_spoken_even_in_dry_run(self):
        """Regression: the live mic pipeline runs dry_run=True, but weather is
        read-only, so a configured provider must still be fetched and spoken.
        Previously the `and not self.dry_run` gate suppressed it and the user
        heard the offline fallback line instead of the real weather."""
        orch = default_orchestrator(
            dry_run=True, weather_fn=lambda: "It's 25 degrees and cloudy.")
        req = decode("what_weather", {}, 0.9)
        result = orch.run(req)
        self.assertTrue(result.handled)
        self.assertEqual(result.reply_text, "It's 25 degrees and cloudy.")
        self.assertEqual(result.execution.payload["answer"],
                         "It's 25 degrees and cloudy.")

    def test_weather_without_provider_keeps_fallback_in_dry_run(self):
        orch = default_orchestrator(dry_run=True, weather_fn=None)
        req = decode("what_weather", {}, 0.9)
        result = orch.run(req)
        self.assertTrue(result.handled)
        self.assertEqual(
            result.reply_text,
            "Weather is not available without a configured source.")


class WhatTimeAnswerSourceTests(unittest.TestCase):
    """what_time: the executor's payload['answer'] is the single source of
    truth for what gets spoken, and the time string is portable."""

    def test_spoken_reply_equals_executor_answer(self):
        orch = default_orchestrator(dry_run=True)
        req = decode("what_time", {}, 0.9)
        result = orch.run(req)
        self.assertTrue(result.handled)
        answer = result.execution.payload["answer"]
        self.assertEqual(result.reply_text, answer)
        self.assertEqual(result.event["reply"], answer)

    def test_time_string_is_portable_no_literal_percent(self):
        from rpi5.facade import _local_time_text
        t = _local_time_text()
        self.assertNotIn("%", t)          # no un-rendered %-I / %M / %p
        self.assertRegex(t, r"^\d{1,2}:\d{2} (AM|PM)$")

    def test_divergent_live_answer_overrides_template(self):
        """A live provider returning a value different from the static
        template must be the one spoken (future-proofs weather/reminders)."""
        # Simulate a live weather/time provider with a different value.
        class FakeProvider:
            def run(self, req):
                return ExecutionResult(
                    ok=True, intent=req.intent, category=req.category,
                    action_code=req.action_code, detail="synced",
                    side_effects=False, payload={"answer": "12:00 PM"})

        orch = Orchestrator({"info": FakeProvider()})
        req = decode("what_time", {}, 0.9)
        result = orch.run(req)
        self.assertTrue(result.handled)
        self.assertEqual(result.reply_text, "12:00 PM")

class WeatherProviderTests(unittest.TestCase):
    """Live OpenWeatherMap provider: formatting, env handling, executor wiring."""

    def test_condition_word_mapping(self):
        from rpi5.weather import _condition_word
        self.assertEqual(_condition_word("Clear", "clear sky"), "clear skies")
        self.assertEqual(_condition_word("Clouds", "broken clouds"), "cloudy")
        self.assertEqual(_condition_word("Rain", "light rain"), "rainy")
        # Unknown main falls back to the description, lower-cased, no period.
        self.assertEqual(_condition_word("Weird", "Scattered showers."), "scattered showers")

    def test_spoken_line_format_from_mock_response(self):
        import rpi5.weather as w
        fake = {
            "cod": "200",
            "main": {"temp": 24.6},
            "weather": [{"main": "Clouds", "description": "broken clouds"}],
        }
        orig = w._fetch_current
        w._fetch_current = lambda loc, key: fake
        try:
            fn = w.make_weather_fn(location="Quezon City", api_key="k")
            line = fn()
        finally:
            w._fetch_current = orig
        self.assertEqual(line, "It's 25 degrees and cloudy in Quezon City.")
        self.assertNotIn("%", line)

    def test_missing_api_key_raises(self):
        import os
        import rpi5.weather as w
        saved = os.environ.pop("OPENWEATHER_API_KEY", None)
        try:
            fn = w.make_weather_fn(location="Quezon City", api_key=None)
            with self.assertRaises(RuntimeError):
                fn()
        finally:
            if saved is not None:
                os.environ["OPENWEATHER_API_KEY"] = saved

    def test_error_response_raises(self):
        import rpi5.weather as w
        orig = w._fetch_current
        w._fetch_current = lambda loc, key: {"cod": "404", "message": "city not found"}
        try:
            fn = w.make_weather_fn(location="Nowhere", api_key="k")
            with self.assertRaises(RuntimeError):
                fn()
        finally:
            w._fetch_current = orig

    def test_executor_speaks_live_weather_answer(self):
        """End-to-end: what_weather -> InfoExecutor -> spoken reply == live answer."""
        from rpi5.executors.info import InfoExecutor
        orch = Orchestrator({"info": InfoExecutor(dry_run=False,
                                                  weather_fn=lambda: "It's 30 degrees and clear skies in Quezon City.")})
        req = decode("what_weather", {}, 0.9)
        result = orch.run(req)
        self.assertTrue(result.handled)
        self.assertEqual(result.reply_text,
                         "It's 30 degrees and clear skies in Quezon City.")
        self.assertEqual(result.execution.payload["answer"], result.reply_text)

    def test_executor_falls_back_without_weather_fn(self):
        from rpi5.executors.info import InfoExecutor
        orch = Orchestrator({"info": InfoExecutor(dry_run=False)})
        req = decode("what_weather", {}, 0.9)
        result = orch.run(req)
        self.assertTrue(result.handled)
        self.assertIn("not available", result.reply_text)


if __name__ == "__main__":
    unittest.main()
