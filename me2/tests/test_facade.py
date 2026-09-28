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


if __name__ == "__main__":
    unittest.main()
