"""Regression tests for reply selection across action-result shapes.

The OOV bug: an out-of-vocabulary utterance flowed through the facade path and
returned an ``OrchestratorResult`` dataclass, but ``reply_wav_name`` /
``build_reply`` assumed a plain dict and called ``action.get(...)``, raising
``AttributeError: 'OrchestratorResult' object has no attribute 'get'``. That
exception propagated out of the state machine (which calls ``play_reply`` with
no guard) and crashed the interaction.

These tests pin down that both the legacy dict shape and the new
``OrchestratorResult`` shape select the correct WAV and produce a spoken reply
without raising.
"""
from __future__ import annotations

import unittest

from rpi5.replies import build_reply, reply_wav_name, REPLY_WAV_BY_INTENT
from rpi5.orchestrator import OrchestratorResult
from rpi5.facade import RejectResult, ActionRequest


def _oov_result() -> OrchestratorResult:
    """Mirror the exact shape the orchestrator returns for an OOV utterance."""
    rej = RejectResult(reason="oov", detail="intent 'oov' is out of vocabulary",
                       reply_text="I don't understand.", intent="oov")
    return OrchestratorResult(handled=False, request=None, execution=None,
                              reject=rej, reply_text="I don't understand.")


def _low_conf_result() -> OrchestratorResult:
    rej = RejectResult(reason="low_confidence",
                       detail="confidence 0.400 < 0.750",
                       reply_text="I'm not sure I caught that. Could you repeat it?",
                       intent="")
    return OrchestratorResult(handled=False, request=None, execution=None,
                              reject=rej,
                              reply_text="I'm not sure I caught that. Could you repeat it?")


def _dry_run_result(intent: str = "turn_on_lights") -> OrchestratorResult:
    req = ActionRequest(intent=intent, category="lights", slots={},
                        action_code="lights.on", reply_text="I would turn the lights on.",
                        dry_run=True, confidence=0.99)
    return OrchestratorResult(handled=True, request=req, execution=None,
                              reject=None, reply_text="I would turn the lights on.")


class ReplyWavNameShapeTests(unittest.TestCase):
    def test_oov_via_orchestrator_result_selects_oov_wav(self):
        # The exact crash trigger: an OrchestratorResult, not a dict.
        self.assertEqual(reply_wav_name({"intent": "oov"}, _oov_result()), "oov.wav")

    def test_oov_via_legacy_dict_selects_oov_wav(self):
        self.assertEqual(
            reply_wav_name({"intent": "oov"},
                           {"status": "rejected", "code": "out_of_vocabulary"}),
            "oov.wav")

    def test_low_confidence_via_orchestrator_result_selects_oov_wav(self):
        self.assertEqual(reply_wav_name({"intent": "turn_on_lights"},
                                        _low_conf_result()), "oov.wav")

    def test_recognized_intent_via_orchestrator_result_selects_its_wav(self):
        self.assertEqual(
            reply_wav_name({"intent": "turn_on_lights"}, _dry_run_result()),
            "turn_on_lights.wav")

    def test_recognized_intent_via_legacy_dict_selects_its_wav(self):
        self.assertEqual(
            reply_wav_name({"intent": "turn_on_lights"},
                           {"status": "dry_run", "code": "lights.on"}),
            "turn_on_lights.wav")

    def test_none_action_falls_back_to_oov(self):
        self.assertEqual(reply_wav_name({"intent": "oov"}, None), "oov.wav")
        self.assertEqual(reply_wav_name(None, None), "oov.wav")


class BuildReplyShapeTests(unittest.TestCase):
    def test_oov_via_orchestrator_result_does_not_raise(self):
        # Previously raised AttributeError before this fix.
        reply = build_reply({"intent": "oov"}, _oov_result())
        self.assertTrue(reply["speak"])
        self.assertEqual(reply["text"], "I don't understand.")

    def test_oov_via_legacy_dict(self):
        reply = build_reply({"intent": "oov"},
                            {"status": "rejected", "code": "out_of_vocabulary"})
        self.assertEqual(reply["text"], "I don't understand.")

    def test_recognized_via_orchestrator_result(self):
        reply = build_reply({"intent": "turn_on_lights"}, _dry_run_result())
        self.assertIn("lights", reply["text"].lower())


class ReplyWavCoverageTests(unittest.TestCase):
    def test_every_mapped_intent_has_a_wav_entry(self):
        # Guard against a mapping pointing at a name that was never generated.
        for intent, wav in REPLY_WAV_BY_INTENT.items():
            self.assertTrue(wav.endswith(".wav"), f"{intent} -> {wav}")


if __name__ == "__main__":
    unittest.main()
