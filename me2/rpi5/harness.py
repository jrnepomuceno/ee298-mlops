"""Replay-first runtime harness for Pi5-VCM.

The harness owns the runtime contract around the model: recognition, a
confidence/OOV gate, dry-run action dispatch, and structured events. Audio
capture and real device integrations can be added as adapters later.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from inference.infer import (
    load_model,
    resolve_checkpoint,
    run_utterance,
    self_test_wavs,
)
from inference.utils import audio_utils
from inference.utils.model_utils import get_device
from .replies import build_reply


ACTION_BY_INTENT = {
    "turn_on_lights": "lights.on",
    "turn_off_lights": "lights.off",
    "dim_lights": "lights.dim",
    "set_temperature": "thermostat.set",
    "play_music": "media.play",
    "pause_music": "media.pause",
    "stop_music": "media.stop",
    "set_timer": "timer.set",
    "set_alarm": "alarm.set",
    "cancel_timer": "timer.cancel",
    "remind": "reminder.create",
    "call": "call.request",
    "what_time": "query.time",
    "what_weather": "query.weather",
    "what_reminders": "query.reminders",
}


@dataclass(frozen=True)
class HarnessConfig:
    checkpoint: str = "inference/best.pt"
    device: str = "auto"
    max_frames: int = 400
    min_confidence: float = 0.75
    warmup: int = 1


class DryRunDispatcher:
    """Allowlisted action dispatcher with no external side effects."""

    def dispatch(self, result: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(result, dict):
            return self._rejected("invalid_result", "invalid_result")

        intent = result.get("intent")
        confidence = float(result.get("intent_confidence", 0.0))
        min_confidence = float(result.get("min_confidence", 0.0))
        slots = result.get("slots") or {}

        if intent == "oov":
            return self._rejected("oov", "out_of_vocabulary")
        if confidence < min_confidence:
            return self._rejected("low_confidence", "confidence_below_threshold")

        action = ACTION_BY_INTENT.get(intent)
        if action is None:
            return self._rejected("unsupported", "intent_not_allowlisted")
        return {
            "status": "dry_run",
            "action": action,
            "slots": slots,
            "side_effects": False,
        }

    @staticmethod
    def _rejected(reason: str, code: str) -> dict[str, Any]:
        return {
            "status": "rejected",
            "action": None,
            "reason": reason,
            "code": code,
            "side_effects": False,
        }


class PiHarness:
    """Load the VCM once and process replayed WAVs or self-test utterances."""

    def __init__(self, config: HarnessConfig,
                 dispatcher: DryRunDispatcher | None = None) -> None:
        self.config = config
        self.device = get_device(config.device)
        checkpoint = resolve_checkpoint(config.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        self.checkpoint = checkpoint
        self.model, self.intents, self.ctc_vocab, _ = load_model(
            str(checkpoint), self.device)
        self.dispatcher = dispatcher or DryRunDispatcher()
        self._warmup()

    def _warmup(self) -> None:
        if self.config.warmup <= 0:
            return
        dummy = torch.zeros(1, 16, 80, device=self.device)
        with torch.no_grad():
            for _ in range(self.config.warmup):
                self.model(dummy)

    def recognize_wav(self, path: str | Path) -> dict[str, Any]:
        wav = audio_utils.load_wav_mono(str(path))
        return self.recognize_audio(wav, Path(path).name)

    def recognize_audio(self, wav: Any, source: str = "microphone") -> dict[str, Any]:
        """Recognize one mono float32 waveform without creating a temporary WAV."""
        if not isinstance(wav, torch.Tensor):
            wav = torch.from_numpy(wav)
        wav = wav.float().reshape(-1)
        result = run_utterance(
            self.model,
            wav,
            self.device,
            self.config.max_frames,
            self.intents,
            self.ctc_vocab,
        )
        return self._event(source, result)

    def recognize_self_test(self) -> list[dict[str, Any]]:
        events = []
        for name, wav in self_test_wavs():
            result = run_utterance(
                self.model,
                wav,
                self.device,
                self.config.max_frames,
                self.intents,
                self.ctc_vocab,
            )
            events.append(self._event(name, result))
        return events

    def _event(self, source: str, result: dict[str, Any]) -> dict[str, Any]:
        result = {**result, "min_confidence": self.config.min_confidence}
        action = self.dispatcher.dispatch(result)
        return {
            "event": "command_processed",
            "request_id": uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "device": str(self.device),
            "checkpoint": str(self.checkpoint),
            "result": result,
            "action": action,
            "reply": build_reply(result, action),
        }


def event_json(event: dict[str, Any]) -> str:
    """Serialize an event for CLI output or a future local service."""
    return json.dumps(event, sort_keys=True)
