"""Replay-first runtime harness for Pi5-VCM.

The harness owns the runtime contract around the model: recognition, a
confidence/OOV gate, dry-run action dispatch, and structured events. Audio
capture and real device integrations can be added as adapters later.

Inference backend
-----------------
The model runs on ONNX Runtime (``vcm_model_int8.onnx``) via
:mod:`inference.ort_infer`, which needs only ``numpy`` + ``onnxruntime``.
This is what makes the harness runnable on the Raspberry Pi 5, whose venv
ships exactly those two packages and nothing else (no torch / torchaudio).
The feature used on-device is a faithful numpy port of the ``kaldi.fbank``
the model was trained on, so intents/slots match the torch path.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from inference.ort_infer import (
    load_session,
    resolve_checkpoint,
    run_utterance,
    self_test_wavs,
    warmup,
)
from inference.features import load_wav_mono
from .replies import build_reply
from .facade import ActionRequest, RejectResult, decode
from .orchestrator import Orchestrator, OrchestratorResult, default_orchestrator


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
    "stop_timer": "timer.cancel",
    "remind": "reminder.create",
    "call": "call.request",
    "what_time": "query.time",
    "what_weather": "query.weather",
    "what_reminders": "query.reminders",
    "volume_up": "volume.up",
    "volume_down": "volume.down",
}


@dataclass(frozen=True)
class HarnessConfig:
    checkpoint: str = "vcm_model_int8.onnx"
    intent_labels_path: str | None = None
    device: str = "cpu"
    max_frames: int = 400
    min_confidence: float = 0.75
    warmup: int = 1
    threads: int = 2


class DryRunDispatcher:
    """Allowlisted action dispatcher with no external side effects."""

    def __init__(self, timer_manager: Any | None = None) -> None:
        self.timer_manager = timer_manager

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
        if intent == "set_timer" and self.timer_manager is not None:
            try:
                return self.timer_manager.start(
                    slots.get("duration"), slots.get("duration_unit", "minute")
                )
            except ValueError:
                return self._rejected("invalid_timer", "invalid_timer_slots")
        if intent == "stop_timer" and self.timer_manager is not None:
            return self.timer_manager.cancel()
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


class FacadePipeline:
    """Facade + orchestrator wired together as the new execution path.

    Replaces the inline gate/slot logic in :class:`DryRunDispatcher` with the
    pure :func:`~rpi5.facade.decode` + :class:`~rpi5.orchestrator.Orchestrator`.
    The model still emits only (intent, slots, confidence); everything from the
    confidence gate onward lives here.
    """

    def __init__(self, orchestrator: Orchestrator | None = None,
                 *, threshold: float = 0.75, dry_run: bool = True,
                 weather_fn: "Callable[[], str] | None" = None,
                 timer_manager: Any | None = None,
                 timer_alarm: Any | None = None,
                 reminder_store: Any | None = None,
                 volume_controller: Any | None = None,
                 media_player: Any | None = None,
                 media_volume: int | None = None,
                 media_before_play: Callable[[str], bool] | None = None,
                 light_driver: Any | None = None,
                 dialer: Any | None = None) -> None:
        self.orchestrator = orchestrator or default_orchestrator(
            dry_run=dry_run, weather_fn=weather_fn, timer_manager=timer_manager,
            timer_alarm=timer_alarm,
            reminder_store=reminder_store, volume_controller=volume_controller,
            media_player=media_player, media_volume=media_volume,
            media_before_play=media_before_play,
            light_driver=light_driver,
            dialer=dialer)
        self.threshold = threshold
        self.dry_run = dry_run

    def process(self, result: dict[str, Any], source: str = "microphone") -> OrchestratorResult:
        intent = result.get("intent", "oov")
        confidence = float(result.get("intent_confidence", 0.0))
        slots = result.get("slots") or {}
        decoded = decode(intent, slots, confidence,
                         threshold=self.threshold, dry_run=self.dry_run)
        return self.orchestrator.run(decoded, source=source)


class PiHarness:
    """Load the VCM (ONNX) once and process replayed WAVs or self-tests."""

    def __init__(self, config: HarnessConfig,
                 dispatcher: DryRunDispatcher | None = None,
                 pipeline: FacadePipeline | None = None) -> None:
        self.config = config
        self.device = "cpu"  # ONNX Runtime CPU execution provider
        checkpoint = resolve_checkpoint(config.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        self.checkpoint = checkpoint
        self.session, self._in_name, self.intents, self.ctc_vocab = load_session(
            str(checkpoint), config.threads)
        outputs = self.session.get_outputs()
        self.intent_only = len(outputs) == 1
        if self.intent_only:
            if outputs[0].name != "intent_logits":
                raise ValueError(
                    f"unsupported intent-only output: {outputs[0].name}")
            if config.intent_labels_path is None:
                raise ValueError(
                    "intent-only ONNX requires --intent-labels for diagnostics")
            labels_path = Path(config.intent_labels_path).expanduser()
            if not labels_path.is_file():
                raise FileNotFoundError(f"intent labels not found: {labels_path}")
            label_data = json.loads(labels_path.read_text(encoding="utf-8"))
            if isinstance(label_data, dict):
                if label_data.get("task") != "intent_classification_only":
                    raise ValueError("intent label file has an unsupported task")
                labels = label_data.get("labels")
            else:
                labels = label_data
            if (not isinstance(labels, list)
                    or not all(isinstance(label, str) for label in labels)):
                raise ValueError("intent label file must contain a string label list")
            output_size = outputs[0].shape[-1]
            if isinstance(output_size, int) and len(labels) != output_size:
                raise ValueError(
                    f"intent label count {len(labels)} does not match model output "
                    f"size {output_size}")
            if len(labels) != len(set(labels)):
                raise ValueError("intent label file contains duplicate labels")
            self.intents = labels
            self.ctc_vocab = []
        elif config.intent_labels_path is not None:
            raise ValueError("--intent-labels is only for single-output intent models")
        self.dispatcher = dispatcher or DryRunDispatcher()
        self.pipeline = pipeline or FacadePipeline(threshold=config.min_confidence)
        self._warmup()

    def _warmup(self) -> None:
        if self.config.warmup <= 0:
            return
        warmup(self.session, self.config.warmup)

    def recognize_wav(self, path: str | Path) -> dict[str, Any]:
        wav = load_wav_mono(str(path))
        return self.recognize_audio(wav, Path(path).name)

    def recognize_audio(self, wav: Any, source: str = "microphone") -> dict[str, Any]:
        """Recognize one mono float32 waveform without a temporary WAV."""
        wav = np.asarray(wav, dtype=np.float32).ravel()
        result = run_utterance(
            self.session,
            wav,
            self.config.max_frames,
            self.intents,
            self.ctc_vocab,
        )
        return self._event(source, result)

    def recognize_and_act(self, wav: Any, source: str = "microphone") -> OrchestratorResult:
        """Recognize + run the full facade/orchestrator pipeline in one call."""
        if self.intent_only:
            raise RuntimeError("intent-only model is diagnostics-only; actions are disabled")
        wav = np.asarray(wav, dtype=np.float32).ravel()
        result = run_utterance(self.session, wav, self.config.max_frames,
                               self.intents, self.ctc_vocab)
        return self.pipeline.process(result, source=source)

    def recognize_self_test(self) -> list[dict[str, Any]]:
        events = []
        for name, wav in self_test_wavs():
            result = run_utterance(
                self.session,
                wav,
                self.config.max_frames,
                self.intents,
                self.ctc_vocab,
            )
            events.append(self._event(name, result))
        return events

    def _event(self, source: str, result: dict[str, Any]) -> dict[str, Any]:
        result = {**result, "min_confidence": self.config.min_confidence}
        if self.intent_only:
            action = {
                "status": "diagnostic_only",
                "action": None,
                "side_effects": False,
            }
            reply = {
                "text": f"Intent-only model predicted {result['intent']}.",
                "speak": False,
                "source": "diagnostic",
            }
        else:
            action = self.dispatcher.dispatch(result)
            reply = build_reply(result, action)
        return {
            "event": "command_processed",
            "request_id": uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "device": str(self.device),
            "checkpoint": str(self.checkpoint),
            "result": result,
            "action": action,
            "reply": reply,
        }


def event_json(event: dict[str, Any]) -> str:
    """Serialize an event for CLI output or a future local service."""
    return json.dumps(event, sort_keys=True)
