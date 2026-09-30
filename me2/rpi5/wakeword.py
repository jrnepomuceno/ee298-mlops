"""Optional pretrained wake-word adapter for the Pi microphone path."""
from __future__ import annotations

import threading
import time
from typing import Any, Callable
from pathlib import Path

import numpy as np


class OpenWakeWordDetector:
    """Detect a pretrained openWakeWord model from 16 kHz audio frames."""

    def __init__(self, model_name: str = "alexa", threshold: float = 0.7,
                 on_score: Callable[[float], None] | None = None,
                 cooldown_s: float = 2.0) -> None:
        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError as exc:
            raise RuntimeError(
                "wake-word mode requires openwakeword; install "
                "requirements-wakeword.txt"
            ) from exc
        paths = openwakeword.get_pretrained_model_paths()
        matching = [path for path in paths
                    if Path(path).stem.startswith(model_name)]
        if not matching:
            raise RuntimeError(
                f"pretrained wake-word model not found: {model_name}")
        self.model_name = Path(matching[0]).stem
        self.threshold = threshold
        self.model = Model(wakeword_model_paths=[matching[0]])
        self._buffer = np.zeros(0, dtype=np.int16)
        self._consecutive_hits = 0
        self.on_score = on_score
        self._lock = threading.Lock()
        # Inter-wake cooldown: after a detection fires, ignore further
        # detections for cooldown_s seconds. This prevents the user's own
        # "Alexa" echo (or a quick double-speak) from re-triggering the
        # detector before the assistant has finished acknowledging.
        self.cooldown_s = cooldown_s
        self._last_fire_time = 0.0

    def accepts(self, frame: np.ndarray) -> bool:
        """Return true when the configured model crosses its threshold."""
        with self._lock:
            # Ignore wake scores during cooldown, but keep the audio buffer bounded.
            if time.monotonic() - self._last_fire_time < self.cooldown_s:
                pcm = np.clip(np.asarray(frame, dtype=np.float32).reshape(-1),
                              -1.0, 1.0)
                self._buffer = np.concatenate(
                    (self._buffer, (pcm * 32767.0).astype(np.int16)))
                if self._buffer.size >= 1280:
                    self._buffer = self._buffer[1280:]
                self._consecutive_hits = 0
                return False

            frame = np.asarray(frame, dtype=np.float32).reshape(-1)
            pcm = np.clip(frame, -1.0, 1.0)
            pcm = (pcm * 32767.0).astype(np.int16)
            self._buffer = np.concatenate((self._buffer, pcm))
            if self._buffer.size < 1280:
                return False
            window = self._buffer[:1280]
            self._buffer = self._buffer[1280:]
            scores: dict[str, Any] = self.model.predict(window)
            score = float(scores.get(self.model_name, 0.0))
            if self.on_score is not None:
                self.on_score(score)

            if score >= self.threshold:
                self._consecutive_hits += 1
            else:
                self._consecutive_hits = 0

            # Require consecutive hits and enforce cooldown after detection
            detected = self._consecutive_hits >= 2
            if detected:
                self._last_fire_time = time.monotonic()
                self._consecutive_hits = 0  # Reset after successful detection

            return detected

    def reset(self) -> None:
        """Forget audio buffered during a response."""
        with self._lock:
            self._buffer = np.zeros(0, dtype=np.int16)
            self._consecutive_hits = 0
            self.model.reset()
            # Re-arm the cooldown so the next legitimate wake can be accepted.
            self._last_fire_time = 0.0
