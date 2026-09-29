"""Regression tests for the microphone echo guard (rpi5/audio.py).

The assistant must not recognise its own voice. While the assistant is
producing audio (a TTS reply, an alarm, a timer ding) the capture loop closes
the mic: every incoming frame is dropped before the VAD / wake-word detector
can latch onto it. Without this guard a spoken reply is re-captured, re-
recognised as a second command, and surfaces as a spurious "could you repeat
it?" plus a double reply.

These tests drive ``microphone_utterances`` with a scripted frame stream (no
real microphone, no sounddevice) by patching the module-level capture helpers,
mirroring the injected-callable pattern in the other rpi5 tests.

Because the capture loop is ``while True`` (a generator that can never be
cleanly unwound once it blocks on an exhausted stream), the driver runs it in
a daemon thread and reads yielded utterances from a bounded queue. The test
process exits with the daemon, so nothing hangs.
"""
from __future__ import annotations

import queue
import threading
import time
import unittest
from unittest import mock

import numpy as np

from rpi5 import audio
from rpi5.audio import VADConfig, microphone_utterances


def _frame(amplitude: float, rate: int = 16_000, ms: int = 30) -> np.ndarray:
    """A constant-amplitude mono frame at the VAD sample rate."""
    n = rate * ms // 1000
    return (np.ones(n, dtype=np.float32) * amplitude)


def _speech_then_gap(speech: int = 30, gap: int = 20) -> list[np.ndarray]:
    """``speech`` loud frames followed by ``gap`` silent frames.

    The loud run opens the VAD; the silent run (below ``stop_rms``) accumulates
    ``silence_frames`` quiet frames and closes the utterance. Without the gap
    the VAD would just run on until ``max_utterance_s``.
    """
    return ([_frame(0.2) for _ in range(speech)]
            + [_frame(0.0) for _ in range(gap)])


class _FakeStream:
    """Finite, closeable frame stream standing in for the pipewire capture.

    Exhaustion blocks on an interruptible wait (it never raises StopIteration,
    which would surface as a RuntimeError through the generator's ``while
    True`` loop per PEP 479). The driver reads from a queue and never relies on
    the stream ending.
    """

    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0
        self._stop = threading.Event()

    def __iter__(self):
        return self

    def __next__(self):
        if self._i < len(self._frames):
            f = self._frames[self._i]
            self._i += 1
            return f
        self._stop.wait(3600)
        raise StopIteration

    def close(self):
        self._stop.set()


def _drive(tts_active, frames, expect_utterances: int, timeout_s: float = 5.0,
           post_reply_gate=None):
    """Run microphone_utterances in a daemon thread; collect utterances.

    ``expect_utterances == 0`` asserts the guard dropped everything (waits the
    full timeout, expecting none). Otherwise waits for that many.
    """
    stream = _FakeStream(frames)
    out: "queue.Queue[np.ndarray]" = queue.Queue()

    def worker():
        with mock.patch.object(audio, "_pipewire_capture",
                               lambda settings: stream), \
             mock.patch.object(audio.shutil, "which",
                               lambda name: "/usr/bin/pw-record"
                               if name == "pw-record" else None):
            try:
                for utt in microphone_utterances(
                        config=VADConfig(),
                        wakeword=None,          # no wake word -> always listening
                        tts_active=tts_active,
                        post_reply_gate=post_reply_gate,
                        command_timeout_s=7.0):
                    out.put(utt)
            except Exception:  # noqa: BLE001 - daemon; surface nothing
                pass

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    got: list[np.ndarray] = []
    deadline = time.monotonic() + timeout_s
    while len(got) < expect_utterances:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            got.append(out.get(timeout=min(remaining, 0.25)))
        except queue.Empty:
            continue
    return got


class TtsEchoGuardTests(unittest.TestCase):
    def test_frames_dropped_while_tts_active(self):
        """Loud frames arriving while tts_active() is True must never be
        captured into an utterance -- the mic is closed."""
        loud = _speech_then_gap()
        flag = threading.Event()
        flag.set()                                 # assistant is "speaking"

        got = _drive(flag.is_set, loud, expect_utterances=0, timeout_s=1.5)

        self.assertEqual(got, [],
                         "frames during active TTS must be dropped, not "
                         "recognised as a second command")

    def test_frames_captured_when_tts_inactive(self):
        """Sanity: with TTS inactive the same loud frames DO form an
        utterance, proving the guard is what suppresses them."""
        loud = _speech_then_gap()
        flag = threading.Event()                  # not set -> mic open

        got = _drive(flag.is_set, loud, expect_utterances=1, timeout_s=3.0)

        self.assertEqual(len(got), 1,
                         "with TTS inactive the loud frames should be captured")
        self.assertGreaterEqual(got[0].shape[0], 1)

    def test_guard_toggles_mid_stream(self):
        """Frames before TTS starts are captured; the loop resumes cleanly
        after TTS ends (the pre-TTS utterance is still delivered)."""
        flag = threading.Event()
        frames = []
        for i in range(160):
            frames.append(_frame(0.2 if i < 30 else 0.0))
            if i == 60:
                flag.set()        # assistant starts speaking mid-stream
            if i == 100:
                flag.clear()      # assistant finishes speaking

        got = _drive(flag.is_set, frames, expect_utterances=1, timeout_s=3.0)

        self.assertEqual(len(got), 1,
                         "the pre-TTS utterance must still be captured")


class PostReplyGateTests(unittest.TestCase):
    """Regression tests for the post-reply echo guard.

    After a reply finishes, ``tts_active`` clears but the room keeps
    echoing the reply for a couple of seconds. That lingering echo is what
    re-latches the wake detector into a phantom wake ("Alexa detected"
    with no user speech) followed by an out-of-vocabulary reply. The
    post-reply gate holds the mic closed for a short window after every
    reply so the trailing echo cannot trigger a second interaction.

    The gate is driven by a *frame counter*, not wall-clock time: the fake
    capture stream is unbounded and drains in microseconds, so a fixed
    sleep would outpace it. Counting frames makes the test deterministic
    while still exercising the real gate path in the capture loop.
    """

    def test_frames_dropped_while_post_reply_gate_active(self):
        """Loud frames arriving while the post-reply gate is armed must
        never be captured into an utterance -- the mic stays closed even
        though tts_active is already cleared."""
        loud = _speech_then_gap()
        tts_flag = threading.Event()            # not set -> tts_active False
        gate = threading.Event()
        gate.set()                              # post-reply window armed

        got = _drive(tts_flag.is_set, loud, expect_utterances=0, timeout_s=1.5,
                     post_reply_gate=gate.is_set)

        self.assertEqual(got, [],
                         "lingering echo during the post-reply guard must be "
                         "dropped, not re-recognised as a second command")

    def test_frames_captured_after_post_reply_gate_expires(self):
        """Once the gate releases the mic reopens and the loud frames DO
        form an utterance, proving the gate (not the silence) is what
        suppressed them."""
        # Silent lead-in (VAD idle while the gate is armed), then a loud
        # phase, then a silence gap that closes the utterance.
        frames = ([_frame(0.0) for _ in range(20)]
                  + [_frame(0.2) for _ in range(30)]
                  + [_frame(0.0) for _ in range(20)])
        tts_flag = threading.Event()            # not set -> tts_active False
        # Arm the gate for the first 25 frames (covering the silent
        # lead-in and the start of the loud phase), then release it so the
        # remaining loud frames reach the VAD and form an utterance.
        counter = {"n": 0}

        def gate():
            counter["n"] += 1
            return counter["n"] <= 25

        got = _drive(tts_flag.is_set, frames, expect_utterances=1, timeout_s=4.0,
                     post_reply_gate=gate)

        self.assertEqual(len(got), 1,
                         "after the guard releases the loud frames should be captured")


if __name__ == "__main__":
    unittest.main()
