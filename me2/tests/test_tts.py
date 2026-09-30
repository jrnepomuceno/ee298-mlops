"""Tests for playback command construction without touching audio devices."""
from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from rpi5.tts import PRE_ROLL_MS, WavPlayer, _with_silence_preroll


class WavPlayerCommandTests(unittest.TestCase):
    def test_pipewire_uses_supported_options(self) -> None:
        player = WavPlayer("pw-play")
        self.assertEqual(player._command("reply.wav"), ["pw-play", "reply.wav"])

    def test_pipewire_target_uses_target_option(self) -> None:
        player = WavPlayer("pw-play", "WILLEN")
        self.assertEqual(player._command("reply.wav"),
                         ["pw-play", "--target=WILLEN", "reply.wav"])

    def test_aplay_keeps_quiet_and_device_options(self) -> None:
        player = WavPlayer("aplay", "plughw:2,0")
        self.assertEqual(player._command("reply.wav"),
                         ["aplay", "-q", "-D", "plughw:2,0", "reply.wav"])

    def test_afplay_has_no_linux_quiet_option(self) -> None:
        player = WavPlayer("afplay")
        self.assertEqual(player._command("reply.wav"), ["afplay", "reply.wav"])


class WavPreRollTests(unittest.TestCase):
    def _check_padded(self, sample_width: int, expected_silence: bytes) -> None:
        channels = 2
        sample_rate = 1000
        source_frames = 4
        frame_bytes = channels * sample_width
        original = bytes([0x11]) * (source_frames * frame_bytes)
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.wav"
            with wave.open(str(source_path), "wb") as source:
                source.setnchannels(channels)
                source.setsampwidth(sample_width)
                source.setframerate(sample_rate)
                source.writeframes(original)

            padded_path = _with_silence_preroll(str(source_path))
            try:
                with wave.open(padded_path, "rb") as padded:
                    silence_frames = sample_rate * PRE_ROLL_MS // 1000
                    self.assertEqual(padded.getnchannels(), channels)
                    self.assertEqual(padded.getsampwidth(), sample_width)
                    self.assertEqual(padded.getframerate(), sample_rate)
                    self.assertEqual(padded.getnframes(), silence_frames + source_frames)
                    data = padded.readframes(padded.getnframes())
                silence_bytes = silence_frames * frame_bytes
                self.assertEqual(data[:silence_bytes], expected_silence * silence_frames * channels)
                self.assertEqual(data[silence_bytes:], original)
            finally:
                Path(padded_path).unlink(missing_ok=True)

    def test_prepends_configured_silence_to_16bit_pcm(self):
        self._check_padded(sample_width=2, expected_silence=b"\x00\x00")

    def test_prepends_centered_silence_to_unsigned_8bit_pcm(self):
        self._check_padded(sample_width=1, expected_silence=b"\x80")


if __name__ == "__main__":
    unittest.main()
