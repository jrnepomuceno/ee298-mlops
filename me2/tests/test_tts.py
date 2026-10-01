"""Tests for playback command construction without touching audio devices."""
from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from rpi5.tts import WavPlayer


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

    @patch("rpi5.tts.subprocess.run")
    def test_play_uses_original_wav_without_pre_roll(self, run: Mock) -> None:
        result = WavPlayer("pw-play").play("reply.wav")
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["pw-play", "reply.wav"])
        self.assertEqual(result["wav"], "reply.wav")
        self.assertNotIn("preroll_ms", result)

    @patch("rpi5.tts.subprocess.Popen")
    def test_play_async_uses_original_wav_without_pre_roll(
            self, popen: Mock) -> None:
        popen.return_value = Mock()
        player = WavPlayer("pw-play")
        result = player.play_async("reply.wav")
        self.assertEqual(popen.call_args.args[0], ["pw-play", "reply.wav"])
        self.assertEqual(result["wav"], "reply.wav")
        self.assertNotIn("preroll_ms", result)
        player.close()


if __name__ == "__main__":
    unittest.main()
