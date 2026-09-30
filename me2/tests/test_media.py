"""Unit tests for the live media (music) path.

Covers: track discovery/filtering, random pick with no-repeat, the
single-source-of-thought spoken reply, pause/resume/stop transport, the
no-player and empty-directory failures, and the dry-run / no-controller
fallbacks. No real audio is spawned -- a fake ``runner`` records the command
line and returns a stub process.
"""
from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from rpi5.facade import decode
from rpi5.media import (AUDIO_EXTS, MediaPlayerController, MediaError,
                        _pick_player)
from rpi5.executors.media import MediaExecutor
from rpi5.orchestrator import default_orchestrator


class FakeProc:
    """Stand-in for a spawned player process."""

    def __init__(self):
        self.returncode = None
        self.signals = []
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class FakeRunner:
    """Records the command line and hands back a :class:`FakeProc`."""

    def __init__(self):
        self.calls = []
        self.procs = []

    def __call__(self, cmd):
        self.calls.append(list(cmd))
        proc = FakeProc()
        self.procs.append(proc)
        return proc


def make_dir(tmp: Path, names=("a.mp3", "b.flac", "c.ogg", "notes.txt")) -> Path:
    for n in names:
        (tmp / n).write_bytes(b"x")
    return tmp


class DiscoverTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_filters_to_audio_extensions(self):
        make_dir(self.tmp)
        ctrl = MediaPlayerController(self.tmp, player="ffplay", runner=FakeRunner())
        names = {p.name for p in ctrl.discover()}
        self.assertEqual(names, {"a.mp3", "b.flac", "c.ogg"})
        self.assertNotIn("notes.txt", names)

    def test_empty_dir_returns_empty(self):
        ctrl = MediaPlayerController(self.tmp, player="ffplay", runner=FakeRunner())
        self.assertEqual(ctrl.discover(), [])

    def test_missing_dir_returns_empty(self):
        ctrl = MediaPlayerController(self.tmp / "nope", player="ffplay",
                                     runner=FakeRunner())
        self.assertEqual(ctrl.discover(), [])

    def test_uppercase_extension_matches(self):
        (self.tmp / "UP.MP3").write_bytes(b"x")
        ctrl = MediaPlayerController(self.tmp, player="ffplay", runner=FakeRunner())
        self.assertIn("UP.MP3", {p.name for p in ctrl.discover()})


class PickTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _dir(self, names):
        d = self.tmp / "tracks"
        d.mkdir()
        make_dir(d, names=names)
        return d

    def test_play_picks_from_directory(self):
        self.tmp = self._dir(("a.mp3", "b.flac", "c.ogg"))
        runner = FakeRunner()
        ctrl = MediaPlayerController(self.tmp, player="ffplay", runner=runner)
        name = ctrl.play()
        self.assertIn(name, {"a.mp3", "b.flac", "c.ogg"})
        self.assertEqual(ctrl.current_track, name)
        self.assertTrue(ctrl.is_playing)
        # command line is [ffplay, -nodisp, -autoexit, -loglevel, quiet, path]
        self.assertEqual(runner.calls[0][0], "ffplay")
        self.assertTrue(runner.calls[0][-1].endswith(name))

    def test_play_avoids_last_track(self):
        self.tmp = self._dir(("a.mp3", "b.flac", "c.ogg"))
        runner = FakeRunner()
        ctrl = MediaPlayerController(self.tmp, player="ffplay", runner=runner,
                                     rng=random.Random(0))
        first = ctrl.play()
        second = ctrl.play()
        self.assertNotEqual(second, first)
        # and the third must differ from the second
        third = ctrl.play()
        self.assertNotEqual(third, second)

    def test_single_track_repeats(self):
        self.tmp = self._dir(("solo.mp3",))
        runner = FakeRunner()
        ctrl = MediaPlayerController(self.tmp, player="ffplay", runner=runner)
        self.assertEqual(ctrl.play(), "solo.mp3")
        self.assertEqual(ctrl.play(), "solo.mp3")


class TransportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        make_dir(self.tmp, names=("a.mp3", "b.flac"))
        self.runner = FakeRunner()
        self.ctrl = MediaPlayerController(self.tmp, player="ffplay",
                                          runner=self.runner)

    def tearDown(self):
        self._tmp.cleanup()

    def test_pause_sends_sigstop(self):
        import signal
        self.ctrl.play()
        self.assertTrue(self.ctrl.pause())
        self.assertTrue(self.ctrl.is_paused)
        self.assertFalse(self.ctrl.is_playing)
        self.assertIn(signal.SIGSTOP, self.runner.procs[-1].signals)

    def test_resume_sends_sigcont(self):
        import signal
        self.ctrl.play()
        self.ctrl.pause()
        self.assertTrue(self.ctrl.resume())
        self.assertFalse(self.ctrl.is_paused)
        self.assertIn(signal.SIGCONT, self.runner.procs[-1].signals)

    def test_stop_terminates(self):
        self.ctrl.play()
        self.assertTrue(self.ctrl.stop())
        self.assertTrue(self.runner.procs[-1].terminated)
        self.assertFalse(self.ctrl.is_playing)
        self.assertIsNone(self.ctrl.current_track)

    def test_stop_when_idle_is_noop(self):
        self.assertFalse(self.ctrl.stop())

    def test_pause_when_idle_is_noop(self):
        self.assertFalse(self.ctrl.pause())

    def test_play_stops_previous(self):
        self.ctrl.play()
        self.ctrl.play()
        # the first process should have been terminated before the second started
        self.assertTrue(self.runner.procs[0].terminated)
        self.assertEqual(len(self.runner.calls), 2)


class FailureTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_empty_dir_raises(self):
        ctrl = MediaPlayerController(self.tmp, player="ffplay", runner=FakeRunner())
        with self.assertRaises(MediaError):
            ctrl.play()

    def test_no_player_raises(self):
        make_dir(self.tmp)
        ctrl = MediaPlayerController(self.tmp, player="definitely-not-a-real-player",
                                     runner=FakeRunner())
        # _pick_player with an explicit unknown binary still returns it; the
        # failure surfaces at spawn. Instead assert detection raises when none
        # of the known players exist and none is forced.
        with self.assertRaises(MediaError):
            _pick_player_forced_none()


def _pick_player_forced_none():
    """Simulate auto-detection finding nothing by monkeypatching shutil.which."""
    import rpi5.media as media_mod
    real = media_mod.shutil.which
    media_mod.shutil.which = lambda _name: None
    try:
        return media_mod._pick_player(None)
    finally:
        media_mod.shutil.which = real


class PlayerDetectionTests(unittest.TestCase):
    def test_ffplay_args(self):
        binary, args = _pick_player("ffplay")
        self.assertEqual(binary, "ffplay")
        self.assertIn("-nodisp", args)
        self.assertIn("-autoexit", args)

    def test_pw_play_args(self):
        binary, args = _pick_player("pw-play")
        self.assertEqual(binary, "pw-play")
        self.assertEqual(args, [])


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        make_dir(self.tmp, names=("a.mp3", "b.flac"))
        self.runner = FakeRunner()
        self.ctrl = MediaPlayerController(self.tmp, player="ffplay",
                                          runner=self.runner)

    def tearDown(self):
        self._tmp.cleanup()

    def _decode(self, intent):
        return decode(intent, {}, confidence=1.0, dry_run=False)

    def test_play_reply_names_track(self):
        ex = MediaExecutor(dry_run=False, player=self.ctrl)
        res = ex.run(self._decode("play_music"))
        self.assertTrue(res.ok)
        self.assertTrue(res.side_effects)
        self.assertTrue(res.detail.startswith("Playing "))
        self.assertEqual(res.payload["answer"], res.detail)

    def test_pause_then_stop(self):
        ex = MediaExecutor(dry_run=False, player=self.ctrl)
        ex.run(self._decode("play_music"))
        res = ex.run(self._decode("pause_music"))
        self.assertTrue(res.ok)
        self.assertEqual(res.detail, "Music paused")
        self.assertEqual(res.payload["answer"], "Music paused")
        res = ex.run(self._decode("stop_music"))
        self.assertTrue(res.ok)
        self.assertEqual(res.detail, "Music stopped")

    def test_stop_when_idle(self):
        ex = MediaExecutor(dry_run=False, player=self.ctrl)
        res = ex.run(self._decode("stop_music"))
        self.assertTrue(res.ok)
        self.assertEqual(res.detail, "No music is playing.")
        self.assertEqual(res.payload["answer"], "No music is playing.")

    def test_pause_when_idle(self):
        ex = MediaExecutor(dry_run=False, player=self.ctrl)
        res = ex.run(self._decode("pause_music"))
        self.assertTrue(res.ok)
        self.assertEqual(res.detail, "No music is playing.")

    def test_play_when_no_music_available(self):
        """Empty music directory: play is acknowledged (ok) with a canned
        reply, not the generic 'Sorry, that didn't work.' apology."""
        empty = self.tmp / "empty"
        empty.mkdir()
        ctrl = MediaPlayerController(directory=empty, runner=self.runner)
        ex = MediaExecutor(dry_run=False, player=ctrl)
        res = ex.run(self._decode("play_music"))
        self.assertTrue(res.ok)
        self.assertIn("couldn't find any music", res.detail)
        self.assertEqual(res.payload["answer"], res.detail)

    def test_no_controller_fails_soft(self):
        ex = MediaExecutor(dry_run=False, player=None)
        res = ex.run(self._decode("play_music"))
        self.assertFalse(res.ok)
        self.assertEqual(res.detail, "no media player attached")

    def test_dry_run_has_no_side_effects(self):
        ex = MediaExecutor(dry_run=True, player=self.ctrl)
        res = ex.run(self._decode("play_music"))
        self.assertTrue(res.ok)
        self.assertFalse(res.side_effects)
        self.assertEqual(self.runner.calls, [])  # nothing spawned

    def test_pipeline_routes_media_live(self):
        orch = default_orchestrator(dry_run=False, media_player=self.ctrl)
        res = orch.run(self._decode("play_music"))
        self.assertTrue(res.handled)
        self.assertTrue(res.event["side_effects"])


if __name__ == "__main__":
    unittest.main()
