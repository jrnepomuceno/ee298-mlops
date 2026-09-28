"""Unit tests for the live call path (baresip dialer).

Covers:

* :func:`normalize_target` -- the safe-SIP-charset gate that keeps a stray
  transcript from injecting commands into the baresip daemon.
* the pluggable :class:`Dialer` registry and :func:`make_dialer` factory
  (unknown name -> dry-run, empty name -> None).
* :class:`BaresipDialer` -- the degrade-when-absent behaviour, the lazy
  daemon start, the ``dial``/``cancel`` command lines written to the daemon's
  stdin, and ``close`` teardown. The daemon is faked with a stub Popen whose
  stdin is a real pipe, exactly like the light tests patch ``subprocess.run``.
* :class:`CommsExecutor` -- the ``call.request`` mapping (contact -> target ->
  spoken reply), the no-dialer and unsupported-action failures, and the dry-run
  fallback. No real network is touched.

Finally, a full-stack test drives the real ``default_orchestrator`` (the exact
object ``run.py`` builds) and asserts the spoken reply for a successful call.
"""
from __future__ import annotations

import io
import unittest
from unittest import mock

from rpi5.facade import decode, RejectResult
from rpi5.calls import (BaresipDialer, Dialer, DIALERS, make_dialer,
                        normalize_target)
from rpi5.executors.comms import CommsExecutor
from rpi5.orchestrator import default_orchestrator


def _req(intent: str, slots: dict | None = None, confidence: float = 0.95):
    """Decode an intent the way the pipeline does, asserting it was accepted."""
    decoded = decode(intent, slots or {}, confidence)
    assert not isinstance(decoded, RejectResult), f"{intent} unexpectedly rejected"
    return decoded


class _FakeProc:
    """Stand-in for a baresip Popen: a real pipe for stdin, a controllable exit."""

    def __init__(self, alive: bool = True) -> None:
        self.stdin = io.BytesIO()
        self._alive = alive

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self._alive = False

    def kill(self):
        self._alive = False

    def wait(self, timeout=None):
        return 0


class TestNormalizeTarget(unittest.TestCase):
    def test_strips_and_collapses_spaces(self):
        self.assertEqual(normalize_target("555 1234"), "5551234")
        self.assertEqual(normalize_target("  +15551234567  "), "+15551234567")

    def test_feature_codes_allowed(self):
        self.assertEqual(normalize_target("*67"), "*67")
        self.assertEqual(normalize_target("123#"), "123#")

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            normalize_target("   ")

    def test_rejects_injection_chars(self):
        # The charset gate blocks command-separator / shell metacharacters and
        # non-SIP alphanumerics. (Whitespace is *stripped*, not rejected -- a
        # spaced number is legitimate; the daemon only ever sees the cleaned
        # token.)
        for bad in ("mom", "555;rm", "a b c", "555\"x", "555&", "555|x", "555`x"):
            with self.assertRaises(ValueError, msg=f"{bad!r} should be rejected"):
                normalize_target(bad)

    def test_newline_is_stripped_not_smuggled(self):
        # A trailing newline (e.g. a transcript line break) is dropped, leaving
        # a valid number -- it never reaches the daemon as a command separator.
        self.assertEqual(normalize_target("555\n"), "555")


class TestMakeDialer(unittest.TestCase):
    def test_empty_name_returns_none(self):
        self.assertIsNone(make_dialer(""))
        self.assertIsNone(make_dialer(None))

    def test_unknown_name_returns_none(self):
        self.assertIsNone(make_dialer("magic-dialer"))

    def test_baresip_registered(self):
        self.assertIn("baresip", DIALERS)
        d = make_dialer("baresip", config="/etc/baresip.conf")
        self.assertIsInstance(d, BaresipDialer)
        self.assertEqual(d.config, "/etc/baresip.conf")


class TestBaresipDialer(unittest.TestCase):
    def _dialer(self, executable: str = "/opt/bin/baresip",
                config: str | None = None) -> BaresipDialer:
        return BaresipDialer(executable=executable, config=config)

    def test_available_false_when_absent(self):
        d = self._dialer(executable="/nonexistent/baresip")
        self.assertFalse(d.available())

    def test_dial_raises_when_absent(self):
        d = self._dialer(executable="/nonexistent/baresip")
        with self.assertRaises(RuntimeError):
            d.dial("5551234")

    def test_dial_starts_daemon_and_writes_command(self):
        d = self._dialer(config="/etc/baresip.conf")
        fake = _FakeProc()
        with mock.patch.object(d, "available", return_value=True), \
             mock.patch("rpi5.calls.time.sleep"), \
             mock.patch("rpi5.calls.subprocess.Popen", return_value=fake) as popen:
            d.dial("5551234")
        # Daemon launched once with the config.
        self.assertEqual(popen.call_count, 1)
        cmd = popen.call_args[0][0]
        self.assertEqual(cmd[:1], ["/opt/bin/baresip"])
        self.assertIn("--config", cmd)
        self.assertIn("/etc/baresip.conf", cmd)
        # The dial command went to the daemon's stdin.
        self.assertEqual(fake.stdin.getvalue(), b"dial 5551234\n")

    def test_second_dial_reuses_daemon(self):
        d = self._dialer()
        fake = _FakeProc()
        with mock.patch.object(d, "available", return_value=True), \
             mock.patch("rpi5.calls.time.sleep"), \
             mock.patch("rpi5.calls.subprocess.Popen", return_value=fake) as popen:
            d.dial("111")
            d.dial("222")
        self.assertEqual(popen.call_count, 1)  # started once
        self.assertEqual(fake.stdin.getvalue(), b"dial 111\ndial 222\n")

    def test_cancel_writes_cancel(self):
        d = self._dialer()
        fake = _FakeProc()
        with mock.patch.object(d, "available", return_value=True), \
             mock.patch("rpi5.calls.time.sleep"), \
             mock.patch("rpi5.calls.subprocess.Popen", return_value=fake):
            d.dial("555")
            d.cancel()
        self.assertEqual(fake.stdin.getvalue(), b"dial 555\ncancel\n")

    def test_close_tears_down(self):
        d = self._dialer()
        fake = _FakeProc()
        with mock.patch.object(d, "available", return_value=True), \
             mock.patch("rpi5.calls.time.sleep"), \
             mock.patch("rpi5.calls.subprocess.Popen", return_value=fake):
            d.dial("555")
        d.close()
        self.assertTrue(fake.stdin.closed)
        self.assertFalse(fake._alive)
        self.assertIsNone(d._proc)

    def test_close_is_idempotent(self):
        d = self._dialer()
        d.close()  # no daemon running
        d.close()  # safe to repeat


class TestCommsExecutor(unittest.TestCase):
    def _executor(self, dialer: Dialer | None = None, dry_run: bool = False):
        return CommsExecutor(dry_run=dry_run, dialer=dialer)

    def test_no_dialer_fails_soft(self):
        res = self._executor(dialer=None, dry_run=False).run(_req("call", {"contact": "5551234"}))
        self.assertFalse(res.ok)
        self.assertIn("no dialer", res.detail)
        self.assertFalse(res.side_effects)

    def test_dry_run_no_side_effects(self):
        res = self._executor(dialer=BaresipDialer(executable="/x"), dry_run=True) \
            .run(_req("call", {"contact": "5551234"}))
        self.assertTrue(res.ok)
        self.assertFalse(res.side_effects)
        self.assertIn("dry-run", res.detail)

    def test_live_call_places_and_reports(self):
        d = BaresipDialer(executable="/opt/bin/baresip")
        fake = _FakeProc()
        with mock.patch.object(d, "available", return_value=True), \
             mock.patch("rpi5.calls.time.sleep"), \
             mock.patch("rpi5.calls.subprocess.Popen", return_value=fake):
            res = self._executor(dialer=d, dry_run=False).run(
                _req("call", {"contact": "555 1234"}))
        self.assertTrue(res.ok)
        self.assertTrue(res.side_effects)
        self.assertEqual(res.payload["target"], "5551234")
        self.assertEqual(res.payload["dialer"], "baresip")
        self.assertEqual(res.payload["answer"], "Calling 555 1234.")
        self.assertEqual(fake.stdin.getvalue(), b"dial 5551234\n")

    def test_live_call_rejects_bad_contact(self):
        d = BaresipDialer(executable="/opt/bin/baresip")
        with mock.patch.object(d, "available", return_value=True):
            res = self._executor(dialer=d, dry_run=False).run(
                _req("call", {"contact": "mom"}))
        self.assertFalse(res.ok)
        self.assertIn("call failed", res.detail)
        self.assertFalse(res.side_effects)

    def test_unsupported_action_rejected(self):
        d = BaresipDialer(executable="/opt/bin/baresip")
        req = _req("call", {"contact": "555"})
        fake = type(req)(**{**req.__dict__, "action_code": "call.transfer"})
        res = self._executor(dialer=d, dry_run=False).run(fake)
        self.assertFalse(res.ok)
        self.assertIn("unsupported comms action", res.detail)


class TestFullStack(unittest.TestCase):
    def test_orchestrator_speaks_call_reply(self):
        d = BaresipDialer(executable="/opt/bin/baresip")
        fake = _FakeProc()
        spoken: list[str] = []
        orch = default_orchestrator(dry_run=False, dialer=d, speak=spoken.append)
        with mock.patch.object(d, "available", return_value=True), \
             mock.patch("rpi5.calls.time.sleep"), \
             mock.patch("rpi5.calls.subprocess.Popen", return_value=fake):
            result = orch.run(_req("call", {"contact": "555 1234"}),
                              source="microphone")
        self.assertTrue(result.handled)
        self.assertTrue(result.execution.side_effects)
        # The spoken reply is the templated "Calling <contact>."
        self.assertIn("Calling 555 1234.", spoken)


if __name__ == "__main__":
    unittest.main()
