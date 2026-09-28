"""Outbound call dialers (baresip).

The ``call`` intent places a SIP call through :program:`baresip`, the
open-source VoIP softphone that runs headless on the Pi. This module owns the
*how* of dialling; :class:`~rpi5.executors.comms.CommsExecutor` owns the *what*
(validating the contact, shaping the spoken reply) and only speaks the generic
:class:`Dialer` interface below.

Pluggable, exactly like the light drivers in :mod:`rpi5.rgb`: the concrete
dialer is selected at startup (``--dialer NAME``), looked up in
:data:`DIALERS`, and a new dial device is one subclass plus one registry line --
never an edit to the executor, the orchestrator, or the model.

Baresip specifics
-----------------
``baresip`` is a long-running daemon, not a one-shot CLI. We therefore start it
once per process (lazily, on the first dial) with ``--config`` pointing at the
account profile, then place each call by piping a single ``dial <target>``
command into its stdin. The daemon is left running so subsequent calls reuse the
registered SIP session; it is torn down by :meth:`Dialer.close` (or process
exit). The target is validated against a conservative SIP-address charset before
anything is written to the daemon, so a stray transcript can't inject commands.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

LOGGER = logging.getLogger("pi5-vcm.calls")

#: Conservative charset for a SIP dial target: digits, ``+`` (international),
#: ``*`` (feature codes), ``#``, and ``,`` (pause). Anything else -- spaces,
#: quotes, semicolons, newlines -- is rejected before it reaches the daemon.
_TARGET_RE = re.compile(r"^[+\d*#,]{1,32}$")


class Dialer(ABC):
    """Interface a :class:`CommsExecutor` uses to place and end calls.

    Implementations remember nothing about *which* contact was last dialed; the
    executor passes an explicit target each time. ``available()`` reports
    whether the underlying tool is present on this host so the assistant can
    answer honestly ("calls aren't reachable here") instead of erroring.
    """

    #: Short name shown in events / logs and used as the registry key.
    name: ClassVar[str] = "dialer"

    @abstractmethod
    def dial(self, target: str) -> None:
        """Place a call to ``target``. Raise on failure; the executor catches."""

    def cancel(self) -> None:
        """End the active call. Default is a no-op (safe to call repeatedly)."""

    @abstractmethod
    def available(self) -> bool:
        """Whether the underlying tool is present on this host."""

    def close(self) -> None:
        """Release resources; safe to call repeatedly."""


class BaresipDialer(Dialer):
    """Place SIP calls through a :program:`baresip` daemon.

    The daemon binary is resolved from ``$BARESIP`` (if set) or ``PATH``. On the
    first :meth:`dial` we launch ``baresip --config <cfg>`` with its stdin held
    open; each subsequent call writes ``dial <target>\\n`` to that stdin. The
    ``--config`` file is the baresip profile (accounts, transports, audio) --
    the same file you'd run ``baresip -c cfg.conf`` with by hand.

    When the binary is absent the dialer degrades to a no-op that reports
    ``available() == False`` rather than raising, so the assistant still answers
    honestly instead of crashing the mic loop.
    """

    name = "baresip"

    #: Seconds to wait for the daemon to register before considering a dial
    #: "too early". Kept short: the dial command itself is queued by baresip
    #: and answered asynchronously, so this is only a readiness nudge.
    _READY_DELAY = 1.0

    def __init__(self, executable: str | None = None,
                 config: str | None = None,
                 account: str | None = None) -> None:
        self.executable = executable or os.environ.get("BARESIP") or "baresip"
        self.config = config
        self.account = account
        self._proc: subprocess.Popen[Any] | None = None

    # -- lifecycle -------------------------------------------------------- #
    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def _ensure_daemon(self) -> None:
        """Start the baresip daemon once, keeping stdin open for commands."""
        if self._proc is not None and self._proc.poll() is None:
            return
        cmd = [self.executable]
        if self.config:
            cmd += ["--config", self.config]
        # "--account" pins which registered account dials when several are
        # configured; harmless when the profile has exactly one.
        if self.account:
            cmd += ["--account", self.account]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # Give the daemon a beat to bind its socket before we queue a dial.
        time.sleep(self._READY_DELAY)

    def _send(self, command: str) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise RuntimeError("baresip daemon is not running")
        proc.stdin.write((command + "\n").encode("ascii"))
        proc.stdin.flush()

    # -- Dialer interface ------------------------------------------------- #
    def dial(self, target: str) -> None:
        if not self.available():
            raise RuntimeError(f"baresip executable '{self.executable}' not found")
        self._ensure_daemon()
        self._send(f"dial {target}")
        LOGGER.info("[calls] dialed %s via baresip", target)

    def cancel(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            self._send("cancel")
        except Exception:  # noqa: BLE001
            LOGGER.warning("[calls] cancel failed", exc_info=True)

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass


def normalize_target(contact: str) -> str:
    """Validate and normalize a dial target.

    Strips surrounding whitespace and collapses internal spaces (a transcript
    like ``"call mom"`` yields ``mom``; a spaced number ``"555 1234"`` yields
    ``5551234``). Raises :class:`ValueError` if the result is empty or contains
    a character outside the safe SIP charset.
    """
    # Strip surrounding whitespace and collapse internal spaces; drop any other
    # whitespace (newlines, tabs) so a transcript can't smuggle a command
    # separator into the daemon's stdin.
    cleaned = "".join(ch for ch in contact if not ch.isspace())
    if not cleaned or not _TARGET_RE.match(cleaned):
        raise ValueError(f"unusable dial target: {contact!r}")
    return cleaned


#: Registry of selectable dialers. ``make_dialer(name)`` looks names up here, so
#: adding a dial device is one line plus one class.
DIALERS: dict[str, type[Dialer]] = {
    BaresipDialer.name: BaresipDialer,
}


def make_dialer(name: str | None, *, executable: str | None = None,
                config: str | None = None,
                account: str | None = None) -> Dialer | None:
    """Instantiate a dialer by registered name.

    Returns ``None`` for a falsy name (calls stay dry-run) or for an unknown
    name (logged to stderr so a typo is visible, and calls fall back to dry-run
    rather than crashing the mic loop). Only the :class:`BaresipDialer` consumes
    ``config``/``account``; other dialers receive ``executable`` only.
    """
    if not name:
        return None
    cls = DIALERS.get(name)
    if cls is None:
        print(f"[calls] unknown dialer '{name}'; "
              f"known: {', '.join(sorted(DIALERS))}; staying dry-run",
              file=sys.stderr)
        return None
    if cls is BaresipDialer:
        return cls(executable=executable, config=config, account=account)
    return cls(executable)
