"""Local Baresip control for the Raspberry Pi intent runner."""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from unicodedata import category

logger = logging.getLogger(__name__)

BARESIP_HOST = "127.0.0.1"
BARESIP_PORT = 4444
BARESIP_START_TIMEOUT_SECONDS = 10.0
BARESIP_COMMAND_TIMEOUT_SECONDS = 5.0
BARESIP_STOP_TIMEOUT_SECONDS = 3.0
BARESIP_MESSAGE_RUN_SECONDS = 8
BARESIP_MESSAGE_TIMEOUT_SECONDS = 12.0
BARESIP_MAX_NETSTRING_BYTES = 8192
MESSAGE_BODY_MAX_BYTES = 900


class BaresipError(RuntimeError):
    """A Baresip process or local control request failed."""


class BaresipClient:
    """Manage Baresip and its single JSON-netstring control connection."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        config_dir: Path | None = None,
        audio_device: str | None = None,
        connection_factory: Callable[..., socket.socket] = socket.create_connection,
        process_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        autostart: bool = True,
    ) -> None:
        if enabled is None:
            setting = os.environ.get("PI5_VCM_BARESIP_ENABLED", "0")
            if setting not in {"0", "1"}:
                raise BaresipError("PI5_VCM_BARESIP_ENABLED must be 0 or 1")
            enabled = setting == "1"
        self.enabled = enabled
        self.config_dir = config_dir or Path(
            os.environ.get(
                "PI5_VCM_BARESIP_CONFIG_DIR",
                str(Path.home() / ".baresip"),
            )
        )
        self.audio_device = audio_device or os.environ.get(
            "PI5_VCM_INPUT_DEVICE",
            "default:CARD=DuoCast",
        )
        self._connection_factory = connection_factory
        self._process_factory = process_factory
        self._process: subprocess.Popen[bytes] | None = None
        self._socket: socket.socket | None = None
        self._reader: threading.Thread | None = None
        self._send_lock = threading.Lock()
        self._condition = threading.Condition()
        self._responses: dict[str, dict[str, object]] = {}
        self._reader_error: str | None = None
        self._closing = False
        self._call_active = False
        self._call_event_sequence = 0

        if self.enabled and autostart:
            self.start()

    def start(self) -> None:
        if not self.enabled:
            raise BaresipError("Baresip integration is disabled")
        if self._socket is not None:
            return
        self._closing = False
        self._reader_error = None
        try:
            self._validate_config()

            deadline = time.monotonic() + BARESIP_START_TIMEOUT_SECONDS
            while True:
                try:
                    connection = self._connection_factory(
                        (BARESIP_HOST, BARESIP_PORT), timeout=1.0
                    )
                except OSError:
                    if self._process is None:
                        executable = shutil.which("baresip")
                        if executable is None:
                            raise BaresipError(
                                "baresip is not installed; install it or disable "
                                "PI5_VCM_BARESIP_ENABLED"
                            )
                        try:
                            self._process = self._process_factory(
                                [executable, "-f", str(self.config_dir)],
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                start_new_session=True,
                            )
                        except OSError as exc:
                            raise BaresipError("could not start baresip") from exc
                    if self._process.poll() is not None:
                        raise BaresipError(
                            "baresip exited before its loopback control service "
                            "became ready"
                        )
                    if time.monotonic() >= deadline:
                        raise BaresipError(
                            "baresip control service did not become ready on "
                            "127.0.0.1:4444"
                        )
                    time.sleep(0.1)
                    continue

                self._attach_connection(connection)
                logger.info("Baresip control connected on 127.0.0.1:4444")
                return
        except BaresipError:
            self.close()
            raise

    def dial_current_contact(self) -> None:
        if not self.enabled:
            raise BaresipError("Baresip integration is disabled")
        with self._condition:
            event_sequence = self._call_event_sequence
        self._send_command("dialcontact")
        with self._condition:
            if self._call_event_sequence == event_sequence:
                self._call_active = True
            self._condition.notify_all()

    def send_message(self, body: str) -> None:
        if not self.enabled:
            raise BaresipError("Baresip integration is disabled")
        self._validate_message_body(body)
        if self._socket is not None and self._process is None:
            raise BaresipError(
                "cannot run the Baresip MESSAGE command while connected to an "
                "externally managed Baresip process"
            )

        restart_process = self._process is not None
        if restart_process:
            self.close()
        try:
            self._run_message_command(body)
        finally:
            if restart_process:
                try:
                    self.start()
                except BaresipError as exc:
                    logger.error(
                        "MESSAGE command finished, but Baresip control could not "
                        "be restarted: %s",
                        exc,
                    )

    def wait_for_call_end(self) -> None:
        with self._condition:
            while self._call_active and self._reader_error is None:
                self._condition.wait()
            if self._reader_error is not None and self._call_active:
                raise BaresipError("Baresip control disconnected during an active call")

    def close(self) -> None:
        self._closing = True
        connection = self._socket
        self._socket = None
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        if self._reader is not None:
            self._reader.join(timeout=2)
            self._reader = None

        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=BARESIP_STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=BARESIP_STOP_TIMEOUT_SECONDS)

    def _validate_config(self) -> None:
        config_path = self.config_dir / "config"
        try:
            config_lines = config_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise BaresipError(
                f"cannot read Baresip configuration: {config_path}"
            ) from exc

        modules: list[str] = []
        listeners: list[str] = []
        audio_sources: list[str] = []
        for line in config_lines:
            fields = line.split("#", 1)[0].split()
            if len(fields) < 2:
                continue
            if fields[0] in {"module", "module_app"}:
                modules.append(Path(fields[1]).name)
            elif fields[0] == "ctrl_tcp_listen":
                listeners.append(fields[1])
            elif fields[0] == "audio_source":
                audio_sources.append(fields[1])

        missing_modules = {"alsa.so", "contact.so", "ctrl_tcp.so"} - set(modules)
        if missing_modules:
            modules_text = ", ".join(sorted(missing_modules))
            raise BaresipError(f"Baresip configuration must load {modules_text}")
        if modules.index("contact.so") > modules.index("ctrl_tcp.so"):
            raise BaresipError(
                "Baresip configuration must load contact.so before ctrl_tcp.so"
            )
        if audio_sources != [f"alsa,{self.audio_device}"]:
            raise BaresipError(
                "Baresip audio_source must be exactly "
                f"alsa,{self.audio_device} to hand the configured runner "
                "microphone to Baresip"
            )
        if listeners != [f"{BARESIP_HOST}:{BARESIP_PORT}"]:
            raise BaresipError(
                "Baresip ctrl_tcp_listen must be exactly "
                "127.0.0.1:4444; do not expose it on a network interface"
            )

    def _attach_connection(self, connection: socket.socket) -> None:
        if self._socket is not None:
            connection.close()
            raise BaresipError("Baresip control is already connected")
        self._socket = connection
        connection.settimeout(0.5)
        self._reader = threading.Thread(
            target=self._read_messages,
            name="pi5-vcm-baresip-control",
            daemon=True,
        )
        self._reader.start()

    def _send_command(self, command: str, params: str | None = None) -> None:
        connection = self._socket
        if connection is None:
            raise BaresipError("Baresip control is not connected")
        token = uuid.uuid4().hex
        request: dict[str, str] = {"command": command, "token": token}
        if params is not None:
            request["params"] = params
        encoded = json.dumps(
            request,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        netstring = str(len(encoded)).encode("ascii") + b":" + encoded + b","

        with self._send_lock:
            try:
                connection.sendall(netstring)
            except OSError as exc:
                raise BaresipError("could not send a Baresip control command") from exc

        deadline = time.monotonic() + BARESIP_COMMAND_TIMEOUT_SECONDS
        with self._condition:
            while token not in self._responses:
                if self._reader_error is not None:
                    raise BaresipError("Baresip control connection closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BaresipError("Baresip control command timed out")
                self._condition.wait(remaining)
            response = self._responses.pop(token)

        if response.get("ok") is not True:
            raise BaresipError(f"Baresip rejected the {command} command")

    def _run_message_command(self, body: str) -> None:
        executable = shutil.which("baresip")
        if executable is None:
            raise BaresipError("baresip is not installed")
        try:
            result = subprocess.run(
                [
                    executable,
                    "-f",
                    str(self.config_dir),
                    "-e",
                    f"/message {body}",
                    "-t",
                    str(BARESIP_MESSAGE_RUN_SECONDS),
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=BARESIP_MESSAGE_TIMEOUT_SECONDS,
            )
        except OSError as exc:
            raise BaresipError("could not run the Baresip MESSAGE command") from exc
        except subprocess.TimeoutExpired as exc:
            raise BaresipError("Baresip MESSAGE command timed out") from exc
        if result.returncode != 0:
            raise BaresipError(
                "Baresip MESSAGE command exited with "
                f"status {result.returncode}"
            )

    def _read_messages(self) -> None:
        connection = self._socket
        if connection is None:
            return
        pending = bytearray()
        error: str | None = None
        try:
            while True:
                try:
                    chunk = connection.recv(4096)
                except socket.timeout:
                    if self._closing:
                        return
                    continue
                if not chunk:
                    if not self._closing:
                        error = "connection closed"
                    return
                pending.extend(chunk)
                for payload in self._extract_netstrings(pending):
                    message = json.loads(payload.decode("utf-8"))
                    self._handle_message(message)
        except (BaresipError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            error = str(exc)
        except OSError as exc:
            if not self._closing:
                error = str(exc)
        finally:
            if error is not None:
                with self._condition:
                    self._reader_error = error
                    self._condition.notify_all()

    @staticmethod
    def _extract_netstrings(pending: bytearray) -> list[bytes]:
        payloads: list[bytes] = []
        while True:
            colon = pending.find(b":")
            if colon < 0:
                if len(pending) > 10:
                    raise BaresipError("invalid Baresip netstring length")
                break
            length_bytes = pending[:colon]
            if not length_bytes or not length_bytes.isdigit():
                raise BaresipError("invalid Baresip netstring length")
            length = int(length_bytes)
            if length > BARESIP_MAX_NETSTRING_BYTES:
                raise BaresipError("Baresip control message exceeded size limit")
            frame_end = colon + 1 + length
            if len(pending) <= frame_end:
                break
            if pending[frame_end] != ord(","):
                raise BaresipError("invalid Baresip netstring terminator")
            payloads.append(bytes(pending[colon + 1 : frame_end]))
            del pending[: frame_end + 1]
        return payloads

    def _handle_message(self, message: object) -> None:
        if not isinstance(message, dict):
            raise BaresipError("Baresip sent an invalid control message")
        if message.get("response") is True:
            token = message.get("token")
            if isinstance(token, str):
                with self._condition:
                    self._responses[token] = message
                    self._condition.notify_all()
            return
        if message.get("event") is True and message.get("class") == "call":
            event_type = message.get("type")
            if not isinstance(event_type, str):
                return
            with self._condition:
                self._call_event_sequence += 1
                if event_type == "CALL_CLOSED":
                    self._call_active = False
                elif event_type.startswith("CALL_"):
                    self._call_active = True
                self._condition.notify_all()

    @staticmethod
    def _validate_message_body(body: str) -> None:
        if not isinstance(body, str) or not body.strip():
            raise BaresipError("PI5_VCM_MESSAGE_TEXT must not be empty")
        if any(category(character) in {"Cc", "Zl", "Zp"} for character in body):
            raise BaresipError(
                "PI5_VCM_MESSAGE_TEXT must not contain control characters"
            )
        try:
            body_bytes = body.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise BaresipError(
                "PI5_VCM_MESSAGE_TEXT must contain valid Unicode text"
            ) from exc
        if len(body_bytes) > MESSAGE_BODY_MAX_BYTES:
            raise BaresipError(
                f"PI5_VCM_MESSAGE_TEXT must be at most {MESSAGE_BODY_MAX_BYTES} "
                "UTF-8 bytes"
            )


def message_text_from_environment() -> str | None:
    """Return the explicitly configured fixed body, without a success fallback."""
    body = os.environ.get("PI5_VCM_MESSAGE_TEXT")
    if body is None:
        return None
    BaresipClient._validate_message_body(body)
    return body
