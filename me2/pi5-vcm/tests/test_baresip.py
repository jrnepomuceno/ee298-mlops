from __future__ import annotations

import json
import socket
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference.pi5_runner import baresip
from inference.pi5_runner.baresip import BaresipClient, BaresipError


def _send_netstring(connection: socket.socket, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    connection.sendall(str(len(encoded)).encode("ascii") + b":" + encoded + b",")


def _receive_netstring(connection: socket.socket) -> dict[str, object]:
    digits = bytearray()
    while True:
        character = connection.recv(1)
        if character == b":":
            break
        if not character:
            raise RuntimeError("mock control connection closed before frame header")
        digits.extend(character)
    length = int(digits)
    payload = bytearray()
    while len(payload) < length + 1:
        chunk = connection.recv(length + 1 - len(payload))
        if not chunk:
            raise RuntimeError("mock control connection closed before frame body")
        payload.extend(chunk)
    assert payload[-1] == ord(",")
    return json.loads(payload[:-1])


def test_control_client_dials_over_mocked_json_netstrings() -> None:
    client_socket, server_socket = socket.socketpair()
    client = BaresipClient(enabled=True, autostart=False)
    client._attach_connection(client_socket)
    received_requests: list[dict[str, object]] = []
    server_errors: list[BaseException] = []
    closed_call = threading.Event()
    release_call = threading.Event()

    def serve_commands() -> None:
        try:
            dial_request = _receive_netstring(server_socket)
            received_requests.append(dial_request)
            _send_netstring(
                server_socket,
                {"event": True, "class": "call", "type": "CALL_ESTABLISHED"},
            )
            _send_netstring(
                server_socket,
                {
                    "response": True,
                    "ok": True,
                    "data": "",
                    "token": dial_request["token"],
                },
            )

            release_call.wait(timeout=2)
            _send_netstring(
                server_socket,
                {"event": True, "class": "call", "type": "CALL_CLOSED"},
            )
            closed_call.set()
        except BaseException as exc:
            server_errors.append(exc)

    server_thread = threading.Thread(target=serve_commands, daemon=True)
    server_thread.start()
    try:
        client.dial_current_contact()
        waiter = threading.Thread(target=client.wait_for_call_end, daemon=True)
        waiter.start()
        waiter.join(timeout=0.05)
        assert waiter.is_alive()
        release_call.set()
        waiter.join(timeout=2)
        assert not waiter.is_alive()
    finally:
        release_call.set()
        client.close()
        server_socket.close()
        server_thread.join(timeout=2)

    assert not server_errors
    assert closed_call.is_set()
    assert len(received_requests) == 1
    assert received_requests[0]["command"] == "dialcontact"


def test_message_uses_baresip_execute_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(baresip.shutil, "which", lambda _: "/usr/bin/baresip")
    monkeypatch.setattr(baresip.subprocess, "run", run)
    client = BaresipClient(enabled=True, config_dir=tmp_path, autostart=False)

    client.send_message("text message here")

    assert commands == [
        (
            [
                "/usr/bin/baresip",
                "-f",
                str(tmp_path),
                "-e",
                "/message text message here",
                "-t",
                "8",
            ],
            {
                "check": False,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "timeout": 12.0,
            },
        )
    ]


def test_message_restarts_owned_baresip_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        terminated = False

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float) -> int:
            return 0

    process = Process()
    monkeypatch.setattr(baresip.shutil, "which", lambda _: "/usr/bin/baresip")
    monkeypatch.setattr(
        baresip.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    client = BaresipClient(enabled=True, autostart=False)
    client._process = process
    restart_calls: list[bool] = []
    client.start = lambda: restart_calls.append(True)

    client.send_message("text message here")

    assert process.terminated
    assert restart_calls == [True]


def test_message_refuses_to_launch_alongside_externally_managed_baresip() -> None:
    client_socket, server_socket = socket.socketpair()
    client = BaresipClient(enabled=True, autostart=False)
    client._attach_connection(client_socket)
    try:
        with pytest.raises(BaresipError, match="externally managed Baresip"):
            client.send_message("text message here")
    finally:
        client.close()
        server_socket.close()


def test_message_reports_nonzero_baresip_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(baresip.shutil, "which", lambda _: "/usr/bin/baresip")
    monkeypatch.setattr(
        baresip.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )
    client = BaresipClient(enabled=True, autostart=False)

    with pytest.raises(BaresipError, match="exited with status 1"):
        client.send_message("text message here")


@pytest.mark.parametrize(
    "body",
    [
        "",
        "   ",
        "line one\nline two",
        "line one\rline two",
        "x" * 901,
    ],
)
def test_message_body_rejects_empty_control_characters_and_oversize(
    body: str,
) -> None:
    with pytest.raises(BaresipError):
        BaresipClient._validate_message_body(body)


def test_message_body_allows_utf8_within_byte_limit() -> None:
    BaresipClient._validate_message_body("Café — see you soon.")


def test_startup_requires_contact_control_module_and_loopback_listener(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config"
    config.write_text(
        "module alsa.so\n"
        "module_app contact.so\n"
        "module_app ctrl_tcp.so\n"
        "audio_source alsa,default\n"
        "ctrl_tcp_listen 127.0.0.1:4444\n",
        encoding="utf-8",
    )
    client = BaresipClient(enabled=False, config_dir=tmp_path)

    client._validate_config()

    config.write_text(
        "module alsa.so\n"
        "module_app contact.so\n"
        "module_app ctrl_tcp.so\n"
        "audio_source alsa,default\n"
        "ctrl_tcp_listen 0.0.0.0:4444\n",
        encoding="utf-8",
    )
    with pytest.raises(BaresipError, match="must be exactly 127.0.0.1:4444"):
        client._validate_config()

    config.write_text(
        "module alsa.so\n"
        "module_app ctrl_tcp.so\n"
        "module_app contact.so\n"
        "audio_source alsa,default\n"
        "ctrl_tcp_listen 127.0.0.1:4444\n",
        encoding="utf-8",
    )
    with pytest.raises(BaresipError, match="contact.so before ctrl_tcp.so"):
        client._validate_config()


def test_startup_rejects_missing_contact_module(tmp_path: Path) -> None:
    (tmp_path / "config").write_text(
        "module alsa.so\n"
        "module_app ctrl_tcp.so\n"
        "audio_source alsa,default\n"
        "ctrl_tcp_listen 127.0.0.1:4444\n",
        encoding="utf-8",
    )
    client = BaresipClient(enabled=False, config_dir=tmp_path)

    with pytest.raises(BaresipError, match="must load contact.so"):
        client._validate_config()


def test_startup_rejects_audio_source_that_would_not_share_runner_mic(
    tmp_path: Path,
) -> None:
    (tmp_path / "config").write_text(
        "module alsa.so\n"
        "module_app contact.so\n"
        "module_app ctrl_tcp.so\n"
        "audio_source alsa,hw:1,0\n"
        "ctrl_tcp_listen 127.0.0.1:4444\n",
        encoding="utf-8",
    )
    client = BaresipClient(enabled=False, config_dir=tmp_path)

    with pytest.raises(BaresipError, match="audio_source must be exactly"):
        client._validate_config()
