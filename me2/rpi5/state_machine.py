"""Testable orchestration for the Pi5-VCM assistant lifecycle."""
from __future__ import annotations

from enum import Enum
from typing import Any, Callable


class State(str, Enum):
    STANDBY = "standby"
    WAKE = "wake"
    ACKNOWLEDGING = "acknowledging"
    LISTENING = "listening"
    PROCESSING = "processing"
    ACTING = "acting"
    CONFIRMING = "confirming"


class HarnessStateMachine:
    """Coordinate wake, command capture, inference, action, and reply.

    Every external operation is injected, making transitions testable without
    a microphone, model, RGB device, or speaker.
    """

    def __init__(
        self,
        *,
        rgb: Any,
        play_ack: Callable[[], Any],
        capture_command: Callable[[float], Any | None],
        infer: Callable[[Any], dict[str, Any]],
        act: Callable[[dict[str, Any]], Any],
        play_reply: Callable[[dict[str, Any], Any], Any],
        command_timeout_s: float = 7.0,
    ) -> None:
        self.rgb = rgb
        self.play_ack = play_ack
        self.capture_command = capture_command
        self.infer = infer
        self.act = act
        self.play_reply = play_reply
        self.command_timeout_s = command_timeout_s
        self.state = State.STANDBY
        self.history: list[State] = [self.state]

    def _set_state(self, state: State) -> None:
        self.state = state
        self.history.append(state)

    def handle_wake(self) -> dict[str, Any] | None:
        """Run one complete interaction; return None on command timeout."""
        if self.state is not State.STANDBY:
            return None

        self._set_state(State.WAKE)
        self._set_state(State.ACKNOWLEDGING)
        self.play_ack()
        self.rgb.wake()

        self._set_state(State.LISTENING)
        command_audio = self.capture_command(self.command_timeout_s)
        if command_audio is None:
            self.rgb.idle()
            self._set_state(State.STANDBY)
            return None

        return self.handle_command(command_audio)

    def acknowledge_wake(self) -> None:
        """Enter LISTENING after the wake acknowledgement has completed."""
        if self.state is not State.STANDBY:
            return
        self._set_state(State.WAKE)
        self._set_state(State.ACKNOWLEDGING)
        self.play_ack()
        self.rgb.wake()
        self._set_state(State.LISTENING)

    def cancel_listening(self) -> None:
        """Return to standby when no command begins before the timeout."""
        if self.state is not State.LISTENING:
            return
        self.rgb.idle()
        self._set_state(State.STANDBY)

    def handle_command(self, command_audio: Any) -> dict[str, Any] | None:
        """Process already-captured command audio through the remaining states."""
        if self.state not in {State.STANDBY, State.LISTENING}:
            return None

        self._set_state(State.PROCESSING)
        self.rgb.processing()
        result = self.infer(command_audio)

        self._set_state(State.ACTING)
        if hasattr(self.rgb, "acting"):
            self.rgb.acting()
        action_result = self.act(result)

        self._set_state(State.CONFIRMING)
        self.rgb.speaking()
        self.play_reply(result, action_result)
        self.rgb.idle()
        self._set_state(State.STANDBY)
        return {"result": result, "action": action_result}
