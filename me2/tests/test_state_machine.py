from __future__ import annotations

import unittest

from rpi5.state_machine import HarnessStateMachine, State


class FakeRgb:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def wake(self) -> None:
        self.calls.append("wake")

    def processing(self) -> None:
        self.calls.append("processing")

    def speaking(self) -> None:
        self.calls.append("speaking")

    def acting(self) -> None:
        self.calls.append("acting")

    def idle(self) -> None:
        self.calls.append("idle")


class StateMachineTests(unittest.TestCase):
    def make_machine(self, command=b"audio"):
        rgb = FakeRgb()
        calls: list[str] = []

        def ack():
            calls.append("ack")

        def capture(timeout):
            calls.append(f"capture:{timeout}")
            return command

        def infer(audio):
            calls.append(f"infer:{audio!r}")
            return {"intent": "turn_on_lights", "slots": {}}

        def act(result):
            calls.append("act")
            return {"status": "dry_run"}

        def reply(result, action):
            calls.append("reply")

        machine = HarnessStateMachine(
            rgb=rgb,
            play_ack=ack,
            capture_command=capture,
            infer=infer,
            act=act,
            play_reply=reply,
        )
        return machine, rgb, calls

    def test_complete_interaction_order(self):
        machine, rgb, calls = self.make_machine()

        outcome = machine.handle_wake()

        self.assertEqual(outcome["result"]["intent"], "turn_on_lights")
        self.assertEqual(
            machine.history,
            [State.STANDBY, State.WAKE, State.ACKNOWLEDGING,
             State.LISTENING,
             State.PROCESSING, State.ACTING, State.CONFIRMING,
             State.STANDBY],
        )
        self.assertEqual(
            rgb.calls, ["wake", "processing", "acting", "speaking", "idle"])
        self.assertEqual(
            calls,
            ["ack", "capture:7.0", "infer:b'audio'", "act", "reply"],
        )

    def test_timeout_returns_to_standby_without_inference(self):
        machine, rgb, calls = self.make_machine(command=None)

        self.assertIsNone(machine.handle_wake())
        self.assertEqual(machine.state, State.STANDBY)
        self.assertEqual(rgb.calls, ["wake", "idle"])
        self.assertEqual(calls, ["ack", "capture:7.0"])

    def test_wake_is_ignored_while_busy(self):
        machine, rgb, calls = self.make_machine()
        machine.state = State.PROCESSING

        self.assertIsNone(machine.handle_wake())
        self.assertEqual(rgb.calls, [])
        self.assertEqual(calls, [])

    def test_cancel_listening_returns_to_standby(self):
        machine, rgb, calls = self.make_machine()
        machine.acknowledge_wake()

        machine.cancel_listening()

        self.assertEqual(machine.state, State.STANDBY)
        self.assertEqual(rgb.calls, ["wake", "idle"])
        self.assertEqual(calls, ["ack"])


if __name__ == "__main__":
    unittest.main()
