from __future__ import annotations

import unittest

from rpi5.harness import DryRunDispatcher
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
            infer=infer,
            act=act,
            play_reply=reply,
        )
        return machine, rgb, calls

    def test_complete_interaction_order(self):
        # Mirror the live mic loop: acknowledge_wake() enters LISTENING, then
        # handle_command() runs the captured audio through the rest.
        machine, rgb, calls = self.make_machine()

        machine.acknowledge_wake()
        outcome = machine.handle_command(
            b"audio", generation=machine.current_generation())

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
        self.assertEqual(calls, ["ack", "infer:b'audio'", "act", "reply"])

    def test_single_execution_no_double_reply(self):
        # Regression: "two Alexas speaking". The action must run exactly once
        # per command. infer_command() only recognizes; execute_action() (via
        # act) is the sole execution point, and play_reply() speaks once.
        machine, rgb, calls = self.make_machine()

        machine.acknowledge_wake()
        machine.handle_command(b"audio",
                               generation=machine.current_generation())

        self.assertEqual(calls.count("act"), 1)
        self.assertEqual(calls.count("reply"), 1)
        self.assertEqual(calls.count("infer:b'audio'"), 1)

    def test_acknowledge_wake_is_ignored_while_busy(self):
        machine, rgb, calls = self.make_machine()
        machine.state = State.PROCESSING

        machine.acknowledge_wake()

        self.assertEqual(machine.state, State.PROCESSING)
        self.assertEqual(rgb.calls, [])
        self.assertEqual(calls, [])

    def test_dispatcher_handles_partial_inference_results(self):
        dispatcher = DryRunDispatcher()

        outcome = dispatcher.dispatch({
            "intent": "turn_on_lights",
            "intent_confidence": 0.91,
            "slots": {},
        })

        self.assertEqual(outcome["status"], "dry_run")
        self.assertEqual(outcome["action"], "lights.on")

    def test_cancel_listening_returns_to_standby(self):
        machine, rgb, calls = self.make_machine()
        machine.acknowledge_wake()

        machine.cancel_listening()

        self.assertEqual(machine.state, State.STANDBY)
        self.assertEqual(rgb.calls, ["wake", "idle"])
        self.assertEqual(calls, ["ack"])

    def test_stale_command_dropped_when_second_wake_arrives(self):
        """Regression: two intents playing at once.

        The live mic loop stamps each captured command with the interaction's
        generation (machine.current_generation()). While the first command is
        mid-flight (slow action, e.g. a live weather lookup) a second wake word
        lands and bumps the generation. The first command now carries a stale
        generation and must be DROPPED so the user hears one reply, not two at
        once. The newest command still processes normally."""
        machine, rgb, calls = self.make_machine()
        machine.acknowledge_wake()                      # gen 1, -> LISTENING
        first_gen = machine.current_generation()

        # Second wake arrives while the first is still resolving: the machine
        # is pulled back to STANDBY (as the loop would after the first reply)
        # and a new interaction is acknowledged, bumping the generation.
        machine.state = State.STANDBY
        machine.acknowledge_wake()                      # gen 2, -> LISTENING
        self.assertNotEqual(first_gen, machine.current_generation())

        # The first (stale) command is dropped even though state allows it.
        self.assertIsNone(machine.handle_command(b"first", generation=first_gen))
        self.assertNotIn("infer:b'first'", calls)
        self.assertNotIn("act", calls)
        self.assertNotIn("reply", calls)

        # The newest command (matching generation) still processes.
        outcome = machine.handle_command(
            b"second", generation=machine.current_generation())
        self.assertIsNotNone(outcome)
        self.assertIn("infer:b'second'", calls)

    def test_command_accepted_from_standby_after_first_completes(self):
        """A fresh command is still processed once the prior one has returned
        the machine to STANDBY (the normal sequential path)."""
        machine, rgb, calls = self.make_machine()
        machine.acknowledge_wake()          # -> LISTENING
        machine.state = State.STANDBY       # simulate first interaction done

        outcome = machine.handle_command(
            b"next command", generation=machine.current_generation())
        self.assertIsNotNone(outcome)
        self.assertIn("infer:b'next command'", calls)


if __name__ == "__main__":
    unittest.main()
