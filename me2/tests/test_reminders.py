"""Unit tests for the reminders store + live what_reminders path (P1+P2+P3).

Covers: store CRUD, JSON persistence round-trip, atomic-write safety, corrupt
and malformed-file recovery, the spoken summarize wording (0/1/2/many, blank
note fallback), the no-store fallback, and the full facade -> orchestrator ->
InfoExecutor path for the ``what_reminders`` intent.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from rpi5.reminders import ReminderStore
from rpi5.harness import FacadePipeline
from rpi5.executors.info import InfoExecutor
from rpi5.executors.reminder import ReminderExecutor
from rpi5.orchestrator import default_orchestrator
from rpi5.facade import decode, ActionRequest


def _mkstore(tmpdir, name="rem.json"):
    return ReminderStore(os.path.join(tmpdir, name))


class StoreCrudTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = _mkstore(self.tmp.name)

    def test_starts_empty(self):
        self.assertEqual(len(self.s), 0)
        self.assertEqual(self.s.list(), [])

    def test_add_returns_record_and_increments_id(self):
        r1 = self.s.add("take out the trash")
        r2 = self.s.add("call Mom")
        self.assertEqual(r1["id"], 1)
        self.assertEqual(r2["id"], 2)
        self.assertEqual([r["id"] for r in self.s.list()], [1, 2])
        self.assertTrue(r1["created_at"])
        self.assertIsNone(r1["due_at"])

    def test_add_strips_note(self):
        r = self.s.add("   buy milk   ")
        self.assertEqual(r["note"], "buy milk")

    def test_remove_existing(self):
        r1 = self.s.add("a")
        r2 = self.s.add("b")
        self.assertTrue(self.s.remove(r1["id"]))
        self.assertEqual([r["id"] for r in self.s.list()], [r2["id"]])

    def test_remove_missing_is_false(self):
        self.assertFalse(self.s.remove(999))

    def test_clear_returns_count(self):
        self.s.add("a")
        self.s.add("b")
        self.assertEqual(self.s.clear(), 2)
        self.assertEqual(len(self.s), 0)

    def test_list_returns_copies(self):
        r = self.s.add("a")
        snapshot = self.s.list()
        snapshot[0]["note"] = "mutated"
        self.assertEqual(self.s.list()[0]["note"], "a")
        self.assertIsNot(r, self.s.list()[0])


class StorePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "sub", "rem.json")

    def test_round_trip(self):
        s = ReminderStore(self.path)
        s.add("take out the trash")
        s.add("call Mom")
        s2 = ReminderStore(self.path)
        self.assertEqual([r["note"] for r in s2.list()],
                         ["take out the trash", "call Mom"])
        self.assertEqual([r["id"] for r in s2.list()], [1, 2])

    def test_creates_parent_dirs(self):
        s = ReminderStore(self.path)
        s.add("x")
        self.assertTrue(os.path.isfile(self.path))

    def test_id_continuity_across_reload(self):
        s = ReminderStore(self.path)
        a = s.add("a")
        b = s.add("b")
        s2 = ReminderStore(self.path)
        c = s2.add("c")
        self.assertEqual((a["id"], b["id"], c["id"]), (1, 2, 3))

    def test_corrupt_file_degrades_to_empty(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as fh:
            fh.write("{this is not valid json")
        s = ReminderStore(self.path)
        self.assertEqual(len(s), 0)
        self.assertEqual(s.summarize(), "You have no reminders.")
        # And it is still usable.
        s.add("recover")
        self.assertEqual(len(ReminderStore(self.path)), 1)

    def test_non_list_json_degrades_to_empty(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump({"oops": 1}, fh)
        self.assertEqual(len(ReminderStore(self.path)), 0)

    def test_malformed_entries_are_skipped(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump([{"id": 1, "note": "keep"}, "junk",
                       {"note": "no id"}, {"id": 5, "note": "also keep"}], fh)
        s = ReminderStore(self.path)
        self.assertEqual([r["id"] for r in s.list()], [1, 5])
        # Next id continues after the max present id.
        self.assertEqual(s.add("new")["id"], 6)

    def test_in_memory_store_writes_nothing(self):
        s = ReminderStore(None)
        self.assertIsNone(s.path)
        s.add("x")
        self.assertEqual(len(s), 1)


class SummarizeTests(unittest.TestCase):
    def _fresh(self):
        return ReminderStore(None)

    def test_zero(self):
        self.assertEqual(self._fresh().summarize(), "You have no reminders.")

    def test_one(self):
        s = self._fresh()
        s.add("take out the trash")
        self.assertEqual(s.summarize(), "You have 1 reminder: take out the trash.")

    def test_two(self):
        s = self._fresh()
        s.add("a")
        s.add("b")
        self.assertEqual(s.summarize(), "You have 2 reminders. 1. a. 2. b.")

    def test_many(self):
        s = self._fresh()
        s.add("a")
        s.add("b")
        s.add("c")
        self.assertEqual(s.summarize(), "You have 3 reminders. 1. a. 2. b. 3. c.")

    def test_blank_note_falls_back_to_label(self):
        s = self._fresh()
        s.add("")
        self.assertEqual(s.summarize(), "You have 1 reminder: reminder.")


class InfoExecutorRemindersTests(unittest.TestCase):
    def _req(self):
        return ActionRequest(intent="what_reminders", category="info",
                             slots={}, action_code="query.reminders",
                             reply_text="", dry_run=True, confidence=0.99)

    def test_reads_from_store_even_in_dry_run(self):
        store = ReminderStore(None)
        store.add("take out the trash")
        store.add("call Mom")
        ex = InfoExecutor(dry_run=True, reminders_fn=store.summarize)
        res = ex.run(self._req())
        self.assertTrue(res.ok)
        self.assertEqual(res.payload["answer"],
                         "You have 2 reminders. 1. take out the trash. 2. call Mom.")
        self.assertFalse(res.side_effects)

    def test_no_store_returns_neutral_line(self):
        ex = InfoExecutor(dry_run=True)
        res = ex.run(self._req())
        self.assertEqual(res.payload["answer"], "You have no reminders.")

    def test_store_exception_is_fail_soft(self):
        def boom():
            raise RuntimeError("disk on fire")
        ex = InfoExecutor(dry_run=True, reminders_fn=boom)
        res = ex.run(self._req())
        self.assertTrue(res.ok)
        self.assertIn("Could not read reminders", res.payload["answer"])


class ReminderExecutorStoreTests(unittest.TestCase):
    def _req(self, note="take out the trash"):
        return ActionRequest(intent="remind", category="remind",
                             slots={"note": note}, action_code="reminder.create",
                             reply_text="", dry_run=False, confidence=0.99)

    def test_live_add_persists_to_store(self):
        store = ReminderStore(None)
        ex = ReminderExecutor(dry_run=False, store=store)
        res = ex.run(self._req("buy milk"))
        self.assertTrue(res.ok)
        self.assertTrue(res.side_effects)
        self.assertEqual([r["note"] for r in store.list()], ["buy milk"])

    def test_no_store_fails_soft(self):
        ex = ReminderExecutor(dry_run=False)
        res = ex.run(self._req())
        self.assertFalse(res.ok)
        self.assertIn("no reminder store", res.detail)


class FacadeWhatRemindersTests(unittest.TestCase):
    """The full path: model result -> decode -> orchestrator -> spoken reply."""

    def _process(self, store):
        pipe = FacadePipeline(dry_run=True, reminder_store=store)
        return pipe.process({"intent": "what_reminders",
                             "intent_confidence": 0.95, "slots": {}})

    def test_decodes_to_query_reminders(self):
        decoded = decode("what_reminders", {}, 0.95, threshold=0.75, dry_run=True)
        self.assertIsInstance(decoded, ActionRequest)
        self.assertEqual(decoded.action_code, "query.reminders")
        self.assertEqual(decoded.category, "info")

    def test_spoken_reply_is_single_source_of_truth(self):
        store = ReminderStore(None)
        store.add("take out the trash")
        store.add("call Mom")
        res = self._process(store)
        self.assertTrue(res.handled)
        self.assertIsNone(res.reject)
        self.assertEqual(res.execution.payload["answer"],
                         "You have 2 reminders. 1. take out the trash. 2. call Mom.")
        self.assertEqual(res.reply_text, res.execution.payload["answer"])

    def test_empty_store_spoken(self):
        res = self._process(ReminderStore(None))
        self.assertEqual(res.reply_text, "You have no reminders.")

    def test_pipeline_without_store_falls_back(self):
        pipe = FacadePipeline(dry_run=True)
        res = pipe.process({"intent": "what_reminders",
                            "intent_confidence": 0.95, "slots": {}})
        self.assertEqual(res.reply_text, "You have no reminders.")


class DefaultOrchestratorWiringTests(unittest.TestCase):
    def test_store_is_reachable_from_both_executors(self):
        store = ReminderStore(None)
        orch = default_orchestrator(dry_run=False, reminder_store=store)
        # The info executor must read from the same store.
        info = orch._executors["info"]
        self.assertEqual(info.reminders_fn, store.summarize)
        # The remind executor must write to the same store.
        remind = orch._executors["remind"]
        self.assertIs(remind.store, store)

    def test_no_store_leaves_info_fn_none(self):
        orch = default_orchestrator(dry_run=False)
        self.assertIsNone(orch._executors["info"].reminders_fn)
        self.assertIsNone(orch._executors["remind"].store)


if __name__ == "__main__":
    unittest.main()
