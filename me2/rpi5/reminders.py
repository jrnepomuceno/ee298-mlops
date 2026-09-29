"""Persistent reminder store.

A tiny, offline, stdlib-only store for reminders. Each reminder is a record
with an id, a note, a creation timestamp, and an (optional) due date. In the
current scope reminders have *indefinite* due dates (``due_at`` stays ``None``)
and the note is free-form text supplied by the caller -- the store does not
care where it came from.

Design goals
------------
* **Offline.** Pure local JSON; no network, no third-party deps.
* **Fail-soft.** A corrupt or unreadable file degrades to an empty store
  instead of raising, so a bad file can never crash the mic loop.
* **Atomic writes.** We write to a temp file in the same directory and
  ``os.replace`` it into place, so a crash mid-write can't truncate the file.
* **Spoken-ready.** :meth:`summarize` renders a single line suitable for TTS.
"""
from __future__ import annotations

import json
import re
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List

DEFAULT_PATH = os.path.join("~", ".me2", "reminders.json")


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ReminderStore:
    """A JSON-backed list of reminders with a spoken summary.

    Parameters
    ----------
    path:
        Where the JSON file lives. ``~`` is expanded. Defaults to
        :data:`DEFAULT_PATH`. Pass ``None`` for an in-memory store (nothing is
        written to disk) -- useful for tests.
    """

    def __init__(self, path: str | None = DEFAULT_PATH) -> None:
        self.path = os.path.expanduser(path) if path else None
        self._lock = threading.Lock()
        self._records: List[Dict[str, Any]] = []
        self._next_id = 1
        self._load()

    # ------------------------------------------------------------------ #
    # Loading / saving
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        """Load records from disk, tolerating a missing or corrupt file."""
        if self.path is None or not os.path.isfile(self.path):
            self._records = []
            self._next_id = 1
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                raise ValueError("expected a JSON list of reminders")
            cleaned: List[Dict[str, Any]] = []
            max_id = 0
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                rid = entry.get("id")
                if not isinstance(rid, int):
                    continue
                cleaned.append({
                    "id": rid,
                    "note": str(entry.get("note", "")),
                    "created_at": str(entry.get("created_at", "")),
                    "due_at": entry.get("due_at"),
                })
                max_id = max(max_id, rid)
            self._records = cleaned
            self._next_id = max_id + 1
        except Exception:  # noqa: BLE001 -- fail soft, never crash the loop
            self._records = []
            self._next_id = 1

    def _save(self) -> None:
        """Atomically persist the current records. No-op for in-memory stores."""
        if self.path is None:
            return
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._records, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            os.replace(tmp_path, self.path)
        except Exception:
            # Clean up the temp file if the swap failed; the original is intact.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean_note(note: str) -> str:
        """Strip ASR artifacts so the stored note reads naturally.

        The model emits ``<unk>`` for words it could not transcribe (e.g.
        "water <unk>" for "water the plants"). We drop those tokens and any
        dangling punctuation so the reminder is not stored/spoken with a raw
        ``<unk>`` marker.
        """
        text = str(note)
        # Remove explicit unknown-token markers in any casing.
        text = re.sub(r"<\s*unk\s*>", " ", text, flags=re.IGNORECASE)
        # Collapse whitespace.
        text = re.sub(r"\s+", " ", text).strip()
        # Trim stray leading/trailing punctuation left by a dropped token.
        text = text.strip(" ,.;:-")
        return text

    def add(self, note: str = "") -> Dict[str, Any]:
        """Add a reminder and persist it. Returns the new record."""
        note = self._clean_note(note)
        with self._lock:
            record = {
                "id": self._next_id,
                "note": note,
                "created_at": _utc_now_iso(),
                "due_at": None,
            }
            self._records.append(record)
            self._next_id += 1
            self._save()
            return dict(record)

    def remove(self, reminder_id: int) -> bool:
        """Remove a reminder by id. Returns True if one was removed."""
        with self._lock:
            before = len(self._records)
            self._records = [r for r in self._records if r["id"] != reminder_id]
            if len(self._records) == before:
                return False
            self._save()
            return True

    def clear(self) -> int:
        """Remove all reminders. Returns how many were removed."""
        with self._lock:
            count = len(self._records)
            self._records = []
            if count:
                self._save()
            return count

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def list(self) -> List[Dict[str, Any]]:
        """Return a copy of the current reminders, oldest first."""
        with self._lock:
            return [dict(r) for r in self._records]

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def summarize(self) -> str:
        """Render the current reminders as an *itemized* spoken list.

        Each reminder is read on its own numbered item so the user hears a
        clear list rather than one run-on sentence::

            You have no reminders.
            You have 1 reminder: take out the trash.
            You have 2 reminders. 1. take out the trash. 2. call Mom.
            You have 3 reminders. 1. a. 2. b. 3. c.
        """
        with self._lock:
            records = list(self._records)
        if not records:
            return "You have no reminders."
        n = len(records)
        notes = [r["note"].strip() for r in records]
        # Fall back to a generic label for any blank note so the item still
        # reads naturally (the model can't always fill the note).
        notes = [note if note else "reminder" for note in notes]
        if n == 1:
            return f"You have 1 reminder: {notes[0]}."
        items = ". ".join(f"{i}. {note}" for i, note in enumerate(notes, start=1))
        return f"You have {n} reminders. {items}."
