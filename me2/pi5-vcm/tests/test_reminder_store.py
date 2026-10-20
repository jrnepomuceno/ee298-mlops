from __future__ import annotations

import json
from datetime import datetime

import pytest

from inference.pi5_runner.reminder_store import (
    JsonReminderStore,
    ReminderStoreError,
)


def test_missing_store_is_empty_and_add_creates_parent_dirs(tmp_path) -> None:
    store_path = tmp_path / "nested" / "config" / "reminders.json"
    store = JsonReminderStore(store_path)

    assert store.list() == []
    reminder = store.add("  Drink water  ")

    assert reminder.id == 1
    assert reminder.action == "Drink water"
    assert datetime.fromisoformat(reminder.created_at).utcoffset() is not None
    assert store_path.is_file()
    assert list(store_path.parent.glob("*.tmp")) == []


def test_store_assigns_monotonic_ids_and_lists_newest_first(tmp_path) -> None:
    store = JsonReminderStore(tmp_path / "reminders.json")

    first = store.add("Drink water")
    second = store.add("Study")

    assert (first.id, second.id) == (1, 2)
    assert store.list() == [second, first]
    assert json.loads(store.path.read_text(encoding="utf-8")) == [
        {
            "id": 1,
            "action": "Drink water",
            "created_at": first.created_at,
        },
        {
            "id": 2,
            "action": "Study",
            "created_at": second.created_at,
        },
    ]


def test_store_rejects_empty_actions(tmp_path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        JsonReminderStore(tmp_path / "reminders.json").add("  ")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{", "invalid JSON"),
        ('{"id": 1}', "must contain a JSON array"),
        ('[{"id": true}]', "invalid fields"),
        (
            '[{"id": 1, "action": "Study", "created_at": "2026-10-20T08:00:00"}]',
            "must include a timezone",
        ),
        (
            '[{"id": 1, "action": "Study", "created_at": "2026-10-20T08:00:00+08:00"},'
            '{"id": 1, "action": "Exercise", '
            '"created_at": "2026-10-20T08:01:00+08:00"}]',
            "duplicate id",
        ),
    ],
)
def test_store_reports_invalid_persisted_data(
    tmp_path,
    payload: str,
    message: str,
) -> None:
    store_path = tmp_path / "reminders.json"
    store_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ReminderStoreError, match=message):
        JsonReminderStore(store_path).list()


def test_store_uses_environment_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    store_path = tmp_path / "custom.json"
    monkeypatch.setenv("PI5_VCM_REMINDERS_FILE", str(store_path))

    store = JsonReminderStore()
    reminder = store.add("Exercise")

    assert store.path == store_path
    assert store.list() == [reminder]
