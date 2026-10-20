"""Persistent local storage for undated action reminders."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_REMINDERS_PATH = Path.home() / ".config" / "pi5-vcm" / "reminders.json"
REMINDERS_PATH_ENV = "PI5_VCM_REMINDERS_FILE"


@dataclass(frozen=True)
class Reminder:
    id: int
    action: str
    created_at: str


class ReminderStoreError(RuntimeError):
    """Raised when persisted reminder data is invalid or cannot be accessed."""


class JsonReminderStore:
    """Store reminders as an atomically replaced JSON document."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            configured_path = os.environ.get(REMINDERS_PATH_ENV)
            path = (
                Path(configured_path).expanduser()
                if configured_path
                else DEFAULT_REMINDERS_PATH
            )
        self.path = path.expanduser()

    def add(self, action: str) -> Reminder:
        normalized_action = action.strip()
        if not normalized_action:
            raise ValueError("reminder action must not be empty")

        reminders = self.list()
        reminder = Reminder(
            id=max((item.id for item in reminders), default=0) + 1,
            action=normalized_action,
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        self._write([*reminders, reminder])
        return reminder

    def list(self) -> list[Reminder]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except UnicodeDecodeError as exc:
            raise ReminderStoreError(
                f"reminder store {self.path} is not valid UTF-8"
            ) from exc
        except OSError as exc:
            raise ReminderStoreError(
                f"could not read reminder store {self.path}: {exc}"
            ) from exc

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ReminderStoreError(
                f"reminder store {self.path} contains invalid JSON"
            ) from exc
        if not isinstance(data, list):
            raise ReminderStoreError(
                f"reminder store {self.path} must contain a JSON array"
            )

        reminders: list[Reminder] = []
        seen_ids: set[int] = set()
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ReminderStoreError(
                    f"reminder store entry {index} must be an object"
                )
            reminder_id = item.get("id")
            action = item.get("action")
            created_at = item.get("created_at")
            if (
                isinstance(reminder_id, bool)
                or not isinstance(reminder_id, int)
                or reminder_id < 1
                or not isinstance(action, str)
                or not action.strip()
                or not isinstance(created_at, str)
            ):
                raise ReminderStoreError(
                    f"reminder store entry {index} has invalid fields"
                )
            if reminder_id in seen_ids:
                raise ReminderStoreError(
                    f"reminder store contains duplicate id {reminder_id}"
                )
            seen_ids.add(reminder_id)
            try:
                created_time = datetime.fromisoformat(created_at)
            except ValueError as exc:
                raise ReminderStoreError(
                    f"reminder store entry {index} has an invalid created_at"
                ) from exc
            if created_time.utcoffset() is None:
                raise ReminderStoreError(
                    f"reminder store entry {index} created_at must include a timezone"
                )
            reminders.append(
                Reminder(
                    id=reminder_id,
                    action=action,
                    created_at=created_at,
                )
            )
        return sorted(reminders, key=lambda item: item.id, reverse=True)

    def _write(self, reminders: list[Reminder]) -> None:
        payload = json.dumps(
            [
                {
                    "id": reminder.id,
                    "action": reminder.action,
                    "created_at": reminder.created_at,
                }
                for reminder in reminders
            ],
            ensure_ascii=False,
            indent=2,
        )
        temporary_path: Path | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(payload)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.path)
        except OSError as exc:
            cleanup_error: OSError | None = None
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as error:
                    cleanup_error = error
            if cleanup_error is not None:
                raise ReminderStoreError(
                    f"could not write reminder store {self.path}: {exc}; "
                    f"temporary file cleanup also failed: {cleanup_error}"
                ) from cleanup_error
            raise ReminderStoreError(
                f"could not write reminder store {self.path}: {exc}"
            ) from exc
