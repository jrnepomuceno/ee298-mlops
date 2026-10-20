"""Persistent preference for the wake and reply indicator color."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

COLOR_HEX = {
    "Red": "ff0000",
    "Blue": "0000ff",
    "Green": "00ff00",
}
COLOR_NAMES = tuple(COLOR_HEX)
DEFAULT_INTERACTION_COLOR = "Blue"
DEFAULT_COLOR_PATH = Path.home() / ".config" / "pi5-vcm" / "interaction-color.json"
COLOR_PATH_ENV = "PI5_VCM_INTERACTION_COLOR_FILE"


class InteractionColorStoreError(RuntimeError):
    """Raised when the saved interaction color cannot be read or written."""


class InteractionColorStore:
    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            configured_path = os.environ.get(COLOR_PATH_ENV)
            path = (
                Path(configured_path).expanduser()
                if configured_path
                else DEFAULT_COLOR_PATH
            )
        self.path = path.expanduser()

    def load(self) -> str:
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return DEFAULT_INTERACTION_COLOR
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InteractionColorStoreError(
                f"could not read interaction color from {self.path}: {exc}"
            ) from exc
        color = document.get("color") if isinstance(document, dict) else None
        if not isinstance(color, str) or color not in COLOR_HEX:
            raise InteractionColorStoreError(
                f"interaction color file {self.path} has an unsupported color"
            )
        return color

    def save(self, color: str) -> None:
        if color not in COLOR_HEX:
            raise ValueError(f"unsupported interaction color: {color!r}")
        payload = json.dumps({"color": color}, indent=2) + "\n"
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
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.path)
        except OSError as exc:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    raise InteractionColorStoreError(
                        f"could not save interaction color to {self.path}: {exc}; "
                        f"temporary file cleanup failed: {cleanup_error}"
                    ) from cleanup_error
            raise InteractionColorStoreError(
                f"could not save interaction color to {self.path}: {exc}"
            ) from exc
