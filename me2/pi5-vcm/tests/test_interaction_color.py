from __future__ import annotations

import json

import pytest

from inference.pi5_runner.interaction_color import (
    COLOR_HEX,
    DEFAULT_INTERACTION_COLOR,
    InteractionColorStore,
    InteractionColorStoreError,
)


def test_store_defaults_to_blue_and_persists_selected_color(tmp_path) -> None:
    path = tmp_path / "config" / "interaction-color.json"
    store = InteractionColorStore(path)

    assert store.load() == DEFAULT_INTERACTION_COLOR == "Blue"

    store.save("Red")

    assert InteractionColorStore(path).load() == "Red"
    assert json.loads(path.read_text(encoding="utf-8")) == {"color": "Red"}


@pytest.mark.parametrize("color", ["Red", "Blue", "Green"])
def test_color_hex_palette_is_bounded(color: str) -> None:
    assert color in COLOR_HEX
    assert len(COLOR_HEX[color]) == 6


def test_color_hex_palette_uses_rgb_values() -> None:
    assert COLOR_HEX == {
        "Red": "ff0000",
        "Blue": "0000ff",
        "Green": "00ff00",
    }


def test_store_rejects_unknown_color(tmp_path) -> None:
    store = InteractionColorStore(tmp_path / "interaction-color.json")

    with pytest.raises(ValueError, match="unsupported interaction color"):
        store.save("Purple")


@pytest.mark.parametrize(
    "contents",
    ["{", "[]", '{"color": "Purple"}', '{"color": []}'],
)
def test_store_reports_invalid_persisted_color(tmp_path, contents: str) -> None:
    path = tmp_path / "interaction-color.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(InteractionColorStoreError):
        InteractionColorStore(path).load()
