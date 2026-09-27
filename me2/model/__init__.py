"""model package: VCM architecture + training code.

Keeps ``from model import VCM`` working after the move into the model/
subfolder, while staying lazy so that importing a torch-free submodule
(e.g. ``model.slots``) does NOT pull in torch. This matters on the Raspberry
Pi 5, whose venv has no torch: the on-device ONNX path imports
``model.slots.parse_slots`` and must not trigger the torch-dependent VCM.
"""
from __future__ import annotations

import importlib as _importlib

__all__ = ["VCM", "ctc_decode_batch", "parse_slots"]


def __getattr__(name):
    # Lazy re-export of the torch-dependent symbols from model.model.
    if name in ("VCM", "ctc_decode_batch"):
        mod = _importlib.import_module(".model", __name__)
        return getattr(mod, name)
    # parse_slots is torch-free; keep it off the torch path entirely.
    if name == "parse_slots":
        mod = _importlib.import_module(".slots", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
