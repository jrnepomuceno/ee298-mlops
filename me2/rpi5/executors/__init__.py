"""Category executors for the Pi5-VCM orchestrator.

Each executor owns one functional category (lights, hvac, media, timer,
remind, comms, info) and implements :meth:`Executor.run`. All ship in
``dry_run=True`` mode: they validate, log, and return success without touching
hardware or the network. Flipping a category to live execution is a one-line
change in that executor, never in the orchestrator or the model.
"""
from __future__ import annotations

from .base import ExecutionResult, Executor
from .light import LightExecutor
from .hvac import HvacExecutor
from .media import MediaExecutor
from .timer import TimerExecutor
from .reminder import ReminderExecutor
from .comms import CommsExecutor
from .info import InfoExecutor
from .volume import VolumeExecutor

__all__ = [
    "ExecutionResult", "Executor",
    "LightExecutor", "HvacExecutor", "MediaExecutor",
    "TimerExecutor", "ReminderExecutor", "CommsExecutor", "InfoExecutor",
    "VolumeExecutor",
]
