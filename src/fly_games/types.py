"""Shared types for platforms, games, and the lab engine."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Action:
    key: str
    label: str
    buttons: tuple[str, ...]
    control: Any = None


@dataclass(frozen=True)
class Stat:
    label: str
    value: str


@dataclass
class AdvanceResult:
    frame: Any
    info: dict
    reward: float
    terminated: bool
    truncated: bool
    frames_executed: int
    samples: list = field(default_factory=list)


@dataclass(frozen=True)
class GameMeta:
    id: str
    title: str
    short_name: str
    platform_id: str
    platform_label: str
    cartridge: str
    subtitle: str
    aspect_width: int
    aspect_height: int
    button_keys: tuple[str, ...]
    hold_frames_options: tuple[int, ...] = (1, 2, 4, 8, 12, 16)
    default_hold_frames: int = 4
    default_seed: int = 123
    # Fly-specific: how many brain steps to settle into a decision (dt=20 ms each).
    brain_steps: int = 24
    # Brain warm-up steps at episode start (silence -> settled baseline).
    brain_warmup: int = 80
