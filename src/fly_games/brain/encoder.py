"""Sensory encoders: map structured game facts onto the fly's own sensory
neurons. This is the *input* side of the fly-brain loop.

The fly's real visual system processes **features**, not raw pixels (see
flybrain/eyes.py and the fly.ai README: the raw photoreceptor signal dies at
the lamina in this LIF model). So we drive the fly's identified visual
projection neuron types:

  channel   neuron type(s)   responds to (in a real fly)
  --------  --------------  ------------------------------------------------
  loom      LPLC2            something getting bigger / approaching
  threat    LC4              fast looming, escape (strongly drives DNp01)
  shot      LPLC1            small approaching objects / projectiles
  chase     LC10a            the moving target the fly pursues

Each channel is driven on the fly's **left** and/or **right** side depending on
where the stimulus is, using a geometric L/R assignment from the fly's forward
direction. Magnitudes follow fly.ai's ENCODER defaults (capped at 0.8).

`SensoryEncoder` resolves the neuron indices for each channel/side once and
provides `events_to_inject(events)` -> (inject_list, input_label). Each game's
`fly.py` subclasses it and implements `encode(observation)` returning `events`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .client import CHANNELS, COMMAND_GROUPS, FlyBrainClient

# Channel magnitude parameters (from fly.ai eyes.ENCODER).
ENCODER_PARAMS = {
    "loom_gain": 10.0,    # not directly used (we use a size/proximity proxy)
    "loom_size": 0.0,
    "chase_base": 0.6,    # LC10a voltage whenever the target is visible
    "chase_gain": 0.2,    # plus this much per unit of angle
    "threat_max": 0.8,    # LC4 voltage when the threat is at close range
    "shot_gain": 10.0,
    "cap": 0.8,           # max voltage any channel adds per step
}


def lateral_sides(
    offset: Sequence[float],
    forward: Sequence[float],
    *,
    behind: float = -24.0,
    both: float = 16.0,
) -> list[str]:
    """Return which fly sides ("L"/"R") a stimulus at `offset` relative to the fly
    should drive, given the fly's `forward` unit vector (screen coords: x right,
    y down).

    left_vec = (fy, -fx) (when forward=(1,0) [facing right], the fly's left is
    up = (0,-1)); right_vec = -left_vec. The stimulus is "on the left" when its
    projection onto left_vec is positive and past the `both` threshold.
    """
    fx, fy = float(forward[0]), float(forward[1])
    dx, dy = float(offset[0]), float(offset[1])
    # normalize forward
    n = math.hypot(fx, fy) or 1.0
    fx, fy = fx / n, fy / n
    left_vec = (fy, -fx)
    right_vec = (-fy, fx)
    along = dx * fx + dy * fy
    lat_left = dx * left_vec[0] + dy * left_vec[1]
    lat_right = dx * right_vec[0] + dy * right_vec[1]
    # behind: stimulus is behind the fly.
    if along < behind:
        return ["L", "R"]
    if lat_left > lat_right and lat_left > both:
        return ["L"]
    if lat_right > lat_left and lat_right > both:
        return ["R"]
    return ["L", "R"]


class SensoryEncoder:
    def __init__(self, client: FlyBrainClient) -> None:
        self.client = client
        b = client.brain
        self._cells: dict[str, dict[str, np.ndarray]] = {}
        for channel, types in CHANNELS.items():
            self._cells[channel] = {}
            for side in ("L", "R"):
                self._cells[channel][side] = b.cells(types, side=side)
        self._params = dict(ENCODER_PARAMS)
        self._size_cache: dict[str, float] = {}

    def _idx(self, channel: str, side: str) -> np.ndarray:
        return self._cells.get(channel, {}).get(side, np.empty(0, dtype=np.int64))

    def _clip(self, value: float) -> float:
        p = self._params
        value = float(value)
        return max(0.0, min(float(p["cap"]), value))

    def _clip_amount(self, value: float) -> float:
        return self._clip(value)

    def _angle(self, dx: float, size_px: float) -> float:
        """Angular size in the eyes.py convention: size / max(|dx|, 8)."""
        return size_px / max(abs(dx), 8.0)

    def _events(self, events: list[tuple[str, str, float, str | None]]) -> tuple[list, dict]:
        """Convert (channel, side, amount, label) tuples to the flybrain
        `inject` list and a flat human-readable input-label dict."""
        inject: list[tuple[np.ndarray, float]] = []
        input_label: dict[str, float] = {}
        for channel, side, amount, label in events:
            idx = self._idx(channel, side)
            if idx is None or idx.size == 0:
                continue
            amt = self._clip_amount(amount)
            if amt <= 0.0:
                continue
            inject.append((idx, amt))
            key = (label or f"{channel}_{side}").replace("-", "_")
            # combine duplicate labels across sides (sum).
            input_label[key] = input_label.get(key, 0.0) + round(amt, 3)
        return inject, input_label

    def events_to_inject(self, events: Sequence[tuple[str, str, float, str | None]]) -> tuple[list, dict]:
        return self._events(list(events))

    # --- generic per-side helpers --------------------------------------------------

    def side_events(
        self,
        offset: Sequence[float],
        forward: Sequence[float],
        loom: float,
        threat: float,
        shot: float,
        chase: float,
        *,
        side_label: str = "event",
    ) -> list[tuple[str, str, float, str | None]]:
        events = []
        for side in lateral_sides(offset, forward):
            if loom > 0:
                events.append(("loom", side, loom, f"looms_{side}"))
            if threat > 0:
                events.append(("threat", side, threat, f"threat_{side}"))
            if shot > 0:
                events.append(("shot", side, shot, f"shot_{side}"))
            if chase > 0:
                events.append(("chase", side, chase, f"chase_{side}"))
        return events

    def encode(self, observation: dict) -> tuple[list, dict]:
        raise NotImplementedError


# Convenience for the engine: a no-op encoder (used if a game has no fly mapping yet).
class NullEncoder(SensoryEncoder):
    def encode(self, observation: dict) -> tuple[list, dict]:
        return [], {}


__all__ = [
    "CHANNELS",
    "COMMAND_GROUPS",
    "ENCODER_PARAMS",
    "NullEncoder",
    "SensoryEncoder",
    "lateral_sides",
]
