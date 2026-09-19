"""Fly-specific policy for Mario.

Validated against the live connectome (probe):
  resting            -> punch_L~8Hz  (the "proceed" state)
  loom+threat L      -> escape_L~18Hz
  loom+threat R      -> escape_R~42Hz
  chase L            -> steer_L~6Hz
  chase R            -> steer_R~4Hz

Mario only faces right, so it expresses two decisions: **proceed** (resting,
walk) and **escape** (hazard -> jump). The escape decision is read from the
giant-fiber (escape) descending rate.
"""

from __future__ import annotations

import numpy as np

from fly_games.brain.encoder import SensoryEncoder

FORWARD = (1.0, 0.0)
HORIZON = 160.0


class MarioFlyEncoder(SensoryEncoder):
    def encode(self, observation: dict) -> tuple[list, dict]:
        events = []
        terrain = (observation.get("terrain") or {}).get("summary", {})

        # Enemy hazards in front.
        for obj in observation.get("nearby_objects") or []:
            if obj.get("defeated") or obj.get("kind") == "flagpole":
                continue
            dx = float(obj.get("dx", 0))
            dy = float(obj.get("dy", 0))
            if dx < -16 or dx >= HORIZON:
                continue
            prox = np.clip(1.0 - dx / HORIZON, 0.0, 1.0)
            contact = obj.get("estimated_contact_in_frames")
            urgency = np.clip(1.0 - float(contact) / 90.0, 0.0, 1.0) if contact is not None else prox * 0.4
            events += self.side_events((dx, dy), FORWARD,
                                       loom=prox, threat=urgency, shot=0.0, chase=0.0)

        # Gap ahead -> jump.
        gap = terrain.get("nearest_empty_column_below_feet")
        if gap:
            edge = float(gap.get("edge_distance_px", 120))
            if 0 < edge < HORIZON:
                prox = float(np.clip(1.0 - edge / HORIZON, 0.0, 1.0))
                events += self.side_events((edge, 0.0), FORWARD,
                                           loom=prox * 1.2, threat=prox * 0.8, shot=0.0, chase=0.0)

        # Wall ahead -> jump to climb.
        wall = terrain.get("nearest_obstacle")
        if wall:
            dist = float(wall.get("distance_px", 120))
            if 0 < dist < HORIZON:
                prox = float(np.clip(1.0 - dist / HORIZON, 0.0, 1.0))
                events += self.side_events((dist, 0.0), FORWARD,
                                           loom=prox * 1.0, threat=0.0, shot=0.0, chase=0.0)

        return self.events_to_inject(events)


def coarse_actions() -> tuple[str, ...]:
    """The fly's decision vocabulary.

    This used to be `("proceed", "escape")`, which meant all twelve descending
    motor urges were squeezed through two labels into two button combos: run
    right, or run right while jumping. `backward_*` and `steer_*` were computed
    every decision and thrown away, so the fly could not express stopping or
    backing off no matter what it was doing - and backing off is the only way
    past a pipe you are already touching.
    """
    return ("proceed", "escape", "retreat", "halt")


def coarse_of(fine_action: str, observation: dict) -> str:
    """Train label: which urge the teacher acted on.

    The old version was `"escape" if "jump" in fine else "proceed"`, which made
    the readout a jump detector rather than a policy.
    """
    if "left" in fine_action:
        return "retreat"
    if fine_action in ("wait", "up", ""):
        return "halt"
    if "jump" in fine_action:
        return "escape"
    return "proceed"


def expand(coarse: str, observation: dict, fine_hint: str | None = None) -> str:
    if coarse == "retreat":
        return fine_hint if fine_hint in ("left", "left_run") else "left_run"
    if coarse == "halt":
        return "wait"
    if coarse == "escape":
        return fine_hint if fine_hint in ("right_jump", "jump", "right_run_jump") else "right_run_jump"
    return "right_run"


def hand_decode(command: dict, observation: dict) -> tuple[str, dict]:
    """Read the fly's escape urge as a Mario decision.

    Deliberately still two-sided, even though the vocabulary is now four. Every
    attempt to let this decoder use `retreat` and `halt` made Mario worse, and
    the reason is worth recording: rates in a 24-step window are quantised at
    2.08 Hz per spike, and across all twelve command groups the fly fires about
    three spikes at rest, so `forward` is zero on nearly every decision. Any
    rule that reads silence as "halt" therefore stops him dead - 656 of 708
    frames walking nowhere - and a low threshold on `backward` turns one stray
    spike into 503 frames of walking left. "proceed" as the unconditional
    residual is what made this decoder work, not an accident of it.

    The four-label space is for the fitted readout, which can learn rates from
    data instead of me guessing thresholds. The hand rule reads two of the four
    and says so.
    """
    escape = max((command.get("escape_L") or {}).get("rate", 0.0),
                 (command.get("escape_R") or {}).get("rate", 0.0))
    names = coarse_actions()
    winner = "escape" if escape > 1.5 else "proceed"
    other = (1.0 - 0.8) / (len(names) - 1)
    return winner, {n: (0.8 if n == winner else other) for n in names}


__all__ = ["FORWARD", "MarioFlyEncoder", "coarse_actions", "coarse_of", "expand", "hand_decode"]
