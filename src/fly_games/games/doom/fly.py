"""Fly-specific policy for Doom.

The marine is a first-person character that can walk, turn, and shoot. The fly
expresses four decisions via its descending output:
  * proceed       -> resting; the expand walks forward.
  * escape        -> a monster is a threat -> shoot / back off.
  * turn_left     -> turn left (monster/goal on the left).
  * turn_right    -> turn right (monster/goal on the right).

The encoder drives the fly's visual channels from the nearest monster /
hazard: a close monster -> loom+threat (escape) plus chase in the monster's
screen side (steer toward it).
"""

from __future__ import annotations

import numpy as np

from fly_games.brain.encoder import SensoryEncoder

SCREEN_CX = 320
SCREEN_HALF = 160


class DoomFlyEncoder(SensoryEncoder):
    def encode(self, observation: dict) -> tuple[list, dict]:
        events = []
        combat = observation.get("combat") or {}
        target = observation.get("best_target")
        objs = observation.get("nearby_objects") or []

        def screen_side(obj) -> float:
            # 0..1 across the screen, from the marine's view. 0.5 = centered.
            sx = obj.get("screen_x")
            if sx is None:
                # fall back to dx relative to facing angle (screen_x = 320 + dx*scale)
                dx = float(obj.get("dx", 0.0) or 0.0)
                return dx / 128.0
            return float(sx) / 640.0

        def side_from_screen(sx: float):
            if sx < 0.42:
                return "L"
            if sx > 0.58:
                return "R"
            return "both"

        def add_monster(obj, threat: float, chase: float):
            sx = screen_side(obj)
            side = side_from_screen(sx)
            # loom: a close monster
            prox = np.clip(1.0 - (obj.get("distance") or 128.0) / 160.0, 0.0, 1.0)
            loom = np.clip(prox * 1.2, 0.0, 1.0)
            if side == "both":
                # Dead ahead: drive both eyes equally, at reduced strength.
                events.extend(self.side_events((0.0, 0.0), (1.0, 0.0),
                                               loom=loom * 0.8, threat=threat * 0.8,
                                               shot=0.0, chase=chase * 0.5))
            else:
                # Off-centre on the 640 px screen -> a lateral offset in the
                # fly's frame, so L and R drive different descending neurons.
                offset = SCREEN_HALF * (sx - 0.5)
                events.extend(self.side_events((offset, 0.0), (1.0, 0.0),
                                               loom=loom, threat=threat,
                                               shot=0.5 if threat > 0.5 else 0.0, chase=chase))

        # Best live monster.
        if target and target.get("role") == "monster":
            dist = float(target.get("distance", 128) or 128)
            will_shoot = combat.get("someone_will_shoot_you") or combat.get("someone_will_eat_you")
            threat = 1.0 if (will_shoot or dist < 72) else 0.6 if dist < 128 else 0.2
            add_monster(target, threat=threat, chase=0.8)

        # Any live monster on screen (secondary).
        for obj in objs:
            if obj.get("role") != "monster" or not obj.get("on_screen"):
                continue
            dist = float(obj.get("distance", 128) or 128)
            if dist >= 90 or obj.get("defeated"):
                continue
            threat = 0.5 if combat.get("someone_will_shoot_you") else 0.3
            add_monster(obj, threat=threat, chase=0.5)

        # Hazard (barrel / mine) -> avoid (loom only).
        for obj in objs:
            if obj.get("role") != "hazard" or not obj.get("on_screen"):
                continue
            dist = float(obj.get("distance", 128) or 128)
            if dist >= 60:
                continue
            sx = screen_side(obj)
            loom = np.clip(1.0 - dist / 80.0, 0.0, 1.0)
            events.extend(self.side_events((SCREEN_HALF * (sx - 0.5), 0.0), (1.0, 0.0),
                                           loom=loom, threat=loom * 0.3,
                                           shot=0.0, chase=0.0))

        return self.events_to_inject(events)


def _threat(observation: dict) -> bool:
    combat = observation.get("combat") or {}
    return bool(
        combat.get("someone_will_shoot_you")
        or combat.get("someone_will_eat_you")
        or combat.get("will_eat_you")
    )


def _monster_side(observation: dict) -> str:
    """'L' / 'R' / 'center' from the best target's screen position."""
    target = observation.get("best_target") or observation.get("nearest") or {}
    sx = target.get("screen_x")
    if sx is None:
        dx = target.get("dx", 0.0) or 0.0
        sx = 320.0 + dx * 2.0
    if sx < 200:
        return "L"
    if sx > 440:
        return "R"
    return "center"


def coarse_actions() -> tuple[str, ...]:
    return ("proceed", "escape", "turn_left", "turn_right")


def coarse_of(fine_action: str, observation: dict) -> str:
    if _threat(observation):
        return "escape"
    side = _monster_side(observation)
    if side == "L":
        return "turn_left"
    if side == "R":
        return "turn_right"
    return "proceed"


def expand(coarse: str, observation: dict, fine_hint: str | None = None) -> str:
    if coarse == "escape":
        return fine_hint if fine_hint and fine_hint != "wait" else "shoot"
    if coarse == "turn_left":
        return fine_hint if fine_hint in ("turn_left", "forward_left") else "turn_left"
    if coarse == "turn_right":
        return fine_hint if fine_hint in ("turn_right", "forward_right") else "turn_right"
    return fine_hint if fine_hint in ("forward", "wait") else "forward"


def hand_decode(command: dict, observation: dict) -> tuple[str, dict]:
    escape = max(command.get("escape_L", {}).get("rate", 0.0),
                 command.get("escape_R", {}).get("rate", 0.0))
    steer_l = command.get("steer_L", {}).get("rate", 0.0)
    steer_r = command.get("steer_R", {}).get("rate", 0.0)
    # Always answer over all four decisions: the UI shows the whole distribution
    # and the engine looks the winning label up in it, so a branch that omits a
    # label reads as a missing decision rather than a zero.
    if escape > 1.5:
        return "escape", {"escape": 0.8, "proceed": 0.1, "turn_left": 0.05, "turn_right": 0.05}
    if steer_l > steer_r and steer_l > 2.0:
        return "turn_left", {"turn_left": 0.7, "proceed": 0.2, "turn_right": 0.05, "escape": 0.05}
    if steer_r > steer_l and steer_r > 2.0:
        return "turn_right", {"turn_right": 0.7, "proceed": 0.2, "turn_left": 0.05, "escape": 0.05}
    return "proceed", {"proceed": 0.75, "escape": 0.1, "turn_left": 0.075, "turn_right": 0.075}


__all__ = ["SCREEN_CX", "DoomFlyEncoder", "coarse_actions", "coarse_of", "expand", "hand_decode"]
