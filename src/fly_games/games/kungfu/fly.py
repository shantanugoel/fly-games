"""Fly-specific policy for Kung Fu / Spartan X.

Thomas can face left or right (odd floors go left, even go right). The fly expresses
two decisions via its descending output:
  * proceed  -> resting state (no hazard); the expand walks in the progress
                direction (from the observation).
  * escape   -> a hazard (enemy in range, projectile, grabbed) drives the
                giant-fiber escape pathway; the expand dodges (jump / crouch).

We deliberately do NOT try to make the fly "punch" or "kick": those are posture
the fly's visual-driven escape/resting states don't express. It dodges instead.
This is an honest basic survivor.
"""

from __future__ import annotations

from fly_games.brain.encoder import SensoryEncoder


class KungFuFlyEncoder(SensoryEncoder):
    def encode(self, observation: dict) -> tuple[list, dict]:
        events = []
        player = observation.get("player") or {}
        combat = observation.get("combat") or {}
        facing = player.get("facing") or "right"
        # Fly forward vector: Thomas faces left (-x) or right (+x).
        forward = (-1.0, 0.0) if facing == "left" else (1.0, 0.0)

        def add_hazard(dx, dy, loom, threat, shot):
            events.extend(self.side_events((dx, dy), forward,
                                           loom=loom, threat=threat, shot=shot, chase=0.0))

        grabbed = bool(player.get("grabbed"))

        # Incoming projectile (knife / boomerang) -> dodge (jump/crouch).
        if combat.get("incoming"):
            # A projectile is ahead of Thomas; place it in front of the fly.
            ahead_dx = 20.0
            add_hazard(ahead_dx, 0.0, loom=1.2, threat=1.0, shot=1.0)

        # Best target / nearest foe in fighting range. The observe pass can
        # legitimately fail to find either Thomas or a foe, and an absent
        # position must not become a phantom hazard at the origin - nor a crash.
        best = combat.get("best_target") or {}
        target = best or observation.get("nearest") or {}
        px, py = player.get("x"), player.get("y")
        tx = target.get("x", px)
        ty = target.get("y", py)
        known = None not in (px, py, tx, ty)
        tdx = float(tx - px) if known else 0.0
        tdy = float(ty - py) if known else 0.0
        # Will it hit / grab you?
        will_hit = combat.get("someone_will_hit_you") or combat.get("someone_will_grab_you")
        if known and will_hit and abs(tdx) < 60:
            add_hazard(tdx, tdy, loom=1.0, threat=1.0, shot=0.0)
        # Grabber in range -> must shrug (escape-ish: we map to dodge).
        if known and best.get("kind") == "grabber" and abs(tdx) < 40:
            add_hazard(tdx, tdy, loom=0.8, threat=0.8, shot=0.0)

        # Any foe that will hit you / grab you -> hazard.
        if known and combat.get("will_grab_you"):
            add_hazard(tdx, tdy, loom=0.8, threat=1.0, shot=0.0)

        # Grabbed: the fly can't move; drive a mild hazard so it "reacts" (shrug).
        if grabbed:
            add_hazard(0.0, 0.0, loom=0.5, threat=0.5, shot=0.0)

        return self.events_to_inject(events)


def _threat(observation: dict) -> bool:
    """Same threat test the encoder uses, so labels are consistent with the fly."""
    combat = observation.get("combat") or {}
    player = observation.get("player") or {}
    if player.get("grabbed"):
        return True
    if combat.get("incoming"):
        return True
    if combat.get("someone_will_hit_you") or combat.get("someone_will_grab_you"):
        return True
    target = combat.get("best_target") or observation.get("nearest") or {}
    if target.get("kind") == "grabber":
        return True
    dx = target.get("dx")
    return dx is not None and abs(dx) < 32


def coarse_actions() -> tuple[str, ...]:
    return ("proceed", "escape")


def coarse_of(fine_action: str, observation: dict) -> str:
    return "escape" if _threat(observation) else "proceed"


def _progress_dir(observation: dict):
    """Direction Thomas should walk toward the stairs (left or right)."""
    progress = observation.get("progress") or {}
    direction = progress.get("direction")
    if not direction:
        floor = int((observation.get("player") or {}).get("floor", 1) or 1)
        direction = "left" if floor % 2 else "right"
    return direction


def expand(coarse: str, observation: dict, fine_hint: str | None = None) -> str:
    player = observation.get("player") or {}
    grabbed = bool(player.get("grabbed"))
    if coarse == "escape":
        if grabbed:
            return "shrug"
        return "jump"
    # proceed: walk in the progress direction.
    direction = _progress_dir(observation)
    if fine_hint in ("left", "right", "crouch"):
        return fine_hint
    return direction


def hand_decode(command: dict, observation: dict) -> tuple[str, dict]:
    escape = max(command.get("escape_L", {}).get("rate", 0.0),
                 command.get("escape_R", {}).get("rate", 0.0))
    if escape > 1.5:
        return "escape", {"escape": 0.8, "proceed": 0.2}
    return "proceed", {"proceed": 0.8, "escape": 0.2}


__all__ = ["KungFuFlyEncoder", "coarse_actions", "coarse_of", "expand", "hand_decode"]
