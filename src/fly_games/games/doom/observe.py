"""Structured Doom observation: mission, motion, combat, and action history."""

from collections import Counter, deque

SCREEN_CX = 320
WALK_ACTIONS = {
    "forward",
    "back",
    "forward_left",
    "forward_right",
    "strafe_left",
    "strafe_right",
    "forward_shoot",
    "forward_use",
}
USE_ACTIONS = {"use", "forward_use"}
WALL_DEPTH = 12.0
CRUISE_DISPLACEMENT = 12.0
STALL_DISPLACEMENT = 6.0
COMPASS = ((0.0, "east"), (90.0, "north"), (180.0, "west"), (270.0, "south"))
WEAPONS = {
    1: "fist",
    2: "pistol",
    3: "shotgun",
    4: "chaingun",
    5: "rocket launcher",
    6: "plasma rifle",
    7: "bfg",
}
MELEE_MONSTERS = {"Demon", "Spectre", "LostSoul"}
HITSCAN_MONSTERS = {"Zombieman", "ShotgunGuy", "ChaingunGuy", "WolfensteinSS"}
PROJECTILE_MONSTERS = {"DoomImp", "Cacodemon", "BaronOfHell", "HellKnight", "Revenant", "Fatso", "Mancubus"}

MISSION = {
    "you_are": (
        "You are the Doom marine — the player, not a spectator. You are inside E1M1 Hangar "
        "with a pistol, and you must stay alive."
    ),
    "level": "Doom (1993) episode 1 map 1, Hangar. First room faces north into a nukage courtyard.",
    "win": (
        "Clear the Hangar: explore rooms, open doors, kill the demons that get in the way, "
        "and reach the exit. Killing one monster does not finish the map."
    ),
    "survive": (
        "Health 0 ends the attempt. Zombiemen and ShotgunGuys shoot. Imps throw fireballs. "
        "Pinky Demons bite if they reach you. Do not stand still in a demon's face — strafe, back up, shoot back."
    ),
    "navigation": (
        "USE opens a door or switch once. A plain wall does not open. If USE did nothing, turn toward open space. "
        "Repeating the same action while x/y stay still means you are stuck — change action."
    ),
}


class DoomMemory:
    def __init__(self):
        self.previous_action = "wait"
        self.recent = deque(maxlen=12)
        self.trail = deque(maxlen=12)
        self.spawn = None
        self.last_xy = None
        self.last_angle = None
        self.last_health = None
        self.last_kills = 0
        self.last_hits = 0
        self.had_progress = False
        self.blocked_streak = 0
        self.use_streak = 0
        self.same_action_streak = 0
        self.turn_bias = "left"
        self.best_distance_from_spawn = 0.0
        self.visited = set()
        self.uses_without_move = 0


def facing_name(angle: float) -> str:
    wrapped = float(angle) % 360.0
    return min(
        COMPASS,
        key=lambda item: min(abs(wrapped - item[0]), 360.0 - abs(wrapped - item[0])),
    )[1]


def _depth_view(session) -> dict:
    getter = getattr(session, "depth_view", None)
    if getter is None:
        return {"ahead": 255.0, "left": 255.0, "right": 255.0}
    return getter()


def _threat_kind(name: str) -> str:
    if name in MELEE_MONSTERS:
        return "melee"
    if name in HITSCAN_MONSTERS:
        return "hitscan"
    if name in PROJECTILE_MONSTERS:
        return "projectile"
    if name == "ExplosiveBarrel":
        return "barrel"
    return "other"


def _urgency(distance: float, on_screen: bool, kind: str) -> str:
    if distance < 72:
        return "contact"
    if distance < 160:
        return "close"
    if on_screen:
        return "seen"
    if kind == "melee" and distance < 280:
        return "closing"
    return "far"


def _encode_objects(session) -> list[dict]:
    encoded = []
    for obj in session.objects():
        item = {
            "kind": obj.name,
            "role": obj.role,
            "category": obj.category,
            "distance": obj.distance,
            "angle_deg": obj.angle_deg,
            "on_screen": obj.on_screen,
            "screen_x": obj.screen_x,
            "dx": obj.dx,
            "dy": obj.dy,
        }
        if obj.role == "monster":
            threat = _threat_kind(obj.name)
            item["attack"] = threat
            item["urgency"] = _urgency(obj.distance, obj.on_screen, threat)
            item["will_eat_you"] = threat == "melee" and obj.distance < 140
            item["will_shoot_you"] = threat in {"hitscan", "projectile"} and obj.on_screen
        encoded.append(item)
    return encoded


def _history_features(memory: DoomMemory) -> dict:
    actions = [row["action"] for row in memory.recent]
    moved = [float(row.get("moved") or 0) for row in memory.recent]
    last_moved = sum(moved[-4:]) if moved else 0.0
    same = memory.same_action_streak
    looping = same >= 3 and last_moved < 10
    use_loop = memory.use_streak >= 1 and memory.uses_without_move >= 1
    counts = Counter(actions[-8:])
    return {
        "last_action": memory.previous_action,
        "last_actions": actions[-8:],
        "same_action_streak": same,
        "use_streak": memory.use_streak,
        "uses_without_moving": memory.uses_without_move,
        "looping": looping,
        "use_already_failed": use_loop,
        "repeated_action": counts.most_common(1)[0][0] if counts else None,
        "recent": [
            {
                "action": row["action"],
                "moved": row.get("moved"),
                "facing": row.get("facing"),
                "health": row.get("health"),
                "event": row.get("event"),
            }
            for row in list(memory.recent)[-8:]
        ],
        "note": (
            "These are YOUR last controller presses this attempt. "
            "If last_actions are use/use/use and x/y did not change, USE is not a door — turn away."
            if use_loop or (looping and memory.previous_action in USE_ACTIONS)
            else "If the same action repeats and moved stays ~0, change action."
        ),
    }


def _tactics(progress: dict, history: dict, combat: dict) -> dict:
    forbidden = []
    prefer_turn = progress.get("open_side")
    if combat.get("must_fight"):
        return {
            "mode": "fight",
            "forbidden": ["use"],
            "prefer_turn": prefer_turn,
            "hint": combat["hint"],
        }
    mode = "explore"
    hint = "Walk into open space. Open doors with a single USE. Kill demons that block the path. Find the Hangar exit."
    if history.get("use_already_failed") or (history.get("use_streak", 0) >= 1 and progress.get("wall_ahead")):
        forbidden.append("use")
        mode = "unstick"
        hint = (
            f"You already pressed USE and the wall did not open. Do not USE again. "
            f"Turn {prefer_turn} toward open depth and walk."
        )
    if history.get("looping") and "forward" in (history.get("last_action") or ""):
        forbidden.append("forward")
        mode = "unstick"
        hint = f"Walking is not moving you. Turn {prefer_turn} and stop walking into the wall."
    if history.get("looping") and (history.get("last_action") or "").startswith("turn"):
        prefer_turn = "right" if prefer_turn == "left" else "left"
        hint = f"Turning one way is not helping. Turn {prefer_turn} or walk if the path is clear."
        mode = "unstick"
    if progress.get("blocked"):
        mode = "unstick"
        if "use" not in forbidden and progress.get("try_use"):
            hint = "Possible door. Press USE once. If nothing happens, turn away — do not mash USE."
        elif "use" in forbidden:
            hint = (
                f"Stuck. Forbidden this step: {forbidden}. Turn {prefer_turn} "
                "toward the larger open_left/open_right depth."
            )
    return {
        "mode": mode,
        "forbidden": forbidden,
        "prefer_turn": prefer_turn,
        "hint": hint,
    }


def observe(session, info, memory: DoomMemory, hold_frames: int) -> dict:
    angle = float(info.get("angle") or 0)
    facing = facing_name(angle)
    x = float(info["x"])
    y = float(info["y"])
    health = int(info["health"])
    kills = int(info.get("kills") or 0)
    hits = int(info.get("hits") or 0)
    if memory.spawn is None:
        memory.spawn = (x, y)
        memory.last_xy = (x, y)
        memory.last_angle = angle
        memory.last_health = health
        memory.last_kills = kills
        memory.last_hits = hits
    dx = x - memory.last_xy[0]
    dy = y - memory.last_xy[1]
    displacement = (dx * dx + dy * dy) ** 0.5
    spawn_dx = x - memory.spawn[0]
    spawn_dy = y - memory.spawn[1]
    from_spawn = (spawn_dx * spawn_dx + spawn_dy * spawn_dy) ** 0.5
    memory.best_distance_from_spawn = max(memory.best_distance_from_spawn, from_spawn)
    cell = (int(x // 64), int(y // 64))
    revisited = cell in memory.visited
    memory.visited.add(cell)

    depth = _depth_view(session)
    ahead = float(depth.get("ahead", 255))
    left = float(depth.get("left", 255))
    right = float(depth.get("right", 255))
    wall_ahead = ahead <= WALL_DEPTH
    walked = memory.previous_action in WALK_ACTIONS
    stall_limit = max(STALL_DISPLACEMENT, 0.7 * hold_frames)
    stalled = walked and displacement < stall_limit and memory.had_progress
    if displacement >= CRUISE_DISPLACEMENT and not wall_ahead:
        memory.had_progress = True
        memory.blocked_streak = 0
        memory.uses_without_move = 0
    elif wall_ahead or stalled:
        memory.blocked_streak += 1
    else:
        memory.blocked_streak = 0
    if abs(left - right) >= 8:
        memory.turn_bias = "left" if left > right else "right"
    blocked = wall_ahead or memory.blocked_streak >= 2
    stopped = displacement < stall_limit
    try_use = (
        wall_ahead
        and stopped
        and memory.use_streak == 0
        and memory.uses_without_move == 0
        and memory.previous_action not in USE_ACTIONS
    )

    health_lost = 0 if memory.last_health is None else max(0, memory.last_health - health)
    kills_gained = max(0, kills - memory.last_kills)
    under_fire = health_lost > 0
    event = None
    if health_lost:
        event = f"took_{health_lost}_damage"
    elif kills_gained:
        event = f"killed_{kills_gained}"
    elif memory.previous_action in USE_ACTIONS and wall_ahead and stopped:
        event = "use_did_nothing"
    elif stalled or (walked and wall_ahead):
        event = "walked_into_wall"

    raw = _encode_objects(session)
    interesting = [obj for obj in raw if obj["role"] in {"monster", "hazard", "item"}]
    interesting.sort(
        key=lambda obj: (
            0 if obj["role"] == "monster" else 1 if obj["role"] == "hazard" else 2,
            obj["distance"],
        )
    )
    monsters = [obj for obj in interesting if obj["role"] == "monster"]
    visible_monsters = [obj for obj in monsters if obj["on_screen"]]
    eaters = [obj for obj in monsters if obj.get("will_eat_you")]
    shooters = [obj for obj in visible_monsters if obj.get("will_shoot_you")]
    nearest = monsters[0] if monsters else (interesting[0] if interesting else None)
    centered = None
    if visible_monsters:
        centered = min(visible_monsters, key=lambda obj: abs((obj["screen_x"] or SCREEN_CX) - SCREEN_CX))
    pickups = [obj for obj in interesting if obj["role"] == "item"][:4]

    must_fight = bool(visible_monsters) and (
        any(obj.get("urgency") in {"contact", "close"} for obj in visible_monsters) or under_fire
    )
    combat = {
        "live_on_screen": len(visible_monsters),
        "live_nearby": len(monsters),
        "must_fight": must_fight,
        "under_fire": under_fire,
        "health_lost": health_lost,
        "someone_will_eat_you": bool(eaters),
        "someone_will_shoot_you": bool(shooters),
        "best_target": centered,
        "hint": (
            "A demon is in your face or shooting you. Aim the crosshair (x=320) at best_target and FIRE. "
            "Strafe or back up if it is melee (Pinky/Demon). Do not USE while fighting."
            if must_fight
            else "No urgent demon in view. Explore. Do not hunt decorative corpses."
        ),
    }

    progress = {
        "displacement": round(displacement, 1),
        "blocked": blocked,
        "wall_ahead": wall_ahead,
        "ahead_depth": round(ahead, 1),
        "open_left_depth": round(left, 1),
        "open_right_depth": round(right, 1),
        "open_side": memory.turn_bias,
        "path_clear": ahead > 40 and not wall_ahead,
        "stuck_streak": memory.blocked_streak,
        "distance_from_spawn": round(from_spawn, 1),
        "best_distance_from_spawn": round(memory.best_distance_from_spawn, 1),
        "making_progress": displacement >= stall_limit and not wall_ahead,
        "revisiting_cell": revisited and not (displacement >= stall_limit),
        "try_use": try_use,
        "trail": list(memory.trail),
    }
    history = _history_features(memory)
    tactics = _tactics(progress, history, combat)
    weapon = WEAPONS.get(int(info.get("weapon") or 2), "pistol")
    ammo = int(info["ammo"])
    last_moves = history.get("last_actions") or []
    forbidden = tactics.get("forbidden") or []
    briefing = (
        f"ROLE: You are the Doom marine — the hero playing this game, not a spectator. "
        f"JOB: Clear E1M1 Hangar: kill demons, open real doors, reach the exit, stay alive. "
        f"STATUS: facing {facing} at ({round(x)}, {round(y)}), hp {health}, {weapon} ammo {ammo}, "
        f"kills {kills}. MODE: {tactics['mode']}. {tactics['hint']} "
        f"YOUR LAST MOVES: {last_moves or ['none']}. "
        f"DO NOT CHOOSE: {forbidden or ['(nothing forbidden)']}. "
        f"EVENT: {event or 'none'}."
    )

    iwad_kind = info.get("iwad_kind") or "unknown"
    map_name = str(info.get("map") or "e1m1").upper()
    observation = {
        "briefing": briefing,
        "you_are": MISSION["you_are"],
        "job": MISSION["win"],
        "mission": MISSION,
        "coordinates": (
            "Map units: +x east, +y north. Facing angle 0=east, 90=north, 180=west, 270=south. "
            "Object angle_deg is relative: 0 ahead, negative left, positive right. "
            "Screen 640x480, crosshair x=320. Depth 0=touching a wall, ~255=far."
        ),
        "player": {
            "name": "Doom marine (you)",
            "health": health,
            "armor": int(info["armor"]),
            "ammo": ammo,
            "bullets": int(info.get("bullets", info["ammo"])),
            "shells": int(info.get("shells", 0)),
            "weapon": weapon,
            "weapon_id": int(info.get("weapon") or 2),
            "x": round(x, 1),
            "y": round(y, 1),
            "angle": round(angle, 1),
            "facing": facing,
            "kills": kills,
            "dead": bool(info.get("dead")),
            "map": info.get("map"),
            "iwad": info.get("iwad"),
            "iwad_kind": iwad_kind,
            "level_name": f"{map_name} Hangar" if iwad_kind in {"original", "shareware"} else map_name,
        },
        "progress": progress,
        "combat": combat,
        "history": history,
        "tactics": tactics,
        "nearby_objects": interesting[:8],
        "pickups": pickups,
        "nearest": nearest,
        "best_target": centered,
        "live_monsters_on_screen": len(visible_monsters),
        "previous_action": memory.previous_action,
        "action_frames": hold_frames,
        "recent_decisions": history["recent"],
        "last_event": event,
        "physics": {
            "control": (
                "Forward walks the way you face. Turn looks. Strafe steps sideways while facing the same way. "
                "ATTACK fires the current weapon. USE taps a door/switch — one tap; mashing a wall does nothing. "
                "history.last_actions is what you already did. tactics.forbidden must not be chosen."
            ),
            "goal": MISSION["win"],
        },
    }
    memory.last_xy = (x, y)
    memory.last_angle = angle
    memory.last_health = health
    memory.last_kills = kills
    memory.last_hits = hits
    memory.trail.append(
        {
            "x": round(x, 1),
            "y": round(y, 1),
            "facing": facing,
            "moved": round(displacement, 1),
            "action": memory.previous_action,
            "event": event,
        }
    )
    return observation


def remember(memory: DoomMemory, before: dict, info: dict, action: str, result) -> None:
    memory.previous_action = action
    if memory.recent and memory.recent[-1]["action"] == action:
        memory.same_action_streak += 1
    else:
        memory.same_action_streak = 1
    px, py = memory.last_xy or (float(info.get("x") or 0), float(info.get("y") or 0))
    moved = ((float(info.get("x") or 0) - px) ** 2 + (float(info.get("y") or 0) - py) ** 2) ** 0.5
    if action in USE_ACTIONS:
        memory.use_streak += 1
        if moved < CRUISE_DISPLACEMENT:
            memory.uses_without_move += 1
    else:
        memory.use_streak = 0
    progress = before.get("progress") or {}
    event = "use_on_wall" if action in USE_ACTIONS and progress.get("wall_ahead") else None
    memory.recent.append(
        {
            "action": action,
            "reward": getattr(result, "reward", 0),
            "kills": info.get("kills"),
            "health": info.get("health"),
            "x": info.get("x"),
            "y": info.get("y"),
            "facing": facing_name(float(info.get("angle") or 0)),
            "moved": round(moved, 1),
            "blocked": progress.get("blocked"),
            "event": event,
        }
    )
