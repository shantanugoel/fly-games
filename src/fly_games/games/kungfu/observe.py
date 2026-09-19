"""Structured Kung Fu observation: mission, combat, motion, and action history."""

from collections import Counter, deque

RAM_LIVES = 0x005C
RAM_STAGE = 0x0058
RAM_MODE = 0x0051
RAM_PAGE = 0x0065
RAM_HERO_ACTION = 0x0069
RAM_PLAYER_X = 0x00D4
RAM_PLAYER_Y = 0x00B6
RAM_HP = 0x04A6
RAM_SCORE = 0x0531
RAM_KILLS = 0x03B1
RAM_ENEMY_X = 0x00D1
RAM_ENEMY_Y = 0x00B3
RAM_ENEMY_TYPE = 0x008A
RAM_ENEMY_FACE = 0x00C3
RAM_ENEMY_POSE = 0x00E2
RAM_ENEMY_ATTACK = 0x00BA
RAM_ENEMY_ENERGY = 0x04A3
RAM_GRAB = 0x0374
RAM_HUGS = 0x0373
RAM_SHRUG = 0x0378
RAM_POSE = 0x036E
RAM_STANCE = 0x036F
RAM_AIR = 0x036A
RAM_BOSS_X = 0x00D3
RAM_BOSS_Y = 0x00B5
RAM_BOSS_HP = 0x04A5
RAM_BOSS_FACE = 0x00C5
RAM_BOSS_ATTACK = 0x00BC
RAM_BOSS_ACTIVE = 0x00E4
RAM_KNIFE_X = 0x03D4
RAM_KNIFE_Y = 0x03D0
RAM_KNIFE_STATE = 0x03EC
RAM_TIMER = 0x0390
ENEMY_INACTIVE = 0x7F
ENEMY_SLOTS = 4
KNIFE_SLOTS = 4

ENEMY_TYPES = {
    0: "Gripper",
    1: "Tiny Gripper",
    2: "Knife Thrower",
    3: "Stick Fighter",
    4: "Boomerang Fighter",
    5: "Bigman",
    6: "Magician",
    7: "Mr. X",
}
GRABBERS = {"Gripper", "Tiny Gripper"}
THROWERS = {"Knife Thrower", "Boomerang Fighter"}
BOSSES = {"Bigman", "Magician", "Mr. X"}
WALK_ACTIONS = {
    "left",
    "right",
    "punch_left",
    "punch_right",
    "kick_left",
    "kick_right",
}
ATTACK_ACTIONS = {
    "punch",
    "kick",
    "punch_left",
    "punch_right",
    "kick_left",
    "kick_right",
    "jump_kick",
    "crouch_punch",
    "crouch_kick",
}
HIT_RANGE = 22
GRAB_RANGE = 28
KNIFE_DANGER = 56
STALL_PX = 2

MISSION = {
    "you_are": (
        "You are Thomas — the hero of NES Kung Fu (Spartan X), the player, not a spectator. "
        "You are in a five-floor temple and you must stay alive."
    ),
    "level": "NES Kung Fu / Spartan X. Floor 1 starts on the right and fights left toward the stairs.",
    "win": (
        "Clear all five floors and rescue Sylvia at the top. Odd floors (1, 3, 5) progress LEFT; "
        "even floors (2, 4) progress RIGHT. Stairs at the far end go up. Killing one foe does not finish the floor."
    ),
    "survive": (
        "The HP bar is YOURS. 0 HP costs a life and restarts the floor. Grippers hug you and drain HP — "
        "shrug immediately (mash Left/Right and A/B). Walking or punching does not break a grab. "
        "Knife throwers and boomerangs hurt from range — jump or crouch, do not walk into the projectile. "
        "Stick fighters and bosses hit if they reach you."
    ),
    "navigation": (
        "Walk the progress direction toward the stairs. Attack grabbers before they hug you (kick around 22px). "
        "Repeating the same walk while x does not change means you are stuck on a foe or wall — kick or jump, "
        "do not mash the same direction. Wrong-way walking undoes progress."
    ),
}


class KungFuMemory:
    def __init__(self):
        self.previous_action = "wait"
        self.recent = deque(maxlen=12)
        self.trail = deque(maxlen=12)
        self.last_x = None
        self.last_world = None
        self.last_hp = None
        self.last_kills = 0
        self.last_score = 0
        self.last_floor = None
        self.had_progress = False
        self.blocked_streak = 0
        self.same_action_streak = 0
        self.walk_without_progress = 0
        self.attack_without_hit = 0
        self.grab_without_shrug = 0
        self.best_progress = None


def score_from_ram(ram) -> int | None:
    score = 0
    for offset in range(6):
        digit = int(ram[RAM_SCORE + offset]) & 0x0F
        if digit > 9:
            return None
        score = score * 10 + digit
    return score


def timer_from_ram(ram) -> int | None:
    value = 0
    for offset in range(4):
        digit = int(ram[RAM_TIMER + offset]) & 0x0F
        if digit > 9:
            return None
        value = value * 10 + digit
    return value


def in_gameplay(ram) -> bool:
    hp = int(ram[RAM_HP])
    lives = int(ram[RAM_LIVES])
    return lives > 0 and 8 <= hp <= 0x60 and score_from_ram(ram) is not None


def grabbed_from_ram(ram) -> bool:
    return int(ram[RAM_GRAB]) > 0


def progress_direction(floor: int) -> str:
    return "left" if int(floor) % 2 else "right"


def _world(page: int, x: int) -> int:
    return int(page) * 256 + int(x)


def _role(kind: str) -> str:
    if kind in GRABBERS:
        return "grabber"
    if kind in THROWERS:
        return "thrower"
    if kind in BOSSES:
        return "boss"
    return "melee"


def _decode_stance(raw: int) -> dict:
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    facing = "right" if low % 2 else "left"
    if low in {2, 3}:
        stance = "crouch"
    elif low in {4, 5}:
        stance = "jump"
    else:
        stance = "stand"
    attack = {1: "kick", 2: "punch"}.get(high, "none")
    return {"facing": facing, "stance": stance, "attack": attack, "raw": raw}


def enemies_from_ram(ram, player_x: int) -> list[dict]:
    enemies = []
    for slot in range(ENEMY_SLOTS):
        pose = int(ram[RAM_ENEMY_POSE - slot])
        if pose in {0, ENEMY_INACTIVE}:
            continue
        x = int(ram[RAM_ENEMY_X - slot])
        y = int(ram[RAM_ENEMY_Y - slot])
        kind_id = int(ram[RAM_ENEMY_TYPE - slot])
        facing = int(ram[RAM_ENEMY_FACE - slot])
        kind = ENEMY_TYPES.get(kind_id, f"type_{kind_id}")
        dx = x - player_x
        attack = int(ram[RAM_ENEMY_ATTACK - slot])
        energy = int(ram[RAM_ENEMY_ENERGY - slot])
        enemies.append(
            {
                "kind": kind,
                "role": _role(kind),
                "type_id": kind_id,
                "x": x,
                "y": y,
                "dx": dx,
                "facing": "left" if facing == 0 else "right",
                "pose": pose,
                "attacking": attack not in {0, ENEMY_INACTIVE},
                "energy": energy,
                "will_grab_you": kind in GRABBERS and abs(dx) < GRAB_RANGE,
                "will_hit_you": abs(dx) < HIT_RANGE,
                "urgency": (
                    "contact"
                    if abs(dx) < HIT_RANGE
                    else "close"
                    if abs(dx) < GRAB_RANGE
                    else "seen"
                ),
            }
        )
    enemies.sort(key=lambda item: abs(item["dx"]))
    return enemies


def knives_from_ram(ram, player_x: int) -> list[dict]:
    knives = []
    for slot in range(KNIFE_SLOTS):
        state = int(ram[RAM_KNIFE_STATE + slot])
        if state == 0:
            continue
        x = int(ram[RAM_KNIFE_X + slot])
        if x in {0, 0xF9}:
            continue
        y = int(ram[RAM_KNIFE_Y + slot])
        dx = x - player_x
        knives.append(
            {
                "kind": "Knife",
                "role": "projectile",
                "x": x,
                "y": y,
                "dx": dx,
                "facing": "left" if state == 0x01 else "right",
                "will_hit_you": abs(dx) < KNIFE_DANGER,
                "urgency": "contact" if abs(dx) < 24 else "close" if abs(dx) < KNIFE_DANGER else "seen",
            }
        )
    knives.sort(key=lambda item: abs(item["dx"]))
    return knives


def boss_from_ram(ram, player_x: int) -> dict | None:
    active = int(ram[RAM_BOSS_ACTIVE])
    if active in {0, ENEMY_INACTIVE}:
        return None
    x = int(ram[RAM_BOSS_X])
    dx = x - player_x
    return {
        "kind": "Boss",
        "role": "boss",
        "x": x,
        "y": int(ram[RAM_BOSS_Y]),
        "dx": dx,
        "hp": int(ram[RAM_BOSS_HP]),
        "facing": "left" if int(ram[RAM_BOSS_FACE]) == 0 else "right",
        "attacking": int(ram[RAM_BOSS_ATTACK]) not in {0, ENEMY_INACTIVE},
        "will_hit_you": abs(dx) < 36,
        "urgency": "contact" if abs(dx) < HIT_RANGE else "close" if abs(dx) < 48 else "seen",
    }


def info_from_ram(ram, info) -> dict:
    payload = dict(info or {})
    payload.update(
        {
            "x": int(ram[RAM_PLAYER_X]),
            "y": int(ram[RAM_PLAYER_Y]),
            "hp": int(ram[RAM_HP]),
            "lives": int(ram[RAM_LIVES]),
            "score": score_from_ram(ram) or 0,
            "stage": int(ram[RAM_STAGE]),
            "kills": int(ram[RAM_KILLS]),
            "grabbed": grabbed_from_ram(ram),
            "page": int(ram[RAM_PAGE]),
        }
    )
    return payload


def _history_features(memory: KungFuMemory) -> dict:
    actions = [row["action"] for row in memory.recent]
    moved = [float(row.get("moved") or 0) for row in memory.recent]
    last_moved = sum(moved[-4:]) if moved else 0.0
    looping = memory.same_action_streak >= 3 and last_moved < 4
    last = memory.previous_action
    wrong_way = any(row.get("wrong_way") for row in list(memory.recent)[-3:])
    return {
        "last_action": last,
        "last_actions": actions[-8:],
        "same_action_streak": memory.same_action_streak,
        "walk_without_progress": memory.walk_without_progress,
        "attack_without_hit": memory.attack_without_hit,
        "grab_without_shrug": memory.grab_without_shrug,
        "looping": looping,
        "wrong_way": wrong_way,
        "repeated_action": Counter(actions[-8:]).most_common(1)[0][0] if actions else None,
        "recent": [
            {
                "action": row["action"],
                "moved": row.get("moved"),
                "hp": row.get("hp"),
                "event": row.get("event"),
            }
            for row in list(memory.recent)[-8:]
        ],
        "note": (
            "These are YOUR last controller presses this attempt. "
            "If last_actions repeat left/left/left and x did not change, you are stuck — kick or jump, do not mash walk. "
            "If you are grabbed, shrug; walking will not break it."
            if looping or memory.grab_without_shrug
            else "If the same action repeats and moved stays ~0, change action."
        ),
    }


def _tactics(progress: dict, history: dict, combat: dict) -> dict:
    direction = progress.get("direction") or "left"
    other = "right" if direction == "left" else "left"
    forbidden = []
    if combat.get("must_shrug"):
        return {
            "mode": "shrug",
            "forbidden": ["left", "right", "wait", "crouch"],
            "progress_direction": direction,
            "hint": (
                "A Gripper is holding you. HP is draining. SHRUG now (mash Left/Right and A/B). "
                "Walking, punching, or jumping will not break the grab."
            ),
        }
    if combat.get("incoming_projectile"):
        return {
            "mode": "dodge",
            "forbidden": [other],
            "progress_direction": direction,
            "hint": "A knife or boomerang is incoming. JUMP or CROUCH. Do not walk into it.",
        }
    if combat.get("must_fight"):
        target = combat.get("best_target") or {}
        face = "left" if int(target.get("dx") or 0) < 0 else "right"
        return {
            "mode": "fight",
            "forbidden": [other] if abs(int(target.get("dx") or 0)) < HIT_RANGE else [],
            "progress_direction": direction,
            "hint": (
                f"A {target.get('kind', 'foe')} is in range (dx={target.get('dx')}). "
                f"Face {face} and KICK. Attack grabbers before they hug you. Do not walk past them."
            ),
        }
    mode = "advance"
    hint = (
        f"Walk {direction} toward the stairs. Kick grabbers around 22px. "
        "Clear the floor, then climb. Rescue Sylvia — do not wander the wrong way."
    )
    if history.get("wrong_way"):
        forbidden.append(other)
        hint = f"You walked the wrong way. This floor progresses {direction}. Do not walk {other}."
        mode = "unstick"
    if history.get("looping") and (history.get("last_action") or "") in WALK_ACTIONS:
        last = history.get("last_action")
        if last in {"left", "right"}:
            forbidden.append(last)
        mode = "unstick"
        hint = (
            f"You already walked {last} and x did not change. You are stuck on a foe or the wall. "
            "Kick the enemy in front or jump. Do not mash the same walk."
        )
    if progress.get("blocked") and mode != "fight":
        mode = "unstick"
        hint = (
            f"Stuck. Forbidden this step: {forbidden or ['(none)']}. "
            f"Kick if a foe is near, otherwise jump, then walk {direction}."
        )
    return {
        "mode": mode,
        "forbidden": forbidden,
        "progress_direction": direction,
        "hint": hint,
    }


def observe(ram, info, memory: KungFuMemory, hold_frames: int) -> dict:
    player_x = int(ram[RAM_PLAYER_X])
    player_y = int(ram[RAM_PLAYER_Y])
    hp_raw = int(ram[RAM_HP])
    hp = 0 if hp_raw == 0xFF else hp_raw
    lives = int(ram[RAM_LIVES])
    stage = int(ram[RAM_STAGE])
    floor = stage + 1
    kills = int(ram[RAM_KILLS])
    score = score_from_ram(ram) or 0
    page = int(ram[RAM_PAGE])
    world = _world(page, player_x)
    grabbed = grabbed_from_ram(ram)
    direction = progress_direction(floor)
    stance = _decode_stance(int(ram[RAM_STANCE]))
    airborne = int(ram[RAM_AIR]) != 0
    timer = timer_from_ram(ram)

    if memory.last_x is None:
        memory.last_x = player_x
        memory.last_world = world
        memory.last_hp = hp
        memory.last_kills = kills
        memory.last_score = score
        memory.last_floor = floor
        memory.best_progress = world
    if memory.last_floor != floor:
        memory.blocked_streak = 0
        memory.walk_without_progress = 0
        memory.best_progress = world
        memory.last_floor = floor

    dx = player_x - int(memory.last_x)
    world_delta = world - int(memory.last_world)
    along = -world_delta if direction == "left" else world_delta
    if direction == "left":
        memory.best_progress = min(memory.best_progress, world)
        off_best = world - int(memory.best_progress)
    else:
        memory.best_progress = max(memory.best_progress, world)
        off_best = int(memory.best_progress) - world
    walked = memory.previous_action in WALK_ACTIONS
    stalled = walked and abs(along) < STALL_PX and memory.had_progress
    if along >= STALL_PX:
        memory.had_progress = True
        memory.blocked_streak = 0
        memory.walk_without_progress = 0
    elif stalled or (walked and abs(along) < STALL_PX):
        memory.blocked_streak += 1
        memory.walk_without_progress += 1
    else:
        memory.blocked_streak = 0
    blocked = memory.blocked_streak >= 2 or (stalled and memory.walk_without_progress >= 2)

    foes = enemies_from_ram(ram, player_x)
    knives = knives_from_ram(ram, player_x)
    boss = boss_from_ram(ram, player_x)
    if boss:
        foes = [boss, *[foe for foe in foes if foe.get("role") != "boss"]]
        foes.sort(key=lambda item: abs(item["dx"]))
    nearest = foes[0] if foes else None
    grabbers = [foe for foe in foes if foe.get("will_grab_you")]
    hitters = [foe for foe in foes if foe.get("will_hit_you")]
    incoming = [knife for knife in knives if knife.get("will_hit_you")]
    ahead = [
        foe
        for foe in foes
        if (direction == "left" and foe["dx"] < 0) or (direction == "right" and foe["dx"] > 0)
    ]
    target = None
    if hitters:
        target = hitters[0]
    elif grabbers:
        target = grabbers[0]
    elif nearest:
        target = nearest

    health_lost = 0 if memory.last_hp is None else max(0, memory.last_hp - hp)
    kills_gained = max(0, kills - memory.last_kills)
    event = None
    if grabbed:
        event = "grabbed"
    elif incoming:
        event = "knife_incoming"
    elif health_lost:
        event = f"took_{health_lost}_damage"
    elif kills_gained:
        event = f"killed_{kills_gained}"
    elif stalled or blocked:
        event = "walked_into_obstacle"

    must_shrug = grabbed
    must_fight = (not grabbed) and bool(hitters or grabbers or (boss and boss.get("will_hit_you")))
    combat = {
        "must_shrug": must_shrug,
        "must_fight": must_fight,
        "incoming_projectile": bool(incoming),
        "someone_will_grab_you": bool(grabbers),
        "someone_will_hit_you": bool(hitters),
        "health_lost": health_lost,
        "best_target": target,
        "boss": boss,
        "hint": (
            "A Gripper is holding you. Shrug now or HP drains."
            if must_shrug
            else "A knife is coming. Jump or crouch."
            if incoming
            else "A foe is in hitting range. Kick. Do not walk into a grab."
            if must_fight
            else "No urgent foe. Walk the progress direction. Attack grabbers before they hug you."
        ),
    }
    progress = {
        "direction": direction,
        "displacement": abs(dx),
        "along_floor": along,
        "blocked": blocked,
        "making_progress": along >= STALL_PX,
        "stuck_streak": memory.blocked_streak,
        "world_x": world,
        "page": page,
        "toward_stairs": along > 0 or (not blocked and walked),
        "best_world": memory.best_progress,
        "off_best": off_best,
        "foes_ahead": len(ahead),
        "trail": list(memory.trail),
    }
    history = _history_features(memory)
    tactics = _tactics(progress, history, combat)
    last_moves = history.get("last_actions") or []
    forbidden = tactics.get("forbidden") or []
    briefing = (
        f"ROLE: You are Thomas — the hero playing NES Kung Fu, not a spectator. "
        f"JOB: Clear five temple floors and rescue Sylvia. Floor {floor} progresses {direction.upper()}. "
        f"STATUS: x={player_x} y={player_y} hp={hp} lives={lives} facing {stance['facing']}. "
        f"MODE: {tactics['mode']}. {tactics['hint']} "
        f"YOUR LAST MOVES: {last_moves or ['none']}. "
        f"DO NOT CHOOSE: {forbidden or ['(nothing forbidden)']}. "
        f"EVENT: {event or 'none'}."
    )
    observation = {
        "briefing": briefing,
        "you_are": MISSION["you_are"],
        "job": MISSION["win"],
        "mission": MISSION,
        "coordinates": (
            "NES pixels, 256x240. x is on-screen fighter position; page is the scrolled screen. "
            "world_x = page*256+x. Floors 1/3/5 run LEFT (world_x down) toward stairs; "
            "floors 2/4 run RIGHT. Enemy dx < 0 is left of you, dx > 0 is right."
        ),
        "player": {
            "name": "Thomas (you)",
            "x": player_x,
            "y": player_y,
            "hp": hp,
            "lives": lives,
            "score": score,
            "kills": kills,
            "floor": floor,
            "stage": stage,
            "page": page,
            "world_x": world,
            "mode": int(ram[RAM_MODE]),
            "facing": stance["facing"],
            "stance": stance["stance"],
            "attack": stance["attack"],
            "airborne": airborne,
            "hero_action": int(ram[RAM_HERO_ACTION]),
            "pose": int(ram[RAM_POSE]),
            "grabbed": grabbed,
            "grab_counter": int(ram[RAM_GRAB]),
            "hug_count": int(ram[RAM_HUGS]),
            "shrug_counter": int(ram[RAM_SHRUG]),
            "time": timer,
        },
        "progress": progress,
        "combat": combat,
        "history": history,
        "tactics": tactics,
        "nearby_objects": (foes[:6] + knives[:4])[:8],
        "projectiles": knives[:4],
        "nearest": nearest,
        "best_target": target,
        "previous_action": memory.previous_action,
        "action_frames": hold_frames,
        "recent_decisions": history["recent"],
        "last_event": event,
        "physics": {
            "control": (
                "Left/right walk. Up jumps. Down crouches. B punches. A kicks. Jump+A is a jump kick. "
                "When player.grabbed is true a Gripper is holding you — choose shrug (mash Left/Right and A/B). "
                "history.last_actions is what you already did. tactics.forbidden must not be chosen."
            ),
            "goal": MISSION["win"],
        },
    }
    memory.last_x = player_x
    memory.last_world = world
    memory.last_hp = hp
    memory.last_kills = kills
    memory.last_score = score
    memory.trail.append(
        {
            "x": player_x,
            "world": world,
            "facing": stance["facing"],
            "moved": along,
            "action": memory.previous_action,
            "event": event,
        }
    )
    return observation


def remember(memory: KungFuMemory, before: dict, info: dict, action: str, result) -> None:
    memory.previous_action = action
    if memory.recent and memory.recent[-1]["action"] == action:
        memory.same_action_streak += 1
    else:
        memory.same_action_streak = 1
    direction = (before.get("progress") or {}).get("direction") or progress_direction(
        int((before.get("player") or {}).get("floor") or 1)
    )
    old_world = memory.last_world
    new_world = _world(int(info.get("page") or 0), int(info.get("x") or 0))
    if old_world is None:
        along = 0
    else:
        delta = new_world - int(old_world)
        along = -delta if direction == "left" else delta
    grabbed = bool(info.get("grabbed"))
    if grabbed and action != "shrug":
        memory.grab_without_shrug += 1
    elif action == "shrug" or not grabbed:
        memory.grab_without_shrug = 0
    if action in ATTACK_ACTIONS:
        kills = int(info.get("kills") or 0)
        score = int(info.get("score") or 0)
        if kills <= memory.last_kills and score <= memory.last_score:
            memory.attack_without_hit += 1
        else:
            memory.attack_without_hit = 0
    wrong_way = action in {"left", "right"} and action != direction
    event = None
    if grabbed and action != "shrug":
        event = "ignored_grab"
    elif wrong_way:
        event = "wrong_way"
    elif action in WALK_ACTIONS and along < STALL_PX:
        event = "no_progress"
    memory.recent.append(
        {
            "action": action,
            "reward": getattr(result, "reward", 0),
            "score": info.get("score"),
            "hp": info.get("hp"),
            "lives": info.get("lives"),
            "x": info.get("x"),
            "moved": along,
            "wrong_way": wrong_way,
            "grabbed": grabbed,
            "event": event,
        }
    )
