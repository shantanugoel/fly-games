"""NES Kung Fu / Spartan X via nes-py and the kungfu2 RAM map."""

from fly_games.games.base import Game
from fly_games.games.kungfu.fly import (
    KungFuFlyEncoder,
    coarse_actions,
    coarse_of,
    expand,
    hand_decode,
)
from fly_games.games.kungfu.observe import (
    HIT_RANGE,
    RAM_GRAB,
    RAM_LIVES,
    KungFuMemory,
    in_gameplay,
    info_from_ram,
    observe,
    progress_direction,
)
from fly_games.platforms.nes import make_nes_rom_env, nes_press
from fly_games.types import Action, AdvanceResult, GameMeta, Stat

ACTIONS = {
    "wait": [],
    "left": ["left"],
    "right": ["right"],
    "jump": ["up"],
    "crouch": ["down"],
    "punch": ["B"],
    "kick": ["A"],
    "punch_left": ["left", "B"],
    "punch_right": ["right", "B"],
    "kick_left": ["left", "A"],
    "kick_right": ["right", "A"],
    "jump_kick": ["up", "A"],
    "crouch_punch": ["down", "B"],
    "crouch_kick": ["down", "A"],
    "shrug": [],
}

# Alternate wiggle + attacks each frame (GameFAQs: mash Left/Right and A/B).
SHRUG_MASH = (("left", "A"), ("right", "B"), ("left", "B"), ("right", "A"))

_LABELS = {key: key.replace("_", " ") for key in ACTIONS}


class KungFuGame(Game):
    meta = GameMeta(
        id="kungfu",
        title="Kung Fu",
        short_name="KUNG-FU",
        platform_id="nes",
        platform_label="NES / nes-py",
        cartridge="Kung Fu / Spartan X (NES)",
        subtitle="You are Thomas - clear 5 floors, rescue Sylvia",
        aspect_width=256,
        aspect_height=240,
        button_keys=("left", "right", "up", "down", "A", "B"),
        default_hold_frames=2,
        brain_steps=24,
        brain_warmup=80,
    )
    actions = tuple(
        Action(key=key, label=_LABELS[key],
               buttons=("left", "right", "A", "B") if key == "shrug" else tuple(buttons))
        for key, buttons in ACTIONS.items()
    )

    def __init__(self):
        self._cartridge_label = None

    def create_env(self):
        from fly_games.assets import find_kungfu_rom
        rom = find_kungfu_rom()
        self._cartridge_label = f"{rom.name} - NES"
        joypad = [tuple(buttons) for key, buttons in ACTIONS.items() if key != "shrug"]
        return make_nes_rom_env(rom, joypad)

    def reset(self, env, seed: int):
        frame, info = env.reset(seed=seed)
        frame, info, _, _ = nes_press(env, frames=90)
        ram = env.unwrapped.ram
        if not in_gameplay(ram):
            frame, info, _, _ = nes_press(env, "start", frames=2)
            for _ in range(80):
                frame, info, _, _ = nes_press(env, frames=2)
                ram = env.unwrapped.ram
                if in_gameplay(ram):
                    break
        frame, info, _, _ = nes_press(env, frames=300)
        return frame, info_from_ram(env.unwrapped.ram, info)

    def new_memory(self):
        return KungFuMemory()

    def observe(self, env, info, memory, hold_frames: int, previous_action: str):
        memory.previous_action = previous_action
        return observe(env.unwrapped.ram, info, memory, hold_frames)

    def scene(self, observation: dict) -> dict:
        player = observation["player"]
        foes = observation.get("nearby_objects") or []
        tactics = observation.get("tactics") or {}
        facts = [
            {"label": "job",
             "value": f"You are Thomas. Floor {player['floor']} go {progress_direction(player['floor'])}. Rescue Sylvia."},
            {"label": "thomas",
             "value": f"x={player['x']} y={player['y']} hp={player['hp']} face={player.get('facing')}"
                      + (" GRABBED" if player.get("grabbed") else "")},
            {"label": "tactics",
             "value": f"{tactics.get('mode')} - {tactics.get('hint', '')[:90]}"},
            {"label": "ahead",
             "value": ", ".join(f"{o['kind']} dx={o['dx']}" for o in foes[:4]) or "none"},
        ]
        overlays = [{"kind": "Thomas", "x": max(0, player["x"] - 8), "y": max(0, player["y"] - 24), "w": 16, "h": 32}]
        for foe in foes[:4]:
            overlays.append({"kind": foe["kind"], "x": max(0, foe["x"] - 8),
                             "y": max(0, int(foe.get("y") or player["y"])) - 24, "w": 16, "h": 32})
        return {"facts": facts, "overlays": overlays,
                "text": "\n".join([
                    observation.get("briefing", ""),
                    facts[2]["value"],
                    f"ahead: {facts[3]['value']}",
                ])}

    def stats(self, observation: dict, info: dict, session: dict) -> list[Stat]:
        player = observation["player"]
        tactics = observation.get("tactics") or {}
        return [
            Stat("SCORE", str(player["score"])),
            Stat("HP", str(player["hp"])),
            Stat("LIVES", str(player["lives"])),
            Stat("FLOOR", str(player["floor"])),
            Stat("MODE", str(tactics.get("mode") or "advance")),
        ]

    def advance(self, env, action_key: str, frames: int, observation: dict) -> AdvanceResult:
        reward = 0.0
        terminated = truncated = False
        info = {}
        frame = None
        executed = 0
        start_score = int((observation.get("player") or {}).get("score", 0))
        start_lives = int((observation.get("player") or {}).get("lives", 0))
        mash = action_key == "shrug"
        index = None if mash else next(i for i, a in enumerate(self.actions) if a.key == action_key)
        for step in range(frames):
            if mash:
                frame, info, terminated, truncated = nes_press(env, *SHRUG_MASH[step % len(SHRUG_MASH)], frames=1)
            else:
                frame, value, terminated, truncated, info = env.step(index)
                reward += float(value)
            executed += 1
            ram = env.unwrapped.ram
            if int(ram[RAM_LIVES]) == 0 and start_lives > 0:
                terminated = True
                break
            if mash and int(ram[RAM_GRAB]) == 0 and step >= 3:
                break
            if terminated or truncated:
                break
        ram = env.unwrapped.ram
        packed = info_from_ram(ram, info)
        packed["x_pos"] = packed["x"]
        score = packed["score"]
        reward += max(0, score - start_score) / 100.0
        return AdvanceResult(
            frame=frame, info=packed, reward=reward,
            terminated=bool(terminated), truncated=bool(truncated),
            frames_executed=executed,
        )

    def scripted(self, observation: dict) -> tuple[str, dict]:
        nearest = observation.get("best_target") or observation.get("nearest")
        player = observation.get("player") or {}
        combat = observation.get("combat") or {}
        progress = observation.get("progress") or {}
        tactics = observation.get("tactics") or {}
        floor = int(player.get("floor", 1) or 1)
        grabbed = bool(player.get("grabbed") or combat.get("must_shrug"))
        desired = tactics.get("progress_direction") or progress.get("direction") or progress_direction(floor)
        dx = None if nearest is None else int(nearest.get("dx", 0))
        jump = False
        attack = False
        kick = False
        shrug = False
        movement = desired
        if grabbed:
            shrug = True
            movement = "wait"
        elif tactics.get("mode") == "dodge" or combat.get("incoming_projectile"):
            jump = True
            movement = "wait"
        elif nearest and abs(dx) < HIT_RANGE:
            movement = "right" if dx >= 0 else "left"
            attack = True
            kick = True
        elif tactics.get("mode") == "unstick" or progress.get("blocked"):
            if nearest and abs(dx) < 48:
                movement = "right" if dx >= 0 else "left"
                attack = True
                kick = True
            else:
                jump = True
                movement = desired
        elif nearest and (
            (desired == "right" and 0 < dx < 40) or (desired == "left" and -40 < dx < 0)
        ):
            movement = desired
        else:
            movement = desired
        movement, jump, attack, kick, shrug = _apply_tactics(observation, movement, jump, attack, kick, shrug)
        action = _compose_kungfu(movement, jump, attack, kick, shrug=shrug, grabbed=grabbed)
        probs = {"right": 0.7, "left": 0.1, "crouch": 0.05, "wait": 0.15}
        probs[movement] = 0.85
        return action, {
            "confidence": 1.0, "probabilities": probs,
            "decisions": {"movement": movement, "attack": 1.0 if attack else 0.0,
                          "source": "scripted"},
        }

    def completed(self, info: dict) -> bool:
        return int(info.get("stage", 0)) >= 5 and int(info.get("lives", 0)) > 0 and int(info.get("hp", 0)) > 0

    # --- fly contract ---
    def fly_encoder(self, client):
        return KungFuFlyEncoder(client)

    def fly_coarse_actions(self) -> tuple[str, ...]:
        return coarse_actions()

    def fly_coarse_of(self, fine_action: str, observation: dict) -> str:
        return coarse_of(fine_action, observation)

    def fly_expand(self, coarse: str, observation: dict, fine_hint: str | None = None) -> str:
        return expand(coarse, observation, fine_hint)

    def fly_hand_decode(self, command: dict, observation: dict) -> tuple[str, dict]:
        return hand_decode(command, observation)


def _apply_tactics(observation: dict, movement: str, jump: bool, attack: bool, kick: bool, shrug: bool) -> tuple:
    player = observation.get("player") or {}
    tactics = observation.get("tactics") or {}
    progress = observation.get("progress") or {}
    combat = observation.get("combat") or {}
    history = observation.get("history") or {}
    grabbed = bool(player.get("grabbed") or combat.get("must_shrug"))
    forbidden = set(tactics.get("forbidden") or [])
    direction = (tactics.get("progress_direction") or progress.get("direction")
                 or progress_direction(int(player.get("floor") or 1)))
    mode = tactics.get("mode")
    target = observation.get("best_target") or combat.get("best_target") or observation.get("nearest")
    if grabbed:
        return "wait", False, False, False, True
    if "left" in forbidden and movement == "left":
        movement = direction if direction != "left" else "wait"
    if "right" in forbidden and movement == "right":
        movement = direction if direction != "right" else "wait"
    if history.get("wrong_way") and movement != direction and movement in {"left", "right"}:
        movement = direction
    if mode == "dodge" or combat.get("incoming_projectile"):
        attack = False
        if not jump and movement != "crouch":
            jump = True
            movement = "wait"
    elif mode == "fight" and target:
        dx = int(target.get("dx", 0) or 0)
        if abs(dx) < 28:
            attack = True
            kick = True
            movement = "right" if dx >= 0 else "left"
    elif mode == "unstick":
        if target and abs(int(target.get("dx", 0) or 0)) < 48:
            dx = int(target["dx"])
            attack = True
            kick = True
            jump = False
            movement = "right" if dx >= 0 else "left"
        elif movement in forbidden or history.get("looping"):
            jump = True
            if movement in forbidden:
                movement = direction
    elif mode in {"advance", None} and movement in {"left", "right", "wait"} and not attack:
        if movement != direction:
            movement = direction
    if combat.get("someone_will_grab_you") and target and abs(int(target.get("dx", 0) or 0)) < 28:
        attack = True
        kick = True
        dx = int(target["dx"])
        movement = "right" if dx >= 0 else "left"
        jump = False
        shrug = False
    return movement, jump, attack, kick, shrug


def _compose_kungfu(movement, jump, attack, kick, shrug=False, grabbed=False):
    if grabbed or shrug:
        return "shrug"
    if jump and attack:
        return "jump_kick"
    if jump:
        return "jump"
    if attack and kick:
        if movement == "left":
            return "kick_left"
        if movement == "crouch":
            return "crouch_kick"
        if movement == "wait":
            return "kick"
        return "kick_right"
    if attack:
        if movement == "left":
            return "punch_left"
        if movement == "crouch":
            return "crouch_punch"
        if movement == "wait":
            return "punch"
        return "punch_right"
    if movement == "crouch":
        return "crouch"
    if movement == "left":
        return "left"
    if movement == "wait":
        return "wait"
    return "right"
