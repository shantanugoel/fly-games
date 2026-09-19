"""Doom E1M1 via ViZDoom (original IWAD when present)."""

from fly_games.games.base import Game
from fly_games.games.doom.fly import coarse_actions, coarse_of, expand, hand_decode
from fly_games.games.doom.observe import DoomMemory, observe, remember
from fly_games.platforms.doom import make_doom
from fly_games.types import Action, AdvanceResult, GameMeta, Stat

ACTIONS = {
    "wait": (),
    "forward": ("MOVE_FORWARD",),
    "back": ("MOVE_BACKWARD",),
    "turn_left": ("TURN_LEFT",),
    "turn_right": ("TURN_RIGHT",),
    "forward_left": ("MOVE_FORWARD", "TURN_LEFT"),
    "forward_right": ("MOVE_FORWARD", "TURN_RIGHT"),
    "strafe_left": ("MOVE_LEFT",),
    "strafe_right": ("MOVE_RIGHT",),
    "shoot": ("ATTACK",),
    "use": ("USE",),
    "forward_shoot": ("MOVE_FORWARD", "ATTACK"),
    "turn_left_shoot": ("TURN_LEFT", "ATTACK"),
    "turn_right_shoot": ("TURN_RIGHT", "ATTACK"),
    "forward_use": ("MOVE_FORWARD", "USE"),
}

_UI_BUTTONS = {
    "wait": (),
    "forward": ("up",),
    "back": ("down",),
    "turn_left": ("left",),
    "turn_right": ("right",),
    "forward_left": ("up", "left"),
    "forward_right": ("up", "right"),
    "strafe_left": ("left",),
    "strafe_right": ("right",),
    "shoot": ("FIRE",),
    "use": ("USE",),
    "forward_shoot": ("up", "FIRE"),
    "turn_left_shoot": ("left", "FIRE"),
    "turn_right_shoot": ("right", "FIRE"),
    "forward_use": ("up", "USE"),
}

_LABELS = {
    "wait": "wait",
    "forward": "walk forward",
    "back": "walk back",
    "turn_left": "turn left",
    "turn_right": "turn right",
    "forward_left": "walk + turn left",
    "forward_right": "walk + turn right",
    "strafe_left": "strafe left",
    "strafe_right": "strafe right",
    "shoot": "shoot",
    "use": "use / open",
    "forward_shoot": "walk + shoot",
    "turn_left_shoot": "turn left + shoot",
    "turn_right_shoot": "turn right + shoot",
    "forward_use": "walk + use",
}


class DoomGame(Game):
    meta = GameMeta(
        id="doom",
        title="Doom",
        short_name="DOOM",
        platform_id="doom",
        platform_label="Doom / ViZDoom",
        cartridge="Doom - E1M1 Hangar",
        subtitle="You are the marine - clear Hangar, don't die",
        aspect_width=640,
        aspect_height=480,
        button_keys=("left", "right", "up", "down", "FIRE", "USE"),
        default_hold_frames=1,
        brain_steps=24,
        brain_warmup=80,
    )
    actions = tuple(Action(key=key, label=_LABELS[key], buttons=_UI_BUTTONS[key]) for key in ACTIONS)

    def create_env(self):
        env = make_doom("e1m1")
        self.cartridge_label = env.cartridge
        return env

    def catalog_entry(self):
        entry = super().catalog_entry()
        entry["cartridge"] = getattr(self, "cartridge_label", None)
        return entry

    def reset(self, env, seed: int):
        return env.reset(seed=seed)

    def new_memory(self):
        return DoomMemory()

    def observe(self, env, info, memory, hold_frames: int, previous_action: str):
        memory.previous_action = previous_action
        return observe(env, info, memory, hold_frames)

    def scene(self, observation: dict) -> dict:
        player = observation["player"]
        target = observation.get("best_target")
        facts = [
            {"label": "job", "value": "You are the marine. Clear Hangar, kill demons, don't die."},
            {"label": "marine",
             "value": f"{player.get('facing')} x={player['x']} y={player['y']} hp={player['health']} "
                      f"{player.get('weapon')} ammo={player['ammo']}"},
            {"label": "target",
             "value": f"{target['kind']} sx={target['screen_x']} d={target.get('distance')}"
                      if target else "none"},
        ]
        overlays = []
        for obj in observation.get("nearby_objects") or []:
            if obj.get("role") not in {"monster", "hazard"}:
                continue
            if obj.get("on_screen") and obj.get("screen_x") is not None:
                overlays.append({"kind": obj["kind"], "x": obj["screen_x"], "y": 160, "w": 32, "h": 64})
        return {"facts": facts, "overlays": overlays,
                "text": (observation.get("briefing")
                         or "You are the Doom marine. Clear Hangar. Kill demons. Don't die.")}

    def stats(self, observation: dict, info: dict, session: dict) -> list[Stat]:
        player = observation["player"]
        return [
            Stat("HEALTH", str(player["health"])),
            Stat("AMMO", str(player["ammo"])),
            Stat("FACE", str(player.get("facing") or "").upper()),
            Stat("KILLS", str(player.get("kills", 0))),
        ]

    def advance(self, env, action_key: str, frames: int, observation: dict) -> AdvanceResult:
        buttons = env.vector(*ACTIONS[action_key])
        reward = 0.0
        terminated = truncated = False
        info = env.info()
        frame = env.frame()
        executed = 0
        for _ in range(frames):
            frame, value, terminated, truncated, info = env.step(buttons)
            reward += float(value)
            executed += 1
            if terminated or truncated or info.get("dead"):
                terminated = True
                break
        return AdvanceResult(frame=frame, info=info, reward=reward,
                             terminated=bool(terminated), truncated=bool(truncated),
                             frames_executed=executed)

    def scripted(self, observation: dict) -> tuple[str, dict]:
        target = observation.get("best_target")
        if not target or target.get("role") != "monster":
            target = None
        combat = observation.get("combat") or {}
        progress = observation.get("progress") or {}
        tactics = observation.get("tactics") or {}
        move = "forward"
        turn = "stay"
        shoot = False
        use = False
        if tactics.get("mode") == "fight" and target and target.get("on_screen"):
            sx = target.get("screen_x")
            angle = target.get("angle_deg") or 0
            if (sx is not None and sx < 280) or angle < -12:
                turn = "left"
                move = "back" if combat.get("someone_will_eat_you") else "stay"
            elif (sx is not None and sx > 360) or angle > 12:
                turn = "right"
                move = "back" if combat.get("someone_will_eat_you") else "stay"
            else:
                shoot = True
                move = "back" if (target.get("distance") or 0) < 72 else "stay"
        elif progress.get("try_use") and "use" not in (tactics.get("forbidden") or []):
            use = True
            move = "stay"
        elif tactics.get("mode") == "unstick" or progress.get("blocked"):
            turn = tactics.get("prefer_turn") or progress.get("open_side") or "left"
            move = "stay" if progress.get("wall_ahead") or "forward" in (tactics.get("forbidden") or []) else "forward"
        move, turn, shoot, use = _apply_tactics(observation, move, turn, shoot, use)
        action = _compose_doom(move, turn, shoot, use)
        return action, {
            "confidence": 1.0,
            "probabilities": {"move": {move: 1.0}, "turn": {turn: 1.0}},
            "decisions": {"move": move, "turn": turn, "shoot": 1.0 if shoot else 0.0,
                          "mode": tactics.get("mode"), "source": "scripted"},
        }

    def completed(self, info: dict) -> bool:
        return bool(info.get("exited"))

    def finish_memory(self, memory, before, env, info, action, result):
        remember(memory, before, info, action, result)

    # --- fly contract ---
    def fly_encoder(self, client):
        from fly_games.games.doom.fly import DoomFlyEncoder
        return DoomFlyEncoder(client)

    def fly_coarse_actions(self):
        return coarse_actions()

    def fly_coarse_of(self, fine_action: str, observation: dict):
        return coarse_of(fine_action, observation)

    def fly_expand(self, coarse: str, observation: dict, fine_hint: str | None = None):
        return expand(coarse, observation, fine_hint)

    def fly_hand_decode(self, command: dict, observation: dict) -> tuple[str, dict]:
        return hand_decode(command, observation)


def _apply_tactics(observation, move, turn, shoot, use):
    tactics = observation.get("tactics") or {}
    progress = observation.get("progress") or {}
    history = observation.get("history") or {}
    combat = observation.get("combat") or {}
    forbidden = set(tactics.get("forbidden") or [])
    prefer = tactics.get("prefer_turn") if tactics.get("prefer_turn") in {"left", "right"} else progress.get("open_side")
    live = int(observation.get("live_monsters_on_screen") or 0)
    target = observation.get("best_target")
    if live <= 0 or (target and target.get("role") != "monster"):
        shoot = False
    if "use" in forbidden or history.get("use_already_failed"):
        use = False
    if "forward" in forbidden and move == "forward":
        move = "stay"
        if turn == "stay":
            turn = prefer or "left"
    mode = tactics.get("mode")
    if not mode and progress.get("blocked"):
        mode = "unstick"
    if mode == "unstick" and not combat.get("must_fight"):
        if use and progress.get("try_use") and "use" not in forbidden:
            move, turn = "stay", "stay"
        else:
            use = False
            if turn == "stay":
                turn = prefer or "left"
            if progress.get("wall_ahead"):
                move = "stay"
    elif mode != "fight" and move == "stay" and turn in {"left", "right"} and not progress.get("blocked"):
        move, turn = "forward", "stay"
    return move, turn, shoot, use


def _compose_doom(move, turn, shoot, use):
    if use:
        return "forward_use" if move == "forward" else "use"
    turning = turn in {"left", "right"}
    walking = move == "forward"
    if shoot and turning:
        return "turn_left_shoot" if turn == "left" else "turn_right_shoot"
    if shoot and walking:
        return "forward_shoot"
    if shoot:
        return "shoot"
    if walking and turning:
        return "forward_left" if turn == "left" else "forward_right"
    if turning:
        return "turn_left" if turn == "left" else "turn_right"
    if move == "back":
        return "back"
    if walking:
        return "forward"
    return "wait"
