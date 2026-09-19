"""Doom (ViZDoom): original IWAD E1M1 when available, else Freedoom Phase 1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import vizdoom as vzd

from fly_games.assets import find_doom_iwad

BUTTONS = (
    "ATTACK",
    "USE",
    "SPEED",
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
)

VARIABLES = (
    vzd.GameVariable.HEALTH,
    vzd.GameVariable.ARMOR,
    vzd.GameVariable.AMMO2,
    vzd.GameVariable.AMMO3,
    vzd.GameVariable.SELECTED_WEAPON,
    vzd.GameVariable.SELECTED_WEAPON_AMMO,
    vzd.GameVariable.POSITION_X,
    vzd.GameVariable.POSITION_Y,
    vzd.GameVariable.ANGLE,
    vzd.GameVariable.KILLCOUNT,
    vzd.GameVariable.DEAD,
    vzd.GameVariable.HITCOUNT,
)

# Live threats / pickups. Corpses (Dead*, Gib*) are map gore, not enemies.
MONSTER_NAMES = {
    "Arachnotron",
    "Archvile",
    "BaronOfHell",
    "BossBrain",
    "Cacodemon",
    "ChaingunGuy",
    "CommanderKeen",
    "Cyberdemon",
    "Demon",
    "DoomImp",
    "Fatso",
    "HellKnight",
    "LostSoul",
    "Mancubus",
    "PainElemental",
    "Revenant",
    "ShotgunGuy",
    "Spectre",
    "SpiderMastermind",
    "WolfensteinSS",
    "Zombieman",
}
ITEM_NAMES = {
    "ArmorBonus",
    "Backpack",
    "Berserk",
    "BFG9000",
    "BlueArmor",
    "BlueCard",
    "BlueSkull",
    "BlurSphere",
    "Cell",
    "CellPack",
    "Chaingun",
    "Chainsaw",
    "Clip",
    "ClipBox",
    "GreenArmor",
    "HealthBonus",
    "Infrared",
    "InvulnerabilitySphere",
    "Medikit",
    "Megasphere",
    "PlasmaRifle",
    "RadSuit",
    "RedCard",
    "RedSkull",
    "RocketAmmo",
    "RocketBox",
    "RocketLauncher",
    "Shell",
    "ShellBox",
    "Shotgun",
    "Soulsphere",
    "Stimpack",
    "SuperShotgun",
    "YellowCard",
    "YellowSkull",
}
_ITEM_CATEGORIES = {
    "Ammo",
    "Armor",
    "Artifact",
    "Health",
    "Key",
    "Power",
    "Powerup",
    "Weapon",
}
_SCENERY_CATEGORIES = {
    "Decoration",
    "DynamicLight",
    "Gibs",
    "Gore",
    "LightSource",
    "Player",
    "Self",
    "Vegetation",
}


def classify_object(name: str, category: str | None = None) -> str:
    """monster / hazard / item / scenery. Gore corpses are scenery."""
    cat = (category or "").strip()
    if cat == "Monster" or name in MONSTER_NAMES:
        return "monster"
    if cat in {"Explosive", "Hazard"} or name == "ExplosiveBarrel":
        return "hazard"
    if cat in _ITEM_CATEGORIES or name in ITEM_NAMES:
        return "item"
    if cat in _SCENERY_CATEGORIES:
        return "scenery"
    if name.startswith(("Dead", "Gib")):
        return "scenery"
    return "scenery"


EPISODE_TIMEOUT = 126_000  # 60 minutes at 35 tics/sec


@dataclass
class DoomObject:
    name: str
    x: float
    y: float
    z: float
    dx: float
    dy: float
    distance: float
    angle_deg: float
    on_screen: bool
    screen_x: float | None = None
    screen_y: float | None = None
    screen_w: float | None = None
    screen_h: float | None = None
    category: str | None = None
    role: str = "scenery"


class DoomSession:
    def __init__(self, map_name="e1m1"):
        self.iwad, self.iwad_kind = find_doom_iwad()
        self.map_name = "map01" if self.iwad_kind == "doom2" else map_name
        self.timeout = EPISODE_TIMEOUT
        self.game = vzd.DoomGame()
        self.game.set_doom_game_path(str(self.iwad))
        self.game.set_doom_map(self.map_name)
        self.game.set_window_visible(False)
        self.game.set_sound_enabled(False)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.set_render_hud(True)
        self.game.set_render_crosshair(True)
        self.game.set_render_weapon(True)
        self.game.set_render_messages(True)
        self.game.set_objects_info_enabled(True)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_episode_start_time(1)
        self.game.set_episode_timeout(self.timeout)
        self.game.set_living_reward(0)
        self.game.set_death_penalty(0)
        self.game.set_mode(vzd.Mode.PLAYER)
        self.game.set_doom_skill(3)
        for button in BUTTONS:
            self.game.add_available_button(getattr(vzd.Button, button))
        for variable in VARIABLES:
            self.game.add_available_game_variable(variable)
        self.buttons = list(self.game.get_available_buttons())
        self.button_index = {button.name: i for i, button in enumerate(self.buttons)}
        self.game.init()
        self._last_state = None
        self.width = 640
        self.height = 480

    @property
    def cartridge(self) -> str:
        lump = self.map_name.upper()
        if self.iwad_kind == "shareware":
            return f"Doom shareware · {lump} Hangar ({self.iwad.name})"
        if self.iwad_kind == "original":
            return f"Doom · {lump} Hangar ({self.iwad.name})"
        if self.iwad_kind == "doom2":
            return f"Doom II · {lump} ({self.iwad.name})"
        return f"Freedoom Phase 1 · {lump} (no Doom IWAD; not original Hangar)"

    def vector(self, *names: str) -> list[bool]:
        pressed = [False] * len(self.buttons)
        for name in names:
            pressed[self.button_index[name]] = True
        return pressed

    def reset(self, seed=None):
        if seed is not None:
            self.game.set_seed(int(seed) & 0xFFFFFFFF)
        self.game.new_episode()
        self._last_state = self.game.get_state()
        return self.frame(), self.info()

    def step(self, buttons: list[bool]):
        reward = float(self.game.make_action(buttons, 1))
        terminated = bool(self.game.is_episode_finished())
        if not terminated:
            self._last_state = self.game.get_state()
        return self.frame(), reward, terminated, False, self.info()

    def frame(self):
        state = self._last_state
        if state is None or state.screen_buffer is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        buffer = state.screen_buffer
        if buffer.ndim == 3 and buffer.shape[0] == 3:
            buffer = np.transpose(buffer, (1, 2, 0))
        return np.ascontiguousarray(buffer)

    def info(self):
        game = self.game
        dead = bool(game.get_game_variable(vzd.GameVariable.DEAD)) or bool(game.is_player_dead())
        finished = bool(game.is_episode_finished())
        tics = int(game.get_episode_time())
        timed_out = finished and not dead and tics >= self.timeout - 2
        exited = finished and not dead and not timed_out
        return {
            "health": int(game.get_game_variable(vzd.GameVariable.HEALTH)),
            "armor": int(game.get_game_variable(vzd.GameVariable.ARMOR)),
            "ammo": int(game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)),
            "bullets": int(game.get_game_variable(vzd.GameVariable.AMMO2)),
            "shells": int(game.get_game_variable(vzd.GameVariable.AMMO3)),
            "weapon": int(game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON)),
            "x": float(game.get_game_variable(vzd.GameVariable.POSITION_X)),
            "y": float(game.get_game_variable(vzd.GameVariable.POSITION_Y)),
            "angle": float(game.get_game_variable(vzd.GameVariable.ANGLE)),
            "kills": int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)),
            "hits": int(game.get_game_variable(vzd.GameVariable.HITCOUNT)),
            "dead": dead,
            "episode_finished": finished,
            "timed_out": timed_out,
            "exited": exited,
            "map": self.map_name,
            "iwad": self.iwad.name,
            "iwad_kind": self.iwad_kind,
            "buttons": [button.name for button in self.buttons],
        }

    def objects(self) -> list[DoomObject]:
        state = self._last_state
        if state is None:
            return []
        player_x = float(self.game.get_game_variable(vzd.GameVariable.POSITION_X))
        player_y = float(self.game.get_game_variable(vzd.GameVariable.POSITION_Y))
        player_angle = float(self.game.get_game_variable(vzd.GameVariable.ANGLE))
        labels = {}
        if state.labels:
            for label in state.labels:
                labels[int(label.object_id)] = label
        objects = []
        for obj in state.objects or []:
            name = str(obj.name)
            if name in {"DoomPlayer", "TeleportFog"}:
                continue
            dx = float(obj.position_x) - player_x
            dy = float(obj.position_y) - player_y
            distance = (dx * dx + dy * dy) ** 0.5
            rel = (np.degrees(np.arctan2(dy, dx)) - player_angle + 180) % 360 - 180
            label = labels.get(int(obj.id))
            category = None if label is None else str(label.object_category)
            objects.append(
                DoomObject(
                    name=name,
                    x=float(obj.position_x),
                    y=float(obj.position_y),
                    z=float(obj.position_z),
                    dx=round(dx, 1),
                    dy=round(dy, 1),
                    distance=round(distance, 1),
                    angle_deg=round(float(rel), 1),
                    on_screen=label is not None,
                    screen_x=None if label is None else float(label.x),
                    screen_y=None if label is None else float(label.y),
                    screen_w=None if label is None else float(label.width),
                    screen_h=None if label is None else float(label.height),
                    category=category,
                    role=classify_object(name, category),
                )
            )
        objects.sort(key=lambda item: item.distance)
        return objects

    def depth_view(self) -> dict:
        """Median depth (0 = touching a wall, 255 = far) in ahead / left / right wedges."""
        state = self._last_state
        depth = None if state is None else getattr(state, "depth_buffer", None)
        if depth is None:
            return {"ahead": 255.0, "left": 255.0, "right": 255.0}
        view = depth[: int(depth.shape[0] * 0.72)]
        height, width = view.shape[:2]
        band = view[max(0, int(height * 0.55) - 14) : int(height * 0.55) + 14]
        center = band[:, width // 2 - 18 : width // 2 + 18]
        left = band[:, width // 8 : width // 8 + 36]
        right = band[:, 7 * width // 8 - 36 : 7 * width // 8]
        return {
            "ahead": float(np.median(center)) if center.size else 255.0,
            "left": float(np.median(left)) if left.size else 255.0,
            "right": float(np.median(right)) if right.size else 255.0,
        }

    def close(self):
        self.game.close()


def make_doom(map_name="e1m1") -> DoomSession:
    return DoomSession(map_name=map_name)
