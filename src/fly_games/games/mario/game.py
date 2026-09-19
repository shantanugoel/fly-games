"""NES Super Mario Bros. via the nes-py platform and SMB1 RAM decoding."""

from fly_games.games.base import Game
from fly_games.games.mario import memory
from fly_games.games.mario.fly import (
    MarioFlyEncoder,
)
from fly_games.games.mario.fly import (
    coarse_actions as _coarse_actions,
)
from fly_games.games.mario.fly import (
    coarse_of as _coarse_of,
)
from fly_games.games.mario.fly import (
    expand as _expand,
)
from fly_games.games.mario.fly import (
    hand_decode as _hand_decode,
)
from fly_games.platforms.nes import make_mario_env
from fly_games.types import Action, AdvanceResult, GameMeta, Stat

# NES SMB1 buttons, in joypad action-list order.
ACTIONS = {
    "wait": [],
    "right": ["right"],
    "right_jump": ["right", "A"],
    "right_run": ["right", "B"],
    "right_run_jump": ["right", "B", "A"],
    "left": ["left"],
    "left_jump": ["left", "A"],
    "jump": ["A"],
}

_LABELS = {key: key.replace("_", " ") for key in ACTIONS}


class MarioGame(Game):
    meta = GameMeta(
        id="mario",
        title="Super Mario Bros.",
        short_name="MARIO",
        platform_id="nes",
        platform_label="NES / nes-py",
        cartridge="Super Mario Bros. (USA)",
        subtitle="RAM facts -> sensory drives -> fly descending neurons -> buttons",
        aspect_width=256,
        aspect_height=240,
        button_keys=("left", "right", "A", "B"),
        default_hold_frames=1,
        brain_steps=24,
        brain_warmup=80,
    )
    actions = tuple(
        Action(key=key, label=_LABELS[key], buttons=tuple(buttons))
        for key, buttons in ACTIONS.items()
    )

    def __init__(self, world=1, stage=1):
        self.world = world
        self.stage = stage

    @property
    def env_id(self) -> str:
        return f"SuperMarioBros-{self.world}-{self.stage}-v0"

    # --- engine contract ---
    def create_env(self):
        return make_mario_env(self.env_id, [tuple(buttons) for buttons in ACTIONS.values()])

    def reset(self, env, seed: int):
        frame, info = env.reset(seed=seed)
        for _ in range(16):
            frame, _, _, _, info = env.step(0)
        return frame, info

    def new_memory(self):
        return memory.ObservationMemory()

    def dump_state(self, env) -> object | None:
        """Snapshot the emulator.

        `super-mario-bros-rl` builds its entire curriculum on this: rather than
        hope an agent reaches the hard parts of a level, it saves a state and
        starts episodes there. Round-trips exactly - all 64 KB of RAM is
        identical after a restore - so a curriculum built this way is
        reproducible, and it matters here because everything past screen one is
        otherwise unmeasurable: Mario used to die at frame 106 of 213.
        """
        unwrapped = getattr(env, "unwrapped", None) if env is not None else None
        dump = getattr(unwrapped, "dump_state", None)
        return dump() if callable(dump) else None

    def load_state(self, env, blob: object) -> bool:
        unwrapped = getattr(env, "unwrapped", None) if env is not None else None
        load = getattr(unwrapped, "load_state", None)
        if blob is None or not callable(load):
            return False
        load(blob)
        return True

    def observe(self, env, info, memory, hold_frames: int, previous_action: str):
        memory.previous_action = previous_action
        return memory.observe(env.unwrapped.ram, info, hold_frames)

    def scene(self, observation: dict) -> dict:
        mario = observation["mario"]
        camera = observation.get("camera_x")
        if camera is None:
            camera = max(0, mario["x"] - 40)
        enemies = observation.get("nearby_objects") or []
        threat = observation.get("nearest_threat")
        gap = (observation.get("terrain") or {}).get("summary", {}).get(
            "nearest_empty_column_below_feet"
        )
        obstacle = (observation.get("terrain") or {}).get("summary", {}).get(
            "nearest_obstacle"
        )
        overlays = [
            {"kind": "Mario", "x": max(0, mario["x"] - camera), "y": mario["y"], "w": 16, "h": 32}
        ]
        for obj in enemies[:6]:
            overlays.append(
                {"kind": obj["kind"], "x": max(0, mario["x"] + obj["dx"] - camera),
                 "y": mario["y"] + obj["dy"], "w": 16, "h": 16}
            )
        facts = [
            {"label": "mario",
             "value": f"x={mario['x']}, y={mario['y']}, {mario.get('status', '?')}, "
                      f"{'grounded' if mario['grounded'] else mario.get('motion', 'airborne')}"},
            {"label": "level", "value": f"{self.world}-{self.stage}"},
            {"label": "enemies",
             "value": ", ".join(f"{e['kind']}@dx{e['dx']}" for e in enemies[:4]) or "none"},
            {"label": "gap", "value": f"dx={gap['edge_distance_px']}" if gap else "not detected"},
            {"label": "wall",
             "value": f"dx={obstacle['distance_px']}" if obstacle else "none"},
            {"label": "threat",
             "value": f"{threat['kind']} dx={threat['dx']}" if threat else "none"},
        ]
        return {"facts": facts, "overlays": overlays, "text": _scene_text(observation, self.world, self.stage)}

    def stats(self, observation: dict, info: dict, session: dict) -> list[Stat]:
        return [
            Stat("DISTANCE", str(session.get("distance", 0))),
            Stat("COINS", str(int(info.get("coins", 0)))),
            Stat("DEATHS", str(session.get("deaths", 0))),
            Stat("LEVEL", f"{info.get('world', self.world)}-{info.get('stage', self.stage)}"),
        ]

    def advance(self, env, action_key: str, frames: int, observation: dict) -> AdvanceResult:
        action_index = self._action_index(action_key)
        previous_x = int(env.unwrapped.ram[0x6D]) * 256 + int(env.unwrapped.ram[0x86])
        previous_y = int(env.unwrapped.ram[0xCE])
        previous_grounded = int(env.unwrapped.ram[0x1D]) == 0
        samples = []
        reward = 0.0
        terminated = truncated = False
        info = {}
        for index in range(frames):
            _, value, terminated, truncated, info = env.step(action_index)
            reward += float(value)
            x = int(env.unwrapped.ram[0x6D]) * 256 + int(env.unwrapped.ram[0x86])
            y = int(env.unwrapped.ram[0xCE])
            grounded = int(env.unwrapped.ram[0x1D]) == 0
            sample = {
                "x": x, "y": y, "grounded": grounded, "frame": index + 1,
                "vx_px_per_frame": x - previous_x, "vy_px_per_frame": y - previous_y,
                "landed": not previous_grounded and grounded,
            }
            samples.append(sample)
            if terminated or truncated or sample["landed"]:
                break
            previous_x, previous_y, previous_grounded = x, y, grounded
        frame = env.render()
        if frame is None:
            frame = getattr(env.unwrapped, "screen", None)
        return AdvanceResult(
            frame=frame, info=info, reward=float(reward),
            terminated=bool(terminated), truncated=bool(truncated),
            frames_executed=len(samples), samples=samples,
        )

    def scripted(self, observation: dict) -> tuple[str, dict]:
        mario = observation["mario"]
        if not mario["grounded"]:
            return "right_run_jump", {}
        if observation.get("jump_already_held"):
            return "right_run", {}
        enemy = any(
            0 < obj["dx"] < 64 and abs(obj["dy"]) < 40 and obj["kind"] != "flagpole"
            for obj in observation["nearby_objects"]
        )
        obstacle = False
        gap = False
        feet_row = max(0, min(12, (mario["y"] + 32 - 32) // 16))
        for col in observation["terrain"]["columns"]:
            if 16 <= col["dx"] <= 48:
                obstacle |= bool(col["tiles"][max(0, feet_row - 1)])
                gap |= not any(col["tiles"][feet_row:])
        action = "right_run_jump" if enemy or obstacle or gap else "right_run"
        return action, {"source": "scripted"}

    def completed(self, info: dict) -> bool:
        return bool(info.get("flag_get"))

    def configure(self, world: int | None = None, stage: int | None = None):
        if world is not None and not 1 <= world <= 8:
            raise ValueError("World must be 1-8.")
        if world is not None:
            self.world = world
        if stage is not None and not 1 <= stage <= 4:
            raise ValueError("Stage must be 1-4.")
        if stage is not None:
            self.stage = stage

    def finish_memory(self, memory, before, env, info, action, result) -> None:
        memory.finish(before, env.unwrapped.ram, info, action,
                      result.frames_executed, result.reward,
                      result.terminated or result.truncated, samples=result.samples)

    # --- fly contract ---
    def fly_encoder(self, client) -> MarioFlyEncoder:
        return MarioFlyEncoder(client)

    def fly_coarse_actions(self) -> tuple[str, ...]:
        return _coarse_actions()

    def fly_coarse_of(self, fine_action: str, observation: dict) -> str:
        return _coarse_of(fine_action, observation)

    def fly_expand(self, coarse: str, observation: dict, fine_hint: str | None = None) -> str:
        return _expand(coarse, observation, fine_hint)

    def fly_hand_decode(self, command: dict, observation: dict) -> tuple[str, dict]:
        return _hand_decode(command, observation)

    @property
    def level_label(self) -> str:
        return f"{self.world}-{self.stage}"


def _scene_text(observation: dict, world: int, stage: int) -> str:
    mario = observation["mario"]
    enemies = observation.get("nearby_objects") or []
    enemy_text = ", ".join(
        f"{e.get('kind')}@dx{e.get('dx')},dy{e.get('dy')}" for e in enemies[:5]
    ) or "none"
    lines = [
        f"frame level {world}-{stage}",
        (f"mario: x={mario['x']}, y={mario['y']}, state={mario.get('status')}, "
         f"on_ground={int(mario['grounded'])}"),
        f"enemies: {enemy_text}",
        f"previous_action: {observation.get('previous_action')}",
    ]
    threat = observation.get("nearest_threat")
    if threat:
        lines.append(f"nearest_threat: {threat['kind']} dx={threat['dx']} "
                     f"contact={threat.get('estimated_contact_in_frames')}")
    return "\n".join(lines)
