"""NES (nes-py) platform helpers.

New NES games: point `make_nes_rom_env` at a legally obtained `.nes` ROM,
wrap the joypad with the game's action lists, then decode that game's RAM.
Mario uses gym-super-mario-bros because it registers SMB1 and exposes level
info; the emulator underneath is still nes-py.
"""

from pathlib import Path

from nes_py.wrappers import JoypadSpace

_NES_BUTTONS = {
    "right": 0b10000000,
    "left": 0b01000000,
    "down": 0b00100000,
    "up": 0b00010000,
    "start": 0b00001000,
    "select": 0b00000100,
    "B": 0b00000010,
    "A": 0b00000001,
}


def nes_press(env, *buttons: str, frames: int = 1):
    """Tap raw NES buttons, including Start, bypassing JoypadSpace."""
    raw = getattr(env, "unwrapped", env)
    mask = 0
    for button in buttons:
        mask |= _NES_BUTTONS[button]
    frame = info = None
    terminated = truncated = False
    for _ in range(frames):
        frame, _, terminated, truncated, info = raw.step(mask)
        if terminated or truncated:
            break
    return frame, info, bool(terminated), bool(truncated)


def make_mario_env(env_id: str, action_lists: list[list[str]], render_mode="rgb_array"):
    import gym_super_mario_bros

    env = gym_super_mario_bros.make(env_id, render_mode=render_mode)
    return JoypadSpace(env, action_lists)


def make_nes_rom_env(rom: str | Path, action_lists: list[list[str]], render_mode="rgb_array"):
    from nes_py import NESEnv

    path = Path(rom)
    if not path.is_file():
        raise FileNotFoundError(
            f"NES ROM not found: {path}. Place a legally obtained ROM there to add this game."
        )
    env = NESEnv(str(path), render_mode=render_mode)
    return JoypadSpace(env, action_lists)
