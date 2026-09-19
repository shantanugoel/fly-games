"""Platform adapters: emulator/runtime backends that games plug into.

A platform owns how to construct an environment and which physical buttons
exist. Games own ROM-specific observations and action maps.
"""

from .doom import DoomSession, make_doom
from .nes import make_mario_env, make_nes_rom_env, nes_press

__all__ = [
    "DoomSession",
    "make_doom",
    "make_mario_env",
    "make_nes_rom_env",
    "nes_press",
]
