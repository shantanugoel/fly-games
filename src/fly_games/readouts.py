"""Where fitted readouts live.

One .npz per game, plus a sidecar .json recording which coarse labels and which
command features it was fitted against, so a stale file fails loudly instead of
silently mislabelling decisions.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "FLY_GAMES_READOUTS"
DEFAULT_DIR = "readouts"


def readout_dir() -> Path:
    return Path(os.getenv(ENV_VAR, DEFAULT_DIR)).expanduser()


def readout_path(game_id: str, base: Path | str | None = None) -> Path:
    root = Path(base).expanduser() if base else readout_dir()
    return root / f"{game_id}.readout.npz"


def available_games(base: Path | str | None = None) -> list[str]:
    root = Path(base).expanduser() if base else readout_dir()
    if not root.is_dir():
        return []
    return sorted(p.stem.replace(".readout", "") for p in root.glob("*.readout.npz"))


__all__ = ["DEFAULT_DIR", "ENV_VAR", "available_games", "readout_dir", "readout_path"]