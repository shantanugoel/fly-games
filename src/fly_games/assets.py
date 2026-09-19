"""Locate optional commercial ROMs / IWADs without putting them in git."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def find_file(
    filenames: tuple[str, ...],
    *,
    env_keys: tuple[str, ...] = (),
    extra_dirs: tuple[Path, ...] = (),
) -> Path | None:
    for key in env_keys:
        raw = os.getenv(key, "").strip()
        if raw:
            path = Path(raw).expanduser()
            if path.is_file():
                return path.resolve()
    root = project_root()
    home = Path.home()
    dirs = (
        root / "roms",
        root,
        root.parent / "kungfu2",
        root.parent / "kungfu_rl_rs",
        home / "Downloads",
        home / "dev" / "kungfu2",
        *extra_dirs,
    )
    wanted = {name.lower() for name in filenames}
    for folder in dirs:
        if not folder.is_dir():
            continue
        for child in folder.iterdir():
            if child.is_file() and child.name.lower() in wanted:
                return child.resolve()
    return None


def find_kungfu_rom() -> Path:
    path = find_file(
        (
            "kungfu.nes",
            "kung_fu.nes",
            "kung fu (japan, usa) (en).nes",
            "kung fu (u).nes",
            "kung fu (usa).nes",
            "spartan x.nes",
            "spartanx.nes",
        ),
        env_keys=("KUNGFU_NES", "JEV_KUNGFU_ROM", "FLY_KUNGFU_ROM"),
    )
    if path is None:
        raise FileNotFoundError(
            "NES Kung Fu / Spartan X ROM not found. Put a legally obtained "
            "kungfu.nes in roms/ or set KUNGFU_NES."
        )
    return path


def find_doom_iwad() -> tuple[Path, str]:
    """Return (path, kind) where kind is shareware, original, doom2, or freedoom."""
    original = find_file(
        (
            "doom.wad",
            "doom1.wad",
            "doomu.wad",
            "doom2.wad",
        ),
        env_keys=("DOOM_IWAD", "JEV_DOOM_IWAD", "FLY_DOOM_IWAD"),
        extra_dirs=(
            Path.home() / "Library/Application Support/Steam/steamapps/common/Ultimate Doom/base",
            Path.home() / "Library/Application Support/Steam/steamapps/common/Doom 2/base",
            Path.home() / "Library/Application Support/GZDoom",
        ),
    )
    if original is not None:
        name = original.name.lower()
        if name.startswith("doom2"):
            kind = "doom2"
        elif name in {"doom1.wad"}:
            kind = "shareware"
        else:
            kind = "original"
        return original, kind
    import vizdoom as vzd

    freedoom = Path(vzd.__file__).resolve().parent / "freedoom1.wad"
    if not freedoom.is_file():
        raise FileNotFoundError("No Doom IWAD and ViZDoom Freedoom Phase 1 is missing.")
    return freedoom, "freedoom"
