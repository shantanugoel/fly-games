# Cartridges

Two of the three titles need a commercial cartridge. They are **not** in this
repository — `.gitignore` excludes `roms/*.nes`, `roms/*.wad` and `roms/*.zip`
— so put your own legally obtained copies here, or point at them with an
environment variable (see `.env.example`).

| Game     | File                                  | Notes |
| -------- | ------------------------------------- | ----- |
| Mario    | *nothing*                              | Super Mario Bros. ships inside `gym-super-mario-bros`; no ROM needed. |
| Kung Fu  | `kungfu.nes` (or `spartan x.nes`)      | `KUNGFU_NES` also works. |
| Doom     | `doom1.wad`, `doom.wad` or `doom2.wad` | `DOOM_IWAD` also works. Falls back to ViZDoom's bundled Freedoom Phase 1 if no IWAD is found. |

Recognised Kung Fu filenames: `kungfu.nes`, `kung_fu.nes`,
`Kung Fu (Japan, USA) (en).nes`, `Kung Fu (U).nes`, `Kung Fu (USA).nes`,
`Spartan X.nes`, `spartanx.nes`.

## Already own them elsewhere?

The lookup searches `roms/`, the project root, `../kungfu2`, `../kungfu_rl_rs`
and `~/Downloads`, so a symlink works as well as a copy:

```sh
ln -s ~/dev/somewhere/kungfu.nes roms/kungfu.nes
```

## Check what was found

```sh
fly-games play --game kungfu --policy scripted --decisions 3
fly-games play --game doom   --policy scripted --decisions 3
```

The header it prints names the cartridge it actually loaded.