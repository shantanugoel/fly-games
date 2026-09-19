# fly-games

Play **Super Mario Bros.**, **Kung Fu / Spartan X** and **Doom** with a real
fruit-fly brain.

The brain is the **MaleCNS v1.0 connectome** — the complete central nervous
system of an adult male *Drosophila melanogaster*: **166,700 neurons** and
**25,582,938 synapses**, wired exactly as electron microscopy found them
(HHMI Janelia / Cambridge / Google Research, 2025–2026). It is a *frozen*
spiking network. Nothing in it is trained, ever.

Each game adds only two small pieces on top of that wiring:

- **Input — an encoder.** Structured game facts are mapped onto the fly's own
  visual-projection neurons, on the correct side of its body:

  | channel | neuron type | what it responds to in a real fly |
  | ------- | ----------- | --------------------------------- |
  | `loom`  | LPLC2       | something approaching / expanding |
  | `threat`| LC4         | fast looming → escape             |
  | `shot`  | LPLC1       | small approaching objects         |
  | `chase` | LC10a       | the target it is pursuing         |

  Raw pixels are *not* used. The fly's visual system processes features, and in
  this model the raw photoreceptor signal dies at the lamina anyway.

- **Output — a readout.** The fly's **descending command neurons** — escape,
  steer, forward, backward, punch, kick, each left and right, twelve populations
  in all, some of them a *single* neuron — are decoded into a controller
  decision by a small linear readout. That readout is the only thing that is
  ever fitted.

## Why the fly reacts

The connectome is a fixed nonlinear dynamical system — a reservoir. Drive its
LC4 neurons and the giant-fibre escape pathway fires; drive LC10a and it does
not. The readout learns "escape-dominant firing → jump, quiet firing → walk".
So the *real wiring* is what the fly reacts with; we only train the small
interpreter bolted onto its motor output.

Stimulate it yourself and watch that happen:

```sh
fly-games probe --types LC4 --volts 0.8 --would-do
#   escape_L        0.00    25.00   +25.00  ######################################
#   escape_R        0.00    20.83   +20.83  ###############################
#   if it were playing mario: escape [zero-shot heuristic]
```

## Setup

Python 3.13+ with [uv](https://docs.astral.sh/uv/); the frontend with
[bun](https://bun.sh).

```sh
uv sync
cp .env.example .env          # optional; every value has a working default

fly-games brain-download      # one-time, ~260 MB
fly-games brain-info          # confirm it loaded
```

Super Mario Bros. needs nothing. Kung Fu and Doom need a cartridge you already
own — see [roms/README.md](roms/README.md).

## Run it

```sh
fly-games lab                 # the observatory at http://127.0.0.1:8000
```

Three tabs, three different questions:

- **Lab** — the connectome, live. Every one of the 140,638 neurons with a known
  soma is drawn (optic lobes either side, ventral nerve cord trailing down, the
  way the published figures do it), with the 6,000 probed neurons coloured by
  superclass and the twelve command populations ringed and named. Brightness is
  spikes in the last decision window. Switch to **3-D** to rotate the actual
  coordinates. Below it: what the fly senses, what its descending neurons are
  firing, and what that resolved into.
- **Probe** — the bench. Stimulate any of the 11,863 annotated cell types
  directly, no emulator and no training involved, and read the delta in each
  command group. This is the tab that shows the biology is real.
- **Replay** — a finished episode, one still per decision. Roughly two thirds of
  an episode's wall clock is the fly thinking, so replay can collapse those
  gaps and play at emulator speed instead.

Headless:

```sh
fly-games play  --game mario --policy fly --decisions 30
fly-games play  --game mario --policy scripted -v      # -v prints the senses
fly-games train --game mario --episodes 8 --decisions 40
fly-games readouts
```

`--game` works before or after the subcommand: `fly-games --game doom play` and
`fly-games play --game doom` are the same thing.

## Train a fly

Training is reservoir computing: watch the scripted policy play, record the
fly's own descending activity at each decision, and fit a ridge decoder from
those rates to the coarse decision. Cross-validation is **leave-one-episode-out**
— decisions inside one episode are wildly correlated, so scoring within an
episode would flatter the readout, and the CLI says so when no episode boundary
was crossed.

```sh
fly-games train --game mario  --episodes 8 --decisions 40
fly-games train --game kungfu --episodes 8 --decisions 40
fly-games train --game doom   --episodes 8 --decisions 40
```

Readouts land in `readouts/<game>.readout.npz` with a sidecar recording the
labels they were fitted against, so a stale file fails loudly instead of
silently mislabelling decisions. Point `FLY_GAMES_READOUTS` elsewhere to keep
several.

## Policies

| policy     | what decides                                                            |
| ---------- | ----------------------------------------------------------------------- |
| `fly`      | the trained readout if there is one, otherwise the zero-shot rule        |
| `fly-hand` | force the zero-shot rule over the descending rates, never the readout    |
| `scripted` | deterministic baseline, no brain — also the label source for training    |

## Frontend

`fly-games lab` serves the built `frontend/dist`. For development:

```sh
cd frontend && bun install && bun run dev    # http://127.0.0.1:5173
```

The dev server proxies `/api` and `/ws` to the lab; set `FLY_GAMES_BACKEND` if
it lives somewhere else.

## How it hangs together

```
observation ──► game.fly_encoder() ──► inject LPLC2/LC4/LPLC1/LC10a (L/R)
                                            │
                                     frozen connectome
                                     24 steps at dt = 20 ms
                                            │
                    twelve descending command groups (rates in Hz)
                          │                        │
              readout.decode()              fly_hand_decode()
                          │                        │
                     coarse decision ──► game.fly_expand() ──► buttons
```

The brain is stepped under a lock — there is one fly, and it cannot be playing
Mario and being probed at the same time. A decision costs about **80 ms of CPU**
once warm (the first one pays ~0.9 s of warm-up), which is why the UI is built
around a visible thinking phase rather than hiding it.

## Layout

```
src/fly_games/
  brain/          connectome client, sensory encoders, the readout
  games/          per-title emulator + fly encoder/decoder
  platforms/      NES and Doom adapters
  engine.py       the loop: decide, advance, record, phase
  app.py          FastAPI + websocket + the static observatory
  probe.py        the stimulation bench
  cli.py          lab / play / train / probe / readouts / brain-info
frontend/         the observatory UI (bun + vite + react + three)
tests/            pytest; skips itself where data or a ROM is missing
roms/             your cartridges (gitignored)
```

## Development

```sh
uv run pytest            # 176 tests, ~17 s; skips what it cannot run
uv run ruff check src tests
cd frontend && bun run build
```

Tests need nothing to be green on a fresh clone: anything requiring the
connectome is marked `needs_brain` and anything requiring a cartridge is marked
with its game, so they skip rather than fail.

## Environment

Everything is optional — see [.env.example](.env.example). The ones worth
knowing: `FLY_DATA` (connectome location), `FLY_DEVICE` (`cpu` or `cuda`),
`FLY_GAMES_PORT`, `FLY_GAMES_READOUTS`, `KUNGFU_NES`, `DOOM_IWAD`.