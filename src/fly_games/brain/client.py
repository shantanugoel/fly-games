"""A thin, decision-oriented wrapper over flybrain.FlyBrain.

`FlyBrainClient` keeps one frozen connectome in memory and turns each game
decision into:

  * `fired`        indices that spiked during the decision window (for the UI).
  * `grid`         a 2-D firing heat-map over the fly's real soma layout.
  * `command`      per-command-group firing counts / rates (escape, steer,
                   forward, backward, punch, kick x L/R) -- the fly's output
                   to its body, i.e. the readout input.
  * `window_trace` a (brain_steps x n_command) 0/1 spike trace of command groups.
  * `field`        per-probe-neuron spike counts, for the 3-D observatory view.
  * `kinds`        firing counts per neuron superclass, for the "what lit up" bars.
  * `input`        the per-decision encoder drive labels for the UI.

The client is stateful: the fly keeps its membrane voltages and noise stream
across decisions (a live animal does not reset between thoughts). Call `reset()`
at episode start to re-seed it.

Everything spatial here is *real*: the point cloud is the MaleCNS soma
coordinates, the left/right split is the annotated `side` column, and the
command groups are the connectome's own `group_*` neuron lists.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from flybrain import FlyBrain

# The twelve descending command groups (flybrain FlyBrain.groups keys). These are
# the fly's *outputs* to its body and the natural readout features.
COMMAND_GROUPS = (
    "escape_L", "escape_R",
    "steer_L", "steer_R",
    "forward_L", "forward_R",
    "backward_L", "backward_R",
    "punch_L", "punch_R",
    "kick_L", "kick_R",
)

# channel -> neuron type list. The fly's identified visual-projection cells and
# what each one is known to respond to. Defined here (not in encoder.py) because
# the point cloud needs them to colour the sensory pools.
CHANNELS: dict[str, list[str]] = {
    "loom": ["LPLC2"],
    "threat": ["LC4"],
    "shot": ["LPLC1"],
    "chase": ["LC10a"],
}

CHANNEL_MEANING = {
    "loom": "something getting bigger / approaching",
    "threat": "fast looming, escape drive",
    "shot": "small approaching objects / projectiles",
    "chase": "the moving target the fly pursues",
}

# How many neurons the browser gets per-decision spike counts for. The full
# window fires ~50k neurons, which is far too much to ship 4x a second; a fixed
# stratified sample keeps the anatomy legible at ~6 kB per decision.
DEFAULT_PROBE_SIZE = 6000

# Bytes per point in /api/brain/points.bin (6 x u16).
POINT_BYTES = 12
NO_PROBE = 0xFFFF

# Stable colours for the 27 connectome superclasses (index order = SUPERCLASSES).
SUPERCLASS_COLORS = (
    "#7ee081", "#4fb3ff", "#ff8f5c", "#ffd166", "#c792ea", "#f78c6c", "#82aaff",
    "#c3e88d", "#f07178", "#ffcb6b", "#89ddff", "#b2ccd6", "#eeffff", "#676e95",
    "#ab47bc", "#0796b5", "#ff5370", "#a6e22e", "#e0e0e0", "#d2a4ff", "#80cbc4",
    "#ffbca6", "#a2b9c9", "#6b7a8f", "#f5f5f5", "#3f51b5", "#00bcd4",
)


_SHARED: FlyBrainClient | None = None
_SHARED_LOCK = threading.RLock()


def shared_client(**kwargs) -> FlyBrainClient:
    """One connectome per process.

    Loading the 166,700 x 25.6M weight matrices twice would double the memory
    for no reason, so the lab engine and the probe bench share a client. The
    lock serialises decisions: the fly cannot be in two places at once.
    """
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = FlyBrainClient(**kwargs)
        return _SHARED


def projection_axes(positions: np.ndarray, superclass: np.ndarray | None = None) -> tuple[int, int]:
    """Return the (horizontal, vertical) connectome axes for the canonical view.

    Every published figure of this nervous system is drawn the same way: the
    head-to-tail axis runs *down* the page, the left-right axis runs across it,
    so the brain sits at the top with its two optic lobes either side and the
    ventral nerve cord trails below like a tail. The connectome's own coordinate
    frame is not aligned to anatomy, so the axes are identified from the tissue
    rather than hardcoded:

      anterior-posterior  the axis that best separates the ventral nerve cord
                          from the central brain, in units of the whole
                          population's spread
      lateral             the axis the optic lobes are spread widest across,
                          relative to the central brain. Spread, not offset:
                          the two lobes straddle the midline, so their mean
                          sits exactly on the brain's

    Falls back to the two widest-spread axes when superclass labels are missing.
    `positions` has NaNs for the ~26k neurons with no soma coordinate, so every
    reduction here is NaN-aware.
    """
    finite = np.isfinite(positions)
    lo = np.where(finite.any(axis=0), np.nanmin(positions, axis=0), 0.0)
    hi = np.where(finite.any(axis=0), np.nanmax(positions, axis=0), 0.0)
    spread = np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64)
    if superclass is None:
        order = np.argsort(spread)[::-1]
        return int(order[0]), int(order[1])

    labels = np.asarray(superclass)
    ok = finite.all(axis=1)
    sigma = np.nanstd(positions, axis=0)
    sigma = np.where(sigma > 0, sigma, 1.0)

    def stat(name: str, how) -> np.ndarray | None:
        mask = ok & (labels == name)
        if mask.sum() < 50:
            return None
        return how(positions[mask], axis=0)

    brain = stat("cb_intrinsic", np.nanmean)
    cord = stat("vnc_intrinsic", np.nanmean)
    lobe_spread = stat("ol_intrinsic", np.nanstd)
    brain_spread = stat("cb_intrinsic", np.nanstd)
    if brain is None:
        order = np.argsort(spread)[::-1]
        return int(order[0]), int(order[1])

    if cord is not None:
        along = np.abs(cord - brain) / sigma
    else:
        along = spread / spread.max()
    body_axis = int(np.argmax(along))

    if lobe_spread is not None and brain_spread is not None:
        laterality = np.asarray(lobe_spread, dtype=np.float64) / np.where(
            np.asarray(brain_spread, dtype=np.float64) > 0, brain_spread, 1.0)
        laterality = np.asarray(laterality, dtype=np.float64)
        laterality[body_axis] = -1.0
        lateral_axis = int(np.argmax(laterality))
    else:
        ranked = np.argsort(spread)[::-1]
        lateral_axis = int(next(a for a in ranked if a != body_axis))
    # Horizontal across the screen, head-to-tail down it.
    return lateral_axis, body_axis


@dataclass(frozen=True)
class BrainDecision:
    """Result of one brain decision (a run of `brain_steps` LIF steps)."""

    fired: np.ndarray                       # indices that spiked in the window
    grid: np.ndarray                        # (grid_h, grid_w) sum of window firing
    command: dict[str, dict[str, float]]    # {group: {"count":, "rate":}}
    window_trace: np.ndarray                # (brain_steps, n_command) 0/1
    input: dict[str, float] | None          # {"channel_side": voltage}
    steps: int
    dt: float
    field: np.ndarray | None = None         # (n_probes,) uint8 spike counts
    kinds: tuple[tuple[str, int], ...] = ()  # superclass -> spike count

    def command_rates(self) -> dict[str, float]:
        """Group name (with side) -> per-step rate in Hz."""
        return {name: v["rate"] for name, v in self.command.items()}

    def feature(self) -> np.ndarray:
        """Flat (len(COMMAND_GROUPS),) per-group rate, in COMMAND_GROUPS order."""
        return np.fromiter(
            (v["rate"] for name, v in self.command.items() if name in COMMAND_GROUPS),
            dtype=np.float32,
            count=len(COMMAND_GROUPS),
        )


class FlyBrainClient:
    """Wrapper around one frozen fly connectome; holds LIF state across calls."""

    def __init__(
        self,
        device: str | None = None,
        data: Path | str | None = None,
        brain_steps: int = 24,
        brain_warmup: int = 80,
        refractory: float = 0.0,
        sensory_input: bool = True,
        seed: int = 64,
        grid_w: int = 128,
        grid_h: int = 72,
        probe_size: int = DEFAULT_PROBE_SIZE,
    ) -> None:
        self.device = device
        self.data = data
        self.seed = seed
        self.brain_steps = int(brain_steps)
        self.brain_warmup = int(brain_warmup)
        self.refractory = float(refractory)
        self.sensory_input = sensory_input
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.probe_size = int(probe_size)
        # The fly is one animal: only one thing may step it at a time.
        self.decision_lock = threading.RLock()
        self._brain: FlyBrain | None = None
        # Projections computed once from brain.positions + brain.groups.
        self._group_of: np.ndarray | None = None    # (n,) int16, -1 = non-command
        self._grid_cell: np.ndarray | None = None   # (n,) int32, -1 = no position
        # Rolling record of which command groups fired, one slot per LIF step.
        # This is what keeps the two clocks honest: the fly steps once per
        # emulator frame, so a window of `brain_steps` slots covers exactly
        # `brain_steps` frames of the world it is reacting to.
        self._ring: np.ndarray | None = None
        self._ring_at = 0
        self._ring_filled = 0
        self._positions: np.ndarray | None = None
        self._loaded = False
        # Spatial / visual metadata, built lazily (a few hundred ms).
        self._cloud: dict | None = None
        self._superclasses: tuple[str, ...] = ()
        self._kind_of: np.ndarray | None = None

    @property
    def brain(self) -> FlyBrain:
        return self._ensure()

    def _ensure(self) -> FlyBrain:
        if self._brain is not None:
            return self._brain
        # flybrain auto-downloads the ~260 MB brain on first FlyBrain() via
        # ensure_data if it is missing under $FLY_DATA (default ~/fly-data).
        self._brain = FlyBrain(
            data=self.data,
            seed=self.seed,
            device=self.device,
            sensory_input=self.sensory_input,
            refractory=self.refractory,
        )
        self._prep_projections()
        self._loaded = True
        return self._brain

    def _prep_projections(self) -> None:
        b = self._ensure()
        pos = b.positions
        if pos is None or pos.size == 0:
            raise RuntimeError("brain.npz has no positions; run `flybrain build` (not `download`).")
        self._positions = np.asarray(pos, dtype=np.float32)
        n = b.n

        # Project onto the two widest-spread axes. Ranking by *range* rather
        # than standard deviation matters here for two reasons: 26,062 neurons
        # have no position at all (so np.std is NaN for every axis and the
        # ranking is meaningless), and anatomically the widest axis is the
        # anterior-posterior one -- so the fly ends up drawn side-on, brain to
        # the left, ventral nerve cord to the right.
        ax0, ax1 = projection_axes(self._positions, getattr(b, "superclass", None))
        self._axes = (ax0, ax1)
        x = self._positions[:, ax0]
        y = self._positions[:, ax1]

        # Command group id per neuron.
        group_of = -np.ones(n, dtype=np.int16)
        for gid, name in enumerate(COMMAND_GROUPS):
            if name in b.groups:
                group_of[b.groups[name]] = gid
        self._group_of = group_of

        # Grid cell per neuron (col * grid_h + row), -1 for NaN position.
        x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        x_span = x_max - x_min if x_max > x_min else 1.0
        y_span = y_max - y_min if y_max > y_min else 1.0
        cell = -np.ones(n, dtype=np.int32)
        valid = np.isfinite(x) & np.isfinite(y)
        xh = np.clip((x[valid] - x_min) / x_span * self.grid_w + 0.5, 0, self.grid_w - 1).astype(int)
        yh = np.clip((y_max - y[valid]) / y_span * self.grid_h + 0.5, 0, self.grid_h - 1).astype(int)
        cell[valid] = xh * self.grid_h + yh
        self._grid_cell = cell
        self._kind_of = None
        self._cloud = None
        try:
            self._n_connections = int(b.weights.size)
        except Exception:                                  # noqa: BLE001 - external lib internals
            self._n_connections = None

    # --------------------------------------------------------------------------
    # lifecycle
    # --------------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        b = self._ensure()
        b.reset(seed if seed is not None else self.seed)
        self.clear_window()

    def clear_window(self) -> None:
        """Forget the trailing spike window. Call whenever the world restarts,
        or a new episode would be judged on the last one's afterimage."""
        self._ring = np.zeros((self.brain_steps, len(COMMAND_GROUPS)), dtype=np.float32)
        self._ring_at = 0
        self._ring_filled = 0

    def advance(self, frames: int, inject: Sequence[tuple[np.ndarray, float]] | None = None) -> None:
        """Step the fly once per emulator frame, holding `inject` constant.

        The fly's clock and the emulator's advance together - one LIF step per
        frame - so the animal never lives faster than the world it is reacting
        to. The old design spent `brain_steps` LIF steps on `hold_frames` frames
        of game, which put half a second of neural time on 67 ms of Mario and
        made the fly free-run ahead of everything it was supposed to be sensing.
        Same spike budget here, but those steps now cover `brain_steps` real
        frames instead of `hold_frames` of them.
        """
        if self._ring is None:
            self.clear_window()
        b = self._ensure()
        gid = self._group_of
        if gid is None:
            raise RuntimeError("brain projections not initialized")
        held = list(inject) if inject else []
        for _ in range(max(0, int(frames))):
            fired = b.step(inject=held)
            row = self._ring[self._ring_at]
            row[:] = 0.0
            g = gid[fired]
            g = g[g >= 0]
            if g.size:
                row[g] = 1.0
            self._ring_at = (self._ring_at + 1) % self.brain_steps
            self._ring_filled = min(self._ring_filled + 1, self.brain_steps)

    def command_rates(self) -> dict[str, dict[str, float]]:
        """Descending rates over the trailing window, in Hz of fly time."""
        span = max(1, self._ring_filled)
        counts = self._ring[:span].sum(axis=0) if self._ring is not None else np.zeros(len(COMMAND_GROUPS))
        dt = self._ensure().dt
        return {name: {"count": float(c), "rate": float(c / span / dt)}
                for name, c in zip(COMMAND_GROUPS, counts)}

    def sense(self, inject, input: dict | None = None, frames: int = 1,
              field: bool = False) -> BrainDecision:
        """Advance the shared clock by `frames`, then read the fly's urges.

        This is the turn-based contract made explicit: the game is frozen while
        the fly steps, the fly is frozen while the game runs, and neither can
        outrun the other. Nothing here is real time, so the ~80 ms a decision
        costs is a training cost, not a control limitation.
        """
        self.advance(frames, inject)
        b = self._ensure()
        span = max(1, self._ring_filled)
        counts = self._ring[:span].sum(axis=0)
        command = {name: {"count": float(c), "rate": float(c / span / b.dt)}
                   for name, c in zip(COMMAND_GROUPS, counts)}
        fired_all = np.flatnonzero(self._ring[:span].sum(axis=0) > 0).astype(np.int64)
        return BrainDecision(
            fired=fired_all,
            grid=np.zeros(self.grid_h * self.grid_w, dtype=np.float32).reshape(self.grid_h, self.grid_w),
            command=command,
            window_trace=self._ring[:span].copy(),
            input=input or {},
            steps=span,
            dt=b.dt,
            field=self.spike_field(fired_all) if field else None,
            kinds={},
        )

    def warmup(self, steps: int | None = None) -> None:
        """Let the silent network settle to its resting rate before real decisions.

        Goes through `advance` rather than stepping directly, so the trailing
        window is prefilled with resting activity instead of opening on zeros -
        an all-zero first decision reads as a dead fly to the readout.
        """
        self.advance(int(steps or self.brain_warmup))

    # --------------------------------------------------------------------------
    # one decision
    # --------------------------------------------------------------------------
    def decide(
        self,
        inject: Sequence[tuple[np.ndarray, float]] | None = None,
        input: dict[str, float] | None = None,
        brain_steps: int | None = None,
        fresh: bool = False,
        field: bool = True,
    ) -> BrainDecision:
        """Run the fly for `brain_steps` LIF steps, holding sensory `inject` constant
        throughout the window (a steady stimulus), and return the decision summary.

        `inject` is a constant list of (idx, amount) pairs applied at every LIF
        step. `input` is a flat human-readable {channel_side: voltage} map for the UI.
        `fresh=True` resets membrane state before stepping (episode restart).
        `field=False` skips the per-neuron spike sample (cheaper, for training runs).
        """
        b = self._ensure()
        steps = int(brain_steps or self.brain_steps)
        gid = self._group_of
        cell = self._grid_cell
        if gid is None or cell is None:
            raise RuntimeError("brain projections not initialized")
        if fresh:
            b.reset(self.seed)

        fired_all_parts = []
        cell_sums = np.zeros(self.grid_h * self.grid_w, dtype=np.float32)
        window_trace = np.zeros((steps, len(COMMAND_GROUPS)), dtype=np.float32)
        group_counts = np.zeros(len(COMMAND_GROUPS), dtype=np.float32)

        for t in range(steps):
            fired = b.step(inject=list(inject) if inject else ())
            fired_all_parts.append(fired)
            # grid bookkeeping.
            fc = cell[fired]
            fc = fc[fc >= 0]
            cell_sums += np.bincount(fc, minlength=self.grid_h * self.grid_w)
            # window trace row.
            gid_fire = gid[fired]
            gid_fire = gid_fire[gid_fire >= 0]
            window_trace[t, gid_fire] = 1.0
            # command counts.
            if gid_fire.size:
                group_counts += np.bincount(gid_fire, minlength=len(COMMAND_GROUPS))

        fired_all = np.concatenate(fired_all_parts) if fired_all_parts else np.empty(0, dtype=np.int64)
        command: dict[str, dict[str, float]] = {
            name: {"count": float(cnt), "rate": float(cnt / steps / b.dt)}
            for name, cnt in zip(COMMAND_GROUPS, group_counts)
        }
        return BrainDecision(
            fired=fired_all,
            grid=cell_sums.reshape(self.grid_h, self.grid_w),
            command=command,
            window_trace=window_trace,
            input=input,
            steps=steps,
            dt=b.dt,
            field=self.spike_field(fired_all) if field else None,
            kinds=self.kind_counts(fired_all),
        )

    # --------------------------------------------------------------------------
    # probe playground: stimulate named cell types, no emulator involved
    # --------------------------------------------------------------------------
    def cell_ids(self, types: Sequence[str], side: str | None = None) -> np.ndarray:
        return self._ensure().cells(list(types), side=side)

    def known_types(self) -> list[str]:
        """Every annotated cell type, for the probe browser's autocomplete."""
        b = self._ensure()
        return sorted({str(t) for t in b.cell_type if str(t)})

    def probe(
        self,
        events: Sequence[tuple[Sequence[str], str | None, float]],
        steps: int | None = None,
        fresh: bool = False,
    ) -> BrainDecision:
        """Drive arbitrary cell types and watch what the descending neurons do.

        `events` is a list of (cell_types, side, voltage). Nothing else is
        required -- no ROM, no emulator, no trained readout. This is the
        observatory's "poke the fly" mode.
        """
        inject: list[tuple[np.ndarray, float]] = []
        labels: dict[str, float] = {}
        for types, side, amount in events:
            idx = self.cell_ids(types, side=side)
            if idx.size == 0:
                continue
            inject.append((idx, float(amount)))
            key = f"{'+'.join(types)}{'_' + side if side else ''}"
            labels[key] = labels.get(key, 0.0) + round(float(amount), 3)
        return self.decide(inject, labels, brain_steps=steps, fresh=fresh)

    # --------------------------------------------------------------------------
    # spatial metadata for the observatory
    # --------------------------------------------------------------------------
    @property
    def superclasses(self) -> tuple[str, ...]:
        if not self._superclasses:
            b = self._ensure()
            names, counts = np.unique(np.asarray(b.superclass), return_counts=True)
            order = np.argsort(counts)[::-1]
            self._superclasses = tuple(str(names[i]) for i in order)
        return self._superclasses

    def _ensure_kind_of(self) -> np.ndarray:
        if self._kind_of is None:
            b = self._ensure()
            lookup = {name: i for i, name in enumerate(self.superclasses)}
            sc = np.asarray(b.superclass)
            kind = np.fromiter((lookup.get(str(s), 0) for s in sc), dtype=np.int16, count=sc.size)
            self._kind_of = kind
        return self._kind_of

    def cloud(self) -> dict:
        """Build (once) the browser point cloud + the stratified probe set.

        Points are the connectome's real soma coordinates with a position; the
        probe set is a fixed subset -- every command and sensory neuron plus a
        stratified sample of the rest -- that we can afford to report per-decision
        spike counts for.
        """
        if self._cloud is not None:
            return self._cloud
        b = self._ensure()
        pos = self._positions
        assert pos is not None
        ax0, ax1 = getattr(self, "_axes", projection_axes(pos, getattr(b, "superclass", None)))
        ax2 = next(i for i in range(3) if i not in (ax0, ax1))
        axes = (ax0, ax1, ax2)
        self._axes = (ax0, ax1)

        pts = pos[:, axes]
        keep = np.isfinite(pts).all(axis=1)
        idx = np.flatnonzero(keep)
        pts = pts[idx]

        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        span = np.where(hi - lo > 0, hi - lo, 1.0).astype(np.float32)
        # Normalise each axis to 0..65535 but keep the body's proportions by
        # scaling every axis against the longest one.
        longest = float(span.max())
        norm = np.clip((pts - lo) / longest, 0.0, 1.0)
        q = (norm * 65535.0).astype(np.uint16)

        group_of = self._group_of[idx]
        kind_of = self._ensure_kind_of()[idx]
        side = np.asarray(b.side)[idx]

        # --- probe set: every command + sensory neuron, then a stratified fill.
        wanted = np.zeros(idx.size, dtype=bool)
        wanted[group_of >= 0] = True
        channel_probe: dict[str, dict[str, list[int]]] = {}
        for channel, types in CHANNELS.items():
            channel_probe[channel] = {}
            for side_key in ("L", "R"):
                ids = self.cell_ids(types, side=side_key)
                local = self._to_local(idx, ids)
                wanted[local] = True
                channel_probe[channel][side_key] = local.tolist()

        remaining = np.flatnonzero(~wanted)
        budget = max(0, self.probe_size - int(wanted.sum()))
        rng = np.random.default_rng(20250919)
        if budget and remaining.size > budget:
            chosen = rng.choice(remaining, size=budget, replace=False)
            wanted[chosen] = True
        elif remaining.size:
            wanted[remaining] = True

        probe_local = np.flatnonzero(wanted)
        probe_id = np.full(idx.size, NO_PROBE, dtype=np.uint16)
        probe_id[probe_local] = np.arange(probe_local.size, dtype=np.uint16)

        # Remap channel probe lists from cloud-local -> probe id.
        inverse = np.full(idx.size, -1, dtype=np.int32)
        inverse[probe_local] = np.arange(probe_local.size, dtype=np.int32)
        channels_out = {
            channel: {s: inverse[np.asarray(v, dtype=np.int64)].tolist()
                      for s, v in sides.items()}
            for channel, sides in channel_probe.items()
        }
        command_out = {}
        for gid, name in enumerate(COMMAND_GROUPS):
            local = np.flatnonzero(group_of == gid)
            command_out[name] = {
                "neurons": len(b.groups.get(name, ())),
                "probe_ids": inverse[local].tolist(),
            }

        self._cloud = {
            "count": int(idx.size),
            "xyz": q,
            "min": lo.astype(np.float32),
            "span": span.astype(np.float32),
            "longest": np.float32(longest),
            "kind": kind_of.astype(np.uint8),
            "command": group_of.astype(np.int8),
            "probe_id": probe_id,
            "probe_points": probe_local.astype(np.uint32),
            "n_probes": int(probe_local.size),
            "neuron_index": idx.astype(np.int64),
            "channels": channels_out,
            "command_neurons": command_out,
            "side": side,
        }
        return self._cloud

    def _to_local(self, idx: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """Map global neuron indices onto their slot in the (filtered) cloud."""
        pos = np.searchsorted(idx, ids)
        pos = np.clip(pos, 0, idx.size - 1)
        ok = idx[pos] == ids
        return pos[ok]

    def spike_field(self, fired: np.ndarray) -> np.ndarray:
        """uint8 spike counts for each probe neuron, in probe-id order."""
        cloud = self.cloud()
        lookup = cloud.get("neuron_lookup")
        if lookup is None:
            lookup = np.full(self._ensure().n, -1, dtype=np.int32)
            lookup[cloud["neuron_index"]] = np.arange(cloud["count"], dtype=np.int32)
            cloud["neuron_lookup"] = lookup
        local = lookup[fired]
        local = local[local >= 0]
        probe_id = cloud["probe_id"][local]
        probe_id = probe_id[probe_id != NO_PROBE]
        counts = np.bincount(probe_id.astype(np.int64), minlength=cloud["n_probes"])
        return np.clip(counts, 0, 255).astype(np.uint8)

    def kind_counts(self, fired: np.ndarray) -> tuple[tuple[str, int], ...]:
        """Spike counts per superclass, biggest first (the 'what lit up' bars)."""
        if fired.size == 0:
            return ()
        kind = self._ensure_kind_of()[fired]
        counts = np.bincount(kind.astype(np.int64), minlength=len(self.superclasses))
        pairs = sorted(
            ((self.superclasses[i], int(c)) for i, c in enumerate(counts) if c),
            key=lambda kv: -kv[1],
        )
        return tuple(pairs[:12])

    def cloud_rev(self) -> str:
        """Short, stable fingerprint of the point cloud blob."""
        rev = getattr(self, "_cloud_rev", None)
        if rev is None:
            import hashlib

            rev = hashlib.sha256(self.points_blob()).hexdigest()[:12]
            self._cloud_rev = rev
        return rev

    def points_blob(self) -> bytes:
        """The whole cloud as one packed binary blob (served once, cached forever).

        Header (64 bytes): magic 'FLYPT1', u16 version, u16 point_bytes, u32 count,
        3xf32 min, 3xf32 span, f32 longest, u32 n_probes, u8 n_superclasses, pad.
        Points (12 B each): u16 x, u16 y, u16 z, u16 command group (0xffff = not a
        command neuron), u16 superclass index, u16 probe id (0xffff = not probed).
        Tail: n_probes x u32 cloud index for each probe id.
        """
        c = self.cloud()
        header = bytearray(64)
        header[0:6] = b"FLYPT1"
        header[6:8] = np.uint16(1).tobytes()
        header[8:10] = np.uint16(POINT_BYTES).tobytes()
        header[10:14] = np.uint32(c["count"]).tobytes()
        header[14:26] = np.asarray(c["min"], dtype="<f4").tobytes()
        header[26:38] = np.asarray(c["span"], dtype="<f4").tobytes()
        header[38:42] = np.float32(c["longest"]).tobytes()
        header[42:46] = np.uint32(c["n_probes"]).tobytes()
        header[46] = np.uint8(len(self.superclasses))
        points = np.zeros((c["count"], POINT_BYTES // 2), dtype=np.uint16)
        points[:, :3] = c["xyz"]
        points[:, 3] = c["command"].astype(np.int16).astype(np.uint16) & 0xFFFF
        points[:, 4] = c["kind"]
        points[:, 5] = c["probe_id"]
        tail = c["probe_points"].astype("<u4").tobytes()
        return bytes(header) + points.reshape(-1).tobytes() + tail

    # --------------------------------------------------------------------------
    # metadata
    # --------------------------------------------------------------------------
    @property
    def meta(self) -> dict:
        b = self._ensure()
        return {
            "device": b.device,
            "batch": b.batch,
            "dt": b.dt,
            "n_neurons": int(b.n),
            "n_connections": getattr(self, "_n_connections", None),
            "grid_h": self.grid_h,
            "grid_w": self.grid_w,
            "command_groups": list(COMMAND_GROUPS),
            "command_neurons": {
                name: len(b.groups[name]) if name in b.groups else 0
                for name in COMMAND_GROUPS
            },
            "seed": self.seed,
        }

    @property
    def command_groups(self) -> list[tuple[str, int]]:
        """(group name, neuron count) for each descending command group."""
        b = self._ensure()
        return [(name, len(b.groups[name]) if name in b.groups else 0)
                for name in COMMAND_GROUPS]

    @property
    def sensory_channels(self) -> list[tuple[str, int]]:
        """(channel name, neuron count) for each sensory drive the encoders use."""
        return [(channel, int(self.cell_ids(types).size))
                for channel, types in CHANNELS.items()]

    def brain_card(self) -> dict:
        """Everything the observatory needs to draw the brain, minus the points."""
        b = self._ensure()
        c = self.cloud()
        names, counts = np.unique(np.asarray(b.superclass), return_counts=True)
        order = np.argsort(counts)[::-1]
        colors = {name: SUPERCLASS_COLORS[i % len(SUPERCLASS_COLORS)]
                  for i, name in enumerate(self.superclasses)}
        return {
            **self.meta,
            "positions": int(c["count"]),
            "unpositioned": int(b.n - c["count"]),
            "probes": int(c["n_probes"]),
            "point_bytes": POINT_BYTES,
            # Content-addressed: the response is `immutable`, so the URL has to
            # change whenever the encoding or the projection does, or returning
            # visitors keep a stale cloud forever.
            "points_url": f"/api/brain/points.bin?rev={self.cloud_rev()}",
            # Which connectome axes the cloud was drawn with, as
            # (horizontal, vertical, depth), so a consumer can tell that the
            # figure is optic-lobes-across / nerve-cord-down without re-deriving
            # it. Not the literal x/y/z of the source data.
            "axes": [int(self._axes[0]), int(self._axes[1]),
                     int(next(i for i in range(3) if i not in self._axes))],
            "superclasses": [
                {"name": str(names[i]), "count": int(counts[i]),
                 "color": colors.get(str(names[i]), "#888888")}
                for i in order
            ],
            "command_groups": [
                {"name": name, **c["command_neurons"].get(name, {})}
                for name in COMMAND_GROUPS
            ],
            "channels": [
                {
                    "name": channel,
                    "types": types,
                    "responds_to": CHANNEL_MEANING.get(channel, ""),
                    "sides": {
                        side: {"neurons": int(self.cell_ids(types, side=side).size),
                               "probe_ids": c["channels"][channel][side]}
                        for side in ("L", "R")
                    },
                }
                for channel, types in CHANNELS.items()
            ],
        }

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.grid_h, self.grid_w)


__all__ = [
    "CHANNELS",
    "CHANNEL_MEANING",
    "COMMAND_GROUPS",
    "DEFAULT_PROBE_SIZE",
    "NO_PROBE",
    "POINT_BYTES",
    "SUPERCLASS_COLORS",
    "BrainDecision",
    "FlyBrainClient",
    "projection_axes",
    "shared_client",
]