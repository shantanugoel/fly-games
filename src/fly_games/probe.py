"""The probe bench: stimulate the fly directly, with no game attached.

The lab's other half. Pick identified cell types, a side, and a voltage, and
watch what the fly's descending command neurons do with it -- and, if a game's
readout has been trained, what that readout would have decided. This is the
fastest way to see the thesis of the whole project: the connectome is frozen,
so LC4 at 0.8 V produces escape firing whether or not anything is chasing the
fly, and the readout is just an interpreter standing behind it.

Nothing here is trained, and nothing here needs a ROM.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as _dc_field

import numpy as np

from fly_games.brain import CHANNEL_MEANING, CHANNELS, COMMAND_GROUPS, shared_client
from fly_games.readouts import readout_path

# A starting menu: the cells the game encoders use, the fly's output neurons,
# and a few other identified types worth poking.
PRESETS = (
    {"name": "startle (LC4, both sides)", "note": "fast looming -> giant fibre escape",
     "events": [{"types": ["LC4"], "side": None, "volts": 0.8}]},
    {"name": "looming left (LPLC2 L)", "note": "object expanding on the left",
     "events": [{"types": ["LPLC2"], "side": "L", "volts": 0.7}]},
    {"name": "looming right (LPLC2 R)", "note": "object expanding on the right",
     "events": [{"types": ["LPLC2"], "side": "R", "volts": 0.7}]},
    {"name": "small target ahead (LPLC1)", "note": "approaching projectile",
     "events": [{"types": ["LPLC1"], "side": None, "volts": 0.7}]},
    {"name": "chase (LC10a)", "note": "the fly's tracked moving target",
     "events": [{"types": ["LC10a"], "side": None, "volts": 0.7}]},
    {"name": "silence", "note": "resting state baseline", "events": []},
)


@dataclass
class ProbeResult:
    """What the fly did with a stimulus."""

    input: dict[str, float]
    command: dict[str, dict[str, float]]
    baseline: dict[str, float]
    kinds: list[tuple[str, int]]
    spikes: int
    steps: int
    dt: float
    latency_ms: float
    field: np.ndarray | None = None
    grid: np.ndarray | None = None
    would_do: dict[str, dict] = _dc_field(default_factory=dict)

    def public(self, include_field: bool = True) -> dict:
        import base64

        out = {
            "input": self.input,
            "command": self.command,
            "baseline": self.baseline,
            "delta": {name: round(self.command[name]["rate"] - self.baseline.get(name, 0.0), 3)
                      for name in self.command},
            "kinds": [[name, count] for name, count in self.kinds],
            "spikes": self.spikes,
            "steps": self.steps,
            "dt": self.dt,
            "latency_ms": self.latency_ms,
            "groups": list(COMMAND_GROUPS),
            "would_do": self.would_do,
        }
        if include_field and self.field is not None:
            out["field"] = base64.b64encode(np.ascontiguousarray(self.field).tobytes()).decode()
        if self.grid is not None:
            grid = np.asarray(self.grid, dtype=np.float64)
            peak = float(grid.max()) or 1.0
            out["grid"] = base64.b64encode(
                np.clip(grid / peak * 255.0, 0, 255).astype(np.uint8).tobytes()).decode()
            out["grid_size"] = list(grid.shape)
        return out


class ProbeBench:
    """Direct stimulation of the frozen connectome."""

    def __init__(self, steps: int = 24):
        self.steps = int(steps)
        self.client = shared_client()

    # --- what can be poked ---------------------------------------------------
    def menu(self) -> dict:
        client = self.client
        types = []
        for channel, names in CHANNELS.items():
            for name in names:
                types.append({
                    "name": name,
                    "role": channel,
                    "responds_to": CHANNEL_MEANING.get(channel, ""),
                    "L": int(client.cell_ids([name], side="L").size),
                    "R": int(client.cell_ids([name], side="R").size),
                })
        return {
            "channels": [{"name": c, "types": t, "responds_to": CHANNEL_MEANING.get(c, "")}
                         for c, t in CHANNELS.items()],
            "cell_types": types,
            "command_groups": [{"name": name, "neurons": count}
                               for name, count in client.command_groups],
            "presets": list(PRESETS),
            "max_volts": 0.8,
            "steps": self.steps,
        }

    def search(self, query: str, limit: int = 40) -> list[dict]:
        """Look up annotated cell types by name fragment, with counts."""
        import collections

        client = self.client
        brain = client.brain
        needle = (query or "").strip().upper()
        counts: dict[str, int] = collections.Counter()
        for name in brain.cell_type:
            name = str(name)
            if needle and needle not in name.upper():
                continue
            counts[name] += 1
            if len(counts) >= limit * 6:
                break
        out = []
        for name, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]:
            out.append({"name": name, "neurons": count,
                        "L": int(client.cell_ids([name], side="L").size),
                        "R": int(client.cell_ids([name], side="R").size)})
        return out

    # --- poke it -------------------------------------------------------------
    def run(self, events: Sequence[dict], steps: int | None = None,
            baseline: dict[str, float] | None = None, games: Sequence[str] = ()) -> ProbeResult:
        """Stimulate and measure. `events` is [{types: [...], side: "L"|"R"|None, volts: 0.8}]."""
        from time import monotonic

        clean = []
        for event in events or []:
            types = [str(t) for t in (event.get("types") or []) if str(t).strip()]
            if not types:
                continue
            side = event.get("side") or None
            volts = max(0.0, min(0.8, float(event.get("volts", 0.5))))
            clean.append((types, side, volts))
        steps = int(steps or self.steps)
        client = self.client
        with client.decision_lock:
            if baseline is None:
                baseline = {name: value["rate"]
                            for name, value in client.decide(field=False).command.items()}
            started = monotonic()
            decision = client.probe(clean, steps=steps)
            latency_ms = round((monotonic() - started) * 1000, 2)
        command = {name: {"count": v["count"], "rate": round(v["rate"], 3)}
                   for name, v in decision.command.items()}
        result = ProbeResult(
            input=dict(decision.input or {}),
            command=command,
            baseline={k: round(v, 3) for k, v in (baseline or {}).items()},
            kinds=list(decision.kinds),
            spikes=int(decision.fired.size),
            steps=decision.steps,
            dt=decision.dt,
            latency_ms=latency_ms,
            field=decision.field,
            grid=np.asarray(decision.grid),
        )
        for game_id in games:
            verdict = self.what_it_would_do(command, game_id)
            if verdict:
                result.would_do[game_id] = verdict
        return result

    def what_it_would_do(self, command: dict, game_id: str) -> dict | None:
        """Decode these command rates with a game's readout (or its zero-shot rule)."""
        from fly_games.brain import FlyReadout
        from fly_games.catalog import get_game

        try:
            game = get_game(game_id)
        except ValueError:
            return None
        path = readout_path(game.meta.id)
        readout = (FlyReadout.load(game.fly_coarse_actions(), path) if path.exists()
                   else FlyReadout(game.fly_coarse_actions(), path))
        feature = np.fromiter((command.get(name, {}).get("rate", 0.0)
                               for name in COMMAND_GROUPS), dtype=np.float64,
                              count=len(COMMAND_GROUPS))
        if readout.trained:
            coarse, probs = readout.decode(feature)
            source = f"readout({readout.model.kind})"
        else:
            coarse, probs = game.fly_hand_decode(command, {})
            source = "zero-shot heuristic"
        return {
            "coarse": coarse,
            "probabilities": {k: round(float(v), 3) for k, v in probs.items()},
            "source": source,
        }


__all__ = ["PRESETS", "ProbeBench", "ProbeResult"]