"""Per-game readout: decode the fly's descending command output into a coarse
controller decision.

This is the *only* trainable piece. The fly connectome is frozen; the readout
is a linear (ridge) decoder fitted on the fly's descending-neuron activity,
trained by **reservoir computing** on data collected with the game's scripted
policy. See fly.ai: input -> encoder -> frozen brain -> trace -> trained linear
readout -> output.

Features: the per-command-group firing rates (COMMAND_GROUPS), i.e.
`FlyBrainClient.decide(...).feature()`. There are twelve of them and they are
tiny populations -- escape/steer/forward/kick are a *single* neuron each -- so
the readout is deliberately low-capacity.

`FlyReadout.fit` trains it (leave-one-episode-out CV so one episode's
correlated samples never leak); `decode` predicts the coarse decision; the
*engine* then maps the coarse decision to a fine button action via
`game.fly_expand`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from flybrain import Readout

from .client import COMMAND_GROUPS


def _softmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


@dataclass
class ReadoutInfo:
    """What the fitted readout is, for the UI's 'how is it deciding' panel."""

    kind: str
    components: int | None
    lam: float
    cv_score: float
    n_samples: int = 0
    trained_at: str | None = None
    # Leave-one-episode-out argmax accuracy of the readout against its own
    # labels, with the majority-class baseline it has to beat. cv_score is a
    # regression R^2 against 0/1 targets and goes negative for classifiers that
    # are right nearly every time, so it cannot be the number a user decides
    # with; this one can.
    accuracy: float | None = None
    baseline: float | None = None

    def public(self) -> dict:
        return {
            "kind": self.kind,
            "components": self.components,
            "lam": self.lam,
            "cv_score": self.cv_score,
            "n_samples": self.n_samples,
            "trained_at": self.trained_at,
            "accuracy": self.accuracy,
            "baseline": self.baseline,
        }


class FlyReadout:
    """A trained linear readout over the fly's command rates, per game."""

    def __init__(self, coarse_actions: Sequence[str], path: str | Path, model=None,
                 info: ReadoutInfo | None = None):
        self.coarse_actions = tuple(coarse_actions)
        self.n = len(self.coarse_actions)
        self.path = Path(path)
        self.model = model
        self.info = info
        self._action_index = {a: i for i, a in enumerate(self.coarse_actions)}

    @property
    def trained(self) -> bool:
        return self.model is not None

    @property
    def sidecar(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".json")

    # --- train ---
    @classmethod
    def fit(
        cls,
        coarse_actions: Sequence[str],
        features: np.ndarray,
        fine_actions: Sequence[str],
        observations: Sequence[dict] | None = None,
        coarse_of: Callable[[str, dict], str] | None = None,
        episode_ids: Sequence[int] | None = None,
        *,
        path: str | Path = "model.npz",
        kind: str = "ridge",
        components: Sequence[int] = (2, 5, 10),
        lambdas: Sequence[float] = (1e-2, 1e-1, 1.0, 10.0),
    ) -> FlyReadout:
        """Fit from training data.

        features: (n_samples, n_command) fly command-rate features.
        fine_actions: the fine action the scripted policy chose at each sample.
        observations: optional per-sample observation dicts (needed by coarse_of).
        coarse_of: (fine_action, observation) -> coarse label.
        episode_ids: one episode id per sample (leave-one-episode-out CV).
        """
        if coarse_of is None:
            raise ValueError("coarse_of is required")
        X = np.asarray(features, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] != len(fine_actions):
            raise ValueError("features must be (n_samples, n_features)")
        n = X.shape[0]
        idx_of = {a: j for j, a in enumerate(coarse_actions)}
        y = np.zeros((n, len(coarse_actions)), dtype=np.float32)
        for i in range(n):
            obs = observations[i] if observations is not None else None
            c = coarse_of(fine_actions[i], obs if obs is not None else {})
            j = idx_of.get(c)
            if j is not None:
                y[i, j] = 1.0
        # With one episode there is no held-out group to score against, so fall
        # back to plain k-fold rather than reporting a meaningless -inf.
        groups = None
        if episode_ids is not None and len(set(episode_ids)) > 1:
            groups = np.asarray(episode_ids)
        # PCA cannot exceed the number of features (there are only 12).
        usable = tuple(k for k in components if k <= X.shape[1]) or (X.shape[1],)
        model = Readout.fit(
            X, y, kind=kind, groups=groups,
            components=usable, lambdas=tuple(lambdas),
        )
        info = ReadoutInfo(
            kind=model.kind, components=model.components, lam=model.lam,
            cv_score=float(model.cv_score), n_samples=n,
            trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        return cls(coarse_actions, path, model=model, info=info)

    # --- predict ---
    def decode(self, feature: Sequence[float]) -> tuple[str, dict[str, float]]:
        """Return (coarse_action, probabilities)."""
        if self.model is None:
            raise RuntimeError("FlyReadout.model is None; train it first.")
        feat = np.asarray(feature, dtype=np.float64).reshape(1, -1)
        raw = np.asarray(self.model.predict(feat), dtype=np.float64).reshape(-1)
        probs = _softmax(raw) if raw.size > 1 else np.array([1.0 - raw[0], raw[0]])
        if probs.size != self.n:
            probs = (np.pad(probs, (0, self.n - probs.size))
                     if probs.size < self.n else probs[: self.n])
        coarse = self.coarse_actions[int(np.argmax(probs))]
        return coarse, {a: float(p) for a, p in zip(self.coarse_actions, probs)}

    def weights(self) -> dict[str, dict[str, float]]:
        """Which command neurons the readout listens to, per coarse action.

        The model is linear on the top principal components, so the effective
        sensitivity of each decision to each command group is `(P / sd) @ w.T`.
        That is the honest, human-readable version of "what the readout was
        taught": positive means this decision's score rises when that descending
        neuron fires more. Values are scaled by one shared factor so the actions
        stay comparable to each other.
        """
        if self.model is None:
            return {}
        _mu, P, sd = self.model.basis
        w = np.asarray(self.model.w, dtype=np.float64)
        if w.ndim == 1:
            w = w.reshape(1, -1)
        # z = ((x - mu) @ P) / sd with P (n_features, k); y = z @ w.T + b
        # => dy/dx = (P / sd) @ w.T, i.e. (w @ (P / sd).T) per output row.
        scaled = np.asarray(P, dtype=np.float64) / np.asarray(sd, dtype=np.float64)
        effective = w @ scaled.T                      # (n_coarse, n_command)
        span = float(np.abs(effective).max()) or 1.0
        names = list(COMMAND_GROUPS[: effective.shape[1]])
        out: dict[str, dict[str, float]] = {}
        for j, action in enumerate(self.coarse_actions):
            row = effective[min(j, effective.shape[0] - 1)]
            out[action] = {name: float(value / span) for name, value in zip(names, row)}
        return out

    # --- save / load ---
    def save(self) -> Path:
        if self.model is None:
            raise RuntimeError("nothing to save; fit the readout first")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.path))
        meta = {
            "coarse_actions": list(self.coarse_actions),
            "features": list(COMMAND_GROUPS),
            **(self.info.public() if self.info else {}),
        }
        self.sidecar.write_text(json.dumps(meta, indent=2))
        return self.path

    @classmethod
    def load(cls, coarse_actions: Sequence[str], path: str | Path) -> FlyReadout:
        path = Path(path)
        if not path.exists():
            return cls(coarse_actions, path, model=None)
        model = Readout.load(path)
        info = ReadoutInfo(kind=model.kind, components=model.components, lam=model.lam,
                           cv_score=float(model.cv_score))
        sidecar = path.with_suffix(path.suffix + ".json")
        if sidecar.is_file():
            try:
                meta = json.loads(sidecar.read_text())
            except (OSError, ValueError):
                meta = {}
            stored = meta.get("coarse_actions")
            if stored and list(stored) != list(coarse_actions):
                raise ValueError(
                    f"{path} was trained for {stored}, not {list(coarse_actions)}. "
                    f"Re-run `fly-games train`."
                )
            info.n_samples = int(meta.get("n_samples", 0))
            info.trained_at = meta.get("trained_at")
            info.accuracy = meta.get("accuracy")
            info.baseline = meta.get("baseline")
        return cls(coarse_actions, path, model=model, info=info)


__all__ = ["COMMAND_GROUPS", "FlyReadout", "ReadoutInfo"]