"""The fly lab engine: one episode, one frozen brain, one decision at a time.

The fly thinks at about 1 Hz (24 LIF steps at dt=20 ms is ~0.8 s of CPU) while
the NES runs at 60 Hz. That asymmetry is the whole design: the emulator
**freezes** while the connectome settles, then the chosen button is held for a
handful of frames, then it freezes again. `phase` reports which of those the
lab is in so the UI can show the thought instead of hiding it.

Every decision records the three things that make the fly legible:

  * what its senses were driven with  (`fly.input`, from the game encoder)
  * what its descending neurons did   (`fly.command`, `fly.field`, `fly.grid`)
  * how that resolved into a button   (`thought.coarse` -> `thought.action`)

The brain is never trained. The only fitted object is the per-game readout.
"""

from __future__ import annotations

import base64
import os
import threading
from collections import Counter, deque
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

import numpy as np
from dotenv import load_dotenv

from fly_games.brain import COMMAND_GROUPS, FlyBrainClient, FlyReadout, shared_client
from fly_games.catalog import get_game

# The one decision every game exposes that is a reflex rather than a plan: all
# three declare it, and it is the only label worth overriding the fitted map for.
REFLEX_LABEL = "escape"
from fly_games.encode import encode_frame, encode_thumb
from fly_games.history import HistoryArchive

# The 2-D fallback view: the fly's real soma positions flattened onto the two
# most elongated axes, at this resolution.
GRID_HISTORY_W, GRID_HISTORY_H = 32, 18


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).decode()


def _scaled(arr: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Normalise a 2-D firing map to uint8, optionally resampling it."""
    grid = np.asarray(arr, dtype=np.float64)
    if shape is not None and grid.shape != shape:
        h, w = shape
        src_h, src_w = grid.shape
        ys = (np.arange(h) * src_h / h).astype(int).clip(0, src_h - 1)
        xs = (np.arange(w) * src_w / w).astype(int).clip(0, src_w - 1)
        grid = grid[np.ix_(ys, xs)]
    peak = float(grid.max())
    if peak <= 0:
        return np.zeros(grid.shape, dtype=np.uint8)
    return np.clip(grid / peak * 255.0, 0, 255).astype(np.uint8)


class Engine:
    """One playable episode plus the fly brain driving it."""

    def __init__(self, game_id: str = "mario", readout_dir: str | Path = "readouts",
                 seed: int | None = None):
        load_dotenv()
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.game = get_game(game_id)
        self.readout_dir = Path(readout_dir)
        self.env = None
        self.memory = None
        self.policy = "fly"                 # fly | fly-hand | scripted
        self.model_status = "ready"
        self.error: str | None = None
        self.running = False
        self.pending_step = False
        self.busy = False
        self.generation = 0
        self.revision = 0
        self.events: deque[dict] = deque(maxlen=8)
        self.archive = HistoryArchive()
        self._brain: FlyBrainClient | None = None
        self.hold_frames = self.game.meta.default_hold_frames
        self.seed = self.game.meta.default_seed if seed is None else int(seed)
        self.previous_action = "wait"
        # Let the fly's own escape reflex outvote a trained readout. See decide().
        self.reflex_override = True
        self.last_brain: dict | None = None
        self.last_scene: str | None = None
        self.input_frame: int | None = None
        self.last_diagnostics: dict = {}
        self.buttons: list[str] = []
        self.last_reason: str | None = None
        self.latency_ms: float | None = None
        self.frames = 0
        self.decisions = 0
        self.done = False
        self.completed = False
        self.last_scores: list = []
        self.last_model_action: str | None = None
        self.last_executed: str | None = None
        self.frame = None
        self.info: dict = {}
        self.observation: dict = {}
        self.session: dict = {}
        # Send the heavy per-neuron arrays only when somebody is watching.
        self.viz = False
        self._readout_cache: FlyReadout | None = None
        self._boot()
        self.worker = threading.Thread(target=self._work, name="fly-games", daemon=True)
        self.worker.start()

    # --------------------------------------------------------------------------
    # brain + readout
    # --------------------------------------------------------------------------
    def _get_brain(self) -> FlyBrainClient:
        if self._brain is None:
            self._brain = shared_client()
        self._brain.brain_steps = self.game.meta.brain_steps
        return self._brain

    def get_readout(self) -> FlyReadout:
        if self._readout_cache is None:
            path = self.readout_path()
            coarse = self.game.fly_coarse_actions()
            self._readout_cache = (FlyReadout.load(coarse, path) if path.exists()
                                   else FlyReadout(coarse, path, model=None))
        return self._readout_cache

    def readout_path(self) -> Path:
        return self.readout_dir / f"{self.game.meta.id}.readout.npz"

    def _forget_readout(self) -> None:
        self._readout_cache = None

    # --------------------------------------------------------------------------
    # boot / reset
    # --------------------------------------------------------------------------
    def _open_env(self):
        if self.env is not None:
            self.game.close(self.env)
            self.env = None
        self.env = self.game.create_env()

    def _reset_episode(self, *, record: bool):
        self.frame, self.info = self.game.reset(self.env, self.seed)
        self.memory = self.game.new_memory()
        self.frames = 0
        self.decisions = 0
        self.done = False
        self.completed = False
        self.previous_action = "wait"
        self.last_scores = []
        self.last_model_action = None
        self.last_executed = None
        self.buttons = []
        self.last_reason = None
        self.latency_ms = None
        self.last_scene = None
        self.input_frame = 0
        self.last_diagnostics = {}
        self.last_brain = None
        self.session = {
            "deaths": 0,
            "hits": 0,
            "distance": 0,
            "start_x": int(self.info.get("x_pos", 0) or 0),
            "best_x": int(self.info.get("x_pos", 0) or 0),
        }
        # reset the fly brain for this episode
        self._get_brain().reset()
        self.observation = self.game.observe(
            self.env, self.info, self.memory, self.hold_frames, "wait"
        )
        self.revision += 1
        if record:
            self.archive.start(self._episode_meta(), encode_thumb(self.frame))
        self.wake.set()

    def _boot(self):
        self._open_env()
        self._reset_episode(record=True)
        self.model_status = "ready"
        self.error = None
        readout = self.get_readout()
        trained = ("trained readout" if readout.trained
                   else "no readout yet -> zero-shot heuristic")
        self.event(f"{self.game.meta.title} ready on {self.game.meta.platform_label}. "
                   f"Policy {self.policy}; {trained}.")

    def _episode_meta(self) -> dict:
        meta = self.game.meta
        return {
            "game": meta.id,
            "game_title": meta.title,
            "short_name": meta.short_name,
            "world": getattr(self.game, "level_label", None),
            "seed": self.seed,
            "policy": self.policy,
            "hold_frames": self.hold_frames,
        }

    def event(self, message: str):
        self.events.appendleft({"time": datetime.now(UTC).strftime("%H:%M:%S"),
                                "message": message})

    def _update_session(self, result):
        info = result.info
        x = info.get("x_pos", 0)
        if x:
            x = int(x)
            self.session["best_x"] = max(self.session["best_x"], x)
            self.session["distance"] = max(0, self.session["best_x"] - self.session["start_x"])
        if result.reward > 0:
            self.session["hits"] = self.session.get("hits", 0) + 1

    # --------------------------------------------------------------------------
    # policy
    # --------------------------------------------------------------------------
    def set_policy(self, policy: str):
        if policy not in ("fly", "fly-hand", "scripted"):
            raise ValueError("Policy must be fly, fly-hand, or scripted.")
        if policy == "fly" and os.getenv("FLY_HAND_POLICY", "0") == "1":
            policy = "fly-hand"
        self.policy = policy
        self._forget_readout()
        self.revision += 1

    def decide(self, observation: dict) -> tuple[str, dict]:
        """Return (fine action key, diagnostics) for the current policy."""
        game = self.game
        if self.policy == "scripted":
            action, diagnostics = game.scripted(observation)
            diagnostics.setdefault("model", "scripted")
            self.last_brain = None
            return action, diagnostics

        brain = self._get_brain()
        encoder = game.fly_encoder(brain)
        inject, input_label = encoder.encode(observation)
        started = monotonic()
        with brain.decision_lock:
            decision = brain.decide(inject, input_label, field=self.viz)
        latency_ms = round((monotonic() - started) * 1000, 2)

        readout = self.get_readout()
        reflex, reflex_probs = game.fly_hand_decode(decision.command, observation)
        source = "fly"
        if self.policy != "fly-hand" and readout.trained:
            coarse, probs = readout.decode(decision.feature())
            model_name = f"fly-readout({readout.model.kind})"
            # The readout is fitted open-loop on states the scripted controller
            # visited, so in play it drifts onto states it was never trained on
            # and its confidence stops meaning much. The escape neurons are not
            # fitted: an over-active giant-fibre reflex means "jump" whatever the
            # map says about a state it has never seen. So the reflex overrides
            # the readout rather than being replaced by it.
            if self.reflex_override and reflex == REFLEX_LABEL and coarse != reflex:
                coarse, probs, source = reflex, reflex_probs, "reflex"
                model_name = f"fly-readout({readout.model.kind})+reflex"
        else:
            coarse, probs = reflex, reflex_probs
            model_name = "fly-hand"

        fine = game.fly_expand(coarse, observation)
        game._action_index(fine)          # fail loudly if a game expands to junk
        self.last_brain = self._brain_snapshot(decision)
        diagnostics = {
            "model": model_name,
            "confidence": float(probs.get(coarse, 0.0)),
            "probabilities": probs,
            "coarse": coarse,
            "latency_ms": latency_ms,
            "source": source,
            "command": {k: v["rate"] for k, v in decision.command.items()},
            "input": decision.input,
        }
        self.previous_action = fine
        return fine, diagnostics

    def _brain_snapshot(self, decision, history: bool = False) -> dict:
        """The per-decision fly record, for the live view or the archive."""
        grid = np.asarray(decision.grid)
        snap = {
            "command": {k: {"count": v["count"], "rate": round(v["rate"], 3)}
                        for k, v in decision.command.items()},
            "input": dict(decision.input or {}),
            "kinds": [[name, count] for name, count in decision.kinds],
            "steps": decision.steps,
            "dt": decision.dt,
            "spikes": int(decision.fired.size),
            "grid": _scaled(grid, (GRID_HISTORY_W, GRID_HISTORY_H) if history else None),
        }
        if not history and decision.field is not None:
            snap["field"] = decision.field
        return snap

    # --------------------------------------------------------------------------
    # run loop
    # --------------------------------------------------------------------------
    def _run_once(self):
        with self.lock:
            if (self.model_status != "ready" or self.done
                    or not (self.running or self.pending_step)):
                return None
            generation = self.generation
            hold = self.hold_frames
            observation = self.observation
            input_frame = self.frames
            self.pending_step = False
            self.busy = True          # phase: thinking
            self.revision += 1
        try:
            action, diagnostics = self.decide(observation)
            latency_ms = diagnostics.get("latency_ms")
            with self.lock:
                if generation != self.generation or self.stop.is_set():
                    return None
                self.busy = False     # phase: acting
                result = self.game.advance(self.env, action, hold, observation)
                self.game.finish_memory(self.memory, observation, self.env,
                                        self.info, action, result)
                self._install_result(action, result, diagnostics, latency_ms,
                                     input_frame, record=True)
                self.revision += 1
            return hold
        except Exception as exc:                              # noqa: BLE001 - worker boundary
            with self.lock:
                self.running = False
                self.error = f"Engine error: {exc}"
                self.revision += 1
            self.event("Stopped after an error.")
            return None
        finally:
            with self.lock:
                self.busy = False
                self.revision += 1

    def _install_result(self, action, result, diagnostics, latency_ms, input_frame, record):
        game = self.game
        self.frame = result.frame
        self.info = result.info
        self.frames += result.frames_executed
        self.decisions += 1
        self.previous_action = action
        self.buttons = list(game.actions[game._action_index(action)].buttons)
        self._update_session(result)
        self.observation = game.observe(self.env, self.info, self.memory,
                                        self.hold_frames, action)
        self.last_scores = diagnostics.get("scores") or []
        scene = game.scene(self.observation)
        self.last_scene = scene.get("text")
        self.input_frame = input_frame
        self.latency_ms = latency_ms
        self.last_diagnostics = diagnostics
        self.last_model_action = self.last_executed = action
        self.last_reason = diagnostics.get("source")
        self.done = result.terminated or result.truncated
        if self.done:
            self.running = False
            self.completed = bool(game.completed(result.info))
            if not self.completed:
                self.session["deaths"] += 1
        label = next((a.label for a in game.actions if a.key == action), action)
        if record:
            brain = self.last_brain
            self.archive.record({
                "input_frame": input_frame,
                "action": action,
                "label": label,
                "buttons": list(self.buttons),
                "frames_executed": result.frames_executed,
                "reward": float(result.reward),
                "latency_ms": latency_ms,
                "scores": list(self.last_scores),
                "scene": scene.get("text"),
                "facts": scene.get("facts"),
                "done": self.done,
                "completed": self.completed,
                "image": encode_thumb(result.frame),
                "model": diagnostics.get("model"),
                "brain": self._history_brain(brain, diagnostics),
            })
        self.revision += 1

    def step(self):
        """Take one decision on the caller's thread and return it.

        Returns ``(action, diagnostics, result)``, or None when there is nothing
        to run. This is the whole loop: it decides, advances the emulator, and
        then re-observes. Callers must not assemble those three themselves, in
        that order, because leaving the re-observe out is invisible in a unit
        test and fatal in play - the policy keeps acting on the frame it started
        with and walks into whatever the reset screen put in front of it.
        """
        with self.lock:
            if self.model_status != "ready" or self.done:
                return None
            observation = self.observation
            hold = self.hold_frames
            input_frame = self.frames
            self.busy = True
            self.revision += 1
        try:
            action, diagnostics = self.decide(observation)
            with self.lock:
                self.busy = False
                result = self.game.advance(self.env, action, hold, observation)
                self.game.finish_memory(self.memory, observation, self.env,
                                        self.info, action, result)
                self._install_result(action, result, diagnostics,
                                     diagnostics.get("latency_ms"),
                                     input_frame, record=True)
            return action, diagnostics, result
        except Exception as exc:
            with self.lock:
                self.error = f"Engine error: {exc}"
            raise
        finally:
            with self.lock:
                self.busy = False
                self.revision += 1

    @staticmethod
    def _history_brain(brain: dict | None, diagnostics: dict) -> dict | None:
        """The compact per-decision fly record kept in the episode archive.

        The heavy per-neuron arrays are live-view only; replay needs the command
        rates, the sensory drives, the readout's belief, and a small heat map.
        """
        if brain is None:
            return None
        return {
            "command": {k: v["rate"] for k, v in brain["command"].items()},
            "input": brain["input"],
            "kinds": brain["kinds"],
            "spikes": brain["spikes"],
            "grid": _b64(brain["grid"]),
            "grid_size": [GRID_HISTORY_W, GRID_HISTORY_H],
            "coarse": diagnostics.get("coarse"),
            "probabilities": diagnostics.get("probabilities"),
        }

    # --------------------------------------------------------------------------
    # training the readout (the only fitted thing)
    # --------------------------------------------------------------------------
    def collect_fly_samples(self, decisions: int, episodes: int = 0,
                            on_episode=None) -> list[tuple[list, str, dict]]:
        """Play with the scripted policy while the fly brain watches, and keep
        (command_rates, fine_action, observation) triples.

        The fly's command rates are the feature; the label is re-derived from
        (fine_action, observation) by the game's `fly_coarse_of`, so it matches
        exactly what the encoder drives.

        With `episodes`, `decisions` is the budget *per episode* and that many
        episodes are collected, so `--episodes 4 --decisions 40` means four runs
        of forty decisions. Episode ids then follow real boundaries, which is
        what makes the readout's leave-one-episode-out cross-validation honest.
        Without it, `decisions` is the total and only deaths split episodes.

        A forced boundary also changes the seed. `_reset_episode` replays the
        same seed, so resetting without reseeding would hand the cross-validator
        identical "different" episodes: a score that looks rigorous and has
        leaked completely is worse than no score at all.
        """
        game = self.game
        brain = self._get_brain()
        encoder = game.fly_encoder(brain)
        per_episode = int(decisions)
        total = per_episode * episodes if episodes else per_episode
        base_seed = self.seed
        samples: list[tuple[list, str, dict]] = []
        episode_ids: list[int] = []
        episode = 1
        since_boundary = 0
        previous = self.previous_action
        obs = self.observation
        try:
            for _ in range(total):
                if obs is None:
                    obs = game.observe(self.env, self.info, self.memory,
                                       self.hold_frames, previous)
                fine_action, _ = game.scripted(obs)
                inject, input_label = encoder.encode(obs)
                with brain.decision_lock:
                    decision = brain.decide(inject, input_label, field=False)
                samples.append(([float(v) for v in decision.feature()], fine_action, dict(obs)))
                episode_ids.append(episode)
                since_boundary += 1
                result = game.advance(self.env, fine_action, self.hold_frames, obs)
                game.finish_memory(self.memory, obs, self.env, self.info, fine_action, result)
                previous = fine_action
                obs = game.observe(self.env, result.info, self.memory,
                                   self.hold_frames, fine_action)
                ended = bool(result.terminated or result.truncated)
                spent = bool(episodes) and since_boundary >= per_episode and episode < episodes
                if ended or spent:
                    self.info = result.info
                    episode += 1
                    since_boundary = 0
                    if episodes:
                        self.seed = base_seed + episode - 1
                    self._reset_episode(record=False)
                    obs = self.observation
                    previous = "wait"
                    if on_episode:
                        on_episode(episode)
        finally:
            self.seed = base_seed
        self._episode_ids = episode_ids
        return samples

    def train_readout_for(self, samples: list[tuple[list, str, dict]],
                          episode_ids: list[int] | None = None) -> dict:
        """Fit a per-game readout from (command_rates, fine_action, obs) samples."""
        game = self.game
        coarse = game.fly_coarse_actions()
        if not samples:
            return {"error": "no training samples"}
        width = len(COMMAND_GROUPS)
        kept = [(f, fine, obs) for f, fine, obs in samples if len(f) == width]
        if not kept:
            return {"error": f"no samples with {width} command-rate features"}
        # Fitting on a handful of decisions produces a readout that looks
        # perfectly good, saves over the working one, and then mislabels every
        # decision it makes. Refuse loudly instead.
        minimum = max(8, 4 * len(coarse))
        if len(kept) < minimum:
            return {
                "error": f"need at least {minimum} samples to fit {len(coarse)} decisions, "
                         f"got {len(kept)}. Raise --decisions or --episodes; "
                         f"the existing readout was left untouched.",
            }
        if episode_ids is None:
            episode_ids = list(getattr(self, "_episode_ids", None) or [1] * len(kept))
        episode_ids = list(episode_ids)[: len(kept)] or [1] * len(kept)
        X = np.asarray([f for f, _, _ in kept], dtype=np.float64)
        readout = FlyReadout.fit(
            coarse, X, [fine for _, fine, _ in kept],
            observations=[obs for _, _, obs in kept],
            coarse_of=game.fly_coarse_of,
            episode_ids=episode_ids,
            path=self.readout_path(),
        )
        score = self._holdout_score(coarse, kept, episode_ids, game, readout)
        if score:
            readout.info.accuracy = score["accuracy"]
            readout.info.baseline = score["baseline"]
        path = readout.save()
        self._forget_readout()
        self.event(f"Trained {game.meta.id} readout on {X.shape[0]} decisions.")
        self.revision += 1
        return {
            "game": game.meta.id,
            "path": str(path),
            "kind": readout.model.kind,
            "components": readout.model.components,
            "lam": readout.model.lam,
            "cv_score": float(readout.model.cv_score),
            "n_samples": int(X.shape[0]),
            "episodes": len(set(episode_ids)),
            "accuracy": readout.info.accuracy,
            "baseline": readout.info.baseline,
            "minority_recall": score.get("minority_recall") if score else None,
        }

    def _holdout_score(self, coarse, kept, episode_ids, game, readout):
        """Leave-one-episode-out argmax accuracy, against the majority baseline.

        This exists because cv_score is a regression R^2 on 0/1 targets and reads
        as failure for a readout that picks the right decision 97% of the time.
        The question a user actually asks is "would this have decided correctly on
        an episode it never saw", so score that directly. Each fold refits on a
        scratch path; the saved model is the one fitted on everything.
        """
        ids = list(episode_ids or [])
        if not ids or len(set(ids)) < 2:
            return None
        import tempfile

        labels = [game.fly_coarse_of(fine, obs) for _f, fine, obs in kept]
        counts = Counter(labels)
        baseline = max(counts.values()) / len(labels)
        minority = min(counts, key=counts.get)
        hits = recall_hits = recall_n = 0
        for held in sorted(set(ids)):
            train = [i for i, e in enumerate(ids) if e != held]
            test = [i for i, e in enumerate(ids) if e == held]
            if not test or not train:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                fold = FlyReadout.fit(
                    coarse, [kept[i][0] for i in train],
                    [kept[i][1] for i in train],
                    observations=[kept[i][2] for i in train],
                    coarse_of=game.fly_coarse_of,
                    episode_ids=[ids[i] for i in train],
                    path=f"{tmp}/fold.npz",
                    components=(readout.model.components,),
                    lambdas=(readout.model.lam,),
                )
                for i in test:
                    picked, _probs = fold.decode(kept[i][0])
                    hits += picked == labels[i]
                    if labels[i] == minority:
                        recall_n += 1
                        recall_hits += picked == minority
        if not hits and not recall_n:
            return None
        return {
            "accuracy": hits / len(labels),
            "baseline": baseline,
            "minority": minority,
            "minority_recall": (recall_hits / recall_n) if recall_n else None,
        }

    def train(self, decisions: int, episodes: int = 0, on_episode=None) -> dict:
        return self.train_readout_for(
            self.collect_fly_samples(decisions, episodes=episodes, on_episode=on_episode))

    # --------------------------------------------------------------------------
    # worker
    # --------------------------------------------------------------------------
    def _work(self):
        while not self.stop.is_set():
            self.wake.wait(0.05)
            self.wake.clear()
            started = monotonic()
            hold = self._run_once()
            if hold is None:
                continue
            self.stop.wait(max(0, hold / 60.0 - (monotonic() - started)))

    # --------------------------------------------------------------------------
    # commands
    # --------------------------------------------------------------------------
    def command(self, command, **options):
        with self.lock:
            if command in {"run", "step"}:
                if command == "step" and (self.running or self.busy or self.pending_step):
                    raise ValueError("Pause and wait for the current decision before stepping.")
                self.running = command == "run"
                self.pending_step = command == "step"
            elif command == "pause":
                self.running = self.pending_step = False
                self.generation += 1
                self.event("Paused.")
            elif command == "reset":
                self.running = self.pending_step = False
                self.generation += 1
                self._reset_episode(record=True)
                self.event("Episode reset.")
            elif command == "select":
                game_id = options.get("game")
                if not game_id:
                    raise ValueError("Select needs a game id.")
                self.running = self.pending_step = False
                self.generation += 1
                self.game = get_game(game_id)
                self.hold_frames = self.game.meta.default_hold_frames
                self.seed = self.game.meta.default_seed
                self._forget_readout()
                self._open_env()
                self._reset_episode(record=True)
                self.event(f"Loaded {self.game.meta.title}.")
            elif command == "configure":
                policy = options.get("policy")
                if policy is not None:
                    self.set_policy(policy)
                frames = options.get("frames")
                if frames is not None:
                    if frames not in self.game.meta.hold_frames_options:
                        raise ValueError("Unsupported hold duration.")
                    self.hold_frames = frames
                seed = options.get("seed")
                if seed is not None:
                    self.seed = int(seed)
                    self._reset_episode(record=True)
                self.generation += 1
                self.event(f"Policy {self.policy}; hold {self.hold_frames} frames.")
            elif command == "viz":
                self.viz = bool(options.get("on", True))
            else:
                raise ValueError(f"Unknown command: {command!r}.")
            self.revision += 1
            self.wake.set()

    # --------------------------------------------------------------------------
    # snapshots
    # --------------------------------------------------------------------------
    def phase(self) -> str:
        if self.error:
            return "error"
        if self.done:
            return "complete" if self.completed else "dead"
        if self.busy:
            return "thinking"
        if self.running:
            return "acting"
        if self.pending_step:
            return "thinking"
        return "paused"

    def _fly_public(self) -> dict | None:
        brain = self.last_brain
        if brain is None:
            return None
        out = {
            "command": brain["command"],
            "input": brain["input"],
            "kinds": brain["kinds"],
            "steps": brain["steps"],
            "dt": brain["dt"],
            "spikes": brain["spikes"],
            "groups": list(COMMAND_GROUPS),
        }
        if "field" in brain:
            out["field"] = _b64(brain["field"])
        out["grid"] = _b64(brain["grid"]) if brain["grid"].dtype == np.uint8 else _b64(_scaled(brain["grid"]))
        out["grid_size"] = list(np.asarray(brain["grid"]).shape)
        return out

    def snapshot(self) -> dict:
        with self.lock:
            scene = self.game.scene(self.observation)
            meta = self.game.meta
            readout = self.get_readout()
            diag = self.last_diagnostics
            entry = self.game.catalog_entry()
            return {
                "revision": self.revision,
                "phase": self.phase(),
                "policy": self.policy,
                "running": self.running,
                "busy": self.busy,
                "done": self.done,
                "completed": self.completed,
                "error": self.error,
                "game": {
                    **entry,
                    "subtitle": meta.subtitle,
                    "aspect_width": meta.aspect_width,
                    "aspect_height": meta.aspect_height,
                    "button_keys": list(meta.button_keys),
                    "coarse_actions": list(self.game.fly_coarse_actions()),
                },
                "screen": {
                    "image": encode_frame(self.frame) if self.frame is not None else None,
                    "facts": scene.get("facts"),
                    "overlays": scene.get("overlays"),
                    "text": scene.get("text"),
                },
                "scene": self.last_scene or scene.get("text"),
                "stats": [
                    {"label": s.label, "value": s.value}
                    for s in self.game.stats(self.observation, self.info, self.session)
                ],
                "progress": {
                    "frames": self.frames,
                    "seconds": self.frames / 60,
                    "decisions": self.decisions,
                    "input_frame": self.input_frame if self.input_frame is not None else self.frames,
                    "hold_frames": self.hold_frames,
                    "seed": self.seed,
                },
                "thought": {
                    "model": diag.get("model"),
                    "coarse": diag.get("coarse"),
                    "probabilities": diag.get("probabilities") or (
                        {"scripted": 1.0} if self.policy == "scripted" else {}),
                    "confidence": diag.get("confidence"),
                    "action": self.last_model_action,
                    "label": next((a.label for a in self.game.actions
                                   if a.key == self.last_model_action), None),
                    "buttons": list(self.buttons),
                    "executed": self.last_executed,
                    "latency_ms": self.latency_ms,
                    "reason": self.last_reason,
                },
                "fly": self._fly_public(),
                "readout": {
                    "trained": readout.trained,
                    "coarse_actions": list(readout.coarse_actions),
                    "path": str(readout.path),
                    "info": readout.info.public() if readout.info else None,
                },
                "actions": [{"key": a.key, "label": a.label, "buttons": list(a.buttons)}
                            for a in self.game.actions],
                "events": list(self.events),
                "episodes": self.archive.summaries(),
            }

    # --- history / replay -------------------------------------------------------
    def history_summaries(self):
        with self.lock:
            return self.archive.summaries()

    def history_episode(self, episode_id, include_images=False):
        with self.lock:
            return self.archive.get(episode_id).public(include_images=include_images)

    def history_step(self, episode_id, index):
        with self.lock:
            episode = self.archive.get(episode_id)
            if index == -1:
                return {"index": -1, "action": None, "label": "episode start",
                        "image": episode.start_image, "facts": [], "scores": []}
            if index < 0 or index >= len(episode.decisions):
                raise ValueError("Unknown history step.")
            return episode.decisions[index].public(include_image=True)

    def close(self):
        """Idempotent: the CLI's `finally` and the server's lifespan both call
        this, and nes-py raises if you close its emulator twice."""
        self.stop.set()
        self.wake.set()
        self.worker.join(timeout=3)
        with self.lock:
            if self.env is not None:
                env, self.env = self.env, None
                self.game.close(env)


__all__ = ["GRID_HISTORY_H", "GRID_HISTORY_W", "Engine"]