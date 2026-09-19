"""Episode archive for the lab history browser and emulator replay."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Decision:
    index: int
    timestamp: str
    input_frame: int
    action: str
    label: str
    buttons: list
    frames_executed: int
    reward: float
    latency_ms: float | None
    scores: list
    scene: str
    facts: list
    done: bool
    completed: bool
    image: str
    model: str | None = None
    usage: dict | None = None
    # fly-brain snapshot: which commands the fly's descending neurons fired,
    # what senses it was driven with, and how it resolved the action.
    brain: dict | None = None

    def public(self, include_image: bool = False) -> dict:
        data = asdict(self)
        if not include_image:
            data.pop("image", None)
        return data

    def replay_row(self, include_image: bool = True) -> dict:
        """Everything a gapless replay needs.

        `wall_ms` is how long this decision actually occupied the lab (the time
        the fly spent thinking, plus the frames it then acted for); `game_ms` is
        only the part the emulator moved. Playing on `game_ms` collapses the
        thinking gaps. The frame thumbnail stays by default -- a replay without
        pictures is just a spreadsheet -- but `facts` and `scores` are fetched
        per-step instead.
        """
        data = asdict(self)
        if not include_image:
            data.pop("image", None)
        data.pop("facts", None)
        data.pop("scores", None)
        data.pop("scene", None)
        thinking_ms = float(self.latency_ms or 0.0)
        data["thinking_ms"] = thinking_ms
        data["game_ms"] = self.frames_executed / 60.0 * 1000.0
        data["wall_ms"] = thinking_ms + data["game_ms"]
        return data


@dataclass
class Episode:
    id: str
    started_at: str
    game: str
    game_title: str
    short_name: str
    world: str
    seed: int
    policy: str
    hold_frames: int
    start_image: str
    status: str = "open"
    decisions: list[Decision] = field(default_factory=list)

    def summary(self) -> dict:
        last = self.decisions[-1] if self.decisions else None
        frames = 0
        if last is not None:
            frames = last.input_frame + last.frames_executed
        return {
            "id": self.id,
            "started_at": self.started_at,
            "game": self.game,
            "game_title": self.game_title,
            "short_name": self.short_name,
            "world": self.world,
            "seed": self.seed,
            "policy": self.policy,
            "hold_frames": self.hold_frames,
            "status": self.status,
            "decisions": len(self.decisions),
            "frames": frames,
            "last_action": last.action if last else None,
            "ended": bool(last.done) if last else False,
            "completed": bool(last.completed) if last else False,
            "thinking_ms": round(sum(float(d.latency_ms or 0.0) for d in self.decisions), 1),
            "game_ms": round(sum(d.frames_executed for d in self.decisions) / 60 * 1000.0, 1),
        }

    def public(self, include_images: bool = False) -> dict:
        data = self.summary()
        data["start_image"] = self.start_image
        data["steps"] = [step.public(include_image=include_images) for step in self.decisions]
        return data

    def replay(self, include_images: bool = True) -> dict:
        """The episode as a gapless-playable track: one row per decision plus the
        timings needed to play it with or without the thinking gaps."""
        data = self.summary()
        data["start_image"] = self.start_image
        data["thinking_ms"] = round(sum(float(d.latency_ms or 0.0) for d in self.decisions), 1)
        data["game_ms"] = round(sum(d.frames_executed for d in self.decisions) / 60 * 1000.0, 1)
        data["steps"] = [step.replay_row(include_images) for step in self.decisions]
        return data


class HistoryArchive:
    def __init__(self, max_episodes: int = 24, max_decisions: int = 800):
        self.max_episodes = max_episodes
        self.max_decisions = max_decisions
        self.episodes: deque[Episode] = deque()
        self.current: Episode | None = None
        self._seq = 0

    def start(self, meta: dict, start_image: str) -> Episode:
        if self.current is not None and not self.current.decisions:
            try:
                self.episodes.remove(self.current)
            except ValueError:
                pass
        elif self.current is not None and self.current.status == "open":
            self.current.status = "closed"
        self._seq += 1
        episode = Episode(
            id=f"ep-{self._seq}",
            started_at=_now(),
            game=meta["game"],
            game_title=meta["game_title"],
            short_name=meta["short_name"],
            world=meta.get("world") or "",
            seed=int(meta["seed"]),
            policy=meta["policy"],
            hold_frames=int(meta["hold_frames"]),
            start_image=start_image,
        )
        self.episodes.append(episode)
        self.current = episode
        while len(self.episodes) > self.max_episodes:
            dropped = self.episodes.popleft()
            if dropped is self.current:
                self.current = episode
        return episode

    def record(self, payload: dict) -> Decision | None:
        if self.current is None:
            return None
        if len(self.current.decisions) >= self.max_decisions:
            return None
        decision = Decision(index=len(self.current.decisions), timestamp=_now(), **payload)
        self.current.decisions.append(decision)
        if decision.done:
            self.current.status = "ended"
        return decision

    def get(self, episode_id: str) -> Episode:
        for episode in self.episodes:
            if episode.id == episode_id:
                return episode
        raise KeyError(episode_id)

    def summaries(self) -> list[dict]:
        return [episode.summary() for episode in reversed(self.episodes)]

    def log_rows(self) -> int:
        return sum(len(episode.decisions) for episode in self.episodes)

    def export(self) -> str:
        lines = []
        for episode in self.episodes:
            header = episode.summary()
            header["type"] = "episode"
            header.pop("last_action", None)
            lines.append(header)
            for step in episode.decisions:
                row = step.public(include_image=False)
                row["type"] = "decision"
                row["episode"] = episode.id
                row["game"] = episode.game
                row["policy"] = episode.policy
                row["seed"] = episode.seed
                lines.append(row)
        return "\n".join(json.dumps(row) for row in lines) + ("\n" if lines else "")
