"""Shared fixtures.

Two rules shape this suite:

* Nothing here may need the 260 MB connectome or a commercial ROM unless the
  test says so. `needs_brain` / `needs_rom` mark the exceptions, so a fresh
  checkout runs green in seconds.
* The API tests use `FakeEngine`. The engine is the one object with real
  background threads and a live emulator behind it; testing the routes against
  it would make the suite slow and flaky without testing the routes any better.
"""

from __future__ import annotations

import numpy as np
import pytest

from fly_games.history import HistoryArchive

try:  # pragma: no cover - environment dependent
    from flybrain import has_data

    BRAIN_READY = bool(has_data())
except (ImportError, RuntimeError, OSError):  # pragma: no cover - flybrain missing
    BRAIN_READY = False

needs_brain = pytest.mark.skipif(
    not BRAIN_READY,
    reason="fly connectome data missing - run `fly-games brain-download`",
)


def rom_available(game_id: str) -> bool:
    """Whether `game_id` can open its emulator right now.

    Checks the cartridge lookup only. Booting a real emulator to find out would
    cost seconds per test and would not tell us anything extra.
    """
    try:
        if game_id == "mario":
            import nes_py  # noqa: F401  (Super Mario Bros. ships inside the gym package)

            return True
        if game_id == "kungfu":
            from fly_games.assets import find_kungfu_rom

            find_kungfu_rom()
            return True
        if game_id == "doom":
            from fly_games.assets import find_doom_iwad

            find_doom_iwad()
            return True
    except Exception:  # noqa: BLE001 - a missing cartridge is not a test failure
        return False
    return False


@pytest.fixture()
def needs_rom():
    """Use as `needs_rom("mario")` at the top of a test that drives an emulator."""

    def _require(game_id: str) -> None:
        if not rom_available(game_id):
            pytest.skip(f"{game_id} needs its ROM/IWAD - see roms/README.md")

    return _require


class FakeEngine:
    """The slice of `Engine` that the HTTP API touches.

    Deliberately keeps a real `HistoryArchive`, because the episode and replay
    routes are about that archive's shape, not about the engine.
    """

    def __init__(self) -> None:
        self.archive = HistoryArchive(max_episodes=5, max_decisions=50)
        self.running = False
        self.busy = False
        self.viz = False
        self.closed = False
        self.commands: list[tuple[str, dict]] = []
        # The websocket pusher only sends when the revision moves, so a fake that
        # never moves it can only ever deliver the first snapshot.
        self.revision = 0
        self.snapshot_payload = {
            "revision": 0,
            "phase": "paused",
            "game": {"id": "mario", "title": "Super Mario Bros.", "cartridge": "Super Mario Bros. (USA)"},
            "policy": "fly",
            "hold_frames": 4,
            "running": False,
            "busy": False,
            "done": False,
            "decisions": 0,
            "frames": 0,
            "log": [],
        }
        self.start_episode()

    # --- episodes ------------------------------------------------------------
    def start_episode(self) -> None:
        self.archive.start(
            {"game": "mario", "game_title": "Super Mario Bros.", "short_name": "Mario",
             "world": "1-1", "seed": 123, "policy": "fly", "hold_frames": 4},
            start_image="data:image/jpeg;base64,AAAA",
        )

    def add_decision(self, **overrides) -> None:
        payload = {
            "input_frame": 0,
            "action": "right_run",
            "label": "right + run",
            "buttons": ["right", "run"],
            "frames_executed": 4,
            "reward": 0.5,
            "latency_ms": 900.0,
            "scores": [],
            "scene": "flat",
            "facts": [["Mario", "x=100"]],
            "done": False,
            "completed": False,
            "image": "data:image/jpeg;base64,BBBB",
            "model": None,
            "usage": None,
            "brain": {"command": {"escape_L": {"rate": 1.0}}, "input": {"looms_L": 0.4}},
        }
        payload.update(overrides)
        self.archive.record(payload)

    # --- engine surface ------------------------------------------------------
    def snapshot(self) -> dict:
        self.snapshot_payload["revision"] = self.revision
        self.snapshot_payload["running"] = self.running
        self.snapshot_payload["busy"] = self.busy
        self.snapshot_payload["phase"] = "acting" if self.running else "paused"
        self.snapshot_payload["decisions"] = len(self.archive.current.decisions) if self.archive.current else 0
        return dict(self.snapshot_payload)

    def command(self, name: str, **options) -> None:
        self.commands.append((name, options))
        self.revision += 1
        if name in {"run", "step"}:
            if name == "step" and self.running:
                raise ValueError("Pause and wait for the current decision before stepping.")
            self.running = name == "run"
        elif name == "pause":
            self.running = False
        elif name == "configure":
            frames = options.get("frames")
            if frames is not None and frames not in (4, 8, 12, 16):
                raise ValueError("Unsupported hold duration.")
            policy = options.get("policy")
            if policy is not None:
                if policy not in {"fly", "fly-hand", "scripted"}:
                    raise ValueError(f"Unknown policy {policy!r}.")
                self.policy = policy
                self.snapshot_payload["policy"] = policy
        elif name == "viz":
            self.viz = bool(options.get("on", True))
        else:
            raise ValueError(f"Unknown command: {name!r}.")

    def history_summaries(self) -> list[dict]:
        return self.archive.summaries()

    def history_episode(self, episode_id: str, include_images: bool = False) -> dict:
        return self.archive.get(episode_id).public(include_images=include_images)

    def history_step(self, episode_id: str, index: int) -> dict:
        episode = self.archive.get(episode_id)
        if index < 0 or index >= len(episode.decisions):
            raise ValueError(f"Step {index} out of range.")
        return episode.decisions[index].public(include_image=True)

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture()
def client(fake_engine: FakeEngine):
    """A TestClient for the real app with a fake engine and no emulator."""
    from fastapi.testclient import TestClient

    from fly_games.app import create_app

    app = create_app(engine_factory=lambda: fake_engine, game_id="mario")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def synthetic_positions():
    """A fake nervous system with a known anatomy.

    axis 0 is lateral (the optic lobes straddle it), axis 1 is dorsal-ventral,
    axis 2 is head-to-tail. Neurons with no soma are NaN, like the real data.
    """
    rng = np.random.default_rng(7)
    labels: list[str] = []
    rows: list[tuple[float, float, float]] = []

    def cloud(name: str, n: int, centre: tuple[float, float, float], spread: tuple[float, float, float]) -> None:
        for _ in range(n):
            rows.append(tuple(rng.normal(centre, spread)))  # type: ignore[arg-type]
            labels.append(name)

    cloud("cb_intrinsic", 400, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))       # central brain
    cloud("ol_intrinsic", 800, (0.0, 0.2, 0.1), (6.0, 1.0, 1.0))       # optic lobes: wide in x only
    cloud("vnc_intrinsic", 300, (0.1, 0.5, 30.0), (0.8, 1.0, 4.0))     # nerve cord: far along z
    positions = np.asarray(rows, dtype=np.float32)
    positions[rng.choice(positions.shape[0], size=50, replace=False)] = np.nan
    return positions, np.asarray(labels)

class StubBrain:
    """Just enough of a flybrain to build a `SensoryEncoder`.

    Every channel/side resolves to a handful of fake neuron indices, so encoder
    tests can assert on *which channels and sides an observation drives* without
    loading 166,700 neurons.
    """

    def cells(self, types, side="any"):
        return np.arange(3, dtype=np.int64)

    @property
    def n(self) -> int:
        return 9


class StubClient:
    """A `FlyBrainClient` stand-in whose brain has indexable but fake cells."""

    def __init__(self) -> None:
        self.brain = StubBrain()


@pytest.fixture()
def stub_client() -> StubClient:
    return StubClient()


@pytest.fixture(scope="session")
def lab_url():
    """The real app on a real ASGI server, for websocket tests only.

    Starlette's `TestClient` websocket harness re-raises the client's own
    disconnect out of `__exit__` in this version — even for a two-line app — so
    it cannot tell a clean shutdown from a broken one. The socket tests need the
    semantics the browser actually gets.
    """
    import socket
    import threading
    import time

    import uvicorn

    from fly_games.app import create_app

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(
        create_app(engine_factory=FakeEngine, game_id="mario"),
        host="127.0.0.1", port=port, log_level="warning", lifespan="on",
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(300):
        if server.started:
            break
        time.sleep(0.05)
    else:  # pragma: no cover - only reachable if uvicorn fails to bind
        pytest.fail("the lab server never started")

    yield f"ws://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=15)
