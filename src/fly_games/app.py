"""The fly observatory server.

Three things live behind this API, and they are deliberately different shapes:

  /api/brain, /api/brain/points.bin
      The connectome. Static, immutable, fetched once: anatomy, superclass
      palette, the sensory channels, the twelve descending command groups, and
      a packed binary point cloud of every soma position.

  /api/state, /ws
      The live episode. Snapshots are pushed only when something actually
      changed -- which, at roughly one fly decision per second, is the pace the
      UI is built around.

  /api/probe
      The bench. Stimulate cell types directly, no emulator involved.

  /api/episodes, /api/episodes/{id}/replay
      What already happened, including the timings that let the UI replay an
      episode with the ~0.9 s thinking gaps collapsed out.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fly_games.brain import COMMAND_GROUPS, shared_client
from fly_games.catalog import GAMES
from fly_games.engine import Engine
from fly_games.probe import ProbeBench
from fly_games.readouts import readout_path

PUSH_INTERVAL_S = 0.08


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def allow_origin(origin: str | None, host: str | None) -> bool:
    """Allow the vite dev server and localhost; nothing else."""
    if not origin:
        return True
    hostname = (urlparse(origin).hostname or "").lower()
    if hostname in {"127.0.0.1", "localhost", "::1", "testserver"}:
        return True
    return bool(host and origin.rstrip("/") in {f"http://{host}", f"https://{host}"})


class Command(BaseModel):
    action: Literal["run", "pause", "step", "reset", "configure", "select", "viz"]
    game: str | None = None
    policy: Literal["fly", "fly-hand", "scripted"] | None = None
    frames: int | None = None
    seed: int | None = None
    on: bool | None = None


class Stimulus(BaseModel):
    # At least one cell type: a stimulus that stimulates nothing is a silent
    # request wearing a stimulus's clothes. "Resting state" is `events: []`.
    types: list[str] = Field(min_length=1)
    side: Literal["L", "R"] | None = None
    volts: float = 0.5


class ProbeRequest(BaseModel):
    # Pydantic deep-copies model defaults, so these mutable defaults are safe.
    events: list[Stimulus] = []
    steps: int | None = None
    games: list[str] = []
    field: bool = True


class WebSocketGuard:
    """Close websocket handshakes for paths the lab does not serve.

    The observatory is mounted at "/", so a websocket aimed at `/` or
    `/api/state` would otherwise fall through to StaticFiles, which asserts on
    the scope type -- turning a stray connection into a 500 and a traceback in
    the server log instead of a clean refusal.
    """

    def __init__(self, app, allowed: tuple[str, ...] = ("/ws",)) -> None:
        self.app = app
        self.allowed = allowed

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "websocket" and scope.get("path") not in self.allowed:
            await send({"type": "websocket.close", "code": 1008,
                        "reason": "the lab websocket lives at /ws"})
            return
        await self.app(scope, receive, send)


def create_app(engine_factory=None, game_id: str = "mario") -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine_factory() if engine_factory else Engine(game_id=game_id)
        app.state.probe = ProbeBench()
        app.state.clients = 0
        app.state.brain_card = None
        app.state.points = None
        yield
        app.state.engine.close()

    app = FastAPI(title="fly-games observatory", version="0.2.0", lifespan=lifespan)
    app.add_middleware(WebSocketGuard)

    def engine() -> Engine:
        return app.state.engine

    # ---------------------------------------------------------------- anatomy
    @app.get("/api/brain")
    def brain():
        """Everything needed to draw the connectome. Cached; it never changes."""
        if app.state.brain_card is None:
            app.state.brain_card = shared_client().brain_card()
        return app.state.brain_card

    @app.get("/api/brain/points.bin")
    def brain_points():
        """Packed soma point cloud. Immutable, and addressed by content revision.

        The card hands out this URL with `?rev=` already filled in, so a long
        max-age is safe: change the projection or the encoding and the URL the
        card advertises changes with it.
        """
        if app.state.points is None:
            app.state.points = shared_client().points_blob()
        return Response(
            content=app.state.points,
            media_type="application/octet-stream",
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "ETag": f'"{shared_client().cloud_rev()}"',
            },
        )

    @app.get("/api/brain/types")
    def brain_types(q: str = Query("", max_length=48), limit: int = Query(40, ge=1, le=200)):
        return {"types": app.state.probe.search(q, limit)}

    # ------------------------------------------------------------- live state
    @app.get("/api/state")
    def state(viz: bool = False):
        """The live snapshot. `viz=true` also ships the per-neuron spike sample and
        the soma heat-map (about 20 kB); the websocket turns them on by itself."""
        if viz:
            engine().viz = True
        return engine().snapshot()

    @app.get("/api/games")
    def games():
        return {"games": [g.catalog_entry() for g in GAMES.values()]}

    @app.get("/api/command")
    def command_help():
        return {"actions": ["run", "pause", "step", "reset", "configure", "select", "viz"],
                "policies": ["fly", "fly-hand", "scripted"]}

    @app.post("/api/command")
    def command(body: Command):
        payload = body.model_dump(exclude_none=True)
        name = payload.pop("action")
        try:
            engine().command(name, **payload)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return engine().snapshot()

    # ------------------------------------------------------------- readouts
    @app.get("/api/readouts")
    def readouts():
        """Per-game readout status: trained or still running on the zero-shot rule."""
        from fly_games.brain import FlyReadout

        out = {}
        for gid, game in GAMES.items():
            path = readout_path(gid)
            coarse = game.fly_coarse_actions()
            try:
                readout = (FlyReadout.load(coarse, path) if path.exists()
                           else FlyReadout(coarse, path))
            except ValueError as exc:            # trained against different labels
                out[gid] = {"trained": False, "error": str(exc)}
                continue
            out[gid] = {
                "trained": readout.trained,
                "coarse_actions": list(readout.coarse_actions),
                "features": list(COMMAND_GROUPS),
                "info": readout.info.public() if readout.info else None,
                "weights": readout.weights(),
            }
        return out

    # ------------------------------------------------------------- replay
    @app.get("/api/episodes")
    def episodes():
        return {"episodes": engine().history_summaries()}

    @app.get("/api/episodes/{episode_id}")
    def episode(episode_id: str, images: bool = False):
        try:
            return engine().history_episode(episode_id, include_images=images)
        except KeyError:
            raise HTTPException(404, f"Unknown episode {episode_id!r}.") from None

    @app.get("/api/episodes/{episode_id}/replay")
    def episode_replay(episode_id: str, images: bool = True):
        """The decision track without the bulky payloads, plus the two timings
        (`thinking_ms`, `game_ms`) needed to play it with or without the gaps."""
        try:
            episode = engine().archive.get(episode_id)
        except KeyError:
            raise HTTPException(404, f"Unknown episode {episode_id!r}.") from None
        return episode.replay(include_images=images)

    @app.get("/api/episodes/{episode_id}/steps/{index}")
    def episode_step(episode_id: str, index: int):
        try:
            return engine().history_step(episode_id, index)
        except KeyError:
            raise HTTPException(404, f"Unknown episode {episode_id!r}.") from None
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/log")
    def log():
        return Response(
            engine().archive.export(),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": 'attachment; filename="fly-games.jsonl"'},
        )

    # ------------------------------------------------------------- the bench
    @app.get("/api/probe")
    def probe_menu():
        return app.state.probe.menu()

    @app.post("/api/probe")
    async def probe_run(body: ProbeRequest):
        """Stimulate the connectome and report what its output neurons did.

        Refused while an episode is actually running: there is one fly, and it
        cannot be playing Mario and being probed at the same time.
        """
        eng = engine()
        if eng.running or eng.busy:
            raise HTTPException(409, "Pause the episode first - the fly can only do one thing at a time.")
        games = [g for g in body.games if g in GAMES]
        result = await asyncio.to_thread(
            app.state.probe.run,
            [e.model_dump() for e in body.events], body.steps, None, games,
        )
        return result.public(include_field=body.field)

    # ------------------------------------------------------------- websocket
    @app.websocket("/ws")
    async def live(ws: WebSocket):
        """Push a snapshot whenever the engine's revision changes, and take
        commands on the same socket.

        Two independent tasks: cancelling a pending `receive()` is not safe in
        Starlette -- it can swallow the message and wedge the connection -- so
        the receiver simply awaits, and only the pusher ever sends.
        """
        if not allow_origin(ws.headers.get("origin"), ws.headers.get("host")):
            await ws.close(code=1008)
            return
        await ws.accept()
        app.state.clients += 1
        eng = engine()
        eng.viz = True

        async def pusher():
            revision = -1
            while True:
                current = eng.snapshot()
                if current["revision"] != revision:
                    revision = current["revision"]
                    await ws.send_text(json.dumps(current))
                await asyncio.sleep(PUSH_INTERVAL_S)

        async def receiver():
            while True:
                raw = await ws.receive_text()
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_text(json.dumps({"error": "expected one JSON object per message"}))
                    continue
                if not isinstance(message, dict):
                    await ws.send_text(json.dumps({"error": "expected a JSON object"}))
                    continue
                name = message.pop("action", None) or message.pop("command", None)
                if not name:
                    await ws.send_text(json.dumps({"error": "message needs an 'action'"}))
                    continue
                message.pop("type", None)
                try:
                    eng.command(name, **message)
                except (TypeError, ValueError) as exc:
                    await ws.send_text(json.dumps({"error": str(exc)}))

        tasks = [asyncio.create_task(pusher()), asyncio.create_task(receiver())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            app.state.clients = max(0, app.state.clients - 1)
            if app.state.clients == 0:
                eng.viz = False

    dist = project_root() / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="observatory")
    return app


app = create_app()


def main(host: str = "127.0.0.1", port: int = 8000) -> int:
    import uvicorn

    uvicorn.run("fly_games.app:app", host=host, port=port, workers=1)
    return 0


__all__ = ["PUSH_INTERVAL_S", "app", "create_app", "main", "project_root"]