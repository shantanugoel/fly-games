"""The live socket, against a real ASGI server.

`/ws` is the only route with two concurrent tasks, and it is the one the browser
spends all its time on. Starlette's TestClient cannot exercise it honestly (see
the `lab_url` fixture), so these run against uvicorn on an ephemeral port.
"""

from __future__ import annotations

import json

import pytest

websockets = pytest.importorskip("websockets")
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect


def recv(ws, timeout: float = 5.0) -> dict:
    raw = ws.recv(timeout=timeout)
    assert isinstance(raw, str)
    return json.loads(raw)


def recv_where(ws, predicate, timeout: float = 8.0, limit: int = 40) -> dict:
    """Snapshots are pushed on every revision change, so a command's reply may be
    preceded by others. Take the first one that satisfies the predicate."""
    for _ in range(limit):
        message = recv(ws, timeout=timeout)
        if predicate(message):
            return message
    raise AssertionError("no matching snapshot arrived")


def test_the_socket_opens_with_a_snapshot(lab_url):
    with connect(lab_url + "/ws") as ws:
        state = recv(ws)
        assert state["phase"] in {"paused", "acting", "thinking"}
        assert state["game"]["id"] == "mario"
        assert "revision" in state


def test_the_socket_pushes_a_snapshot_per_change(lab_url):
    with connect(lab_url + "/ws") as ws:
        first = recv(ws)
        ws.send(json.dumps({"action": "run"}))
        acting = recv_where(ws, lambda m: m.get("running") is True)
        assert acting["phase"] == "acting"
        assert acting["revision"] != first["revision"]
        ws.send(json.dumps({"action": "pause"}))
        assert recv_where(ws, lambda m: m.get("running") is False)["phase"] == "paused"


def test_commands_and_replies_share_the_socket(lab_url):
    with connect(lab_url + "/ws") as ws:
        recv(ws)
        ws.send(json.dumps({"action": "configure", "policy": "scripted"}))
        assert recv_where(ws, lambda m: m.get("policy") == "scripted")["policy"] == "scripted"


def test_a_command_on_the_socket_reports_errors_without_dropping_the_connection(lab_url):
    """A typo in the UI must not sever the live feed."""
    with connect(lab_url + "/ws") as ws:
        recv(ws)
        ws.send(json.dumps({"action": "configure", "frames": 7}))
        assert recv_where(ws, lambda m: "error" in m)["error"]
        ws.send(json.dumps({"action": "pause"}))
        assert recv_where(ws, lambda m: "phase" in m)


def test_a_message_that_is_not_json_is_refused_in_place(lab_url):
    with connect(lab_url + "/ws") as ws:
        recv(ws)
        ws.send("not json at all")
        assert "json" in recv_where(ws, lambda m: "error" in m)["error"].lower()
        ws.send(json.dumps({"action": "pause"}))
        assert recv_where(ws, lambda m: "phase" in m)


def test_a_message_with_no_action_is_refused_in_place(lab_url):
    with connect(lab_url + "/ws") as ws:
        recv(ws)
        ws.send(json.dumps({"policy": "fly"}))
        assert "action" in recv_where(ws, lambda m: "error" in m)["error"]


def test_a_socket_aimed_at_any_other_path_is_refused_cleanly(lab_url):
    """The observatory is mounted at `/`, so a stray websocket would otherwise
    fall through to StaticFiles, which asserts on the scope type: a 500 and a
    traceback instead of a refusal."""
    for path in ("/", "/api/state", "/wsx"):
        with pytest.raises(InvalidStatus) as excinfo, connect(lab_url + path):
            pass
        assert excinfo.value.response.status_code == 403, path


def test_a_second_client_still_gets_snapshots_after_the_first_leaves(lab_url):
    """The pusher must not wedge when a tab goes away mid-stream."""
    with connect(lab_url + "/ws") as first:
        recv(first)
    with connect(lab_url + "/ws") as second:
        assert "phase" in recv(second)