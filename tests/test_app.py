"""The HTTP surface, against a fake engine.

The engine owns threads and an emulator; the routes own shapes, status codes and
a few decisions that are easy to get wrong (which timings go to the replay
track, when a probe must be refused, what a stray websocket gets).
"""

from __future__ import annotations

import pytest

# --- live state ---------------------------------------------------------------

def test_state_returns_the_current_snapshot(client):
    body = client.get("/api/state").json()
    assert body["phase"] == "paused"
    assert body["game"]["id"] == "mario"


def test_state_with_viz_turns_the_heat_map_on(client, fake_engine):
    client.get("/api/state?viz=true")
    assert fake_engine.viz is True


def test_games_lists_the_bundled_titles(client):
    games = client.get("/api/games").json()["games"]
    assert {g["id"] for g in games} == {"mario", "kungfu", "doom"}
    assert all({"title", "platform_label", "cartridge"} <= set(g) for g in games)


def test_command_help_advertises_the_protocol(client):
    body = client.get("/api/command").json()
    assert set(body["actions"]) == {"run", "pause", "step", "reset", "configure", "select", "viz"}
    assert set(body["policies"]) == {"fly", "fly-hand", "scripted"}


def test_run_and_pause_drive_the_engine(client, fake_engine):
    client.post("/api/command", json={"action": "run"})
    assert fake_engine.running is True
    client.post("/api/command", json={"action": "pause"})
    assert fake_engine.running is False


def test_a_command_returns_the_new_snapshot(client):
    body = client.post("/api/command", json={"action": "run"}).json()
    assert body["running"] is True
    assert body["phase"] == "acting"


def test_an_unsupported_hold_duration_is_a_conflict_not_a_crash(client):
    response = client.post("/api/command", json={"action": "configure", "frames": 7})
    assert response.status_code == 409
    assert "hold" in response.json()["detail"].lower()


def test_stepping_while_running_is_refused(client, fake_engine):
    client.post("/api/command", json={"action": "run"})
    response = client.post("/api/command", json={"action": "step"})
    assert response.status_code == 409


def test_an_unknown_action_is_rejected_by_validation(client):
    assert client.post("/api/command", json={"action": "self_destruct"}).status_code == 422


def test_an_unknown_policy_is_rejected_by_validation(client):
    assert client.post("/api/command", json={"action": "configure", "policy": "gpt"}).status_code == 422


# --- episodes and replay ------------------------------------------------------

def test_episodes_are_summarised_newest_first(client, fake_engine):
    fake_engine.add_decision()
    body = client.get("/api/episodes").json()["episodes"]
    assert body[0]["id"] == "ep-1"
    assert body[0]["decisions"] >= 1


def test_an_unknown_episode_is_404(client):
    assert client.get("/api/episodes/ep-999").status_code == 404
    assert client.get("/api/episodes/ep-999/replay").status_code == 404
    assert client.get("/api/episodes/ep-999/steps/0").status_code == 404


def test_an_out_of_range_step_is_404(client, fake_engine):
    fake_engine.add_decision()
    assert client.get("/api/episodes/ep-1/steps/99").status_code == 404


def test_episode_details_omit_images_unless_asked(client, fake_engine):
    fake_engine.add_decision()
    assert "image" not in client.get("/api/episodes/ep-1").json()["steps"][0]
    assert "image" in client.get("/api/episodes/ep-1?images=true").json()["steps"][0]


def test_the_replay_track_carries_both_timings(client, fake_engine):
    """Gapless playback needs `game_ms` (the emulator moved) and `wall_ms`
    (thinking included). Without both, the UI cannot collapse the gaps."""
    fake_engine.add_decision(frames_executed=6, latency_ms=850.0)
    steps = client.get("/api/episodes/ep-1/replay").json()["steps"]
    assert steps[0]["game_ms"] == pytest.approx(100.0)
    assert steps[0]["wall_ms"] == pytest.approx(950.0)
    assert steps[0]["thinking_ms"] == pytest.approx(850.0)


def test_the_replay_track_keeps_its_pictures(client, fake_engine):
    fake_engine.add_decision()
    steps = client.get("/api/episodes/ep-1/replay").json()["steps"]
    assert steps[0]["image"].startswith("data:")


def test_the_log_export_is_ndjson(client, fake_engine):
    fake_engine.add_decision()
    response = client.get("/api/log")
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert "attachment" in response.headers["content-disposition"]
    assert response.text.count("\n") >= 2


# --- readouts -----------------------------------------------------------------

def test_readout_status_names_the_features_and_the_decisions(client):
    body = client.get("/api/readouts").json()
    assert set(body) == {"mario", "kungfu", "doom"}
    for entry in body.values():
        assert isinstance(entry["trained"], bool)
        assert entry["coarse_actions"]
        assert len(entry["features"]) == 12


# --- the bench ----------------------------------------------------------------

def test_the_probe_menu_describes_the_channels(client):
    pytest.importorskip("flybrain")
    from flybrain import has_data

    if not has_data():
        pytest.skip("connectome data missing")
    menu = client.get("/api/probe").json()
    assert {c["name"] for c in menu["channels"]} == {"loom", "threat", "shot", "chase"}
    assert menu["max_volts"] == 0.8


def test_a_probe_is_refused_while_the_fly_is_playing(client, fake_engine):
    client.post("/api/command", json={"action": "run"})
    response = client.post("/api/probe", json={"events": [{"types": ["LC4"]}]})
    assert response.status_code == 409
    assert "pause" in response.json()["detail"].lower()


def test_a_stimulus_with_no_cell_types_is_rejected(client):
    assert client.post("/api/probe", json={"events": [{"types": []}]}).status_code == 422


def test_a_side_that_is_not_left_or_right_is_rejected(client):
    assert client.post("/api/probe", json={"events": [{"types": ["LC4"], "side": "up"}]}).status_code == 422


# --- the websocket -----------------------------------------------------------
# Exercised against a real ASGI server in test_websocket.py: this version of
# Starlette's TestClient re-raises the client's own disconnect on close, so it
# cannot distinguish a clean shutdown from a broken one.

# --- lifecycle ----------------------------------------------------------------

def test_the_engine_is_closed_when_the_app_shuts_down(fake_engine):
    from fastapi.testclient import TestClient

    from fly_games.app import create_app

    app = create_app(engine_factory=lambda: fake_engine)
    with TestClient(app):
        assert fake_engine.closed is False
    assert fake_engine.closed is True


def test_the_point_cloud_url_is_content_addressed():
    """The blob is served `immutable`; if the URL did not change with the
    encoding, returning visitors would keep a stale cloud forever."""
    pytest.importorskip("flybrain")
    from flybrain import has_data

    if not has_data():
        pytest.skip("connectome data missing")
    from fly_games.brain import shared_client

    url = shared_client().brain_card()["points_url"]
    assert url.startswith("/api/brain/points.bin?rev=")
    assert len(url.rsplit("rev=", 1)[1]) == 12