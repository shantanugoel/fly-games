"""The episode archive, and the two timings gapless replay is built on."""

from __future__ import annotations

import json

import pytest

from fly_games.history import Decision, Episode, HistoryArchive

META = {"game": "mario", "game_title": "Super Mario Bros.", "short_name": "Mario",
        "world": "1-1", "seed": 123, "policy": "fly", "hold_frames": 4}


def payload(**overrides) -> dict:
    base = {"input_frame": 0, "action": "right_run", "label": "right + run",
            "buttons": ["right", "B"], "frames_executed": 4, "reward": 0.0,
            "latency_ms": 900.0, "scores": [], "scene": "flat", "facts": [],
            "done": False, "completed": False, "image": "data:,X"}
    base.update(overrides)
    return base


@pytest.fixture()
def archive() -> HistoryArchive:
    return HistoryArchive(max_episodes=3, max_decisions=4)


def test_start_then_record_appends_to_the_current_episode(archive):
    episode = archive.start(META, "data:,start")
    assert archive.record(payload()) is not None
    assert episode.decisions[0].index == 0
    assert archive.log_rows() == 1


def test_an_episode_with_no_decisions_is_not_kept(archive):
    first = archive.start(META, "data:,a")
    archive.start(META, "data:,b")
    assert first not in archive.episodes
    assert len(archive.episodes) == 1


def test_restarting_closes_the_previous_episode(archive):
    first = archive.start(META, "data:,a")
    archive.record(payload())
    archive.start(META, "data:,b")
    assert first.status == "closed"


def test_a_finishing_decision_ends_the_episode(archive):
    episode = archive.start(META, "data:,a")
    archive.record(payload(done=True, completed=True))
    assert episode.status == "ended"
    assert episode.summary()["ended"] is True
    assert episode.summary()["completed"] is True


def test_record_is_ignored_without_an_episode(archive):
    assert archive.record(payload()) is None


def test_decision_and_episode_caps_are_enforced(archive):
    archive.start(META, "data:,a")
    for _ in range(10):
        archive.record(payload())
    assert len(archive.episodes[0].decisions) == 4

    for _ in range(5):
        archive.start(META, "data:,x")
        archive.record(payload())
    assert len(archive.episodes) == 3


def test_summaries_are_newest_first(archive):
    archive.start(META, "data:,a")
    archive.record(payload())
    archive.start(META, "data:,b")
    archive.record(payload())
    assert [row["id"] for row in archive.summaries()] == ["ep-2", "ep-1"]


def test_get_raises_keyerror_for_unknown_episodes(archive):
    with pytest.raises(KeyError):
        archive.get("ep-99")


# --- the replay timings -------------------------------------------------------

def test_replay_row_separates_thinking_from_acting():
    decision = Decision(index=0, timestamp="now", **payload(frames_executed=12, latency_ms=900.0))
    row = decision.replay_row(include_image=False)
    # 12 emulator frames at 60 Hz is 200 ms of world time; the other 900 ms was
    # the fly thinking. Playing on game_ms collapses that gap.
    assert row["game_ms"] == pytest.approx(200.0)
    assert row["thinking_ms"] == pytest.approx(900.0)
    assert row["wall_ms"] == pytest.approx(1100.0)


def test_replay_row_keeps_the_picture_and_drops_the_bulk():
    row = Decision(index=0, timestamp="now", **payload()).replay_row()
    assert row["image"].startswith("data:")
    assert "facts" not in row and "scores" not in row and "scene" not in row


def test_replay_row_without_images_is_the_cheap_variant():
    row = Decision(index=0, timestamp="now", **payload()).replay_row(include_image=False)
    assert "image" not in row


def test_a_missing_latency_is_treated_as_no_thinking():
    row = Decision(index=0, timestamp="now", **payload(latency_ms=None)).replay_row()
    assert row["thinking_ms"] == 0.0
    assert row["wall_ms"] == pytest.approx(row["game_ms"])


def test_public_hides_the_image_unless_asked():
    decision = Decision(index=0, timestamp="now", **payload())
    assert "image" not in decision.public()
    assert decision.public(include_image=True)["image"] == "data:,X"


def test_episode_totals_add_up():
    episode = Episode(id="ep-1", started_at="now", game="mario", game_title="Mario",
                      short_name="Mario", world="1-1", seed=1, policy="fly",
                      hold_frames=4, start_image="data:,")
    for frames, latency in ((4, 900.0), (8, 1100.0)):
        episode.decisions.append(Decision(index=len(episode.decisions), timestamp="now",
                                          **payload(frames_executed=frames, latency_ms=latency)))
    summary = episode.summary()
    assert summary["thinking_ms"] == pytest.approx(2000.0)
    assert summary["game_ms"] == pytest.approx((4 + 8) / 60 * 1000.0)
    assert summary["decisions"] == 2


def test_replay_track_is_playable_without_the_gaps():
    episode = Episode(id="ep-1", started_at="now", game="mario", game_title="Mario",
                      short_name="Mario", world="1-1", seed=1, policy="fly",
                      hold_frames=4, start_image="data:,")
    for _ in range(3):
        episode.decisions.append(Decision(index=len(episode.decisions), timestamp="now",
                                          **payload(frames_executed=6, latency_ms=800.0)))
    track = episode.replay()
    assert len(track["steps"]) == 3
    assert all(step["game_ms"] < step["wall_ms"] for step in track["steps"])
    assert track["start_image"] == "data:,"


def test_export_is_one_valid_json_object_per_line(archive):
    archive.start(META, "data:,a")
    archive.record(payload())
    archive.record(payload(action="jump", done=True))
    lines = [line for line in archive.export().splitlines() if line]
    rows = [json.loads(line) for line in lines]
    assert [row["type"] for row in rows] == ["episode", "decision", "decision"]
    assert rows[1]["episode"] == "ep-1"
    assert rows[1]["seed"] == 123
    assert all("image" not in row for row in rows)


def test_export_of_an_empty_archive_is_empty(archive):
    assert archive.export() == ""


def test_history_brain_snapshot_survives_the_round_trip(archive):
    """The lab's decision panel reads `brain` back out of the archive, so the
    command rates have to be recorded, not just used."""
    archive.start(META, "data:,a")
    archive.record(payload(brain={"command": {"escape_L": {"rate": 18.8}}, "input": {"looms_L": 0.32}}))
    row = archive.current.decisions[0].public()
    assert row["brain"]["command"]["escape_L"]["rate"] == 18.8