"""The lab engine: the phase machine, sample collection, and fitting.

This is where the fly, the emulator and the archive meet, and it is where the
subtle bugs live — a sample filter that can never match, an attribute read before
it is assigned, a phase that lies about what the fly is doing.

Needs the connectome and a Super Mario Bros. cartridge.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import needs_brain
from fly_games.brain.client import COMMAND_GROUPS

pytestmark = [needs_brain, pytest.mark.usefixtures("needs_mario")]


@pytest.fixture()
def needs_mario(needs_rom):
    needs_rom("mario")


@pytest.fixture()
def engine(tmp_path):
    from fly_games.engine import Engine

    eng = Engine(game_id="mario", readout_dir=tmp_path / "readouts")
    try:
        yield eng
    finally:
        eng.close()


# --- boot and snapshot --------------------------------------------------------

def test_the_engine_boots_paused_with_a_full_snapshot(engine):
    assert engine.phase() == "paused"
    snapshot = engine.snapshot()
    assert snapshot["game"]["id"] == "mario"
    assert snapshot["policy"] == "fly"
    assert snapshot["phase"] == "paused"
    assert snapshot["running"] is False
    assert snapshot["screen"]["image"].startswith("data:image")
    assert snapshot["progress"]["decisions"] == 0


def test_the_snapshot_names_the_readout_that_is_actually_in_use(engine):
    """The panel claims 'trained readout' or 'zero-shot rule'; saying the wrong
    one would make a trained fly look untrained and vice versa."""
    readout = engine.snapshot()["readout"]
    assert readout["trained"] is False
    assert readout["info"] is None
    assert readout["coarse_actions"] == list(engine.game.fly_coarse_actions())


def test_an_unknown_policy_is_refused(engine):
    with pytest.raises(ValueError):
        engine.set_policy("gpt")


def test_an_unsupported_hold_duration_is_refused(engine):
    with pytest.raises(ValueError, match="hold"):
        engine.command("configure", frames=7)


def test_closing_twice_is_harmless(engine):
    engine.close()
    engine.close()


# --- the phase machine --------------------------------------------------------

def test_running_flips_the_phase(engine):
    engine.command("run")
    assert engine.running is True
    assert engine.phase() in {"acting", "thinking"}
    engine.command("pause")
    assert engine.phase() == "paused"


@pytest.mark.parametrize(("flags", "expected"), [
    ({"running": True}, "acting"),
    ({"busy": True}, "thinking"),
    ({"pending_step": True}, "thinking"),
    ({"done": True}, "dead"),
    ({"done": True, "completed": True}, "complete"),
    ({"error": "boom"}, "error"),
    ({}, "paused"),
])
def test_the_phase_tells_the_truth_about_what_the_fly_is_doing(engine, flags, expected):
    """The UI is built around the thinking gap: 'THINKING' while the fly
    integrates, 'ACTING' while the emulator moves. A phase that skips the first
    half makes the fly look instant, and that is the whole story of this project.
    """
    for name, value in flags.items():
        setattr(engine, name, value)
    assert engine.phase() == expected


# --- deciding -----------------------------------------------------------------

def test_a_decision_names_a_real_action_and_explains_itself(engine):
    engine.set_policy("fly")
    action, diagnostics = engine.decide(engine.observation)
    assert action in {a.key for a in engine.game.actions}
    assert diagnostics["coarse"] in engine.game.fly_coarse_actions()
    assert set(diagnostics["probabilities"]) == set(engine.game.fly_coarse_actions())
    assert sum(diagnostics["probabilities"].values()) == pytest.approx(1.0, abs=0.05)
    assert diagnostics["source"] == "fly"


def test_a_decision_records_the_fly_s_own_command_rates(engine):
    """The whole point of the lab is seeing *why* the fly did what it did."""
    engine.set_policy("fly")
    _action, diagnostics = engine.decide(engine.observation)
    assert set(diagnostics["command"]) == set(COMMAND_GROUPS)
    assert all(np.isfinite(rate) and rate >= 0.0 for rate in diagnostics["command"].values())
    assert diagnostics["latency_ms"] > 0
    assert isinstance(diagnostics["input"], dict), "the senses that drove this decision"


def test_the_scripted_policy_never_touches_the_brain(engine):
    engine.set_policy("scripted")
    _action, diagnostics = engine.decide(engine.observation)
    assert diagnostics["model"] == "scripted"
    assert "command" not in diagnostics
    assert engine.last_brain is None


def test_the_previous_action_is_known_from_the_first_decision_onward(engine):
    """`observe()` is handed the action that just ran, so it can tell 'a wall I am
    walking into' from 'a wall I have stopped at'. It has to exist before the
    first decision, not after it."""
    engine.set_policy("fly")
    assert engine.previous_action is not None
    engine.decide(engine.observation)
    assert engine.previous_action is not None


# --- training -----------------------------------------------------------------

def test_collected_samples_have_the_shape_the_readout_needs(engine):
    samples = engine.collect_fly_samples(4, episodes=2)
    assert len(samples) >= 4
    features, fine, observation = samples[0]
    assert len(features) == len(COMMAND_GROUPS), "features are the twelve command rates"
    assert fine in {a.key for a in engine.game.actions}
    assert isinstance(observation, dict)


def test_collected_samples_are_labelled_by_episode(engine):
    """Episode ids are what make the cross-validation honest: samples from one
    episode are wildly correlated, so scoring inside an episode is not scoring.

    `episodes` is a cap on how many boundaries to wait for, not a promise -- if
    the fly survives, every sample legitimately belongs to episode 1, and the
    caller has to be able to see that.
    """
    samples = engine.collect_fly_samples(8, episodes=2)
    ids = engine._episode_ids
    assert len(ids) == len(samples)
    assert ids == sorted(ids) and ids[0] == 1
    assert all(isinstance(value, int) and value >= 1 for value in ids)


def test_training_produces_a_readout_from_collected_samples(engine):
    """The filter that selects usable samples once silently dropped every one of
    them by comparing a 12-wide feature vector against a 2-wide label set."""
    samples = engine.collect_fly_samples(10, episodes=2)
    meta = engine.train_readout_for(samples)
    assert "error" not in meta, meta.get("error")
    assert meta["n_samples"] >= 8
    assert meta["episodes"] == len(set(engine._episode_ids))
    assert meta["episodes"] >= 1
    assert meta["path"].endswith("mario.readout.npz")
    assert engine.get_readout().trained


def test_a_trained_readout_takes_over_the_decision(engine):
    engine.set_policy("fly")
    assert engine.snapshot()["readout"]["trained"] is False
    engine.train(decisions=10, episodes=2)
    readout = engine.snapshot()["readout"]
    assert readout["trained"] is True
    assert readout["info"]["kind"] in {"ridge", "pls"}
    action, diagnostics = engine.decide(engine.observation)
    assert diagnostics["model"].startswith("fly-readout")
    assert action in {a.key for a in engine.game.actions}


def test_forcing_the_hand_rule_ignores_the_trained_readout(engine):
    engine.train(decisions=10, episodes=2)
    engine.set_policy("fly-hand")
    _action, diagnostics = engine.decide(engine.observation)
    assert diagnostics["model"] == "fly-hand", "the hand rule must be forceable"


def test_training_too_few_samples_refuses_instead_of_overwriting(engine):
    """A readout fitted on a couple of decisions looks fine, saves over the
    working one, and then mislabels everything. Better to keep the old one."""
    samples = engine.collect_fly_samples(4, episodes=2)
    meta = engine.train_readout_for(samples[:2])
    assert "error" in meta
    assert "at least" in meta["error"]
    assert not engine.readout_path().exists()
    assert engine.get_readout().trained is False


# --- episodes -----------------------------------------------------------------

def test_a_reset_starts_a_fresh_recorded_episode(engine):
    engine.command("reset")
    summaries = engine.history_summaries()
    assert summaries[0]["game"] == "mario"
    assert summaries[0]["decisions"] == 0


def run_one(engine):
    """One full decide / advance / record cycle, the way the worker drives it."""
    engine.pending_step = True
    assert engine._run_once() is not None
    return engine


def test_the_archive_keeps_the_fly_s_side_of_each_decision(engine):
    engine.set_policy("fly")
    run_one(engine)
    episode = engine.archive.current
    assert episode is not None and episode.decisions
    brain = episode.decisions[-1].brain
    assert brain is not None
    assert set(brain["command"]) == set(COMMAND_GROUPS)


def test_replay_rows_separate_thinking_from_acting(engine):
    engine.set_policy("fly")
    run_one(engine)
    track = engine.archive.current.replay()
    step = track["steps"][-1]
    assert step["thinking_ms"] > 0, "the fly spent measurable time integrating"
    assert step["wall_ms"] > step["game_ms"]


def test_the_snapshot_carries_a_compact_brain_record(engine):
    """The websocket ships this on every decision; if it grew a full spike array
    the browser would fall over."""
    engine.set_policy("fly")
    run_one(engine)
    fly = engine.snapshot()["fly"]
    assert fly is not None, "the fly's own record of the decision it just made"
    assert set(fly["command"]) == set(COMMAND_GROUPS)
    assert fly["groups"] == list(COMMAND_GROUPS)
    assert isinstance(fly["steps"], int) and fly["steps"] > 0
    # The senses that drove it. Empty is legitimate and expected at the start of
    # a level: nothing is approaching, so nothing is injected.
    assert isinstance(fly["input"], dict)
    # The per-neuron heat map only rides along when someone asked for it.
    assert "field" not in fly

# --- the loop has to re-observe ----------------------------------------------
# `play` used to hand the fly the same frame forever. The loop was inlined in
# the CLI and skipped the re-observe, so every decision was made against the
# reset screen: Mario never moved and died about two seconds in. Nothing in the
# suite caught it, because each test either refreshed by hand or never read the
# observation between steps.

def test_each_step_leaves_the_fly_looking_at_the_new_world(engine):
    engine.set_policy("fly")
    seen = []
    for _ in range(12):
        seen.append(dict(engine.observation.get("mario") or {}))
        if engine.step() is None:
            break
    xs = [row.get("x") for row in seen]
    assert len(seen) > 2, "the engine refused to step at all"
    assert len(set(xs)) > 1, f"the fly decided {len(seen)} times against x={xs[0]}: nothing moved"


def test_a_step_reports_the_action_it_actually_took(engine):
    engine.set_policy("scripted")
    stepped = engine.step()
    assert stepped is not None
    action, _diagnostics, result = stepped
    assert action in {a.key for a in engine.game.actions}
    assert result.frames_executed == engine.hold_frames
    assert engine.frames >= result.frames_executed


def test_a_step_refuses_once_the_episode_is_over(engine):
    engine.set_policy("scripted")
    engine.done = True
    assert engine.step() is None


def test_a_seed_reaches_the_emulator(tmp_path):
    """`play --seed` is the only way to see run-to-run spread, so it has to
    actually change what the emulator starts from."""
    from fly_games.engine import Engine

    default = Engine(game_id="mario", readout_dir=tmp_path / "a")
    forced = Engine(game_id="mario", readout_dir=tmp_path / "b", seed=1234)
    try:
        assert default.seed == default.game.meta.default_seed
        assert forced.seed == 1234
    finally:
        default.close()
        forced.close()


# --- the escape reflex outranks the fitted map --------------------------------

def test_a_firing_escape_reflex_overrides_a_trained_readout(engine, monkeypatch):
    """The readout is fitted open-loop; in play it meets states it never saw.

    Measured on Mario: from decision 16 the escape neurons run at 17-50 Hz while
    the readout answers `proceed`, and the fly dies at 27. With the override it
    reaches 150. Before this, the readout simply replaced the rule that worked.
    """
    engine.set_policy("fly")

    # Force the disagreement: a readout that always says proceed, a reflex that
    # always says escape.
    class AlwaysProceed:
        trained = True

        class model:
            kind = "stub"

        def decode(self, feature):
            return "proceed", {"proceed": 0.6, "escape": 0.4}

    engine._readout_cache = AlwaysProceed()
    monkeypatch.setattr(engine.game, "fly_hand_decode",
                        lambda command, obs: ("escape", {"escape": 0.8, "proceed": 0.2}))

    engine.reflex_override = True
    _action, diagnostics = engine.decide(engine.observation)
    assert diagnostics["coarse"] == "escape"
    assert diagnostics["source"] == "reflex"

    engine.reflex_override = False
    _action, diagnostics = engine.decide(engine.observation)
    assert diagnostics["coarse"] == "proceed"
    assert diagnostics["source"] == "fly"
