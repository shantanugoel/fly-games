"""The per-title contract in `games/base.py`, checked against all three games.

These run without the connectome: they exercise the game-side half of the fly
loop (coarse labels, expansion, the zero-shot decode), not the brain.
"""

from __future__ import annotations

import pytest

from fly_games.brain.client import COMMAND_GROUPS
from fly_games.catalog import GAMES, get_game

QUIET = {name: {"count": 0, "rate": 0.0} for name in COMMAND_GROUPS}


def driven(**rates: float) -> dict:
    """A command dict with the given groups firing at the given Hz."""
    out = {name: {"count": 0, "rate": 0.0} for name in COMMAND_GROUPS}
    for name, rate in rates.items():
        assert name in out, name
        out[name] = {"count": int(rate), "rate": rate}
    return out


def test_the_three_bundled_titles_are_registered():
    assert set(GAMES) == {"mario", "kungfu", "doom"}


def test_get_game_names_the_bundled_titles_on_error():
    with pytest.raises(ValueError, match="Bundled games:"):
        get_game("zelda")


@pytest.mark.parametrize("game_id", sorted(GAMES))
def test_meta_is_self_consistent(game_id: str):
    game = GAMES[game_id]
    meta = game.meta
    assert meta.id == game_id
    assert meta.default_hold_frames in meta.hold_frames_options
    assert meta.brain_steps > 0 and meta.brain_warmup > 0
    keys = [action.key for action in game.actions]
    assert len(keys) == len(set(keys)), "duplicate action keys"
    assert set(meta.button_keys) <= {b for action in game.actions for b in action.buttons}


@pytest.mark.parametrize("game_id", sorted(GAMES))
def test_every_game_implements_the_fly_contract(game_id: str):
    game = GAMES[game_id]
    coarse = game.fly_coarse_actions()
    assert coarse, "a game must expose at least one coarse decision"
    assert len(set(coarse)) == len(coarse)

    observation = {"text": "", "terrain": {}, "nearby_objects": []}
    # The zero-shot decoder must always answer with one of the coarse labels and
    # a probability distribution over all of them.
    for command in (QUIET, driven(escape_L=30.0), driven(steer_L=9.0, forward_R=4.0)):
        choice, probabilities = game.fly_hand_decode(command, observation)
        assert choice in coarse, (game_id, choice)
        assert set(probabilities) == set(coarse)
        assert sum(probabilities.values()) == pytest.approx(1.0, abs=0.05)

    # Every coarse decision has to expand to a real controller action.
    keys = {action.key for action in game.actions}
    for label in coarse:
        assert game.fly_expand(label, observation) in keys


@pytest.mark.parametrize("game_id", sorted(GAMES))
def test_training_labels_are_drawn_from_the_coarse_set(game_id: str):
    game = GAMES[game_id]
    coarse = set(game.fly_coarse_actions())
    for action in (action.key for action in game.actions):
        assert game.fly_coarse_of(action, {}) in coarse


@pytest.mark.parametrize("game_id", sorted(GAMES))
def test_an_empty_world_drives_nothing(game_id: str, stub_client):
    """No hazards, no sensory injection.

    Worth pinning: an encoder that invents stimuli from an empty observation
    would have the fly reacting to things that are not there, and the symptom (a
    constant escape rate) looks like a brain property rather than a bug.
    """
    inject, labels = GAMES[game_id].fly_encoder(stub_client).encode(
        {"text": "", "terrain": {}, "nearby_objects": []})
    assert inject == []
    assert labels == {}


def test_threat_drives_escape_and_quiet_does_not():
    """The one behavioural claim the README makes about the whole loop."""
    game = get_game("mario")
    quiet, _ = game.fly_hand_decode(QUIET, {})
    frightened, _ = game.fly_hand_decode(driven(escape_L=18.0), {})
    # "proceed" is the unconditional residual, and that is load-bearing rather
    # than sloppy: the fly fires ~3 spikes at rest across all twelve command
    # groups, so nearly every decision is silent, and any rule that reads
    # silence as "halt" stops Mario dead. See hand_decode.
    assert quiet == "proceed"
    assert frightened == "escape"
    assert game.fly_expand("escape", {}) == "right_run_jump"
    assert game.fly_expand("proceed", {}) == "right_run"


def test_the_wider_vocabulary_is_expressible():
    """`backward_*` was computed every decision and thrown away, which capped the
    fly at "run right" whatever it did. Backing off is the only way past a pipe
    you are already accelerating into, so it has to be reachable - by the fitted
    readout, which is what the four-label space is for."""
    game = get_game("mario")
    assert game.fly_expand("retreat", {}) == "left_run"
    assert game.fly_expand("halt", {}) == "wait"
    assert set(game.fly_coarse_actions()) == {"proceed", "escape", "retreat", "halt"}
    assert game.fly_coarse_of("left_run", {}) == "retreat"
    assert game.fly_coarse_of("wait", {}) == "halt"


def test_doom_reads_the_steering_channels_separately():
    """Doom has to turn toward a monster, so steer_L and steer_R must not
    collapse into one decision the way they do in the side-on games."""
    game = get_game("doom")
    left, left_probs = game.fly_hand_decode(driven(steer_L=9.0), {})
    right, _ = game.fly_hand_decode(driven(steer_R=9.0), {})
    assert (left, right) == ("turn_left", "turn_right")
    assert left_probs["turn_left"] > left_probs["turn_right"]


def test_kungfu_turns_the_sides_over_with_thomas(stub_client):
    """Thomas faces left on odd floors and right on even ones.

    The decoder collapses both escapes into one "escape" decision on purpose, so
    the side information only matters on the input side: the same world-space
    offset has to drive the fly's left when Thomas faces right and its right when
    he faces left. If it did not, half the floors would be dodged the wrong way.
    """
    game = get_game("kungfu")

    def sides_for(facing: str) -> set[str]:
        observation = {
            "player": {"x": 100.0, "y": 100.0, "facing": facing},
            "combat": {"someone_will_hit_you": True,
                       "best_target": {"x": 100.0 + (30.0 if facing == "right" else -30.0),
                                       "y": 60.0}},
        }
        _inject, labels = game.fly_encoder(stub_client).encode(observation)
        return {label.rsplit("_", 1)[-1] for label in labels}

    assert sides_for("right") == {"L"}
    assert sides_for("left") == {"R"}


@pytest.mark.parametrize("observation", [
    {},
    {"player": {}, "combat": {}},
    {"player": {"facing": "left"}, "combat": {"someone_will_hit_you": True}},
    {"player": {"x": None, "y": None}, "combat": {"best_target": {"x": 10, "y": 10}}},
    {"player": {"x": 100, "y": 60}, "combat": {"best_target": None, "nearest": None}},
])
def test_kungfu_survives_an_observation_with_no_positions(stub_client, observation):
    """The observe pass can fail to find Thomas or a foe, and that must not crash
    the decision loop.

    `dict.get(key, {})` is not enough here: the observe pass writes an explicit
    `None` when it finds nothing, and `get` hands that straight back.
    """
    inject, labels = get_game("kungfu").fly_encoder(stub_client).encode(observation)
    assert isinstance(inject, list) and isinstance(labels, dict)


def test_kungfu_invents_no_hazard_when_nothing_threatens(stub_client):
    """A phantom hazard would have the fly dodging an empty floor, and the
    escape rate that causes would then be baked into the trained readout."""
    for observation in (
        {},
        {"player": {"x": 100, "y": 60, "facing": "right"}, "combat": {}},
        {"player": {"x": 100, "y": 60, "facing": "right"},
         "combat": {"best_target": {"x": 400, "y": 60}}},
    ):
        assert get_game("kungfu").fly_encoder(stub_client).encode(observation) == ([], {})


# Each title's observe pass has its own shape, so each gets its own hazard.
HAZARD_OBSERVATION = {
    "mario": {
        "terrain": {"summary": {"nearest_obstacle": {"distance_px": 20}}},
        "nearby_objects": [{"dx": 20, "dy": 0, "kind": "goomba",
                            "estimated_contact_in_frames": 10}],
    },
    "kungfu": {
        "player": {"x": 100.0, "y": 100.0, "facing": "right"},
        "combat": {"someone_will_hit_you": True,
                   "best_target": {"x": 120.0, "y": 100.0, "kind": "thug"}},
    },
    "doom": {
        "best_target": {"role": "monster", "distance": 40.0, "screen_x": 180.0},
        "nearby_objects": [{"role": "monster", "on_screen": True, "distance": 40.0,
                            "screen_x": 180.0}],
        "combat": {"someone_will_shoot_you": True},
    },
}


@pytest.mark.parametrize("game_id", sorted(GAMES))
def test_a_close_hazard_drives_more_than_an_empty_room(game_id: str, stub_client):
    """The encoder has to actually respond to the world. If it did not, every
    decision would be the resting state and the readout would be fitted on a
    constant."""
    encoder = GAMES[game_id].fly_encoder(stub_client)
    _inject, labels = encoder.encode(HAZARD_OBSERVATION[game_id])
    assert labels, f"{game_id} drove nothing with a monster 40 px away"
    assert all(value > 0.0 for value in labels.values())
    # Labels accumulate when several hazards share a name; the 0.8 cap is per
    # injection, which is what the connectome is actually driven with.
    _inject2, _labels2 = encoder.encode({})
    assert _inject2 == []