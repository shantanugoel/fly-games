"""Sensory encoding geometry: which of the fly's sides a stimulus drives.

The fly faces right on a NES screen, so "up the screen" is its left. Getting
this backwards would send every escape command to the wrong side of the body,
and because the readout is fitted on whatever the encoder produced, the bug
would hide.
"""

from __future__ import annotations

import numpy as np
import pytest

from fly_games.brain.encoder import ENCODER_PARAMS, SensoryEncoder, lateral_sides
from fly_games.brain.readout import COMMAND_GROUPS

FORWARD = (1.0, 0.0)


def sides(dx: float, dy: float, **kwargs) -> list[str]:
    return lateral_sides((dx, dy), FORWARD, **kwargs)


def test_ahead_is_bilateral():
    assert sides(80.0, 0.0) == ["L", "R"]


def test_above_is_left_and_below_is_right():
    # Screen coordinates: y grows downwards, and the fly's left is up.
    assert sides(80.0, -80.0) == ["L"]
    assert sides(80.0, 80.0) == ["R"]


def test_straight_up_and_down_are_pure_lateral():
    assert sides(0.0, -100.0) == ["L"]
    assert sides(0.0, 100.0) == ["R"]


def test_behind_is_bilateral_whatever_the_lateral_offset():
    assert sides(-200.0, 0.0) == ["L", "R"]
    assert sides(-200.0, -120.0) == ["L", "R"]


def test_near_the_midline_is_still_bilateral():
    assert sides(80.0, 5.0) == ["L", "R"]


def test_a_fly_facing_left_mirrors_the_sides():
    # Facing left, +x is behind, so the stimuli have to be at -x to be ahead.
    leftward = (-1.0, 0.0)
    assert lateral_sides((-80.0, -80.0), leftward) == ["R"]
    assert lateral_sides((-80.0, 80.0), leftward) == ["L"]


def test_a_diagonal_forward_vector_still_defines_a_left():
    # Facing down-right, the fly's left is down-left... except that screen "down"
    # is +y, so its left vector is (+y, -x) = down-left only for a right-facing
    # fly; for down-right the left vector is (1, -1), i.e. up-right.
    assert lateral_sides((60.0, -60.0), (1.0, 1.0)) == ["L"]
    assert lateral_sides((-60.0, 60.0), (1.0, 1.0)) == ["R"]


def test_zero_forward_vector_does_not_divide_by_zero():
    assert lateral_sides((10.0, 0.0), (0.0, 0.0)) == ["L", "R"]


def test_voltages_are_capped_at_the_fly_ai_maximum():
    assert ENCODER_PARAMS["cap"] == 0.8


def test_events_become_one_injection_per_channel_and_side(stub_client):
    encoder = SensoryEncoder(stub_client)
    inject, labels = encoder.events_to_inject([
        ("loom", "L", 0.4, "looms_L"),
        ("threat", "L", 0.6, "threat_L"),
    ])
    assert len(inject) == 2
    assert labels == {"looms_L": 0.4, "threat_L": 0.6}


def test_overvoltage_is_clipped_not_rejected(stub_client):
    encoder = SensoryEncoder(stub_client)
    _inject, labels = encoder.events_to_inject([("loom", "L", 4.0, "looms_L")])
    assert labels["looms_L"] == pytest.approx(0.8)


def test_zero_drive_is_dropped(stub_client):
    encoder = SensoryEncoder(stub_client)
    inject, labels = encoder.events_to_inject([("loom", "L", 0.0, "looms_L")])
    assert inject == [] and labels == {}


def test_negative_drive_is_dropped(stub_client):
    encoder = SensoryEncoder(stub_client)
    inject, labels = encoder.events_to_inject([("threat", "R", -1.0, "threat_R")])
    assert inject == [] and labels == {}


def test_duplicate_labels_accumulate_across_sides(stub_client):
    encoder = SensoryEncoder(stub_client)
    _inject, labels = encoder.events_to_inject([
        ("loom", "L", 0.3, "loom"),
        ("loom", "R", 0.3, "loom"),
    ])
    assert labels == {"loom": pytest.approx(0.6)}


def test_side_events_expand_one_stimulus_onto_both_sides(stub_client):
    encoder = SensoryEncoder(stub_client)
    events = encoder.side_events((80.0, 0.0), FORWARD, loom=0.5, threat=0.4, shot=0.0, chase=0.0)
    channels = {event[0] for event in events}
    names = {event[3] for event in events}
    assert channels == {"loom", "threat"}
    assert names == {"looms_L", "looms_R", "threat_L", "threat_R"}


def test_command_groups_are_the_twelve_descending_populations():
    assert len(COMMAND_GROUPS) == 12
    assert set(COMMAND_GROUPS) == {
        f"{behaviour}_{side}"
        for behaviour in ("escape", "steer", "forward", "backward", "punch", "kick")
        for side in ("L", "R")
    }


def test_a_game_encoder_must_implement_encode():
    class Blank(SensoryEncoder):
        pass

    with pytest.raises(NotImplementedError):
        Blank.__new__(Blank).encode({})


def test_angular_size_saturates_at_eight_pixels(stub_client):
    encoder = SensoryEncoder(stub_client)
    assert encoder._angle(100.0, 10.0) == pytest.approx(0.1)
    assert encoder._angle(2.0, 10.0) == pytest.approx(1.25)


def test_the_same_observation_always_drives_the_same_neurons(stub_client):
    """The fly is a deterministic dynamical system; the encoder must be too, or
    a re-run of an episode is not the same experiment."""
    encoder = SensoryEncoder(stub_client)
    events = [("loom", "L", 0.4, "looms_L"), ("chase", "R", 0.7, "chase_R")]
    first_idx, first_labels = encoder.events_to_inject(events)
    second_idx, second_labels = encoder.events_to_inject(events)
    assert first_labels == second_labels
    assert len(first_idx) == len(second_idx)
    for (a_idx, a_amt), (b_idx, b_amt) in zip(first_idx, second_idx, strict=True):
        assert np.array_equal(a_idx, b_idx)
        assert a_amt == b_amt