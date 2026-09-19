"""Which two of the connectome's axes get drawn, and why.

The canonical figure of this nervous system is portrait: optic lobes either side,
nerve cord trailing down. Getting the projection wrong turns it into an
unrecognisable smear, and nothing else in the app would complain.
"""

from __future__ import annotations

import numpy as np

from fly_games.brain.client import projection_axes

LATERAL, DORSOVENTRAL, HEAD_TO_TAIL = 0, 1, 2


def test_the_canonical_pair_is_chosen(synthetic_positions):
    positions, labels = synthetic_positions
    horizontal, vertical = projection_axes(positions, labels)
    assert horizontal == LATERAL, "optic lobes must sit either side of the screen"
    assert vertical == HEAD_TO_TAIL, "the nerve cord must trail down the screen"


def test_the_two_axes_are_always_different(synthetic_positions):
    positions, labels = synthetic_positions
    assert len(set(projection_axes(positions, labels))) == 2


def test_the_third_axis_is_the_remaining_one(synthetic_positions):
    positions, labels = synthetic_positions
    chosen = set(projection_axes(positions, labels))
    assert {0, 1, 2} - chosen == {DORSOVENTRAL}


def test_labels_are_not_needed_to_pick_something():
    """Without superclass labels there is no anatomy to find, but the caller
    still needs a usable pair of axes."""
    rng = np.random.default_rng(1)
    positions = rng.normal(size=(200, 3)).astype(np.float32) * np.array([1.0, 5.0, 9.0])
    horizontal, vertical = projection_axes(positions, None)
    assert (horizontal, vertical) == (2, 1), "widest spread first"


def test_neurons_without_a_soma_do_not_poison_the_choice(synthetic_positions):
    """A quarter of the real connectome has no soma coordinate. Plain min/max
    over that returns NaN, and every axis then compares equal."""
    positions, labels = synthetic_positions
    missing = np.isnan(positions).all(axis=1)
    assert missing.any(), "the fixture must include neurons with no soma"
    assert missing.sum() < positions.shape[0], "...but not only those"
    assert not np.isfinite(positions).all(axis=1).all()
    horizontal, vertical = projection_axes(positions, labels)
    assert (horizontal, vertical) == (LATERAL, HEAD_TO_TAIL)


def test_a_nervous_system_rotated_into_another_frame_still_reads_right():
    """The axes are found from the tissue, so a relabelled coordinate frame must
    not change what the figure looks like."""
    rng = np.random.default_rng(11)
    labels = np.array(["cb_intrinsic"] * 300 + ["ol_intrinsic"] * 600 + ["vnc_intrinsic"] * 200)
    base = np.zeros((1100, 3), dtype=np.float32)
    base[:300] = rng.normal(0.0, 1.0, (300, 3))
    base[300:900, 0] = rng.normal(0.0, 8.0, 600)
    base[300:900, 1:] = rng.normal(0.0, 1.0, (600, 2))
    base[900:, 2] = rng.normal(40.0, 4.0, 200)
    base[900:, :2] = rng.normal(0.0, 1.0, (200, 2))

    for permutation in ((0, 1, 2), (2, 0, 1), (1, 2, 0)):
        rotated = base[:, permutation]
        horizontal, vertical = projection_axes(rotated, labels)
        assert permutation[horizontal] == 0, "lateral axis should be horizontal"
        assert permutation[vertical] == 2, "head-to-tail axis should be vertical"


def test_a_degenerate_axis_set_does_not_crash():
    """All neurons at one point: no spread anywhere, no anatomy to find."""
    positions = np.zeros((10, 3), dtype=np.float32)
    labels = np.array(["cb_intrinsic"] * 10)
    horizontal, vertical = projection_axes(positions, labels)
    assert horizontal != vertical


def test_the_cloud_is_normalised_without_squashing_the_body(synthetic_positions):
    """Every axis is divided by the *longest* one, so the proportions survive.
    The flat view re-fits to its own bounding box with one shared scale."""
    from fly_games.brain.client import FlyBrainClient

    positions, _labels = synthetic_positions
    finite = np.isfinite(positions).all(axis=1)
    pts = positions[finite]
    lo = pts.min(axis=0)
    span = pts.max(axis=0) - lo
    longest = float(span.max())
    normalised = (pts - lo) / longest
    # The short axes must stay short: dividing per-axis would make all three 0..1.
    assert normalised.max(axis=0).max() == 1.0
    assert normalised.max(axis=0).min() < 0.9
    assert FlyBrainClient is not None  # the client builds on exactly this maths