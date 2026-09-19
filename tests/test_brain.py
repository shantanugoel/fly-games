"""The connectome itself.

Skipped entirely without the data (`fly-games brain-download`). Everything here
uses `shared_client()`, which loads the brain once for the whole session, so the
cost is one load plus a few ~100 ms decision windows.

These are the tests that check the project's actual claim: that the *real
wiring* produces a readable motor response to a sensory stimulus.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import needs_brain
from fly_games.brain.client import COMMAND_GROUPS
from fly_games.catalog import GAMES

pytestmark = needs_brain


@pytest.fixture(scope="module")
def brain():
    from fly_games.brain import shared_client

    client = shared_client()
    assert client.brain is not None, "the connectome failed to load"
    return client


def test_the_shared_client_is_a_singleton(brain):
    from fly_games.brain import shared_client

    assert shared_client() is brain


def test_the_connectome_is_the_published_size(brain):
    meta = brain.meta
    assert meta["n_neurons"] == 166_700
    assert meta["n_connections"] == 25_582_938
    assert meta["dt"] == pytest.approx(0.02)


def test_the_twelve_command_groups_are_all_present(brain):
    groups = dict(brain.command_groups)
    assert set(groups) == set(COMMAND_GROUPS)
    assert all(count >= 1 for count in groups.values())


def test_every_sensory_channel_resolves_to_real_neurons(brain):
    from fly_games.brain import CHANNELS

    channels = dict(brain.sensory_channels)
    assert set(channels) == set(CHANNELS)
    assert all(count > 0 for count in channels.values())


# --- the point cloud ----------------------------------------------------------

def test_the_cloud_holds_every_neuron_with_a_soma(brain):
    cloud = brain.cloud()
    assert cloud["count"] == 140_638
    assert cloud["count"] < brain.meta["n_neurons"], "some neurons have no soma coordinate"
    assert np.isfinite(cloud["xyz"]).all()


def test_the_cloud_probes_every_command_neuron(brain):
    """If a command neuron were not in the probe set, its rate would read as zero
    forever and the readout would be fitted on silence."""
    cloud = brain.cloud()
    card = brain.brain_card()
    assert {entry["name"] for entry in card["command_groups"]} == set(COMMAND_GROUPS)
    for entry in card["command_groups"]:
        assert entry["probe_ids"], f"{entry['name']} has no probe id"
        assert len(entry["probe_ids"]) == entry["neurons"]
        for probe_id in entry["probe_ids"]:
            assert 0 <= probe_id < cloud["n_probes"]
    # The command neurons themselves are drawn in the cloud, marked by group.
    assert set(np.unique(cloud["command"][cloud["command"] >= 0])) == set(range(len(COMMAND_GROUPS)))


def test_the_blob_decodes_back_to_the_same_cloud(brain):
    blob = brain.points_blob()
    assert blob[:6] == b"FLYPT1"
    count = int(np.frombuffer(blob[10:14], dtype="<u4", count=1)[0])
    assert count == brain.cloud()["count"]
    assert len(blob) == 64 + count * 12 + brain.cloud()["n_probes"] * 4


def test_the_cloud_revision_is_stable_and_url_addressed(brain):
    assert brain.cloud_rev() == brain.cloud_rev()
    assert brain.brain_card()["points_url"].endswith(f"rev={brain.cloud_rev()}")


def test_the_vertical_axis_puts_the_head_at_the_top(brain):
    """The figure is drawn brain-up with the nerve cord trailing down, so the
    ventral nerve cord must sit at *high* vertical coordinates."""
    _horizontal, vertical, _depth = brain.brain_card()["axes"]
    positions = np.asarray(brain.brain.positions, dtype=np.float64)
    labels = np.asarray(brain.brain.superclass)
    present = np.isfinite(positions).all(axis=1)
    cord = present & (labels == "vnc_intrinsic")
    central = present & (labels == "cb_intrinsic")
    assert cord.sum() > 1000 and central.sum() > 1000
    assert positions[cord, vertical].mean() > positions[central, vertical].mean()


# --- the fly's response to being stimulated -----------------------------------

def rates(events: list[dict], steps: int = 24) -> dict[str, float]:
    """Descending command rates (Hz) after driving some cell types."""
    from fly_games.probe import ProbeBench

    result = ProbeBench(steps=steps).run(events, steps=steps)
    return {name: entry["rate"] for name, entry in result.public(include_field=False)["command"].items()}


def test_silence_is_quiet():
    """Baseline for everything below. A non-zero floor is fine — the connectome
    is spontaneously active — but escape must not be saturated at rest."""
    resting = rates([])
    assert max(resting["escape_L"], resting["escape_R"]) < 10.0


def test_a_looming_threat_drives_the_giant_fiber_escape_pathway():
    """The claim the README makes, stated as a test.

    LC4 is the fly's own selective-looming neuron and DNp01/DNp02 the descending
    escape command it is wired to. Stimulating LC4 has to raise the escape rate,
    or the loop has no biology in it and the readout is fitting noise.
    """
    resting = rates([])
    startle = rates([{"types": ["LC4"], "side": None, "volts": 0.8}])
    rise = max(startle["escape_L"], startle["escape_R"]) - max(resting["escape_L"], resting["escape_R"])
    assert rise > 5.0, f"escape only rose {rise:+.2f} Hz under LC4 stimulation"


def test_the_response_is_specific_to_the_stimulated_channel():
    """Chase cells (LC10a) are not the escape pathway. If every stimulus raised
    escape equally, the measurement would be an artefact of the window."""
    startle = rates([{"types": ["LC4"], "side": None, "volts": 0.8}])
    chase = rates([{"types": ["LC10a"], "side": None, "volts": 0.8}])
    assert max(startle["escape_L"], startle["escape_R"]) > max(chase["escape_L"], chase["escape_R"])


def test_a_unilateral_stimulus_is_stronger_on_its_own_side():
    """Escape neurons are left/right populations. A one-sided stimulus should not
    produce a symmetric output, or the side information is being lost."""
    left = rates([{"types": ["LPLC2"], "side": "L", "volts": 0.8}], steps=24)
    right = rates([{"types": ["LPLC2"], "side": "R", "volts": 0.8}], steps=24)
    left_bias = left["escape_L"] - left["escape_R"]
    right_bias = right["escape_L"] - right["escape_R"]
    assert left_bias > right_bias, (left_bias, right_bias)


def test_a_decision_exposes_all_twelve_command_rates(brain):
    decision = brain.decide([], {})
    command = decision.command
    assert set(command) == set(COMMAND_GROUPS)
    assert all({"count", "rate"} <= set(entry) for entry in command.values())
    assert len(decision.feature()) == len(COMMAND_GROUPS)


def test_the_feature_vector_is_the_command_rates(brain):
    """The readout is fitted on `feature()`, so it must be the same numbers the
    UI displays as command rates — otherwise the panel describes a different
    decision from the one the model saw."""
    decision = brain.decide([], {})
    feature = np.asarray(decision.feature(), dtype=np.float64)
    rates_vector = np.array([decision.command[name]["rate"] for name in COMMAND_GROUPS])
    assert feature.shape == rates_vector.shape
    assert np.allclose(feature, rates_vector, atol=1e-6)


def test_the_spike_field_is_one_entry_per_probed_neuron(brain):
    cloud = brain.cloud()
    field = brain.spike_field(np.zeros(brain.meta["n_neurons"], dtype=np.int32))
    assert len(field) == cloud["n_probes"]


def test_the_bench_reports_what_the_fly_would_do(brain):
    """The probe tab's payoff: the same command rates, decoded by each game."""
    from fly_games.probe import ProbeBench

    result = ProbeBench(steps=24).run(
        [{"types": ["LC4"], "side": None, "volts": 0.8}], games=["mario", "doom"])
    verdicts = result.public(include_field=False)["would_do"]
    assert set(verdicts) == {"mario", "doom"}
    for game_id, verdict in verdicts.items():
        assert verdict["coarse"] in GAMES[game_id].fly_coarse_actions()
        # Which interpreter produced it. Whether a readout has been trained is
        # local state, so assert only that the bench says which one it used.
        source = verdict["source"]
        assert "readout" in source or "zero-shot" in source, source
        assert verdict["probabilities"]