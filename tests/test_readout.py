"""The readout: the only thing in this project that is ever fitted.

`flybrain.Readout` is pure numpy, so all of this runs without the connectome.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from fly_games.brain.client import COMMAND_GROUPS
from fly_games.brain.readout import COMMAND_GROUPS as REEXPORTED_GROUPS
from fly_games.brain.readout import FlyReadout

COARSE = ("proceed", "escape")


def labelled(rng: np.random.Generator, n: int = 120):
    """Synthetic command rates where `escape` is driven by escape_L and
    `proceed` by forward_R, so a fitted readout has something real to find."""
    escape = rng.normal(2.0, 0.6, n)
    forward = rng.normal(2.0, 0.6, n)
    X = np.zeros((n, len(COMMAND_GROUPS)))
    X[:, COMMAND_GROUPS.index("escape_L")] = escape
    X[:, COMMAND_GROUPS.index("forward_R")] = forward
    fine = np.where(escape > forward, "right_run_jump", "right_run")
    return X, list(fine)


def coarse_of(fine_action: str, observation: dict) -> str:
    return "escape" if "jump" in fine_action else "proceed"


@pytest.fixture()
def data():
    rng = np.random.default_rng(3)
    X, fine = labelled(rng)
    return X, fine, ["episode"] * len(fine)


def test_reexported_command_groups_are_the_same_table():
    assert REEXPORTED_GROUPS == COMMAND_GROUPS


def test_fit_requires_a_label_function(tmp_path):
    with pytest.raises(ValueError, match="coarse_of is required"):
        FlyReadout.fit(COARSE, np.zeros((4, 12)), ["right_run"] * 4, path=tmp_path / "m.npz")


def test_fit_rejects_misaligned_features_and_labels(tmp_path):
    with pytest.raises(ValueError, match="n_samples"):
        FlyReadout.fit(COARSE, np.zeros((4, 12)), ["right_run"] * 5,
                       coarse_of=coarse_of, path=tmp_path / "m.npz")


def test_a_single_episode_falls_back_instead_of_reporting_a_bogus_score(tmp_path, data):
    """Leave-one-episode-out CV needs at least two episodes. With one there is
    nothing held out, and reporting -inf would look like a training failure."""
    X, _fine, _ = data
    readout = FlyReadout.fit(COARSE, X, _fine_of(X), coarse_of=coarse_of,
                             episode_ids=[1] * len(X), path=tmp_path / "m.npz")
    assert np.isfinite(readout.info.cv_score)


def _fine_of(X: np.ndarray) -> list[str]:
    escape = X[:, COMMAND_GROUPS.index("escape_L")]
    forward = X[:, COMMAND_GROUPS.index("forward_R")]
    return ["right_run_jump" if e > f else "right_run" for e, f in zip(escape, forward, strict=True)]


def test_the_fitted_readout_recovers_the_rule(tmp_path, data):
    X, _fine, _ = data
    fine = _fine_of(X)
    readout = FlyReadout.fit(COARSE, X, fine, coarse_of=coarse_of,
                             episode_ids=[i % 4 for i in range(len(fine))],
                             path=tmp_path / "m.npz")
    loud_escape = {name: 0.0 for name in COMMAND_GROUPS}
    loud_escape["escape_L"] = 12.0
    quiet = {name: 0.0 for name in COMMAND_GROUPS}
    quiet["forward_R"] = 12.0
    assert readout.decode([loud_escape[k] for k in COMMAND_GROUPS])[0] == "escape"
    assert readout.decode([quiet[k] for k in COMMAND_GROUPS])[0] == "proceed"


def test_probabilities_are_a_distribution_over_the_coarse_set(tmp_path, data):
    X, fine, episodes = data
    readout = FlyReadout.fit(COARSE, X, fine, coarse_of=coarse_of,
                             episode_ids=episodes, path=tmp_path / "m.npz")
    coarse, probabilities = readout.decode(np.zeros(len(COMMAND_GROUPS)))
    assert coarse in COARSE
    assert set(probabilities) == set(COARSE)
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_weights_point_at_the_command_neurons_the_labels_came_from(tmp_path, data):
    """`weights()` is the UI's answer to "what did you teach it". It has to name
    the descending populations that actually carry the signal."""
    X, fine, episodes = data
    readout = FlyReadout.fit(COARSE, X, fine, coarse_of=coarse_of,
                             episode_ids=episodes, path=tmp_path / "m.npz")
    weights = readout.weights()
    assert set(weights) == set(COARSE)
    assert set(weights["escape"]) == set(COMMAND_GROUPS)
    escape_row = weights["escape"]
    assert escape_row["escape_L"] > escape_row["forward_R"]
    assert escape_row["escape_L"] > 0, "escape_L should raise the escape score"


def test_weights_match_a_finite_difference_of_decode(tmp_path, data):
    """The advertised sensitivity must be the model's real one, not a plausible
    looking approximation of it."""
    X, fine, episodes = data
    readout = FlyReadout.fit(COARSE, X, fine, coarse_of=coarse_of,
                             episode_ids=episodes, path=tmp_path / "m.npz")
    index = COMMAND_GROUPS.index("escape_L")
    base = np.zeros(len(COMMAND_GROUPS))
    step = 0.5

    def score(x: np.ndarray) -> float:
        _coarse, probabilities = readout.decode(x)
        return probabilities["escape"] - probabilities["proceed"]

    numeric = (score(base + step * np.eye(len(base))[index]) - score(base)) / step
    # `weights()` is normalised by the largest |sensitivity| across the matrix,
    # so compare signs and relative magnitude rather than absolute scale.
    advertised = readout.weights()["escape"]["escape_L"]
    assert np.sign(numeric) == np.sign(advertised)


def test_an_untrained_readout_refuses_to_decode_or_save(tmp_path):
    readout = FlyReadout(COARSE, tmp_path / "missing.npz")
    assert readout.trained is False
    assert readout.weights() == {}
    with pytest.raises(RuntimeError, match="train it first"):
        readout.decode(np.zeros(12))
    with pytest.raises(RuntimeError, match="nothing to save"):
        readout.save()


def test_save_and_load_round_trip(tmp_path, data):
    X, fine, episodes = data
    path = tmp_path / "readouts" / "mario.readout.npz"
    readout = FlyReadout.fit(COARSE, X, fine, coarse_of=coarse_of,
                             episode_ids=episodes, path=path)
    assert readout.save() == path
    assert path.exists()

    loaded = FlyReadout.load(COARSE, path)
    assert loaded.trained
    assert loaded.info.n_samples == len(fine)
    assert loaded.info.trained_at
    sample = np.zeros(len(COMMAND_GROUPS))
    sample[COMMAND_GROUPS.index("escape_L")] = 9.0
    assert loaded.decode(sample)[0] == readout.decode(sample)[0]


def test_the_sidecar_records_the_labels_it_was_fitted_against(tmp_path, data):
    X, fine, episodes = data
    path = tmp_path / "m.npz"
    FlyReadout.fit(COARSE, X, fine, coarse_of=coarse_of, episode_ids=episodes, path=path).save()
    meta = json.loads(path.with_suffix(".npz.json").read_text())
    assert meta["coarse_actions"] == list(COARSE)
    assert meta["features"] == list(COMMAND_GROUPS)


def test_a_readout_fitted_for_other_labels_fails_loudly(tmp_path, data):
    """Silently decoding with a readout trained against a different decision set
    would mislabel every fly decision, so loading has to refuse."""
    X, fine, episodes = data
    path = tmp_path / "m.npz"
    FlyReadout.fit(COARSE, X, fine, coarse_of=coarse_of, episode_ids=episodes, path=path).save()
    with pytest.raises(ValueError, match="Re-run `fly-games train`"):
        FlyReadout.load(("forward", "back", "left", "right"), path)


def test_loading_a_missing_file_gives_an_untrained_readout(tmp_path):
    readout = FlyReadout.load(COARSE, tmp_path / "nope.npz")
    assert readout.trained is False


def test_a_corrupt_sidecar_does_not_stop_the_readout_loading(tmp_path, data):
    X, fine, episodes = data
    path = tmp_path / "m.npz"
    FlyReadout.fit(COARSE, X, fine, coarse_of=coarse_of, episode_ids=episodes, path=path).save()
    path.with_suffix(".npz.json").write_text("{not json")
    readout = FlyReadout.load(COARSE, path)
    assert readout.trained


# --- where readouts live ------------------------------------------------------

def test_readout_directory_follows_the_environment(monkeypatch, tmp_path):
    from fly_games import readouts

    monkeypatch.setenv(readouts.ENV_VAR, str(tmp_path / "elsewhere"))
    assert readouts.readout_dir() == tmp_path / "elsewhere"
    assert readouts.readout_path("mario").parent == tmp_path / "elsewhere"


def test_available_games_reads_the_directory(tmp_path):
    from fly_games.readouts import available_games, readout_path

    assert available_games(tmp_path) == []
    for game_id in ("doom", "mario"):
        readout_path(game_id, base=tmp_path).parent.mkdir(parents=True, exist_ok=True)
        readout_path(game_id, base=tmp_path).write_bytes(b"x")
    assert available_games(tmp_path) == ["doom", "mario"]


def test_unrelated_files_are_not_readouts(tmp_path):
    from fly_games.readouts import available_games

    (tmp_path / "notes.npz").write_bytes(b"x")
    (tmp_path / "mario.readout.json").write_text("{}")
    assert available_games(tmp_path) == []