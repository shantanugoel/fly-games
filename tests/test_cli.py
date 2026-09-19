"""The CLI is the documented public interface, so its shape is worth pinning."""

from __future__ import annotations

import argparse

import pytest

from fly_games import cli


def parse(argv: list[str]):
    return cli.build_parser().parse_args(argv)


def test_every_command_has_a_subcommand_and_a_handler():
    parser = cli.build_parser()
    choices = set(parser._subparsers._group_actions[0].choices)
    assert choices == set(cli.COMMANDS)


def test_defaults_match_the_readme():
    args = parse(["play"])
    assert (args.game, args.policy, args.decisions) == ("mario", "fly", 30)
    args = parse(["train"])
    assert (args.episodes, args.decisions) == (8, 40)
    args = parse(["lab"])
    assert (args.host, args.port) == ("127.0.0.1", 8000)


def test_game_flag_works_before_and_after_the_subcommand():
    assert parse(["--game", "doom", "play"]).game == "doom"
    assert parse(["play", "--game", "doom"]).game == "doom"
    # The subcommand only overrides when it is actually given.
    assert parse(["--game", "kungfu", "play"]).game == "kungfu"


def test_probe_would_do_distinguishes_absent_bare_and_listed():
    # `--would-do` with no names means every game; without the flag, none.
    assert parse(["probe", "--types", "LC4"]).would_do is None
    assert parse(["probe", "--types", "LC4", "--would-do"]).would_do == []
    assert parse(["probe", "--types", "LC4", "--would-do", "doom"]).would_do == ["doom"]


def test_probe_accepts_repeated_type_flags_and_a_side():
    args = parse(["probe", "--types", "LC4,LPLC2", "--types", "LC10a", "--side", "L", "--volts", "0.6"])
    assert args.types == ["LC4,LPLC2", "LC10a"]
    assert (args.side, args.volts, args.steps) == ("L", 0.6, 24)


def test_unknown_game_is_rejected():
    with pytest.raises(SystemExit):
        parse(["play", "--game", "zelda"])


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        parse([])


def test_run_rejects_an_unknown_command():
    with pytest.raises(ValueError, match="Unknown command"):
        cli.run(argparse.Namespace(cmd="nope"))


META = {"path": "readouts/mario.readout.npz", "kind": "ridge", "components": 2,
        "lam": 0.1, "cv_score": 0.42, "n_samples": 40, "episodes": 1}


def _fake_train(monkeypatch, meta: dict) -> None:
    from fly_games.engine import Engine

    monkeypatch.setattr(Engine, "collect_fly_samples", lambda *_a, **_k: [("x",)] * 40)
    monkeypatch.setattr(Engine, "train_readout_for", lambda *_a, **_k: meta)
    monkeypatch.setattr(Engine, "close", lambda *_a, **_k: None)
    monkeypatch.setattr(Engine, "set_policy", lambda *_a, **_k: None)


def test_a_single_episode_training_run_says_the_score_is_weaker(monkeypatch, capsys):
    """Leave-one-episode-out is the only CV that means anything here, because
    decisions within one episode are correlated. If no boundary was crossed the
    printed score is plain k-fold and must not be presented as the real thing."""
    _fake_train(monkeypatch, dict(META))
    assert cli._train(parse(["train"])) == 0
    assert "k-fold" in capsys.readouterr().out


def test_a_multi_episode_training_run_does_not_apologise(monkeypatch, capsys):
    _fake_train(monkeypatch, dict(META, episodes=6))
    assert cli._train(parse(["train"])) == 0
    assert "k-fold" not in capsys.readouterr().out


def test_a_refused_training_run_reports_the_reason(monkeypatch, capsys):
    _fake_train(monkeypatch, {"error": "need at least 8 samples"})
    assert cli._train(parse(["train"])) == 1
    assert "need at least 8 samples" in capsys.readouterr().out


def test_run_dispatches_to_the_handler(monkeypatch):
    seen = {}

    def fake(_args):
        seen["called"] = True
        return 0

    monkeypatch.setitem(cli.COMMANDS, "readouts", fake)
    assert cli.run(parse(["readouts"])) == 0
    assert seen == {"called": True}