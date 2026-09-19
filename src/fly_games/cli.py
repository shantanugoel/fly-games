"""fly-games command line.

  fly-games lab      [--host 127.0.0.1] [--port 8000]   the observatory in a browser
  fly-games play     [--policy fly] [--decisions 30]    headless, prints every thought
  fly-games train    [--episodes 8] [--decisions 40]    fit a game's readout
  fly-games probe    --types LC4 --side L --volts 0.8   stimulate the connectome
  fly-games readouts                                    what has been fitted
  fly-games brain-info / brain-download                 the connectome itself

`--game` works before or after the subcommand: `fly-games --game doom play` and
`fly-games play --game doom` are the same thing.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

GAMES = ("mario", "kungfu", "doom")
POLICIES = ("fly", "fly-hand", "scripted")


# --------------------------------------------------------------------------- lab
def _lab(args: argparse.Namespace) -> int:
    from fly_games.app import create_app

    os.environ["FLY_GAMES_GAME"] = args.game
    print(f"fly-games observatory ({args.game}) on http://{args.host}:{args.port}")
    print("Ctrl+C to stop.")
    import uvicorn

    uvicorn.run(create_app(game_id=args.game), host=args.host, port=args.port)
    return 0


# -------------------------------------------------------------------------- play
def _play(args: argparse.Namespace) -> int:
    """Run the fly through a real episode and print each thought as it happens."""
    from fly_games.engine import Engine

    engine = Engine(game_id=args.game, seed=args.seed,
                    **({"readout_dir": args.readouts} if args.readouts else {}))
    engine.set_policy(args.policy)
    coarse_labels = engine.game.fly_coarse_actions()
    readout = engine.get_readout()
    if readout.trained and (readout.info.cv_score or 0) <= 0:
        print("note: this readout's cross-validation score is not better than "
              "predicting the mean,\n"
              "      so it replaces a hand rule that works with a map that does not.\n"
              "      Compare with: --readouts /tmp/empty")
    print(f"{engine.game.meta.title} - policy {engine.policy} - "
          f"{'readout ' + str(readout.info.kind) if readout.trained else 'zero-shot rule'}"
          f" - seed {engine.seed}")
    print(f"coarse decisions: {', '.join(coarse_labels)}")
    print(f"{'#':>4}  {'frames':>7}  {'reward':>7}  {'decision':<12} -> {'action':<16} "
          f"{'p':>5}  {'ms':>6}")
    print("-" * 78)
    steps = 0
    try:
        while steps < args.decisions:
            stepped = engine.step()
            if stepped is None:
                break
            action, diagnostics, result = stepped
            steps += 1
            probs = diagnostics.get("probabilities") or {}
            coarse = diagnostics.get("coarse") or "scripted"
            confidence = probs.get(coarse, 1.0) if probs else 1.0
            drives = " ".join(f"{k}={v:.2f}" for k, v in sorted((diagnostics.get("input") or {}).items()))
            print(f"{steps:>4}  {engine.frames:>7}  {result.reward:>+7.2f}  {coarse:<12} -> "
                  f"{action:<16} {confidence:>5.2f}  {diagnostics.get('latency_ms') or 0:>6.0f}"
                  + (f"\n       senses: {drives}" if args.verbose and drives else ""))
            if result.terminated or result.truncated:
                print(f"{'':>4}  {'':>7}  {'':>7}  "
                      f"{'episode over' + (' (completed)' if engine.game.completed(result.info) else '')}")
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        engine.close()
    return 0


# ------------------------------------------------------------------------- train
def _train(args: argparse.Namespace) -> int:
    """Collect fly command-rate traces under the scripted policy, then fit the
    per-game readout. This is the only thing in the project that is ever fitted."""
    from fly_games.engine import Engine

    engine = Engine(game_id=args.game)
    episodes = max(1, args.episodes)
    total = args.decisions * episodes
    print(f"Watching the scripted policy play {args.game} for {total} decisions "
          f"({episodes} episode{'s' if episodes != 1 else ''} of {args.decisions}) "
          f"while the fly brain records its own descending activity...")
    started = time.time()
    try:
        samples = engine.collect_fly_samples(args.decisions, episodes=episodes)
        print(f"  {len(samples)} decisions in {time.time() - started:.0f}s.")
        meta = engine.train_readout_for(samples)
        if "error" in meta:
            print(f"training failed: {meta['error']}")
            return 1
        print(f"Saved {args.game} readout -> {meta['path']}")
        print(f"  kind={meta['kind']}  components={meta['components']}  lam={meta['lam']:g}")
        print(f"  samples={meta['n_samples']}  episodes={meta['episodes']}  "
              f"cv_score={meta['cv_score']:.4f} (held-out ridge MSE, higher is better)")
        if meta["episodes"] < 2:
            # Leave-one-episode-out is what makes this score mean anything:
            # decisions inside one episode are wildly correlated, so scoring
            # within an episode flatters the readout. Say so rather than
            # printing a number that looks comparable to a real one.
            print("  note: this was a single episode, so the score is plain k-fold, "
                  "not leave-one-episode-out. Raise --episodes.")
        if meta["cv_score"] <= 0:
            # A readout at or below the mean has learned nothing, and it is
            # worse than nothing: it replaces the hand rule that already works.
            # Mario does this today - the encoder folds enemies, gaps and walls
            # onto the same two channels, so the label is not predictable from
            # the features and more data only fits the ambiguity harder.
            print("  WARNING: cv_score <= 0 means this readout is no better than "
                  "predicting the mean,\n"
                  "           and it now overrides the zero-shot rule that worked. "
                  "Try:\n"
                  f"             fly-games play --game {args.game} --readouts /tmp/empty"
                  f"   # the hand rule\n"
                  f"             rm {meta['path']}")
        print(f"Now: fly-games play --game {args.game} --policy fly")
        return 0
    finally:
        engine.close()


# ------------------------------------------------------------------------- probe
def _probe(args: argparse.Namespace) -> int:
    from fly_games.probe import ProbeBench

    bench = ProbeBench(steps=args.steps)
    events = []
    for raw in args.types or []:
        types = [t.strip() for t in raw.split(",") if t.strip()]
        if types:
            events.append({"types": types, "side": args.side, "volts": args.volts})
    # `--would-do` bare means every game; `--would-do mario doom` means those.
    games = [] if args.would_do is None else (list(args.would_do) or list(GAMES))
    result = bench.run(events, steps=args.steps, games=games)
    payload = result.public(include_field=False)
    label = ", ".join(f"{'+'.join(e['types'])}{'' if not e['side'] else ' ' + e['side']}"
                      for e in events) or "silence"
    print(f"stimulus: {label} at {args.volts:g} V for {result.steps} steps "
          f"({result.steps * result.dt:.2f}s of fly time)")
    print(f"  {result.spikes:,} spikes across the window, {result.latency_ms:.0f} ms of CPU")
    print(f"  {'command group':<14} {'rest':>8} {'driven':>8} {'delta':>8}")
    for name in payload["groups"]:
        rest = payload["baseline"].get(name, 0.0)
        driven = payload["command"][name]["rate"]
        delta = payload["delta"][name]
        bar = "#" * min(30, int(driven * 1.5))
        print(f"  {name:<14} {rest:>8.2f} {driven:>8.2f} {delta:>+8.2f}  {bar}")
    if payload["kinds"]:
        print("  most active classes: " + ", ".join(f"{n} ({c:,})" for n, c in payload["kinds"][:5]))
    for game_id, verdict in payload["would_do"].items():
        top = ", ".join(f"{k} {v:.2f}" for k, v in
                        sorted(verdict["probabilities"].items(), key=lambda kv: -kv[1]))
        print(f"  if it were playing {game_id}: {verdict['coarse']} "
              f"[{verdict['source']}]  {top}")
    return 0


# ---------------------------------------------------------------------- readouts
def _readouts(_args: argparse.Namespace) -> int:
    from fly_games.brain import FlyReadout
    from fly_games.catalog import GAMES
    from fly_games.readouts import readout_path

    for game_id, game in GAMES.items():
        path = readout_path(game_id)
        coarse = game.fly_coarse_actions()
        if not path.exists():
            print(f"{game_id:<8} not trained - using the zero-shot rule "
                  f"(fly-games train --game {game_id})")
            continue
        readout = FlyReadout.load(coarse, path)
        info = readout.info
        print(f"{game_id:<8} {path}  {info.kind}  components={info.components}  "
              f"lam={info.lam:g}  cv={info.cv_score:.4f}  n={info.n_samples}  {info.trained_at}")
        print(f"         decisions: {', '.join(coarse)}")
    return 0


# --------------------------------------------------------------------- brain cli
def _brain_info(_args: argparse.Namespace) -> int:
    try:
        from fly_games.brain import CHANNELS, shared_client

        client = shared_client()
        meta = client.meta
        connections = meta["n_connections"] or 0
        print(f"MaleCNS v1.0 - {meta['n_neurons']:,} neurons, {connections:,} connections")
        print(f"  device={meta['device']}  dt={meta['dt'] * 1000:.0f} ms  "
              f"seed={meta['seed']}  positions={client.cloud()['count']:,}")
        print("  descending command groups (the fly's output to its body):")
        for name, count in client.command_groups:
            print(f"    {name:<12} {count:>3} neuron{'s' if count != 1 else ''}")
        print("  sensory channels (what the game encoders drive):")
        for channel, types in CHANNELS.items():
            print(f"    {channel:<8} {'/'.join(types):<8} {dict(client.sensory_channels)[channel]:>4} neurons")
        print("  readouts:")
        _readouts(_args)
        return 0
    except Exception as exc:                                # noqa: BLE001
        print(f"Could not load the fly brain: {exc}")
        print("Run:  fly-games brain-download")
        return 1


def _brain_download(_args: argparse.Namespace) -> int:
    print("Downloading the MaleCNS v1.0 fly connectome data (one-time, ~260 MB)...")
    result = subprocess.run(["flybrain", "download"], capture_output=True, text=True, check=False)
    sys.stdout.write(result.stdout or result.stderr)
    return 0 if result.returncode == 0 else 1


COMMANDS = {
    "lab": _lab,
    "play": _play,
    "train": _train,
    "probe": _probe,
    "readouts": _readouts,
    "brain-info": _brain_info,
    "brain-download": _brain_download,
}


def run(args: argparse.Namespace) -> int:
    handler = COMMANDS.get(args.cmd)
    if handler is None:
        raise ValueError(f"Unknown command {args.cmd!r}")
    return handler(args)


def _game_flag(parser: argparse.ArgumentParser) -> None:
    """--game that only overrides the global when explicitly given."""
    parser.add_argument("--game", default=argparse.SUPPRESS, choices=GAMES,
                        help="which title to use")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fly-games",
        description="Play mario, kung-fu/spartan-x and doom with a real fruit-fly brain.",
    )
    parser.add_argument("--game", default="mario", choices=GAMES, help="which title to use")
    sub = parser.add_subparsers(dest="cmd", required=True)

    lab = sub.add_parser("lab", help="the observatory in a browser")
    lab.add_argument("--host", default=os.getenv("FLY_GAMES_HOST", "127.0.0.1"))
    lab.add_argument("--port", type=int, default=int(os.getenv("FLY_GAMES_PORT", "8000")))

    play = sub.add_parser("play", help="run a bounded episode without the UI")
    _game_flag(play)
    play.add_argument("--policy", choices=POLICIES, default="fly")
    play.add_argument("--decisions", type=int, default=30, help="how many fly decisions to make")
    play.add_argument("--seed", type=int, default=None,
                      help="emulator seed (default: the game's own); vary it to see run-to-run spread")
    play.add_argument("--readouts", default=None,
                      help="where to load readouts from (default: ./readouts; point it at an "
                           "empty directory to fly on the zero-shot rule)")
    play.add_argument("-v", "--verbose", action="store_true",
                      help="also print the sensory drives behind each decision")

    train = sub.add_parser("train", help="fit a game's readout (the only trained part)")
    _game_flag(train)
    train.add_argument("--episodes", type=int, default=8,
                       help="how many separate episodes to collect (each gets its "
                            "own seed, which is what makes the CV honest)")
    train.add_argument("--decisions", type=int, default=40,
                       help="decisions per episode (total = episodes x decisions)")

    probe = sub.add_parser("probe", help="stimulate cell types and watch the output neurons")
    _game_flag(probe)
    probe.add_argument("--types", action="append", metavar="CELLTYPE[,CELLTYPE]",
                       help="cell type to stimulate; repeat for more. e.g. --types LC4")
    probe.add_argument("--side", choices=("L", "R"), default=None, help="only that side")
    probe.add_argument("--volts", type=float, default=0.8)
    probe.add_argument("--steps", type=int, default=24, help="LIF steps in the decision window")
    probe.add_argument("--would-do", nargs="*", choices=GAMES, default=None,
                       metavar="GAME",
                       help="also decode the result with a game's readout; "
                            "no names means all three")

    sub.add_parser("readouts", help="which games have a trained readout")
    sub.add_parser("brain-info", help="connectome size, device, channels, command groups")
    sub.add_parser("brain-download", help="fetch the connectome data (~260 MB)")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Load .env before anything reads the environment, so `probe` and
    # `brain-info` honour FLY_DATA the same way `lab` and `play` do.
    from dotenv import load_dotenv

    load_dotenv()
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))