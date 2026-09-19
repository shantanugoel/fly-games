"""Game registry. Importing this module loads the bundled titles."""

from fly_games.games.doom.game import DoomGame
from fly_games.games.kungfu.game import KungFuGame
from fly_games.games.mario.game import MarioGame

GAMES = {
    game.meta.id: game
    for game in (MarioGame(), KungFuGame(), DoomGame())
}


def get_game(game_id: str):
    try:
        return GAMES[game_id]
    except KeyError as exc:
        known = ", ".join(GAMES)
        raise ValueError(f"Unknown game {game_id!r}. Bundled games: {known}") from exc
