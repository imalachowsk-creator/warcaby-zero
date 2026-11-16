"""Playing whole games between two ``Player`` objects."""

from __future__ import annotations

from dataclasses import dataclass, field

from .game import Game, Move, State
from .players import Player


@dataclass
class GameRecord:
    """A finished game: the moves and the result seen by player 0 (+1, 0, -1)."""

    moves: list[Move] = field(default_factory=list)
    result: float = 0.0
    final_state: State | None = None


def play_game(game: Game, first: Player, second: Player) -> GameRecord:
    """One game; ``first`` plays as player 0 (White in draughts)."""
    players = (first, second)
    state = game.initial_state()
    record = GameRecord()
    while (value := game.terminal_value(state)) is None:
        move = players[game.current_player(state)].choose(state)
        record.moves.append(move)
        state = game.next_state(state, move)
    record.result = value if game.current_player(state) == 0 else -value
    record.final_state = state
    return record


@dataclass
class MatchResult:
    """Wins, draws and losses of the first player of a match."""

    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """Share of the points won, a draw counting as half: 0..1."""
        return (self.wins + 0.5 * self.draws) / max(self.games, 1)

    def add(self, result: float) -> None:
        if result > 0:
            self.wins += 1
        elif result < 0:
            self.losses += 1
        else:
            self.draws += 1

    def __str__(self) -> str:
        return f"+{self.wins} ={self.draws} -{self.losses} ({self.score:.0%})"


def play_match(game: Game, player: Player, opponent: Player, games: int) -> MatchResult:
    """``games`` games with colours alternating; the result is from ``player``'s side."""
    result = MatchResult()
    for number in range(games):
        if number % 2 == 0:
            result.add(play_game(game, player, opponent).result)
        else:
            result.add(-play_game(game, opponent, player).result)
    return result
