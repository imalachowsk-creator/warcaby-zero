"""Opponents that do not learn: a random mover and a classic minimax search.

They are the yardstick for the learning agent. Random is the floor. Minimax with
alpha-beta is the "traditional" way to play a board game: look a few moves ahead
and score the positions at the horizon with a hand-written rule (here: count the
material). Its strength is easy to dial with the search depth.
"""

from __future__ import annotations

import random
from typing import Protocol

from .game import Game, Move, State


class Player(Protocol):
    name: str

    def choose(self, state: State) -> Move: ...


class RandomPlayer:
    def __init__(self, game: Game, seed: int | None = None) -> None:
        self.game = game
        self.rng = random.Random(seed)
        self.name = "random"

    def choose(self, state: State) -> Move:
        return self.rng.choice(self.game.legal_moves(state))


class MinimaxPlayer:
    """Depth-limited negamax with alpha-beta pruning.

    *Negamax*: the game is zero-sum, so "my best result" is minus "the opponent's
    best result" - one function serves both sides.

    *Alpha-beta*: once a reply is found that already refutes a move, the remaining
    replies to it need not be searched. The result is exactly that of a full
    search, just found sooner.

    At the horizon the search does not stop while a capture is pending
    (``Game.is_quiet``); counting material in the middle of an exchange would give
    nonsense. Equal moves are broken at random so that games differ.
    """

    def __init__(self, game: Game, depth: int, seed: int | None = None) -> None:
        self.game = game
        self.depth = depth
        self.rng = random.Random(seed)
        self.name = f"minimax-{depth}"
        self.nodes = 0

    def choose(self, state: State) -> Move:
        game = self.game
        moves = list(game.legal_moves(state))
        if len(moves) == 1:
            return moves[0]
        self.rng.shuffle(moves)
        best_move, alpha = moves[0], -float("inf")
        for move in moves:
            value = -self._search(
                game.next_state(state, move), self.depth - 1, -float("inf"), -alpha
            )
            if value > alpha:
                alpha, best_move = value, move
        return best_move

    def _search(self, state: State, depth: int, alpha: float, beta: float) -> float:
        """Value of ``state`` for the player to move, within the window (alpha, beta)."""
        self.nodes += 1
        game = self.game
        terminal = game.terminal_value(state)
        if terminal is not None:
            # a win found sooner is worth slightly more, a loss later slightly less
            return terminal * (10.0 + depth)
        if depth <= 0 and game.is_quiet(state):
            return game.heuristic(state)
        best = -float("inf")
        for move in game.legal_moves(state):
            value = -self._search(game.next_state(state, move), depth - 1, -beta, -alpha)
            if value > best:
                best = value
                if value > alpha:
                    alpha = value
                    if alpha >= beta:
                        break
        return best
