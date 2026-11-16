"""Evaluators answer the two questions MCTS asks about a position.

Given a batch of positions an evaluator returns, for each one,

* **priors** - a probability for every legal move ("where to look first"),
* a **value** in [-1, 1] for the player to move ("how good is this").

The real one wraps the neural network. The others need no training and exist
for tests, for debugging and as a reference point: with them MCTS is an ordinary
search algorithm, which shows how much of the playing strength comes from the
search and how much from the network.
"""

from __future__ import annotations

import random
from typing import Protocol

import numpy as np
import torch

from .game import Game, State
from .net import PolicyValueNet


class Evaluator(Protocol):
    def evaluate(self, states: list[State]) -> tuple[list[np.ndarray], np.ndarray]:
        """Priors aligned with ``game.legal_moves(state)``, and one value per state."""
        ...


class NetEvaluator:
    """Priors and values from the network, computed for a whole batch at once."""

    def __init__(self, game: Game, net: PolicyValueNet, device: torch.device | str = "cpu") -> None:
        self.game = game
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        #: number of positions evaluated so far
        self.positions = 0
        self.batches = 0

    def evaluate(self, states: list[State]) -> tuple[list[np.ndarray], np.ndarray]:
        game = self.game
        observations = np.stack([game.encode(state) for state in states])
        with torch.inference_mode():
            logits, values = self.net(torch.from_numpy(observations).to(self.device))
        logits = logits.float().cpu().numpy()
        values = values.float().cpu().numpy().astype(np.float64)
        self.positions += len(states)
        self.batches += 1

        priors = []
        for row, state in zip(logits, states):
            # Only the outputs of legal moves count; the rest are ignored, and
            # the softmax is taken over the legal moves alone.
            indices = [game.action_index(state, move) for move in game.legal_moves(state)]
            legal = row[indices].astype(np.float64)
            legal = np.exp(legal - legal.max())
            priors.append(legal / legal.sum())
        return priors, values


class UniformEvaluator:
    """Knows nothing: every move equally likely, every position even."""

    def __init__(self, game: Game) -> None:
        self.game = game

    def evaluate(self, states: list[State]) -> tuple[list[np.ndarray], np.ndarray]:
        priors = []
        for state in states:
            count = len(self.game.legal_moves(state))
            priors.append(np.full(count, 1.0 / count))
        return priors, np.zeros(len(states))


class HeuristicEvaluator(UniformEvaluator):
    """Uniform priors, value from the game's hand-written heuristic."""

    def evaluate(self, states: list[State]) -> tuple[list[np.ndarray], np.ndarray]:
        priors, _ = super().evaluate(states)
        return priors, np.array([self.game.heuristic(state) for state in states])


class RolloutEvaluator(UniformEvaluator):
    """Uniform priors, value from one random game played to the end.

    This is "classic" MCTS as it was before neural networks.
    """

    def __init__(self, game: Game, seed: int | None = None) -> None:
        super().__init__(game)
        self.rng = random.Random(seed)

    def evaluate(self, states: list[State]) -> tuple[list[np.ndarray], np.ndarray]:
        priors, _ = super().evaluate(states)
        return priors, np.array([self._rollout(state) for state in states])

    def _rollout(self, state: State) -> float:
        game = self.game
        player = game.current_player(state)
        while (value := game.terminal_value(state)) is None:
            state = game.next_state(state, self.rng.choice(game.legal_moves(state)))
        return value if game.current_player(state) == player else -value
