"""The interface every game has to implement.

Search (MCTS, minimax), self-play, training and the app only ever talk to this
interface, never to draughts directly. Adding a 10x10 board, or a different game
altogether, means writing another ``Game`` and nothing else.

Assumptions baked into the interface:

* two players, numbered 0 and 1, who strictly alternate,
* zero-sum: one side's win is the other side's loss,
* states are immutable; ``next_state`` returns a new one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np

State = Any
Move = Hashable


class Game(ABC):
    """Rules of a game plus its encoding for a neural network."""

    #: short identifier, stored in checkpoints so a network is never loaded for the wrong game
    name: str
    #: shape of the network input, ``(planes, height, width)``
    observation_shape: tuple[int, int, int]
    #: number of policy outputs
    action_size: int

    # --- rules -----------------------------------------------------------------

    @abstractmethod
    def initial_state(self) -> State: ...

    @abstractmethod
    def current_player(self, state: State) -> int:
        """Player to move: 0 or 1."""

    @abstractmethod
    def legal_moves(self, state: State) -> Sequence[Move]:
        """Legal moves of the player to move. Empty when the game is over."""

    @abstractmethod
    def next_state(self, state: State, move: Move) -> State: ...

    @abstractmethod
    def terminal_value(self, state: State) -> float | None:
        """``None`` while the game goes on, otherwise the result seen by the player
        to move: +1 win, 0 draw, -1 loss."""

    # --- encoding for the network ------------------------------------------------

    @abstractmethod
    def encode(self, state: State) -> np.ndarray:
        """Network input, always from the point of view of the player to move."""

    @abstractmethod
    def action_index(self, state: State, move: Move) -> int:
        """Index of ``move`` in the policy vector, in the same point of view as ``encode``.

        Two legal moves may share an index; they then share the network's prior
        and the search tells them apart.
        """

    def symmetries(
        self, state: State, observation: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Equivalent ``(observation, policy)`` pairs used to augment training data.

        The default is "no symmetries": just the pair itself.
        """
        return [(observation, policy)]

    # --- convenience -------------------------------------------------------------

    @abstractmethod
    def render(self, state: State) -> str:
        """Human-readable board for the terminal and for test failures."""

    def outcome_for(self, state: State, player: int) -> float | None:
        """Result seen by ``player``; ``None`` while the game goes on."""
        value = self.terminal_value(state)
        if value is None:
            return None
        return value if self.current_player(state) == player else -value
