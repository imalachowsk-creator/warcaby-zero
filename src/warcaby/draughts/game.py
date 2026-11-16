"""Draughts as a ``Game``: the rules plus the encoding a neural network sees.

**Board encoding.** The network always looks at the board from the side of the
player to move: for Black the board is rotated by 180 degrees and the colours are
swapped, so "my men move up" holds in every input. This is the one true symmetry
of draughts, and using it means the network learns each pattern once instead of
once per colour. The input has 6 planes of ``size x size``:

0. my men, 1. my kings, 2. opponent's men, 3. opponent's kings,
4. how close the game is to a no-progress draw (0..1, constant over the plane),
5. how often this position has occurred before (0, 0.5, 1).

Planes 4 and 5 are there because the rules depend on them: without them two
inputs that look the same could have different values.

**Move encoding.** A whole move - also a multi-capture - is a single action,
identified by the square it starts on and the square it ends on:
``index = from * squares + to`` (32 * 32 = 1024 outputs on 8x8). Two different
captures can share both squares; they then share one output of the network and
the search, which has one branch per legal move, tells them apart.
"""

from __future__ import annotations

import numpy as np

from ..game import Game
from .rules import DRAW, WHITE, Move, Rules, State, _bits

NUM_PLANES = 6
MAN_VALUE = 1.0
KING_VALUE = 3.0


class Draughts(Game):
    def __init__(
        self,
        size: int = 8,
        no_progress_limit: int | None = None,
        max_plies: int | None = None,
    ) -> None:
        self.rules = Rules(size, no_progress_limit=no_progress_limit, max_plies=max_plies)
        self.size = size
        self.name = f"draughts{size}x{size}"
        self.num_squares = self.rules.num_squares
        self.observation_shape = (NUM_PLANES, size, size)
        self.action_size = self.num_squares * self.num_squares

        last = self.num_squares - 1
        cells = [row * size + col for row, col in self.rules.coords]
        # position of each square inside a flattened plane, per player to move
        self._cells = (cells, [cells[last - sq] for sq in range(self.num_squares)])
        self._rows = [row for row, _ in self.rules.coords]
        self._policy_permutations = self._build_policy_permutations()

    # --- rules ---------------------------------------------------------------------

    def initial_state(self) -> State:
        return self.rules.initial_state()

    def current_player(self, state: State) -> int:
        return state.player

    def legal_moves(self, state: State) -> list[Move]:
        return self.rules.legal_moves(state)

    def next_state(self, state: State, move: Move) -> State:
        return self.rules.apply(state, move)

    def terminal_value(self, state: State) -> float | None:
        winner = self.rules.winner(state)
        if winner is None:
            return None
        if winner == DRAW:
            return 0.0
        return 1.0 if winner == state.player else -1.0

    def is_quiet(self, state: State) -> bool:
        moves = self.rules.legal_moves(state)
        return not moves or not moves[0].captures

    def heuristic(self, state: State) -> float:
        """Material balance with a small bonus for advanced men, in (-1, 1)."""
        me, other = state.player, 1 - state.player
        mine = self._material(state, me)
        theirs = self._material(state, other)
        return 0.9 * (mine - theirs) / max(mine + theirs, 1e-9)

    def _material(self, state: State, player: int) -> float:
        men = state.men[player]
        total = MAN_VALUE * men.bit_count() + KING_VALUE * state.kings[player].bit_count()
        rows = self._rows
        last_row = self.size - 1
        for sq in _bits(men):
            advanced = rows[sq] if player == WHITE else last_row - rows[sq]
            total += 0.03 * advanced
        return total

    # --- encoding ------------------------------------------------------------------

    def encode(self, state: State) -> np.ndarray:
        me = state.player
        area = self.size * self.size
        flat = np.zeros(NUM_PLANES * area, dtype=np.float32)
        cells = self._cells[me]
        for plane, mask in enumerate(
            (state.men[me], state.kings[me], state.men[1 - me], state.kings[1 - me])
        ):
            offset = plane * area
            for sq in _bits(mask):
                flat[offset + cells[sq]] = 1.0
        flat[4 * area : 5 * area] = min(state.no_progress / self.rules.no_progress_limit, 1.0)
        flat[5 * area :] = min(state.repetitions - 1, 2) / 2.0
        return flat.reshape(self.observation_shape)

    def canonical_square(self, state: State, sq: int) -> int:
        """Square number as the player to move sees it."""
        return sq if state.player == WHITE else self.num_squares - 1 - sq

    def action_index(self, state: State, move: Move) -> int:
        start, end = move.path[0], move.path[-1]
        if state.player != WHITE:
            last = self.num_squares - 1
            start, end = last - start, last - end
        return start * self.num_squares + end

    def symmetries(
        self, state: State, observation: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Draughts has almost no symmetries: a mirror image puts the pieces on
        light squares, and turning the board makes men walk backwards. Only when
        no men are left do the directions stop mattering; then the two diagonal
        reflections and the half-turn give equivalent positions."""
        if state.men[0] or state.men[1]:
            return [(observation, policy)]
        transposed = observation.transpose(0, 2, 1)
        boards = (
            observation,
            transposed,
            transposed[:, ::-1, ::-1],
            observation[:, ::-1, ::-1],
        )
        return [
            (np.ascontiguousarray(board), self._permute_policy(policy, index))
            for index, board in enumerate(boards)
        ]

    def _build_policy_permutations(self) -> list[np.ndarray]:
        size, squares = self.size, self.num_squares
        index = self.rules.index
        transforms = (
            lambda r, c: (r, c),
            lambda r, c: (c, r),
            lambda r, c: (size - 1 - c, size - 1 - r),
            lambda r, c: (size - 1 - r, size - 1 - c),
        )
        permutations = []
        for transform in transforms:
            square_map = np.array([index[transform(r, c)] for r, c in self.rules.coords])
            start, end = np.divmod(np.arange(squares * squares), squares)
            permutations.append(square_map[start] * squares + square_map[end])
        return permutations

    def _permute_policy(self, policy: np.ndarray, index: int) -> np.ndarray:
        permuted = np.empty_like(policy)
        permuted[self._policy_permutations[index]] = policy
        return permuted

    # --- convenience ---------------------------------------------------------------

    def render(self, state: State) -> str:
        return self.rules.render(state)

    def move_name(self, move: Move) -> str:
        return self.rules.move_name(move)
