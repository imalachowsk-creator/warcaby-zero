"""Monte Carlo Tree Search guided by an evaluator (AlphaZero style).

The idea in one paragraph: the search grows a tree of positions, one new
position per *simulation*. Each simulation walks down from the current position,
at every step choosing the move that looks best *so far*, until it reaches a
position not seen before. It asks the evaluator about that position (priors for
its moves, a value), adds it to the tree, and passes the value back up the path
so that every move on the way remembers "when I was tried, things turned out
this well". After many simulations the moves that kept turning out well have
been visited most, and the visit counts are the search's verdict.

"Looks best so far" is the PUCT rule. For each move ``a`` of a position::

    score(a) = Q(a) + c_puct * P(a) * sqrt(N_total) / (1 + N(a))

* ``Q(a)`` - the average value seen below ``a`` (exploitation: prefer what worked),
* ``P(a)`` - the evaluator's prior (prefer what the network suggests),
* ``N(a)`` - how often ``a`` was tried; the second term shrinks as a move gets
  visited, so the search does not get stuck on its first idea (exploration).

Values are always stored from the point of view of the player who is about to
move in that position, and the sign flips at every step up the tree, because a
position that is good for me is bad for my opponent.

The tree can be kept between moves: after a move is played, the subtree below
it becomes the new tree, so the thinking done so far is not thrown away.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .evaluators import Evaluator
from .game import Game, Move, State


@dataclass
class MCTSConfig:
    #: weight of the exploration term in PUCT
    c_puct: float = 1.5
    #: share of the root priors replaced by random noise (0 turns it off)
    dirichlet_epsilon: float = 0.0
    #: concentration of the noise for a position with ~10 moves; scaled by the move count
    dirichlet_alpha: float = 1.0


class Node:
    """A position in the tree, with statistics for each of its moves."""

    __slots__ = (
        "state",
        "terminal",
        "moves",
        "priors",
        "visits",
        "value_sums",
        "children",
        "value",
    )

    def __init__(self, state: State, terminal: float | None) -> None:
        self.state = state
        #: result for the player to move if the game is over here, else ``None``
        self.terminal = terminal
        #: legal moves; ``None`` until the evaluator has seen this position
        self.moves: list[Move] | None = None
        self.priors: np.ndarray | None = None
        self.visits: np.ndarray | None = None
        self.value_sums: np.ndarray | None = None
        self.children: list[Node | None] | None = None
        #: the evaluator's value for the player to move
        self.value = 0.0

    @property
    def expanded(self) -> bool:
        return self.moves is not None


class MCTS:
    """One search tree, for one game."""

    def __init__(
        self,
        game: Game,
        evaluator: Evaluator,
        config: MCTSConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.game = game
        self.evaluator = evaluator
        self.config = config or MCTSConfig()
        self.rng = rng or np.random.default_rng()
        self.root: Node | None = None
        self._pending: tuple[Node, list[tuple[Node, int]]] | None = None

    # --- managing the root ---------------------------------------------------------

    def reset(self, state: State) -> None:
        """Start a fresh tree at ``state``."""
        self.root = self._make_node(state)
        self._pending = None

    def advance(self, move: Move) -> None:
        """Play ``move`` at the root and keep the subtree below it."""
        root = self.root
        child = None
        if root.expanded:
            child = root.children[root.moves.index(move)]
        if child is None:
            child = self._make_node(self.game.next_state(root.state, move))
        self.root = child
        self._pending = None
        if child.expanded:
            self._add_root_noise()

    def _make_node(self, state: State) -> Node:
        return Node(state, self.game.terminal_value(state))

    def _add_root_noise(self) -> None:
        """Mix random noise into the root priors (self-play only).

        Without it the search would only ever try what the network already
        believes in, and a good move the network underrates would never be
        discovered. The noise guarantees every move gets looked at sometimes.
        """
        epsilon = self.config.dirichlet_epsilon
        root = self.root
        if epsilon <= 0 or len(root.moves) < 2:
            return
        alpha = self.config.dirichlet_alpha * min(1.0, 10.0 / len(root.moves))
        noise = self.rng.dirichlet(np.full(len(root.moves), alpha))
        root.priors = (1 - epsilon) * root.priors + epsilon * noise

    # --- one simulation, in two halves ---------------------------------------------
    #
    # ``select`` walks down the tree. If it ends in a position the evaluator has
    # not seen, it returns that position and waits; ``backup`` then finishes the
    # simulation. The split lets a caller collect the waiting positions of many
    # trees and send them to the network as one batch.

    def select(self) -> Node | None:
        """Walk down to a leaf. Returns the node that needs evaluating, or
        ``None`` when the walk ended in a finished game (already backed up)."""
        node = self.root
        path: list[tuple[Node, int]] = []
        while True:
            if node.terminal is not None:
                self._propagate(path, node.terminal)
                return None
            if node.moves is None:
                self._pending = (node, path)
                return node
            action = self._best_action(node)
            child = node.children[action]
            if child is None:
                child = self._make_node(self.game.next_state(node.state, node.moves[action]))
                node.children[action] = child
            path.append((node, action))
            node = child

    def _best_action(self, node: Node) -> int:
        visits = node.visits
        total = visits.sum()
        # A move never tried is assumed to be as good as the position looks so far.
        untried = (node.value + node.value_sums.sum()) / (1.0 + total)
        q = np.where(visits > 0, node.value_sums / np.maximum(visits, 1.0), untried)
        u = self.config.c_puct * node.priors * math.sqrt(total + 1.0) / (1.0 + visits)
        return int(np.argmax(q + u))

    def backup(self, priors: np.ndarray, value: float) -> None:
        """Finish the simulation started by ``select`` with the evaluator's answer."""
        node, path = self._pending
        self._pending = None
        count = len(priors)
        node.moves = self.game.legal_moves(node.state)
        node.priors = np.asarray(priors, dtype=np.float64)
        node.visits = np.zeros(count)
        node.value_sums = np.zeros(count)
        node.children = [None] * count
        node.value = float(value)
        if node is self.root:
            self._add_root_noise()
        self._propagate(path, float(value))

    @staticmethod
    def _propagate(path: list[tuple[Node, int]], value: float) -> None:
        """Pass a leaf's value up the path, flipping its sign at every step."""
        for node, action in reversed(path):
            value = -value
            node.visits[action] += 1
            node.value_sums[action] += value

    # --- reading the result ----------------------------------------------------------

    def visit_distribution(self) -> np.ndarray:
        """Share of the simulations spent on each root move (the training target)."""
        visits = self.root.visits
        total = visits.sum()
        if total == 0:
            return np.full(len(visits), 1.0 / len(visits))
        return visits / total

    def root_value(self) -> float:
        """The search's estimate of the root for the player to move."""
        root = self.root
        total = root.visits.sum()
        return float((root.value + root.value_sums.sum()) / (1.0 + total))

    def choose(self, temperature: float = 0.0) -> Move:
        """Pick a root move from the visit counts.

        ``temperature`` 0 plays the most visited move; 1 samples in proportion
        to the visits, which makes self-play games differ from each other.
        """
        root = self.root
        visits = root.visits
        if temperature <= 1e-6 or visits.sum() == 0:
            best = np.flatnonzero(visits == visits.max())
            return root.moves[int(self.rng.choice(best))]
        weights = visits ** (1.0 / temperature)
        return root.moves[int(self.rng.choice(len(visits), p=weights / weights.sum()))]

    def move_stats(self) -> list[dict]:
        """Per-move numbers for display: prior, visits and average value."""
        root = self.root
        total = max(root.visits.sum(), 1.0)
        stats = []
        for index, move in enumerate(root.moves):
            visits = root.visits[index]
            stats.append(
                {
                    "move": move,
                    "prior": float(root.priors[index]),
                    "visits": int(visits),
                    "share": float(visits / total),
                    "value": float(root.value_sums[index] / visits) if visits else None,
                }
            )
        return stats


def run_searches(trees: list[MCTS], simulations: int) -> None:
    """Run ``simulations`` simulations in every tree.

    The trees advance together: each selects a leaf, all leaves that belong to
    the same evaluator go to it in one batch, each tree backs up its answer.
    A GPU evaluates a batch of 64 positions hardly slower than one, so this is
    what makes playing many games at once cheap.
    """
    remaining = {id(tree): simulations for tree in trees}
    active = list(trees)
    while active:
        waiting: dict[int, list[MCTS]] = defaultdict(list)
        evaluators = {}
        still_active = []
        for tree in active:
            budget = remaining[id(tree)]
            leaf = None
            while budget > 0 and leaf is None:
                leaf = tree.select()
                budget -= 1
            remaining[id(tree)] = budget
            if leaf is not None:
                waiting[id(tree.evaluator)].append(tree)
                evaluators[id(tree.evaluator)] = tree.evaluator
                still_active.append(tree)
        for key, group in waiting.items():
            priors, values = evaluators[key].evaluate([tree._pending[0].state for tree in group])
            for tree, tree_priors, value in zip(group, priors, values):
                tree.backup(tree_priors, value)
        active = [tree for tree in still_active if remaining[id(tree)] > 0]


class MCTSPlayer:
    """A ``Player`` that searches from scratch on every move."""

    def __init__(
        self,
        game: Game,
        evaluator: Evaluator,
        simulations: int,
        config: MCTSConfig | None = None,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        self.game = game
        self.simulations = simulations
        self.tree = MCTS(game, evaluator, config, np.random.default_rng(seed))
        self.name = name or f"mcts-{simulations}"

    def choose(self, state: State) -> Move:
        moves = self.game.legal_moves(state)
        if len(moves) == 1:
            return moves[0]
        self.tree.reset(state)
        run_searches([self.tree], self.simulations)
        return self.tree.choose(temperature=0.0)
