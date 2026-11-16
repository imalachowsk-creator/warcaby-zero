from __future__ import annotations

import numpy as np
import pytest
import torch

from warcaby.draughts import BLACK, Draughts
from warcaby.evaluators import HeuristicEvaluator, NetEvaluator, RolloutEvaluator, UniformEvaluator
from warcaby.match import play_match
from warcaby.mcts import MCTS, MCTSConfig, MCTSPlayer, run_searches
from warcaby.net import NetConfig, PolicyValueNet
from warcaby.players import RandomPlayer

from .test_rules import pos

GAME = Draughts(8, max_plies=200)
TINY = NetConfig(blocks=1, channels=8, policy_channels=2, value_hidden=8)


def searched(state, evaluator, simulations, **config) -> MCTS:
    tree = MCTS(GAME, evaluator, MCTSConfig(**config), np.random.default_rng(0))
    tree.reset(state)
    run_searches([tree], simulations)
    return tree


def best_move_name(tree: MCTS) -> str:
    return GAME.move_name(tree.choose(temperature=0.0))


class CountingEvaluator(UniformEvaluator):
    def __init__(self, game):
        super().__init__(game)
        self.calls = 0
        self.positions = 0

    def evaluate(self, states):
        self.calls += 1
        self.positions += len(states)
        return super().evaluate(states)


# --- the network -----------------------------------------------------------------------


def test_network_output_shapes_and_ranges():
    net = PolicyValueNet.for_game(GAME, TINY)
    batch = torch.from_numpy(np.stack([GAME.encode(GAME.initial_state())] * 3))
    logits, values = net(batch)
    assert logits.shape == (3, GAME.action_size)
    assert values.shape == (3,)
    assert torch.all(values.abs() <= 1)


def test_network_export_restore_round_trip():
    net = PolicyValueNet.for_game(GAME, TINY).eval()
    copy = PolicyValueNet.restore(net.export()).eval()
    batch = torch.from_numpy(GAME.encode(GAME.initial_state())[None])
    with torch.inference_mode():
        for original, restored in zip(net(batch), copy(batch)):
            assert torch.allclose(original, restored)


def test_net_evaluator_gives_a_distribution_over_legal_moves():
    evaluator = NetEvaluator(GAME, PolicyValueNet.for_game(GAME, TINY))
    states = [GAME.initial_state(), pos(w="c3", b="b4 d4"), pos(W="a1", B="b8", player=BLACK)]
    priors, values = evaluator.evaluate(states)
    assert len(priors) == 3 and values.shape == (3,)
    for state, prior in zip(states, priors):
        assert len(prior) == len(GAME.legal_moves(state))
        assert prior.sum() == pytest.approx(1.0)
        assert np.all(prior > 0)
    assert np.all(np.abs(values) <= 1)


# --- the search ------------------------------------------------------------------------


def test_visit_counts_add_up():
    tree = searched(GAME.initial_state(), UniformEvaluator(GAME), 50)
    # the first simulation only expands the root
    assert tree.root.visits.sum() == 49
    assert tree.visit_distribution().sum() == pytest.approx(1.0)
    assert len(tree.move_stats()) == 7


# In this position a3-b4 wins on the spot: Black's only man has no move left.
# c3-d4 also wins by force (a5-b4 is Black's only move and a3xc5 takes the last
# man), while c3-b4 simply gives a man away.
WIN_IN_ONE = dict(w="a3 c3", b="a5")
WINNING_MOVES = {"a3-b4", "c3-d4"}


def test_search_finds_a_forced_win():
    tree = searched(pos(**WIN_IN_ONE), UniformEvaluator(GAME), 200)
    assert best_move_name(tree) in WINNING_MOVES
    stats = {GAME.move_name(s["move"]): s for s in tree.move_stats()}
    assert stats["a3-b4"]["value"] == pytest.approx(1.0)
    assert stats["c3-b4"]["visits"] < stats["a3-b4"]["visits"]
    assert tree.root_value() > 0.5


def test_value_is_negative_for_the_side_that_is_lost():
    # Black to move: whatever the man does, it is taken and Black has nothing left
    state = pos(w="e1 g1", b="f4", player=BLACK)
    tree = searched(state, UniformEvaluator(GAME), 400)
    assert tree.root_value() < -0.5


def test_search_avoids_hanging_a_piece():
    state = pos(w="c3 g1", b="e5 h8")
    tree = searched(state, HeuristicEvaluator(GAME), 300)
    assert best_move_name(tree) != "c3-d4"


def test_search_finds_the_sacrifice_that_wins_material():
    state = pos(w="a1 b2 c3 e3", b="c5 e5 h8")
    tree = searched(state, HeuristicEvaluator(GAME), 600)
    assert best_move_name(tree) == "c3-d4"


def test_rollout_search_finds_a_forced_win():
    tree = searched(pos(**WIN_IN_ONE), RolloutEvaluator(GAME, seed=0), 300)
    assert best_move_name(tree) in WINNING_MOVES
    assert tree.root_value() > 0.5


def test_priors_steer_the_search():
    """With few simulations the search mostly follows the evaluator's priors."""

    class Biased(UniformEvaluator):
        def evaluate(self, states):
            priors, values = super().evaluate(states)
            for prior in priors:
                if len(prior) > 2:
                    prior[:] = 0.02
                    prior[2] = 1 - 0.02 * (len(prior) - 1)
            return priors, values

    tree = searched(GAME.initial_state(), Biased(GAME), 30)
    assert int(np.argmax(tree.root.visits)) == 2


def test_root_noise_changes_only_the_priors_and_keeps_them_a_distribution():
    quiet = searched(GAME.initial_state(), UniformEvaluator(GAME), 1)
    noisy = searched(GAME.initial_state(), UniformEvaluator(GAME), 1, dirichlet_epsilon=0.25)
    assert np.allclose(quiet.root.priors, 1 / 7)
    assert not np.allclose(noisy.root.priors, 1 / 7)
    assert noisy.root.priors.sum() == pytest.approx(1.0)


def test_advance_keeps_the_subtree():
    tree = searched(GAME.initial_state(), UniformEvaluator(GAME), 200)
    move = tree.choose()
    index = tree.root.moves.index(move)
    child = tree.root.children[index]
    visits_below = child.visits.sum()
    assert visits_below > 0
    tree.advance(move)
    assert tree.root is child
    assert tree.root.state == GAME.next_state(GAME.initial_state(), move)
    run_searches([tree], 10)
    assert tree.root.visits.sum() == visits_below + 10


def test_advance_to_an_unvisited_move_starts_a_new_subtree():
    tree = searched(GAME.initial_state(), UniformEvaluator(GAME), 2)
    unvisited = tree.root.moves[int(np.argmin(tree.root.visits))]
    tree.advance(unvisited)
    assert not tree.root.expanded
    run_searches([tree], 5)
    assert tree.root.visits.sum() == 4


def test_temperature_one_samples_in_proportion_to_visits():
    tree = searched(GAME.initial_state(), UniformEvaluator(GAME), 100)
    picks = {GAME.move_name(tree.choose(temperature=1.0)) for _ in range(200)}
    assert len(picks) > 1
    assert len({GAME.move_name(tree.choose(temperature=0.0)) for _ in range(20)}) <= 2


def test_searches_are_batched_across_trees():
    evaluator = CountingEvaluator(GAME)
    trees = []
    for seed in range(8):
        tree = MCTS(GAME, evaluator, rng=np.random.default_rng(seed))
        tree.reset(GAME.initial_state())
        trees.append(tree)
    run_searches(trees, 25)
    assert evaluator.calls == 25
    assert evaluator.positions == 8 * 25
    assert all(tree.root.visits.sum() == 24 for tree in trees)


def test_trees_with_different_evaluators_are_batched_separately():
    first, second = CountingEvaluator(GAME), CountingEvaluator(GAME)
    trees = []
    for evaluator in (first, first, second):
        tree = MCTS(GAME, evaluator, rng=np.random.default_rng(0))
        tree.reset(GAME.initial_state())
        trees.append(tree)
    run_searches(trees, 10)
    assert (first.positions, second.positions) == (20, 10)


def test_search_through_finished_games_needs_no_evaluation():
    # Once the whole (tiny) game tree is known, simulations end in terminal
    # positions and cost no evaluator calls.
    evaluator = CountingEvaluator(GAME)
    tree = searched(pos(**WIN_IN_ONE), evaluator, 300)
    assert evaluator.positions < 300
    assert tree.root.visits.sum() == 299


def test_mcts_player_beats_random():
    player = MCTSPlayer(GAME, HeuristicEvaluator(GAME), simulations=60, seed=0)
    result = play_match(GAME, player, RandomPlayer(GAME, seed=0), 6)
    assert result.wins >= 5, str(result)


def test_search_works_with_an_untrained_network():
    evaluator = NetEvaluator(GAME, PolicyValueNet.for_game(GAME, TINY))
    tree = searched(GAME.initial_state(), evaluator, 40, dirichlet_epsilon=0.25)
    assert tree.root.visits.sum() == 39
    assert tree.choose(temperature=1.0) in GAME.legal_moves(GAME.initial_state())
