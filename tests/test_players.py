from __future__ import annotations

import pytest

from warcaby.draughts import BLACK, WHITE, Draughts
from warcaby.match import MatchResult, play_game, play_match
from warcaby.players import MinimaxPlayer, RandomPlayer

from .test_rules import pos

GAME = Draughts(8, max_plies=200)


def test_heuristic_is_zero_at_the_start_and_antisymmetric():
    assert GAME.heuristic(GAME.initial_state()) == pytest.approx(0.0)
    white_view = pos(w="c3 e3", b="f6", W="a1")
    black_view = pos(w="c3 e3", b="f6", W="a1", player=BLACK)
    assert GAME.heuristic(white_view) > 0
    assert GAME.heuristic(black_view) == pytest.approx(-GAME.heuristic(white_view))


def test_quiet_means_no_capture_pending():
    assert GAME.is_quiet(GAME.initial_state())
    assert not GAME.is_quiet(pos(w="d4", b="e5"))


def test_terminal_value_is_seen_by_the_player_to_move():
    lost = pos(w="a1", b="b2 c3")  # White to move and blocked
    assert GAME.terminal_value(lost) == -1.0
    assert GAME.outcome_for(lost, WHITE) == -1.0
    assert GAME.outcome_for(lost, BLACK) == 1.0
    assert GAME.terminal_value(GAME.initial_state()) is None


def test_minimax_does_not_hang_a_piece():
    # c3-d4 walks into e5xc3; every other move is safe
    state = pos(w="c3 g1", b="e5 h8")
    for seed in range(5):
        move = MinimaxPlayer(GAME, depth=2, seed=seed).choose(state)
        assert GAME.move_name(move) != "c3-d4"


def test_minimax_finds_a_sacrifice_that_wins_material():
    # c3-d4 gives up a man: e5xc3 is forced (c5 cannot take because e3 is
    # occupied, and the capture cannot go on over b2 because a1 is occupied).
    # Then b2 takes two with b2xd4xb6.
    state = pos(w="a1 b2 c3 e3", b="c5 e5 h8")
    for seed in range(5):
        move = MinimaxPlayer(GAME, depth=3, seed=seed).choose(state)
        assert GAME.move_name(move) == "c3-d4"


def test_alpha_beta_agrees_with_plain_minimax():
    """Pruning must not change the value of the position, only the work done."""

    def plain(state, depth):
        terminal = GAME.terminal_value(state)
        if terminal is not None:
            return terminal * (10.0 + depth)
        if depth <= 0 and GAME.is_quiet(state):
            return GAME.heuristic(state)
        return max(-plain(GAME.next_state(state, m), depth - 1) for m in GAME.legal_moves(state))

    rng_player = RandomPlayer(GAME, seed=3)
    state = GAME.initial_state()
    searcher = MinimaxPlayer(GAME, depth=4)
    for _ in range(25):
        if GAME.terminal_value(state) is not None:
            break
        pruned = searcher._search(state, 4, -float("inf"), float("inf"))
        assert pruned == pytest.approx(plain(state, 4))
        state = GAME.next_state(state, rng_player.choose(state))


def test_play_game_records_moves_and_result():
    record = play_game(GAME, RandomPlayer(GAME, seed=1), RandomPlayer(GAME, seed=2))
    assert record.result in (-1.0, 0.0, 1.0)
    assert GAME.terminal_value(record.final_state) is not None
    state = GAME.initial_state()
    for move in record.moves:
        assert move in GAME.legal_moves(state)
        state = GAME.next_state(state, move)
    assert state == record.final_state


def test_match_result_score():
    result = MatchResult(wins=3, draws=2, losses=5)
    assert result.games == 10
    assert result.score == pytest.approx(0.4)


def test_minimax_beats_random():
    result = play_match(GAME, MinimaxPlayer(GAME, depth=2, seed=0), RandomPlayer(GAME, seed=0), 10)
    assert result.wins >= 9, str(result)


@pytest.mark.slow
def test_deeper_minimax_beats_shallower():
    result = play_match(
        GAME, MinimaxPlayer(GAME, depth=4, seed=0), MinimaxPlayer(GAME, depth=1, seed=1), 10
    )
    assert result.score >= 0.7, str(result)
