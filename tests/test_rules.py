"""Rule-by-rule tests of the draughts engine.

Positions are given by square names (``a1`` is White's left corner, White moves
up the board). Dark squares are a1, c1, ... on odd ranks and b2, d2, ... on even
ranks.
"""

from __future__ import annotations

import random

import pytest

from warcaby.draughts.rules import BLACK, DRAW, WHITE, Move, Rules, State

from .reference import reference_moves

R = Rules(8)


def pos(w="", b="", W="", B="", player=WHITE, rules: Rules = R) -> State:  # noqa: N803
    """Position with white men ``w``, black men ``b``, white kings ``W``, black kings ``B``."""

    def mask(squares: str) -> int:
        return sum(1 << rules.square(name) for name in squares.split())

    masks = (mask(w), mask(b), mask(W), mask(B))
    union = masks[0] | masks[1] | masks[2] | masks[3]
    assert sum(m.bit_count() for m in masks) == union.bit_count(), "two pieces on one square"
    return State(men=(masks[0], masks[1]), kings=(masks[2], masks[3]), player=player)


def names(state: State, rules: Rules = R) -> set[str]:
    return {rules.move_name(move) for move in rules.legal_moves(state)}


def play(state: State, *move_names: str, rules: Rules = R) -> State:
    for name in move_names:
        state = rules.apply(state, rules.find_move(state, name))
    return state


# --- set-up ------------------------------------------------------------------------


def test_initial_position_8x8():
    state = R.initial_state()
    assert state.men[WHITE].bit_count() == 12
    assert state.men[BLACK].bit_count() == 12
    assert state.kings == (0, 0)
    assert state.player == WHITE
    assert names(state) == {"a3-b4", "c3-b4", "c3-d4", "e3-d4", "e3-f4", "g3-f4", "g3-h4"}


def test_initial_position_10x10():
    rules = Rules(10)
    state = rules.initial_state()
    assert state.men[WHITE].bit_count() == 20
    assert state.men[BLACK].bit_count() == 20
    assert len(rules.legal_moves(state)) == 9


def test_black_has_the_mirrored_first_moves():
    state = play(R.initial_state(), "a3-b4")
    assert state.player == BLACK
    assert names(state) == {"b6-a5", "b6-c5", "d6-c5", "d6-e5", "f6-e5", "f6-g5", "h6-g5"}


def test_board_size_must_be_even():
    with pytest.raises(ValueError):
        Rules(7)


def test_position_from_a_diagram():
    state = R.from_text(
        """
        . . . . . . . B
        . . . . . . . .
        . . . . . b . .
        . . . . . . . .
        . . . w . . . .
        . . . . . . . .
        . . . . . . . .
        W . . . . . . .
        """,
        player=BLACK,
    )
    assert state == pos(w="d4", b="f6", W="a1", B="h8", player=BLACK)


def test_diagram_rejects_pieces_on_light_squares():
    with pytest.raises(ValueError, match="light square"):
        R.from_text("w . . . . . . .\n" + ". . . . . . . .\n" * 7)


def test_render_round_trip():
    state = play(R.initial_state(), "c3-d4", "f6-e5")
    diagram = "\n".join(line[3:] for line in R.render(state).splitlines()[:8])
    assert R.from_text(diagram, player=state.player) == state


# --- quiet moves ---------------------------------------------------------------------


def test_man_moves_one_square_forward_only():
    assert names(pos(w="d4", b="h8")) == {"d4-c5", "d4-e5"}
    assert names(pos(w="a1", b="d6", player=BLACK)) == {"d6-c5", "d6-e5"}


def test_man_is_blocked_by_pieces_in_front():
    # d4: c5 is an own man, e5 an enemy that cannot be taken because f6 is occupied
    assert names(pos(w="d4 c5", b="e5 f6")) == {"c5-b6", "c5-d6"}


def test_king_flies_any_distance_until_blocked():
    state = pos(w="g7", W="d4", b="b8")
    king_moves = {name for name in names(state) if name.startswith("d4")}
    assert king_moves == {
        "d4-e5", "d4-f6",  # stopped by the own man on g7
        "d4-c5", "d4-b6", "d4-a7",
        "d4-e3", "d4-f2", "d4-g1",
        "d4-c3", "d4-b2", "d4-a1",
    }  # fmt: skip


def test_man_is_crowned_on_the_last_row():
    after = play(pos(w="c7", b="h8"), "c7-d8")
    assert R.piece_at(after, R.square("d8")) == "W"
    assert after.men[WHITE] == 0


def test_black_man_is_crowned_on_the_first_row():
    after = play(pos(w="a1", b="d2", player=BLACK), "d2-e1")
    assert R.piece_at(after, R.square("e1")) == "B"
    assert after.men[BLACK] == 0


# --- captures ------------------------------------------------------------------------


def test_capture_is_mandatory():
    state = pos(w="d4 b2", b="e5")
    # the man on b2 could move, but capturing takes precedence
    assert names(state) == {"d4xf6"}
    after = play(state, "d4xf6")
    assert after.men[BLACK] == 0
    assert R.piece_at(after, R.square("f6")) == "w"


def test_capture_needs_a_free_square_behind():
    # c5 can be taken (b6 is free), e5 cannot (f6 is occupied)
    assert names(pos(w="d4", b="c5 e5 f6")) == {"d4xb6"}


def test_man_captures_backwards():
    assert names(pos(w="d4", b="c3")) == {"d4xb2"}


def test_multi_capture_is_a_single_move():
    state = pos(w="c3", b="d4 f6")
    assert names(state) == {"c3xe5xg7"}
    after = play(state, "c3xe5xg7")
    assert after.men[BLACK] == 0
    assert after.player == BLACK
    assert after.ply == 1


def test_must_take_the_most_pieces():
    # taking b4 wins one man; taking d4 continues over f6, so only that is legal
    assert names(pos(w="c3", b="b4 d4 f6")) == {"c3xe5xg7"}


def test_majority_counts_pieces_not_their_rank():
    # one king is worth less than two men under the majority rule
    assert names(pos(w="c3", b="d4 f6", B="b4")) == {"c3xe5xg7"}


def test_majority_rule_applies_across_pieces():
    # h2 could take g3, but c3 can take two, so only c3 may capture
    assert names(pos(w="c3 h2", b="d4 f6 g3")) == {"c3xe5xg7"}


def test_equal_captures_leave_a_choice():
    assert names(pos(w="c3", b="b4 d4")) == {"c3xa5", "c3xe5"}


def test_king_captures_from_a_distance_and_chooses_where_to_land():
    assert names(pos(W="b2", b="e5")) == {"b2xf6", "b2xg7", "b2xh8"}


def test_king_cannot_jump_two_pieces_in_a_row():
    assert not any(move.captures for move in R.legal_moves(pos(W="b2", b="e5 f6")))


def test_king_cannot_jump_its_own_piece():
    assert not any(move.captures for move in R.legal_moves(pos(W="b2", w="d4", b="f6")))


def test_king_must_land_where_the_capture_continues():
    # After taking c3 the king may stop on d4, e5, f6, g7 or h8, but only from
    # e5 can it also take g3, and taking the most pieces is mandatory.
    assert names(pos(W="a1", b="c3 g3")) == {"a1xe5xh2"}


def test_king_capture_turns_a_corner():
    state = pos(W="c1", b="d2 d4")
    assert names(state) == {"c1xe3xc5", "c1xe3xb6", "c1xe3xa7"}
    assert play(state, "c1xe3xc5").men[BLACK] == 0


def test_captured_pieces_stay_on_the_board_until_the_move_ends():
    # The "Turkish strike". The king takes d4, e7 and c7 and arrives on b6.
    # From there f2 is on its diagonal, but d4 - already taken, not yet removed -
    # is in the way. Were d4 removed at once, a fourth capture would be forced.
    state = pos(W="a1", b="d4 e7 c7 f2")
    assert names(state) == {"a1xf6xd8xb6", "a1xf6xd8xa5"}
    after = play(state, "a1xf6xd8xb6")
    assert after.men[BLACK] == 1 << R.square("f2")


def test_man_passing_the_last_row_stays_a_man():
    # b6 takes c7, lands on d8 (the last row) and must go on over e7 to f6
    state = pos(w="b6", b="c7 e7")
    assert names(state) == {"b6xd8xf6"}
    after = play(state, "b6xd8xf6")
    assert R.piece_at(after, R.square("f6")) == "w"
    assert after.kings[WHITE] == 0


def test_man_ending_a_capture_on_the_last_row_is_crowned_but_stops():
    # From d8 a king could take g5 from a distance. The piece arrived as a man,
    # so the move ends on d8 and only then is it crowned.
    state = pos(w="b6", b="c7 g5")
    assert names(state) == {"b6xd8"}
    after = play(state, "b6xd8")
    assert R.piece_at(after, R.square("d8")) == "W"
    assert after.men[BLACK].bit_count() == 1


def test_capture_can_end_on_its_starting_square():
    state = pos(w="c3", b="b4 d4 b6 d6")
    moves = R.legal_moves(state)
    # clockwise and anticlockwise take the same men and end on the same square,
    # so they are one move
    assert len(moves) == 1
    assert len(moves[0].captures) == 4
    assert moves[0].start == moves[0].end == R.square("c3")
    after = R.apply(state, moves[0])
    assert after.men[BLACK] == 0
    assert R.piece_at(after, R.square("c3")) == "w"


def test_no_piece_is_ever_taken_twice():
    rng = random.Random(7)
    rules = Rules(8, max_plies=300)
    for _ in range(100):
        state = rules.initial_state()
        while rules.winner(state) is None:
            moves = rules.legal_moves(state)
            for move in moves:
                assert len(set(move.captures)) == len(move.captures)
            state = rules.apply(state, rng.choice(moves))


# --- draws and the end of the game -------------------------------------------------------

# In these positions the kings never share a diagonal, so no capture interferes.


def test_no_progress_counter_resets_on_a_man_move():
    state = play(pos(W="a1", w="h4", B="b8"), "a1-b2", "b8-a7")
    assert state.no_progress == 2
    assert len(state.history) == 2
    state = play(state, "h4-g5")
    assert state.no_progress == 0
    assert state.history == ()


def test_no_progress_counter_resets_on_a_capture():
    # the black king steps next to the white man and gets taken
    state = play(pos(W="a1", w="c5", B="b8"), "a1-b2", "b8-d6")
    assert state.no_progress == 2
    state = play(state, "c5xe7")
    assert state.no_progress == 0
    assert state.history == ()


def test_player_without_pieces_loses():
    state = pos(w="d4", b="e5")
    assert R.winner(state) is None
    after = play(state, "d4xf6")
    assert R.legal_moves(after) == []
    assert R.winner(after) == WHITE


def test_blocked_player_loses():
    # a1 cannot move (b2 is occupied) and cannot capture (c3 is occupied)
    state = pos(w="a1", b="b2 c3")
    assert R.legal_moves(state) == []
    assert R.winner(state) == BLACK


def test_threefold_repetition_is_a_draw():
    rules = Rules(8, no_progress_limit=1000)
    shuffle = ("a1-b2", "b8-a7", "b2-a1", "a7-b8")
    state = play(pos(W="a1", B="b8"), *shuffle, rules=rules)
    assert state.repetitions == 2
    assert rules.winner(state) is None
    state = play(state, *shuffle, rules=rules)
    assert state.repetitions == 3
    assert rules.winner(state) == DRAW


def test_repetition_needs_the_same_side_to_move():
    start = pos(W="a1", B="b8")
    # White returns to a1 in three moves, Black to b8 in two
    state = play(start, "a1-c3", "b8-a7", "c3-b2", "a7-b8", "b2-a1")
    assert (state.men, state.kings) == (start.men, start.kings)
    assert state.player == BLACK
    assert state.repetitions == 1


def test_no_progress_limit_is_a_draw():
    rules = Rules(8, no_progress_limit=6)
    state = play(pos(W="a1", B="b8"), "a1-b2", "b8-a7", "b2-c3", "a7-b8", "c3-d2", rules=rules)
    assert rules.winner(state) is None
    state = play(state, "b8-a7", rules=rules)
    assert state.no_progress == 6
    assert rules.winner(state) == DRAW


def test_default_no_progress_limits():
    assert Rules(8).no_progress_limit == 30
    assert Rules(10).no_progress_limit == 50


def test_max_plies_is_a_draw():
    rules = Rules(8, max_plies=2)
    state = play(rules.initial_state(), "c3-d4", rules=rules)
    assert rules.winner(state) is None
    state = play(state, "f6-g5", rules=rules)
    assert rules.winner(state) == DRAW


def test_losing_takes_precedence_over_a_draw():
    rules = Rules(8, max_plies=1)
    after = play(pos(w="d4", b="e5"), "d4xf6", rules=rules)
    assert rules.winner(after) == WHITE


# --- notation ----------------------------------------------------------------------------


def test_square_names():
    assert R.square_name(0) == "a1"
    assert R.square_name(31) == "h8"
    assert R.square("c3") == R.index[(2, 2)]
    with pytest.raises(ValueError):
        R.square("a2")


def test_find_move_accepts_an_abbreviated_capture():
    state = pos(w="c3", b="d4 f6")
    assert R.find_move(state, "c3xg7") == Move(
        (R.square("c3"), R.square("e5"), R.square("g7")), (R.square("d4"), R.square("f6"))
    )
    with pytest.raises(ValueError):
        R.find_move(state, "c3-b4")


# --- whole games ---------------------------------------------------------------------------


def as_dict(rules: Rules, state: State) -> dict:
    board = {}
    for sq, coords in enumerate(rules.coords):
        piece = rules.piece_at(state, sq)
        if piece:
            board[coords] = piece
    return board


def signatures(rules: Rules, state: State) -> set:
    return {
        (
            rules.coords[move.start],
            rules.coords[move.end],
            frozenset(rules.coords[sq] for sq in move.captures),
        )
        for move in rules.legal_moves(state)
    }


@pytest.mark.parametrize(("size", "games"), [(6, 150), (8, 150), (10, 40)])
def test_random_games_match_the_reference_generator(size, games):
    rules = Rules(size, max_plies=300)
    rng = random.Random(size)
    positions = captures = king_captures = 0
    for _ in range(games):
        state = rules.initial_state()
        while rules.winner(state) is None:
            moves = rules.legal_moves(state)
            expected = reference_moves(size, as_dict(rules, state), state.player)
            assert signatures(rules, state) == expected, rules.render(state)
            assert len(moves) == len(expected), "the engine returned duplicate moves"
            positions += 1
            move = rng.choice(moves)
            if move.captures:
                captures += 1
                king_captures += state.kings[state.player] >> move.start & 1
            state = rules.apply(state, move)
    # make sure the comparison really covered the interesting cases
    assert positions > 1000
    assert captures > 100
    assert king_captures > 10


@pytest.mark.parametrize("size", [6, 8, 10])
def test_random_games_keep_the_board_consistent(size):
    rules = Rules(size, max_plies=400)
    rng = random.Random(100 + size)
    results = {WHITE: 0, BLACK: 0, DRAW: 0}

    def count(state: State, player: int) -> int:
        return (state.men[player] | state.kings[player]).bit_count()

    for _ in range(60):
        state = rules.initial_state()
        while rules.winner(state) is None:
            before = state
            move = rng.choice(rules.legal_moves(state))
            state = rules.apply(state, move)

            masks = (state.men[0], state.men[1], state.kings[0], state.kings[1])
            union = masks[0] | masks[1] | masks[2] | masks[3]
            assert sum(mask.bit_count() for mask in masks) == union.bit_count()
            assert union < 1 << rules.num_squares
            # no man is left standing on its promotion row
            assert state.men[WHITE] & rules.promotion[WHITE] == 0
            assert state.men[BLACK] & rules.promotion[BLACK] == 0
            mover, other = before.player, 1 - before.player
            assert count(state, mover) == count(before, mover)
            assert count(state, other) == count(before, other) - len(move.captures)
            assert state.player == other
            assert state.ply == before.ply + 1
        results[rules.winner(state)] += 1
    assert results[WHITE] and results[BLACK], "random games should sometimes be decisive"


def perft(rules: Rules, state: State, depth: int) -> int:
    """Number of move sequences of length ``depth`` from ``state``."""
    if depth == 0:
        return 1
    return sum(
        perft(rules, rules.apply(state, move), depth - 1) for move in rules.legal_moves(state)
    )


def test_perft_10x10_matches_published_counts():
    # Published for international draughts by Ed Gilbert on the World Draughts
    # Forum (damforum.nl, "perft" thread). The 10x10 game runs on exactly the
    # same code as 8x8, so this is an outside check of the whole move generator.
    rules = Rules(10)
    counts = [perft(rules, rules.initial_state(), depth) for depth in range(1, 7)]
    assert counts == [9, 81, 658, 4265, 27117, 167140]


@pytest.mark.slow
def test_perft_10x10_depth_7_matches_published_count():
    rules = Rules(10)
    assert perft(rules, rules.initial_state(), 7) == 1049442


def test_perft_8x8_is_stable():
    # Depths 1-4 equal the well-known checkers counts (no rule difference shows
    # up that early). Deeper values are this engine's own, pinned so that an
    # accidental change in move generation is noticed.
    counts = [perft(R, R.initial_state(), depth) for depth in range(1, 7)]
    assert counts == [7, 49, 302, 1469, 7473, 37628]
