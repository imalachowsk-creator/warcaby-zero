from __future__ import annotations

import random

import numpy as np
import pytest

from warcaby.draughts import BLACK, WHITE, Draughts, Move, State

from .test_rules import pos

GAME = Draughts(8)
SQUARES = GAME.num_squares


def swap_sides(state: State) -> State:
    """The same position with colours swapped and the board turned by 180 degrees."""

    def turn(mask: int) -> int:
        return sum(1 << (SQUARES - 1 - sq) for sq in range(SQUARES) if mask >> sq & 1)

    swapped = State(
        men=(turn(state.men[1]), turn(state.men[0])),
        kings=(turn(state.kings[1]), turn(state.kings[0])),
        player=1 - state.player,
        no_progress=state.no_progress,
    )
    return swapped


def random_positions(count: int, seed: int = 0):
    rng = random.Random(seed)
    game = Draughts(8, max_plies=150)
    produced = 0
    while produced < count:
        state = game.initial_state()
        while game.terminal_value(state) is None and produced < count:
            if rng.random() < 0.2:
                yield state
                produced += 1
            state = game.next_state(state, rng.choice(game.legal_moves(state)))


def test_shapes():
    assert GAME.observation_shape == (6, 8, 8)
    assert GAME.action_size == 32 * 32
    observation = GAME.encode(GAME.initial_state())
    assert observation.shape == (6, 8, 8)
    assert observation.dtype == np.float32
    assert Draughts(10).observation_shape == (6, 10, 10)
    assert Draughts(10).action_size == 50 * 50


def test_initial_position_planes():
    observation = GAME.encode(GAME.initial_state())
    assert observation[0].sum() == 12  # my men
    assert observation[1].sum() == 0
    assert observation[2].sum() == 12  # their men
    assert observation[3].sum() == 0
    assert observation[4].sum() == 0  # no-progress clock
    assert observation[5].sum() == 0  # repetitions
    # my men are on the first three rows, on dark squares
    assert observation[0, :3].sum() == 12
    assert observation[0, 0, 0] == 1 and observation[0, 0, 1] == 0
    assert observation[2, 5:].sum() == 12


def test_pieces_land_on_their_squares():
    state = pos(w="c3", W="h8", b="b6", B="e1")
    observation = GAME.encode(state)
    assert observation[0, 2, 2] == 1  # c3 = row 2, column 2
    assert observation[1, 7, 7] == 1
    assert observation[2, 5, 1] == 1
    assert observation[3, 0, 4] == 1
    assert observation[:4].sum() == 4


def test_both_players_see_the_board_from_their_own_side():
    for state in random_positions(200):
        assert np.array_equal(GAME.encode(state), GAME.encode(swap_sides(state)))


def test_after_one_move_black_sees_what_white_saw():
    first = GAME.initial_state()
    second = GAME.next_state(first, GAME.rules.find_move(first, "c3-d4"))
    seen_by_black = GAME.encode(second)
    assert seen_by_black[0].sum() == 12 and seen_by_black[0, :3].sum() == 12
    # White's man on d4 appears, from Black's side, on row 4 (counted from Black)
    assert seen_by_black[2, 4, 4] == 1


def test_clock_and_repetition_planes():
    game = Draughts(8, no_progress_limit=10)
    state = pos(W="a1", B="b8")
    for name in ("a1-b2", "b8-a7", "b2-a1", "a7-b8"):
        state = game.next_state(state, game.rules.find_move(state, name))
    observation = game.encode(state)
    assert np.allclose(observation[4], 0.4)
    assert np.allclose(observation[5], 0.5)  # seen once before


def test_action_indices_are_mirrored_between_the_players():
    for state in random_positions(200, seed=1):
        mirrored = swap_sides(state)
        own = sorted(GAME.action_index(state, move) for move in GAME.legal_moves(state))
        other = sorted(GAME.action_index(mirrored, move) for move in GAME.legal_moves(mirrored))
        assert own == other
        assert all(0 <= index < GAME.action_size for index in own)


def test_action_index_is_from_times_squares_plus_to():
    state = GAME.initial_state()
    move = GAME.rules.find_move(state, "c3-d4")
    c3, d4 = GAME.rules.square("c3"), GAME.rules.square("d4")
    assert GAME.action_index(state, move) == c3 * 32 + d4
    # the mirrored move for Black, f6-e5, gets the same index
    black = GAME.next_state(state, GAME.rules.find_move(state, "a3-b4"))
    assert GAME.action_index(black, GAME.rules.find_move(black, "f6-e5")) == c3 * 32 + d4


def test_moves_sharing_an_index_are_rare():
    positions = collisions = 0
    for state in random_positions(2000, seed=2):
        indices = [GAME.action_index(state, move) for move in GAME.legal_moves(state)]
        positions += 1
        collisions += len(indices) != len(set(indices))
    assert collisions / positions < 0.01


# --- symmetries ----------------------------------------------------------------------


def test_positions_with_men_have_no_symmetries():
    state = GAME.initial_state()
    policy = np.zeros(GAME.action_size)
    assert len(GAME.symmetries(state, GAME.encode(state), policy)) == 1


def transform_state(state: State, transform) -> State:
    rules = GAME.rules

    def move_mask(mask: int) -> int:
        return sum(
            1 << rules.index[transform(*rules.coords[sq])]
            for sq in range(SQUARES)
            if mask >> sq & 1
        )

    return State(
        men=(move_mask(state.men[0]), move_mask(state.men[1])),
        kings=(move_mask(state.kings[0]), move_mask(state.kings[1])),
        player=state.player,
    )


TRANSFORMS = [
    lambda r, c: (r, c),
    lambda r, c: (c, r),
    lambda r, c: (7 - c, 7 - r),
    lambda r, c: (7 - r, 7 - c),
]


@pytest.mark.parametrize("player", [WHITE, BLACK])
def test_king_endgames_have_four_symmetries(player):
    state = pos(W="c3 h2", B="b8 e7", player=player)
    moves = GAME.legal_moves(state)
    policy = np.zeros(GAME.action_size)
    for weight, move in enumerate(moves, start=1):
        policy[GAME.action_index(state, move)] = weight
    variants = GAME.symmetries(state, GAME.encode(state), policy)
    assert len(variants) == 4

    rules = GAME.rules
    for (observation, permuted), transform in zip(variants, TRANSFORMS):
        # The symmetric position, built square by square, must encode to the
        # transformed observation...
        image = transform_state(state, transform)
        assert np.array_equal(observation, GAME.encode(image))
        # ...and each move must carry its weight to the transformed move.
        assert permuted.sum() == pytest.approx(policy.sum())
        for weight, move in enumerate(moves, start=1):
            start = rules.index[transform(*rules.coords[move.start])]
            end = rules.index[transform(*rules.coords[move.end])]
            image_move = Move((start, end))
            assert image_move in GAME.legal_moves(image)
            assert permuted[GAME.action_index(image, image_move)] == weight
