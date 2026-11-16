from __future__ import annotations

from PIL import Image

from warcaby.draughts import Draughts
from warcaby.match import play_game
from warcaby.players import MinimaxPlayer, RandomPlayer
from warcaby.render import describe_result, draw_frame, game_frames, save_gif

from .test_rules import pos

GAME = Draughts(8, max_plies=80)


def test_frame_shows_pieces_on_their_squares():
    frame = draw_frame(GAME, {GAME.rules.square("a1"): "w", GAME.rules.square("h8"): "B"}, size=240)
    assert frame.size == (240, 240 + 54)
    cell = 30
    a1 = frame.getpixel((cell // 2, 240 - cell // 2))
    h8 = frame.getpixel((240 - cell // 2, cell // 2 + 6))  # just below the crown
    empty_dark = frame.getpixel((cell + cell // 2, 240 - cell - cell // 2))  # b2
    assert sum(a1) > 600  # a light piece
    assert sum(h8) < 200  # a dark piece
    assert 250 < sum(empty_dark) < 500  # the bare dark square


def test_multi_capture_gets_a_frame_per_jump():
    moves = GAME.legal_moves(pos(w="c3", b="d4 f6"))
    assert [GAME.move_name(m) for m in moves] == ["c3xe5xg7"]
    # start + one intermediate landing + the finished move + the result
    game = Draughts(8)
    game.initial_state = lambda: pos(w="c3", b="d4 f6")
    frames, durations = game_frames(game, moves, ("a", "b"), "a wins", size=160)
    assert len(frames) == len(durations) == 4


def test_a_whole_game_becomes_a_gif(tmp_path):
    players = (MinimaxPlayer(GAME, 2, seed=0), RandomPlayer(GAME, seed=0))
    names = tuple(player.name for player in players)
    record = play_game(GAME, *players)
    frames, durations = game_frames(
        GAME, record.moves, names, describe_result(record, names), size=160
    )
    path = save_gif(frames, durations, tmp_path / "out" / "game.gif")
    with Image.open(path) as gif:
        assert gif.n_frames == len(frames) >= len(record.moves) + 2
    assert describe_result(record, names) == "minimax-2 wins (white)"
