"""Turn a game into an animated GIF.

::

    python -m warcaby.render --model runs/laptop --opponent minimax --depth 4 --out docs/game.gif

plays one game between the network and the chosen opponent and saves it frame
by frame. A multi-capture is shown jump by jump, and - as the rules say - the
captured pieces disappear only when the whole move is finished.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .draughts import Draughts, Move, State
from .match import GameRecord, play_game
from .mcts import MCTSPlayer
from .players import MinimaxPlayer, RandomPlayer

LIGHT = "#ecdfc8"
DARK = "#a17c5b"
HIGHLIGHT = "#d9b44a"
WHITE_FILL, WHITE_EDGE = "#faf6ec", "#b9ac93"
BLACK_FILL, BLACK_EDGE = "#2a2623", "#0c0b0a"
CROWN = "#c9971c"
CAPTION_BG, CAPTION_INK, CAPTION_MUTED = "#fcfcfb", "#171614", "#7d7b75"

#: frames are drawn this many times larger and scaled down, which smooths the edges
SUPERSAMPLE = 3


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def draw_frame(
    game: Draughts,
    pieces: dict[int, str],
    highlights: tuple[int, ...] = (),
    title: str = "",
    subtitle: str = "",
    size: int = 440,
) -> Image.Image:
    """One picture of the board. ``pieces`` maps square numbers to w/W/b/B."""
    scale = SUPERSAMPLE
    n = game.size
    cell = size * scale // n
    board = cell * n
    caption = 54 * scale
    image = Image.new("RGB", (board, board + caption), CAPTION_BG)
    draw = ImageDraw.Draw(image)

    def corner(row: int, col: int) -> tuple[int, int]:
        return col * cell, (n - 1 - row) * cell

    for row in range(n):
        for col in range(n):
            x, y = corner(row, col)
            fill = DARK if (row + col) % 2 == 0 else LIGHT
            draw.rectangle([x, y, x + cell - 1, y + cell - 1], fill=fill)
    for sq in highlights:
        x, y = corner(*game.rules.coords[sq])
        draw.rectangle([x, y, x + cell - 1, y + cell - 1], fill=HIGHLIGHT)

    for sq, kind in pieces.items():
        x, y = corner(*game.rules.coords[sq])
        cx, cy = x + cell // 2, y + cell // 2
        white = kind in "wW"
        fill, edge = (WHITE_FILL, WHITE_EDGE) if white else (BLACK_FILL, BLACK_EDGE)
        rim = WHITE_EDGE if white else "#57504a"
        radius = int(cell * 0.38)
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=fill, outline=edge, width=max(2, cell // 28),
        )  # fmt: skip
        inner = int(cell * 0.27)
        draw.ellipse(
            [cx - inner, cy - inner, cx + inner, cy + inner], outline=rim, width=max(1, cell // 45)
        )
        if kind in "WB":
            u = cell / 100
            shape = [(-17, 10), (-20, -10), (-9, 0), (0, -14), (9, 0), (20, -10), (17, 10)]
            draw.polygon([(cx + px * u, cy + py * u) for px, py in shape], fill=CROWN)

    draw.text((14 * scale, board + 8 * scale), title, fill=CAPTION_INK, font=_font(17 * scale))
    draw.text(
        (14 * scale, board + 31 * scale), subtitle, fill=CAPTION_MUTED, font=_font(13 * scale)
    )
    return image.resize((size // n * n, (board + caption) // scale), Image.LANCZOS)


def _pieces(game: Draughts, state: State) -> dict[int, str]:
    rules = game.rules
    return {sq: kind for sq in range(rules.num_squares) if (kind := rules.piece_at(state, sq))}


def game_frames(
    game: Draughts, moves: list[Move], names: tuple[str, str], result_text: str, size: int = 440
) -> tuple[list[Image.Image], list[int]]:
    """Frames of a whole game and how long each is shown (milliseconds)."""
    subtitle = f"{names[0]} (white)  vs  {names[1]} (black)"
    state = game.initial_state()
    frames = [draw_frame(game, _pieces(game, state), (), "Start", subtitle, size)]
    durations = [900]
    for ply, move in enumerate(moves):
        number = f"{ply // 2 + 1}{'.' if ply % 2 == 0 else '...'} {game.move_name(move)}"
        before = _pieces(game, state)
        kind = before[move.start]
        # intermediate landings of a multi-capture: the taken pieces are still there
        for step in move.path[1:-1]:
            jumping = {sq: k for sq, k in before.items() if sq != move.start}
            jumping[step] = kind
            frames.append(draw_frame(game, jumping, (move.start, step), number, subtitle, size))
            durations.append(260)
        state = game.next_state(state, move)
        frames.append(
            draw_frame(game, _pieces(game, state), (move.start, move.end), number, subtitle, size)
        )
        durations.append(520)
    frames.append(draw_frame(game, _pieces(game, state), (), result_text, subtitle, size))
    durations.append(3500)
    return frames, durations


def save_gif(frames: list[Image.Image], durations: list[int], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # One shared palette keeps the file small and the colours steady between
    # frames. It is taken from several frames spread over the game, so that
    # colours appearing only later (highlights, crowns) are in it too.
    picks = [frames[round(i * (len(frames) - 1) / 5)] for i in range(6)]
    mosaic = Image.new("RGB", (sum(pick.width for pick in picks), picks[0].height))
    for index, pick in enumerate(picks):
        mosaic.paste(pick, (index * pick.width, 0))
    palette = mosaic.quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    quantized[0].save(
        path, save_all=True, append_images=quantized[1:], duration=durations, loop=0, optimize=True
    )
    return path


def describe_result(record: GameRecord, names: tuple[str, str]) -> str:
    if record.result > 0:
        return f"{names[0]} wins (white)"
    if record.result < 0:
        return f"{names[1]} wins (black)"
    return "Draw"


def main(argv: list[str] | None = None) -> None:
    from .app.session import find_newest_generation, load_model

    parser = argparse.ArgumentParser(description="Record a game of the network as a GIF.")
    parser.add_argument("--model", type=Path, help="run directory or gen_XXXX.pt (default: newest)")
    parser.add_argument("--opponent", choices=["minimax", "random", "self"], default="minimax")
    parser.add_argument("--depth", type=int, default=4, help="depth of the minimax opponent")
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--colour", choices=["white", "black"], default="white",
                        help="the side the network plays")  # fmt: skip
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("docs/game.gif"))
    args = parser.parse_args(argv)

    model = load_model(args.model or find_newest_generation(Path("runs")))
    game = Draughts(model.game.size, max_plies=200)
    network = MCTSPlayer(game, model.evaluator, args.simulations, seed=args.seed, name=model.label)
    if args.opponent == "minimax":
        opponent = MinimaxPlayer(game, args.depth, seed=args.seed)
    elif args.opponent == "random":
        opponent = RandomPlayer(game, seed=args.seed)
    else:
        opponent = MCTSPlayer(
            game, model.evaluator, args.simulations, seed=args.seed + 1, name=model.label
        )
    players = (network, opponent) if args.colour == "white" else (opponent, network)
    names = (players[0].name, players[1].name)
    record = play_game(game, *players)
    frames, durations = game_frames(game, record.moves, names, describe_result(record, names))
    path = save_gif(frames, durations, args.out)
    print(f"{describe_result(record, names)} in {len(record.moves)} plies -> {path}")


if __name__ == "__main__":
    main()
