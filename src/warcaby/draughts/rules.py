"""Rules of international draughts on an N x N board.

With ``size=8`` this is Brazilian draughts, with ``size=10`` international (Polish)
draughts. The rules are identical, only the board differs:

* men move one square diagonally forward, kings any distance along a diagonal,
* capturing is mandatory, men capture backwards too,
* a multi-capture is one move, and only the sequences that take the largest
  number of pieces are legal,
* captured pieces leave the board only after the whole move, so a piece cannot
  be jumped twice and still blocks the way until the move is over,
* a man that merely passes the last row during a capture stays a man; it is
  crowned only when the move ends there,
* a player with no legal move loses.

Draws: the same position (with the same side to move) for the third time, a run
of ``no_progress_limit`` plies with only king moves and no capture, or the hard
cap ``max_plies`` (there to keep self-play games finite).

Only dark squares are used. They are numbered row by row starting from White's
left corner (a1 = 0), and the board is stored as four bit masks, one per piece
kind and colour. Python integers have no size limit, so the same code serves an
8x8 board (32 squares) and a 10x10 one (50 squares).
"""

from __future__ import annotations

from typing import NamedTuple

WHITE, BLACK = 0, 1
DRAW = -1

#: (row step, column step); directions 0 and 1 are "forward" for White, 2 and 3 for Black
DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_FORWARD = ((0, 1), (2, 3))

_FILES = "abcdefghijklmnopqrstuvwxyz"


class Move(NamedTuple):
    """One complete move: every square the piece stands on, and what it takes."""

    path: tuple[int, ...]
    captures: tuple[int, ...] = ()

    @property
    def start(self) -> int:
        return self.path[0]

    @property
    def end(self) -> int:
        return self.path[-1]


class State:
    """An immutable position. Create new ones with ``Rules.apply``."""

    __slots__ = ("men", "kings", "player", "no_progress", "history", "ply", "_moves")

    def __init__(
        self,
        men: tuple[int, int],
        kings: tuple[int, int],
        player: int = WHITE,
        no_progress: int = 0,
        history: tuple = (),
        ply: int = 0,
    ) -> None:
        self.men = men
        self.kings = kings
        self.player = player
        #: plies since the last capture or man move
        self.no_progress = no_progress
        #: keys of the earlier positions since the last capture or man move
        self.history = history
        self.ply = ply
        self._moves: list[Move] | None = None

    @property
    def key(self) -> tuple[int, int, int, int, int]:
        """What makes two positions "the same" for the repetition rule."""
        return (self.men[0], self.men[1], self.kings[0], self.kings[1], self.player)

    @property
    def repetitions(self) -> int:
        """How many times this position has occurred, counting now."""
        return 1 + self.history.count(self.key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, State) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return f"State(men={self.men}, kings={self.kings}, player={self.player}, ply={self.ply})"


def _bits(mask: int):
    """Indices of the set bits of ``mask``, lowest first."""
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


class Rules:
    """Move generation and game results for one board size."""

    def __init__(
        self,
        size: int = 8,
        no_progress_limit: int | None = None,
        max_plies: int | None = None,
    ) -> None:
        if size < 4 or size % 2:
            raise ValueError("board size must be an even number, at least 4")
        self.size = size
        self.half = size // 2
        self.num_squares = size * self.half
        # 15 moves per side on 8x8, 25 per side on 10x10, like the official rules
        self.no_progress_limit = (
            no_progress_limit if no_progress_limit is not None else (50 if size >= 10 else 30)
        )
        self.max_plies = max_plies

        self.coords: list[tuple[int, int]] = [
            (row, col) for row in range(size) for col in range(size) if (row + col) % 2 == 0
        ]
        self.index: dict[tuple[int, int], int] = {rc: sq for sq, rc in enumerate(self.coords)}

        # rays[sq][d]: the squares met when walking from sq in direction d
        self.rays: list[tuple[tuple[int, ...], ...]] = []
        for row, col in self.coords:
            per_direction = []
            for d_row, d_col in DIRECTIONS:
                ray = []
                r, c = row + d_row, col + d_col
                while 0 <= r < size and 0 <= c < size:
                    ray.append(self.index[(r, c)])
                    r, c = r + d_row, c + d_col
                per_direction.append(tuple(ray))
            self.rays.append(tuple(per_direction))

        def row_mask(row: int) -> int:
            return sum(1 << sq for sq, (r, _) in enumerate(self.coords) if r == row)

        #: squares on which a man of each colour is crowned
        self.promotion = (row_mask(size - 1), row_mask(0))
        rows_per_side = self.half - 1
        self._initial_men = (
            sum(row_mask(r) for r in range(rows_per_side)),
            sum(row_mask(r) for r in range(size - rows_per_side, size)),
        )

    # --- positions ---------------------------------------------------------------

    def initial_state(self) -> State:
        return State(men=self._initial_men, kings=(0, 0))

    def from_text(self, diagram: str, player: int = WHITE) -> State:
        """Build a position from a diagram, top row first.

        ``w``/``b`` are men, ``W``/``B`` kings, ``.`` (or ``-``) an empty square.
        Spaces are ignored, so rows can be written with gaps for readability.
        """
        rows = [line.replace(" ", "") for line in diagram.strip().splitlines()]
        rows = [line for line in rows if line]
        if len(rows) != self.size or any(len(line) != self.size for line in rows):
            raise ValueError(f"expected {self.size} rows of {self.size} squares")
        men, kings = [0, 0], [0, 0]
        for line_number, line in enumerate(rows):
            row = self.size - 1 - line_number
            for col, char in enumerate(line):
                if char in ".-":
                    continue
                if char not in "wWbB":
                    raise ValueError(f"unknown piece {char!r}")
                if (row, col) not in self.index:
                    raise ValueError(f"piece on a light square: {self.square_name_rc(row, col)}")
                bit = 1 << self.index[(row, col)]
                colour = WHITE if char in "wW" else BLACK
                (kings if char.isupper() else men)[colour] |= bit
        return State(men=(men[0], men[1]), kings=(kings[0], kings[1]), player=player)

    def render(self, state: State) -> str:
        lines = []
        for row in range(self.size - 1, -1, -1):
            cells = []
            for col in range(self.size):
                sq = self.index.get((row, col))
                cells.append("." if sq is None else self.piece_at(state, sq) or ".")
            lines.append(f"{row + 1:>2} " + " ".join(cells))
        lines.append("   " + " ".join(_FILES[: self.size]))
        side = "white" if state.player == WHITE else "black"
        lines.append(f"   {side} to move, ply {state.ply}")
        return "\n".join(lines)

    def piece_at(self, state: State, sq: int) -> str | None:
        bit = 1 << sq
        if state.men[WHITE] & bit:
            return "w"
        if state.men[BLACK] & bit:
            return "b"
        if state.kings[WHITE] & bit:
            return "W"
        if state.kings[BLACK] & bit:
            return "B"
        return None

    # --- notation ----------------------------------------------------------------

    def square_name_rc(self, row: int, col: int) -> str:
        return f"{_FILES[col]}{row + 1}"

    def square_name(self, sq: int) -> str:
        return self.square_name_rc(*self.coords[sq])

    def square(self, name: str) -> int:
        """Square number from a name such as ``"c3"``."""
        col, row = _FILES.index(name[0]), int(name[1:]) - 1
        try:
            return self.index[(row, col)]
        except KeyError:
            raise ValueError(f"{name} is not a dark square") from None

    def move_name(self, move: Move) -> str:
        """``c3-d4`` for a quiet move, ``c3xe5xg7`` for a capture."""
        separator = "x" if move.captures else "-"
        return separator.join(self.square_name(sq) for sq in move.path)

    def find_move(self, state: State, name: str) -> Move:
        """The legal move written as ``name``; the capture path may be abbreviated
        to its first and last square when that is unambiguous."""
        moves = self.legal_moves(state)
        exact = [m for m in moves if self.move_name(m) == name]
        if exact:
            return exact[0]
        squares = name.replace("x", "-").split("-")
        ends = (self.square(squares[0]), self.square(squares[-1]))
        matching = [m for m in moves if (m.start, m.end) == ends]
        if len(matching) != 1:
            raise ValueError(f"{name}: {len(matching)} legal moves match")
        return matching[0]

    # --- move generation -----------------------------------------------------------

    def legal_moves(self, state: State) -> list[Move]:
        if state._moves is None:
            state._moves = self._generate(state)
        return state._moves

    def _generate(self, state: State) -> list[Move]:
        me = state.player
        my_men, my_kings = state.men[me], state.kings[me]
        opponent = state.men[1 - me] | state.kings[1 - me]
        occupied = my_men | my_kings | opponent

        # Captures first: if there is one, quiet moves are not allowed at all.
        sequences: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for sq in _bits(my_men):
            self._man_jumps(sq, occupied ^ (1 << sq), opponent, 0, (sq,), (), sequences)
        for sq in _bits(my_kings):
            self._king_jumps(sq, occupied ^ (1 << sq), opponent, 0, (sq,), (), sequences)

        if sequences:
            most = max(len(taken) for _, taken in sequences)
            moves = []
            seen = set()
            for path, taken in sequences:
                if len(taken) != most:
                    continue
                # Different routes that start and end on the same squares and take
                # the same pieces lead to the same position: keep one of them.
                signature = (path[0], path[-1], frozenset(taken))
                if signature not in seen:
                    seen.add(signature)
                    moves.append(Move(path, taken))
            return moves

        moves = []
        rays = self.rays
        forward = _FORWARD[me]
        for sq in _bits(my_men):
            for direction in forward:
                ray = rays[sq][direction]
                if ray and not (occupied >> ray[0]) & 1:
                    moves.append(Move((sq, ray[0])))
        for sq in _bits(my_kings):
            for ray in rays[sq]:
                for target in ray:
                    if (occupied >> target) & 1:
                        break
                    moves.append(Move((sq, target)))
        return moves

    def _man_jumps(self, sq, occupied, opponent, taken_mask, path, taken, out) -> None:
        """Depth-first search of a man's capture sequences from ``sq``.

        ``occupied`` has the capturing piece's start square cleared (it may pass
        over or return to it) but still contains the pieces taken so far.
        """
        extended = False
        for ray in self.rays[sq]:
            if len(ray) < 2:
                continue
            over, landing = ray[0], ray[1]
            bit = 1 << over
            if opponent & bit and not taken_mask & bit and not (occupied >> landing) & 1:
                extended = True
                self._man_jumps(
                    landing, occupied, opponent, taken_mask | bit,
                    path + (landing,), taken + (over,), out,
                )  # fmt: skip
        if not extended and taken:
            out.append((path, taken))

    def _king_jumps(self, sq, occupied, opponent, taken_mask, path, taken, out) -> None:
        """Like ``_man_jumps`` for a flying king: it may start the jump from a
        distance and land on any free square behind the captured piece."""
        extended = False
        for ray in self.rays[sq]:
            length = len(ray)
            i = 0
            while i < length and not (occupied >> ray[i]) & 1:
                i += 1
            if i >= length - 1:
                continue  # nothing on this diagonal, or no room behind the piece
            over = ray[i]
            bit = 1 << over
            if not opponent & bit or taken_mask & bit:
                continue  # own piece, or a piece already taken in this move
            j = i + 1
            while j < length and not (occupied >> ray[j]) & 1:
                extended = True
                self._king_jumps(
                    ray[j], occupied, opponent, taken_mask | bit,
                    path + (ray[j],), taken + (over,), out,
                )  # fmt: skip
                j += 1
        if not extended and taken:
            out.append((path, taken))

    # --- playing -----------------------------------------------------------------

    def apply(self, state: State, move: Move) -> State:
        me = state.player
        start_bit, end_bit = 1 << move.path[0], 1 << move.path[-1]
        men, kings = list(state.men), list(state.kings)

        man_moved = bool(men[me] & start_bit)
        if man_moved:
            men[me] ^= start_bit
            if end_bit & self.promotion[me]:
                kings[me] |= end_bit
            else:
                men[me] |= end_bit
        else:
            # remove first, then add: a king may finish a capture where it started
            kings[me] = (kings[me] ^ start_bit) | end_bit

        if move.captures:
            keep = ~sum(1 << sq for sq in move.captures)
            men[1 - me] &= keep
            kings[1 - me] &= keep

        irreversible = man_moved or bool(move.captures)
        return State(
            men=(men[0], men[1]),
            kings=(kings[0], kings[1]),
            player=1 - me,
            no_progress=0 if irreversible else state.no_progress + 1,
            history=() if irreversible else state.history + (state.key,),
            ply=state.ply + 1,
        )

    def winner(self, state: State) -> int | None:
        """``None`` while the game goes on, else ``WHITE``, ``BLACK`` or ``DRAW``."""
        if not self.legal_moves(state):
            return 1 - state.player
        if state.repetitions >= 3:
            return DRAW
        if state.no_progress >= self.no_progress_limit:
            return DRAW
        if self.max_plies is not None and state.ply >= self.max_plies:
            return DRAW
        return None
