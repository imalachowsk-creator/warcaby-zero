"""A slow, deliberately naive move generator used to cross-check the real one.

It shares no code and no data structures with ``warcaby.draughts.rules``: the
board is a dictionary of coordinates and every step is spelled out. If both
implementations agree on thousands of random positions, a bug would have to be
made twice in two different ways.

A move is described by what matters for the resulting position:
``(start, end, frozenset of captured squares)``, all as ``(row, col)``.
"""

from __future__ import annotations

DIAGONALS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _inside(size, pos):
    return 0 <= pos[0] < size and 0 <= pos[1] < size


def _is_enemy(piece, player):
    return piece.lower() == ("b" if player == 0 else "w")


def _is_own(piece, player):
    return piece.lower() == ("w" if player == 0 else "b")


def _capture_sequences(size, board, player, is_king, origin, pos, taken):
    """Every complete capture sequence continuing from ``pos``: ``(end, taken)``."""

    def free(square):
        return _inside(size, square) and (square not in board or square == origin)

    results = []
    row, col = pos
    for d_row, d_col in DIAGONALS:
        if not is_king:
            middle = (row + d_row, col + d_col)
            landing = (row + 2 * d_row, col + 2 * d_col)
            if (
                middle in board
                and middle != origin
                and _is_enemy(board[middle], player)
                and middle not in taken
                and free(landing)
            ):
                results += _capture_sequences(
                    size, board, player, is_king, origin, landing, taken | {middle}
                )
            continue

        distance = 1
        while free((row + d_row * distance, col + d_col * distance)):
            distance += 1
        victim = (row + d_row * distance, col + d_col * distance)
        if not _inside(size, victim):
            continue
        if not _is_enemy(board[victim], player) or victim in taken:
            continue
        distance += 1
        while free((row + d_row * distance, col + d_col * distance)):
            landing = (row + d_row * distance, col + d_col * distance)
            results += _capture_sequences(
                size, board, player, is_king, origin, landing, taken | {victim}
            )
            distance += 1

    if not results and taken:
        return [(pos, frozenset(taken))]
    return results


def reference_moves(size, board, player):
    """Set of ``(start, end, captured)`` for the legal moves of ``player``.

    ``board`` maps ``(row, col)`` to ``"w"``, ``"W"``, ``"b"`` or ``"B"``.
    """
    captures = []
    for pos, piece in board.items():
        if _is_own(piece, player):
            for end, taken in _capture_sequences(
                size, board, player, piece.isupper(), pos, pos, frozenset()
            ):
                captures.append((pos, end, taken))
    if captures:
        most = max(len(taken) for _, _, taken in captures)
        return {move for move in captures if len(move[2]) == most}

    moves = set()
    forward = 1 if player == 0 else -1
    for pos, piece in board.items():
        if not _is_own(piece, player):
            continue
        row, col = pos
        for d_row, d_col in DIAGONALS:
            if piece.isupper():
                distance = 1
                while True:
                    target = (row + d_row * distance, col + d_col * distance)
                    if not _inside(size, target) or target in board:
                        break
                    moves.add((pos, target, frozenset()))
                    distance += 1
            elif d_row == forward:
                target = (row + d_row, col + d_col)
                if _inside(size, target) and target not in board:
                    moves.add((pos, target, frozenset()))
    return moves
