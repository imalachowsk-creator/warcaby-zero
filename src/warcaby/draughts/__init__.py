"""International draughts on an N x N board."""

from .game import Draughts
from .rules import BLACK, DRAW, WHITE, Move, Rules, State

__all__ = ["BLACK", "DRAW", "WHITE", "Draughts", "Move", "Rules", "State"]
