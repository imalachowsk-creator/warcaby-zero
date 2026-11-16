"""One game played in the app, and what the AI thinks about it.

Nothing here knows about HTTP: the web server only translates requests into
calls on ``GameSession`` and sends ``to_dict()`` back.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..draughts import BLACK, DRAW, WHITE, Draughts, Move, State
from ..evaluators import NetEvaluator, UniformEvaluator
from ..mcts import MCTS, MCTSConfig, run_searches
from ..net import PolicyValueNet

COLOURS = ("white", "black")


@dataclass
class Model:
    """A network ready to play, or a stand-in when none has been trained yet."""

    game: Draughts
    evaluator: object
    label: str
    trained: bool


def find_newest_generation(runs_dir: Path) -> Path | None:
    files = sorted(runs_dir.glob("*/generations/gen_*.pt"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def load_model(path: Path | None, device: str = "cpu") -> Model:
    """Load a generation file, or the newest generation of a run directory."""
    if path is not None and path.is_dir():
        files = sorted((path / "generations").glob("gen_*.pt"))
        if not files:
            raise FileNotFoundError(f"no generations in {path}")
        path = files[-1]
    if path is None:
        game = Draughts(8)
        return Model(game, UniformEvaluator(game), "search without a network", trained=False)
    stored = torch.load(path, map_location="cpu", weights_only=True)
    net = PolicyValueNet.restore(stored["net"])
    game = Draughts(net.observation_shape[1])
    label = f"{path.parent.parent.name} gen {stored['generation']}"
    return Model(game, NetEvaluator(game, net, device), label, trained=True)


class GameSession:
    """A game between two seats, each a human or the AI."""

    def __init__(
        self,
        game_id: str,
        model: Model,
        white: str,
        black: str,
        ai_seats: tuple[bool, bool],
        simulations: int,
        seed: int | None = None,
    ) -> None:
        self.id = game_id
        self.model = model
        self.game = model.game
        self.names = (white, black)
        self.ai_seats = ai_seats
        self.simulations = simulations
        self.rng = np.random.default_rng(seed)
        self.lock = threading.Lock()

        self.state: State = self.game.initial_state()
        self.history: list[Move] = []
        self.last_move: dict | None = None
        self.ai_move: dict | None = None
        self.analysis: dict | None = None
        self.resigned: int | None = None
        #: set once the result has been written to the database
        self.saved = False

    # --- playing ---------------------------------------------------------------------

    @property
    def over(self) -> bool:
        return self.resigned is not None or self.game.terminal_value(self.state) is not None

    def play_human(self, move_id: int) -> None:
        if self.over:
            raise ValueError("the game is over")
        if self.ai_seats[self.state.player]:
            raise ValueError("it is the AI's turn")
        moves = self.game.legal_moves(self.state)
        if not 0 <= move_id < len(moves):
            raise ValueError("no such move")
        self.ai_move = None
        self._apply(moves[move_id])

    def play_ai_if_due(self) -> None:
        """Let the AI answer while it is its turn."""
        while not self.over and self.ai_seats[self.state.player]:
            report = self._search(self.state)
            best = max(report["moves"], key=lambda entry: (entry["visits"], entry["prior"]))
            move = self.game.legal_moves(self.state)[best["id"]]
            self._apply(move)
            self.ai_move = {**self.last_move, "chosen": best["id"], **report}

    def analyse(self) -> None:
        """What the AI thinks of the current position (for a human to move)."""
        self.analysis = None if self.over else self._search(self.state)

    def resign(self) -> None:
        if not self.over:
            # with two humans the player to move resigns; against the AI, the human
            if self.ai_seats[self.state.player]:
                self.resigned = 1 - self.state.player
            else:
                self.resigned = self.state.player
            self.analysis = None

    def _apply(self, move: Move) -> None:
        mover = self.state.player
        self.last_move = {**self._describe(move), "by": COLOURS[mover]}
        self.history.append(move)
        self.state = self.game.next_state(self.state, move)
        self.analysis = None

    def _search(self, state: State) -> dict:
        game = self.game
        moves = game.legal_moves(state)
        tree = MCTS(game, self.model.evaluator, MCTSConfig(), self.rng)
        tree.reset(state)
        run_searches([tree], max(self.simulations, 2))
        mover_value = tree.root_value()
        return {
            "simulations": self.simulations,
            "to_move": COLOURS[state.player],
            # the evaluation as White's expected result, -1..1
            "value_white": mover_value if state.player == WHITE else -mover_value,
            # what the network says before any search
            "network_value_white": (tree.root.value if state.player == WHITE else -tree.root.value),
            "moves": [
                {
                    "id": index,
                    **self._describe(moves[index]),
                    "prior": stats["prior"],
                    "visits": stats["visits"],
                    "share": stats["share"],
                    # for the side to move, -1..1; None if the search never tried it
                    "value": stats["value"],
                }
                for index, stats in enumerate(tree.move_stats())
            ],
        }

    # --- describing ------------------------------------------------------------------

    def _describe(self, move: Move) -> dict:
        coords = self.game.rules.coords
        return {
            "name": self.game.move_name(move),
            "path": [list(coords[sq]) for sq in move.path],
            "captures": [list(coords[sq]) for sq in move.captures],
        }

    def result(self) -> dict | None:
        """``None`` while the game goes on, else the winner and why."""
        if self.resigned is not None:
            return {"winner": COLOURS[1 - self.resigned], "reason": "resignation"}
        rules = self.game.rules
        winner = rules.winner(self.state)
        if winner is None:
            return None
        if winner != DRAW:
            return {"winner": COLOURS[winner], "reason": "no_moves"}
        if self.state.repetitions >= 3:
            reason = "repetition"
        elif self.state.no_progress >= rules.no_progress_limit:
            reason = "no_progress"
        else:
            reason = "length"
        return {"winner": "draw", "reason": reason}

    def to_dict(self) -> dict:
        rules = self.game.rules
        state = self.state
        pieces = []
        for sq, (row, col) in enumerate(rules.coords):
            kind = rules.piece_at(state, sq)
            if kind:
                pieces.append({"row": row, "col": col, "kind": kind})
        human_to_move = not self.over and not self.ai_seats[state.player]
        legal = []
        if human_to_move:
            legal = [
                {"id": index, **self._describe(move)}
                for index, move in enumerate(self.game.legal_moves(state))
            ]
        return {
            "game": self.id,
            "size": rules.size,
            "pieces": pieces,
            "turn": COLOURS[state.player],
            "ply": len(self.history),
            "players": {
                colour: {"name": self.names[seat], "ai": self.ai_seats[seat]}
                for seat, colour in enumerate(COLOURS)
            },
            "legal": legal,
            "result": self.result(),
            "history": [self.game.move_name(move) for move in self.history],
            "last_move": self.last_move,
            "ai_move": self.ai_move,
            "analysis": self.analysis,
            "simulations": self.simulations,
            "model": {"label": self.model.label, "trained": self.model.trained},
            "no_progress": {"count": state.no_progress, "limit": rules.no_progress_limit},
        }


__all__ = ["BLACK", "WHITE", "GameSession", "Model", "find_newest_generation", "load_model"]
