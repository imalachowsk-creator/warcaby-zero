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


class ModelLibrary:
    """Every saved generation under a runs directory, loaded when first used.

    A model is named ``"<run>/<generation>"``, e.g. ``"laptop/12"``. The
    directory is read again on every call, so generations saved by a training
    run that is still going show up without restarting the app.
    """

    def __init__(
        self,
        runs_dir: Path,
        default_run: str | None = None,
        default_id: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.runs_dir = Path(runs_dir)
        self.default_run = default_run
        self._pinned = default_id
        self.device = device
        self._loaded: dict[tuple[str, float], Model] = {}
        self._fallback: Model | None = None

    @classmethod
    def for_path(cls, path: Path | None, runs_dir: Path, device: str = "cpu") -> ModelLibrary:
        """``path`` may be a generation file (always the default), a run directory
        (its newest generation is the default) or ``None`` (the newest of all)."""
        if path is None:
            return cls(runs_dir, device=device)
        path = Path(path)
        if path.is_dir():
            return cls(path.parent, default_run=path.name, device=device)
        run_dir = path.parent.parent
        generation = int(path.stem.split("_")[1])
        return cls(run_dir.parent, default_id=f"{run_dir.name}/{generation}", device=device)

    def _files(self) -> dict[str, Path]:
        files = {}
        for path in sorted(self.runs_dir.glob("*/generations/gen_*.pt")):
            run = path.parent.parent.name
            files[f"{run}/{int(path.stem.split('_')[1])}"] = path
        return files

    def entries(self) -> list[dict]:
        """All models, newest generation first within each run."""
        listed = []
        for model_id in self._files():
            run, generation = model_id.rsplit("/", 1)
            listed.append({"id": model_id, "run": run, "generation": int(generation)})
        return sorted(listed, key=lambda entry: (entry["run"], -entry["generation"]))

    def default_id(self) -> str | None:
        files = self._files()
        if self._pinned in files:
            return self._pinned
        if self.default_run:
            files = {k: v for k, v in files.items() if k.startswith(self.default_run + "/")}
        if not files:
            return None
        return max(files, key=lambda model_id: files[model_id].stat().st_mtime)

    def get(self, model_id: str | None = None) -> Model:
        """The model with this name; ``None`` means the default one."""
        files = self._files()
        model_id = model_id or self.default_id()
        if model_id is None:
            if self._fallback is None:
                self._fallback = load_model(None)
            return self._fallback
        if model_id not in files:
            raise KeyError(model_id)
        key = (model_id, files[model_id].stat().st_mtime)
        if key not in self._loaded:
            self._loaded[key] = load_model(files[model_id], self.device)
        return self._loaded[key]


class FixedLibrary:
    """A library with exactly one model (used by tests)."""

    def __init__(self, model: Model) -> None:
        self.model = model

    def entries(self) -> list[dict]:
        return []

    def default_id(self) -> str | None:
        return None

    def get(self, model_id: str | None = None) -> Model:
        if model_id is not None:
            raise KeyError(model_id)
        return self.model


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


__all__ = [
    "BLACK", "WHITE", "FixedLibrary", "GameSession", "Model", "ModelLibrary",
    "find_newest_generation", "load_model",
]  # fmt: skip
