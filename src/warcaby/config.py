"""All the knobs of a training run, with two ready-made settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

from .draughts import Draughts
from .game import Game
from .net import NetConfig


@dataclass
class Config:
    # --- game ---------------------------------------------------------------------
    board_size: int = 8
    #: games longer than this are scored as draws, so self-play never stalls
    max_plies: int = 200

    # --- network ------------------------------------------------------------------
    blocks: int = 6
    channels: int = 64

    # --- search -------------------------------------------------------------------
    simulations: int = 100
    c_puct: float = 1.5
    #: share of the root priors replaced by noise in self-play
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float = 1.0
    #: for this many plies self-play samples its move from the visit counts
    #: (variety); afterwards it plays the most visited move (quality)
    temperature_plies: int = 20

    # --- self-play ----------------------------------------------------------------
    games_per_iteration: int = 400
    #: games played side by side inside one worker, sharing network batches
    parallel_games: int = 64
    #: worker processes; 0 or 1 plays in the main process
    workers: int = 4
    #: device for playing ("auto" = GPU if there is one)
    play_device: str = "auto"

    # --- replay buffer --------------------------------------------------------------
    #: positions kept; older ones are dropped
    buffer_size: int = 300_000

    # --- training -----------------------------------------------------------------
    batch_size: int = 256
    #: how many times, on average, each new position is used in a gradient step
    #: before newer data arrives
    reuse: float = 4.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_device: str = "auto"

    # --- gating: the new network must beat the current best --------------------------
    arena_games: int = 40
    #: plies sampled from the visit counts so that the arena games differ
    arena_temperature_plies: int = 6
    #: share of the points the candidate needs to replace the best network
    accept_score: float = 0.55

    # --- progress checks against fixed opponents --------------------------------------
    #: every this many iterations (0 = never)
    eval_every: int = 5
    eval_games: int = 20
    eval_minimax_depths: list[int] = field(default_factory=lambda: [2, 4])

    seed: int = 0

    # ----------------------------------------------------------------------------------

    def make_game(self) -> Game:
        return Draughts(self.board_size, max_plies=self.max_plies)

    def net_config(self) -> NetConfig:
        return NetConfig(blocks=self.blocks, channels=self.channels)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> Config:
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})


def quick() -> Config:
    """A few minutes per iteration on a laptop: proves the whole loop works and
    that the network starts to learn. Not meant to produce a strong player."""
    return Config(
        blocks=2,
        channels=32,
        simulations=40,
        games_per_iteration=96,
        parallel_games=32,
        workers=4,
        play_device="cpu",
        buffer_size=40_000,
        batch_size=128,
        arena_games=20,
        eval_every=2,
        eval_games=12,
        eval_minimax_depths=[1, 2],
        temperature_plies=16,
    )


def kaggle() -> Config:
    """The real run: a free Kaggle GPU session (4 CPU cores, one or two T4s)."""
    return Config()


PRESETS = {"quick": quick, "kaggle": kaggle}
