"""Playing many games at once, in one process or in several.

The same machinery serves three purposes:

* **self-play** - one network plays both sides and every position is recorded
  as a training example,
* **arena** - two networks play each other to decide which is better,
* **evaluation** - a network plays a fixed opponent (random, minimax).

Two levels of parallelism keep the GPU busy:

1. inside a process, ``parallel`` games run side by side and their network
   questions are answered in one batch (see ``run_searches``),
2. several worker processes do this at the same time, one per CPU core,
   because walking the search trees is plain Python and uses one core each.
"""

from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .buffer import concatenate_samples, empty_samples, pack_observation, pack_policy
from .config import Config
from .evaluators import NetEvaluator
from .game import Game, State
from .mcts import MCTS, MCTSConfig, run_searches
from .net import PolicyValueNet, export_from_bytes
from .players import MinimaxPlayer, RandomPlayer


@dataclass
class AgentSpec:
    """A description of a participant that can be sent to another process."""

    #: "net" (MCTS guided by a network), "random" or "minimax"
    kind: str = "net"
    #: the network, as produced by ``export_to_bytes``
    net: bytes | None = None
    simulations: int = 100
    c_puct: float = 1.5
    dirichlet_epsilon: float = 0.0
    dirichlet_alpha: float = 1.0
    #: plies during which the move is sampled from the visit counts
    temperature_plies: int = 0
    #: search depth of a minimax player
    depth: int = 2


@dataclass
class PlayResult:
    #: one entry per game: +1, 0 or -1 for the first agent (for White in self-play)
    results: list[float] = field(default_factory=list)
    plies: list[int] = field(default_factory=list)
    #: training examples, when recording was requested
    samples: dict[str, np.ndarray] | None = None
    #: positions sent to the network
    evaluations: int = 0
    seconds: float = 0.0

    @property
    def score(self) -> float:
        """Share of the points won by the first agent."""
        return float(np.mean([(result + 1) / 2 for result in self.results]))

    def tally(self) -> tuple[int, int, int]:
        """Wins, draws and losses of the first agent."""
        wins = sum(result > 0 for result in self.results)
        losses = sum(result < 0 for result in self.results)
        return wins, len(self.results) - wins - losses, losses


class SearchAgent:
    """One search tree playing one game."""

    def __init__(self, game: Game, evaluator, spec: AgentSpec, rng: np.random.Generator) -> None:
        config = MCTSConfig(spec.c_puct, spec.dirichlet_epsilon, spec.dirichlet_alpha)
        self.tree = MCTS(game, evaluator, config, rng)
        self.simulations = spec.simulations
        self.temperature_plies = spec.temperature_plies


class _Slot:
    """A game in progress."""

    __slots__ = ("index", "state", "agents", "plies", "positions", "searched")

    def __init__(self, index: int, state: State, agents: tuple) -> None:
        self.index = index
        self.state = state
        self.agents = agents
        self.plies = 0
        self.positions: list = []
        self.searched = False


def play_games(
    game: Game,
    first: AgentSpec,
    second: AgentSpec | None,
    games: int,
    parallel: int = 32,
    device: str = "cpu",
    seed: int = 0,
    record: bool = False,
) -> PlayResult:
    """Play ``games`` games, up to ``parallel`` of them side by side.

    With ``second=None`` the first agent plays itself (self-play). Otherwise the
    two agents swap colours every game. ``record=True`` collects every position
    as a training example: what the board looked like, how the search split its
    visits, and - filled in when the game ends - who won.
    """
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    self_play = second is None
    evaluators: dict[int, NetEvaluator] = {}

    def make_agent(spec: AgentSpec):
        child_seed = int(rng.integers(2**31))
        if spec.kind == "random":
            return RandomPlayer(game, child_seed)
        if spec.kind == "minimax":
            return MinimaxPlayer(game, spec.depth, child_seed)
        if id(spec) not in evaluators:
            net = PolicyValueNet.restore(export_from_bytes(spec.net))
            evaluators[id(spec)] = NetEvaluator(game, net, device)
        return SearchAgent(game, evaluators[id(spec)], spec, np.random.default_rng(child_seed))

    started_games = 0

    def new_slot() -> _Slot:
        nonlocal started_games
        index = started_games
        started_games += 1
        agent = make_agent(first)
        if self_play:
            agents = (agent, agent)
        else:
            opponent = make_agent(second)
            agents = (agent, opponent) if index % 2 == 0 else (opponent, agent)
        state = game.initial_state()
        for participant in {id(a): a for a in agents}.values():
            if isinstance(participant, SearchAgent):
                participant.tree.reset(state)
        return _Slot(index, state, agents)

    result = PlayResult(results=[0.0] * games, plies=[0] * games)
    recorded: list[dict[str, np.ndarray]] = []
    slots = [new_slot() for _ in range(min(parallel, games))]

    while slots:
        # 1. every game whose player to move has a real choice thinks, all together
        trees, budgets = [], []
        for slot in slots:
            agent = slot.agents[game.current_player(slot.state)]
            slot.searched = isinstance(agent, SearchAgent) and len(game.legal_moves(slot.state)) > 1
            if slot.searched:
                trees.append(agent.tree)
                budgets.append(agent.simulations)
        if trees:
            run_searches(trees, budgets)

        # 2. every game makes its move
        survivors = []
        for slot in slots:
            state = slot.state
            agent = slot.agents[game.current_player(state)]
            moves = game.legal_moves(state)
            if isinstance(agent, SearchAgent):
                if slot.searched:
                    temperature = 1.0 if slot.plies < agent.temperature_plies else 0.0
                    move = agent.tree.choose(temperature)
                    distribution = agent.tree.visit_distribution()
                else:
                    # a forced move needs no thinking
                    move, distribution = moves[0], np.ones(1)
                if record:
                    slot.positions.append(_example(game, state, moves, distribution, rng))
            else:
                move = agent.choose(state)

            slot.state = game.next_state(state, move)
            slot.plies += 1
            for participant in {id(a): a for a in slot.agents}.values():
                if isinstance(participant, SearchAgent):
                    participant.tree.advance(move)

            if game.terminal_value(slot.state) is None:
                survivors.append(slot)
                continue

            # 3. the game is over: now every recorded position learns who won
            for_player_0 = game.outcome_for(slot.state, 0)
            first_is_player_0 = self_play or slot.index % 2 == 0
            result.results[slot.index] = for_player_0 if first_is_player_0 else -for_player_0
            result.plies[slot.index] = slot.plies
            if record and slot.positions:
                recorded.append(_finish_examples(slot.positions, for_player_0))
            if started_games < games:
                survivors.append(new_slot())
        slots = survivors

    if record:
        result.samples = concatenate_samples(recorded or [empty_samples(game.observation_shape)])
    result.evaluations = sum(evaluator.positions for evaluator in evaluators.values())
    result.seconds = time.perf_counter() - started
    return result


def _example(game: Game, state: State, moves, distribution, rng: np.random.Generator) -> tuple:
    policy = np.zeros(game.action_size)
    for move, probability in zip(moves, distribution):
        policy[game.action_index(state, move)] += probability
    variants = game.symmetries(state, game.encode(state), policy)
    observation, policy = variants[int(rng.integers(len(variants)))]
    indices, probs = pack_policy(policy)
    return pack_observation(observation), indices, probs, game.current_player(state)


def _finish_examples(positions: list[tuple], for_player_0: float) -> dict[str, np.ndarray]:
    return {
        "observations": np.stack([position[0] for position in positions]),
        "policy_indices": np.stack([position[1] for position in positions]),
        "policy_probs": np.stack([position[2] for position in positions]),
        # the result as seen by whoever was to move in that position
        "values": np.array(
            [for_player_0 if position[3] == 0 else -for_player_0 for position in positions],
            dtype=np.float32,
        ),
    }


# --- worker processes ----------------------------------------------------------------

_WORKER_INDEX = 0


def _init_worker(counter) -> None:
    global _WORKER_INDEX
    with counter.get_lock():
        _WORKER_INDEX = counter.value
        counter.value += 1
    # each worker walks its trees on one core; more threads would only fight
    torch.set_num_threads(1)


def resolve_play_device(name: str, worker_index: int = 0) -> str:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and torch.cuda.device_count() > 1:
        # spread the workers over the GPUs (Kaggle offers two T4s)
        name = f"cuda:{worker_index % torch.cuda.device_count()}"
    return name


def _play_task(task: dict) -> PlayResult:
    game = Config.from_dict(task["config"]).make_game()
    return play_games(
        game,
        task["first"],
        task["second"],
        task["games"],
        parallel=task["parallel"],
        device=resolve_play_device(task["device"], _WORKER_INDEX),
        seed=task["seed"],
        record=task["record"],
    )


class PlayPool:
    """Plays games on ``config.workers`` processes (or in this one if that is < 2).

    The processes are started once and reused, because starting Python with
    PyTorch (and a GPU context) takes seconds.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._pool = None
        if config.workers > 1:
            # "spawn" gives each worker a clean interpreter, which CUDA requires
            context = multiprocessing.get_context("spawn")
            self._pool = context.Pool(
                config.workers, initializer=_init_worker, initargs=(context.Value("i", 0),)
            )

    def play(
        self,
        first: AgentSpec,
        second: AgentSpec | None,
        games: int,
        seed: int,
        record: bool = False,
    ) -> PlayResult:
        started = time.perf_counter()
        workers = max(1, self.config.workers) if self._pool else 1
        workers = min(workers, games)
        shares = [games // workers + (index < games % workers) for index in range(workers)]
        tasks = [
            {
                "config": self.config.to_dict(),
                "first": first,
                "second": second,
                "games": share,
                "parallel": self.config.parallel_games,
                "device": self.config.play_device,
                "seed": seed * 1000 + index,
                "record": record,
            }
            for index, share in enumerate(shares)
        ]
        parts = self._pool.map(_play_task, tasks) if self._pool else [_play_task(tasks[0])]

        merged = PlayResult()
        for part in parts:
            merged.results += part.results
            merged.plies += part.plies
            merged.evaluations += part.evaluations
        if record:
            merged.samples = concatenate_samples([part.samples for part in parts])
        merged.seconds = time.perf_counter() - started
        return merged

    def close(self) -> None:
        if self._pool:
            self._pool.close()
            self._pool.join()
            self._pool = None
