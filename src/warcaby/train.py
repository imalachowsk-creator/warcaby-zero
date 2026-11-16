"""The AlphaZero training loop: self-play -> training -> arena.

Each iteration:

1. **Self-play.** The best network so far plays games against itself, with MCTS
   choosing the moves. Every position is stored with two labels: how the search
   split its visits among the moves, and who eventually won.
2. **Training.** The candidate network learns from the stored positions: its
   policy is pulled towards the search's visit distribution, its value towards
   the game results.
3. **Arena.** The candidate plays the best network. Only if it scores clearly
   above half does it become the new best, and a new *generation* is born.
   Self-play data always comes from a network that has proven itself.

Where does the knowledge come from? From two facts only: the rules, and the
results of finished games. The search is what turns them into improvement: it
looks ahead with the current network and comes back with a better move choice
than the network alone would make. Training makes the network imitate that
improved choice, so the next search starts from a better place.

Everything needed to continue lives in the run directory and is rewritten after
every iteration, so a run can be stopped at any point and resumed later::

    checkpoint.pt      networks, optimiser, counters, history
    buffer.npz         the replay buffer
    generations/       every network that became "best", for the Elo ranking
    metrics.jsonl      one line of numbers per iteration
    tb/                TensorBoard logs
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .buffer import ReplayBuffer
from .config import PRESETS, Config
from .net import PolicyValueNet, export_to_bytes, pick_device
from .selfplay import AgentSpec, PlayPool


def atomic_torch_save(obj, path: Path) -> None:
    """Write to a temporary file first, so a crash never leaves half a checkpoint."""
    temporary = path.with_name(path.name + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


class Trainer:
    def __init__(self, config: Config, run_dir: Path | str, log=None) -> None:
        self.run_dir = Path(run_dir)
        # flush, so progress shows up at once in a notebook or a log file
        self.log = log or (lambda message: print(message, flush=True))
        (self.run_dir / "generations").mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.run_dir / "checkpoint.pt"
        checkpoint = None
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            # the run's own settings win over whatever was passed in
            config = Config.from_dict(checkpoint["config"])

        self.config = config
        self.game = config.make_game()
        self.device = pick_device(config.train_device)
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)

        self.candidate = PolicyValueNet.for_game(self.game, config.net_config())
        self.buffer = ReplayBuffer(config.buffer_size, self.game.observation_shape)
        self.iteration = 0
        self.generation = 0
        self.history: list[dict] = []

        if checkpoint is not None:
            if checkpoint["game"] != self.game.name:
                raise ValueError(f"checkpoint is for {checkpoint['game']}, not {self.game.name}")
            self.candidate = PolicyValueNet.restore(checkpoint["candidate"])
            self.best = checkpoint["best"]
            self.iteration = checkpoint["iteration"]
            self.generation = checkpoint["generation"]
            self.history = checkpoint["history"]
            self.rng = np.random.default_rng(config.seed + self.iteration)
            if (self.run_dir / "buffer.npz").exists():
                self.buffer.load(self.run_dir / "buffer.npz")
        else:
            # generation 0: a network with random weights
            self.best = self.candidate.export()

        self.candidate.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.candidate.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        if checkpoint is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.log(
                f"resumed {self.run_dir}: iteration {self.iteration}, generation "
                f"{self.generation}, {len(self.buffer)} positions in the buffer"
            )
        else:
            self._save_generation()
            self._save_checkpoint()

        self.pool = PlayPool(config)
        self._writer = None

    # --- the loop -------------------------------------------------------------------

    def run(self, iterations: int | None = None, hours: float | None = None) -> None:
        """Train for ``iterations`` more iterations and/or until ``hours`` have passed.

        The time limit is checked before an iteration starts, using the longest
        iteration so far as the estimate, so the run stops cleanly before a
        session with a hard limit (Kaggle) is cut off.
        """
        started = time.monotonic()
        longest = 0.0
        done = 0
        while iterations is None or done < iterations:
            elapsed = time.monotonic() - started
            if hours is not None and elapsed + 1.15 * longest > hours * 3600:
                self.log(f"time budget reached after {elapsed / 3600:.2f} h")
                break
            iteration_started = time.monotonic()
            self.run_iteration()
            longest = max(longest, time.monotonic() - iteration_started)
            done += 1
            if iterations is None and hours is None:
                break

    def run_iteration(self) -> dict:
        config = self.config
        started = time.monotonic()
        self.iteration += 1
        metrics: dict = {"iteration": self.iteration}

        # 1. self-play with the best network
        played = self.pool.play(
            self._spec(
                self.best,
                dirichlet_epsilon=config.dirichlet_epsilon,
                temperature_plies=config.temperature_plies,
            ),
            None,
            config.games_per_iteration,
            seed=config.seed * 100_003 + self.iteration,
            record=True,
        )
        self.buffer.add(played.samples)
        new_positions = len(played.samples["values"])
        white, draws, black = played.tally()
        metrics.update(
            selfplay_games=len(played.results),
            selfplay_positions=new_positions,
            selfplay_plies=float(np.mean(played.plies)),
            selfplay_white_wins=white / len(played.results),
            selfplay_draws=draws / len(played.results),
            selfplay_black_wins=black / len(played.results),
            selfplay_seconds=played.seconds,
            selfplay_evaluations_per_second=played.evaluations / max(played.seconds, 1e-9),
            buffer_size=len(self.buffer),
        )

        # 2. train the candidate on the buffer
        steps = max(1, round(new_positions * config.reuse / config.batch_size))
        train_started = time.monotonic()
        metrics.update(self._train(steps))
        metrics.update(train_steps=steps, train_seconds=time.monotonic() - train_started)

        # 3. arena: does the candidate beat the best network?
        candidate = self.candidate.export()
        arena = self.pool.play(
            self._spec(candidate, temperature_plies=config.arena_temperature_plies),
            self._spec(self.best, temperature_plies=config.arena_temperature_plies),
            config.arena_games,
            seed=config.seed * 100_003 + self.iteration + 50_000,
        )
        wins, draws, losses = arena.tally()
        accepted = arena.score >= config.accept_score
        if accepted:
            self.best = candidate
            self.generation += 1
            self._save_generation()
        metrics.update(
            arena_score=arena.score,
            arena_wins=wins,
            arena_draws=draws,
            arena_losses=losses,
            arena_seconds=arena.seconds,
            accepted=accepted,
            generation=self.generation,
        )

        # 4. now and then: how does the best network do against fixed opponents?
        if config.eval_every and self.iteration % config.eval_every == 0:
            metrics.update(self.evaluate_baselines())

        metrics["seconds"] = time.monotonic() - started
        self.history.append(metrics)
        self._save_checkpoint()
        self._write_logs(metrics)
        self.log(self._summary(metrics))
        return metrics

    def _spec(self, exported: dict, **overrides) -> AgentSpec:
        config = self.config
        settings = dict(
            net=export_to_bytes(exported),
            simulations=config.simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
        )
        settings.update(overrides)
        return AgentSpec(**settings)

    def _train(self, steps: int) -> dict:
        config = self.config
        net = self.candidate.train()
        policy_losses, value_losses = [], []
        for _ in range(steps):
            batch = self.buffer.sample(config.batch_size, self.rng)
            observations = torch.from_numpy(batch["observations"]).to(self.device).float() / 255.0
            indices = torch.from_numpy(batch["policy_indices"].astype(np.int64)).to(self.device)
            probs = torch.from_numpy(batch["policy_probs"]).to(self.device)
            outcomes = torch.from_numpy(batch["values"]).to(self.device)

            logits, values = net(observations)
            # Policy: cross-entropy between the search's visit distribution and
            # the network's move probabilities. Padding entries have probability
            # 0 and so contribute nothing.
            log_probs = F.log_softmax(logits, dim=1).gather(1, indices)
            policy_loss = -(probs * log_probs).sum(dim=1).mean()
            # Value: squared error between the prediction and the real result.
            value_loss = F.mse_loss(values, outcomes)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
        net.eval()
        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
        }

    def evaluate_baselines(self) -> dict:
        """Score of the best network against random play and minimax."""
        config = self.config
        me = self._spec(self.best, temperature_plies=config.arena_temperature_plies)
        opponents = [("random", AgentSpec(kind="random"))]
        opponents += [
            (f"minimax{depth}", AgentSpec(kind="minimax", depth=depth))
            for depth in config.eval_minimax_depths
        ]
        scores = {}
        for number, (name, opponent) in enumerate(opponents):
            played = self.pool.play(
                me,
                opponent,
                config.eval_games,
                seed=config.seed * 100_003 + self.iteration + 70_000 + number,
            )
            scores[f"eval_{name}"] = played.score
        return scores

    # --- saving and logging -----------------------------------------------------------

    def _save_generation(self) -> None:
        path = self.run_dir / "generations" / f"gen_{self.generation:04d}.pt"
        atomic_torch_save(
            {"game": self.game.name, "generation": self.generation,
             "iteration": self.iteration, "net": self.best},
            path,
        )  # fmt: skip

    def _save_checkpoint(self) -> None:
        self.buffer.save(self.run_dir / "buffer.npz")
        atomic_torch_save(
            {
                "game": self.game.name,
                "config": self.config.to_dict(),
                "iteration": self.iteration,
                "generation": self.generation,
                "candidate": self.candidate.export(),
                "best": self.best,
                "optimizer": self.optimizer.state_dict(),
                "history": self.history,
            },
            self.run_dir / "checkpoint.pt",
        )
        with open(self.run_dir / "metrics.jsonl", "w") as handle:
            for entry in self.history:
                handle.write(json.dumps(entry) + "\n")

    def _write_logs(self, metrics: dict) -> None:
        if self._writer is None:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(str(self.run_dir / "tb"))
        groups = {
            "selfplay": "selfplay/",
            "arena": "arena/",
            "eval": "eval/",
            "policy_loss": "loss/",
            "value_loss": "loss/",
            "train": "train/",
        }
        for name, value in metrics.items():
            if name == "iteration" or value is None:
                continue
            prefix = next((p for key, p in groups.items() if name.startswith(key)), "run/")
            self._writer.add_scalar(prefix + name, float(value), self.iteration)
        self._writer.flush()

    @staticmethod
    def _summary(metrics: dict) -> str:
        parts = [
            f"iter {metrics['iteration']:>3}",
            f"gen {metrics['generation']:>3}",
            f"self-play {metrics['selfplay_games']} games/{metrics['selfplay_seconds']:.0f}s"
            f" (plies {metrics['selfplay_plies']:.0f}, draws {metrics['selfplay_draws']:.0%})",
            f"loss p {metrics['policy_loss']:.3f} v {metrics['value_loss']:.3f}",
            f"arena {metrics['arena_score']:.0%} "
            + ("ACCEPTED" if metrics["accepted"] else "rejected"),
        ]
        evals = [f"{k[5:]} {v:.0%}" for k, v in metrics.items() if k.startswith("eval_")]
        if evals:
            parts.append("vs " + ", ".join(evals))
        parts.append(f"{metrics['seconds']:.0f}s")
        return " | ".join(parts)

    def close(self) -> None:
        self.pool.close()
        if self._writer is not None:
            self._writer.close()


def parse_overrides(pairs: list[str]) -> dict:
    """``["simulations=50", "eval_minimax_depths=[1,2]"]`` as a dictionary."""
    overrides = {}
    for pair in pairs:
        name, _, raw = pair.partition("=")
        overrides[name] = json.loads(raw)
    return overrides


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a draughts network by self-play.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    parser.add_argument("--run-dir", default="runs/quick")
    parser.add_argument("--iterations", type=int, help="iterations to run in this session")
    parser.add_argument("--hours", type=float, help="stop before this much time has passed")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="NAME=VALUE",
        help="override settings of a new run, e.g. --set simulations=60 workers=2",
    )  # fmt: skip
    args = parser.parse_args(argv)

    config = PRESETS[args.preset]()
    for name, value in parse_overrides(args.set).items():
        if not hasattr(config, name):
            parser.error(f"unknown setting {name!r}")
        setattr(config, name, value)

    trainer = Trainer(config, args.run_dir)
    try:
        trainer.run(iterations=args.iterations, hours=args.hours)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
