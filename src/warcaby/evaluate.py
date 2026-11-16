"""Measuring a finished (or running) training run.

``tournament`` lets the saved generations play each other and the fixed
opponents (random, minimax at several depths), then fits Elo ratings::

    python -m warcaby.evaluate tournament --run-dir runs/laptop
    python -m warcaby.evaluate plots --run-dir runs/laptop

Results are written to ``<run-dir>/eval/tournament.json`` after every pairing,
so an interrupted tournament continues where it stopped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import Config
from .elo import PairResult, fit_elo
from .net import export_to_bytes
from .selfplay import AgentSpec, PlayPool


def load_run_config(run_dir: Path) -> Config:
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    return Config.from_dict(checkpoint["config"])


def list_generations(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "generations").glob("gen_*.pt"))


def pick_evenly(items: list, count: int) -> list:
    """Up to ``count`` items spread evenly, always with the first and the last."""
    if len(items) <= count:
        return list(items)
    positions = sorted({round(i * (len(items) - 1) / (count - 1)) for i in range(count)})
    return [items[position] for position in positions]


def load_generation(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def run_tournament(
    run_dir: Path,
    generations: int = 8,
    games: int = 20,
    minimax_depths: tuple[int, ...] = (2, 4, 6),
    simulations: int | None = None,
    log=print,
) -> dict:
    config = load_run_config(run_dir)
    simulations = simulations or config.simulations
    out_path = run_dir / "eval" / "tournament.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    specs: dict[str, AgentSpec] = {}
    generation_names = []
    for path in pick_evenly(list_generations(run_dir), generations):
        stored = load_generation(path)
        name = f"gen{stored['generation']}"
        generation_names.append(name)
        specs[name] = AgentSpec(
            net=export_to_bytes(stored["net"]),
            simulations=simulations,
            c_puct=config.c_puct,
            temperature_plies=config.arena_temperature_plies,
        )
    specs["random"] = AgentSpec(kind="random")
    for depth in minimax_depths:
        specs[f"minimax{depth}"] = AgentSpec(kind="minimax", depth=depth)

    settings = {"games": games, "simulations": simulations}
    done: dict[str, dict] = {}
    if out_path.exists():
        previous = json.loads(out_path.read_text())
        if previous.get("settings") == settings:
            done = {f"{p['first']}|{p['second']}": p for p in previous["pairs"]}

    names = list(specs)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    pool = PlayPool(config)
    try:
        for number, (first, second) in enumerate(pairs):
            key = f"{first}|{second}"
            if key not in done:
                played = pool.play(specs[first], specs[second], games, seed=90_000 + number)
                wins, draws, losses = played.tally()
                done[key] = dict(first=first, second=second, wins=wins, draws=draws, losses=losses)
                log(f"[{number + 1}/{len(pairs)}] {first} vs {second}: +{wins} ={draws} -{losses}")
            report = _report(done, names, generation_names, settings)
            out_path.write_text(json.dumps(report, indent=1))
    finally:
        pool.close()
    return report


def _report(done: dict, names: list[str], generation_names: list[str], settings: dict) -> dict:
    wanted = [p for p in done.values() if p["first"] in names and p["second"] in names]
    results = [PairResult(**pair) for pair in wanted]
    return {
        "settings": settings,
        "players": names,
        "generations": generation_names,
        "pairs": wanted,
        # random play is the natural zero of the scale
        "elo": fit_elo(results, anchor="random"),
    }


def summary_table(report: dict) -> str:
    """A Markdown table: every player's rating and its score against the baselines."""
    elo = report["elo"]
    baselines = [name for name in report["players"] if name not in report["generations"]]
    scores: dict[tuple[str, str], str] = {}
    for pair in report["pairs"]:
        scores[(pair["first"], pair["second"])] = (
            f"+{pair['wins']} ={pair['draws']} -{pair['losses']}"
        )
        scores[(pair["second"], pair["first"])] = (
            f"+{pair['losses']} ={pair['draws']} -{pair['wins']}"
        )
    header = "| player | Elo | " + " | ".join(f"vs {name}" for name in baselines) + " |"
    lines = [header, "|---|---:|" + "---|" * len(baselines)]
    for name in sorted(elo, key=elo.get, reverse=True):
        cells = [scores.get((name, baseline), "-") for baseline in baselines]
        lines.append(f"| {name} | {elo[name]:.0f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a training run.")
    commands = parser.add_subparsers(dest="command", required=True)

    tournament = commands.add_parser("tournament", help="generations vs each other and baselines")
    tournament.add_argument("--run-dir", required=True, type=Path)
    tournament.add_argument("--generations", type=int, default=8, help="how many to include")
    tournament.add_argument("--games", type=int, default=20, help="games per pairing")
    tournament.add_argument("--minimax", type=int, nargs="*", default=[2, 4, 6])
    tournament.add_argument("--simulations", type=int, help="default: as in training")

    plots = commands.add_parser("plots", help="draw the progress charts")
    plots.add_argument("--run-dir", required=True, type=Path)
    plots.add_argument("--out", type=Path, help="default: <run-dir>/plots")

    args = parser.parse_args(argv)
    if args.command == "tournament":
        report = run_tournament(
            args.run_dir, args.generations, args.games, tuple(args.minimax), args.simulations
        )
        table = summary_table(report)
        (args.run_dir / "eval" / "summary.md").write_text(table + "\n")
        print(table)
    else:
        from .plots import make_plots

        for path in make_plots(args.run_dir, args.out):
            print(path)


if __name__ == "__main__":
    main()
