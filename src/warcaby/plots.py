"""Progress charts for a training run, drawn from its log files.

* ``training.png`` - losses and self-play statistics per iteration
  (from ``metrics.jsonl``),
* ``baselines.png`` - score of the best network against random and minimax,
  measured during training,
* ``arena.png`` - how each candidate did against the best network,
* ``elo.png`` - Elo of the generations (needs ``evaluate tournament`` first).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# One palette for all charts: series colours in a fixed order, quiet grey chrome.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _style(ax, title: str, xlabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _figure(rows: int, cols: int, width: float, height: float):
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _percent_axis(ax) -> None:
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])


def load_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plot_training(metrics: list[dict], path: Path) -> Path:
    iterations = [m["iteration"] for m in metrics]
    panels = [
        ("Policy loss (cross-entropy with the search's visits)", "policy_loss", None),
        ("Value loss (squared error against the game result)", "value_loss", None),
        ("Average self-play game length (plies)", "selfplay_plies", None),
        ("Share of self-play games drawn", "selfplay_draws", "percent"),
    ]
    fig, axes = _figure(2, 2, 10, 6.2)
    for ax, (title, key, kind) in zip(axes.flat, panels):
        _style(ax, title, "iteration")
        ax.plot(iterations, [m[key] for m in metrics], color=SERIES[0], linewidth=2)
        if kind == "percent":
            _percent_axis(ax)
    return _save(fig, path)


def plot_baselines(metrics: list[dict], path: Path) -> Path | None:
    keys = sorted(
        {key for m in metrics for key in m if key.startswith("eval_")},
        key=lambda key: (key != "eval_random", len(key), key),
    )
    if not keys:
        return None
    fig, axes = _figure(1, 1, 8, 4.4)
    ax = axes[0][0]
    _style(ax, "Score of the best network against fixed opponents", "iteration")
    for colour, key in zip(SERIES, keys):
        points = [(m["iteration"], m[key]) for m in metrics if key in m]
        xs, ys = zip(*points)
        ax.plot(
            xs, ys, color=colour, linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=key[5:], clip_on=False,
        )  # fmt: skip
    ax.axhline(0.5, color=AXIS, linewidth=1)
    _percent_axis(ax)
    ax.legend(
        frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncols=4,
        loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.3,
    )  # fmt: skip
    # the legend sits in its own row between the title and the plot
    ax.title.set_position((0, 1.0))
    ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=11, color=INK, pad=30)
    return _save(fig, path)


def plot_arena(metrics: list[dict], path: Path, accept_score: float = 0.55) -> Path:
    fig, axes = _figure(1, 1, 8, 4.4)
    ax = axes[0][0]
    _style(ax, "Arena: candidate's score against the best network", "iteration")
    accepted = [(m["iteration"], m["arena_score"]) for m in metrics if m["accepted"]]
    rejected = [(m["iteration"], m["arena_score"]) for m in metrics if not m["accepted"]]
    if accepted:
        ax.scatter(
            *zip(*accepted), s=42, color=SERIES[0], edgecolors=SURFACE, linewidths=1.5,
            label="accepted: becomes the new best", zorder=3, clip_on=False,
        )  # fmt: skip
    if rejected:
        ax.scatter(
            *zip(*rejected), s=42, facecolors=SURFACE, edgecolors=INK_MUTED, linewidths=1.5,
            label="rejected: keeps training", zorder=3, clip_on=False,
        )  # fmt: skip
    ax.axhline(accept_score, color=AXIS, linewidth=1)
    ax.annotate(
        f"needed to be accepted: {accept_score:.0%}", (1, accept_score),
        xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset points",
        ha="right", fontsize=8, color=INK_MUTED,
    )  # fmt: skip
    _percent_axis(ax)
    ax.legend(
        frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncols=2,
        loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.3,
    )  # fmt: skip
    # the legend sits in its own row between the title and the plot
    ax.title.set_position((0, 1.0))
    ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=11, color=INK, pad=30)
    return _save(fig, path)


def plot_elo(report: dict, path: Path) -> Path:
    elo = report["elo"]
    generations = [(int(name[3:]), elo[name]) for name in report["generations"]]
    baselines = [(name, elo[name]) for name in report["players"] if not name.startswith("gen")]
    fig, axes = _figure(1, 1, 8, 4.8)
    ax = axes[0][0]
    _style(ax, "Elo by generation (random play = 0)", "generation")
    for name, rating in baselines:
        ax.axhline(rating, color=AXIS, linewidth=1)
        ax.annotate(
            name, (1, rating), xycoords=("axes fraction", "data"), xytext=(4, -3),
            textcoords="offset points", fontsize=8, color=INK_MUTED, annotation_clip=False,
        )  # fmt: skip
    xs, ys = zip(*generations)
    ax.plot(
        xs, ys, color=SERIES[0], linewidth=2, marker="o", markersize=6,
        markeredgecolor=SURFACE, markeredgewidth=1.5, clip_on=False, zorder=3,
    )  # fmt: skip
    ax.annotate(
        f"{ys[-1]:.0f}", (xs[-1], ys[-1]), xytext=(0, 8), textcoords="offset points",
        ha="center", fontsize=9, color=INK,
    )  # fmt: skip
    fig.subplots_adjust(right=0.86)
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def make_plots(run_dir: Path | str, out_dir: Path | str | None = None) -> list[Path]:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    metrics = load_metrics(run_dir)
    if metrics:
        written.append(plot_training(metrics, out_dir / "training.png"))
        baselines = plot_baselines(metrics, out_dir / "baselines.png")
        if baselines:
            written.append(baselines)
        written.append(plot_arena(metrics, out_dir / "arena.png"))
    tournament = run_dir / "eval" / "tournament.json"
    if tournament.exists():
        written.append(plot_elo(json.loads(tournament.read_text()), out_dir / "elo.png"))
    return written
