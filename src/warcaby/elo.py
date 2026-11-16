"""Elo ratings from the results of a tournament.

Elo puts all players on one scale such that the rating *difference* predicts the
result: the expected score of A against B is ``1 / (1 + 10 ** ((Rb - Ra) / 400))``.
A 100-point lead means about 64% of the points, 200 points about 76%, 400 points
about 91%.

Here the ratings are not updated game by game as on a chess site. All results
are known at once, so we look for the set of ratings that explains them best
(maximum likelihood for the Bradley-Terry model, with a draw counted as half a
win and half a loss). This does not depend on the order the games were played in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PairResult:
    """Games between two players, from the point of view of ``first``."""

    first: str
    second: str
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / max(self.games, 1)


def expected_score(rating: float, opponent: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent - rating) / 400.0))


def fit_elo(
    results: list[PairResult],
    anchor: str | None = None,
    prior_draws: float = 1.0,
    iterations: int = 5000,
) -> dict[str, float]:
    """Ratings that best explain ``results``; ``anchor`` is placed at 0.

    ``prior_draws`` adds that many imaginary drawn games to every pairing. A
    player who won all 20 games is "infinitely" stronger as far as the data can
    tell; the imaginary draw keeps the estimate finite. Differences beyond about
    600 points therefore mean "won practically everything", not a measured gap.
    """
    players = sorted({name for r in results for name in (r.first, r.second)})
    points = dict.fromkeys(players, 0.0)
    games: dict[tuple[str, str], float] = {}
    for r in results:
        if r.games == 0:
            continue
        points[r.first] += r.wins + 0.5 * r.draws + 0.5 * prior_draws
        points[r.second] += r.losses + 0.5 * r.draws + 0.5 * prior_draws
        key = (r.first, r.second) if r.first < r.second else (r.second, r.first)
        games[key] = games.get(key, 0.0) + r.games + prior_draws

    # Minorisation-maximisation for Bradley-Terry strengths: repeatedly set each
    # strength to (points won) / (games weighted by how hard the opponents are).
    strength = dict.fromkeys(players, 1.0)
    for _ in range(iterations):
        updated = {}
        for player in players:
            denominator = 0.0
            for (a, b), count in games.items():
                if player == a:
                    denominator += count / (strength[a] + strength[b])
                elif player == b:
                    denominator += count / (strength[a] + strength[b])
            updated[player] = points[player] / denominator if denominator else strength[player]
        mean_log = sum(math.log(v) for v in updated.values()) / len(updated)
        updated = {p: v / math.exp(mean_log) for p, v in updated.items()}
        change = max(abs(math.log(updated[p] / strength[p])) for p in players)
        strength = updated
        if change < 1e-10:
            break

    ratings = {player: 400.0 * math.log10(strength[player]) for player in players}
    offset = ratings[anchor] if anchor in ratings else 0.0
    return {player: rating - offset for player, rating in ratings.items()}
