from __future__ import annotations

import json

import pytest

from warcaby.elo import PairResult, expected_score, fit_elo
from warcaby.evaluate import pick_evenly, run_tournament, summary_table
from warcaby.plots import make_plots
from warcaby.train import Trainer

from .test_training import tiny_config


def test_expected_score():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1700, 1500) == pytest.approx(0.76, abs=0.01)
    assert expected_score(1500, 1700) == pytest.approx(0.24, abs=0.01)


def test_fit_recovers_known_rating_gaps():
    # results generated exactly from ratings 0 / 200 / 400
    truth = {"a": 0.0, "b": 200.0, "c": 400.0}
    results = []
    for first, second in (("a", "b"), ("b", "c"), ("a", "c")):
        games = 10_000
        wins = round(games * expected_score(truth[first], truth[second]))
        results.append(PairResult(first, second, wins, 0, games - wins))
    elo = fit_elo(results, anchor="a", prior_draws=0.0)
    assert elo["a"] == 0.0
    assert elo["b"] == pytest.approx(200, abs=3)
    assert elo["c"] == pytest.approx(400, abs=3)


def test_draws_count_as_half():
    elo = fit_elo([PairResult("a", "b", 0, 10, 0)], anchor="a")
    assert elo["b"] == pytest.approx(0.0)


def test_a_clean_sweep_gives_a_finite_rating():
    elo = fit_elo([PairResult("strong", "weak", 20, 0, 0)], anchor="weak")
    assert 400 < elo["strong"] < 1000


def test_ratings_do_not_depend_on_who_is_listed_first():
    one = fit_elo([PairResult("a", "b", 7, 1, 2)], anchor="a")
    other = fit_elo([PairResult("b", "a", 2, 1, 7)], anchor="a")
    assert one["b"] == pytest.approx(other["b"])
    assert one["b"] < 0


def test_pick_evenly_keeps_first_and_last():
    assert pick_evenly(list(range(4)), 8) == [0, 1, 2, 3]
    picked = pick_evenly(list(range(21)), 5)
    assert picked == [0, 5, 10, 15, 20]


def test_tournament_and_plots_on_a_tiny_run(tmp_path):
    run_dir = tmp_path / "run"
    trainer = Trainer(tiny_config(accept_score=0.0), run_dir, log=lambda m: None)
    trainer.run(iterations=2)
    trainer.close()

    messages = []
    report = run_tournament(
        run_dir, generations=2, games=2, minimax_depths=(1,), log=messages.append
    )
    assert report["generations"] == ["gen0", "gen2"]
    assert set(report["players"]) == {"gen0", "gen2", "random", "minimax1"}
    assert len(report["pairs"]) == 6 == len(messages)
    assert report["elo"]["random"] == 0.0
    assert all(pair["wins"] + pair["draws"] + pair["losses"] == 2 for pair in report["pairs"])
    stored = json.loads((run_dir / "eval" / "tournament.json").read_text())
    assert stored["elo"] == report["elo"]

    # running again plays nothing new
    again = run_tournament(
        run_dir, generations=2, games=2, minimax_depths=(1,), log=messages.append
    )
    assert len(messages) == 6
    assert again["pairs"] == report["pairs"]

    table = summary_table(report)
    assert "| player | Elo | vs random | vs minimax1 |" in table
    assert table.count("\n") == 5

    plots = make_plots(run_dir)
    assert {path.name for path in plots} == {
        "training.png", "baselines.png", "arena.png", "elo.png"
    }  # fmt: skip
    assert all(path.stat().st_size > 5000 for path in plots)
