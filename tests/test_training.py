from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from warcaby.buffer import (
    MAX_POLICY_ENTRIES,
    ReplayBuffer,
    pack_observation,
    pack_policy,
)
from warcaby.config import Config, quick
from warcaby.net import PolicyValueNet, export_from_bytes, export_to_bytes
from warcaby.selfplay import AgentSpec, play_games
from warcaby.train import Trainer, parse_overrides


def tiny_config(**overrides) -> Config:
    settings = dict(
        board_size=6,
        max_plies=60,
        blocks=1,
        channels=8,
        simulations=8,
        games_per_iteration=6,
        parallel_games=4,
        workers=0,
        play_device="cpu",
        train_device="cpu",
        buffer_size=2000,
        batch_size=16,
        arena_games=4,
        eval_every=1,
        eval_games=2,
        eval_minimax_depths=[1],
        temperature_plies=6,
    )
    settings.update(overrides)
    return Config(**settings)


def net_spec(config: Config, seed: int = 0, **overrides) -> AgentSpec:
    torch.manual_seed(seed)
    net = PolicyValueNet.for_game(config.make_game(), config.net_config())
    return AgentSpec(net=export_to_bytes(net.export()), simulations=config.simulations, **overrides)


# --- buffer ------------------------------------------------------------------------------


def fake_samples(count: int, start: int = 0) -> dict:
    return {
        "observations": np.full((count, 6, 6, 6), 7, dtype=np.uint8),
        "policy_indices": np.zeros((count, MAX_POLICY_ENTRIES), dtype=np.int16),
        "policy_probs": np.zeros((count, MAX_POLICY_ENTRIES), dtype=np.float32),
        "values": np.arange(start, start + count, dtype=np.float32),
    }


def test_buffer_keeps_only_the_newest_positions():
    buffer = ReplayBuffer(10, (6, 6, 6))
    buffer.add(fake_samples(6))
    assert len(buffer) == 6
    buffer.add(fake_samples(7, start=6))
    assert len(buffer) == 10
    assert buffer.total_added == 13
    assert list(buffer.contents()["values"]) == list(range(3, 13))


def test_buffer_accepts_more_than_its_capacity_at_once():
    buffer = ReplayBuffer(5, (6, 6, 6))
    buffer.add(fake_samples(12))
    assert list(buffer.contents()["values"]) == [7, 8, 9, 10, 11]
    assert buffer.total_added == 12


def test_buffer_sample_and_save_load(tmp_path):
    buffer = ReplayBuffer(10, (6, 6, 6))
    buffer.add(fake_samples(13))
    batch = buffer.sample(32, np.random.default_rng(0))
    assert batch["observations"].shape == (32, 6, 6, 6)
    assert set(batch["values"]) <= set(range(3, 13))

    buffer.save(tmp_path / "buffer.npz")
    restored = ReplayBuffer(10, (6, 6, 6))
    restored.load(tmp_path / "buffer.npz")
    assert restored.total_added == 13
    assert np.array_equal(restored.contents()["values"], buffer.contents()["values"])
    # and it keeps working as a ring afterwards
    restored.add(fake_samples(2, start=100))
    assert list(restored.contents()["values"])[-2:] == [100, 101]
    assert len(restored) == 10


def test_pack_policy_round_trip():
    policy = np.zeros(1024)
    policy[[5, 700, 1023]] = [0.2, 0.5, 0.3]
    indices, probs = pack_policy(policy)
    rebuilt = np.zeros(1024)
    np.add.at(rebuilt, indices.astype(int), probs)
    assert np.allclose(rebuilt, policy)
    assert probs.sum() == pytest.approx(1.0)


def test_pack_policy_keeps_the_most_visited_moves_when_there_are_too_many():
    policy = np.arange(1, 101, dtype=float)
    indices, probs = pack_policy(policy / policy.sum())
    assert set(indices.astype(int)) == set(range(100 - MAX_POLICY_ENTRIES, 100))
    assert probs.sum() == pytest.approx(1.0)


def test_pack_observation_uses_the_full_byte_range():
    packed = pack_observation(np.array([0.0, 0.5, 1.0], dtype=np.float32))
    assert list(packed) == [0, 128, 255]


# --- playing ---------------------------------------------------------------------------------


def test_self_play_records_one_example_per_ply():
    config = tiny_config()
    game = config.make_game()
    spec = net_spec(config, dirichlet_epsilon=0.25, temperature_plies=6)
    played = play_games(game, spec, None, games=5, parallel=3, seed=1, record=True)

    assert len(played.results) == 5
    assert all(result in (-1.0, 0.0, 1.0) for result in played.results)
    samples = played.samples
    assert len(samples["values"]) == sum(played.plies)
    assert samples["observations"].shape[1:] == game.observation_shape
    assert np.allclose(samples["policy_probs"].sum(axis=1), 1.0, atol=1e-5)
    assert samples["policy_indices"].max() < game.action_size
    assert set(np.unique(samples["values"])) <= {-1.0, 0.0, 1.0}
    assert played.evaluations > 0


def test_self_play_value_targets_alternate_with_the_side_to_move():
    config = tiny_config()
    game = config.make_game()
    played = play_games(game, net_spec(config), None, games=1, parallel=1, seed=3, record=True)
    values = played.samples["values"]
    result = played.results[0]  # for White
    expected = [result if ply % 2 == 0 else -result for ply in range(len(values))]
    assert list(values) == expected


def test_self_play_is_reproducible():
    config = tiny_config()
    game = config.make_game()
    spec = net_spec(config, dirichlet_epsilon=0.25, temperature_plies=6)
    first = play_games(game, spec, None, games=3, parallel=2, seed=5, record=True)
    second = play_games(game, spec, None, games=3, parallel=2, seed=5, record=True)
    assert first.results == second.results
    assert np.array_equal(first.samples["policy_probs"], second.samples["policy_probs"])


def test_match_against_random_reports_the_first_agents_score():
    config = tiny_config(simulations=30)
    game = config.make_game()
    random_spec = AgentSpec(kind="random")
    minimax = AgentSpec(kind="minimax", depth=3)
    played = play_games(game, minimax, random_spec, games=10, parallel=4, seed=0)
    assert played.score >= 0.8
    reversed_roles = play_games(game, random_spec, minimax, games=10, parallel=4, seed=0)
    assert reversed_roles.score <= 0.2
    wins, draws, losses = played.tally()
    assert wins + draws + losses == 10


def test_network_bytes_round_trip():
    config = tiny_config()
    net = PolicyValueNet.for_game(config.make_game(), config.net_config())
    restored = PolicyValueNet.restore(export_from_bytes(export_to_bytes(net.export())))
    for name, tensor in net.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name])


# --- the loop ----------------------------------------------------------------------------------


def test_overrides_are_parsed_as_json():
    assert parse_overrides(
        ["simulations=50", "eval_minimax_depths=[1, 2]", 'play_device="cpu"']
    ) == {
        "simulations": 50,
        "eval_minimax_depths": [1, 2],
        "play_device": "cpu",
    }


def test_presets_build_a_game_and_a_network():
    config = quick()
    net = PolicyValueNet.for_game(config.make_game(), config.net_config())
    assert net.action_size == 1024
    assert Config.from_dict(config.to_dict()) == config


def test_training_iteration_and_resume(tmp_path):
    run_dir = tmp_path / "run"
    messages = []
    trainer = Trainer(tiny_config(), run_dir, log=messages.append)
    assert (run_dir / "generations" / "gen_0000.pt").exists()
    metrics = trainer.run_iteration()
    trainer.close()

    assert metrics["iteration"] == 1
    assert metrics["selfplay_games"] == 6
    assert metrics["buffer_size"] == metrics["selfplay_positions"] > 0
    assert np.isfinite(metrics["policy_loss"]) and np.isfinite(metrics["value_loss"])
    assert 0 <= metrics["arena_score"] <= 1
    assert "eval_random" in metrics and "eval_minimax1" in metrics
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "buffer.npz").exists()
    assert list((run_dir / "tb").iterdir())
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["iteration"] == 1

    # a new Trainer on the same directory continues where the first one stopped,
    # ignoring the configuration passed in
    resumed = Trainer(tiny_config(simulations=999), run_dir, log=messages.append)
    assert resumed.config.simulations == 8
    assert resumed.iteration == 1
    assert len(resumed.buffer) == metrics["buffer_size"]
    assert resumed.generation == metrics["generation"]
    for name, tensor in trainer.candidate.state_dict().items():
        assert torch.equal(tensor.cpu(), resumed.candidate.state_dict()[name].cpu())
    second = resumed.run_iteration()
    resumed.close()
    assert second["iteration"] == 2
    assert second["buffer_size"] > metrics["buffer_size"]
    assert len((run_dir / "metrics.jsonl").read_text().splitlines()) == 2
    assert any("resumed" in message for message in messages)


def test_candidate_replaces_best_only_above_the_threshold(tmp_path):
    always = Trainer(
        tiny_config(accept_score=0.0, eval_every=0), tmp_path / "a", log=lambda m: None
    )
    metrics = always.run_iteration()
    always.close()
    assert metrics["accepted"] and metrics["generation"] == 1
    assert (tmp_path / "a" / "generations" / "gen_0001.pt").exists()

    never = Trainer(
        tiny_config(accept_score=1.01, eval_every=0), tmp_path / "b", log=lambda m: None
    )
    before = never.best
    metrics = never.run_iteration()
    never.close()
    assert not metrics["accepted"] and metrics["generation"] == 0
    assert never.best is before
    assert not (tmp_path / "b" / "generations" / "gen_0001.pt").exists()


def test_training_reduces_the_loss_on_a_fixed_buffer(tmp_path):
    trainer = Trainer(tiny_config(eval_every=0), tmp_path / "run", log=lambda m: None)
    game = trainer.game
    played = play_games(game, net_spec(trainer.config), None, games=8, parallel=4, record=True)
    trainer.buffer.add(played.samples)
    first = trainer._train(5)
    for _ in range(10):
        last = trainer._train(20)
    trainer.close()
    assert last["policy_loss"] < first["policy_loss"]
    assert last["value_loss"] < first["value_loss"]


def test_time_budget_stops_the_run(tmp_path):
    messages = []
    trainer = Trainer(tiny_config(eval_every=0), tmp_path / "run", log=messages.append)
    trainer.run(hours=0.0)
    trainer.close()
    assert trainer.iteration == 0
    assert any("time budget" in message for message in messages)


@pytest.mark.slow
def test_worker_processes_play_games(tmp_path):
    trainer = Trainer(tiny_config(workers=2, eval_every=0), tmp_path / "run", log=lambda m: None)
    metrics = trainer.run_iteration()
    trainer.close()
    assert metrics["selfplay_games"] == 6
    assert metrics["selfplay_positions"] > 0
