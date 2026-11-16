from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient

from warcaby.app.server import create_app
from warcaby.app.session import GameSession, Model, load_model
from warcaby.app.storage import ResultStore
from warcaby.draughts import Draughts
from warcaby.evaluators import UniformEvaluator
from warcaby.net import NetConfig, PolicyValueNet


def uniform_model() -> Model:
    game = Draughts(8)
    return Model(game, UniformEvaluator(game), "test", trained=False)


@pytest.fixture
def client(tmp_path):
    store = ResultStore(tmp_path / "games.db")
    with TestClient(create_app(uniform_model(), store)) as test_client:
        test_client.store = store
        yield test_client


def new_game(client, **settings) -> dict:
    body = {"name": "Ala", "simulations": 8, **settings}
    response = client.post("/api/new", json=body)
    assert response.status_code == 200, response.text
    return response.json()


# --- storage -------------------------------------------------------------------------


def test_standings_split_results_by_opponent(tmp_path):
    store = ResultStore(tmp_path / "db" / "games.db")
    store.add("Ala", "AI", False, True, "white", "no_moves", ["c3-d4"], model="m", simulations=50)
    store.add("AI", "Ala", True, False, "white", "resignation", ["c3-d4", "f6-e5"])
    store.add("Ala", "Olek", False, False, "draw", "repetition", [])
    store.add("Olek", "Ala", False, False, "white", "no_moves", [])

    standings = {entry["name"]: entry for entry in store.standings()}
    assert set(standings) == {"Ala", "Olek"}  # the AI is not listed
    assert standings["Ala"]["games"] == 4
    assert standings["Ala"]["vs_ai"] == {"wins": 1, "draws": 0, "losses": 1}
    assert standings["Ala"]["vs_people"] == {"wins": 0, "draws": 1, "losses": 1}
    assert standings["Olek"]["vs_people"] == {"wins": 1, "draws": 1, "losses": 0}
    # both have 1.5 points; the one with more games is listed first
    assert store.standings()[0]["name"] == "Ala"

    recent = store.recent(2)
    assert [game["white"] for game in recent] == ["Olek", "Ala"]
    assert recent[1]["winner"] == "draw"
    # a second store on the same file sees the same games
    assert len(ResultStore(tmp_path / "db" / "games.db").recent()) == 4


# --- session -------------------------------------------------------------------------


def test_session_alternates_between_human_and_ai():
    session = GameSession("g", uniform_model(), "Ala", "AI", (False, True), simulations=8, seed=0)
    session.play_ai_if_due()
    assert session.history == []  # White is human: nothing happens yet
    session.play_human(0)
    session.play_ai_if_due()
    assert len(session.history) == 2
    assert session.ai_move["by"] == "black"
    assert session.ai_move["moves"][session.ai_move["chosen"]]["name"] == session.ai_move["name"]
    with pytest.raises(ValueError):
        session.play_human(99)


def test_analysis_lists_every_legal_move():
    session = GameSession(
        "g", uniform_model(), "Ala", "Olek", (False, False), simulations=30, seed=0
    )
    session.analyse()
    analysis = session.analysis
    assert len(analysis["moves"]) == 7
    assert sum(move["visits"] for move in analysis["moves"]) == 29
    assert sum(move["share"] for move in analysis["moves"]) == pytest.approx(1.0)
    assert -1 <= analysis["value_white"] <= 1
    assert analysis["to_move"] == "white"


def test_resigning_against_the_ai_counts_as_the_humans_loss():
    session = GameSession("g", uniform_model(), "AI", "Ala", (True, False), simulations=8, seed=0)
    session.play_ai_if_due()
    session.resign()
    assert session.result() == {"winner": "white", "reason": "resignation"}
    with pytest.raises(ValueError):
        session.play_human(0)


def test_load_model_from_a_generation_file(tmp_path):
    game = Draughts(8)
    net = PolicyValueNet.for_game(game, NetConfig(blocks=1, channels=8))
    generations = tmp_path / "myrun" / "generations"
    generations.mkdir(parents=True)
    torch.save({"game": game.name, "generation": 3, "iteration": 5, "net": net.export()},
               generations / "gen_0003.pt")  # fmt: skip
    for path in (generations / "gen_0003.pt", tmp_path / "myrun"):
        model = load_model(path)
        assert model.trained and model.label == "myrun gen 3"
        assert model.game.size == 8
    assert not load_model(None).trained


# --- HTTP ----------------------------------------------------------------------------


def test_index_and_static_files_are_served(client):
    assert "warcaby-zero" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    # browsers must not keep an old copy of the page after the code changes
    assert client.get("/static/app.js").headers["cache-control"] == "no-cache"
    assert client.get("/api/info").json()["size"] == 8


def test_new_game_as_white_waits_for_the_human(client):
    state = new_game(client)
    assert state["turn"] == "white" and state["ply"] == 0
    assert len(state["pieces"]) == 24
    assert len(state["legal"]) == 7
    assert state["players"]["white"] == {"name": "Ala", "ai": False}
    assert state["players"]["black"]["ai"] is True
    assert len(state["analysis"]["moves"]) == 7
    move = state["legal"][0]
    assert set(move) == {"id", "name", "path", "captures"}


def test_new_game_as_black_lets_the_ai_start(client):
    state = new_game(client, colour="black")
    assert state["ply"] == 1 and state["turn"] == "black"
    assert state["ai_move"]["by"] == "white"
    assert state["last_move"]["name"] == state["history"][0]


def test_move_gets_an_answer_and_illegal_requests_are_rejected(client):
    state = new_game(client)
    after = client.post("/api/move", json={"game": state["game"], "move": 2}).json()
    assert after["ply"] == 2 and after["turn"] == "white"
    assert after["history"][0] == state["legal"][2]["name"]
    assert client.get("/api/state", params={"game": state["game"]}).json()["ply"] == 2
    assert client.post("/api/move", json={"game": state["game"], "move": 999}).status_code == 409
    assert client.post("/api/move", json={"game": "nope", "move": 0}).status_code == 404
    assert client.post("/api/new", json={"colour": "green"}).status_code == 422


def test_analysis_can_be_skipped_and_requested_later(client):
    state = new_game(client, analysis=False)
    assert state["analysis"] is None
    analysed = client.post("/api/analyse", json={"game": state["game"]}).json()
    assert len(analysed["analysis"]["moves"]) == 7


def test_two_humans_take_turns_without_the_ai_moving(client):
    state = new_game(client, opponent="human", opponent_name="Olek")
    assert state["players"]["black"] == {"name": "Olek", "ai": False}
    after = client.post("/api/move", json={"game": state["game"], "move": 0}).json()
    assert after["ply"] == 1 and after["turn"] == "black"
    assert after["ai_move"] is None
    assert len(after["legal"]) == 7


def test_a_finished_game_is_recorded_once(client):
    state = new_game(client)
    game = state["game"]
    # resigning before any move leaves no trace
    untouched = new_game(client, name="Nikt")
    client.post("/api/resign", json={"game": untouched["game"]})
    assert client.get("/api/results").json()["recent"] == []

    client.post("/api/move", json={"game": game, "move": 0})
    over = client.post("/api/resign", json={"game": game}).json()
    assert over["result"] == {"winner": "black", "reason": "resignation"}
    assert over["legal"] == []
    client.post("/api/resign", json={"game": game})  # a second click changes nothing
    results = client.get("/api/results").json()
    assert len(results["recent"]) == 1
    assert results["recent"][0]["white"] == "Ala" and results["recent"][0]["black"] == "AI"
    assert results["recent"][0]["plies"] == 2
    assert results["standings"][0]["vs_ai"] == {"wins": 0, "draws": 0, "losses": 1}


def test_a_whole_game_can_be_played_through_the_api(client):
    state = new_game(client, analysis=False)
    for _ in range(400):
        if state["result"]:
            break
        state = client.post(
            "/api/move", json={"game": state["game"], "move": 0, "analysis": False}
        ).json()
    assert state["result"] is not None
    assert state["result"]["winner"] in ("white", "black", "draw")
    assert len(client.get("/api/results").json()["recent"]) == 1
