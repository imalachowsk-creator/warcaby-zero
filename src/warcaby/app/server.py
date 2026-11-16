"""The web server of the app: a thin JSON layer over ``GameSession``."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .session import GameSession, Model
from .storage import ResultStore

STATIC = Path(__file__).parent / "static"
AI_NAME = "AI"


class NewGame(BaseModel):
    #: "ai" to play the network, "human" for two people at one computer
    opponent: str = "ai"
    #: the colour of the (first) human: "white" or "black"
    colour: str = "white"
    name: str = Field(default="Gość", max_length=40)
    #: the second human's name when ``opponent`` is "human"
    opponent_name: str = Field(default="Znajomy", max_length=40)
    simulations: int = Field(default=200, ge=2, le=5000)
    analysis: bool = True


class MoveRequest(BaseModel):
    game: str
    move: int
    analysis: bool = True


class GameRequest(BaseModel):
    game: str
    analysis: bool = True


def create_app(model: Model, store: ResultStore) -> FastAPI:
    app = FastAPI(title="warcaby-zero")
    sessions: dict[str, GameSession] = {}
    # one search at a time: the network is shared by all games
    thinking = threading.Lock()

    @app.middleware("http")
    async def no_cache(request, call_next):
        # the page and its script change with the code; never serve a stale copy
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response

    def session(game_id: str) -> GameSession:
        if game_id not in sessions:
            raise HTTPException(404, "unknown game")
        return sessions[game_id]

    def respond(current: GameSession, analysis: bool) -> dict:
        """Let the AI move if it is due, analyse, record a finished game."""
        with thinking:
            current.play_ai_if_due()
            if analysis and not current.over and current.analysis is None:
                current.analyse()
        result = current.result()
        if result and not current.saved and current.history:
            current.saved = True
            store.add(
                white=current.names[0],
                black=current.names[1],
                white_is_ai=current.ai_seats[0],
                black_is_ai=current.ai_seats[1],
                winner=result["winner"],
                reason=result["reason"],
                moves=[current.game.move_name(move) for move in current.history],
                model=model.label if any(current.ai_seats) else None,
                simulations=current.simulations if any(current.ai_seats) else None,
            )
        return current.to_dict()

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/info")
    def info() -> dict:
        return {"model": {"label": model.label, "trained": model.trained}, "size": model.game.size}

    @app.post("/api/new")
    def new_game(request: NewGame) -> dict:
        if request.colour not in ("white", "black") or request.opponent not in ("ai", "human"):
            raise HTTPException(422, "bad colour or opponent")
        name = request.name.strip() or "Gość"
        human_is_white = request.colour == "white"
        if request.opponent == "ai":
            other, ai_seats = AI_NAME, (not human_is_white, human_is_white)
        else:
            other, ai_seats = request.opponent_name.strip() or "Znajomy", (False, False)
        white, black = (name, other) if human_is_white else (other, name)
        current = GameSession(
            uuid.uuid4().hex[:12], model, white, black, ai_seats, request.simulations
        )
        sessions[current.id] = current
        with current.lock:
            return respond(current, request.analysis)

    @app.get("/api/state")
    def state(game: str) -> dict:
        return session(game).to_dict()

    @app.post("/api/move")
    def move(request: MoveRequest) -> dict:
        current = session(request.game)
        with current.lock:
            try:
                current.play_human(request.move)
            except ValueError as error:
                raise HTTPException(409, str(error)) from None
            return respond(current, request.analysis)

    @app.post("/api/analyse")
    def analyse(request: GameRequest) -> dict:
        current = session(request.game)
        with current.lock:
            return respond(current, analysis=True)

    @app.post("/api/resign")
    def resign(request: GameRequest) -> dict:
        current = session(request.game)
        with current.lock:
            current.resign()
            return respond(current, analysis=False)

    @app.get("/api/results")
    def results() -> dict:
        return {"standings": store.standings(), "recent": store.recent(15)}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
