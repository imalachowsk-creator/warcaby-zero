"""Results of the games played in the app, kept in a small SQLite file."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    played_at TEXT NOT NULL,
    white TEXT NOT NULL,
    black TEXT NOT NULL,
    white_is_ai INTEGER NOT NULL,
    black_is_ai INTEGER NOT NULL,
    winner TEXT NOT NULL,           -- 'white', 'black' or 'draw'
    reason TEXT NOT NULL,
    plies INTEGER NOT NULL,
    model TEXT,
    simulations INTEGER,
    moves TEXT NOT NULL
)
"""


class ResultStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def add(
        self,
        white: str,
        black: str,
        white_is_ai: bool,
        black_is_ai: bool,
        winner: str,
        reason: str,
        moves: list[str],
        model: str | None = None,
        simulations: int | None = None,
        played_at: str | None = None,
    ) -> int:
        played_at = played_at or datetime.now().isoformat(timespec="seconds")
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "INSERT INTO games (played_at, white, black, white_is_ai, black_is_ai, winner,"
                " reason, plies, model, simulations, moves) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    played_at, white, black, int(white_is_ai), int(black_is_ai), winner,
                    reason, len(moves), model, simulations, " ".join(moves),
                ),
            )  # fmt: skip
            return cursor.lastrowid

    def recent(self, limit: int = 20) -> list[dict]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM games ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def standings(self) -> list[dict]:
        """One row per human player: results against the AI and against people."""
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT * FROM games ORDER BY id").fetchall()
        table: dict[str, dict] = {}
        for row in rows:
            for colour, other in (("white", "black"), ("black", "white")):
                if row[f"{colour}_is_ai"]:
                    continue
                entry = table.setdefault(
                    row[colour],
                    {
                        "name": row[colour],
                        "games": 0,
                        "vs_ai": {"wins": 0, "draws": 0, "losses": 0},
                        "vs_people": {"wins": 0, "draws": 0, "losses": 0},
                        "last_played": row["played_at"],
                    },
                )
                bucket = entry["vs_ai" if row[f"{other}_is_ai"] else "vs_people"]
                if row["winner"] == "draw":
                    bucket["draws"] += 1
                elif row["winner"] == colour:
                    bucket["wins"] += 1
                else:
                    bucket["losses"] += 1
                entry["games"] += 1
                entry["last_played"] = row["played_at"]

        def points(entry: dict) -> float:
            return sum(
                entry[key]["wins"] + 0.5 * entry[key]["draws"] for key in ("vs_ai", "vs_people")
            )

        return sorted(table.values(), key=lambda e: (-points(e), -e["games"], e["name"]))
