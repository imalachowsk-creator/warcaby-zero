"""Start the app: ``python -m warcaby.app`` and open http://127.0.0.1:8000"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .server import create_app
from .session import ModelLibrary
from .storage import ResultStore


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play draughts against the trained network.")
    parser.add_argument(
        "--model", type=Path,
        help="a run directory or a gen_XXXX.pt file (default: the newest generation in runs/)",
    )  # fmt: skip
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="where runs are kept")
    parser.add_argument("--results", type=Path, default=Path("data/games.db"))
    parser.add_argument("--device", default="cpu", help="cpu is fastest for one game at a time")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    runs = args.runs
    if args.model is None and not any(runs.glob("*/generations/gen_*.pt")):
        # nothing trained here yet: use the network that ships with the repository
        runs = Path(__file__).resolve().parents[3] / "models"
    library = ModelLibrary.for_path(args.model, runs, args.device)
    model = library.get()
    if model.trained:
        print(f"default model: {model.label} ({len(library.entries())} generations to choose from)")
    else:
        print("no trained network found in runs/ - playing with plain search, without a network")
    app = create_app(library, ResultStore(args.results))
    print(f"open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
