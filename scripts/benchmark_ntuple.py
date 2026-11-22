#!/usr/bin/env python3
"""Reproducible fresh-board greedy evaluation for an N-tuple checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from destroyer.game import Game2048, freeze_board, legal_actions  # noqa: E402
from destroyer.ntuple import NtupleValue  # noqa: E402


def wilson(successes: int, trials: int, z: float = 1.96) -> list[float]:
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * (p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) ** 0.5 / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.games < 1:
        parser.error("games must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    value = NtupleValue.load(args.checkpoint)
    records: list[dict] = []
    started = time.perf_counter()
    for index in range(args.games):
        game_started = time.perf_counter()
        game = Game2048(random.Random(args.seed + index))
        while True:
            board = freeze_board(game.board)
            if not legal_actions(board):
                break
            action, _, _ = value.best_afterstate(board)
            if action is None:
                break
            game.move(action)
        records.append({
            "seed": args.seed + index,
            "score": game.score,
            "max_tile": game.max_tile,
            "moves": game.moves,
            "elapsed_s": time.perf_counter() - game_started,
        })

    scores = [item["score"] for item in records]
    tiles = [item["max_tile"] for item in records]
    successes = sum(tile >= 2048 for tile in tiles)
    summary = {
        "agent": "N-TUPLE GREEDY",
        "checkpoint": str(args.checkpoint),
        "games": args.games,
        "seed_start": args.seed,
        "seed_end": args.seed + args.games - 1,
        "p2048": successes / args.games,
        "p2048_wilson_95": wilson(successes, args.games),
        "p4096": sum(tile >= 4096 for tile in tiles) / args.games,
        "p8192": sum(tile >= 8192 for tile in tiles) / args.games,
        "p16384": sum(tile >= 16384 for tile in tiles) / args.games,
        "mean_score": statistics.fmean(scores),
        "median_score": statistics.median(scores),
        "min_score": min(scores),
        "max_score": max(scores),
        "mean_moves": statistics.fmean(item["moves"] for item in records),
        "max_tile_overall": max(tiles),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report = {"summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"report={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
