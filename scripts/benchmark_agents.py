#!/usr/bin/env python3
"""Reproducible headless evaluation for 2048 baseline and hybrid agents."""

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

from destroyer.agent import DQNAgent  # noqa: E402
from destroyer.game import ACTIONS, Game2048, apply_move  # noqa: E402
from destroyer.ntuple import NtupleValue  # noqa: E402
from destroyer.search import Expectimax, PositionHeuristic  # noqa: E402


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * ((p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) ** 0.5) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def play_one(agent_name: str, seed: int, evaluator, afterstate_evaluator,
             depth: int, node_budget: int, time_budget_ms: float, dqn_agent=None):
    spawn_rng = random.Random(seed)
    action_rng = random.Random(seed ^ 0x9E3779B97F4A7C15)
    game = Game2048(spawn_rng)
    search = Expectimax(
        evaluator,
        max_depth=depth,
        node_budget=node_budget,
        time_budget_ms=time_budget_ms,
        include_merge_reward=afterstate_evaluator is not evaluator,
        fallback_evaluator=afterstate_evaluator,
    )
    search_nodes = 0
    search_cache_hits = 0
    completed_depths = 0

    while True:
        legal = game.legal_actions()
        if not legal:
            break

        if agent_name == "random":
            action = action_rng.choice(legal)
        elif agent_name == "greedy":
            candidates = [(candidate, apply_move(game.board, candidate)) for candidate in legal]
            add_reward = afterstate_evaluator is not evaluator
            action = max(candidates, key=lambda item: afterstate_evaluator(item[1].board) + (item[1].score_delta if add_reward else 0.0))[0]
        elif agent_name == "expectimax":
            result = search.choose(game.board)
            action = result.action
            search_nodes += result.nodes
            search_cache_hits += result.cache_hits
            completed_depths += result.completed_depth
        elif agent_name == "dqn":
            action = ACTIONS[dqn_agent.act(game.board, legal, explore=False)]
        else:
            raise ValueError(f"Unknown agent: {agent_name}")

        game.move(action)

    return {
        "seed": seed,
        "score": game.score,
        "max_tile": game.max_tile,
        "moves": game.moves,
        "merges": game.merges,
        "search_nodes": search_nodes,
        "search_cache_hits": search_cache_hits,
        "completed_depth_sum": completed_depths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("random", "greedy", "expectimax", "dqn"), default="expectimax")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20_480)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--node-budget", type=int, default=250_000)
    parser.add_argument("--time-budget-ms", type=float, default=100.0)
    parser.add_argument("--ntuple", type=Path, help="Optional trained N-tuple .npz evaluator")
    parser.add_argument("--dqn-checkpoint", type=Path, help="DQN .pt checkpoint (required for --agent dqn)")
    parser.add_argument("--output", type=Path, help="Optional JSONL file for per-game results")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    if args.games < 1 or args.progress_every < 1:
        parser.error("games and progress-every must be positive")
    if args.agent == "dqn" and not args.dqn_checkpoint:
        parser.error("--dqn-checkpoint is required when --agent dqn")

    if args.ntuple:
        ntuple = NtupleValue.load(args.ntuple)
        evaluator = ntuple.state_value
        afterstate_evaluator = ntuple.value
    else:
        evaluator = PositionHeuristic()
        afterstate_evaluator = evaluator
    dqn_agent = None
    if args.agent == "dqn":
        dqn_agent = DQNAgent(device="cpu")
        dqn_agent.load(args.dqn_checkpoint)
    started = time.perf_counter()
    results = []
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_stream = args.output.open("w", encoding="utf-8")
    else:
        output_stream = None
    try:
        for index in range(args.games):
            result = play_one(
                args.agent, args.seed + index, evaluator, afterstate_evaluator,
                args.depth, args.node_budget, args.time_budget_ms, dqn_agent
            )
            results.append(result)
            if output_stream:
                output_stream.write(json.dumps(result) + "\n")
                output_stream.flush()
            if (index + 1) % args.progress_every == 0 or index + 1 == args.games:
                elapsed_now = time.perf_counter() - started
                successes = sum(item["max_tile"] >= 2048 for item in results)
                print(
                    f"progress={index + 1}/{args.games} p2048={successes}/{index + 1} "
                    f"elapsed={elapsed_now:.1f}s",
                    flush=True,
                )
    finally:
        if output_stream:
            output_stream.close()
    elapsed = time.perf_counter() - started

    tiles = [item["max_tile"] for item in results]
    scores = [item["score"] for item in results]
    rates = {
        str(target): sum(tile >= target for tile in tiles) / args.games
        for target in (2048, 4096, 8192, 16384, 32768, 65536)
    }
    low, high = wilson_interval(sum(tile >= 2048 for tile in tiles), args.games)
    report = {
        "agent": args.agent,
        "games": args.games,
        "seed_start": args.seed,
        "mean_score": statistics.fmean(scores),
        "median_score": statistics.median(scores),
        "mean_moves": statistics.fmean(item["moves"] for item in results),
        "max_tile_overall": max(tiles),
        "tile_rates": rates,
        "p2048_wilson_95": [low, high],
        "mean_search_nodes_per_game": statistics.fmean(item["search_nodes"] for item in results),
        "mean_search_cache_hits_per_game": statistics.fmean(item["search_cache_hits"] for item in results),
        "mean_completed_search_depth": statistics.fmean(item["completed_depth_sum"] / max(1, item["moves"]) for item in results),
        "elapsed_seconds": elapsed,
    }
    print(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
