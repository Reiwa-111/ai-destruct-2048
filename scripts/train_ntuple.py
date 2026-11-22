#!/usr/bin/env python3
"""Train and independently validate the D4-symmetric N-tuple TD agent."""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from destroyer.game import Game2048, freeze_board, legal_actions  # noqa: E402
from destroyer.ntuple import NtupleValue, PositionPool  # noqa: E402


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * (p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) ** 0.5 / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def evaluate_greedy(value: NtupleValue, games: int, seed: int) -> dict:
    """Evaluate fresh games only; no exploration, restarts or learning."""
    scores: list[int] = []
    max_tiles: list[int] = []
    move_counts: list[int] = []

    for index in range(games):
        spawn_rng = random.Random(seed + index)
        game = Game2048(spawn_rng)
        while True:
            board = freeze_board(game.board)
            if not legal_actions(board):
                break
            action, _, _ = value.best_afterstate(board)
            if action is None:
                break
            game.move(action)
        scores.append(game.score)
        max_tiles.append(game.max_tile)
        move_counts.append(game.moves)

    successes = sum(tile >= 2048 for tile in max_tiles)
    low, high = wilson_interval(successes, games)
    return {
        "games": games,
        "seed_start": seed,
        "p2048": successes / games,
        "p2048_wilson_95": [low, high],
        "p4096": sum(tile >= 4096 for tile in max_tiles) / games,
        "p8192": sum(tile >= 8192 for tile in max_tiles) / games,
        "mean_score": statistics.fmean(scores),
        "median_score": statistics.median(scores),
        "mean_max_tile": statistics.fmean(max_tiles),
        "mean_moves": statistics.fmean(move_counts),
        "max_tile_overall": max(max_tiles),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=700_000)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--td-lambda", type=float, default=0.0,
                       help="Eligibility-trace parameter for TD(lambda); 0 preserves TD(0)")
    parser.add_argument("--epsilon-start", type=float, default=0.0)
    parser.add_argument("--epsilon-end", type=float, default=0.0)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "ntuple_latest.npz")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--eval-seed", type=int, default=1_000_000)
    parser.add_argument("--position-pool", type=Path, default=ROOT / "models" / "ntuple_positions.npz")
    parser.add_argument("--pool-capacity", type=int, default=50_000)
    parser.add_argument("--restart-prob", type=float, default=0.0)
    parser.add_argument("--restart-min-tile", type=int, default=2048)
    parser.add_argument("--capture-min-tile", type=int, default=512)
    parser.add_argument("--capture-every", type=int, default=8)
    parser.add_argument("--capture-positions", action="store_true",
                         help="Capture reachable training positions in Python for a later restart curriculum")
    args = parser.parse_args()

    positive = (args.episodes, args.checkpoint_every, args.log_every, args.eval_every,
                args.eval_games, args.pool_capacity, args.capture_every)
    if any(value < 1 for value in positive):
        parser.error("episode, interval, game and capacity values must be positive")
    if not 0.0 <= args.epsilon_end <= args.epsilon_start <= 1.0:
        parser.error("epsilon values must satisfy 0 <= end <= start <= 1")
    if not 0.0 <= args.restart_prob <= 1.0:
        parser.error("restart-prob must be in [0, 1]")
    if not 0.0 <= args.td_lambda <= 1.0:
        parser.error("td-lambda must be in [0, 1]")
    if args.restart_min_tile < 2 or args.restart_min_tile & (args.restart_min_tile - 1):
        parser.error("restart-min-tile must be a power of two >= 2")

    value = NtupleValue.load(args.resume) if args.resume else NtupleValue()
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.position_pool.parent.mkdir(parents=True, exist_ok=True)
    pool = (
        PositionPool.load(args.position_pool, capacity=args.pool_capacity)
        if args.position_pool.exists()
        else PositionPool(args.pool_capacity)
    )
    print(
        f"backend={value.backend} table_bytes={value.tables.nbytes} "
        f"patterns={len(value.tables)} tuple_size=6 pool={len(pool)} "
        f"position_capture={bool(args.capture_positions or args.restart_prob > 0)}",
        flush=True,
    )

    window_scores: list[int] = []
    window_tiles: list[int] = []
    window_td_errors: list[float] = []
    started = time.perf_counter()
    last_eval_episode = 0

    for episode in range(1, args.episodes + 1):
        progress = (episode - 1) / max(1, args.episodes - 1)
        epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * progress
        initial_board = None
        if len(pool) and rng.random() < args.restart_prob:
            try:
                initial_board = pool.sample(rng, min_tile=args.restart_min_tile)
            except LookupError:
                pass

        stats = value.train_episode(
            rng=rng,
            epsilon=epsilon,
            alpha=args.alpha,
            gamma=args.gamma,
            td_lambda=args.td_lambda,
            initial_board=initial_board,
            position_callback=pool.add if (args.capture_positions or args.restart_prob > 0) else None,
            capture_min_tile=args.capture_min_tile,
            capture_every=args.capture_every,
        )
        window_scores.append(stats.score)
        window_tiles.append(stats.max_tile)
        window_td_errors.append(stats.td_error_abs_mean)

        if episode % args.log_every == 0 or episode == 1:
            recent_scores = window_scores[-args.log_every:]
            recent_tiles = window_tiles[-args.log_every:]
            recent_td = window_td_errors[-args.log_every:]
            elapsed = time.perf_counter() - started
            print(
                f"episode={episode} epsilon={epsilon:.4f} "
                f"train_score_mean={statistics.fmean(recent_scores):.1f} "
                f"train_max_tile_mean={statistics.fmean(recent_tiles):.1f} "
                f"td_abs_mean={statistics.fmean(recent_td):.3f} "
                f"restart_prob={args.restart_prob:.3f} td_lambda={args.td_lambda:.3f} pool={len(pool)} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

        checkpoint_due = episode % args.checkpoint_every == 0
        evaluation_due = episode % args.eval_every == 0
        if checkpoint_due:
            value.save(args.output)
            pool.save(args.position_pool)
            print(f"checkpoint={args.output} pool_checkpoint={args.position_pool}", flush=True)

        if evaluation_due:
            report = evaluate_greedy(value, args.eval_games, args.eval_seed)
            report_text = " ".join(
                f"{key}={item:.4f}" if isinstance(item, float) else f"{key}={item}"
                for key, item in report.items()
            )
            print(f"GREEDY_EVAL episode={episode} {report_text}", flush=True)
            last_eval_episode = episode

    value.save(args.output)
    pool.save(args.position_pool)
    print(f"final_checkpoint={args.output} final_pool={args.position_pool}", flush=True)
    if last_eval_episode != args.episodes:
        report = evaluate_greedy(value, args.eval_games, args.eval_seed)
        report_text = " ".join(
            f"{key}={item:.4f}" if isinstance(item, float) else f"{key}={item}"
            for key, item in report.items()
        )
        print(f"FINAL_GREEDY_EVAL episode={args.episodes} {report_text}", flush=True)
    print(f"total_elapsed={time.perf_counter() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
