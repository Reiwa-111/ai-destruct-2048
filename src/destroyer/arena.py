"""Headless evaluation arena shared by the GUI and benchmark tooling.

The arena owns the "play N reproducible games" workflow so the GUI never has
to reimplement agent selection, seed handling, progress reporting or the
statistics/confidence interval calculation.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .agent import DQNAgent
from .game import ACTIONS, Game2048, apply_move
from .ntuple import NtupleValue
from .search import Expectimax, PositionHeuristic

AGENT_DQN = "DQN"
AGENT_EXPECTIMAX = "EXPECTIMAX"
AGENT_NTUPLE = "N-TUPLE + EXPECTIMAX"
AGENT_RANDOM = "RANDOM"
AGENT_GREEDY = "GREEDY"
AGENT_MODES = (AGENT_DQN, AGENT_EXPECTIMAX, AGENT_NTUPLE)
ALL_AGENTS = AGENT_MODES + (AGENT_RANDOM, AGENT_GREEDY)
TILE_TARGETS = (2048, 4096, 8192, 16384, 32768, 65536)


class EvaluationCancelled(Exception):
    """Raised inside a game when the caller cancels an evaluation."""


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if trials <= 0:
        return 0.0, 0.0
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * ((p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) ** 0.5) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


@dataclass(frozen=True)
class GameRecord:
    seed: int
    score: int
    max_tile: int
    moves: int
    merges: int
    search_nodes: int = 0
    search_cache_hits: int = 0
    completed_depth_sum: int = 0
    elapsed_s: float = 0.0
    search_latency_mean_ms: float = 0.0
    search_latency_p50_ms: float = 0.0
    search_latency_p95_ms: float = 0.0
    search_latency_p99_ms: float = 0.0
    search_latency_max_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationReport:
    agent: str
    games_requested: int
    seed_start: int
    records: list[GameRecord]
    elapsed_s: float
    cancelled: bool = False
    config: dict = field(default_factory=dict)

    @property
    def games(self) -> int:
        return len(self.records)

    @property
    def scores(self) -> list[int]:
        return [record.score for record in self.records]

    @property
    def tiles(self) -> list[int]:
        return [record.max_tile for record in self.records]

    def tile_rate(self, target: int) -> float:
        if not self.records:
            return 0.0
        return sum(tile >= target for tile in self.tiles) / len(self.records)

    def summary(self) -> dict:
        if not self.records:
            return {
                "agent": self.agent,
                "games": 0,
                "games_requested": self.games_requested,
                "seed_start": self.seed_start,
                "cancelled": self.cancelled,
                "elapsed_seconds": self.elapsed_s,
                "config": self.config,
            }
        scores = self.scores
        tiles = self.tiles
        moves = [record.moves for record in self.records]
        successes = sum(tile >= 2048 for tile in tiles)
        low, high = wilson_interval(successes, len(self.records))
        return {
            "agent": self.agent,
            "games": len(self.records),
            "games_requested": self.games_requested,
            "seed_start": self.seed_start,
            "seed_end": self.seed_start + len(self.records) - 1,
            "cancelled": self.cancelled,
            "mean_score": statistics.fmean(scores),
            "median_score": statistics.median(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "mean_moves": statistics.fmean(moves),
            "max_tile_overall": max(tiles),
            "tile_rates": {str(target): self.tile_rate(target) for target in TILE_TARGETS},
            "p2048_wilson_95": [low, high],
            "mean_search_nodes_per_game": statistics.fmean(
                record.search_nodes for record in self.records
            ),
            "mean_search_cache_hits_per_game": statistics.fmean(
                record.search_cache_hits for record in self.records
            ),
            "mean_completed_search_depth": statistics.fmean(
                record.completed_depth_sum / max(1, record.moves) for record in self.records
            ),
            "mean_search_latency_ms": statistics.fmean(
                record.search_latency_mean_ms for record in self.records
            ),
            "mean_search_latency_p95_ms": statistics.fmean(
                record.search_latency_p95_ms for record in self.records
            ),
            "max_search_latency_ms": max(
                record.search_latency_max_ms for record in self.records
            ),
            "elapsed_seconds": self.elapsed_s,
            "games_per_second": len(self.records) / self.elapsed_s if self.elapsed_s > 0 else 0.0,
            "config": self.config,
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.summary(),
            "config": self.config,
            "records": [record.to_dict() for record in self.records],
        }
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return destination


def create_search(
    agent: str,
    *,
    ntuple: NtupleValue | None = None,
    depth: int = 3,
    node_budget: int = 500_000,
    time_budget_ms: float | None = 35.0,
    downgrade_threshold: int | None = None,
    downgrade_steps: int = 1,
) -> Expectimax | None:
    """Build the search object required by an agent name, if any."""

    if agent == AGENT_EXPECTIMAX:
        return Expectimax(
            PositionHeuristic(),
            max_depth=depth,
            node_budget=node_budget,
            time_budget_ms=time_budget_ms,
            downgrade_threshold=downgrade_threshold,
            downgrade_steps=downgrade_steps,
        )
    if agent == AGENT_NTUPLE:
        if ntuple is None:
            raise ValueError("N-tuple evaluator is required for the N-TUPLE agent")
        return Expectimax(
            ntuple.state_value,
            max_depth=depth,
            node_budget=node_budget,
            time_budget_ms=time_budget_ms,
            fallback_evaluator=ntuple.value,
            downgrade_threshold=downgrade_threshold,
            downgrade_steps=downgrade_steps,
        )
    if agent in (AGENT_DQN, AGENT_RANDOM, AGENT_GREEDY):
        return None
    raise ValueError(f"Unknown agent: {agent}")


def _greedy_action(game: Game2048, ntuple: NtupleValue | None, heuristic: PositionHeuristic) -> str:
    best_action = game.legal_actions()[0]
    best_value = -float("inf")
    for action in game.legal_actions():
        afterstate = apply_move(game.board, action)
        if ntuple is not None:
            value = afterstate.score_delta + ntuple.value(afterstate.board)
        else:
            value = heuristic(afterstate.board)
        if value > best_value:
            best_action, best_value = action, value
    return best_action


def play_game(
    agent: str,
    seed: int,
    *,
    dqn_agent: DQNAgent | None = None,
    search: Expectimax | None = None,
    ntuple: NtupleValue | None = None,
    cancel=None,
) -> GameRecord:
    """Play one reproducible game from a fresh board."""

    if agent not in ALL_AGENTS:
        raise ValueError(f"Unknown agent: {agent}")
    if agent == AGENT_DQN and dqn_agent is None:
        raise ValueError("dqn_agent is required for the DQN agent")
    if agent in (AGENT_EXPECTIMAX, AGENT_NTUPLE) and search is None:
        raise ValueError(f"search is required for the {agent} agent")

    spawn_rng = random.Random(seed)
    action_rng = random.Random(seed ^ 0x9E3779B97F4A7C15)
    game = Game2048(spawn_rng)
    heuristic = PositionHeuristic()
    search_nodes = 0
    search_cache_hits = 0
    completed_depth_sum = 0
    search_latencies_ms: list[float] = []
    started = time.perf_counter()

    while True:
        if cancel is not None and cancel.is_set():
            raise EvaluationCancelled
        if agent == AGENT_RANDOM:
            legal = game.legal_actions()
            if not legal:
                break
            action = action_rng.choice(legal)
        elif agent == AGENT_GREEDY:
            legal = game.legal_actions()
            if not legal:
                break
            action = _greedy_action(game, ntuple, heuristic)
        elif agent == AGENT_DQN:
            legal = game.legal_actions()
            if not legal:
                break
            assert dqn_agent is not None
            action = ACTIONS[dqn_agent.act(game.board, legal, explore=False)]
        else:
            assert search is not None
            result = search.choose(game.board)
            if result is None:
                break
            if result.action not in ACTIONS:
                raise RuntimeError(f"{agent} returned an invalid action during evaluation")
            action = result.action
            search_nodes += result.nodes
            search_cache_hits += result.cache_hits
            completed_depth_sum += result.completed_depth
            search_latencies_ms.append(result.elapsed_ms)

        game.move(action)

    if search_latencies_ms:
        ordered_latencies = sorted(search_latencies_ms)

        def percentile(fraction: float) -> float:
            index = max(0, min(len(ordered_latencies) - 1,
                              math.ceil(fraction * len(ordered_latencies)) - 1))
            return float(ordered_latencies[index])

        latency_mean = statistics.fmean(ordered_latencies)
        latency_p50 = percentile(0.50)
        latency_p95 = percentile(0.95)
        latency_p99 = percentile(0.99)
        latency_max = ordered_latencies[-1]
    else:
        latency_mean = latency_p50 = latency_p95 = latency_p99 = latency_max = 0.0

    return GameRecord(
        seed=seed,
        score=game.score,
        max_tile=game.max_tile,
        moves=game.moves,
        merges=game.merges,
        search_nodes=search_nodes,
        search_cache_hits=search_cache_hits,
        completed_depth_sum=completed_depth_sum,
        elapsed_s=time.perf_counter() - started,
        search_latency_mean_ms=latency_mean,
        search_latency_p50_ms=latency_p50,
        search_latency_p95_ms=latency_p95,
        search_latency_p99_ms=latency_p99,
        search_latency_max_ms=latency_max,
    )


def evaluate(
    agent: str,
    games: int,
    *,
    seed_start: int = 1_000_000,
    dqn_agent: DQNAgent | None = None,
    ntuple: NtupleValue | None = None,
    depth: int = 3,
    node_budget: int = 500_000,
    time_budget_ms: float | None = 35.0,
    downgrade_threshold: int | None = None,
    downgrade_steps: int = 1,
    cancel=None,
    on_game: Callable[[int, int, GameRecord], None] | None = None,
    search: Expectimax | None = None,
) -> EvaluationReport:
    """Evaluate an agent on a fixed seed range and return a full report."""

    if games < 1:
        raise ValueError("games must be positive")
    if agent not in ALL_AGENTS:
        raise ValueError(f"Unknown agent: {agent}")
    if agent == AGENT_DQN and dqn_agent is None:
        raise ValueError("dqn_agent is required for the DQN agent")
    if agent == AGENT_NTUPLE and ntuple is None:
        raise ValueError("ntuple is required for the N-TUPLE agent")

    own_search = search is None and agent in (AGENT_EXPECTIMAX, AGENT_NTUPLE)
    if own_search:
        search = create_search(
            agent,
            ntuple=ntuple,
            depth=depth,
            node_budget=node_budget,
            time_budget_ms=time_budget_ms,
            downgrade_threshold=downgrade_threshold,
            downgrade_steps=downgrade_steps,
        )

    started = time.perf_counter()
    records: list[GameRecord] = []
    cancelled = False
    try:
        for index in range(games):
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            try:
                record = play_game(
                    agent,
                    seed_start + index,
                    dqn_agent=dqn_agent,
                    search=search,
                    ntuple=ntuple,
                    cancel=cancel,
                )
            except EvaluationCancelled:
                cancelled = True
                break
            records.append(record)
            if on_game is not None:
                on_game(index + 1, games, record)
    finally:
        if own_search and search is not None:
            search.close()

    return EvaluationReport(
        agent=agent,
        games_requested=games,
        seed_start=seed_start,
        records=records,
        elapsed_s=time.perf_counter() - started,
        cancelled=cancelled,
        config={
            "depth": depth,
            "node_budget": node_budget,
            "time_budget_ms": time_budget_ms,
            "ntuple": ntuple is not None,
            "downgrade_threshold": downgrade_threshold,
            "downgrade_steps": downgrade_steps,
        },
    )
