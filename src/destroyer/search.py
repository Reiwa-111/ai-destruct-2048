from __future__ import annotations

from dataclasses import dataclass
import ctypes
from math import inf
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import numpy as np

from .game import ACTIONS, Board, apply_move, freeze_board, legal_actions, spawn_outcomes

ValueFunction = Callable[[Board], float]

_NATIVE_SEARCH_LIBRARY = None
_NATIVE_SEARCH_CHECKED = False


def _load_native_search():
    global _NATIVE_SEARCH_LIBRARY, _NATIVE_SEARCH_CHECKED
    if _NATIVE_SEARCH_CHECKED:
        return _NATIVE_SEARCH_LIBRARY
    _NATIVE_SEARCH_CHECKED = True
    path = Path(__file__).resolve().parents[2] / "native" / "libdestroyer_ntuple.so"
    if not path.is_file():
        return None
    try:
        library = ctypes.CDLL(str(path))
        byte_ptr = ctypes.POINTER(ctypes.c_uint8)
        float_ptr = ctypes.POINTER(ctypes.c_float)
        uint64_ptr = ctypes.POINTER(ctypes.c_uint64)

        library.destroyer_search_create.argtypes = []
        library.destroyer_search_create.restype = ctypes.c_void_p
        library.destroyer_search_destroy.argtypes = [ctypes.c_void_p]
        library.destroyer_search_destroy.restype = None
        library.destroyer_search_clear.argtypes = [ctypes.c_void_p]
        library.destroyer_search_clear.restype = None
        library.destroyer_search_choose.argtypes = [
            ctypes.c_void_p, byte_ptr, float_ptr, ctypes.c_int,
            ctypes.c_int, ctypes.c_uint64, ctypes.c_double,
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int), uint64_ptr, uint64_ptr,
            ctypes.POINTER(ctypes.c_int),
        ]
        library.destroyer_search_choose.restype = ctypes.c_int
        library.destroyer_apply_move.argtypes = [
            byte_ptr, ctypes.c_int, byte_ptr, uint64_ptr,
            ctypes.POINTER(ctypes.c_int),
        ]
        library.destroyer_apply_move.restype = ctypes.c_int
        library.destroyer_classic_heuristic.argtypes = [byte_ptr]
        library.destroyer_classic_heuristic.restype = ctypes.c_float
        library.destroyer_ntuple_afterstate_value.argtypes = [byte_ptr, float_ptr]
        library.destroyer_ntuple_afterstate_value.restype = ctypes.c_float
        library.destroyer_ntuple_state_value.argtypes = [byte_ptr, float_ptr]
        library.destroyer_ntuple_state_value.restype = ctypes.c_float
        _NATIVE_SEARCH_LIBRARY = library
    except (OSError, AttributeError):
        _NATIVE_SEARCH_LIBRARY = None
    return _NATIVE_SEARCH_LIBRARY


def board_to_ranks(board: Iterable[Iterable[int]]) -> np.ndarray:
    current = freeze_board(board)
    ranks = [0 if value == 0 else min(value.bit_length() - 1, 31)
             for row in current for value in row]
    return np.asarray(ranks, dtype=np.uint8)


def downgrade_board(board: Iterable[Iterable[int]], *, threshold: int = 32768) -> Board:
    """TDL-style root downgrade: halve tiles above the largest missing tile.

    Applied once before search. Empty cells and merge structure are preserved.
    """
    if threshold < 2 or threshold & (threshold - 1):
        raise ValueError("threshold must be a power-of-two tile >= 2")
    current = freeze_board(board)
    ranks = {tile.bit_length() - 1 for row in current for tile in row if tile}
    maximum = max(ranks, default=0)
    if maximum < threshold.bit_length() - 1 or maximum <= 1:
        return current
    missing_rank = next((rank for rank in range(maximum - 1, 0, -1) if rank not in ranks), None)
    if missing_rank is None:
        return current
    return tuple(
        tuple(tile if tile == 0 or tile.bit_length() - 1 <= missing_rank else tile // 2
              for tile in row)
        for row in current
    )


def native_apply_move(board: Iterable[Iterable[int]], action: str):
    """Testing/diagnostic adapter for the native line-lookup move engine."""
    library = _load_native_search()
    if library is None:
        raise RuntimeError("Native search library is unavailable")
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}")
    ranks = board_to_ranks(board)
    output = np.empty(16, dtype=np.uint8)
    reward = ctypes.c_uint64()
    merges = ctypes.c_int()
    moved = library.destroyer_apply_move(
        ranks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ACTIONS.index(action),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.byref(reward),
        ctypes.byref(merges),
    )
    decoded = tuple(
        tuple(0 if int(output[row * 4 + col]) == 0 else 1 << int(output[row * 4 + col])
              for col in range(4))
        for row in range(4)
    )
    return decoded, int(reward.value), int(merges.value), bool(moved)


def native_ntuple_value(board: Iterable[Iterable[int]], tables: np.ndarray, *, afterstate: bool = True) -> float:
    """Evaluate an afterstate or player-to-move state with the native tuple kernel."""
    library = _load_native_search()
    if library is None:
        raise RuntimeError("Native search library is unavailable")
    expected_shape = (4, 16 ** 6)
    if tables.shape != expected_shape or tables.dtype != np.float32 or not tables.flags.c_contiguous:
        raise ValueError(f"N-tuple tables must be contiguous float32 with shape {expected_shape}")
    ranks = board_to_ranks(board)
    function = (
        library.destroyer_ntuple_afterstate_value
        if afterstate else library.destroyer_ntuple_state_value
    )
    return float(function(
        ranks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        tables.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    ))


@dataclass(frozen=True)
class SearchResult:
    action: str
    value: float
    completed_depth: int
    nodes: int
    complete: bool
    cache_hits: int = 0
    elapsed_ms: float = 0.0


class _SearchBudgetExceeded(Exception):
    pass


class PositionHeuristic:
    """Classic row/column heuristic baseline inspired by nneonneo's solver.

    This is a hand-written search evaluator, not a learned value function.
    It uses a scale separate from the game's merge score, so Expectimax must
    not add immediate merge rewards when this evaluator is selected.
    """

    includes_merge_reward = False
    terminal_value = 0.0
    lost_penalty = 200_000.0
    monotonicity_power = 4.0
    monotonicity_weight = 47.0
    sum_power = 3.5
    sum_weight = 11.0
    merges_weight = 700.0
    empty_weight = 270.0

    @staticmethod
    def _rank(value: int) -> int:
        return 0 if value == 0 else value.bit_length() - 1

    def __call__(self, board: Iterable[Iterable[int]]) -> float:
        current = freeze_board(board)
        grid = [[self._rank(value) for value in row] for row in current]
        def line_score(line: list[int]) -> float:
            empty = 0
            merges = 0
            previous = 0
            merge_counter = 0
            tile_sum = 0.0
            mono_left = 0.0
            mono_right = 0.0

            for rank in line:
                tile_sum += rank ** self.sum_power
                if rank == 0:
                    empty += 1
                else:
                    if previous == rank:
                        merge_counter += 1
                    elif merge_counter:
                        merges += 1 + merge_counter
                        merge_counter = 0
                    previous = rank

            if merge_counter:
                merges += 1 + merge_counter

            for index in range(1, 4):
                previous_rank = line[index - 1]
                rank = line[index]
                if previous_rank > rank:
                    mono_left += previous_rank ** self.monotonicity_power - rank ** self.monotonicity_power
                else:
                    mono_right += rank ** self.monotonicity_power - previous_rank ** self.monotonicity_power

            return (
                self.empty_weight * empty
                + self.merges_weight * merges
                - self.monotonicity_weight * min(mono_left, mono_right)
                - self.sum_weight * tile_sum
            )

        columns = [[grid[row][col] for row in range(4)] for col in range(4)]
        if not any(value == 0 for row in current for value in row) and not any(
            current[row][col] == current[row + 1][col]
            for row in range(3) for col in range(4)
        ) and not any(
            current[row][col] == current[row][col + 1]
            for row in range(4) for col in range(3)
        ):
            return 0.0
        return 8.0 * self.lost_penalty + sum(line_score(line) for line in grid + columns)


class Expectimax:
    """Expectimax search for stochastic 2048, with an injected leaf evaluator.

    depth counts player decisions. Each action is followed by the exact chance
    distribution of the next 2/4 tile. Search is iterative-deepening and keeps
    the last fully completed result if the node budget interrupts deeper work.
    """

    def __init__(
        self,
        evaluator: ValueFunction,
        max_depth: int = 3,
        node_budget: int = 250_000,
        time_budget_ms: float | None = None,
        include_merge_reward: bool | None = None,
        max_cache_entries: int = 500_000,
        fallback_evaluator: ValueFunction | None = None,
        native: bool = True,
        downgrade_threshold: int | None = None,
        downgrade_steps: int = 1,
    ):
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if node_budget < 1:
            raise ValueError("node_budget must be positive")
        if time_budget_ms is not None and time_budget_ms <= 0:
            raise ValueError("time_budget_ms must be positive when supplied")
        if downgrade_threshold is not None and (
            downgrade_threshold < 2 or downgrade_threshold & (downgrade_threshold - 1)
        ):
            raise ValueError("downgrade_threshold must be a power-of-two tile >= 2")
        if downgrade_steps != 1:
            raise ValueError("root tile-downgrading supports one rank shift")
        self.evaluator = evaluator
        evaluator_owner = getattr(evaluator, "__self__", None)
        evaluator_tables = getattr(evaluator_owner, "tables", None)
        self._uses_ntuple_evaluator = (
            isinstance(evaluator_tables, np.ndarray)
            and evaluator_tables.shape == (4, 16 ** 6)
            and evaluator_tables.dtype == np.float32
            and evaluator_tables.flags.c_contiguous
        )
        if downgrade_threshold is not None and not self._uses_ntuple_evaluator:
            raise ValueError("tile downgrading is supported only with an N-tuple evaluator")
        self.max_depth = int(max_depth)
        self.node_budget = int(node_budget)
        self.time_budget_s = None if time_budget_ms is None else float(time_budget_ms) / 1000.0
        self.include_merge_reward = (
            bool(getattr(evaluator, "includes_merge_reward", True))
            if include_merge_reward is None else bool(include_merge_reward)
        )
        self.fallback_evaluator = fallback_evaluator or evaluator
        self.downgrade_threshold = downgrade_threshold
        self.downgrade_steps = int(downgrade_steps)
        self.max_cache_entries = max(1, int(max_cache_entries))
        self._native_library = None
        self._native_context = None
        self._native_tables = None
        self.native_mode = None
        if native:
            owner = getattr(evaluator, "__self__", None)
            if isinstance(evaluator, PositionHeuristic):
                self.native_mode = "classic"
            elif owner is not None and isinstance(getattr(owner, "tables", None), np.ndarray):
                candidate_tables = owner.tables
                if (
                    candidate_tables.shape == (4, 16 ** 6)
                    and candidate_tables.dtype == np.float32
                    and candidate_tables.flags.c_contiguous
                ):
                    self.native_mode = "ntuple"
                    self._native_tables = candidate_tables
            if self.native_mode is not None:
                self._native_library = _load_native_search()
                if self._native_library is not None:
                    self._native_context = self._native_library.destroyer_search_create()
                else:
                    self.native_mode = None
        self.nodes = 0
        self.cache_hits = 0
        self._player_cache: dict[tuple[Board, int], float] = {}
        self._chance_cache: dict[tuple[Board, int], float] = {}
        self._deadline = inf

    @property
    def backend(self) -> str:
        """Human-readable search backend description."""

        return f"native-{self.native_mode}" if self.native_mode else "python"

    @property
    def cache_entries(self) -> int:
        return len(self._player_cache) + len(self._chance_cache)

    def configure(
        self,
        *,
        max_depth: int | None = None,
        node_budget: int | None = None,
        time_budget_ms: float | None | object = ...,
    ) -> None:
        """Apply new search limits without discarding the transposition cache."""

        if max_depth is not None:
            if max_depth < 1:
                raise ValueError("max_depth must be at least 1")
            self.max_depth = int(max_depth)
        if node_budget is not None:
            if node_budget < 1:
                raise ValueError("node_budget must be positive")
            self.node_budget = int(node_budget)
        if time_budget_ms is not ...:
            if time_budget_ms is not None and time_budget_ms <= 0:
                raise ValueError("time_budget_ms must be positive when supplied")
            self.time_budget_s = None if time_budget_ms is None else float(time_budget_ms) / 1000.0

    def _visit(self) -> None:
        self.nodes += 1
        if self.nodes > self.node_budget:
            raise _SearchBudgetExceeded
        # perf_counter() is comparatively expensive; checking every 32 nodes
        # keeps the budget accurate while making deep Python searches faster.
        if (self.nodes & 31) == 0 and perf_counter() >= self._deadline:
            raise _SearchBudgetExceeded

    def clear_cache(self) -> None:
        """Clear cached values, e.g. after updating/reloading a learned evaluator."""
        self._player_cache.clear()
        self._chance_cache.clear()
        if self._native_context:
            self._native_library.destroyer_search_clear(self._native_context)

    def close(self) -> None:
        if self._native_context:
            self._native_library.destroyer_search_destroy(self._native_context)
            self._native_context = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _choose_native(self, current: Board) -> SearchResult | None:
        started = perf_counter()
        ranks = board_to_ranks(current)
        tables_ptr = None
        use_ntuple = self.native_mode == "ntuple"
        if use_ntuple:
            tables_ptr = self._native_tables.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        action = ctypes.c_int()
        value = ctypes.c_double()
        depth = ctypes.c_int()
        nodes = ctypes.c_uint64()
        cache_hits = ctypes.c_uint64()
        complete = ctypes.c_int()
        time_budget_ms = 0.0 if self.time_budget_s is None else self.time_budget_s * 1000.0
        downgrade_rank = 0 if self.downgrade_threshold is None else self.downgrade_threshold.bit_length() - 1
        ok = self._native_library.destroyer_search_choose(
            self._native_context,
            ranks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            tables_ptr,
            int(use_ntuple),
            self.max_depth,
            self.node_budget,
            time_budget_ms,
            downgrade_rank,
            self.downgrade_steps if downgrade_rank else 0,
            ctypes.byref(action),
            ctypes.byref(value),
            ctypes.byref(depth),
            ctypes.byref(nodes),
            ctypes.byref(cache_hits),
            ctypes.byref(complete),
        )
        if not ok:
            return None
        self.nodes = int(nodes.value)
        self.cache_hits = int(cache_hits.value)
        return SearchResult(
            action=ACTIONS[action.value],
            value=float(value.value),
            completed_depth=int(depth.value),
            nodes=self.nodes,
            complete=bool(complete.value),
            cache_hits=self.cache_hits,
            elapsed_ms=(perf_counter() - started) * 1000.0,
        )

    def _player_value(self, board: Board, decisions_left: int) -> float:
        key = (board, decisions_left)
        cached = self._player_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        self._visit()
        actions = legal_actions(board)
        if not actions:
            terminal_value = float(getattr(self.evaluator, "terminal_value", 0.0))
            self._player_cache[key] = terminal_value
            return terminal_value
        if decisions_left <= 0:
            value = float(self.evaluator(board))
            self._player_cache[key] = value
            return value

        best = -inf
        for action in actions:
            afterstate = apply_move(board, action)
            continuation = self._chance_value(afterstate.board, decisions_left - 1)
            value = continuation + (afterstate.score_delta if self.include_merge_reward else 0.0)
            if value > best:
                best = value
        self._player_cache[key] = best
        return best

    def _chance_value(self, afterstate: Board, decisions_left: int) -> float:
        key = (afterstate, decisions_left)
        cached = self._chance_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self._visit()
        outcomes = spawn_outcomes(afterstate)
        expected = 0.0
        for probability, spawned_board in outcomes:
            expected += probability * self._player_value(spawned_board, decisions_left)
        self._chance_cache[key] = expected
        return expected

    def choose(self, board: Iterable[Iterable[int]]) -> SearchResult | None:
        started = perf_counter()
        current = freeze_board(board)
        if self._native_context:
            native_result = self._choose_native(current)
            if native_result is not None:
                return SearchResult(
                    action=native_result.action,
                    value=native_result.value,
                    completed_depth=native_result.completed_depth,
                    nodes=native_result.nodes,
                    complete=native_result.complete,
                    cache_hits=native_result.cache_hits,
                    elapsed_ms=(perf_counter() - started) * 1000.0,
                )
            # The optional C++ search can fail for reasons outside our control
            # (for example a partially rebuilt library). Fall through to the
            # Python implementation instead of returning no move at all.

        search_board = (
            downgrade_board(current, threshold=self.downgrade_threshold)
            if self.downgrade_threshold is not None and self._uses_ntuple_evaluator
            else current
        )
        actions = legal_actions(search_board)
        if not actions:
            return None

        self.nodes = 0
        self.cache_hits = 0
        if self.cache_entries > self.max_cache_entries:
            self._player_cache.clear()
            self._chance_cache.clear()
        self._deadline = (
            inf if self.time_budget_s is None else perf_counter() + self.time_budget_s
        )

        # Always have a meaningful legal fallback, even when the time budget
        # expires before the first complete chance layer. This is a fast greedy
        # afterstate choice, not a claimed completed Expectimax depth.
        fallback_action = actions[0]
        fallback_value = -inf
        for action in actions:
            afterstate = apply_move(search_board, action)
            value = float(self.fallback_evaluator(afterstate.board))
            if self.include_merge_reward:
                value += afterstate.score_delta
            if value > fallback_value:
                fallback_action, fallback_value = action, value
        best_completed = SearchResult(fallback_action, fallback_value, 0, 0, False)

        for depth in range(1, self.max_depth + 1):
            depth_best_action = actions[0]
            depth_best_value = -inf
            try:
                for action in actions:
                    afterstate = apply_move(search_board, action)
                    continuation = self._chance_value(afterstate.board, depth - 1)
                    value = continuation + (afterstate.score_delta if self.include_merge_reward else 0.0)
                    if value > depth_best_value:
                        depth_best_action = action
                        depth_best_value = value
            except _SearchBudgetExceeded:
                break

            best_completed = SearchResult(
                action=depth_best_action,
                value=depth_best_value,
                completed_depth=depth,
                nodes=self.nodes,
                complete=depth == self.max_depth,
                cache_hits=self.cache_hits,
            )

        elapsed_ms = (perf_counter() - started) * 1000.0
        if best_completed.completed_depth == 0:
            # Even the shallowest search did not finish. Return a legal fallback
            # rather than leaking an arbitrary or illegal model action.
            return SearchResult(
                best_completed.action, best_completed.value, 0,
                self.nodes, False, self.cache_hits, elapsed_ms,
            )
        return SearchResult(
            action=best_completed.action,
            value=best_completed.value,
            completed_depth=best_completed.completed_depth,
            nodes=self.nodes,
            complete=best_completed.completed_depth == self.max_depth,
            cache_hits=self.cache_hits,
            elapsed_ms=elapsed_ms,
        )
