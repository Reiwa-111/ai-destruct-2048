from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
BOARD_SIZE = 4
SPAWN_TWO_PROBABILITY = 0.9

Board = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class MoveResult:
    """Deterministic result of a player move, before the random tile spawn."""

    board: Board
    score_delta: int
    moved: bool
    merges: int


@dataclass(frozen=True)
class Step:
    """Public game-step result, including the post-move random spawn."""

    board: list[list[int]]
    score_delta: int
    moved: bool
    game_over: bool
    merges: int
    spawned: tuple[int, int, int] | None = None
    max_tile: int = 0


def freeze_board(board: Iterable[Iterable[int]]) -> Board:
    frozen = tuple(tuple(int(value) for value in row) for row in board)
    if len(frozen) != BOARD_SIZE or any(len(row) != BOARD_SIZE for row in frozen):
        raise ValueError("2048 board must be exactly 4x4")
    for row in frozen:
        for value in row:
            if value < 0 or (value and (value < 2 or value & (value - 1))):
                raise ValueError(f"Tile value must be zero or a power of two: {value}")
    return frozen


@lru_cache(maxsize=65_536)
def _merge_line(line: tuple[int, ...]) -> tuple[tuple[int, ...], int, int]:
    packed = [value for value in line if value]
    output: list[int] = []
    score_delta = 0
    merges = 0
    index = 0
    while index < len(packed):
        if index + 1 < len(packed) and packed[index] == packed[index + 1]:
            merged = packed[index] * 2
            output.append(merged)
            score_delta += merged
            merges += 1
            index += 2
        else:
            output.append(packed[index])
            index += 1
    output.extend([0] * (BOARD_SIZE - len(output)))
    return tuple(output), score_delta, merges


@lru_cache(maxsize=131_072)
def _apply_move_cached(current: Board, action: str) -> MoveResult:
    output = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    score_delta = 0
    merges = 0

    if action in ("LEFT", "RIGHT"):
        for row_index in range(BOARD_SIZE):
            line = current[row_index]
            if action == "RIGHT":
                line = tuple(reversed(line))
            merged, gained, count = _merge_line(line)
            if action == "RIGHT":
                merged = tuple(reversed(merged))
            output[row_index] = list(merged)
            score_delta += gained
            merges += count
    else:
        for column in range(BOARD_SIZE):
            line = tuple(current[row][column] for row in range(BOARD_SIZE))
            if action == "DOWN":
                line = tuple(reversed(line))
            merged, gained, count = _merge_line(line)
            if action == "DOWN":
                merged = tuple(reversed(merged))
            for row in range(BOARD_SIZE):
                output[row][column] = merged[row]
            score_delta += gained
            merges += count

    result = freeze_board(output)
    return MoveResult(
        board=result,
        score_delta=score_delta,
        moved=result != current,
        merges=merges,
    )


def apply_move(board: Iterable[Iterable[int]], action: str) -> MoveResult:
    """Apply only the deterministic slide/merge operation; never spawn a tile."""
    action = action.upper()
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}")
    return _apply_move_cached(freeze_board(board), action)


@lru_cache(maxsize=32_768)
def _legal_actions_cached(current: Board) -> tuple[str, ...]:
    return tuple(action for action in ACTIONS if _apply_move_cached(current, action).moved)


def legal_actions(board: Iterable[Iterable[int]]) -> list[str]:
    current = freeze_board(board)
    return list(_legal_actions_cached(current))


def spawn_tile(
    board: Iterable[Iterable[int]], rng: random.Random
) -> tuple[Board, tuple[int, int] | None, int | None]:
    """Spawn one tile using the standard 90% 2 / 10% 4 distribution."""

    current = [list(row) for row in freeze_board(board)]
    empty = [(row, col) for row in range(BOARD_SIZE) for col in range(BOARD_SIZE) if not current[row][col]]
    if not empty:
        return freeze_board(current), None, None
    # Fixed two-draw sampling keeps the RNG stream independent of the number
    # of empty cells, which improves paired-seed reproducibility in benchmarks.
    slot = min(int(rng.random() * len(empty)), len(empty) - 1)
    row, col = empty[slot]
    value = 4 if rng.random() >= SPAWN_TWO_PROBABILITY else 2
    current[row][col] = value
    return freeze_board(current), (row, col), value


@lru_cache(maxsize=16_384)
def _spawn_outcomes_cached(current: Board) -> tuple[tuple[float, Board], ...]:
    current = [list(row) for row in current]
    empty = [(row, col) for row in range(BOARD_SIZE) for col in range(BOARD_SIZE) if not current[row][col]]
    if not empty:
        return ((1.0, freeze_board(current)),)

    cell_probability = 1.0 / len(empty)
    outcomes: list[tuple[float, Board]] = []
    for row, col in empty:
        for value, probability in ((2, 0.9), (4, 0.1)):
            spawned = [line[:] for line in current]
            spawned[row][col] = value
            outcomes.append((cell_probability * probability, freeze_board(spawned)))
    return tuple(outcomes)


def spawn_outcomes(board: Iterable[Iterable[int]]) -> list[tuple[float, Board]]:
    """Enumerate exact chance outcomes after an afterstate.

    Each empty cell receives a 2 with probability 0.9 / E and a 4 with
    probability 0.1 / E, where E is the number of empty cells.
    """

    return list(_spawn_outcomes_cached(freeze_board(board)))


def encode(board: Iterable[Iterable[int]]) -> np.ndarray:
    """Encode cells as log2(tile), with empty=0 and no high-tile saturation.

    The historical scale (rank / 15) is retained for compatibility with
    existing DQN checkpoints. Values above rank 15 are intentionally allowed
    to exceed 1.0 instead of collapsing 32768, 65536, 131072, ... to one value.
    """

    current = freeze_board(board)
    values = [0.0 if value == 0 else (value.bit_length() - 1) / 15.0
              for row in current for value in row]
    return np.asarray(values, dtype=np.float32)


class Game2048:
    """A deterministic 2048 episode with an optional undo history.

    The undo stack stores the pre-move board together with the RNG state, so
    undoing and replaying the same move reproduces the same spawned tile.
    """

    def __init__(self, rng: random.Random | None = None, max_undo: int = 200):
        self.rng = rng if rng is not None else random.Random()
        self.max_undo = max(1, int(max_undo))
        self._undo: deque[tuple] = deque(maxlen=self.max_undo)
        self.board: list[list[int]] = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.score = 0
        self.moves = 0
        self.merges = 0
        self.best_tile = 0
        self.reset()

    def reset(self) -> list[list[int]]:
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.score = 0
        self.moves = 0
        self.merges = 0
        self.best_tile = 0
        self._undo.clear()
        self.spawn()
        self.spawn()
        self.best_tile = self.max_tile
        return [row[:] for row in self.board]

    def spawn(self) -> bool:
        spawned, _, value = spawn_tile(self.board, self.rng)
        if value is None:
            return False
        self.board = [list(row) for row in spawned]
        self.best_tile = max(self.best_tile, self.max_tile)
        return True

    @staticmethod
    def _line(line: list[int]) -> tuple[list[int], int, int]:
        merged, score_delta, merges = _merge_line(tuple(line))
        return list(merged), score_delta, merges

    def _push_undo(self) -> None:
        self._undo.append((
            tuple(tuple(row) for row in self.board),
            self.score,
            self.moves,
            self.merges,
            self.best_tile,
            self.rng.getstate(),
        ))

    def move(self, action: str) -> Step:
        result = apply_move(self.board, action)
        spawned = None
        if result.moved:
            self._push_undo()
            self.board = [list(row) for row in result.board]
            self.score += result.score_delta
            self.moves += 1
            self.merges += result.merges
            spawned_board, cell, value = spawn_tile(self.board, self.rng)
            if value is not None and cell is not None:
                self.board = [list(row) for row in spawned_board]
                spawned = (int(cell[0]), int(cell[1]), int(value))
            self.best_tile = max(self.best_tile, self.max_tile)
        return Step(
            board=[row[:] for row in self.board],
            score_delta=result.score_delta,
            moved=result.moved,
            game_over=self.game_over(),
            merges=result.merges,
            spawned=spawned,
            max_tile=self.max_tile,
        )

    def undo(self) -> bool:
        """Restore the position before the last successful move."""

        if not self._undo:
            return False
        board, score, moves, merges, best_tile, rng_state = self._undo.pop()
        self.board = [list(row) for row in board]
        self.score = int(score)
        self.moves = int(moves)
        self.merges = int(merges)
        self.best_tile = int(best_tile)
        self.rng.setstate(rng_state)
        return True

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_move(self, action: str) -> bool:
        return apply_move(self.board, action).moved

    def legal_actions(self) -> list[str]:
        return legal_actions(self.board)

    def _simulate(self, action: str) -> list[list[int]]:
        return [list(row) for row in apply_move(self.board, action).board]

    def game_over(self) -> bool:
        return not legal_actions(self.board)

    @property
    def max_tile(self) -> int:
        return max(max(row) for row in self.board)

    @property
    def empty_count(self) -> int:
        return sum(value == 0 for row in self.board for value in row)

    def state_dict(self) -> dict:
        """Serializable snapshot of the visible episode state."""

        return {
            "board": [row[:] for row in self.board],
            "score": self.score,
            "moves": self.moves,
            "merges": self.merges,
            "best_tile": self.best_tile,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.board = [list(row) for row in freeze_board(state["board"])]
        self.score = int(state.get("score", 0))
        self.moves = int(state.get("moves", 0))
        self.merges = int(state.get("merges", 0))
        self.best_tile = int(state.get("best_tile", self.max_tile))
        if state.get("rng_state") is not None:
            self.rng.setstate(state["rng_state"])
        self._undo.clear()
