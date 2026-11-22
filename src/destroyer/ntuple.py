from __future__ import annotations

import random
import ctypes
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .game import ACTIONS, Board, Game2048, apply_move, freeze_board, legal_actions, spawn_tile

CELL_STATES = 16  # Empty plus tile ranks 1..15 (2 through 32768).
TUPLE_SIZE = 6
PATTERN_VERSION = 4
# Four documented geometric families: L, shifted L, 3x2 block and diagonal
# zigzag. The engine, feature indexing, trainer and native implementation here
# are independently implemented.
PATTERNS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5),       # L: first row and first two cells of row 2
    (4, 5, 6, 7, 8, 9),       # shifted L
    (0, 1, 4, 5, 8, 9),       # 3x2 block
    (0, 1, 5, 6, 9, 10),      # diagonal zigzag
)
PATTERN_ARRAY = np.asarray(PATTERNS, dtype=np.int32)
INDEX_WEIGHTS = np.asarray(tuple(CELL_STATES ** power for power in range(TUPLE_SIZE - 1, -1, -1)), dtype=np.int64)
PATTERN_IDS = np.broadcast_to(np.arange(len(PATTERNS), dtype=np.int32), (8, len(PATTERNS)))
_NATIVE_LIBRARY = None
_NATIVE_CHECKED = False


def _load_native_library():
    global _NATIVE_LIBRARY, _NATIVE_CHECKED
    if _NATIVE_CHECKED:
        return _NATIVE_LIBRARY
    _NATIVE_CHECKED = True
    library_path = Path(__file__).resolve().parents[2] / "native" / "libdestroyer_ntuple.so"
    if not library_path.is_file():
        return None
    try:
        library = ctypes.CDLL(str(library_path))
        byte_ptr = ctypes.POINTER(ctypes.c_uint8)
        uint64_ptr = ctypes.POINTER(ctypes.c_uint64)
        float_ptr = ctypes.POINTER(ctypes.c_float)
        library.destroyer_ntuple_value.argtypes = [uint64_ptr, float_ptr]
        library.destroyer_ntuple_value.restype = ctypes.c_double
        library.destroyer_ntuple_update.argtypes = [uint64_ptr, float_ptr, ctypes.c_double, ctypes.c_double]
        library.destroyer_ntuple_update.restype = ctypes.c_double
        library.destroyer_ntuple_train_episode.argtypes = [
            float_ptr, ctypes.c_uint64, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_uint64, byte_ptr, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, byte_ptr, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            uint64_ptr, uint64_ptr, uint64_ptr,
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_double),
        ]
        library.destroyer_ntuple_train_episode.restype = ctypes.c_int
        library.destroyer_ntuple_train_episode_lambda.argtypes = [
            float_ptr, ctypes.c_uint64, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_uint64, byte_ptr,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, byte_ptr, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), uint64_ptr, uint64_ptr, uint64_ptr,
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_double),
        ]
        library.destroyer_ntuple_train_episode_lambda.restype = ctypes.c_int
        _NATIVE_LIBRARY = library
    except (OSError, AttributeError):
        _NATIVE_LIBRARY = None
    return _NATIVE_LIBRARY


def _rotate90(board: Board) -> Board:
    return tuple(tuple(board[3 - col][row] for col in range(4)) for row in range(4))


def board_symmetries(board: Iterable[Iterable[int]]) -> tuple[Board, ...]:
    """Return the eight D4 rotations/reflections of a board."""

    current = freeze_board(board)
    reflected = tuple(tuple(reversed(row)) for row in current)
    result: list[Board] = []
    for source in (current, reflected):
        rotated = source
        for _ in range(4):
            result.append(rotated)
            rotated = _rotate90(rotated)
    return tuple(result)


@dataclass(frozen=True)
class EpisodeStats:
    score: int
    max_tile: int
    moves: int
    merges: int
    td_error_abs_mean: float = 0.0
    td_updates: int = 0


class NtupleValue:
    """Four six-cell N-tuples evaluated over all eight D4 symmetries.

    Four 16^6 float32 tables occupy 256 MiB. Tile ranks 0..15 encode empty
    and 2..32768; higher tiles currently share rank 15. A later endgame
    evaluator can cover 65536+ without multiplying the core tables.
    """

    def __init__(self, tables: np.ndarray | None = None):
        expected_shape = (len(PATTERNS), CELL_STATES ** TUPLE_SIZE)
        if tables is None:
            self.tables = np.zeros(expected_shape, dtype=np.float32)
        else:
            array = np.asarray(tables, dtype=np.float32)
            if array.shape != expected_shape:
                raise ValueError(
                    f"Incompatible N-tuple checkpoint shape {array.shape}; "
                    f"this version requires {expected_shape}. Older 4-tuple "
                    "development checkpoints cannot be resumed with this architecture."
                )
            self.tables = array.copy()
        self._native = _load_native_library()

    @property
    def backend(self) -> str:
        return "native-c++" if self._native is not None else "numpy"

    def _native_board(self, board: Iterable[Iterable[int]]) -> np.ndarray:
        return np.asarray(freeze_board(board), dtype=np.uint64).reshape(16)

    def _indices(self, board: Iterable[Iterable[int]]) -> np.ndarray:
        indices = np.empty((8, len(PATTERNS)), dtype=np.int64)
        for symmetry_index, symmetry in enumerate(board_symmetries(board)):
            flat = np.fromiter((value for row in symmetry for value in row), dtype=np.int64, count=16)
            ranks = np.zeros(16, dtype=np.int64)
            occupied = flat > 0
            ranks[occupied] = np.floor(np.log2(flat[occupied])).astype(np.int64)
            np.minimum(ranks, CELL_STATES - 1, out=ranks)
            indices[symmetry_index] = (ranks[PATTERN_ARRAY] * INDEX_WEIGHTS).sum(axis=1)
        return indices

    def value(self, board: Iterable[Iterable[int]]) -> float:
        if self._native is not None:
            native_board = self._native_board(board)
            return float(self._native.destroyer_ntuple_value(
                native_board.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                self.tables.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ))
        indices = self._indices(board)
        return float(self.tables[PATTERN_IDS, indices].sum(dtype=np.float64) / len(indices))

    def state_value(self, board: Iterable[Iterable[int]], gamma: float = 1.0) -> float:
        """Convert afterstate values into a value for a player-to-move state."""
        _, _, best_value = self.best_afterstate(board, gamma)
        return 0.0 if best_value == -float("inf") else float(best_value)

    __call__ = state_value

    def update(self, board: Iterable[Iterable[int]], target: float, alpha: float = 0.1) -> float:
        """Apply one semi-gradient TD update; return the pre-update TD error."""

        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self._native is not None:
            native_board = self._native_board(board)
            return float(self._native.destroyer_ntuple_update(
                native_board.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                self.tables.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                float(target),
                float(alpha),
            ))
        indices = self._indices(board)
        error = float(target) - self.value(board)
        # value() averages the eight D4 views, so each active feature has
        # derivative 1/8 with respect to its table entry.
        increment = alpha * error / len(indices)
        np.add.at(self.tables, (PATTERN_IDS, indices), increment)
        return error

    def best_afterstate(
        self,
        board: Iterable[Iterable[int]],
        gamma: float = 1.0,
        tie_rng: random.Random | None = None,
    ):
        current = freeze_board(board)
        best_action: str | None = None
        best_result = None
        best_value = -float("inf")
        for action in legal_actions(current):
            result = apply_move(current, action)
            value = result.score_delta + gamma * self.value(result.board)
            if value > best_value + 1e-9:
                best_action, best_result = action, result
                best_value = value
        return best_action, best_result, best_value

    def train_episode(
        self,
        rng: random.Random | None = None,
        epsilon: float = 0.05,
        alpha: float = 0.1,
        gamma: float = 1.0,
        max_moves: int = 100_000,
        initial_board: Iterable[Iterable[int]] | None = None,
        position_callback: Callable[[Board], None] | None = None,
        capture_min_tile: int = 512,
        capture_every: int = 16,
        td_lambda: float = 0.0,
    ) -> EpisodeStats:
        """Run one on-policy game with greedy TD-afterstate bootstrapping."""

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= td_lambda <= 1.0:
            raise ValueError("td_lambda must be in [0, 1]")
        if max_moves < 1:
            raise ValueError("max_moves must be positive")
        if capture_min_tile < 2 or capture_min_tile & (capture_min_tile - 1):
            raise ValueError("capture_min_tile must be a power-of-two tile >= 2")
        if capture_min_tile > (1 << 31):
            raise ValueError("capture_min_tile cannot exceed the native engine's 2^31 rank limit")
        if capture_every < 1:
            raise ValueError("capture_every must be positive")
        random_source = rng if rng is not None else random.Random()
        if self._native is not None:
            native_initial = None
            has_initial = int(initial_board is not None)
            if initial_board is not None:
                frozen = freeze_board(initial_board)
                if not legal_actions(frozen):
                    raise ValueError("initial_board must be a non-terminal position")
                if max(max(row) for row in frozen) > (1 << 31):
                    raise ValueError("initial_board exceeds the native engine's 2^31 tile limit")
                native_initial = np.asarray(
                    [0 if tile == 0 else tile.bit_length() - 1
                     for row in frozen for tile in row],
                    dtype=np.uint8,
                )

            score = ctypes.c_uint64()
            moves = ctypes.c_uint64()
            merges = ctypes.c_uint64()
            max_rank = ctypes.c_uint8()
            mean_td_error = ctypes.c_double()
            capture_capacity = 64 if position_callback is not None else 0
            captured_positions = (
                np.empty((capture_capacity, 16), dtype=np.uint8)
                if capture_capacity else None
            )
            captured_count = ctypes.c_uint32()
            captured_pointer = (
                captured_positions.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
                if captured_positions is not None else None
            )
            initial_pointer = (
                native_initial.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
                if native_initial is not None else None
            )
            ok = self._native.destroyer_ntuple_train_episode_lambda(
                self.tables.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_uint64(random_source.getrandbits(64)),
                float(epsilon), float(alpha), float(gamma), float(td_lambda),
                ctypes.c_uint64(max_moves),
                initial_pointer, has_initial,
                int(capture_min_tile.bit_length() - 1), int(capture_every),
                captured_pointer, ctypes.c_uint32(capture_capacity), ctypes.byref(captured_count),
                ctypes.byref(score), ctypes.byref(moves), ctypes.byref(merges),
                ctypes.byref(max_rank), ctypes.byref(mean_td_error),
            )
            if not ok:
                raise RuntimeError("Native N-tuple training episode failed")
            if position_callback is not None and captured_positions is not None:
                for flat in captured_positions[:captured_count.value]:
                    captured_board = tuple(
                        tuple(0 if int(flat[row * 4 + col]) == 0
                              else 1 << int(flat[row * 4 + col]) for col in range(4))
                        for row in range(4)
                    )
                    position_callback(captured_board)
            rank = int(max_rank.value)
            return EpisodeStats(
                score=int(score.value),
                max_tile=0 if rank == 0 else 1 << rank,
                moves=int(moves.value),
                merges=int(merges.value),
                td_error_abs_mean=float(mean_td_error.value),
                td_updates=int(moves.value),
            )

        # Keep exploration draws separate from environment/spawn randomness.
        action_rng = random.Random(random_source.getrandbits(64))
        game = Game2048(random_source)
        if initial_board is not None:
            restored = freeze_board(initial_board)
            if not legal_actions(restored):
                raise ValueError("initial_board must be a non-terminal position")
            game.board = [list(row) for row in restored]
        total_score = 0
        total_merges = 0
        td_error_abs_sum = 0.0
        td_updates = 0
        previous_afterstate: Board | None = None
        traces: dict[tuple[int, int], float] = {}

        def lambda_update(board: Board, target: float) -> float:
            if td_lambda == 0.0:
                return self.update(board, target, alpha)
            error = float(target) - self.value(board)
            decay = gamma * td_lambda
            for feature in list(traces):
                traces[feature] *= decay
                if abs(traces[feature]) < 1.0e-7:
                    del traces[feature]
            indices = self._indices(board)
            counts: dict[tuple[int, int], int] = {}
            for pattern_id in range(len(PATTERNS)):
                for index in indices[:, pattern_id]:
                    key = (pattern_id, int(index))
                    counts[key] = counts.get(key, 0) + 1
            for feature, count in counts.items():
                traces[feature] = traces.get(feature, 0.0) + count / len(indices)
            for (pattern_id, index), trace in traces.items():
                self.tables[pattern_id, index] += alpha * error * trace
            return error

        for _ in range(max_moves):
            current = freeze_board(game.board)
            actions = legal_actions(current)
            if not actions:
                if previous_afterstate is not None:
                    td_error = lambda_update(previous_afterstate, 0.0)
                    td_error_abs_sum += abs(td_error)
                    td_updates += 1
                break

            if action_rng.random() < epsilon:
                action = action_rng.choice(actions)
                current_afterstate = apply_move(current, action)
            else:
                action, current_afterstate, _ = self.best_afterstate(current, gamma)
                assert action is not None and current_afterstate is not None

            # Update the previous afterstate from the action that is actually
            # selected now. This is the on-policy TD-afterstate ordering.
            if previous_afterstate is not None:
                td_target = current_afterstate.score_delta + gamma * self.value(current_afterstate.board)
                td_error = lambda_update(previous_afterstate, td_target)
                td_error_abs_sum += abs(td_error)
                td_updates += 1

            previous_afterstate = current_afterstate.board
            total_score += current_afterstate.score_delta
            total_merges += current_afterstate.merges

            next_board, _, _ = spawn_tile(current_afterstate.board, random_source)
            game.board = [list(row) for row in next_board]
            game.score += current_afterstate.score_delta
            game.moves += 1
            game.merges += current_afterstate.merges
            if (
                position_callback is not None
                and game.moves % capture_every == 0
                and game.max_tile >= capture_min_tile
            ):
                position_callback(freeze_board(game.board))
        else:
            # Hitting the safety cap is not a terminal game; report the played prefix.
            pass

        return EpisodeStats(
            score=total_score,
            max_tile=max(max(row) for row in game.board),
            moves=game.moves,
            merges=total_merges,
            td_error_abs_mean=td_error_abs_sum / max(1, td_updates),
            td_updates=td_updates,
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez(
                    stream,
                    tables=self.tables,
                    format_version=np.asarray([PATTERN_VERSION], dtype=np.int32),
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> "NtupleValue":
        with np.load(path) as data:
            if "tables" not in data:
                raise ValueError("Checkpoint has no N-tuple tables")
            version = int(data["format_version"][0]) if "format_version" in data else 1
            if version != PATTERN_VERSION:
                raise ValueError(
                    f"N-tuple checkpoint format {version} is not supported; "
                    f"current format is {PATTERN_VERSION}."
                )
            return cls(data["tables"])


class PositionPool:
    """Bounded deduplicated pool of reachable non-terminal training positions."""

    def __init__(self, capacity: int = 10_000):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._positions: deque[Board] = deque()
        self._known: set[Board] = set()
        self._rank_buckets: dict[int, list[Board]] = {}
        self._rank_index: dict[Board, tuple[int, int]] = {}

    def __len__(self) -> int:
        return len(self._positions)

    def add(self, board: Iterable[Iterable[int]]) -> bool:
        position = freeze_board(board)
        maximum = max(value for row in position for value in row)
        if maximum < 2 or not legal_actions(position) or position in self._known:
            return False
        if len(self._positions) >= self.capacity:
            removed = self._positions.popleft()
            self._known.remove(removed)
            old_rank, old_index = self._rank_index.pop(removed)
            bucket = self._rank_buckets[old_rank]
            last = bucket.pop()
            if old_index < len(bucket):
                bucket[old_index] = last
                self._rank_index[last] = (old_rank, old_index)
            if not bucket:
                del self._rank_buckets[old_rank]
        self._positions.append(position)
        self._known.add(position)
        rank = maximum.bit_length() - 1
        bucket = self._rank_buckets.setdefault(rank, [])
        self._rank_index[position] = (rank, len(bucket))
        bucket.append(position)
        return True

    def sample(self, rng: random.Random, min_tile: int = 2) -> Board:
        if not self._positions:
            raise LookupError("position pool is empty")
        if min_tile < 2 or min_tile & (min_tile - 1):
            raise ValueError("min_tile must be a power of two >= 2")
        min_rank = min_tile.bit_length() - 1
        eligible = [(rank, bucket) for rank, bucket in self._rank_buckets.items() if rank >= min_rank]
        total = sum(len(bucket) for _, bucket in eligible)
        if not total:
            raise LookupError(f"position pool has no positions with max tile >= {min_tile}")
        index = rng.randrange(total)
        for _, bucket in eligible:
            if index < len(bucket):
                return bucket[index]
            index -= len(bucket)
        raise RuntimeError("position-pool rank index became inconsistent")

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        positions = np.asarray(self._positions, dtype=np.uint32).reshape((-1, 16))
        try:
            with temporary.open("wb") as stream:
                np.savez(
                    stream,
                    positions=positions,
                    capacity=np.asarray([self.capacity], dtype=np.int32),
                    format_version=np.asarray([1], dtype=np.int32),
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path, capacity: int | None = None) -> "PositionPool":
        with np.load(path) as data:
            positions = data["positions"]
            saved_capacity = int(data["capacity"][0]) if "capacity" in data else len(positions)
        result = cls(capacity or max(1, saved_capacity))
        for flat in positions:
            result.add(tuple(tuple(int(flat[row * 4 + col]) for col in range(4)) for row in range(4)))
        return result
