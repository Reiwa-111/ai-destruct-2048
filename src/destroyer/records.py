"""Persistent personal records for completed 2048 games."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


def _non_negative_int(value, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer")
    result = int(value)
    if result < 0 or result != value:
        raise ValueError(f"{field} must be a non-negative integer")
    return result


def _tile_value(value, field: str) -> int:
    result = _non_negative_int(value, field)
    if result and result & (result - 1):
        raise ValueError(f"{field} must be zero or a power of two")
    return result


@dataclass(frozen=True)
class RecordBook:
    """Monotonic best score / tile records plus the last completed game."""

    best_score: int = 0
    best_tile: int = 0
    games_played: int = 0
    last_score: int = 0
    last_tile: int = 0

    @classmethod
    def load(cls, path: str | Path) -> "RecordBook":
        source = Path(path)
        if not source.exists():
            return cls()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported or invalid records file: {source}")
        return cls(
            best_score=_non_negative_int(payload.get("best_score", 0), "best_score"),
            best_tile=_tile_value(payload.get("best_tile", 0), "best_tile"),
            games_played=_non_negative_int(payload.get("games_played", 0), "games_played"),
            last_score=_non_negative_int(payload.get("last_score", 0), "last_score"),
            last_tile=_tile_value(payload.get("last_tile", 0), "last_tile"),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, **asdict(self)}
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return destination

    def with_completed_game(
        self, path: str | Path, *, score: int, max_tile: int
    ) -> "RecordBook":
        """Persist a completed game and return the updated immutable record book."""
        return self.with_completed_games(path, ((score, max_tile),))


    def with_completed_games(
        self, path: str | Path, games: Iterable[tuple[int, int]]
    ) -> "RecordBook":
        """Persist a batch of (score, max_tile) results with one atomic write."""

        normalized = [
            (_non_negative_int(score, "score"), _tile_value(max_tile, "max_tile"))
            for score, max_tile in games
        ]
        if not normalized:
            return self
        updated = RecordBook(
            best_score=max(self.best_score, max(score for score, _ in normalized)),
            best_tile=max(self.best_tile, max(tile for _, tile in normalized)),
            games_played=self.games_played + len(normalized),
            last_score=normalized[-1][0],
            last_tile=normalized[-1][1],
        )
        updated.save(path)
        return updated
