import json

import pytest

from destroyer.records import RecordBook


def test_record_book_persists_best_score_and_tile_independently(tmp_path):
    path = tmp_path / "records.json"
    records = RecordBook.load(path)
    records = records.with_completed_game(path, score=12_345, max_tile=1024)
    records = records.with_completed_game(path, score=900, max_tile=2048)

    restored = RecordBook.load(path)
    assert restored == RecordBook(
        best_score=12_345,
        best_tile=2048,
        games_played=2,
        last_score=900,
        last_tile=2048,
    )
    assert not list(tmp_path.glob(".*.tmp"))


def test_record_book_keeps_existing_records_when_new_game_is_worse(tmp_path):
    path = tmp_path / "records.json"
    first = RecordBook().with_completed_game(path, score=1000, max_tile=512)
    second = first.with_completed_game(path, score=500, max_tile=256)
    assert second.best_score == 1000
    assert second.best_tile == 512
    assert second.games_played == 2


def test_record_book_persists_batch_of_evaluation_games_once(tmp_path):
    path = tmp_path / "records.json"
    initial = RecordBook().with_completed_game(path, score=1000, max_tile=512)
    updated = initial.with_completed_games(
        path,
        [(227_188, 16_384), (131_952, 8192), (80_000, 4096)],
    )

    assert RecordBook.load(path) == RecordBook(
        best_score=227_188,
        best_tile=16_384,
        games_played=4,
        last_score=80_000,
        last_tile=4096,
    )
    assert updated == RecordBook.load(path)


@pytest.mark.parametrize(
    "score,tile",
    [(-1, 2), (1, 3), (1, -2)],
)
def test_record_book_rejects_invalid_game_result(tmp_path, score, tile):
    with pytest.raises(ValueError):
        RecordBook().with_completed_game(
            tmp_path / "records.json", score=score, max_tile=tile
        )


def test_record_book_rejects_unknown_schema(tmp_path):
    path = tmp_path / "records.json"
    path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        RecordBook.load(path)
