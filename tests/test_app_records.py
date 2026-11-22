import random
from types import SimpleNamespace
from unittest.mock import Mock

from destroyer import app
from destroyer.game import Game2048
from destroyer.records import RecordBook


def test_app_saves_game_over_records_once(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "RECORDS_PATH", tmp_path / "records.json")
    instance = app.App.__new__(app.App)
    instance.records = RecordBook()
    instance.game = SimpleNamespace(score=24_680, max_tile=4096)
    instance.game_best_score = 24_680
    instance.game_best_tile = 4096
    instance.high_score = 0
    instance.high_tile = 0
    instance._game_record_saved = False
    instance.log = Mock()
    instance._save_state = Mock()

    instance._finish_game_record()
    instance._finish_game_record()

    assert RecordBook.load(app.RECORDS_PATH).best_score == 24_680
    assert RecordBook.load(app.RECORDS_PATH).best_tile == 4096
    assert RecordBook.load(app.RECORDS_PATH).games_played == 1
    assert instance.high_score == 24_680
    assert instance.high_tile == 4096
    assert instance._game_record_saved is True
    instance._save_state.assert_called_once()


def test_app_does_not_mark_failed_record_save_as_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "RECORDS_PATH", tmp_path / "missing" / "records.json")
    instance = app.App.__new__(app.App)
    instance.records = RecordBook()
    instance.game = SimpleNamespace(score=100, max_tile=128)
    instance.game_best_score = 100
    instance.game_best_tile = 128
    instance.high_score = 0
    instance.high_tile = 0
    instance._game_record_saved = False
    instance.log = Mock()
    instance._save_state = Mock()

    # Force persistence to fail rather than letting the helper create its parent.
    monkeypatch.setattr(RecordBook, "with_completed_game", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk full")))
    instance._finish_game_record()

    assert instance._game_record_saved is False
    instance.log.assert_called_once()
    instance._save_state.assert_not_called()


def test_evaluation_results_update_persistent_personal_records(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "RECORDS_PATH", tmp_path / "records.json")
    initial = RecordBook(
        best_score=129_376,
        best_tile=8192,
        games_played=2,
        last_score=129_376,
        last_tile=8192,
    )
    initial.save(app.RECORDS_PATH)

    instance = app.App.__new__(app.App)
    instance.records = initial
    instance.high_score = initial.best_score
    instance.high_tile = initial.best_tile
    instance.evaluating = True
    instance.eval_cancel = SimpleNamespace(clear=Mock())
    instance.eval_progress = SimpleNamespace(configure=Mock())
    instance.eval_progress_label = SimpleNamespace(config=Mock())
    instance._update_eval_summary = Mock()
    instance.log = Mock()
    instance._auto_save_report = Mock()
    instance._sync_controls = Mock()
    instance._update_status = Mock()
    instance._save_state = Mock()

    report = SimpleNamespace(
        games_requested=2,
        games=2,
        cancelled=False,
        agent="N-TUPLE + EXPECTIMAX",
        records=[
            SimpleNamespace(score=227_188, max_tile=16_384),
            SimpleNamespace(score=131_952, max_tile=8192),
        ],
        summary=lambda: {
            "mean_score": 179_570.0,
            "median_score": 179_570.0,
            "tile_rates": {"2048": 1.0, "4096": 1.0, "8192": 0.5},
            "elapsed_seconds": 12.3,
        },
    )

    instance._on_eval_done(report)

    persisted = RecordBook.load(app.RECORDS_PATH)
    assert persisted.best_score == 227_188
    assert persisted.best_tile == 16_384
    assert persisted.games_played == 4
    assert persisted.last_score == 131_952
    assert persisted.last_tile == 8192
    assert instance.high_score == 227_188
    assert instance.high_tile == 16_384
    instance._save_state.assert_called_once()
def test_undo_after_game_over_allows_recording_replayed_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "RECORDS_PATH", tmp_path / "records.json")
    initial = RecordBook().with_completed_game(app.RECORDS_PATH, score=1000, max_tile=512)
    game = Game2048(random.Random(1234))
    action = game.legal_actions()[0]
    assert game.move(action).moved

    instance = app.App.__new__(app.App)
    instance.training = False
    instance.game = game
    instance.records = initial
    instance.high_score = initial.best_score
    instance.high_tile = initial.best_tile
    instance.game_best_score = 1000
    instance.game_best_tile = 512
    instance._game_record_saved = True
    instance.stop_autoplay = Mock()
    instance.overlay = "gameover"
    instance.win_shown = True
    instance.last_spawn = None
    instance.last_search_result = None
    instance.log = Mock()
    instance.render = Mock()
    instance._save_state = Mock()

    instance.undo()
    assert instance._game_record_saved is False
    assert instance.game_best_score == game.score
    assert instance.game_best_tile == game.max_tile

    instance.game_best_score = 1500
    instance.game_best_tile = 1024
    instance._finish_game_record()

    saved = RecordBook.load(app.RECORDS_PATH)
    assert saved.games_played == 2
    assert saved.last_score == 1500
    assert saved.last_tile == 1024
    assert instance._game_record_saved is True
