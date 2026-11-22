import random

import numpy as np
import pytest

from destroyer.game import Game2048, apply_move, legal_actions
from destroyer.ntuple import NtupleValue, PositionPool, board_symmetries
from destroyer.search import Expectimax, PositionHeuristic, downgrade_board, native_apply_move, native_ntuple_value


def test_d4_symmetries_are_unique_for_asymmetric_board():
    board = (
        (2, 4, 8, 16),
        (0, 32, 64, 128),
        (256, 512, 0, 1024),
        (2, 0, 4, 8),
    )
    assert len(set(board_symmetries(board))) == 8


def test_tile_downgrade_preserves_empty_cells_and_rank_order():
    board = (
        (0, 2, 4, 8),
        (16, 32, 64, 128),
        (256, 512, 1024, 2048),
        (4096, 8192, 0, 32768),
    )
    downgraded = downgrade_board(board, threshold=32768)
    assert downgraded[0][0] == 0
    assert downgraded[0][1:4] == (2, 4, 8)
    assert downgraded[-1][-1] == 16384
    assert downgraded[-1][-3] == 8192
    assert downgraded[-1][-2] == 0
    original_nonempty = [tile for row in board for tile in row if tile]
    transformed_nonempty = [tile for row in downgraded for tile in row if tile]
    assert sorted(transformed_nonempty) == sorted([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    assert len(original_nonempty) == len(transformed_nonempty)


def test_tile_downgrade_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="threshold"):
        downgrade_board(((2, 0, 0, 0),) * 4, threshold=0)


def test_tile_downgrade_is_root_transform_and_requires_threshold_and_missing_rank():
    low_board = ((2, 4, 8, 16), (32, 64, 128, 256), (512, 1024, 2048, 4096), (8192, 0, 0, 0))
    assert downgrade_board(low_board, threshold=32768) == low_board
    complete_ranks = tuple(tuple(1 << rank for rank in range(1, 5)) for _ in range(4))
    complete_ranks = tuple(tuple(row) for row in complete_ranks)
    assert downgrade_board(complete_ranks, threshold=8) == complete_ranks


def test_native_and_python_expectimax_match_with_tile_downgrading():
    ntuple = NtupleValue()
    rng = np.random.default_rng(8182)
    ntuple.tables[:] = rng.normal(0.0, 0.2, ntuple.tables.shape).astype(np.float32)
    board = (
        (32768, 8192, 4096, 2048),
        (1024, 512, 256, 128),
        (64, 32, 16, 8),
        (4, 2, 0, 0),
    )
    native = Expectimax(
        ntuple.state_value, max_depth=1, node_budget=100_000, native=True,
        downgrade_threshold=32768, downgrade_steps=1,
    )
    python = Expectimax(
        ntuple.state_value, max_depth=1, node_budget=100_000, native=False,
        downgrade_threshold=32768, downgrade_steps=1,
    )
    if native.native_mode != "ntuple":
        pytest.skip("optional C++ search backend has not been built")
    native_result = native.choose(board)
    python_result = python.choose(board)
    assert native_result is not None and python_result is not None
    assert native_result.completed_depth == python_result.completed_depth == 1
    assert native_result.action == python_result.action
    assert native_result.value == pytest.approx(python_result.value, rel=1e-5, abs=1e-3)


def test_native_and_python_fallbacks_apply_tile_downgrading_consistently():
    ntuple = NtupleValue()
    board = (
        (32768, 8192, 4096, 2048),
        (1024, 512, 256, 128),
        (64, 32, 16, 8),
        (4, 2, 0, 0),
    )
    native = Expectimax(
        ntuple.state_value, max_depth=3, node_budget=1, native=True,
        fallback_evaluator=ntuple.value, downgrade_threshold=32768,
    )
    python = Expectimax(
        ntuple.state_value, max_depth=3, node_budget=1, native=False,
        fallback_evaluator=ntuple.value, downgrade_threshold=32768,
    )
    if native.native_mode != "ntuple":
        pytest.skip("optional C++ search backend has not been built")
    native_result = native.choose(board)
    python_result = python.choose(board)
    assert native_result is not None and python_result is not None
    assert native_result.completed_depth == python_result.completed_depth == 0
    assert native_result.action == python_result.action
    assert native_result.value == pytest.approx(python_result.value, abs=1e-4)


def test_tile_downgrading_is_rejected_for_classic_heuristic():
    with pytest.raises(ValueError, match="only with an N-tuple evaluator"):
        Expectimax(PositionHeuristic(), downgrade_threshold=32768)


@pytest.mark.parametrize("node_budget", [1, 4, 12, 64])
def test_native_and_python_expectimax_match_node_budget_interruption(node_budget):
    game = Game2048(random.Random(8128))
    native = Expectimax(
        PositionHeuristic(), max_depth=3, node_budget=node_budget,
        time_budget_ms=None, native=True,
    )
    python = Expectimax(
        PositionHeuristic(), max_depth=3, node_budget=node_budget,
        time_budget_ms=None, native=False,
    )
    if native.native_mode != "classic":
        pytest.skip("optional C++ search backend has not been built")
    native_result = native.choose(game.board)
    python_result = python.choose(game.board)
    assert native_result is not None and python_result is not None
    assert native_result.completed_depth == python_result.completed_depth
    assert native_result.nodes == python_result.nodes
    assert native_result.action == python_result.action
    assert native_result.value == pytest.approx(python_result.value, rel=2e-5, abs=1.0)


def test_ntuple_value_is_symmetry_invariant_and_learns_toward_target():
    value = NtupleValue()
    assert value.tables.shape == (4, 16 ** 6)
    assert value.tables.nbytes == 256 * 1024 * 1024
    board = (
        (2, 4, 8, 16),
        (0, 32, 64, 128),
        (256, 512, 0, 1024),
        (2, 0, 4, 8),
    )
    baseline = value(board)
    assert all(abs(value(symmetry) - baseline) < 1e-6 for symmetry in board_symmetries(board))

    error = value.update(board, target=100.0, alpha=0.1)
    assert error > 0
    assert value(board) > baseline


def test_ntuple_update_matches_gradient_of_eight_symmetry_average():
    value = NtupleValue()
    rng = random.Random(20261121)
    board = tuple(
        tuple(0 if (rank := rng.randrange(0, 16)) == 0 else 1 << rank for _ in range(4))
        for _ in range(4)
    )
    indices = value._indices(board)
    assert all(len(set(indices[:, pattern].tolist())) == 8 for pattern in range(4))

    before = value.value(board)
    target = 123.0
    alpha = 0.1
    error = target - before
    returned_error = value.update(board, target=target, alpha=alpha)
    after = value.value(board)

    # There are 32 unique active features, each with derivative 1/8.
    # A semi-gradient step therefore changes V by alpha * error * 32/8^2.
    expected_delta = alpha * error * (32.0 / 64.0)
    assert returned_error == pytest.approx(error)
    assert after - before == pytest.approx(expected_delta, rel=1e-5, abs=1e-5)


def test_ntuple_state_value_includes_the_best_next_move_reward():
    value = NtupleValue()
    board = (
        (2, 2, 4, 8),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )
    action, afterstate, expected = value.best_afterstate(board)
    assert action is not None and afterstate is not None
    assert value.state_value(board) == expected
    assert expected >= afterstate.score_delta


def test_ntuple_checkpoint_round_trip_and_version_guard(tmp_path):
    value = NtupleValue()
    board = (
        (2, 4, 8, 16),
        (32, 64, 128, 256),
        (512, 1024, 2048, 4096),
        (8192, 16384, 32768, 0),
    )
    for symmetry_indices in value._indices(board):
        for pattern_id, index in enumerate(symmetry_indices):
            value.tables[pattern_id, index] = pattern_id + index / 10_000_000
    path = tmp_path / 'ntuple_v4.npz'
    value.save(path)
    assert not list(tmp_path.glob('.*.tmp'))
    restored = NtupleValue.load(path)
    assert np.array_equal(value.tables, restored.tables)
    assert restored.value(board) == pytest.approx(value.value(board))

    legacy_path = tmp_path / 'old_v3.npz'
    np.savez(legacy_path, tables=np.zeros_like(value.tables), format_version=np.asarray([3],dtype=np.int32))
    with pytest.raises(ValueError, match='format 3'):
        NtupleValue.load(legacy_path)


def test_expectimax_returns_legal_move_and_completes_shallow_search():
    game = Game2048(random.Random(7))
    search = Expectimax(PositionHeuristic(), max_depth=2, node_budget=50_000)
    result = search.choose(game.board)
    assert result is not None
    assert result.action in legal_actions(game.board)
    assert result.completed_depth >= 1
    assert result.nodes > 0


def test_classic_heuristic_does_not_mix_raw_merge_reward_into_search():
    heuristic = PositionHeuristic()
    search = Expectimax(heuristic, max_depth=1, node_budget=50_000)
    assert heuristic.includes_merge_reward is False
    assert search.include_merge_reward is False


def test_expectimax_reuses_completed_transpositions_between_moves():
    game = Game2048(random.Random(29))
    search = Expectimax(PositionHeuristic(), max_depth=1, node_budget=50_000)
    first = search.choose(game.board)
    second = search.choose(game.board)
    assert first is not None and second is not None
    assert second.action == first.action
    assert second.cache_hits > 0
    assert second.nodes <= first.nodes


def test_expectimax_reports_no_action_for_terminal_board():
    board = (
        (2, 4, 8, 16),
        (32, 64, 128, 256),
        (512, 1024, 2, 4),
        (8, 16, 32, 64),
    )
    search = Expectimax(PositionHeuristic(), max_depth=2, node_budget=100)
    assert search.choose(board) is None


def test_expectimax_time_budget_always_returns_a_legal_fallback():
    game = Game2048(random.Random(19))
    search = Expectimax(PositionHeuristic(), max_depth=8, node_budget=1_000_000, time_budget_ms=1)
    result = search.choose(game.board)
    assert result is not None
    assert result.action in legal_actions(game.board)


def test_expectimax_fallback_uses_the_supplied_afterstate_evaluator():
    game = Game2048(random.Random(191))

    def prefer_right(board):
        return 10.0 if board[0][3] else 0.0

    search = Expectimax(
        evaluator=lambda _board: 0.0,
        fallback_evaluator=prefer_right,
        max_depth=8,
        node_budget=1,
        time_budget_ms=1,
        include_merge_reward=False,
    )
    result = search.choose(game.board)
    assert result is not None
    assert result.completed_depth == 0
    assert result.action in legal_actions(game.board)


def test_ntuple_training_episode_runs_with_seeded_rng():
    value = NtupleValue()
    stats = value.train_episode(
        rng=random.Random(1234),
        epsilon=1.0,
        alpha=0.01,
        max_moves=300,
    )
    assert stats.moves > 0
    assert stats.score >= 0
    assert stats.max_tile >= 2
    assert stats.merges >= 0


def test_ntuple_training_validates_discount_and_episode_budget():
    value = NtupleValue()
    with pytest.raises(ValueError, match='gamma'):
        value.train_episode(rng=random.Random(1), gamma=1.1)
    with pytest.raises(ValueError, match='alpha'):
        value.train_episode(rng=random.Random(1), alpha=0.0)
    with pytest.raises(ValueError, match='max_moves'):
        value.train_episode(rng=random.Random(1), max_moves=0)
    with pytest.raises(ValueError, match='td_lambda'):
        value.train_episode(rng=random.Random(1), td_lambda=1.1)


def test_native_td_lambda_trace_training_changes_weights_without_changing_random_game():
    value_td0 = NtupleValue()
    value_lambda = NtupleValue()
    if value_td0.backend != 'native-c++':
        pytest.skip('optional C++ N-tuple backend has not been built')
    seed = 99173
    stats_td0 = value_td0.train_episode(
        rng=random.Random(seed), epsilon=1.0, alpha=0.01, gamma=1.0,
        td_lambda=0.0, max_moves=250,
    )
    stats_lambda = value_lambda.train_episode(
        rng=random.Random(seed), epsilon=1.0, alpha=0.01, gamma=1.0,
        td_lambda=0.7, max_moves=250,
    )
    assert (stats_td0.score, stats_td0.moves, stats_td0.max_tile) == (
        stats_lambda.score, stats_lambda.moves, stats_lambda.max_tile
    )
    assert not np.array_equal(value_td0.tables, value_lambda.tables)


def test_python_td_lambda_fallback_trains_with_sparse_traces():
    value = NtupleValue()
    value._native = None
    stats = value.train_episode(
        rng=random.Random(417), epsilon=1.0, alpha=0.01, gamma=1.0,
        td_lambda=0.5, max_moves=40,
    )
    assert stats.moves == 40
    assert stats.td_updates > 0
    assert np.count_nonzero(value.tables) > 0


def test_native_trainer_captures_reachable_positions_without_python_game_loop():
    value = NtupleValue()
    if value.backend != 'native-c++':
        pytest.skip('optional C++ N-tuple backend has not been built')
    captured = []
    stats = value.train_episode(
        rng=random.Random(7722),
        epsilon=1.0,
        alpha=0.05,
        max_moves=500,
        position_callback=captured.append,
        capture_min_tile=8,
        capture_every=1,
    )
    assert stats.moves > 0
    assert captured
    assert all(max(tile for row in board for tile in row) >= 8 for board in captured)
    assert all(legal_actions(board) for board in captured)


def test_position_pool_deduplicates_and_round_trips(tmp_path):
    pool = PositionPool(capacity=3)
    board = (
        (2, 4, 8, 16),
        (32, 64, 128, 256),
        (512, 1024, 2, 4),
        (8, 16, 0, 0),
    )
    assert pool.add(board)
    assert not pool.add(board)
    assert pool.sample(random.Random(7), min_tile=512) == board
    with pytest.raises(LookupError):
        pool.sample(random.Random(7), min_tile=2048)
    path = tmp_path / 'positions.npz'
    pool.save(path)
    assert not list(tmp_path.glob('.*.tmp'))
    restored = PositionPool.load(path)
    assert len(restored) == 1
    assert restored.sample(random.Random(1)) == board


def test_native_ntuple_backend_matches_numpy_fallback():
    native = NtupleValue()
    if native.backend != 'native-c++':
        pytest.skip('optional C++ backend has not been built')
    fallback = NtupleValue(native.tables)
    fallback._native = None
    board = (
        (2, 4, 8, 16),
        (32, 64, 128, 256),
        (512, 1024, 2048, 4096),
        (8192, 16384, 32768, 0),
    )
    assert native.value(board) == pytest.approx(fallback.value(board), abs=1e-5)
    native_error = native.update(board, target=12345.0, alpha=0.03)
    fallback_error = fallback.update(board, target=12345.0, alpha=0.03)
    assert native_error == pytest.approx(fallback_error, abs=1e-5)
    assert np.allclose(native.tables, fallback.tables, atol=1e-6)

    # The search library has its own leaf-evaluation kernel. Compare its
    # precomputed symmetry-index path against the independent N-tuple backend.
    rng = random.Random(4519)
    for _ in range(50):
        board = tuple(
            tuple(0 if (rank := rng.randrange(0, 16)) == 0 else 1 << rank for _ in range(4))
            for _ in range(4)
        )
        indices = native._indices(board)
        for symmetry in range(8):
            for pattern in range(4):
                native.tables[pattern, indices[symmetry, pattern]] = rng.uniform(-100.0, 100.0)
        assert native_ntuple_value(board, native.tables) == pytest.approx(
            native.value(board), abs=1e-5
        )


def test_native_move_engine_matches_python_on_random_boards():
    pytest.importorskip('ctypes')
    from destroyer.search import _load_native_search
    if _load_native_search() is None:
        pytest.skip('optional C++ search backend has not been built')
    rng = random.Random(9901)
    for _ in range(100):
        board = []
        for _row in range(4):
            row = []
            for _col in range(4):
                rank = rng.choices(range(0, 12), weights=[8] + [1] * 11)[0]
                row.append(0 if rank == 0 else 1 << rank)
            board.append(row)
        for action in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
            python_result = apply_move(board, action)
            native_board, native_reward, native_merges, native_moved = native_apply_move(board, action)
            assert native_board == python_result.board, (
                f"action={action} board={board} native={native_board} "
                f"python={python_result.board} reward={native_reward}/{python_result.score_delta}"
            )
            assert native_reward == python_result.score_delta, f"action={action} board={board}"
            assert native_merges == python_result.merges, f"action={action} board={board}"
            assert native_moved == python_result.moved, f"action={action} board={board}"


def test_native_classic_heuristic_matches_python():
    from destroyer.search import _load_native_search, board_to_ranks
    library = _load_native_search()
    if library is None:
        pytest.skip('optional C++ search backend has not been built')
    rng = random.Random(9912)
    heuristic = PositionHeuristic()
    for _ in range(100):
        board = tuple(
            tuple(0 if (rank := rng.randrange(0, 12)) == 0 else 1 << rank for _ in range(4))
            for _ in range(4)
        )
        ranks = board_to_ranks(board)
        native_value = library.destroyer_classic_heuristic(
            ranks.ctypes.data_as(__import__('ctypes').POINTER(__import__('ctypes').c_uint8))
        )
        assert native_value == pytest.approx(heuristic(board), rel=2e-5, abs=0.5)


def test_native_expectimax_returns_legal_action_with_bounded_budget():
    game = Game2048(random.Random(27))
    search = Expectimax(PositionHeuristic(), max_depth=4, node_budget=100_000, time_budget_ms=10)
    if search.native_mode != 'classic':
        pytest.skip('optional C++ search backend has not been built')
    result = search.choose(game.board)
    assert result is not None
    assert result.action in legal_actions(game.board)
    assert result.nodes >= 0
    assert result.completed_depth >= 0


def test_native_and_python_expectimax_match_full_shallow_search():
    game = Game2048(random.Random(2718))
    native = Expectimax(PositionHeuristic(), max_depth=1, node_budget=100_000, native=True)
    python = Expectimax(PositionHeuristic(), max_depth=1, node_budget=100_000, native=False)
    if native.native_mode != 'classic':
        pytest.skip('optional C++ search backend has not been built')
    native_result = native.choose(game.board)
    python_result = python.choose(game.board)
    assert native_result is not None and python_result is not None
    assert native_result.completed_depth == python_result.completed_depth == 1
    assert native_result.action == python_result.action
    assert native_result.value == pytest.approx(python_result.value, rel=2e-5, abs=1.0)


@pytest.mark.parametrize("node_budget", [4, 12, 24, 64])
def test_native_and_python_expectimax_match_node_budget_interruption(node_budget):
    game = Game2048(random.Random(8128))
    native = Expectimax(
        PositionHeuristic(), max_depth=3, node_budget=node_budget,
        time_budget_ms=None, native=True,
    )
    python = Expectimax(
        PositionHeuristic(), max_depth=3, node_budget=node_budget,
        time_budget_ms=None, native=False,
    )
    if native.native_mode != "classic":
        pytest.skip("optional C++ search backend has not been built")
    native_result = native.choose(game.board)
    python_result = python.choose(game.board)
    assert native_result is not None and python_result is not None
    assert native_result.completed_depth == python_result.completed_depth
    assert native_result.nodes == python_result.nodes
    assert native_result.action == python_result.action
    assert native_result.value == pytest.approx(python_result.value, rel=2e-5, abs=1.0)


def test_native_and_python_ntuple_expectimax_match_full_shallow_search():
    game = Game2048(random.Random(2719))
    ntuple = NtupleValue()
    native = Expectimax(ntuple.state_value, max_depth=1, node_budget=100_000, native=True)
    python = Expectimax(ntuple.state_value, max_depth=1, node_budget=100_000, native=False)
    if native.native_mode != 'ntuple':
        pytest.skip('optional C++ search backend has not been built')
    native_result = native.choose(game.board)
    python_result = python.choose(game.board)
    assert native_result is not None and python_result is not None
    assert native_result.completed_depth == python_result.completed_depth == 1
    assert native_result.action == python_result.action
    assert native_result.value == pytest.approx(python_result.value, rel=2e-5, abs=1.0)


def test_native_search_ntuple_kernel_matches_python_ntuple_kernel():
    library = __import__('destroyer.search', fromlist=['_load_native_search'])._load_native_search()
    if library is None:
        pytest.skip('optional C++ search backend has not been built')
    value = NtupleValue()
    rng = np.random.default_rng(2048)
    boards = []
    for _ in range(12):
        board = tuple(
            tuple(0 if (rank := int(rng.choice(12, p=[0.5] + [0.5 / 11] * 11))) == 0 else 1 << rank
                  for _ in range(4))
            for _ in range(4)
        )
        boards.append(board)
        for symmetry_indices in value._indices(board):
            for pattern_id, index in enumerate(symmetry_indices):
                value.tables[pattern_id, index] = np.float32(rng.normal())

    for board in boards:
        assert native_ntuple_value(board, value.tables, afterstate=True) == pytest.approx(
            value.value(board), rel=2e-5, abs=1e-5
        )
        assert native_ntuple_value(board, value.tables, afterstate=False) == pytest.approx(
            value.state_value(board), rel=2e-5, abs=1e-4
        )
    with pytest.raises(ValueError, match='shape'):
        native_ntuple_value(boards[0], np.zeros((4, 4), dtype=np.float32))
