from scripts.compare_reports import seed_ranges


def test_seed_ranges_compress_only_consecutive_seeds():
    assert seed_ranges([10, 11, 12, 15, 18, 19]) == [[10, 12], [15, 15], [18, 19]]


def test_seed_ranges_empty_input():
    assert seed_ranges([]) == []
