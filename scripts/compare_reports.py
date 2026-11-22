#!/usr/bin/env python3
"""Compare two arena JSON reports on overlapping seeds using paired statistics."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def load_records(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"No game records in {path}")
    indexed = {int(record["seed"]): record for record in records}
    if len(indexed) != len(records):
        raise ValueError(f"Duplicate seeds in {path}")
    return indexed


def seed_ranges(seeds: list[int]) -> list[list[int]]:
    """Compress an exact seed set into inclusive contiguous ranges."""
    if not seeds:
        return []
    ranges: list[list[int]] = []
    start = previous = seeds[0]
    for seed in seeds[1:]:
        if seed == previous + 1:
            previous = seed
            continue
        ranges.append([start, previous])
        start = previous = seed
    ranges.append([start, previous])
    return ranges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")

    baseline = load_records(args.baseline)
    candidate = load_records(args.candidate)
    seeds = sorted(baseline.keys() & candidate.keys())
    if not seeds:
        parser.error("reports have no overlapping seeds")
    score_deltas = [candidate[s]["score"] - baseline[s]["score"] for s in seeds]
    wins = sum(delta > 0 for delta in score_deltas)
    losses = sum(delta < 0 for delta in score_deltas)
    rng = random.Random(args.seed)
    bootstrap_means = sorted(
        statistics.fmean(rng.choice(score_deltas) for _ in score_deltas)
        for _ in range(args.bootstrap)
    )
    low = bootstrap_means[int(0.025 * (len(bootstrap_means) - 1))]
    high = bootstrap_means[int(0.975 * (len(bootstrap_means) - 1))]
    targets = (2048, 4096, 8192, 16384, 32768)
    result = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "paired_games": len(seeds),
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "paired_seed_ranges": seed_ranges(seeds),
        "score_delta_mean": statistics.fmean(score_deltas),
        "score_delta_median": statistics.median(score_deltas),
        "score_wins": wins,
        "score_losses": losses,
        "score_ties": len(seeds) - wins - losses,
        "paired_bootstrap_mean_delta_95": [low, high],
        "tile_rate_delta": {
            str(tile): statistics.fmean(candidate[s]["max_tile"] >= tile for s in seeds)
            - statistics.fmean(baseline[s]["max_tile"] >= tile for s in seeds)
            for tile in targets
        },
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
