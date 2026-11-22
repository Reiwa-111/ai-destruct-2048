# Benchmark methodology

## Purpose

Benchmarks measure playing strength and search cost. Training metrics are not evaluation metrics. The project distinguishes fresh-start greedy evaluation, hybrid search evaluation, and paired feature comparisons.

## Evaluation protocol

- **Fresh starts:** each evaluation game begins from the standard two-tile board; restart-position pools are not used.
- **Deterministic seeds:** each game uses a consecutive seed from the configured starting value. Report the inclusive seed range.
- **No learning during evaluation:** agents do not explore, update weights, or modify their checkpoint during holdout games.
- **Paired comparisons:** candidate and baseline should use the same seed range and environment. Compare score deltas per seed, not only aggregate averages.
- **Configuration isolation:** change one variable per experiment whenever possible. Report search depth, node budget, time budget, evaluator/checkpoint, native backend, and any optional transforms.
- **Metrics:** score mean/median/range; max-tile attainment rates; game length; search nodes/cache hits; completed depth; and per-decision latency distribution when available.
- **Uncertainty:** use Wilson confidence intervals for tile-attainment proportions and paired bootstrap intervals for mean score differences. A small sample is exploratory.

## Provenance checklist

For every result intended for publication, record:

- Exact checkpoint filename, format/version, and SHA-256. A filename alone is not a model identity.
- Source commit and native shared-library SHA-256; record compiler version and build flags for native runs.
- Backend actually used (`native-c++` or Python fallback), search depth, node budget, time target, and optional transforms.
- Runtime environment, CPU/GPU device, Python and relevant numerical-library versions.
- Game count, complete inclusive seed range, raw JSON report, and the report's SHA-256.

Keep the checkpoint and raw report available alongside the hash. A hash proves identity, not availability: a benchmark cannot be independently rerun from a model that is not distributed or otherwise accessible.

## Commands

Greedy N-tuple holdout:

```bash
python scripts/benchmark_ntuple.py \
  --checkpoint models/ntuple_latest.npz \
  --games 100 \
  --seed 5000000 \
  --output reports/ntuple_holdout.json
```

Hybrid holdout:

```bash
python scripts/benchmark_hybrid.py \
  --games 100 \
  --seed 1000000 \
  --depth 3 \
  --node-budget 500000 \
  --time-budget-ms 1.0 \
  --ntuple models/ntuple_latest.npz \
  --output reports/hybrid_holdout.json
```

Paired report comparison (for compatible per-game JSON reports):

```bash
python scripts/compare_reports.py reports/baseline.json reports/candidate.json
```

The paired comparator reports the exact overlapping seed set as contiguous ranges (`paired_seed_ranges`); the start/end values alone should not be read as proof that every seed in that interval was present.

## Interpretation

- Observed success rate is not the same as the true success probability. A 100/100 result still has a non-zero lower confidence bound.
- Median and mean score describe different parts of the distribution; both should be shown.
- A time budget is a search stopping target, not a hard real-time guarantee. System scheduling, memory contention, and runtime overhead can produce outliers.
- Results from different checkpoints or different seed ranges are not paired comparisons.
- Historical benchmark reports do not become reproducible merely because their seed range is recorded. The model checkpoint and compatible source revision must also be available.
- Keep raw output locally for audit; commit compact result summaries and methodology instead of every exploratory run.
