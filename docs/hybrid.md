# N-tuple + Expectimax hybrid

The hybrid uses Expectimax to plan several moves ahead and an N-tuple value function to estimate leaf positions. This is the project's main playing agent.

## Search outline

```text
Player node
  -> enumerate legal moves
      -> deterministic slide/merge
          -> chance node
                -> spawn 2 (90% / empty-cell count)
                -> spawn 4 (10% / empty-cell count)
                    -> recurse or evaluate leaf
```

Expectimax uses the exact spawn distribution from the game engine. The search includes iterative deepening, transposition caching, a node budget, and a wall-clock budget. Both backends count player and chance nodes toward the node budget; individual spawn outcomes are not counted as separate chance nodes. If an iteration is interrupted, the result comes from the last completed iteration or the legal fallback. A configured depth is a cap, not a promise that every decision reaches that depth.

The transposition cache persists across calls to `choose()`. If a mutable learned evaluator is updated in place, call `clear_cache()` before the next search so cached values are not reused with changed weights. The GUI replaces its solver when loading a new N-tuple checkpoint.

## Evaluators

- **Classic heuristic:** hand-written row/column positional scoring based on empty cells, merge opportunities, monotonicity, and tile ranks. It is a baseline, not a learned model.
- **N-tuple:** learned value function, with D4-symmetric features.
- **Merge reward:** the search can include immediate merge score when the evaluator is the learned afterstate function; the classic heuristic uses a separate scale.

## Benchmarking the hybrid

Example:

```bash
python scripts/benchmark_hybrid.py \
  --games 100 \
  --seed 1000000 \
  --depth 3 \
  --node-budget 500000 \
  --time-budget-ms 1.0 \
  --ntuple models/ntuple_latest.npz \
  --output reports/hybrid_eval.json
```

The runner reports score and tile attainment, search nodes, cache hits, completed depth, and latency distributions. Use the same seed range for paired agent comparisons. Search settings and checkpoint must be held constant except for the feature under test.

## Tile downgrading (experimental)

The optional TDL-style transform is root-only. Once the configured threshold is reached, it finds the largest missing tile rank below the current maximum and halves occupied tiles above that missing rank. The transform is applied once to the root search position; it is not applied to real game transitions or individual leaf states. It is disabled by default because the small paired evaluation is inconclusive.

Example:

```bash
python scripts/benchmark_hybrid.py \
  --games 100 --seed 1000000 --depth 3 \
  --node-budget 500000 --time-budget-ms 1.0 \
  --downgrade-threshold 32768 --downgrade-steps 1 \
  --ntuple models/ntuple_latest.npz \
  --output reports/hybrid_downgrade.json
```

Only one rank shift is supported. Do not assume it improves strength; it remains an opt-in research feature.
